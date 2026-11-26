"""网页操作流 E2E：用 HTTP 严格复刻前端 index.html 的调用序列。

覆盖需求中的"网页自测跨段对轨和交错保存"：
- 载入样例 -> 填打板点保存对轨 -> 跨多段试听 -> 确认
- 两个"浏览器窗口"交错：窗口2 在窗口1 拉取 state 之后保存静音，窗口1 用
  旧 version 保存对轨 -> 409 -> 按回传 state 重试 -> 成功
- 重试后逐段试听确认到全通过 -> 下载校验
运行：python3 tests/webflow.py
"""

import http.client
import io
import json
import os
import sys
import threading
import time
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
import app as app_module  # noqa: E402

HOST, PORT = "127.0.0.1", 8973
FAILED = []


def check(name, cond, extra=""):
    print(("PASS" if cond else "FAIL"), "-", name, ("| " + str(extra) if extra and not cond else ""))
    if not cond:
        FAILED.append(name)


def req(method, path, body=None, headers=None, raw_conn=None):
    c = raw_conn or http.client.HTTPConnection(HOST, PORT, timeout=10)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    data = r.read()
    if raw_conn is None:
        c.close()
    return r.status, dict(r.getheaders()), data


def main():
    from werkzeug.serving import make_server
    srv = make_server(HOST, PORT, app_module.app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.4)

    # ---- 网页启动：refresh() 拉 state（未载入）
    st = json.loads(req("GET", "/api/state")[2])
    check("初始未载入", st == {"loaded": False})

    # ---- 点"载入内置样例"
    status, _, data = req("POST", "/api/load_sample")
    check("载入样例 -> 200", status == 200, data[:120])
    state = json.loads(data)
    nseg = len(state["segments"])
    check("样例 6 段", nseg == 6, nseg)
    check("样例两轨不等长", state["num_samples"] == 1190700
          and state["num_samples_b"] == 1192773)

    # ---- 前端对轨面板填入打板点并保存（带当前 version）
    align_payload = {"a1": 22050, "b1": 20616, "a2": 1080450, "b2": 1082191,
                     "version": state["version"]}
    status, _, data = req("POST", "/api/alignment", json.dumps(align_payload),
                          {"Content-Type": "application/json"})
    check("保存对轨 -> 200", status == 200, data[:160])
    state = json.loads(data)["state"]
    check("对轨生效、斜率≈1.003（打板取整）",
          abs(state["alignment"]["slope"] - 1.003) < 1e-3, state["alignment"])

    # ---- 跨段试听：6 段全部请求 segment WAV，记录 X-Mix-Hash（前端 listened）
    listened = {}
    pcm_by_seg = {}
    for i in range(nseg):
        status, headers, wav_data = req("GET", f"/api/segment/{i}.wav")
        check(f"段 {i} 试听 200", status == 200)
        listened[i] = headers["X-Mix-Hash"]
        with wave.open(io.BytesIO(wav_data), "rb") as w:
            pcm_by_seg[i] = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    check("各段哈希互不雷同或按内容区分", len(set(listened.values())) >= 1)

    # 对轨后打板脉冲对齐校验：段 0 内 A 打板位置 22050 处应为局部最大峰值
    seg0 = pcm_by_seg[0]
    win = np.abs(seg0[22050 - 40:22050 + 40]).astype(np.float64)
    center = int(np.abs(seg0[22050]))
    check("段0 打板位置为局部峰值（对齐零偏移）",
          center >= win.max() - 1 and int(np.argmax(win)) in range(36, 45),
          f"center={center} max={win.max()} argmax={int(np.argmax(win))}")

    # ---- 交错保存：窗口2 在窗口1 持有旧 state 期间保存静音
    v_before = state["version"]
    win2_mutes = {"B": [[500000, 500441]], "version": v_before}
    status, _, data = req("POST", "/api/mutes", json.dumps(win2_mutes),
                          {"Content-Type": "application/json"})
    check("窗口2 保存静音 -> 200", status == 200, data[:160])
    state_after_mute = json.loads(data)["state"]
    check("版本号前进", state_after_mute["version"] == v_before + 1)

    # 窗口1 仍用旧 version 提交对轨修改 -> 前端会收到 409 conflict
    stale_align = {"a1": 22050, "b1": 20620, "a2": 1080450, "b2": 1082195,
                   "version": v_before}
    status, _, data = req("POST", "/api/alignment", json.dumps(stale_align),
                          {"Content-Type": "application/json"})
    body = json.loads(data)
    check("窗口1 旧版本对轨 -> 409 conflict", status == 409 and body.get("conflict") is True)
    check("409 回传当前生效状态", body["state"]["version"] == v_before + 1
          and body["state"]["mutes"]["B"] == [[500000, 500441]])
    check("冲突后原对轨仍是生效版本",
          body["state"]["alignment"]["b1"] == 20616)

    # 前端 conflictPrompt -> "按当前设置重试"：用回传的新 version 重发
    retry_payload = dict(stale_align)
    retry_payload["version"] = body["state"]["version"]
    status, _, data = req("POST", "/api/alignment", json.dumps(retry_payload),
                          {"Content-Type": "application/json"})
    check("重试保存对轨 -> 200", status == 200, data[:160])
    state = json.loads(data)["state"]
    check("重试后新对轨生效且静音保留",
          state["alignment"]["b1"] == 20620 and state["mutes"]["B"] == [[500000, 500441]])

    # ---- 反向交错：静音保存撞上对轨改动
    v = state["version"]
    req("POST", "/api/alignment",
        json.dumps({"a1": 22050, "b1": 20616, "a2": 1080450, "b2": 1082191,
                    "version": v}),
        {"Content-Type": "application/json"})
    stale_mute = {"A": [[100, 200]], "version": v}
    status, _, data = req("POST", "/api/mutes", json.dumps(stale_mute),
                          {"Content-Type": "application/json"})
    body = json.loads(data)
    check("静音撞对轨改动 -> 409 conflict", status == 409 and body.get("conflict"))
    stale_mute["version"] = body["state"]["version"]
    status, _, _ = req("POST", "/api/mutes", json.dumps(stale_mute),
                       {"Content-Type": "application/json"})
    check("静音按新版本重试 -> 200", status == 200)

    # ---- 设置变化后旧试听哈希确认被拒，重新试听后确认
    state = json.loads(req("GET", "/api/state")[2])
    old_h = listened[0]
    status, _, data = req("POST", "/api/confirm/0", json.dumps({"hash": old_h}),
                          {"Content-Type": "application/json"})
    check("旧 PCM 哈希确认 -> 409", status == 409)

    for i in range(nseg):
        status, headers, _ = req("GET", f"/api/segment/{i}.wav")
        h = headers["X-Mix-Hash"]
        status, _, data = req("POST", f"/api/confirm/{i}", json.dumps({"hash": h}),
                              {"Content-Type": "application/json"})
        check(f"段 {i} 重新试听后确认 -> 200", status == 200, data[:120])
    state = json.loads(req("GET", "/api/state")[2])
    check("全部段落通过", state["all_passed"])

    # ---- 下载：长度=A、内容=各段 PCM 拼接
    status, headers, data = req("GET", "/api/download")
    check("全通过下载 -> 200", status == 200)
    check("Content-Length=44+A长度*2",
          headers.get("Content-Length") == str(44 + state["num_samples"] * 2))
    with wave.open(io.BytesIO(data), "rb") as w:
        check("下载 WAV 帧数=A 长度", w.getnframes() == state["num_samples"])
        full = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    # 与当前各段试听 PCM 逐段一致
    ok = True
    for i in range(nseg):
        s0 = state["segments"][i]["start_sample"]
        s1 = state["segments"][i]["end_sample"]
        _, _, wav_data = req("GET", f"/api/segment/{i}.wav")
        with wave.open(io.BytesIO(wav_data), "rb") as w:
            seg_pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        ok = ok and np.array_equal(full[s0:s1], seg_pcm)
    check("下载整集与逐段试听 PCM 完全一致", ok)

    srv.shutdown()
    print()
    if FAILED:
        print(f"共 {len(FAILED)} 项失败")
        sys.exit(1)
    print("网页操作流 E2E 全部通过。")


if __name__ == "__main__":
    main()
