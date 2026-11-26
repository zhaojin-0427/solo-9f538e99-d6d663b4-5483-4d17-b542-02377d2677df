"""真实 HTTP 环境自测：多线程服务器 + 原始 http.client（含打板对轨）。

验证：
1. 跨段对轨的完整 HTTP 闭环：上传（两轨不等长）-> 保存对轨 -> 逐段试听确认；
2. 对轨/静音交错保存：版本冲突 409，按提示带新版本重试成功；
3. 对轨重算（锁外按段）进行中，另一连接的保存/试听立即响应不被阻塞；
4. 浏览器停读（下载挂着不读）时仍可保存设置；
5. 下载进行中改对轨/静音，已开始的下载继续交付当时通过的混音（含对轨快照）；
6. 下载中途断连后服务器正常服务；跨"窗口"（独立连接）状态一致。

运行：python3 tests/livetest.py
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

HOST, PORT = "127.0.0.1", 8972
RATE, DUR = 8000, 12.0
N = int(RATE * DUR)          # A 长度 96000；3 段 40000/40000/16000
SEG = RATE * 5
N_B = N + 4000               # B 机更长（时钟偏快），演示不等长上传与补零

FAILED = []


def check(name, cond, extra=""):
    print(("PASS" if cond else "FAIL"), "-", name, ("| " + str(extra) if extra and not cond else ""))
    if not cond:
        FAILED.append(name)


def wav_bytes(arr, rate=RATE):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(arr, dtype="<i2").tobytes())
    return buf.getvalue()


def multipart(fields):
    boundary = "----livetestboundary"
    body = b""
    for name, fname, data in fields:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                 f"filename=\"{fname}\"\r\nContent-Type: audio/wav\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def conn():
    return http.client.HTTPConnection(HOST, PORT, timeout=10)


def request(method, path, body=None, headers=None):
    c = conn()
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, dict(r.getheaders()), data


def read_exact(resp, n):
    buf = b""
    while len(buf) < n:
        chunk = resp.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return buf


def main():
    from werkzeug.serving import make_server
    srv = make_server(HOST, PORT, app_module.app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.4)

    rng = np.random.default_rng(11)
    a = np.full(N, 1000, dtype=np.int16)
    a[5000:5100] = 30000
    b = rng.integers(-3000, 3000, size=N_B).astype(np.int16)
    wa, wb = wav_bytes(a), wav_bytes(b)

    # ------------------------------------------------ 1. 不等长上传
    body, ctype = multipart([("track_a", "a.wav", wa), ("track_b", "b.wav", wb)])
    status, _, data = request("POST", "/api/upload", body, {"Content-Type": ctype})
    check("不等长上传 -> 200", status == 200, data[:200])
    st = json.loads(data)
    check("成片长度=A、B 长度独立", st["num_samples"] == N and st["num_samples_b"] == N_B)
    check("初始无对轨", st["alignment"] is None)
    nseg = len(st["segments"])
    check("3 段", nseg == 3)

    # ------------------------------------------------ 2. 跨段对轨
    align = {"a1": 12345, "b1": 9000, "a2": 87654, "b2": 90000}
    status, hdrs, seg_before = request("GET", "/api/segment/1.wav")
    check("对轨前可试听", status == 200)
    status, _, data = request("POST", "/api/alignment",
                              json.dumps({**align, "version": st["version"]}),
                              {"Content-Type": "application/json"})
    check("保存对轨 -> 200", status == 200, data[:200])
    st = json.loads(data)["state"]
    check("对轨状态回传", st["alignment"]["a1"] == align["a1"]
          and abs(st["alignment"]["slope"] - (90000 - 9000) / (87654 - 12345)) < 1e-12)
    status, hdrs, seg_after = request("GET", "/api/segment/1.wav")
    check("对轨后试听内容改变", status == 200 and seg_after != seg_before)

    # 非法对轨 -> 400 且不影响生效混音
    bad = {"a1": 100, "b1": 50, "a2": 50, "b2": 60, "version": st["version"]}
    status, _, data = request("POST", "/api/alignment", json.dumps(bad),
                              {"Content-Type": "application/json"})
    check("非递增打板点 -> 400", status == 400)
    status, _, cur = request("GET", "/api/segment/1.wav")
    check("400 后生效混音不变", cur == seg_after)

    # 全部段按生效对轨试听 + 确认
    status, _, data = request("GET", "/api/state")
    st = json.loads(data)
    for seg in st["segments"]:
        status, headers, _ = request("GET", f"/api/segment/{seg['index']}.wav")
        h = headers.get("X-Mix-Hash")
        status, _, data = request("POST", f"/api/confirm/{seg['index']}",
                                  json.dumps({"hash": h}), {"Content-Type": "application/json"})
        assert status == 200, (seg["index"], status, data)
    status, _, data = request("GET", "/api/state")
    check("全部通过", json.loads(data)["all_passed"])

    # ------------------------------------------------ 3. 交错保存：版本冲突 + 重试
    v = json.loads(data)["version"]
    # 先在另一连接静音保存（版本前进），再用旧版本提交对轨
    status, _, data = request("POST", "/api/mutes",
                              json.dumps({"B": [[60000, 60500]], "version": v}),
                              {"Content-Type": "application/json"})
    check("静音保存 -> 200", status == 200, data[:200])
    stale = {**align, "a1": 100, "b1": 50, "a2": 90000, "b2": 91000, "version": v}
    status, _, data = request("POST", "/api/alignment", json.dumps(stale),
                              {"Content-Type": "application/json"})
    check("旧版本对轨提交 -> 409 conflict", status == 409
          and json.loads(data).get("conflict") is True)
    fresh = json.loads(data)["state"]
    check("冲突响应回传新状态与新静音",
          fresh["version"] == v + 1 and fresh["mutes"]["B"] == [[60000, 60500]])
    # 按新版本重试 -> 成功，静音保留
    retry = {**stale, "version": fresh["version"]}
    status, _, data = request("POST", "/api/alignment", json.dumps(retry),
                              {"Content-Type": "application/json"})
    check("带新版本重试 -> 200", status == 200, data[:200])
    st2 = json.loads(data)["state"]
    check("重试后对轨生效且静音仍在",
          st2["alignment"]["a1"] == 100 and st2["mutes"]["B"] == [[60000, 60500]])

    # ------------------------------------------------ 4. 重算进行中不阻塞他处
    # compute_hook 在锁外重算前阻塞，直到另一连接完成一次试听 + 静音保存
    gate = threading.Event()

    def hook():
        gate.wait(5)

    app_module.EP.compute_hook = hook
    align_done = {}

    def slow_align():
        align_done["status"], _, align_done["data"] = request(
            "POST", "/api/alignment",
            json.dumps({"a1": 12345, "b1": 9000, "a2": 87654, "b2": 90000,
                        "version": app_module.EP.version}),
            {"Content-Type": "application/json"})

    t = threading.Thread(target=slow_align)
    t.start()
    time.sleep(0.4)                       # 等慢对轨进入锁外重算
    t0 = time.time()
    status, _, _ = request("GET", "/api/segment/0.wav")
    dt_listen = time.time() - t0
    check("重算期间试听立即响应(<2s)", status == 200 and dt_listen < 2, f"{dt_listen:.2f}s")
    t0 = time.time()
    status, _, _ = request("POST", "/api/mutes",
                           json.dumps({"A": [[10, 20]], "version": app_module.EP.version}),
                           {"Content-Type": "application/json"})
    dt_save = time.time() - t0
    check("重算期间静音保存立即响应(<2s)", status == 200 and dt_save < 2, f"{dt_save:.2f}s")
    gate.set()
    t.join(timeout=10)
    app_module.EP.compute_hook = None
    # 慢对轨提交时发现版本已被静音保存推进 -> 409，静音生效、对轨未变
    check("慢对轨因交错改动 -> 409", align_done["status"] == 409, align_done.get("data", b"")[:200])
    st_now = json.loads(request("GET", "/api/state")[2])
    check("冲突后静音生效、对轨维持",
          st_now["mutes"]["A"] == [[10, 20]] and st_now["alignment"]["a1"] == 100)

    # ------------------------------------------------ 5. 下载快照：停读 + 改对轨/静音
    # 重新全部试听确认（前面的改动使段落失效）
    status, _, data = request("GET", "/api/state")
    st = json.loads(data)
    for seg in st["segments"]:
        status, headers, wav_seg = request("GET", f"/api/segment/{seg['index']}.wav")
        h = headers.get("X-Mix-Hash")
        # 留存段 1 当前（快照前）混音，用于比对下载内容
        if seg["index"] == 1:
            seg1_snapshot = wav_seg[44:]
        status, _, _ = request("POST", f"/api/confirm/{seg['index']}",
                               json.dumps({"hash": h}), {"Content-Type": "application/json"})
        assert status == 200
    check("重新确认后全部通过", json.loads(request("GET", "/api/state")[2])["all_passed"])

    dl = conn()
    dl.request("GET", "/api/download")
    resp = dl.getresponse()
    check("下载 -> 200", resp.status == 200)
    head_seg0 = read_exact(resp, 44 + SEG * 2)
    check("已读头+段0", len(head_seg0) == 44 + SEG * 2)

    # 下载挂着不读，另一连接改静音再改对轨——都须立即成功
    t0 = time.time()
    status, _, _ = request("POST", "/api/mutes",
                           json.dumps({"B": [[70000, 70900]],
                                       "version": app_module.EP.version}),
                           {"Content-Type": "application/json"})
    check("停读期间保存静音 -> 200", status == 200)
    status, _, _ = request("POST", "/api/alignment",
                           json.dumps({"a1": 500, "b1": 300, "a2": 90000, "b2": 89000,
                                       "version": app_module.EP.version}),
                           {"Content-Type": "application/json"})
    check("停读期间保存对轨 -> 200", status == 200)
    check("设置保存响应迅速(<5s)", time.time() - t0 < 5)

    rest = read_exact(resp, 1 << 30)
    dl.close()
    seg1 = rest[:SEG * 2]
    check("下载中段 1 仍为开始时的混音（对轨/静音均快照）", seg1 == seg1_snapshot)
    check("下载总长正确", len(head_seg0) + len(rest) == 44 + N * 2,
          len(head_seg0) + len(rest))

    # 当前段 1 已是新生效混音；改动后未重听 -> 新下载被门控
    status, _, cur = request("GET", "/api/segment/1.wav")
    check("当前段 1 已不同于快照", cur[44:] != seg1)
    check("改动后新下载 -> 403", request("GET", "/api/download")[0] == 403)

    # ------------------------------------------------ 6. 断连后正常 + 跨窗口一致
    dl2 = conn()
    dl2.request("GET", "/api/state")
    r2 = dl2.getresponse()
    read_exact(r2, 100)
    dl2.close()
    time.sleep(0.3)
    status, _, data = request("GET", "/api/state")
    st = json.loads(data)
    check("断连后服务器正常", status == 200 and st["loaded"])
    check("跨窗口对轨一致", st["alignment"]["a1"] == 500 and st["alignment"]["b2"] == 89000)
    check("跨窗口静音一致", st["mutes"]["B"] == [[70000, 70900]])

    # ------------------------------------------------ 7. 换集后旧集提交被拒
    old_id = st["episode_id"]
    body, ctype = multipart([("track_a", "a.wav", wav_bytes(a[:50000])),
                             ("track_b", "b.wav", wav_bytes(b[:60000]))])
    status, _, data = request("POST", "/api/upload", body, {"Content-Type": ctype})
    check("换集上传 -> 200", status == 200)
    new_id = json.loads(data)["episode_id"]
    check("集 ID 更换", new_id != old_id)
    status, _, data = request("POST", "/api/alignment",
                              json.dumps({"a1": 12345, "b1": 9000, "a2": 87654,
                                          "b2": 90000, "version": st["version"]}),
                              {"Content-Type": "application/json"})
    check("旧集版本提交 -> 409", status == 409 and json.loads(data).get("conflict"))
    check("拒绝后新集仍无对轨",
          json.loads(request("GET", "/api/state")[2])["alignment"] is None)

    srv.shutdown()
    print()
    if FAILED:
        print(f"共 {len(FAILED)} 项失败")
        sys.exit(1)
    print("真实 HTTP 环境自测全部通过。")


if __name__ == "__main__":
    main()
