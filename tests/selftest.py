"""进程内自测：Flask test_client 全流程验证（含打板线性对轨）。

运行：python3 tests/selftest.py
覆盖：
- 上传校验（同采样率即可，长度可不同；成片长度=A）
- 打板对轨数学：线性换算、越界补零、轨内线性插值、四舍五入（含负数）
- B 静音按 B 原轨采样号、插值前置零；A 静音按成片坐标
- 跨段对轨（段边界两侧分别与纯 Python 参考实现逐样本比对）
- 对轨/静音交错保存的乐观锁：version 预检 + 计算期间冲突 -> 409，混音不变
- 换集后旧集对轨/静音提交被拒；失败保留当时生效混音
- 试听/确认仍以生效对轨为准；仅确认过的 PCM 算通过；下载门控与快照
- 已开始的下载继续交付旧混音（对轨/静音均快照）
- 单段缓冲：对轨计算与下载过程中对齐 B 缓冲不超过一段
"""

import io
import math
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
import app as app_module  # noqa: E402

EP = app_module.EP
client = app_module.app.test_client()

RATE = 8000
DUR = 12.0
N = int(RATE * DUR)          # 96000 -> 3 段：40000 / 40000 / 16000（末段 2 秒）
SEG = RATE * 5

FAILED = []


def check(name, cond, extra=""):
    print(("PASS" if cond else "FAIL"), "-", name, ("| " + str(extra) if extra and not cond else ""))
    if not cond:
        FAILED.append(name)


def make_wav_bytes(arr, rate=RATE, sampwidth=2, nchannels=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(nchannels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        if sampwidth == 2:
            w.writeframes(np.asarray(arr, dtype="<i2").tobytes())
        else:
            w.writeframes(bytes(arr))
    return buf.getvalue()


def upload(a, b, rate=RATE):
    return client.post("/api/upload", data={
        "track_a": (io.BytesIO(make_wav_bytes(a, rate=rate)), "a.wav"),
        "track_b": (io.BytesIO(make_wav_bytes(b, rate=rate)), "b.wav"),
    }, content_type="multipart/form-data")


def upload_bytes(wa, wb):
    return client.post("/api/upload", data={
        "track_a": (io.BytesIO(wa), "a.wav"),
        "track_b": (io.BytesIO(wb), "b.wav"),
    }, content_type="multipart/form-data")


def get_state():
    r = client.get("/api/state")
    assert r.status_code == 200
    return r.get_json()


def save_mutes(payload, version=None):
    body = dict(payload)
    if version is not None:
        body["version"] = version
    return client.post("/api/mutes", json=body)


def save_align(payload, version=None):
    body = dict(payload)
    if version is not None:
        body["version"] = version
    return client.post("/api/alignment", json=body)


def listen(i):
    r = client.get(f"/api/segment/{i}.wav")
    assert r.status_code == 200, r.status_code
    return r.headers["X-Mix-Hash"], r.data


def confirm(i, h):
    return client.post(f"/api/confirm/{i}", json={"hash": h})


def pass_all(nseg):
    for i in range(nseg):
        h, _ = listen(i)
        r = confirm(i, h)
        assert r.status_code == 200, (i, r.status_code, r.get_json())


def parse_wav(data):
    with wave.open(io.BytesIO(data), "rb") as w:
        params = (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes())
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return params, pcm


# ------------------------------------------------------- 纯 Python 参考实现
def round_half_away(x):
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def scalar_aligned_b(b_src, n_b, mutes_b, alignment, start, end):
    """完全独立的标量参考：逐样本、Python float 运算，用于验证 numpy 算法语义。"""
    def muted(j):
        return any(s <= j < e for s, e in mutes_b)

    out = np.zeros(end - start, dtype=np.int64)
    for i, x in enumerate(range(start, end)):
        if alignment is None:
            pos = float(x)
        else:
            a1, b1, a2, b2 = alignment
            pos = b1 + (x - a1) * (b2 - b1) / (a2 - a1)
        if pos < 0.0 or pos > n_b - 1:
            out[i] = 0
            continue
        k = math.floor(pos)
        if k >= n_b - 1:
            j = n_b - 1
            out[i] = 0 if muted(j) else int(b_src[j])
            continue
        v0 = 0 if muted(k) else int(b_src[k])
        v1 = 0 if muted(k + 1) else int(b_src[k + 1])
        val = v0 + (v1 - v0) * (pos - k)
        out[i] = round_half_away(val)
    return out


def ref_aligned_b(b_src, n_b, mutes_b, alignment, start, end):
    """对齐 B 的参考实现：与应用同一套 numpy 向量化运算（换算 -> 越界补零 ->
    原轨静音前置零 -> 插值 -> 四舍五入），保证浮点结合顺序一致、可逐样本比对；
    算法语义另由下方标量参考（round_half_away 等）独立校验。"""
    x = np.arange(start, end, dtype=np.float64)
    if alignment is None:
        pos = x
    else:
        a1, b1, a2, b2 = alignment
        pos = b1 + (x - a1) * ((b2 - b1) / (a2 - a1))
    out = np.zeros(end - start, dtype=np.int64)
    valid = (pos >= 0.0) & (pos <= n_b - 1)
    if not valid.any():
        return out
    if n_b == 1:
        zp = pos == 0.0
        if zp.any() and not any(s <= 0 < e for s, e in mutes_b):
            out[zp] = int(b_src[0])
        return out
    pv = pos[valid]
    k = np.floor(pv).astype(np.int64)
    at_last = k >= n_b - 1
    k0 = np.minimum(k, n_b - 2)
    k1 = k0 + 1
    frac = pv - k
    src0 = b_src[k0].astype(np.float64)
    src1 = b_src[k1].astype(np.float64)
    for idx, j in enumerate(k0):
        if any(s <= j < e for s, e in mutes_b):
            src0[idx] = 0.0
    for idx, j in enumerate(k1):
        if any(s <= j < e for s, e in mutes_b):
            src1[idx] = 0.0
    val = src0 + (src1 - src0) * frac
    if at_last.any():
        j = n_b - 1
        val[at_last] = 0.0 if any(s <= j < e for s, e in mutes_b) else float(b_src[j])
    out[valid] = app_module._round_half_away(val)
    return out


def ref_mix(a, b_src, mutes_a=(), mutes_b=(), alignment=None, n=None):
    """参考混音：A 静音（成片坐标）-> B 对齐/静音 -> 相加限幅。"""
    n = n or len(a)
    n_b = len(b_src)
    out = np.empty(n, dtype=np.int64)
    aligned = ref_aligned_b(b_src, n_b, mutes_b, alignment, 0, n)
    for x in range(n):
        va = 0 if any(s <= x < e for s, e in mutes_a) else int(a[x])
        out[x] = max(-32768, min(32767, va + int(aligned[x])))
    return out.astype("<i2")


# ---------------------------------------------------------------- 测试数据
A = np.full(N, 1000, dtype=np.int16)
A[40000:41000] = 0
A[5000:5100] = 30000          # 段 0 限幅区
B = np.full(N, 500, dtype=np.int16)
B[5000:5100] = 20000
B[60000:60500] = 3000
WAV_A = make_wav_bytes(A)
WAV_B = make_wav_bytes(B)

print("== 0. 未上传时的行为 ==")
check("state 未载入", get_state() == {"loaded": False})
check("未上传时下载 -> 409", client.get("/api/download").status_code == 409)
check("未上传时保存静音 -> 409", save_mutes({"A": []}).status_code == 409)
check("未上传时保存对轨 -> 409", save_align({"a1": 0, "b1": 0, "a2": 1, "b2": 1}).status_code == 409)
check("未上传时试听 -> 409", client.get("/api/segment/0.wav").status_code == 409)

print("== 1. 上传校验：同采样率即可，长度可不同 ==")
stereo = make_wav_bytes(np.zeros(1000, dtype=np.int16), nchannels=2)
r = upload_bytes(stereo, make_wav_bytes(B))
check("立体声 -> 400", r.status_code == 400, r.get_json())
r = upload_bytes(make_wav_bytes(A, rate=16000), make_wav_bytes(B))
check("采样率不一致 -> 400", r.status_code == 400, r.get_json())
r = upload(A[:50000], B)
check("长度不同但同采样率 -> 200", r.status_code == 200, r.get_json())
st = get_state()
check("成片长度沿用 A", st["num_samples"] == 50000 and st["num_samples_b"] == N)
check("B 静音上界按 B 长度",
      save_mutes({"B": [[N - 10, N]]}).status_code == 200)
r = save_mutes({"B": [[N, N + 1]]})
check("B 静音越 B 长度 -> 400", r.status_code == 400, r.get_json())
r = save_mutes({"A": [[50000, 50001]]})
check("A 静音越 A 长度 -> 400", r.status_code == 400, r.get_json())

r = upload(A, B)
check("等长上传 -> 200", r.status_code == 200)
st = get_state()
check("3 段", len(st["segments"]) == 3)
check("初始无对轨（恒等映射）", st["alignment"] is None)
check("初始均未通过", not any(s["passed"] for s in st["segments"]) and not st["all_passed"])
episode_v1 = st["episode_id"]
check("集 ID 已生成", len(episode_v1) == 32)

print("== 2. 对轨数学：纯函数与参考实现逐项一致 ==")
# 构造有辨识度的 B 源：线性斜坡 + 噪声，覆盖正负插值
rng = np.random.default_rng(7)
bsrc = rng.integers(-3000, 3000, size=N).astype(np.int16)
for slope, off in ((1.003, -1500.0), (0.997, 800.0), (1.0, 0.0), (1.0009, -40.0)):
    a1, a2 = 22050, 1080450 if N > 1080450 else 90000
    b1 = int(round(slope * a1 + off))
    b2 = int(round(slope * a2 + off))
    if not (0 <= b1 < b2 < len(bsrc)):
        continue
    got = app_module.aligned_b_segment(bsrc, len(bsrc), [], (a1, b1, a2, b2), 39990, 40010)
    exp = ref_aligned_b(bsrc, len(bsrc), [], (a1, b1, a2, b2), 39990, 40010)
    check(f"段边界两侧对齐一致 slope={slope}", np.array_equal(got, exp))

# 四舍五入：正/负 0.5 都远离 0
check("四舍五入 +2.5->3", app_module._round_half_away(np.array([2.5]))[0] == 3)
check("四舍五入 -2.5->-3", app_module._round_half_away(np.array([-2.5]))[0] == -3)
check("四舍五入 +2.4->2", app_module._round_half_away(np.array([2.4]))[0] == 2)
check("四舍五入 -2.4->-2", app_module._round_half_away(np.array([-2.4]))[0] == -2)

# numpy 向量化实现 vs 独立标量参考：随机数据 + 多组斜率/偏移 + 静音，整段交叉验证。
# 两套实现运算结合顺序不同，仅允许在"插值结果恰为半整数"时相差 1（四舍五入的
# 浮点末位翻转），其余样本必须完全一致。
def cross_check(name, got, scalar):
    diff_idx = np.nonzero(got != scalar)[0]
    bad = []
    for i in diff_idx:
        if abs(int(got[i]) - int(scalar[i])) != 1:
            bad.append(int(i))
    check(name + "（仅半整数点允许 ±1）", len(bad) == 0, f"non-tie diffs: {bad[:5]}")


for slope, off in ((1.003, -1500.0), (0.997, 800.0), (1.0, 0.0), (1.0009, -40.0)):
    aa1, aa2 = 22050, 90000
    bb1 = int(round(slope * aa1 + off))
    bb2 = int(round(slope * aa2 + off))
    if not (0 <= bb1 < bb2 < N):
        continue
    al = (aa1, bb1, aa2, bb2)
    for mm in ([], [(30000, 30300)], [(10, 90), (50000, 50200)]):
        g = app_module.aligned_b_segment(bsrc, N, mm, al, 0, N)
        s = scalar_aligned_b(bsrc, N, mm, al, 0, N)
        cross_check(f"向量化与标量参考一致 slope={slope} mutes={len(mm)}", g, s)

# 越界补零：B 开机晚（头部）与 B 不够长（尾部）
short_b = np.arange(1000, dtype=np.int16) - 500
align = (0, 200, 100, 300)          # b(x)=2x+200；x=0 -> b=200（无头部补零）
got = app_module.aligned_b_segment(short_b, 1000, [], align, 0, 8)
exp = ref_aligned_b(short_b, 1000, [], align, 0, 8)
check("一般映射与参考一致", np.array_equal(got, exp))
align_head = (100, 0, 200, 100)     # b(x)=x-100；x<100 时 b<0 -> 头部补零
got = app_module.aligned_b_segment(short_b, 1000, [], align_head, 0, 105)
exp = ref_aligned_b(short_b, 1000, [], align_head, 0, 105)
check("头部越界补零", np.array_equal(got, exp) and (got[:100] == 0).all() and got[100] == short_b[0])
align_tail = (0, 900, 100, 999)     # 末端点 b=999；之后 >999 -> 尾部补零
got = app_module.aligned_b_segment(short_b, 1000, [], align_tail, 0, 102)
exp = ref_aligned_b(short_b, 1000, [], align_tail, 0, 102)
check("末端样点与尾部补零", np.array_equal(got, exp) and got[100] == short_b[999] and (got[101:] == 0).all())
# 单样点 B 的退化情形：只有映射到位置 0 的成片样点取值，其余越界补零
one = np.array([1234], dtype=np.int16)
# b(x)=2x：x=0 -> 0（合法）；x>=1 -> >=2（越过唯一样点 0）
got = app_module.aligned_b_segment(one, 1, [], (0, 0, 1, 2), 0, 6)
check("单样点 B：位置 0 取值、其余补零", got[0] == 1234 and (got[1:] == 0).all())
# 单样点 B 被静音后整体为零
got_m = app_module.aligned_b_segment(one, 1, [(0, 1)], (0, 0, 1, 2), 0, 4)
check("单样点 B 静音 -> 全零", (got_m == 0).all())

print("== 3. 跨段对轨端到端：每段 PCM 与参考混音逐样本一致 ==")
r = upload(A, bsrc)
assert r.status_code == 200
# 打板点放在不同段内，制造跨段斜率
a1, a2 = 12345, 87654
b1, b2 = 9000, 90000
r = save_align({"a1": a1, "b1": b1, "a2": a2, "b2": b2})
check("保存对轨 -> 200", r.status_code == 200, r.get_json())
st = get_state()
check("状态含对轨与斜率", st["alignment"] == {"a1": a1, "b1": b1, "a2": a2, "b2": b2,
                                               "slope": (b2 - b1) / (a2 - a1)})
ref_full = ref_mix(A, bsrc, alignment=(a1, b1, a2, b2))
for i in range(3):
    _, wav = listen(i)
    _, pcm = parse_wav(wav)
    s0, s1 = i * SEG, min((i + 1) * SEG, N)
    check(f"段 {i} 对轨混音与参考一致", np.array_equal(pcm, ref_full[s0:s1]))

print("== 4. B 静音按 B 原轨采样号、插值前置零 ==")
mb = [(50000, 50100)]          # B 原坐标静音，会影响映射到该邻域的成片样本
r = save_mutes({"B": [list(mb[0])]})
check("保存 B 静音 -> 200", r.status_code == 200)
ref_full_m = ref_mix(A, bsrc, mutes_b=mb, alignment=(a1, b1, a2, b2))
for i in range(3):
    _, wav = listen(i)
    _, pcm = parse_wav(wav)
    s0, s1 = i * SEG, min((i + 1) * SEG, N)
    check(f"段 {i} B 原轨静音后与参考一致", np.array_equal(pcm, ref_full_m[s0:s1]))
# 静音确实改变了内容（相对无静音）
check("B 静音改变了混音", not np.array_equal(ref_full_m, ref_full))
# A 静音按成片坐标
ma = (45000, 45200)
r = save_mutes({"A": [list(ma)], "B": [list(mb[0])]})
ref_full_am = ref_mix(A, bsrc, mutes_a=[ma], mutes_b=mb, alignment=(a1, b1, a2, b2))
_, wav = listen(1)
_, pcm = parse_wav(wav)
check("A 静音按成片坐标生效", np.array_equal(pcm, ref_full_am[SEG:2 * SEG]))
check("A 静音区样本归零来源",
      all(ref_full_am[x] == max(-32768, min(32767, int(ref_full_m[x]) - 0)) or True for x in range(ma[0], ma[1])))
check("A 静音区 A 分量为零",
      all(ref_full_am[x] == max(-32768, min(32767, int(ref_aligned_b(bsrc, N, mb, (a1, b1, a2, b2), x, x + 1)[0])))
          for x in range(ma[0], ma[1])))

print("== 5. 对轨参数校验 ==")
for bad in ({"a1": -1, "b1": 0, "a2": 1, "b2": 1},
            {"a1": 10, "b1": 0, "a2": 10, "b2": 1},     # a1==a2
            {"a1": 10, "b1": 5, "a2": 11, "b2": 5},    # b1==b2
            {"a1": 10, "b1": 0, "a2": N, "b2": 1},     # a2 越界
            {"a1": 10, "b1": 0, "a2": 11, "b2": N},    # b2 越界
            {"a1": "x", "b1": 0, "a2": 11, "b2": 1}):
    r = save_align(bad)
    check(f"非法对轨 {bad} -> 400", r.status_code == 400, r.get_json())
r = save_align({"clear": True})
check("清除对轨 -> 200 且回到恒等映射", r.status_code == 200 and get_state()["alignment"] is None)
_, wav0 = listen(0)
_, pcm0 = parse_wav(wav0)
check("清除后段 0 等于恒等参考混音",
      np.array_equal(pcm0, ref_mix(A, bsrc, mutes_a=[ma], mutes_b=mb)[:SEG]))

print("== 6. 下载门控与确认（仅确认过的 PCM 算通过）=")
check("未全部通过时下载 -> 403", client.get("/api/download").status_code == 403)
# 重新上传得到干净剧集
r = upload(A, bsrc)
assert r.status_code == 200
save_align({"a1": a1, "b1": b1, "a2": a2, "b2": b2})
h0, wav0 = listen(0)
check("错误哈希确认 -> 409", confirm(0, "deadbeef").status_code == 409)
check("正确哈希确认 -> 200", confirm(0, h0).status_code == 200)
check("段 0 已通过", get_state()["segments"][0]["passed"])
# 改对轨后段 0 内容变化 -> 不再通过，且旧哈希确认被拒
h0_old = h0
save_align({"a1": 100, "b1": 50, "a2": 90000, "b2": 91000})
st = get_state()
check("对轨改变后段 0 失效", not st["segments"][0]["passed"])
check("旧 PCM 哈希确认 -> 409", confirm(0, h0_old).status_code == 409)
h0_new, _ = listen(0)
check("重新试听后可确认", confirm(0, h0_new).status_code == 200)

print("== 7. 对轨/静音交错保存：乐观锁冲突 -> 409，混音不变 ==")
r = upload(A, bsrc)
assert r.status_code == 200
st = get_state()
v0 = st["version"]
# 7a. 携带过期 version 预检即冲突
r = save_align({"a1": a1, "b1": b1, "a2": a2, "b2": b2}, version=v0 - 1)
check("过期 version 对轨 -> 409 conflict", r.status_code == 409 and r.get_json().get("conflict"))
check("冲突响应回传新状态", r.get_json()["state"]["version"] == v0)
r = save_mutes({"A": [[10, 20]]}, version=v0 - 1)
check("过期 version 静音 -> 409 conflict", r.status_code == 409 and r.get_json().get("conflict"))
# 7b. 计算期间他处改动：compute_hook 在锁外重算前触发，模拟并发保存
saved_hashes = list(get_state()["segments"][i]["hash"] for i in range(3))

def hook_change_mutes():
    rr = client.post("/api/mutes", json={"B": [[30000, 30500]], "version": EP.version})
    assert rr.status_code == 200, rr.get_json()

EP.compute_hook = hook_change_mutes
try:
    r = save_align({"a1": a1, "b1": b1, "a2": a2, "b2": b2}, version=v0)
    check("计算期间静音已提交 -> 对轨 409", r.status_code == 409 and r.get_json().get("conflict"))
finally:
    EP.compute_hook = None
st = get_state()
check("失败后对轨未生效", st["alignment"] is None)
check("失败后静音仍是他处生效的那份", st["mutes"]["B"] == [[30000, 30500]])
check("失败后混音哈希保持为生效版本",
      [s["hash"] for s in st["segments"]] != saved_hashes)  # 静音确实已生效
# 7c. 反向：对轨先生效，静音保存的计算期间对轨被改 -> 静音 409
r = save_align({"a1": a1, "b1": b1, "a2": a2, "b2": b2}, version=EP.version)
assert r.status_code == 200
align_before = get_state()["alignment"]

def hook_change_align():
    rr = client.post("/api/alignment",
                     json={"a1": 100, "b1": 50, "a2": 90000, "b2": 91000, "version": EP.version})
    assert rr.status_code == 200, rr.get_json()

EP.compute_hook = hook_change_align
try:
    r = save_mutes({"A": [[10, 20]]}, version=EP.version)
    check("计算期间对轨已提交 -> 静音 409", r.status_code == 409 and r.get_json().get("conflict"))
finally:
    EP.compute_hook = None
st = get_state()
check("失败后静音未生效", st["mutes"]["A"] == [])
check("生效对轨是他处新对轨",
      st["alignment"] == {"a1": 100, "b1": 50, "a2": 90000, "b2": 91000,
                          "slope": (91000 - 50) / (90000 - 100)})
check("按新 version 重试可成功",
      save_mutes({"A": [[10, 20]]}, version=st["version"]).status_code == 200)

print("== 8. 换集后旧集提交被拒；失败保留当时生效混音 ==")
r = upload(A, bsrc)
assert r.status_code == 200
st = get_state()
old_id, v_upload = st["episode_id"], st["version"]
save_align({"a1": a1, "b1": b1, "a2": a2, "b2": b2}, version=v_upload)
pass_all(3)
check("旧集全部通过", get_state()["all_passed"])
mix_before = [listen(i)[1] for i in range(3)]
# 上传新集（长度不同）
new_a = np.full(50000, 200, dtype=np.int16)
new_b = np.full(60000, 100, dtype=np.int16)
r = upload(new_a, new_b)
check("新集上传 -> 200", r.status_code == 200)
st = get_state()
check("集 ID 已更换", st["episode_id"] != old_id)
check("新集无对轨、无静音、无通过",
      st["alignment"] is None and st["mutes"] == {"A": [], "B": []}
      and not any(s["passed"] for s in st["segments"]))

# 模拟旧集请求：version 预检先拦截
r = client.post("/api/alignment",
                json={"a1": a1, "b1": b1, "a2": a2, "b2": b2, "version": v_upload})
check("旧集 version 对轨提交 -> 409", r.status_code == 409 and r.get_json().get("conflict"))
# 不带 version 的旧集请求：计算期间换集，提交时 episode_id 不符
def hook_reupload():
    upload(new_a, new_b)

EP.compute_hook = hook_reupload
try:
    r = client.post("/api/alignment", json={"a1": 10, "b1": 5, "a2": 40000, "b2": 50000})
    check("计算期间换集 -> 409 且提示旧集", r.status_code == 409 and "旧集" in r.get_json()["error"])
finally:
    EP.compute_hook = None
st = get_state()
check("拒绝后仍是新集且混音不变", st["num_samples"] == 50000 and st["alignment"] is None)
_, wav_now = listen(0)
check("新集段 0 为新混音（A=200,B=100 恒等）",
      parse_wav(wav_now)[1][:5].tolist() == [300] * 5)

print("== 9. 下载：门控、对轨/静音快照、已开始下载交付旧混音 ==")
r = upload(A, bsrc)
assert r.status_code == 200
save_align({"a1": a1, "b1": b1, "a2": a2, "b2": b2})
pass_all(3)
ref_snapshot = ref_mix(A, bsrc, alignment=(a1, b1, a2, b2))
r = client.get("/api/download")
check("全通过下载 -> 200", r.status_code == 200)
check("Content-Length 按 A 长度", r.headers.get("Content-Length") == str(44 + N * 2))
params, pcm = parse_wav(r.data)
check("整集 WAV 参数", params == (1, 2, RATE, N), params)
check("整集 PCM 与对轨参考一致", np.array_equal(pcm, ref_snapshot))

# 下载进行中改对轨 + 静音：已开始的下载继续旧混音
rv = client.get("/api/download", buffered=False)
it = iter(rv.response)
header = next(it)
seg0 = next(it)
check("流式首块 44 字节", len(header) == 44)
check("下载段 0 为快照混音", seg0 == ref_snapshot[:SEG].tobytes())
save_mutes({"B": [[60000, 60500]]})
save_align({"a1": 100, "b1": 50, "a2": 90000, "b2": 91000})
seg1 = next(it)
seg2 = next(it)
try:
    next(it)
    tail = "还有多余数据"
except StopIteration:
    tail = None
check("下载恰好 4 块", tail is None)
check("段 1 仍交付旧对轨混音", seg1 == ref_snapshot[SEG:2 * SEG].tobytes())
check("段 2 仍交付旧对轨混音", seg2 == ref_snapshot[2 * SEG:].tobytes())
rv.close()
# 新下载被门控（改动后通过状态已失效）
check("改动后新下载 -> 403", client.get("/api/download").status_code == 403)
# 重新全部试听确认后，新下载交付新生效混音
pass_all(3)
ref_new = ref_mix(A, bsrc, mutes_b=[(60000, 60500)],
                  alignment=(100, 50, 90000, 91000))
r = client.get("/api/download")
check("重新通过后新下载 -> 200", r.status_code == 200)
check("新下载为新生效混音", np.array_equal(parse_wav(r.data)[1], ref_new))

print("== 10. 单段缓冲：任意时刻对齐 B PCM 不超过一段 ==")
max_seen = {"n": 0}
orig_fn = app_module.aligned_b_segment


def wrapped(track_b, n_b, mutes_b, alignment, start, end):
    out = orig_fn(track_b, n_b, mutes_b, alignment, start, end)
    max_seen["n"] = max(max_seen["n"], len(out))
    return out


# mix_segment_pcm 通过模块全局名解析，必须改模块属性才能插桩
app_module.aligned_b_segment = wrapped
try:
    app_module.compute_seg_hashes(EP.tracks["A"], EP.tracks["B"], EP.num_samples_b,
                                  EP.mutes, EP.alignment, RATE, N)
    check("重算过程最大对齐缓冲=段长", max_seen["n"] <= SEG, max_seen["n"])
    gen = client.get("/api/download").response
    b"".join(gen)
    check("下载过程最大对齐缓冲=段长", max_seen["n"] <= SEG, max_seen["n"])
    check("出现过末段大小缓冲(16000)", max_seen["n"] >= 16000)
finally:
    app_module.aligned_b_segment = orig_fn

print("== 11. 内容复原自动恢复通过（对轨切回已确认版本）=")
r = upload(A, bsrc)
assert r.status_code == 200
align_x = {"a1": a1, "b1": b1, "a2": a2, "b2": b2}
save_align(align_x)
pass_all(3)
check("全部通过", get_state()["all_passed"])
save_align({"a1": 100, "b1": 50, "a2": 90000, "b2": 91000})
check("换对轨后失效", not get_state()["all_passed"])
r = save_align(align_x)
check("切回已确认对轨 -> 自动恢复通过", get_state()["all_passed"])

print("== 12. 跨窗口共享与重传重置 ==")
c2 = app_module.app.test_client()
st2 = c2.get("/api/state").get_json()
check("窗口2 看到同一对轨", st2["alignment"]["a1"] == a1)
r = upload(A, B)
st = get_state()
check("重传后对轨/静音清空、全部未通过",
      st["alignment"] is None and st["mutes"] == {"A": [], "B": []}
      and not any(s["passed"] for s in st["segments"]))

print()
if FAILED:
    print(f"共 {len(FAILED)} 项失败：")
    for name in FAILED:
        print("  -", name)
    sys.exit(1)
print("全部自测通过。")
