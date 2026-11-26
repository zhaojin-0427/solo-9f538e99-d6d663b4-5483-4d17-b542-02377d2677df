"""广播剧混音审听台 —— Flask 后端（含打板线性对轨）。

声音导演交片前逐段审听：
- 上传同一集的两条 16 位单声道 PCM WAV（轨 A / 轨 B），同采样率即可，长度可不同
- 按两次打板在两轨上的采样号 (a1,b1)、(a2,b2) 把 B 轨线性换算到 A 轨时间轴：
  成片沿用 A 轨的时间轴与长度；B 取样位置落在首末样点之外补零，轨内线性插值
  后四舍五入（half away from zero）
- B 轨静音始终按 B 原轨采样号，在插值前置零；A 轨静音按 A（成片）采样号
- 两轨相加并限幅到 16 位得到混音；混音按 5 秒一段（末段可不足 5 秒）
- 逐段试听并确认通过；确认时校验"刚试听的 PCM"与该段当前内容一致（哈希）
- 对轨与静音交错保存：提交带版本号，提交时若另一项设置已变 -> 409 并回传新
  状态，提示按新设置重试；任何冲突失败都不改变当时生效的混音
- 换集（重新上传）后旧集的对轨提交一律拒绝
- 全部段落通过后才可下载整集 WAV；下载开始即快照（轨道引用 + 静音表 + 对齐
  参数），按段流式生成，内存中最多另存一段对齐后的 B 轨 PCM
"""

import hashlib
import os
import struct
import threading
import uuid
import wave

import numpy as np
from flask import Flask, Response, jsonify, render_template, request

SEGMENT_SECONDS = 5
INT16_MIN, INT16_MAX = -32768, 32767
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.environ.get("SAMPLES_DIR", os.path.normpath(os.path.join(BASE_DIR, "..", "samples")))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


# ---------------------------------------------------------------------------
# WAV 读取
# ---------------------------------------------------------------------------

def read_wav_16mono(fileobj):
    """读取并校验 16 位单声道 PCM WAV，返回 (int16 numpy 数组, 采样率)。"""
    try:
        with wave.open(fileobj, "rb") as w:
            if w.getcomptype() != "NONE":
                raise ValueError("只支持未压缩的 PCM WAV")
            if w.getnchannels() != 1:
                raise ValueError(f"只支持单声道（该文件为 {w.getnchannels()} 声道）")
            if w.getsampwidth() != 2:
                raise ValueError(f"只支持 16 位采样（该文件为 {w.getsampwidth() * 8} 位）")
            rate = w.getframerate()
            nframes = w.getnframes()
            raw = w.readframes(nframes)
    except wave.Error as exc:
        raise ValueError(f"无法解析 WAV 文件：{exc}") from exc
    data = np.frombuffer(raw, dtype="<i2").copy()
    if data.size != nframes:
        raise ValueError("WAV 数据长度与文件头不符")
    return data, rate


# ---------------------------------------------------------------------------
# 对轨与混音（纯函数，便于下载快照在无锁状态下使用）
#
# 内存约定：所有按段函数只分配"一段"大小的工作缓冲；调用方逐段调用、用完即
# 释放，应用层在任意时刻最多另存一段对齐后的 B 轨 PCM。
# ---------------------------------------------------------------------------

def _round_half_away(values):
    """四舍五入（half away from zero）：2.5->3，-2.5->-3。"""
    return np.where(values >= 0.0,
                    np.floor(values + 0.5),
                    np.ceil(values - 0.5)).astype(np.int32)


def _muted_index_mask(idx, mutes):
    """判断 B 原轨采样号数组 idx 中哪些落在静音区间（区间互不重叠、已排序）。"""
    if not mutes:
        return np.zeros(idx.shape, dtype=bool)
    starts = np.fromiter((m[0] for m in mutes), dtype=np.int64, count=len(mutes))
    ends = np.fromiter((m[1] for m in mutes), dtype=np.int64, count=len(mutes))
    # 区间 [s, e) 内 <=> 不超过 i 的起点数 > 不超过 i 的终点数
    return (np.searchsorted(starts, idx, side="right")
            > np.searchsorted(ends, idx, side="right"))


def aligned_b_segment(track_b, n_b, mutes_b, alignment, start, end):
    """生成 A（成片）坐标 [start, end) 上对齐后的 B 轨 int32 缓冲。

    alignment 为 None 时采用恒等映射 b(x)=x（不对轨）。
    步骤：线性换算 B 取样位置 -> 首末样点之外补零 -> B 静音按原轨采样号
    在插值前置零 -> 轨内线性插值 -> 四舍五入。
    """
    x = np.arange(start, end, dtype=np.float64)
    if alignment is None:
        pos = x
    else:
        a1, b1, a2, b2 = alignment
        slope = (b2 - b1) / (a2 - a1)
        pos = b1 + (x - a1) * slope

    out = np.zeros(end - start, dtype=np.int32)
    valid = (pos >= 0.0) & (pos <= n_b - 1)
    if not valid.any():
        return out

    if n_b == 1:
        # 唯一合法位置就是 0；静音按原轨采样号处理，其余位置补零
        zero_pos = (pos == 0.0)
        if zero_pos.any() and not _muted_index_mask(np.zeros(1, dtype=np.int64), mutes_b)[0]:
            out[zero_pos] = int(track_b[0])
        return out

    pv = pos[valid]
    k = np.floor(pv).astype(np.int64)
    at_last = k >= n_b - 1
    k0 = np.minimum(k, n_b - 2)          # 插值左样点（末端点回退到 n_b-2）
    k1 = k0 + 1                          # 插值右样点
    frac = pv - k

    src0 = track_b[k0].astype(np.float64)
    src1 = track_b[k1].astype(np.float64)
    # B 静音：先把原轨静音样点置零，再做插值
    m0 = _muted_index_mask(k0, mutes_b)
    m1 = _muted_index_mask(k1, mutes_b)
    if m0.any():
        src0[m0] = 0.0
    if m1.any():
        src1[m1] = 0.0
    val = src0 + (src1 - src0) * frac
    # 末端样点没有 k+1，直接取该样点（若被静音则为 0）
    if at_last.any():
        last_val = 0.0 if _muted_index_mask(
            np.array([n_b - 1], dtype=np.int64), mutes_b)[0] else float(track_b[n_b - 1])
        val[at_last] = last_val
    out[valid] = _round_half_away(val)
    return out


def apply_mutes(buf, mutes, seg_start):
    """把 A（成片）绝对采样坐标系下的静音区间应用到段缓冲（int32）上。"""
    seg_end = seg_start + buf.shape[0]
    for s, e in mutes:
        lo = max(s, seg_start) - seg_start
        hi = min(e, seg_end) - seg_start
        if hi > lo:
            buf[lo:hi] = 0


def mix_segment_pcm(track_a, track_b, n_b, mutes_a, mutes_b, alignment, start, end):
    """生成 [start, end) 区间的混音 PCM 字节。

    A 轨静音按成片坐标置零；B 轨先按原轨坐标静音、再线性对轨（含补零/插值/
    四舍五入）；两轨相加 -> 限幅到 16 位。
    """
    sa = track_a[start:end].astype(np.int32)
    apply_mutes(sa, mutes_a, start)
    sb = aligned_b_segment(track_b, n_b, mutes_b, alignment, start, end)
    mixed = sa + sb
    np.clip(mixed, INT16_MIN, INT16_MAX, out=mixed)
    return mixed.astype("<i2").tobytes()


def pcm_hash(pcm):
    return hashlib.sha256(pcm).hexdigest()


def wav_header(num_samples, sample_rate):
    """44 字节规范 PCM WAV 头（单声道 16 位）。"""
    data_size = num_samples * 2
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b"data", data_size,
    )


def compute_seg_hashes(track_a, track_b, n_b, mutes, alignment, sample_rate, num_samples):
    """逐段计算当前混音哈希；任意时刻只保留一段对齐 B PCM 工作缓冲。"""
    seg_len = sample_rate * SEGMENT_SECONDS
    hashes = []
    for start in range(0, num_samples, seg_len):
        end = min(start + seg_len, num_samples)
        pcm = mix_segment_pcm(track_a, track_b, n_b,
                              mutes["A"], mutes["B"], alignment, start, end)
        hashes.append(pcm_hash(pcm))
    return hashes


# ---------------------------------------------------------------------------
# 剧集状态（单剧集、跨窗口共享；所有可变字段仅在锁内读写）
# ---------------------------------------------------------------------------

class Episode:
    def __init__(self):
        self.lock = threading.RLock()
        self.loaded = False
        self.episode_id = ""                # 每次上传换集重新生成，旧集提交据此拒绝
        self.sample_rate = 0
        self.num_samples = 0                # 成片长度 = A 轨长度，时间轴以 A 为准
        self.num_samples_b = 0
        self.tracks = {"A": None, "B": None}   # int16 numpy 数组，载入后不再原地修改
        self.mutes = {"A": [], "B": []}        # [(start, end), ...]，保存时整体替换
        self.alignment = None                   # None=恒等映射；否则 (a1,b1,a2,b2)
        self.seg_hashes = []
        self.confirmed_hashes = []
        self.version = 0                       # 任何设置改动都递增，交错保存的乐观锁
        # 测试缝：重型哈希计算（锁外）开始前调用一次，用于模拟"计算期间他处改设置"
        self.compute_hook = None
        self._hook_in_progress = False

    def seg_len(self):
        return self.sample_rate * SEGMENT_SECONDS

    def is_passed(self, i):
        # 当前混音内容曾被试听确认过即视为通过：设置改动后内容未变的段自动保持
        # 通过；内容改回某个已确认版本时也自动恢复通过。
        return self.seg_hashes[i] in self.confirmed_hashes[i]

    def all_passed(self):
        return bool(self.seg_hashes) and all(self.is_passed(i) for i in range(len(self.seg_hashes)))

    def alignment_dict(self):
        if self.alignment is None:
            return None
        a1, b1, a2, b2 = self.alignment
        return {"a1": a1, "b1": b1, "a2": a2, "b2": b2,
                "slope": (b2 - b1) / (a2 - a1)}

    def to_dict(self):
        """调用方须持有锁。"""
        seg_len = self.seg_len()
        segments = []
        for i, h in enumerate(self.seg_hashes):
            start = i * seg_len
            end = min(start + seg_len, self.num_samples)
            segments.append({
                "index": i,
                "start_sample": start,
                "end_sample": end,
                "t0": start / self.sample_rate,
                "t1": end / self.sample_rate,
                "hash": h,
                "passed": self.is_passed(i),
            })
        return {
            "loaded": True,
            "episode_id": self.episode_id,
            "sample_rate": self.sample_rate,
            "num_samples": self.num_samples,
            "num_samples_b": self.num_samples_b,
            "duration": self.num_samples / self.sample_rate,
            "segment_seconds": SEGMENT_SECONDS,
            "segments": segments,
            "all_passed": self.all_passed(),
            "mutes": {"A": [list(r) for r in self.mutes["A"]],
                      "B": [list(r) for r in self.mutes["B"]]},
            "alignment": self.alignment_dict(),
            "version": self.version,
        }


EP = Episode()


def install_episode(track_a, track_b, rate):
    """校验通过后整体替换剧集（新集 ID、空静音、恒等对轨）。"""
    if track_a.shape[0] == 0 or track_b.shape[0] == 0:
        return None, (jsonify({"error": "音频为空"}), 400)
    n_a = int(track_a.shape[0])
    n_b = int(track_b.shape[0])
    hashes = compute_seg_hashes(track_a, track_b, n_b,
                                {"A": [], "B": []}, None, rate, n_a)
    with EP.lock:
        EP.tracks = {"A": track_a, "B": track_b}
        EP.sample_rate = rate
        EP.num_samples = n_a
        EP.num_samples_b = n_b
        EP.mutes = {"A": [], "B": []}
        EP.alignment = None
        EP.seg_hashes = hashes
        EP.confirmed_hashes = [set() for _ in hashes]
        EP.episode_id = uuid.uuid4().hex
        EP.loaded = True
        EP.version += 1
        return EP.to_dict(), None


# ---------------------------------------------------------------------------
# 页面与 API
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def api_state():
    with EP.lock:
        if not EP.loaded:
            return jsonify({"loaded": False})
        return jsonify(EP.to_dict())


@app.post("/api/upload")
def api_upload():
    if "track_a" not in request.files or "track_b" not in request.files:
        return jsonify({"error": "需要同时上传 track_a 与 track_b 两个 WAV 文件"}), 400
    try:
        track_a, rate_a = read_wav_16mono(request.files["track_a"])
    except ValueError as exc:
        return jsonify({"error": f"轨 A：{exc}"}), 400
    try:
        track_b, rate_b = read_wav_16mono(request.files["track_b"])
    except ValueError as exc:
        return jsonify({"error": f"轨 B：{exc}"}), 400
    if rate_a != rate_b:
        return jsonify({"error": f"两轨采样率不一致（{rate_a} Hz vs {rate_b} Hz）"}), 400
    # 长度允许不同：成片以 A 轨时间轴与长度为准，B 经打板对轨后越界补零
    state, err = install_episode(track_a, track_b, rate_a)
    if err:
        return err
    return jsonify(state)


@app.post("/api/load_sample")
def api_load_sample():
    try:
        with open(os.path.join(SAMPLES_DIR, "sample_a.wav"), "rb") as f:
            track_a, rate_a = read_wav_16mono(f)
        with open(os.path.join(SAMPLES_DIR, "sample_b.wav"), "rb") as f:
            track_b, rate_b = read_wav_16mono(f)
    except FileNotFoundError:
        return jsonify({"error": "样例文件缺失（samples/sample_a.wav, sample_b.wav）"}), 500
    except ValueError as exc:
        return jsonify({"error": f"样例文件损坏：{exc}"}), 500
    if rate_a != rate_b:
        return jsonify({"error": "样例采样率不一致"}), 500
    state, err = install_episode(track_a, track_b, rate_a)
    if err:
        return err
    return jsonify(state)


def _as_int(value):
    if isinstance(value, bool):
        raise ValueError
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise ValueError


def _snapshot_context():
    """锁内抓取计算所需的不可变上下文。"""
    return {
        "episode_id": EP.episode_id,
        "version": EP.version,
        "tracks": EP.tracks,
        "rate": EP.sample_rate,
        "n_a": EP.num_samples,
        "n_b": EP.num_samples_b,
        "mutes": {"A": list(EP.mutes["A"]), "B": list(EP.mutes["B"])},
        "alignment": EP.alignment,
    }


def _conflict_response(ctx, *, episode_changed):
    """冲突失败：不修改任何状态，回传当前生效状态供前端重试。"""
    with EP.lock:
        if not EP.loaded:
            return jsonify({"error": "剧集已被更换，请刷新页面", "conflict": True}), 409
        payload = {"conflict": True, "state": EP.to_dict()}
    if episode_changed:
        payload["error"] = "该提交属于旧集，已被拒绝（当前生效混音保持不变）"
    else:
        payload["error"] = "另一项设置已变更，请按当前设置核对后重试"
    return jsonify(payload), 409


def _check_client_version(body, ctx):
    """请求携带 version 时做第一道乐观锁校验；返回错误响应或 None。"""
    if "version" not in body:
        return None
    try:
        client_version = _as_int(body["version"])
    except ValueError:
        return jsonify({"error": "version 须为整数"}), 400
    if client_version != ctx["version"]:
        return _conflict_response(ctx, episode_changed=False)
    return None


def _run_compute_hook():
    # 重入保护：hook 内发起的保存请求也会走到这里，跳过内层调用
    if EP._hook_in_progress:
        return
    hook = EP.compute_hook
    if hook is None:
        return
    EP._hook_in_progress = True
    try:
        hook()
    finally:
        EP._hook_in_progress = False


@app.post("/api/mutes")
def api_save_mutes():
    """保存静音改动。任一区间非法则整体失败，原混音与通过状态保持不变。

    请求体：{"A": [[start, end], ...], "B": [...], "version": 可选}；
    缺省的轨保持原样。A 区间按成片（A 轨）采样号校验，B 区间按 B 原轨采样号
    校验。携带 version 且与服务端不一致 -> 409 冲突，回传新状态供重试。
    """
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "请求体须为 JSON 对象"}), 400
    with EP.lock:
        if not EP.loaded:
            return jsonify({"error": "尚未上传音频"}), 409
        ctx = _snapshot_context()

    verr = _check_client_version(body, ctx)
    if verr is not None:
        return verr

    bounds = {"A": ctx["n_a"], "B": ctx["n_b"]}
    new_mutes = {}
    for t in ("A", "B"):
        if t not in body:
            new_mutes[t] = ctx["mutes"][t]
            continue
        raw_list = body[t]
        if not isinstance(raw_list, list):
            return jsonify({"error": f"轨 {t} 的静音列表格式错误"}), 400
        bound = bounds[t]
        ranges = []
        for item in raw_list:
            try:
                s, e = _as_int(item[0]), _as_int(item[1])
            except (ValueError, TypeError, IndexError):
                return jsonify({"error": f"轨 {t} 存在非法的静音区间：{item!r}"}), 400
            if not (0 <= s < e <= bound):
                coord = "成片（A 轨）" if t == "A" else "B 原轨"
                return jsonify({"error": f"轨 {t} 静音区间 [{s}, {e}) 须满足 "
                                         f"0 ≤ 起 < 止 ≤ {bound}（{coord}采样号）"}), 400
            ranges.append((s, e))
        ranges.sort()
        new_mutes[t] = ranges

    # 锁外按段重算哈希（不阻塞其他窗口的试听/确认/保存）；任何异常都发生在提交前
    _run_compute_hook()
    new_hashes = compute_seg_hashes(ctx["tracks"]["A"], ctx["tracks"]["B"], ctx["n_b"],
                                    new_mutes, ctx["alignment"], ctx["rate"], ctx["n_a"])

    with EP.lock:
        if not EP.loaded:
            return jsonify({"error": "剧集已被更换，请刷新页面", "conflict": True}), 409
        if EP.episode_id != ctx["episode_id"]:
            return _conflict_response(ctx, episode_changed=True)
        if EP.tracks is not ctx["tracks"] or EP.version != ctx["version"]:
            # 计算期间另一项设置（对轨/静音）已提交
            return _conflict_response(ctx, episode_changed=False)
        old_hashes = EP.seg_hashes
        invalidated = [i for i in range(len(new_hashes))
                       if old_hashes[i] in EP.confirmed_hashes[i]
                       and new_hashes[i] not in EP.confirmed_hashes[i]]
        EP.mutes = new_mutes
        EP.seg_hashes = new_hashes
        EP.version += 1
        state = EP.to_dict()
    return jsonify({"ok": True, "invalidated": invalidated, "state": state})


def _parse_alignment(body, n_a, n_b):
    """校验对轨参数，返回 (alignment 或 None, 错误响应)。"""
    if body.get("clear") is True:
        return None, None
    try:
        a1, b1, a2, b2 = (_as_int(body[k]) for k in ("a1", "b1", "a2", "b2"))
    except (KeyError, ValueError, TypeError):
        return None, (jsonify({"error": "需要整数采样号 a1/b1/a2/b2（或 {\"clear\": true} 清除对轨）"}), 400)
    if not (0 <= a1 < a2 < n_a):
        return None, (jsonify({"error": f"A 轨打板点须满足 0 ≤ a1 < a2 < {n_a}（当前 {a1}, {a2}）"}), 400)
    if not (0 <= b1 < b2 < n_b):
        return None, (jsonify({"error": f"B 轨打板点须满足 0 ≤ b1 < b2 < {n_b}（当前 {b1}, {b2}）"}), 400)
    return (a1, b1, a2, b2), None


@app.post("/api/alignment")
def api_save_alignment():
    """保存打板对轨。成片时间轴与长度始终沿用 A 轨。

    请求体：{"a1", "b1", "a2", "b2", "version"?} 或 {"clear": true, "version"?}。
    版本不一致 / 计算期间他处改动 / 属于旧集 -> 409，当前生效混音不变。
    """
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "请求体须为 JSON 对象"}), 400
    with EP.lock:
        if not EP.loaded:
            return jsonify({"error": "尚未上传音频"}), 409
        ctx = _snapshot_context()

    verr = _check_client_version(body, ctx)
    if verr is not None:
        return verr

    new_alignment, ferr = _parse_alignment(body, ctx["n_a"], ctx["n_b"])
    if ferr is not None:
        return ferr

    # 锁外按段重算：长录音下不持全局锁，其他窗口仍可保存静音、试听、确认
    _run_compute_hook()
    new_hashes = compute_seg_hashes(ctx["tracks"]["A"], ctx["tracks"]["B"], ctx["n_b"],
                                    ctx["mutes"], new_alignment, ctx["rate"], ctx["n_a"])

    with EP.lock:
        if not EP.loaded:
            return jsonify({"error": "剧集已被更换，请刷新页面", "conflict": True}), 409
        if EP.episode_id != ctx["episode_id"]:
            return _conflict_response(ctx, episode_changed=True)
        if EP.tracks is not ctx["tracks"] or EP.version != ctx["version"]:
            return _conflict_response(ctx, episode_changed=False)
        old_hashes = EP.seg_hashes
        invalidated = [i for i in range(len(new_hashes))
                       if old_hashes[i] in EP.confirmed_hashes[i]
                       and new_hashes[i] not in EP.confirmed_hashes[i]]
        EP.alignment = new_alignment
        EP.seg_hashes = new_hashes
        EP.version += 1
        state = EP.to_dict()
    return jsonify({"ok": True, "invalidated": invalidated, "state": state})


@app.get("/api/segment/<int:idx>.wav")
def api_segment(idx):
    with EP.lock:
        if not EP.loaded:
            return jsonify({"error": "尚未上传音频"}), 409
        nseg = len(EP.seg_hashes)
        if not (0 <= idx < nseg):
            return jsonify({"error": "段落不存在"}), 404
        tracks = EP.tracks
        n_b = EP.num_samples_b
        mutes = {"A": list(EP.mutes["A"]), "B": list(EP.mutes["B"])}
        alignment = EP.alignment
        rate = EP.sample_rate
        n = EP.num_samples
    seg_len = rate * SEGMENT_SECONDS
    start = idx * seg_len
    end = min(start + seg_len, n)
    pcm = mix_segment_pcm(tracks["A"], tracks["B"], n_b,
                          mutes["A"], mutes["B"], alignment, start, end)
    return Response(
        wav_header(end - start, rate) + pcm,
        mimetype="audio/wav",
        headers={
            "X-Mix-Hash": pcm_hash(pcm),
            "Cache-Control": "no-store",
            "Content-Disposition": f'inline; filename="segment_{idx + 1}.wav"',
        },
    )


@app.post("/api/confirm/<int:idx>")
def api_confirm(idx):
    """确认通过：客户端回传刚试听到的混音哈希，须与该段当前内容一致。"""
    body = request.get_json(silent=True) or {}
    client_hash = str(body.get("hash", ""))
    with EP.lock:
        if not EP.loaded:
            return jsonify({"error": "尚未上传音频"}), 409
        if not (0 <= idx < len(EP.seg_hashes)):
            return jsonify({"error": "段落不存在"}), 404
        current = EP.seg_hashes[idx]
        if client_hash != current:
            return jsonify({"error": "该段内容已变化，请重新试听后再确认",
                            "current_hash": current}), 409
        EP.confirmed_hashes[idx].add(client_hash)
        return jsonify(EP.to_dict())


@app.get("/api/download")
def api_download():
    """全部段落通过后才可下载。开始下载即快照当时通过的混音来源（轨道引用 +
    静音表 + 对轨参数），之后任何对轨/静音改动/重新上传都不影响本次下载；
    逐段生成，内存中最多另存一段对齐后的 B 轨 PCM。"""
    with EP.lock:
        if not EP.loaded:
            return jsonify({"error": "尚未上传音频"}), 409
        if not EP.all_passed():
            return jsonify({"error": "仍有段落未通过审听，不能下载"}), 403
        tracks = EP.tracks                # 数组载入后不原地修改，引用即快照
        n_b = EP.num_samples_b
        mutes_a = list(EP.mutes["A"])      # 静音表保存时整体替换，拷贝列表即快照
        mutes_b = list(EP.mutes["B"])
        alignment = EP.alignment
        rate = EP.sample_rate
        n = EP.num_samples

    seg_len = rate * SEGMENT_SECONDS
    track_a, track_b = tracks["A"], tracks["B"]

    def generate():
        yield wav_header(n, rate)
        for start in range(0, n, seg_len):
            end = min(start + seg_len, n)
            yield mix_segment_pcm(track_a, track_b, n_b,
                                  mutes_a, mutes_b, alignment, start, end)

    return Response(
        generate(),
        mimetype="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="episode_mix.wav"',
            "Content-Length": str(44 + n * 2),
            "Cache-Control": "no-store",
        },
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "文件过大（上限 512 MB）"}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=port, threads=8)
    except ImportError:
        app.run(host="0.0.0.0", port=port, threaded=True)
