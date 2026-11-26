"""生成可直接试听的样例：samples/sample_a.wav 与 sample_b.wav。

- A 机：27 秒 @ 44.1 kHz，16 位单声道 PCM（成片时间轴，6 个审听段：
  5×5 秒 + 末段 2 秒）
- B 机：采样率同为 44.1 kHz，但时钟偏快（理想位置 x 的 B 采样号
  b(x) ≈ 1.003·x − 1500，开机晚 1500 个 A 采样），长度与 A 不同，
  用来演示打板线性对轨、头部补零与跨段漂移
- 两次打板（短促脉冲，便于在审听页定位）：
    第一次：A 采样 22050（0.5 s），B 采样 20616
    第二次：A 采样 1080450（24.5 s），B 采样 1082191
  两个编号在各自轨内严格递增；把这四个数填入页面对轨面板即可
- 10.0–11.0 秒（A 时间轴）两轨同相同幅叠加，对轨后仍触发 16 位限幅（第 3 段）
"""

import os
import wave

import numpy as np

RATE = 44100
DUR = 27.0
N_A = int(RATE * DUR)

# B 机时钟模型：B 采样号 = SLOPE * A 位置 + OFFSET
SLOPE = 1.003
OFFSET = -1500.0

# 打板点（A 时间轴位置）
CLAP_A1 = int(0.5 * RATE)       # 22050
CLAP_A2 = int(24.5 * RATE)      # 1080450

rng = np.random.default_rng(20261126)


def add_note(track, freq, start, dur, amp, decay=3.0):
    """在 A 时间轴 [start, start+dur) 上叠加一个乐音。"""
    i0 = int(start * RATE)
    i1 = min(N_A, int((start + dur) * RATE))
    if i1 <= i0:
        return
    t = np.arange(i1 - i0) / RATE
    env = np.exp(-decay * t / dur)
    tone = (np.sin(2 * np.pi * freq * t)
            + 0.3 * np.sin(4 * np.pi * freq * t)
            + 0.1 * np.sin(6 * np.pi * freq * t)) / 1.4
    track[i0:i1] += amp * env * tone


def add_clap(track, center, half=8):
    """在采样号 center 附近放一个短促打板脉冲（向两侧指数衰减）。"""
    lo = max(0, center - half)
    hi = min(len(track), center + half + 1)
    idx = np.arange(lo, hi)
    track[lo:hi] += 0.9 * np.exp(-np.abs(idx - center) / 2.0)


def to_int16(x):
    return (np.clip(x, -1.0, 1.0) * 32767).astype("<i2")


def write_wav(path, samples):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(samples.tobytes())


def resample_to_b(ideal_a):
    """把 A 时间轴上的理想 B 内容按 B 机时钟模型重采样到 B 自身采样网格。

    对每个 B 采样号 j，其对应的 A 位置 x = (j - OFFSET) / SLOPE；
    在理想内容上线性插值（x 落在 [0, N_A) 之外取 0）。
    """
    n_b = int(np.ceil(SLOPE * (N_A - 1) + OFFSET)) + 1
    j = np.arange(n_b, dtype=np.float64)
    x = (j - OFFSET) / SLOPE
    out = np.zeros(n_b, dtype=np.float64)
    valid = (x >= 0.0) & (x <= N_A - 1)
    xv = x[valid]
    k = np.minimum(np.floor(xv).astype(np.int64), N_A - 2)
    frac = xv - k
    out[valid] = ideal_a[k] + (ideal_a[k + 1] - ideal_a[k]) * frac
    return out


def main():
    track_a = np.zeros(N_A)
    ideal_b = np.zeros(N_A)

    # A 机内容：五声音阶随机漫步旋律，每 0.45 秒一个音
    scale = [440.0, 523.25, 587.33, 659.25, 783.99, 880.0]
    pos = 2
    t = 0.0
    while t < DUR - 0.5:
        add_note(track_a, scale[pos], t, 0.5, 0.45)
        pos = int(np.clip(pos + rng.integers(-2, 3), 0, len(scale) - 1))
        t += 0.45

    # B 机内容：Am - F - C - G 低音进行 + 轻底噪
    roots = [110.0, 87.31, 130.81, 98.0]
    t = 0.0
    k = 0
    while t < DUR:
        add_note(ideal_b, roots[k % 4], t, 1.7, 0.5, decay=1.5)
        add_note(ideal_b, roots[k % 4] * 1.5, t, 1.7, 0.18, decay=1.5)
        t += 1.8
        k += 1
    ideal_b += 0.015 * rng.standard_normal(N_A)

    # 10.0–11.0 秒：两轨叠加同相 220 Hz 大幅正弦（对轨后仍在同一段内触发限幅）
    i0, i1 = int(10.0 * RATE), int(11.0 * RATE)
    tt = np.arange(i1 - i0) / RATE
    loud = 0.85 * np.sin(2 * np.pi * 220.0 * tt)
    track_a[i0:i1] += loud
    ideal_b[i0:i1] += loud

    # 打板：先在两轨的 A 时间轴内容上打点，B 的点随重采样落到 B 采样网格
    add_clap(track_a, CLAP_A1)
    add_clap(track_a, CLAP_A2)
    add_clap(ideal_b, CLAP_A1)
    add_clap(ideal_b, CLAP_A2)

    track_b_float = resample_to_b(ideal_b)

    # B 轨打板采样号（导演在 B 机上读到的编号，严格递增）
    clap_b1 = int(round(SLOPE * CLAP_A1 + OFFSET))
    clap_b2 = int(round(SLOPE * CLAP_A2 + OFFSET))
    # 重采样会把脉冲摊薄，在 B 自己的采样网格上把两个打板点补成同样的脉冲
    add_clap(track_b_float, clap_b1)
    add_clap(track_b_float, clap_b2)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    write_wav(os.path.join(out_dir, "sample_a.wav"), to_int16(track_a))
    write_wav(os.path.join(out_dir, "sample_b.wav"), to_int16(track_b_float))

    print("sample_a.wav:", N_A, "samples")
    print("sample_b.wav:", len(track_b_float), "samples")
    print("clap on A:", CLAP_A1, CLAP_A2)
    print("clap on B:", clap_b1, clap_b2)


if __name__ == "__main__":
    main()
