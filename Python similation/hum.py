import numpy as np
import librosa
from scipy.signal import find_peaks

def _runs_from_mask(mask):
    """返回 mask=True 的连续段列表 [(s,e), ...]，e为开区间"""
    runs = []
    i = 0
    N = len(mask)
    while i < N:
        if not mask[i]:
            i += 1
            continue
        s = i
        while i < N and mask[i]:
            i += 1
        e = i
        runs.append((s, e))
    return runs


def _is_hum_tonal_bursts(y, sr, hop=256, n_fft=1024,
                         voiced_k=0.18,          # voiced阈值 = noise_floor + voiced_k*dyn
                         flat_thr=0.28,          # flatness越小越“有音高”
                         zcr_thr=0.14,           # hum通常zcr偏低
                         min_voiced_seg_sec=0.12,
                         min_seg_count=2,        # 至少出现2段（断断续续的hum）
                         tonal_ratio_thr=0.60):  # voiced里有多少比例是“tonal”
    """
    针对你这种 hum：短暂、间隔回0、音高不稳定
    用“tonal(低flatness)+低ZCR”来识别，不要求稳定f0
    """
    eps = 1e-9

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    if rms.size < 10:
        return False, {}

    noise_floor = float(np.percentile(rms, 10))
    peak_max = float(np.max(rms))
    dyn = max(peak_max - noise_floor, eps)

    frame_sec = hop / sr
    voiced = rms > (noise_floor + voiced_k * dyn)

    # 计算 flatness + zcr（逐帧）
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) + eps
    flat = librosa.feature.spectral_flatness(S=S)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)[0]

    # 只看 voiced 帧
    v_flat = flat[voiced]
    v_zcr = zcr[voiced]

    if v_flat.size < 5:
        return False, {
            "voiced_ratio": float(np.mean(voiced)),
            "tonal_ratio": 0.0,
            "zcr_med": 1.0,
            "reason": "too few voiced frames"
        }

    tonal_mask = (v_flat < flat_thr) & (v_zcr < zcr_thr)
    tonal_ratio = float(np.mean(tonal_mask))
    voiced_ratio = float(np.mean(voiced))
    zcr_med = float(np.median(v_zcr))

    # 统计 voiced 的连续段数量（断断续续出现几次）
    runs = _runs_from_mask(voiced)
    min_frames = int(min_voiced_seg_sec / frame_sec)
    good_runs = [(s, e) for (s, e) in runs if (e - s) >= min_frames]
    seg_count = len(good_runs)

    # 判定：必须“有足够tonal的voiced帧” + “至少出现多段”
    ok = (tonal_ratio >= tonal_ratio_thr) and (seg_count >= min_seg_count)

    info = {
        "voiced_ratio": round(voiced_ratio, 3),
        "tonal_ratio": round(tonal_ratio, 3),
        "zcr_med": round(zcr_med, 3),
        "seg_count": int(seg_count),
        "flat_thr": flat_thr,
        "zcr_thr": zcr_thr,
        "reason": "ok" if ok else "not tonal enough / not enough segments"
    }
    return ok, info


def _burst_shape_ok(rms, pk, noise_floor, frame_sec, dyn,
                    win_sec=1.5,         # ✅ 窗口变大：适配“慢慢爬升”的哭声
                    floor_tol=0.25,      # ✅ 放宽：谷值允许离底噪稍远一点
                    rise_thr=0.18,       # ✅ 放宽：rise 幅度阈值
                    fall_thr=0.18,       # ✅ 放宽：fall 幅度阈值
                    up_ratio_thr=0.55,   # ✅ 上升趋势比例
                    down_ratio_thr=0.55  # ✅ 下降趋势比例
                    ):
    """
    判断一个峰是否符合 cry 的“爬升->下降回到底噪”形状（只用 RMS）
    """
    N = len(rms)
    w = int(win_sec / frame_sec)
    l0 = max(0, pk - w)
    r0 = min(N, pk + w + 1)

    if pk - l0 < 5 or r0 - pk < 5:
        return False, {}

    # 找左谷/右谷（同时拿到位置）
    left_seg = rms[l0:pk]
    right_seg = rms[pk:r0]

    li = int(np.argmin(left_seg)) + l0
    ri = int(np.argmin(right_seg)) + pk

    left_min = float(rms[li])
    right_min = float(rms[ri])
    peak = float(rms[pk])

    rise = peak - left_min
    fall = peak - right_min

    # 基础幅度门槛
    if rise < rise_thr * dyn or fall < fall_thr * dyn:
        return False, {
            "reason": "rise/fall too small",
            "left_min": left_min, "right_min": right_min, "peak": peak,
            "rise": rise, "fall": fall, "li": li, "ri": ri
        }

    # 回落到底噪附近（谷值接近底噪）
    floor_limit = noise_floor + floor_tol * dyn
    if left_min > floor_limit or right_min > floor_limit:
        return False, {
            "reason": "not return to floor",
            "left_min": left_min, "right_min": right_min, "peak": peak,
            "floor_limit": floor_limit, "li": li, "ri": ri
        }

    # ✅ “爬升”趋势：左谷->峰 大部分差分为正
    up = np.diff(rms[li:pk+1])
    down = np.diff(rms[pk:ri+1])

    up_ratio = float(np.mean(up > 0)) if up.size else 0.0
    down_ratio = float(np.mean(down < 0)) if down.size else 0.0

    ok = (up_ratio >= up_ratio_thr) and (down_ratio >= down_ratio_thr)

    info = {
        "left_min": round(left_min, 6),
        "right_min": round(right_min, 6),
        "peak": round(peak, 6),
        "rise": round(rise, 6),
        "fall": round(fall, 6),
        "up_ratio": round(up_ratio, 3),
        "down_ratio": round(down_ratio, 3),
        "li": int(li),
        "ri": int(ri),
        "reason": "ok" if ok else "trend not strong",
    }
    return ok, info


def classify_cry_laugh(wav_path, sr=16000, hop=256, n_fft=1024):
    y, _ = librosa.load(wav_path, sr=sr, mono=True)

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    if rms.size < 10:
        return "unknown (too short)", {}

    frame_sec = hop / sr
    duration_sec = len(y) / sr
    eps = 1e-9

    noise_floor = float(np.percentile(rms, 10))
    peak_max = float(np.max(rms))
    dyn = max(peak_max - noise_floor, eps)

    height_thr = noise_floor + 0.2 * dyn
    prom_thr = 0.1 * dyn
    peaks, props = find_peaks(rms, height=height_thr, prominence=prom_thr, distance=2)

    # ===== Stage 0: hum (NEW) =====
    is_hum, hum_info = _is_hum_tonal_bursts(y, sr, hop=hop, n_fft=n_fft)

    # ===== Stage 1: laugh =====
    peaks_per_sec = float(len(peaks) / max(duration_sec, 1e-6))

    # 这些阈值你后面可以再微调
    FAST_GAP_SEC = 0.35          # “连续哈”的峰间隔上限
    MIN_CLUSTER_PEAKS = 3        # 一坨至少3个峰才算“连续峰簇”
    MIN_CLUSTER_COUNT = 2        # 这种峰簇出现>=2次 => laugh
    LONG_RUN_PEAKS = 4           # 单次连续峰 >=4 也可直接算 laugh
    MIN_TOTAL_PEAKS = 6          # 总峰数太少就别判 laugh（避免误判）

    if len(peaks) >= 3:
        intervals = np.diff(peaks) * frame_sec
        short_mask = intervals < FAST_GAP_SEC
        short_intervals = intervals[short_mask]

        fast_repeat_ratio = float(np.mean(short_mask))          # 短间隔占比
        median_interval = float(np.median(intervals))           # 全局间隔中位数

        # 只对短间隔算稳定性（避免中间大停顿拉爆 CV）
        if short_intervals.size >= 2:
            cv_fast = float(np.std(short_intervals) / (np.mean(short_intervals) + eps))
        else:
            cv_fast = 10.0

        # 计算连续短间隔的最长 run（run 长度对应“连续峰”个数）
        max_fast_run = 1
        cur = 1
        for is_short in short_mask:
            if is_short:
                cur += 1
                max_fast_run = max(max_fast_run, cur)
            else:
                cur = 1

        # ✅ 新增：计算“短连续峰簇”的数量（允许中间有很大间隔）
        # 每个 True-run 表示一坨连续峰，峰数 = run_len + 1
        cluster_count = 0
        i = 0
        while i < len(short_mask):
            if short_mask[i]:
                run_start = i
                while i < len(short_mask) and short_mask[i]:
                    i += 1
                run_len = i - run_start
                cluster_peaks = run_len + 1
                if cluster_peaks >= MIN_CLUSTER_PEAKS:
                    cluster_count += 1
            else:
                i += 1

    else:
        fast_repeat_ratio = 0.0
        median_interval = 999.0
        cv_fast = 10.0
        max_fast_run = 1
        cluster_count = 0

    # 原来那套“典型 laugh（峰多+快+稳定）”规则（保留）
    old_laugh_rule = (
        (peaks_per_sec >= 2.0) and
        (fast_repeat_ratio >= 0.55) and
        ((cv_fast <= 0.35) or (max_fast_run >= LONG_RUN_PEAKS) or (median_interval <= 0.30))
    )

    # ✅ 新规则：就算整体峰率不高，只要“短连续峰簇”出现 >=2 次，也判 laugh
    cluster_laugh_rule = (
        (len(peaks) >= MIN_TOTAL_PEAKS) and
        (cluster_count >= MIN_CLUSTER_COUNT) and
        (max_fast_run >= MIN_CLUSTER_PEAKS) and
        (fast_repeat_ratio >= 0.40)
    )

    is_laugh = old_laugh_rule or cluster_laugh_rule


# ===== Stage 2: cry =====
    cry_good = 0
    cry_checked = 0
    per_peak_debug = []

    if (not is_laugh) and (len(peaks) >= 1):
        for pk in peaks:
            ok, dbg = _burst_shape_ok(rms, int(pk), noise_floor, frame_sec, dyn)
            cry_checked += 1
            cry_good += int(ok)
            per_peak_debug.append({"pk": int(pk), "ok": ok, **dbg})

        burst_ratio = cry_good / max(cry_checked, 1)

        # ✅ 峰很少时（<=3），只要命中>=1且比例>=0.34 就算 cry
        if len(peaks) <= 3:
            is_cry = (cry_good >= 1) and (burst_ratio >= 0.34)
        else:
            is_cry = (cry_good >= 2) and (burst_ratio >= 0.60)
    else:
        burst_ratio = 0.0
        is_cry = False


    if is_hum:
        label = "hum"
    elif is_laugh:
        label = "laugh"
    elif is_cry:
        label = "cry"
    else:
        label = "unknown"


    info = {
        "duration_sec": round(duration_sec, 3),
        "noise_floor": round(noise_floor, 6),
        "peak_max": round(peak_max, 6),
        "dyn": round(dyn, 6),
        "num_peaks": int(len(peaks)),

        "peaks_per_sec": round(peaks_per_sec, 3),
        "fast_repeat_ratio": round(fast_repeat_ratio, 3),
        "median_interval": round(median_interval, 3),
        "cv_fast": round(cv_fast, 3),
        "max_fast_run": int(max_fast_run),

        "cry_good_peaks": int(cry_good),
        "cry_checked_peaks": int(cry_checked),
        "cry_burst_ratio": round(float(burst_ratio), 3),
        "per_peak_debug": per_peak_debug[:8],

        "hum_info": hum_info,

    }
    return label, info


# === test ===
#test_files = [r"C:\Users\Stacee\Desktop\cry5.wav"]
test_files = [r"C:\Users\Stacee\Desktop\hum\hum4.wav"]
#test_files = [r"C:\Users\Stacee\Desktop\laugh\laugh_1.m4a_2.wav"]


for f in test_files:
    label, info = classify_cry_laugh(f)
    print("\nFile:", f)
    print("Pred:", label)
    print("Info:", info)
