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
                         voiced_k=0.18,
                         flat_thr=0.28,
                         zcr_thr=0.14,
                         min_voiced_seg_sec=0.12,
                         min_seg_count=2,
                         tonal_ratio_thr=0.60):
    """
    针对你这种 hum：短暂、间隔回0、音高不稳定
    用"tonal(低flatness)+低ZCR"来识别，不要求稳定f0
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

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) + eps
    flat = librosa.feature.spectral_flatness(S=S)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)[0]

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

    runs = _runs_from_mask(voiced)
    min_frames = int(min_voiced_seg_sec / frame_sec)
    good_runs = [(s, e) for (s, e) in runs if (e - s) >= min_frames]
    seg_count = len(good_runs)

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
                    win_sec=1.5,
                    floor_tol=0.25,
                    rise_thr=0.18,
                    fall_thr=0.18,
                    up_ratio_thr=0.55,
                    down_ratio_thr=0.55):
    """
    判断一个峰是否符合 cry 的"爬升->下降回到底噪"形状（只用 RMS）
    """
    N = len(rms)
    w = int(win_sec / frame_sec)
    l0 = max(0, pk - w)
    r0 = min(N, pk + w + 1)

    if pk - l0 < 5 or r0 - pk < 5:
        return False, {}

    left_seg = rms[l0:pk]
    right_seg = rms[pk:r0]

    li = int(np.argmin(left_seg)) + l0
    ri = int(np.argmin(right_seg)) + pk

    left_min = float(rms[li])
    right_min = float(rms[ri])
    peak = float(rms[pk])

    rise = peak - left_min
    fall = peak - right_min

    if rise < rise_thr * dyn or fall < fall_thr * dyn:
        return False, {
            "reason": "rise/fall too small",
            "left_min": left_min, "right_min": right_min, "peak": peak,
            "rise": rise, "fall": fall, "li": li, "ri": ri
        }

    floor_limit = noise_floor + floor_tol * dyn
    if left_min > floor_limit or right_min > floor_limit:
        return False, {
            "reason": "not return to floor",
            "left_min": left_min, "right_min": right_min, "peak": peak,
            "floor_limit": floor_limit, "li": li, "ri": ri
        }

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


# ============ 新增：Talk 检测函数 ============
def _is_talk_pattern(y, sr, rms, peaks, noise_floor, dyn, frame_sec,
                     min_syllable_gap=0.08,      # 音节间隔下限（秒）
                     max_syllable_gap=0.55,      # 音节间隔上限（秒）
                     regularity_cv_thr=0.65,     # 间隔规律性（CV阈值）
                     min_peaks=3,                # 至少3个峰（音节）
                     max_peaks=15,               # 说话不会有太多峰（vs laugh）
                     envelope_smoothness_thr=0.12, # 包络平滑度阈值
                     speech_centroid_lo=600,     # 说话频谱重心范围
                     speech_centroid_hi=2800,
                     harmonicity_thr=0.35,       # 谐波性下限（比hum低，比laugh高）
                     floor_return_ratio_thr=0.40 # 至少40%的峰间回到底噪
                     ):
    """
    检测 Talk（说话）的特征：
    1. 有规律的音节间隔（0.1-0.5秒）
    2. 包络相对平滑（vs cry的不稳定）
    3. 频繁回到底噪（音节间）
    4. 中等谐波性（有共振峰但不如hum纯）
    5. 频谱重心在语音范围（800-2500Hz）
    """
    eps = 1e-9
    
    # 基础检查
    if len(peaks) < min_peaks or len(peaks) > max_peaks:
        return False, {"reason": f"peaks out of range: {len(peaks)}"}
    
    # === 特征1: 峰间隔分析（音节节奏）===
    peak_times = peaks * frame_sec
    intervals = np.diff(peak_times)
    
    # 过滤出合理的音节间隔
    valid_intervals = intervals[(intervals >= min_syllable_gap) & (intervals <= max_syllable_gap)]
    
    if len(valid_intervals) < 2:
        return False, {"reason": "too few valid intervals"}
    
    # 间隔的规律性（CV = std/mean，越小越规律）
    interval_mean = float(np.mean(valid_intervals))
    interval_std = float(np.std(valid_intervals))
    interval_cv = interval_std / (interval_mean + eps)
    
    # 说话应该有一定规律性，但不如节拍那么严格
    if interval_cv > regularity_cv_thr:
        return False, {
            "reason": "intervals too irregular",
            "interval_cv": round(interval_cv, 3),
            "interval_mean": round(interval_mean, 3)
        }
    
    # === 特征2: 包络平滑度（vs cry的起伏）===
    # 计算RMS的二阶差分（曲率）来衡量平滑度
    rms_smooth = librosa.util.normalize(rms)  # 归一化
    rms_diff2 = np.abs(np.diff(rms_smooth, n=2))  # 二阶差分
    envelope_smoothness = float(np.mean(rms_diff2))
    
    if envelope_smoothness > envelope_smoothness_thr:
        return False, {
            "reason": "envelope too rough (like cry)",
            "smoothness": round(envelope_smoothness, 4)
        }
    
    # === 特征3: 回到底噪的频率（音节间停顿）===
    floor_threshold = noise_floor + 0.20 * dyn
    
    # 检查每对相邻峰之间是否有回到底噪的点
    floor_returns = 0
    for i in range(len(peaks) - 1):
        start = int(peaks[i])
        end = int(peaks[i+1])
        segment = rms[start:end]
        if np.any(segment <= floor_threshold):
            floor_returns += 1
    
    floor_return_ratio = floor_returns / max(len(peaks) - 1, 1)
    
    if floor_return_ratio < floor_return_ratio_thr:
        return False, {
            "reason": "not enough floor returns",
            "floor_return_ratio": round(floor_return_ratio, 3)
        }
    
    # === 特征4: 频谱分析（谐波性 + 频谱重心）===
    hop = 256  # 与主函数一致
    n_fft = 1024
    
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) + eps
    
    # 谐波性：用 spectral flatness 的反向指标
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    harmonicity = 1.0 - np.median(flatness)
    
    if harmonicity < harmonicity_thr:
        return False, {
            "reason": "not harmonic enough",
            "harmonicity": round(float(harmonicity), 3)
        }
    
    # 频谱重心
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    centroid_median = float(np.median(centroid))
    
    if not (speech_centroid_lo <= centroid_median <= speech_centroid_hi):
        return False, {
            "reason": "centroid out of speech range",
            "centroid_median": round(centroid_median, 1)
        }
    
    # === 所有条件满足 ===
    info = {
        "interval_mean": round(interval_mean, 3),
        "interval_cv": round(interval_cv, 3),
        "envelope_smoothness": round(envelope_smoothness, 4),
        "floor_return_ratio": round(floor_return_ratio, 3),
        "harmonicity": round(float(harmonicity), 3),
        "centroid_median": round(centroid_median, 1),
        "num_peaks": len(peaks),
        "valid_intervals": len(valid_intervals),
        "reason": "ok"
    }
    
    return True, info


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

    # ===== Stage 0: hum =====
    is_hum, hum_info = _is_hum_tonal_bursts(y, sr, hop=hop, n_fft=n_fft)

    # ===== Stage 1: laugh =====
    peaks_per_sec = float(len(peaks) / max(duration_sec, 1e-6))

    FAST_GAP_SEC = 0.35
    MIN_CLUSTER_PEAKS = 3
    MIN_CLUSTER_COUNT = 2
    LONG_RUN_PEAKS = 4
    MIN_TOTAL_PEAKS = 6

    if len(peaks) >= 3:
        intervals = np.diff(peaks) * frame_sec
        short_mask = intervals < FAST_GAP_SEC
        short_intervals = intervals[short_mask]

        fast_repeat_ratio = float(np.mean(short_mask))
        median_interval = float(np.median(intervals))

        if short_intervals.size >= 2:
            cv_fast = float(np.std(short_intervals) / (np.mean(short_intervals) + eps))
        else:
            cv_fast = 10.0

        max_fast_run = 1
        cur = 1
        for is_short in short_mask:
            if is_short:
                cur += 1
                max_fast_run = max(max_fast_run, cur)
            else:
                cur = 1

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

    old_laugh_rule = (
        (peaks_per_sec >= 2.0) and
        (fast_repeat_ratio >= 0.55) and
        ((cv_fast <= 0.35) or (max_fast_run >= LONG_RUN_PEAKS) or (median_interval <= 0.30))
    )

    cluster_laugh_rule = (
        (len(peaks) >= MIN_TOTAL_PEAKS) and
        (cluster_count >= MIN_CLUSTER_COUNT) and
        (max_fast_run >= MIN_CLUSTER_PEAKS) and
        (fast_repeat_ratio >= 0.40)
    )

    is_laugh = old_laugh_rule or cluster_laugh_rule

    # ===== Stage 2: talk (NEW) =====
    # Talk 优先级在 laugh 之后，cry 之前
    is_talk = False
    talk_info = {}
    
    if not is_hum and not is_laugh:
        is_talk, talk_info = _is_talk_pattern(
            y, sr, rms, peaks, noise_floor, dyn, frame_sec
        )

    # ===== Stage 3: cry =====
    cry_good = 0
    cry_checked = 0
    per_peak_debug = []

    if (not is_laugh) and (not is_talk) and (len(peaks) >= 1):
        for pk in peaks:
            ok, dbg = _burst_shape_ok(rms, int(pk), noise_floor, frame_sec, dyn)
            cry_checked += 1
            cry_good += int(ok)
            per_peak_debug.append({"pk": int(pk), "ok": ok, **dbg})

        burst_ratio = cry_good / max(cry_checked, 1)

        if len(peaks) <= 3:
            is_cry = (cry_good >= 1) and (burst_ratio >= 0.34)
        else:
            is_cry = (cry_good >= 2) and (burst_ratio >= 0.60)
    else:
        burst_ratio = 0.0
        is_cry = False

    # ===== 最终判定（优先级：hum > laugh > talk > cry > unknown）=====
    if is_hum:
        label = "hum"
    elif is_laugh:
        label = "laugh"
    elif is_talk:
        label = "talk"
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
        "talk_info": talk_info,  # 新增
    }
    return label, info


# === test ===
test_files = [
    r"C:\Users\Stacee\Desktop\talk5.wav",       # 测试 talk
    #r"C:\Users\Stacee\Desktop\cry6.wav",        # 测试 cry
    #r"C:\Users\Stacee\Desktop\laugh\laugh11.wav", # 测试 laugh
]

for f in test_files:
    label, info = classify_cry_laugh(f)
    print("\n" + "="*70)
    print("File:", f)
    print("Prediction:", label)
    print("\nDetailed Info:")
    for k, v in info.items():
        print(f"  {k}: {v}")


