#ifndef CL_DEBUG_DUMP
#define CL_DEBUG_DUMP 1
#endif

#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===================== Config =====================
#ifndef CL_SR_DEFAULT
#define CL_SR_DEFAULT   16000
#endif

#ifndef CL_HOP
#define CL_HOP          256
#endif

#ifndef CL_NFFT
#define CL_NFFT         1024
#endif

// 最近窗口长度（帧数）。5.12s @ 16k/256 = 320帧
#ifndef CL_MAX_FRAMES
#define CL_MAX_FRAMES   320
#endif

#ifndef CL_MAX_PEAKS
#define CL_MAX_PEAKS    64
#endif

// ===================== Labels =====================
typedef enum {
    CL_UNKNOWN = 0,
    CL_HUM     = 1,
    CL_LAUGH   = 2,
    CL_CRY     = 3
} CL_Label;

typedef struct {
    // common
    float duration_sec;
    float noise_floor;
    float peak_max;
    float dyn;

    // peaks / laugh stats
    int   num_peaks;
    float peaks_per_sec;
    float fast_repeat_ratio;
    float median_interval;
    float cv_fast;
    int   max_fast_run;
    int   cluster_count;

    // hum stats
    float hum_voiced_ratio;
    float hum_tonal_ratio;
    float hum_zcr_med;
    int   hum_seg_count;

    // cry stats
    int   cry_good_peaks;
    int   cry_checked_peaks;
    float cry_burst_ratio;

    // talk blocker (方案2：只挡，不输出)
    int   talk_blocked;              // 1 表示判到 talk -> 强制 UNKNOWN
    float talk_interval_mean;
    float talk_interval_cv;
    float talk_smoothness;
    float talk_floor_return_ratio;
    float talk_harmonicity;
    float talk_centroid_median;

} CL_Debug;

// ===================== State =====================
typedef struct {
    int   sr;
    int   hop;
    int   n_fft;
    float frame_sec;

    // sliding frame buffer (mono float)
    float frame_buf[CL_NFFT];
    int   fb_fill;
    int   hop_acc;

    // hann window
    float hann[CL_NFFT];

    // FFT scratch
    float fft_re[CL_NFFT];
    float fft_im[CL_NFFT];

    // ring history
    float rms_hist[CL_MAX_FRAMES];
    float zcr_hist[CL_MAX_FRAMES];
    float flat_hist[CL_MAX_FRAMES];
    float cent_hist[CL_MAX_FRAMES];
    int   hist_len;
    int   hist_head;

    // unfolded temp (old->new)
    float rms_tmp[CL_MAX_FRAMES];
    float zcr_tmp[CL_MAX_FRAMES];
    float flat_tmp[CL_MAX_FRAMES];
    float cent_tmp[CL_MAX_FRAMES];
    bool  voiced_tmp[CL_MAX_FRAMES];

    // peaks
    int peaks[CL_MAX_PEAKS];
} CL_State;

// ===================== API =====================
void      cl_init(CL_State *st, int sample_rate);
void      cl_reset(CL_State *st);

// feed mono int16 blocks
// return true if produced >=1 new feature frame
bool      cl_feed_mono_block(CL_State *st, const int16_t *mono, int n);

// classify latest window
CL_Label  cl_classify_latest(CL_State *st, CL_Debug *dbg);

// label string
const char* cl_label_str(CL_Label lab);

#ifdef __cplusplus
}
#endif
