#ifndef CRY_LAUGH_H
#define CRY_LAUGH_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /*
      Output label of the classifier.
      - CL_UNKNOWN: not enough evidence / uncertain
      - CL_CRY: cry-like envelope pattern
      - CL_LAUGH: laugh-like peak repetition pattern
    */
    typedef enum
    {
        CL_UNKNOWN = 0,
        CL_CRY = 1,
        CL_LAUGH = 2
    } CL_Label;

    /*
      Optional debug metrics (useful for tuning).
      You can pass NULL to cl_classify_latest() if you don't need this.
    */
    typedef struct
    {
        float duration_sec; // duration of the RMS window used for classification
        float noise_floor;  // 10th percentile of RMS (floor estimate)
        float peak_max;     // max RMS in the window
        float dyn;          // peak_max - noise_floor
        int num_peaks;      // number of detected peaks in RMS

        // laugh-related metrics
        float peaks_per_sec;
        float fast_repeat_ratio;
        float median_interval;
        float cv_fast;
        int max_fast_run;
        int cluster_count;

        // cry-related metrics
        int cry_good_peaks;
        int cry_checked_peaks;
        float cry_burst_ratio;
    } CL_Debug;

    /*
      Internal state of the classifier.
      This stores:
      - sliding energy window for RMS (FRAME_LEN)
      - ring buffer of RMS frames (last few seconds)
      - small smoothing state
      IMPORTANT: Create ONE instance and keep feeding samples into it.
    */
    typedef struct
    {
        // Configuration
        int sr; // sample rate in Hz (recommended: 16000)

        // RMS sliding window state (energy sum over last FRAME_LEN samples)
        int64_t sum_energy;
        int samp_idx;
        int hop_count;
        int64_t energy_ring[1024]; // FRAME_LEN fixed at 1024

        // RMS ring buffer (stores last MAX_RMS_FRAMES RMS values)
        float rms_ring[240];
        int rms_widx;
        int rms_count;
        bool new_rms;

        // Optional smoothing (EMA)
        float ema;
        bool ema_init;
    } CL_State;

    /*
      Initialize the classifier state.
      sample_rate must match your audio pipeline sample rate.
      For best matching with the original Python thresholds, use 16000 Hz.
    */
    void cl_init(CL_State *st, int sample_rate);

    /*
      Feed denoised mono PCM samples into the classifier.
      - mono: pointer to int16 PCM samples
      - n: number of samples
      Recommended: feed in blocks of 256 samples (matches HOP=256),
      but any block size works (internally it is sample-by-sample).
    */
    void cl_feed_mono_block(CL_State *st, const int16_t *mono, int n);

    /*
      Feed stereo PCM samples (if your teammate outputs L/R separately).
      Internally this uses energy fusion: E = (L^2 + R^2)/2 to avoid phase cancel.
    */
    void cl_feed_stereo_block(CL_State *st, const int16_t *left, const int16_t *right, int n);

    /*
      Whether a new RMS frame has been produced since last clear.
      RMS is produced roughly every HOP samples (HOP=256).
    */
    bool cl_has_new_rms(const CL_State *st);

    /* Clear the "new RMS produced" flag. */
    void cl_clear_new_rms(CL_State *st);

    /*
      Classify using the most recent RMS window (e.g., last ~3 seconds).
      - st: classifier state (read-only)
      - debug: optional pointer to receive debug metrics; pass NULL if not needed
      Returns: CL_LAUGH / CL_CRY / CL_UNKNOWN
    */
    CL_Label cl_classify_latest(const CL_State *st, CL_Debug *debug);

#ifdef __cplusplus
}
#endif

#endif
