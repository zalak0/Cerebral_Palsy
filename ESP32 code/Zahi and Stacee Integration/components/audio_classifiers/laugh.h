#ifndef LAUGH_H
#define LAUGH_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

  typedef enum
  {
    CL_UNKNOWN = 0,
    CL_LAUGH = 2
  } CL_Label;

  typedef struct
  {
    float duration_sec;
    float noise_floor;
    float peak_max;
    float dyn;
    int num_peaks;

    // laugh metrics
    float peaks_per_sec;
    float fast_repeat_ratio;
    float median_interval;
    float cv_fast;
    int max_fast_run;
    int cluster_count;
  } CL_Debug;

  typedef struct
  {
    int sr;

    // RMS sliding window
    int64_t sum_energy;
    int samp_idx;
    int hop_count;
    int64_t energy_ring[1024]; // FRAME_LEN fixed = 1024

    // RMS ring buffer (~last few seconds)
    float rms_ring[240]; // MAX_RMS_FRAMES fixed = 240
    int rms_widx;
    int rms_count;
    bool new_rms;

    // smoothing
    float ema;
    bool ema_init;
  } CL_State;

  void cl_init(CL_State *st, int sample_rate);
  void cl_feed_mono_block(CL_State *st, const int16_t *mono, int n);
  bool cl_has_new_rms(const CL_State *st);
  void cl_clear_new_rms(CL_State *st);
  CL_Label cl_classify_latest(const CL_State *st, CL_Debug *debug);

#ifdef __cplusplus
}
#endif

#endif
