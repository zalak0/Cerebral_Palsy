#ifndef MALE_REJECT_FFT_H
#define MALE_REJECT_FFT_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

  typedef struct
  {
    float frame_rms;
    float low_mid_ratio;
    float centroid_hz;
    int score;
    bool male_like;
  } MR_Debug;

  void mr_init(int sr);
  void mr_feed_mono_block(const int16_t *mono, int n);
  bool mr_is_male_like(void);
  void mr_get_debug(MR_Debug *out);

#ifdef __cplusplus
}
#endif

#endif
