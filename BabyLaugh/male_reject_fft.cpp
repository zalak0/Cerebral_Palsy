#include "male_reject_fft.h"
#include <arduino.h>
#include <arduinoFFT.h>
#include <math.h>
#include <string.h>

/* Tunable parameters */
static const int   FFT_N = 1024;
static const float RMS_MIN           = 0.008f;
static const float LOW_MID_RATIO_THR = 0.55f;
static const float CENTROID_THR_HZ   = 900.0f;

static const int SCORE_ON_FRAMES  = 3;
static const int SCORE_OFF_FRAMES = 2;
static const int SCORE_MAX        = 6;

/* Internal state */
static int g_sr = 16000;
static int16_t g_frame16[FFT_N];
static int g_fill = 0;

static float vReal[FFT_N];
static float vImag[FFT_N];

static ArduinoFFT<float> FFT(vReal, vImag, FFT_N, 16000);

static MR_Debug g_dbg = {0};
static int g_score = 0;

static inline int hz_to_bin(float hz) {
  int k = (int)floorf(hz * (float)FFT_N / (float)g_sr + 0.5f);
  if (k < 0) k = 0;
  if (k > FFT_N / 2) k = FFT_N / 2;
  return k;
}

static float frame_rms_norm(const int16_t *x, int n) {
  if (n <= 0) return 0.0f;
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    double s = (double)x[i] / 32768.0;
    sum += s * s;
  }
  return (float)sqrt(sum / (double)n);
}

static void analyze_frame_1024(void) {
  float rms = frame_rms_norm(g_frame16, FFT_N);
  g_dbg.frame_rms = rms;

  if (rms < RMS_MIN) {
    if (g_score > 0) g_score--;
    g_dbg.score = g_score;
    g_dbg.male_like = (g_score >= SCORE_ON_FRAMES);
    return;
  }

  for (int i = 0; i < FFT_N; i++) {
    vReal[i] = (float)g_frame16[i];
    vImag[i] = 0.0f;
  }

  FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);
  FFT.compute(FFTDirection::Forward);
  FFT.complexToMagnitude();

  int k_low0 = hz_to_bin(80.0f);
  int k_low1 = hz_to_bin(200.0f);
  int k_mid0 = hz_to_bin(200.0f);
  int k_mid1 = hz_to_bin(2000.0f);
  int k_ce0  = hz_to_bin(80.0f);
  int k_ce1  = hz_to_bin(3500.0f);

  double E_low = 0.0, E_mid = 0.0;
  for (int k = k_low0; k <= k_low1; k++) { double m = (double)vReal[k]; E_low += m * m; }
  for (int k = k_mid0; k <= k_mid1; k++) { double m = (double)vReal[k]; E_mid += m * m; }

  float ratio = (float)(E_low / (E_mid + 1e-9));
  g_dbg.low_mid_ratio = ratio;

  double num = 0.0, den = 0.0;
  for (int k = k_ce0; k <= k_ce1; k++) {
    double mag = (double)vReal[k];
    double hz  = ((double)k * (double)g_sr) / (double)FFT_N;
    num += hz * mag;
    den += mag;
  }
  float centroid = (den > 1e-9) ? (float)(num / den) : 9999.0f;
  g_dbg.centroid_hz = centroid;

  bool hit = (ratio > LOW_MID_RATIO_THR) && (centroid < CENTROID_THR_HZ);

  if (hit) {
    if (g_score < SCORE_MAX) g_score++;
  } else {
    g_score -= SCORE_OFF_FRAMES;
    if (g_score < 0) g_score = 0;
  }

  g_dbg.score = g_score;
  g_dbg.male_like = (g_score >= SCORE_ON_FRAMES);
}

void mr_init(int sr) {
  if (sr <= 0) sr = 16000;
  g_sr = sr;
  FFT = ArduinoFFT<float>(vReal, vImag, FFT_N, (float)g_sr);

  g_fill = 0;
  memset(g_frame16, 0, sizeof(g_frame16));
  memset(&g_dbg, 0, sizeof(g_dbg));
  g_score = 0;
}

void mr_feed_mono_block(const int16_t *mono, int n) {
  if (!mono || n <= 0) return;

  for (int i = 0; i < n; i++) {
    g_frame16[g_fill++] = mono[i];
    if (g_fill >= FFT_N) {
      analyze_frame_1024();
      g_fill = 0;
    }
  }
}

bool mr_is_male_like(void) {
  return g_dbg.male_like;
}

void mr_get_debug(MR_Debug *out) {
  if (!out) return;
  *out = g_dbg;
}
