#include <Arduino.h>
#include <cmath>
#include <cstring>

#include "driver/i2s_std.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"

// ESP-SR (AFE / DOA)
#include "esp_afe_sr_iface.h"
#include "esp_afe_sr_models.h"
#include "model_path.h"
#include "esp_afe_doa.h"

// Your modules
#include "laugh.h"
#include "male_reject_fft.h"

static const char* TAG = "BabyLaugh";

/* =========================
   Pin definitions (edit if needed)
   ========================= */
static const int PIN_BCLK = 12;
static const int PIN_WS   = 13;
static const int PIN_DIN  = 11;

static const int PIN_POT  = 7;

// Outputs
static const int PIN_DETECT = 8;   // loud enough
static const int PIN_SCREAM = 9;   // very loud
static const int PIN_LAUGH  = 10;  // laugh detected (repurpose your old PIN_CRY)
static const int PIN_REAL   = 6;   // valid baby-zone + not rejected

/* =========================
   Audio parameters
   ========================= */
static const int SAMPLE_RATE   = 16000;
static const int FRAME_SAMPLES = 1024;   // I2S read frame (stereo => x2 samples)

/* =========================
   I2S handle
   ========================= */
static i2s_chan_handle_t rx_chan;

/* =========================
   AFE variables
   ========================= */
static const esp_afe_sr_iface_t *afe_handle = NULL;
static esp_afe_sr_data_t *afe_data = NULL;
static int16_t *afe_feed_buf = NULL;
static bool use_afe = false;
static int afe_feed_chunksize = 0;
static int afe_feed_channels = 0;

/* =========================
   DOA variables
   ========================= */
static afe_doa_handle_t *doa_handle = NULL;
static bool use_doa = false;

// Baby monitoring zones (degrees: 0-180)
static const float BABY_ZONE_MIN   = 120.0f;
static const float BABY_ZONE_MAX   = 180.0f;
static const float IGNORE_ZONE_MIN = 0.0f;
static const float IGNORE_ZONE_MAX = 90.0f;

/* =========================
   Laugh classifier + FFT gate state
   ========================= */
static CL_State g_cl;

/* =========================
   Utilities
   ========================= */
static void print_memory_info() {
  ESP_LOGI(TAG, "=== Memory Info ===");
  ESP_LOGI(TAG, "Free heap: %lu bytes", (unsigned long)esp_get_free_heap_size());
  ESP_LOGI(TAG, "Free PSRAM: %lu bytes", (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  ESP_LOGI(TAG, "Largest free block PSRAM: %lu bytes", (unsigned long)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
  ESP_LOGI(TAG, "Min free heap: %lu bytes", (unsigned long)esp_get_minimum_free_heap_size());
  ESP_LOGI(TAG, "==================");
}

// Read potentiometer with averaging
static int readPotAvg() {
  long sum = 0;
  for (int i = 0; i < 8; i++) {
    sum += analogRead(PIN_POT);
    delayMicroseconds(200);
  }
  return (int)(sum / 8);
}

// Simple map to double range
static double mapDouble(int x, int in_min, int in_max, double out_min, double out_max) {
  if (x < in_min) x = in_min;
  if (x > in_max) x = in_max;
  double t = (double)(x - in_min) / (double)(in_max - in_min);
  return out_min + t * (out_max - out_min);
}

/* =========================
   I2S setup (stereo)
   ========================= */
static void setupI2S() {
  ESP_LOGI(TAG, "Setting up I2S for DUAL microphones (stereo)...");

  i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_chan));

  i2s_std_config_t std_cfg = {};
  std_cfg.clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE);
  std_cfg.slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO);

  std_cfg.gpio_cfg.mclk = I2S_GPIO_UNUSED;
  std_cfg.gpio_cfg.bclk = (gpio_num_t)PIN_BCLK;
  std_cfg.gpio_cfg.ws   = (gpio_num_t)PIN_WS;
  std_cfg.gpio_cfg.dout = I2S_GPIO_UNUSED;
  std_cfg.gpio_cfg.din  = (gpio_num_t)PIN_DIN;
  std_cfg.gpio_cfg.invert_flags.mclk_inv = false;
  std_cfg.gpio_cfg.invert_flags.bclk_inv = false;
  std_cfg.gpio_cfg.invert_flags.ws_inv   = false;

  ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_chan, &std_cfg));
  ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));

  ESP_LOGI(TAG, "I2S stereo ready.");
}

/* =========================
   AFE setup (SE/NS/VAD)
   ========================= */
static void setupAFE() {
  ESP_LOGI(TAG, "========================================");
  ESP_LOGI(TAG, "Initializing AFE...");
  print_memory_info();

  // Load SR models list (ESP-SR)
  srmodel_list_t *models = esp_srmodel_init("model");

  // "MM" means 2 microphones (stereo interleaved)
  afe_config_t *afe_cfg = afe_config_init("MM", models, AFE_TYPE_SR, AFE_MODE_LOW_COST);

  // Configure: enable SE/NS/VAD only
  afe_cfg->aec_init = false;
  afe_cfg->se_init  = true;
  afe_cfg->ns_init  = true;
  afe_cfg->vad_init = true;

  afe_cfg->wakenet_init = false;
  afe_cfg->vad_mode = VAD_MODE_3;

  afe_cfg->afe_ringbuf_size = 100;
  afe_cfg->memory_alloc_mode = AFE_MEMORY_ALLOC_MORE_PSRAM;

  ESP_LOGI(TAG, "PCM Config - total=%d, mic=%d, ref=%d",
           afe_cfg->pcm_config.total_ch_num,
           afe_cfg->pcm_config.mic_num,
           afe_cfg->pcm_config.ref_num);

  afe_handle = esp_afe_handle_from_config(afe_cfg);
  afe_data   = afe_handle->create_from_config(afe_cfg);

  afe_feed_chunksize = afe_handle->get_feed_chunksize(afe_data);
  afe_feed_channels  = afe_handle->get_feed_channel_num(afe_data);

  ESP_LOGI(TAG, "AFE feed chunk=%d samples, channels=%d", afe_feed_chunksize, afe_feed_channels);

  size_t buf_size = (size_t)afe_feed_chunksize * (size_t)afe_feed_channels * sizeof(int16_t);
  afe_feed_buf = (int16_t*)heap_caps_calloc(1, buf_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

  if (!afe_feed_buf) {
    ESP_LOGE(TAG, "Failed to allocate AFE feed buffer (%u bytes). Disable AFE.", (unsigned)buf_size);
    afe_handle->destroy(afe_data);
    afe_data = NULL;
    esp_srmodel_deinit(models);
    use_afe = false;
    return;
  }

  use_afe = true;
  ESP_LOGI(TAG, "AFE initialized OK (PSRAM buffer=%u bytes).", (unsigned)buf_size);
  print_memory_info();

  // You can deinit model list after AFE is created
  esp_srmodel_deinit(models);
}

/* =========================
   DOA setup
   ========================= */
static void setupDOA() {
  ESP_LOGI(TAG, "========================================");
  ESP_LOGI(TAG, "Initializing DOA (Direction of Arrival)...");

  float mic_distance_m = 0.046f;  // TODO: set your real mic spacing in meters

  doa_handle = afe_doa_create(
    "MM",
    SAMPLE_RATE,
    20.0f,            // degrees resolution
    mic_distance_m,
    FRAME_SAMPLES
  );

  if (!doa_handle) {
    ESP_LOGW(TAG, "DOA create failed. Continue without DOA.");
    use_doa = false;
    return;
  }

  use_doa = true;
  ESP_LOGI(TAG, "DOA OK. mic_spacing=%.3fm, resolution=20deg", mic_distance_m);
  ESP_LOGI(TAG, "Baby zone: %.0f-%.0f deg, Ignore zone: %.0f-%.0f deg",
           BABY_ZONE_MIN, BABY_ZONE_MAX, IGNORE_ZONE_MIN, IGNORE_ZONE_MAX);
  ESP_LOGI(TAG, "========================================");
}

void setup() {
  Serial.begin(115200);
  delay(300);

  ESP_LOGI(TAG, "Boot.");
  setupI2S();
  setupAFE();
  setupDOA();

  pinMode(PIN_DETECT, OUTPUT);
  pinMode(PIN_SCREAM, OUTPUT);
  pinMode(PIN_LAUGH,  OUTPUT);
  pinMode(PIN_REAL,   OUTPUT);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  // Init laugh classifier + FFT gate
  cl_init(&g_cl, SAMPLE_RATE);
  mr_init(SAMPLE_RATE);

  ESP_LOGI(TAG, "========================================");
  ESP_LOGI(TAG, "AFE: %s, DOA: %s", use_afe ? "ON" : "OFF", use_doa ? "ON" : "OFF");
  ESP_LOGI(TAG, "Ready.");
  ESP_LOGI(TAG, "========================================");
}

void loop() {
  static int32_t i2s_buf[FRAME_SAMPLES * 2];      // stereo int32 container
  static int16_t audio_stereo[FRAME_SAMPLES * 2]; // stereo int16 (L,R interleaved)
  static int16_t mono16[FRAME_SAMPLES];           // mono int16 for gate+classifier

  static float last_angle = 90.0f;
  static uint32_t last_cls_ms = 0;
  static uint32_t laugh_until_ms = 0;

  size_t bytes_read = 0;

  // 1) Read I2S stereo
  esp_err_t err = i2s_channel_read(rx_chan, i2s_buf, sizeof(i2s_buf), &bytes_read, pdMS_TO_TICKS(200));
  if (err != ESP_OK || bytes_read == 0) {
    ESP_LOGW(TAG, "I2S read timeout/error");
    return;
  }

  int frames = (int)(bytes_read / (sizeof(int32_t) * 2)); // number of stereo frames

  // 2) Convert 32-bit stereo to 16-bit stereo
  // NOTE: shift value depends on your mic data format. ">>11" is an empirical choice.
  for (int i = 0; i < frames; i++) {
    int32_t sampleL = i2s_buf[i * 2 + 0] >> 11;
    int32_t sampleR = i2s_buf[i * 2 + 1] >> 11;
    audio_stereo[i * 2 + 0] = (int16_t)sampleL;
    audio_stereo[i * 2 + 1] = (int16_t)sampleR;
  }

  // 3) AFE processing (optional)
  int16_t *processed_audio = audio_stereo; // default: raw
  int processed_samples = frames * 2;      // default: stereo samples count
  bool vad_detected = false;

  if (use_afe && afe_handle && afe_data && afe_feed_buf) {
    // Feed: AFE expects interleaved stereo int16 in chunksize
    int feed_frames = min(afe_feed_chunksize, frames);
    for (int i = 0; i < feed_frames; i++) {
      afe_feed_buf[i * 2 + 0] = audio_stereo[i * 2 + 0];
      afe_feed_buf[i * 2 + 1] = audio_stereo[i * 2 + 1];
    }

    afe_handle->feed(afe_data, afe_feed_buf);
    vTaskDelay(pdMS_TO_TICKS(1));

    // Fetch twice to get most recent output
    afe_fetch_result_t *r1 = afe_handle->fetch(afe_data);
    afe_fetch_result_t *r2 = afe_handle->fetch(afe_data);

    afe_fetch_result_t *r = (r2 && r2->ret_value == ESP_OK && r2->data) ? r2 : r1;

    if (r && r->ret_value == ESP_OK && r->data) {
      processed_audio   = r->data; // usually mono output
      processed_samples = (int)(r->data_size / sizeof(int16_t));
      vad_detected      = (r->vad_state == VAD_SPEECH);
    }
  }

  // 4) DOA processing (optional) - uses RAW stereo
  float angle = last_angle;
  bool from_baby_zone = true;
  bool from_ignore_zone = false;

  if (use_doa && doa_handle) {
    angle = afe_doa_process(doa_handle, audio_stereo);

    from_baby_zone   = (angle >= BABY_ZONE_MIN && angle <= BABY_ZONE_MAX);
    from_ignore_zone = (angle >= IGNORE_ZONE_MIN && angle <= IGNORE_ZONE_MAX);

    last_angle = angle;

    if (from_ignore_zone) {
      // Early exit: ignore this direction
      digitalWrite(PIN_DETECT, LOW);
      digitalWrite(PIN_SCREAM, LOW);
      digitalWrite(PIN_LAUGH,  LOW);
      digitalWrite(PIN_REAL,   LOW);
      delay(20);
      return;
    }
  }

  // 5) Build MONO buffer for FFT gate + laugh classifier
  int mono_len = 0;

  if (use_afe && processed_audio != audio_stereo) {
    // AFE output assumed mono
    mono_len = min(processed_samples, FRAME_SAMPLES);
    memcpy(mono16, processed_audio, (size_t)mono_len * sizeof(int16_t));
  } else {
    // No AFE output -> average L/R to mono
    mono_len = min(frames, FRAME_SAMPLES);
    for (int i = 0; i < mono_len; i++) {
      int32_t L = audio_stereo[i * 2 + 0];
      int32_t R = audio_stereo[i * 2 + 1];
      mono16[i] = (int16_t)((L + R) / 2);
    }
    // If AFE is OFF, treat VAD as "true when not silent" by using loudness later
    vad_detected = true;
  }

  // 6) Feed FFT gate + laugh classifier
  mr_feed_mono_block(mono16, mono_len);
  cl_feed_mono_block(&g_cl, mono16, mono_len);

  // 7) Loudness (RMS dB) from mono16
  double sum = 0.0;
  for (int i = 0; i < mono_len; i++) {
    double s = (double)mono16[i];
    sum += s * s;
  }
  int sample_count = (mono_len > 0) ? mono_len : 1;
  double rms = sqrt(sum / (double)sample_count);
  double rms_db = 20.0 * log10(rms + 1e-9);

  // Threshold from potentiometer
  int pot = readPotAvg();
  double voice_db  = mapDouble(pot, 0, 4095, 30.0, 90.0);
  double scream_db = voice_db + 20.0;

  bool loud_enough = (rms_db > voice_db);
  bool screaming   = (rms_db > scream_db);

  // 8) FFT gate decision
  bool reject = mr_is_male_like(); // true => likely adult-male-like speech, reject trigger

  // 9) Run laugh classification periodically (not every loop)
  uint32_t now = millis();
  if (now - last_cls_ms >= 250) {
    last_cls_ms = now;

    // Gate conditions (edit if you want)
    bool can_classify = loud_enough && vad_detected && from_baby_zone && !reject;

    CL_Label lab = CL_UNKNOWN;
    CL_Debug dbg = {};

    if (can_classify) {
      lab = cl_classify_latest(&g_cl, &dbg);
      const char* lab_str = (lab == CL_LAUGH) ? "laugh" : "unknown";
      ESP_LOGI(TAG, "Classifier: %s (peaks=%d, pps=%.2f, fast=%.2f, run=%d, clusters=%d)",lab_str,dbg.num_peaks, dbg.peaks_per_sec, dbg.fast_repeat_ratio, dbg.max_fast_run, dbg.cluster_count);

      if (lab == CL_LAUGH) {
        laugh_until_ms = now + 800; // latch laugh output for 0.8s
      }

    else {ESP_LOGI(TAG, "Classifier skipped: loud=%d vad=%d baby=%d reject=%d",
           loud_enough ? 1 : 0,
           vad_detected ? 1 : 0,
           from_baby_zone ? 1 : 0,
           reject ? 1 : 0);}

    }

    // Optional logs (every 1s approx)
    static int log_ctr = 0;
    log_ctr++;
    if (log_ctr % 4 == 0) {
      MR_Debug md;
      mr_get_debug(&md);

      ESP_LOGI(TAG,
        "RMS: %.1fdB thr:%.1f angle:%.1f baby:%d VAD:%d reject:%d LAUGH:%d | FFT ratio=%.2f cent=%.0f score=%d",
        rms_db, voice_db, last_angle,
        from_baby_zone ? 1 : 0,
        vad_detected ? 1 : 0,
        reject ? 1 : 0,
        (now < laugh_until_ms) ? 1 : 0,
        md.low_mid_ratio, md.centroid_hz, md.score
      );
    }
  }

  // 10) Outputs
  // Basic status LEDs/pins
  digitalWrite(PIN_SCREAM, screaming ? HIGH : LOW);
  digitalWrite(PIN_DETECT, (!screaming && loud_enough) ? HIGH : LOW);

  // "REAL" means: from baby direction + loud enough + not rejected
  bool real_ok = (!screaming && loud_enough && from_baby_zone && !reject);
  digitalWrite(PIN_REAL, real_ok ? HIGH : LOW);

  // Laugh output (latched)
  digitalWrite(PIN_LAUGH, (now < laugh_until_ms) ? HIGH : LOW);

  delay(20);
}

