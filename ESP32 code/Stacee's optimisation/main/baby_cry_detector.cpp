#include <Arduino.h>
#include <cmath>
#include "driver/i2s_std.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_afe_sr_iface.h"
#include "esp_afe_sr_models.h"
#include "model_path.h"
#include "esp_heap_caps.h"
#include "esp_afe_doa.h"
#include "led_strip.h"  // ESP-IDF LED strip component

// Stacee's modules
#include "laugh.h"
#include "male_reject_fft.h"

static const char* TAG = "BabyCry";

// Pin definitions for both microphones
static const int PIN_BCLK = 12;
static const int PIN_WS   = 13;
static const int PIN_DIN  = 11;

// Control pin
static const int PIN_POT = 7;

// Output and status pins
static const int PIN_CRY = 2;
static const int PIN_DETECT = 8;
static const int PIN_SCREAM = 9;
static const int PIN_REAL = 6;  //PIN_REAL

// LED Strip configuration
#define LED_PIN     10
#define NUM_LEDS    8
#define LED_TYPE    WS2812B
#define COLOR_ORDER GRB
#define BRIGHTNESS 128      // 0-255 (50% brightness)

// Replace CRGB array with NeoPixel
static led_strip_handle_t led_strip = NULL;

// Audio parameters
static const int SAMPLE_RATE = 16000;
static const int FRAME_SAMPLES = 1024;

// i2s channel handle
static i2s_chan_handle_t rx_chan;

// AFE variables
static const esp_afe_sr_iface_t *afe_handle = NULL;
static esp_afe_sr_data_t *afe_data = NULL;
static int16_t *afe_feed_buf = NULL;  // Make this global so we can free it
static bool use_afe = false;
static int afe_feed_chunksize = 0;
static int afe_feed_channels = 0;

// DOA variables
static afe_doa_handle_t *doa_handle = NULL;
static bool use_doa = false;

// Baby monitoring zones (degrees: 0-180)
// Adjust these based on the crib position
static const float BABY_ZONE_MIN = 120.0;   // Baby zone: 120° to 180° (front center)
static const float BABY_ZONE_MAX = 180.0;
static const float IGNORE_ZONE_MIN = 0.0;   // Ignore: 0° to 90° (Back side)
static const float IGNORE_ZONE_MAX = 90.0;

// LED visualization settings
static const float LED_MIN_DB = 30.0;
static const float LED_MAX_DB = 90.0;

// Laugh classifier + FFT gate state
static CL_State g_cl;

/* =========================
   FINAL decision state machine
   ========================= */
typedef enum {
  DETECT_ARMED = 0,    // allow laugh detection
  DETECT_LOCKED        // laugh already triggered; wait quiet then re-arm
} DetectState;

static DetectState g_state = DETECT_ARMED;
static uint32_t g_quiet_since_ms = 0;

// Tuning: how long it must be quiet before re-arming
static const uint32_t QUIET_REARM_MS = 1500;

// How often we run the classifier while ARMED (fast enough for 3s laugh)
static const uint32_t CLS_PERIOD_MS = 250;

// Laugh trigger requirement (your rule)
static const int MIN_TOTAL_PEAKS_FOR_FINAL = 6;
static const int MIN_CLUSTER_FOR_FINAL = 2;

void print_memory_info() {
  ESP_LOGI(TAG, "=== Memory Info ===");
  ESP_LOGI(TAG, "Free heap: %lu bytes", esp_get_free_heap_size());
  ESP_LOGI(TAG, "Free PSRAM: %lu bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  ESP_LOGI(TAG, "Largest free block PSRAM: %lu bytes", heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
  ESP_LOGI(TAG, "Min free heap: %lu bytes", esp_get_minimum_free_heap_size());
  ESP_LOGI(TAG, "==================");
}

void setupI2S() {
  ESP_LOGI(TAG, "Setting up I2S for DUAL microphones (stereo)...");

  i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_chan));

  i2s_std_config_t std_cfg = {};
  std_cfg.clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE);
  std_cfg.slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO);  // STEREO!

  std_cfg.gpio_cfg.mclk = I2S_GPIO_UNUSED;
  std_cfg.gpio_cfg.bclk = (gpio_num_t)PIN_BCLK;
  std_cfg.gpio_cfg.ws   = (gpio_num_t)PIN_WS;
  std_cfg.gpio_cfg.dout = I2S_GPIO_UNUSED;
  std_cfg.gpio_cfg.din  = (gpio_num_t)PIN_DIN;  // Reads both channels
  std_cfg.gpio_cfg.invert_flags.mclk_inv = false;
  std_cfg.gpio_cfg.invert_flags.bclk_inv = false;
  std_cfg.gpio_cfg.invert_flags.ws_inv   = false;

  ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_chan, &std_cfg));
  ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));

  ESP_LOGI(TAG, "Dual I2S microphones setup complete (stereo mode)!");
}

void setupLEDs() {
  ESP_LOGI(TAG, "========================================");
  ESP_LOGI(TAG, "Initializing LED Strip with RMT...");

  // LED strip configuration
  led_strip_config_t strip_config = {
    .strip_gpio_num = LED_PIN,
    .max_leds = NUM_LEDS,
    .led_pixel_format = LED_PIXEL_FORMAT_GRB,  // WS2812B uses GRB
    .led_model = LED_MODEL_WS2812,
    .flags = {
      .invert_out = false,
    }
  };

  // RMT backend configuration
  led_strip_rmt_config_t rmt_config = {
    .clk_src = RMT_CLK_SRC_DEFAULT,
    .resolution_hz = 10 * 1000 * 1000, // 10 MHz
    .flags = {
      .with_dma = false,  // DMA disabled to avoid conflicts with I2S
    }
  };

  ESP_ERROR_CHECK(led_strip_new_rmt_device(&strip_config, &rmt_config, &led_strip));

  ESP_LOGI(TAG, "LED strip created with RMT backend");

  // Test pattern - rainbow
  for (int i = 0; i < NUM_LEDS; i++) {
    uint8_t hue = (i * 255) / NUM_LEDS;
    uint8_t r, g, b;

    if (hue < 85) {
      r = 255 - hue * 3;
      g = hue * 3;
      b = 0;
    } else if (hue < 170) {
      hue -= 85;
      r = 0;
      g = 255 - hue * 3;
      b = hue * 3;
    } else {
      hue -= 170;
      r = hue * 3;
      g = 0;
      b = 255 - hue * 3;
    }

    r = (r * BRIGHTNESS) / 255;
    g = (g * BRIGHTNESS) / 255;
    b = (b * BRIGHTNESS) / 255;

    led_strip_set_pixel(led_strip, i, r, g, b);
  }
  led_strip_refresh(led_strip);
  delay(500);

  led_strip_clear(led_strip);

  ESP_LOGI(TAG, "LED Strip initialized: %d LEDs on GPIO %d", NUM_LEDS, LED_PIN);
  ESP_LOGI(TAG, "========================================");
}

void setupAFE() {
  ESP_LOGI(TAG, "========================================");
  ESP_LOGI(TAG, "Initializing AFE...");

  print_memory_info();

  srmodel_list_t *models = esp_srmodel_init("model");

  afe_config_t *afe_cfg = afe_config_init(
    "MM",
    models,
    AFE_TYPE_SR,
    AFE_MODE_LOW_COST
  );

  afe_cfg->aec_init = false;
  afe_cfg->se_init = true;
  afe_cfg->ns_init = true;
  afe_cfg->vad_init = true;
  afe_cfg->wakenet_init = false;
  afe_cfg->vad_mode = VAD_MODE_3;
  afe_cfg->afe_ringbuf_size = 100;
  afe_cfg->memory_alloc_mode = AFE_MEMORY_ALLOC_MORE_PSRAM;

  ESP_LOGI(TAG, "PCM Config - Total channels: %d, Mic channels: %d, Ref channels: %d",
           afe_cfg->pcm_config.total_ch_num,
           afe_cfg->pcm_config.mic_num,
           afe_cfg->pcm_config.ref_num);

  ESP_LOGI(TAG, "AFE config created (ringbuf_size=100, PSRAM mode)");
  print_memory_info();

  afe_handle = esp_afe_handle_from_config(afe_cfg);

  ESP_LOGI(TAG, "Got AFE handle");
  print_memory_info();

  afe_data = afe_handle->create_from_config(afe_cfg);

  ESP_LOGI(TAG, "AFE instance created successfully!");
  print_memory_info();

  afe_feed_chunksize = afe_handle->get_feed_chunksize(afe_data);
  afe_feed_channels = afe_handle->get_feed_channel_num(afe_data);

  ESP_LOGI(TAG, "Feed chunk size: %d samples", afe_feed_chunksize);
  ESP_LOGI(TAG, "Feed channels: %d", afe_feed_channels);

  size_t buf_size = afe_feed_chunksize * afe_feed_channels * sizeof(int16_t);
  afe_feed_buf = (int16_t *)heap_caps_calloc(1, buf_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

  if (!afe_feed_buf) {
    ESP_LOGE(TAG, "Failed to allocate %d byte feed buffer in PSRAM", (int)buf_size);
    afe_handle->destroy(afe_data);
    afe_data = NULL;
    esp_srmodel_deinit(models);
    use_afe = false;
    return;
  }

  ESP_LOGI(TAG, "AFE feed buffer allocated: %d bytes in PSRAM", (int)buf_size);

  ESP_LOGI(TAG, "========================================");
  ESP_LOGI(TAG, "AFE initialized successfully!");
  ESP_LOGI(TAG, "========================================");

  print_memory_info();

  use_afe = true;

  esp_srmodel_deinit(models);
}

void setupDOA() {
  ESP_LOGI(TAG, "========================================");
  ESP_LOGI(TAG, "Initializing DOA (Direction of Arrival)...");

  float mic_distance_meters = 0.0485;  // Update this with your actual measurement!

  doa_handle = afe_doa_create(
    "MM",
    SAMPLE_RATE,
    20.0,
    mic_distance_meters,
    FRAME_SAMPLES
  );

  if (doa_handle == NULL) {
    ESP_LOGE(TAG, "Failed to create DOA instance");
    ESP_LOGW(TAG, "DOA disabled - continuing without spatial filtering");
    use_doa = false;
    return;
  }

  use_doa = true;

  ESP_LOGI(TAG, "DOA initialized successfully!");
  ESP_LOGI(TAG, "Microphone spacing: %.3f meters (%.0f mm)",
           mic_distance_meters, mic_distance_meters * 1000);
  ESP_LOGI(TAG, "Angular resolution: 20 degrees");
  ESP_LOGI(TAG, "Detection range: 0° to 180°");
  ESP_LOGI(TAG, "Baby monitoring zone: %.0f° to %.0f°", BABY_ZONE_MIN, BABY_ZONE_MAX);
  ESP_LOGI(TAG, "Ignore zone: %.0f° to %.0f°", IGNORE_ZONE_MIN, IGNORE_ZONE_MAX);
  ESP_LOGI(TAG, "========================================");
}

void updateLEDs_VUMeter(double rms_db, double voice_db) {
  double led_min_db = voice_db - 10.0;
  double led_max_db = voice_db + 20.0;

  int num_lit = map(constrain(rms_db, led_min_db, led_max_db),
                    led_min_db, led_max_db, 0, NUM_LEDS);

  led_strip_clear(led_strip);

  for (int i = 0; i < num_lit; i++) {
    uint8_t r, g, b;

    if (i < NUM_LEDS / 3) {
      r = 0; g = 255; b = 0;
    } else if (i < NUM_LEDS * 2 / 3) {
      r = 255; g = 255; b = 0;
    } else {
      r = 255; g = 0; b = 0;
    }

    r = (r * BRIGHTNESS) / 255;
    g = (g * BRIGHTNESS) / 255;
    b = (b * BRIGHTNESS) / 255;

    led_strip_set_pixel(led_strip, i, r, g, b);
  }

  led_strip_refresh(led_strip);
}

int readPotAvg() {
  long sum = 0;
  for (int i = 0; i < 8; i++) {
    sum += analogRead(PIN_POT);
    delayMicroseconds(200);
  }
  return sum / 8;
}

void setup() {
  Serial.begin(115200);
  delay(500);

  ESP_LOGI(TAG, "Serial initialized at 115200 baud");

  setupLEDs();
  delay(100);
  setupI2S();
  setupAFE();
  setupDOA();

  pinMode(PIN_DETECT, OUTPUT);
  pinMode(PIN_SCREAM, OUTPUT);
  pinMode(PIN_CRY, OUTPUT);
  pinMode(PIN_REAL, OUTPUT);
  ESP_LOGI(TAG, "GPIO pins configured");

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  ESP_LOGI(TAG, "ADC configured");

  cl_init(&g_cl, SAMPLE_RATE);
  mr_init(SAMPLE_RATE);

  ESP_LOGI(TAG, "========================================");
  ESP_LOGI(TAG, "AFE Status: %s", use_afe ? "ENABLED" : "DISABLED");
  ESP_LOGI(TAG, "Setup complete! Starting detection...");
  ESP_LOGI(TAG, "========================================");
}

static uint32_t hb = 0;
uint32_t now_ms = millis();
if (now_ms - hb > 1000) {
  hb = now_ms;
  ESP_LOGW(TAG, "loop alive");
}


void loop() {
  static int32_t i2s_buf[FRAME_SAMPLES * 2];
  static int16_t audio_stereo[FRAME_SAMPLES * 2];
  static int16_t mono16[FRAME_SAMPLES];

  static int loop_count = 0;
  static float last_sound_angle = 90.0f;
  static uint32_t last_cls_ms = 0;
  static uint32_t laugh_until_ms = 0;

  size_t bytes_read = 0;

  // Read stereo data from single I2S channel
  ESP_LOGW(TAG, "LOOP: before i2s_channel_read");

  esp_err_t err = i2s_channel_read(
    rx_chan,
    i2s_buf,
    sizeof(i2s_buf),
    &bytes_read,
    pdMS_TO_TICKS(10)   // ✅ 先改成 10ms，避免卡死
  );

  ESP_LOGW(TAG, "LOOP: after i2s_channel_read err=%d bytes=%u", (int)err, (unsigned)bytes_read);

  if (err != ESP_OK || bytes_read == 0) {
    ESP_LOGW(TAG, "I2S read timeout/error");
    return;
  }


  int frames = bytes_read / (sizeof(int32_t) * 2);

  // Convert 32-bit stereo to 16-bit stereo
  for (int i = 0; i < frames; i++) {
    int32_t sampleL = i2s_buf[i * 2 + 0] >> 11;
    int32_t sampleR = i2s_buf[i * 2 + 1] >> 11;
    audio_stereo[i * 2 + 0] = (int16_t)sampleL;
    audio_stereo[i * 2 + 1] = (int16_t)sampleR;
  }

  // Process with AFE if enabled
  int16_t *processed_audio = audio_stereo;
  int processed_frames = frames;
  bool vad_detected = false;

  if (use_afe && afe_handle && afe_data && afe_feed_buf) {
    for (int i = 0; i < afe_feed_chunksize && i < frames; i++) {
      afe_feed_buf[i * 2 + 0] = audio_stereo[i * 2 + 0];
      afe_feed_buf[i * 2 + 1] = audio_stereo[i * 2 + 1];
    }

    afe_handle->feed(afe_data, afe_feed_buf);
    vTaskDelay(pdMS_TO_TICKS(1));

    afe_fetch_result_t *afe_result = afe_handle->fetch(afe_data);
    if (afe_result && afe_result->ret_value == ESP_OK && afe_result->data) {
      processed_audio = afe_result->data;
      processed_frames = afe_result->data_size / sizeof(int16_t);
      vad_detected = (afe_result->vad_state == VAD_SPEECH);
    }

    afe_fetch_result_t *afe_result2 = afe_handle->fetch(afe_data);
    if (afe_result2 && afe_result2->ret_value == ESP_OK && afe_result2->data) {
      processed_audio = afe_result2->data;
      processed_frames = afe_result2->data_size / sizeof(int16_t);
      vad_detected = (afe_result2->vad_state == VAD_SPEECH);
    }
  }

  // Calculate RMS
  int64_t sum_left = 0;
  int64_t sum_right = 0;
  int64_t max_left = 0;
  int64_t max_right = 0;

  for (int i = 0; i < FRAME_SAMPLES; i++) {
    int32_t left = i2s_buf[i * 2];
    int32_t right = i2s_buf[i * 2 + 1];

    if (abs(left) > max_left) max_left = abs(left);
    if (abs(right) > max_right) max_right = abs(right);

    left = left >> 11;
    right = right >> 11;

    sum_left += (int64_t)left * left;
    sum_right += (int64_t)right * right;
  }

  double rms_left = sqrt((double)sum_left / FRAME_SAMPLES);
  double rms_right = sqrt((double)sum_right / FRAME_SAMPLES);
  double rms_db_left = 20.0 * log10(rms_left + 1.0);
  double rms_db_right = 20.0 * log10(rms_right + 1.0);

  ESP_LOGI(TAG, "Max samples: L=%lld, R=%lld", max_left, max_right);
  ESP_LOGI(TAG, "RMS: L=%.1f dB, R=%.1f dB", rms_db_left, rms_db_right);

  double rms_db = max(rms_db_left, rms_db_right);

  int pot = readPotAvg();
  double voice_db = map(pot, 0, 4095, 30, 90);
  double scream_db = voice_db + 20.0;

  bool loud_enough = rms_db > voice_db;
  bool screaming = rms_db > scream_db;

  // DOA processing
  float sound_angle = last_sound_angle;
  bool sound_from_baby_zone = (last_sound_angle >= BABY_ZONE_MIN &&
                              last_sound_angle <= BABY_ZONE_MAX);

  if (use_doa && doa_handle) {
    sound_angle = afe_doa_process(doa_handle, audio_stereo);

    sound_from_baby_zone = (sound_angle >= BABY_ZONE_MIN &&
                            sound_angle <= BABY_ZONE_MAX);

    bool sound_from_ignore_zone = (sound_angle >= IGNORE_ZONE_MIN &&
                                   sound_angle <= IGNORE_ZONE_MAX);

    const char* zone_name = sound_from_baby_zone ? "BABY" :
                            sound_from_ignore_zone ? "IGNORE" : "OTHER";
    ESP_LOGI(TAG, "DOA: %.1f° - Zone: %s", sound_angle, zone_name);

    last_sound_angle = sound_angle;
  }

  // Build MONO buffer for FFT gate + laugh classifier
  int mono_len = 0;

  if (use_afe && processed_audio != audio_stereo) {
    mono_len = min(processed_frames, FRAME_SAMPLES);
    memcpy(mono16, processed_audio, (size_t)mono_len * sizeof(int16_t));
  } else {
    mono_len = min(frames, FRAME_SAMPLES);
    for (int i = 0; i < mono_len; i++) {
      int32_t L = audio_stereo[i * 2 + 0];
      int32_t R = audio_stereo[i * 2 + 1];
      mono16[i] = (int16_t)((L + R) / 2);
    }
    vad_detected = true;
  }

  // Feed FFT gate + laugh classifier
  mr_feed_mono_block(mono16, mono_len);
  cl_feed_mono_block(&g_cl, mono16, mono_len);

  bool reject = mr_is_male_like(); // true => likely adult-male-like speech

  uint32_t now = millis();

  // =========================
  // STATE MACHINE:
  //  - ARMED: allow laugh detection
  //  - LOCKED: stop detection until quiet
  // =========================

  // While LOCKED -> wait for quiet then re-arm
  if (g_state == DETECT_LOCKED) {
    if (!loud_enough) {
      if (g_quiet_since_ms == 0) g_quiet_since_ms = now;
      else if (now - g_quiet_since_ms >= QUIET_REARM_MS) {
        g_state = DETECT_ARMED;
        g_quiet_since_ms = 0;
        ESP_LOGI(TAG, "System re-armed after quiet");
      }
    } else {
      g_quiet_since_ms = 0;
    }
  }

  // ARMED detection (periodic)
  bool can_classify = loud_enough && vad_detected && sound_from_baby_zone && !reject;

  if (g_state == DETECT_ARMED && can_classify) {
    if (now - last_cls_ms >= CLS_PERIOD_MS) {
      last_cls_ms = now;

      CL_Debug dbg = {};
      CL_Label lab = cl_classify_latest(&g_cl, &dbg);

      const char* lab_str = (lab == CL_LAUGH) ? "laugh" : "unknown";
      ESP_LOGI(TAG, "Classifier: %s (peaks=%d, pps=%.2f, fast=%.2f, run=%d, clusters=%d)",
               lab_str, dbg.num_peaks, dbg.peaks_per_sec, dbg.fast_repeat_ratio, dbg.max_fast_run, dbg.cluster_count);

      // ✅ FINAL: trigger immediately when 2 clusters observed
      bool final_laugh =
        (lab == CL_LAUGH) &&
        (dbg.num_peaks >= MIN_TOTAL_PEAKS_FOR_FINAL) &&
        (dbg.cluster_count >= MIN_CLUSTER_FOR_FINAL);

      if (final_laugh) {
        laugh_until_ms = now + 800;

        ESP_LOGW(TAG,
          "=== FINAL DECISION: LAUGH === (clusters=%d peaks=%d) -> LOCKED until quiet",
          dbg.cluster_count, dbg.num_peaks
        );

        g_state = DETECT_LOCKED;
        g_quiet_since_ms = 0;
      }
    }
  } else {
    // occasional skip log (avoid spam)
    static int skip_ctr = 0;
    skip_ctr++;
    if (skip_ctr % 20 == 0) {
      ESP_LOGI(TAG, "Classifier skipped: state=%d loud=%d vad=%d baby=%d reject=%d",
               (int)g_state,
               loud_enough ? 1 : 0,
               vad_detected ? 1 : 0,
               sound_from_baby_zone ? 1 : 0,
               reject ? 1 : 0);
    }
  }

  // Optional logs (every ~1s)
  static int log_ctr = 0;
  log_ctr++;
  if (log_ctr % 4 == 0) {
    MR_Debug md;
    mr_get_debug(&md);

    ESP_LOGI(TAG,
      "RMS: %.1fdB thr:%.1f angle:%.1f baby:%d VAD:%d reject:%d LAUGH_LATCH:%d | FFT ratio=%.2f cent=%.0f score=%d | state=%d",
      rms_db, voice_db, last_sound_angle,
      sound_from_baby_zone ? 1 : 0,
      vad_detected ? 1 : 0,
      reject ? 1 : 0,
      (now < laugh_until_ms) ? 1 : 0,
      md.low_mid_ratio, md.centroid_hz, md.score,
      (int)g_state
    );
  }

  updateLEDs_VUMeter(rms_db, voice_db);

  // Output control (original)
  if (screaming) {
    digitalWrite(PIN_SCREAM, HIGH);
    digitalWrite(PIN_CRY, LOW);
    digitalWrite(PIN_DETECT, LOW);
  } else if (loud_enough) {
    digitalWrite(PIN_SCREAM, LOW);
    digitalWrite(PIN_CRY, LOW);
    digitalWrite(PIN_DETECT, HIGH);
  } else {
    digitalWrite(PIN_SCREAM, LOW);
    digitalWrite(PIN_CRY, LOW);
    digitalWrite(PIN_DETECT, LOW);
  }

  if (!screaming && loud_enough && sound_from_baby_zone) {
    digitalWrite(PIN_REAL, HIGH);
  } else {
    digitalWrite(PIN_REAL, LOW);
  }

  loop_count++;
  if (loop_count % 20 == 0) {
    ESP_LOGI(TAG, "Stats - RMS: %.1f dB, Threshold: %.1f, Angle: %.1f°, Zone: %s, VAD: %s",
             rms_db, voice_db, last_sound_angle,
             sound_from_baby_zone ? "BABY" : "OTHER",
             vad_detected ? "SPEECH" : "SILENCE");
  }

  delay(50);
}

extern "C" void app_main(void) {
  ESP_LOGI(TAG, "app_main() called - initializing Arduino...");
  initArduino();
  ESP_LOGI(TAG, "Arduino initialized - calling setup()...");
  setup();
  ESP_LOGI(TAG, "Setup complete - entering main loop...");
  while (true) {
    loop();
    vTaskDelay(1);
  }
}