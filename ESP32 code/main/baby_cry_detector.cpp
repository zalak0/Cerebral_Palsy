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

static const char* TAG = "BabyCry";

static const int PIN_BCLK = 12;
static const int PIN_WS   = 13;
static const int PIN_DIN  = 11;

static const int PIN_POT = 7;
static const int PIN_DETECT = 8;
static const int PIN_SCREAM = 9;
static const int PIN_CRY = 10;

static const int SAMPLE_RATE = 16000;
static const int FRAME_SAMPLES = 512;

// ZCR history for variation tracking
static const int ZCR_HISTORY_SIZE = 10;
static float zcr_history[ZCR_HISTORY_SIZE];
static int zcr_history_index = 0;
static bool zcr_history_filled = false;

static i2s_chan_handle_t rx_chan;

// AFE variables
static const esp_afe_sr_iface_t *afe_handle = NULL;
static esp_afe_sr_data_t *afe_data = NULL;
static int16_t *afe_feed_buf = NULL;  // Make this global so we can free it
static bool use_afe = false;
static int afe_feed_chunksize = 0;
static int afe_feed_channels = 0;

void setupI2S() {
  ESP_LOGI(TAG, "Setting up I2S...");
  
  i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_chan));

  i2s_std_config_t std_cfg = {};
  std_cfg.clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE);
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
  
  ESP_LOGI(TAG, "I2S setup complete!");
}

void print_memory_info() {
  ESP_LOGI(TAG, "=== Memory Info ===");
  ESP_LOGI(TAG, "Free heap: %lu bytes", esp_get_free_heap_size());
  ESP_LOGI(TAG, "Free PSRAM: %lu bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  ESP_LOGI(TAG, "Largest free block PSRAM: %lu bytes", heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
  ESP_LOGI(TAG, "Min free heap: %lu bytes", esp_get_minimum_free_heap_size());
  ESP_LOGI(TAG, "==================");
}

void setupAFE() {
  ESP_LOGI(TAG, "========================================");
  ESP_LOGI(TAG, "Initializing AFE...");
  
  print_memory_info();
  
  // Load SR models
  srmodel_list_t *models = esp_srmodel_init("model");
  
  // Create AFE config - use PSRAM aggressively
  afe_config_t *afe_cfg = afe_config_init(
    "MR",
    models,
    AFE_TYPE_SR,
    AFE_MODE_LOW_COST
  );

  // Configure for minimal memory usage
  afe_cfg->aec_init = false;
  afe_cfg->se_init = false;
  afe_cfg->ns_init = false;
  afe_cfg->vad_init = true;
  afe_cfg->wakenet_init = false;
  afe_cfg->vad_mode = VAD_MODE_3;
  afe_cfg->afe_ringbuf_size = 100;  // Minimum ring buffer
  afe_cfg->memory_alloc_mode = AFE_MEMORY_ALLOC_MORE_PSRAM;  // Force PSRAM usage
  
  ESP_LOGI(TAG, "AFE config created (ringbuf_size=5, PSRAM mode)");
  print_memory_info();
  
  // Get AFE interface
  afe_handle = esp_afe_handle_from_config(afe_cfg);
  
  ESP_LOGI(TAG, "Got AFE handle");
  print_memory_info();
  
  // Create AFE instance
  afe_data = afe_handle->create_from_config(afe_cfg);
  
  ESP_LOGI(TAG, "AFE instance created successfully!");
  print_memory_info();
  
  // Get AFE parameters
  afe_feed_chunksize = afe_handle->get_feed_chunksize(afe_data);
  afe_feed_channels = afe_handle->get_feed_channel_num(afe_data);
  
  ESP_LOGI(TAG, "Feed chunk size: %d samples", afe_feed_chunksize);
  ESP_LOGI(TAG, "Feed channels: %d", afe_feed_channels);
  
  // Allocate feed buffer in PSRAM
  size_t buf_size = afe_feed_chunksize * afe_feed_channels * sizeof(int16_t);
  afe_feed_buf = (int16_t *)heap_caps_calloc(1, buf_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

  
  if (!afe_feed_buf) {
    ESP_LOGE(TAG, "Failed to allocate %d byte feed buffer in PSRAM", buf_size);
    afe_handle->destroy(afe_data);
    afe_data = NULL;
    esp_srmodel_deinit(models);
    use_afe = false;
    return;
  }
  
  ESP_LOGI(TAG, "AFE feed buffer allocated: %d bytes in PSRAM", buf_size);
  
  ESP_LOGI(TAG, "========================================");
  ESP_LOGI(TAG, "AFE initialized successfully!");
  ESP_LOGI(TAG, "========================================");
  
  print_memory_info();
  
  use_afe = true;
  
  // Clean up model list (but keep AFE data!)
  esp_srmodel_deinit(models);
}

void cleanupAFE() {
  if (afe_feed_buf) {
    free(afe_feed_buf);
    afe_feed_buf = NULL;
    ESP_LOGI(TAG, "AFE feed buffer freed");
  }
  
  if (afe_data && afe_handle) {
    afe_handle->destroy(afe_data);
    afe_data = NULL;
    ESP_LOGI(TAG, "AFE instance destroyed");
  }
  
  use_afe = false;
}

int readPotAvg() {
  long sum = 0;
  for (int i = 0; i < 8; i++) {
    sum += analogRead(PIN_POT);
    delayMicroseconds(200);
  }
  return sum / 8;
}

float zcr_calculation(int16_t buf[], int frames) {
  int zeroCrossings = 0;
  
  for (int i = 1; i < frames; i++) {
    if ((buf[i-1] >= 0 && buf[i] < 0) || (buf[i-1] < 0 && buf[i] >= 0)) {
      zeroCrossings++;
    }
  }

  return (float)zeroCrossings / frames;
}

float calculate_zcr_variation() {
  if (!zcr_history_filled && zcr_history_index < 5) {
    return 0.0;
  }
  
  int size = zcr_history_filled ? ZCR_HISTORY_SIZE : zcr_history_index;
  
  float sum = 0.0;
  for (int i = 0; i < size; i++) {
    sum += zcr_history[i];
  }
  float mean = sum / size;
  
  float variance = 0.0;
  for (int i = 0; i < size; i++) {
    float diff = zcr_history[i] - mean;
    variance += diff * diff;
  }
  float std_dev = std::sqrt(variance / size);
  
  return std_dev;
}

int find_pitch_period(int16_t buf[], int frames) {
  int min_period = 5;
  int max_period = 200;
  
  if (max_period >= frames) {
    max_period = frames - 1;
  }
  
  float max_correlation = 0.0;
  int best_period = min_period;
  
  for (int period = min_period; period <= max_period; period++) {
    float correlation = 0.0;
    int count = 0;
    
    for (int i = 0; i < frames - period; i++) {
      correlation += (float)buf[i] * (float)buf[i + period];
      count++;
    }
    
    if (count > 0) {
      correlation /= count;
    }
    
    if (correlation > max_correlation) {
      max_correlation = correlation;
      best_period = period;
    }
  }
  
  return best_period;
}

void setup() {  
  Serial.begin(115200);
  delay(500);
  
  ESP_LOGI(TAG, "Serial initialized at 115200 baud");
  
  setupI2S();
  setupAFE();  // AFE after I2S
  
  pinMode(PIN_DETECT, OUTPUT);
  pinMode(PIN_SCREAM, OUTPUT);
  pinMode(PIN_CRY, OUTPUT);
  ESP_LOGI(TAG, "GPIO pins configured");

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  ESP_LOGI(TAG, "ADC configured");
  
  for (int i = 0; i < ZCR_HISTORY_SIZE; i++) {
    zcr_history[i] = 0.0;
  }
  
  ESP_LOGI(TAG, "========================================");
  ESP_LOGI(TAG, "AFE Status: %s", use_afe ? "ENABLED" : "DISABLED");
  ESP_LOGI(TAG, "Setup complete! Starting detection...");
  ESP_LOGI(TAG, "========================================");
}

void loop() {
  static int32_t i2s_buf[FRAME_SAMPLES * 2];
  static int16_t audio_mono[FRAME_SAMPLES];
  static int loop_count = 0;
  size_t bytes_read = 0;

  // Read from I2S
  esp_err_t err = i2s_channel_read(rx_chan, i2s_buf, sizeof(i2s_buf), &bytes_read, pdMS_TO_TICKS(200));
  if (err != ESP_OK || bytes_read == 0) {
    ESP_LOGW(TAG, "I2S read timeout/error");
    return;
  }

  int frames = bytes_read / (sizeof(int32_t) * 2);
  
  // Convert 32-bit stereo to 16-bit mono
  for (int i = 0; i < frames; i++) {
    int32_t sampleL = i2s_buf[i * 2 + 0] >> 11;
    audio_mono[i] = (int16_t)sampleL;
  }

  // Process with AFE if enabled
  int16_t *processed_audio = audio_mono;
  int processed_frames = frames;
  bool vad_detected = false;
  
  if (use_afe && afe_handle && afe_data && afe_feed_buf) {
    // Prepare feed buffer
    for (int i = 0; i < afe_feed_chunksize && i < frames; i++) {
      if (afe_feed_channels == 1) {
        afe_feed_buf[i] = audio_mono[i];
      } else if (afe_feed_channels == 2) {
        afe_feed_buf[i * 2 + 0] = audio_mono[i];  // Mic
        afe_feed_buf[i * 2 + 1] = 0;              // Reference (silence)
      }
    }
    
    // Feed the data
    afe_handle->feed(afe_data, afe_feed_buf);
    
    // ALWAYS fetch after feeding - this is critical!
    // Fetch drains the ringbuffer regardless of feed result
    afe_fetch_result_t *afe_result = afe_handle->fetch(afe_data);
    
    if (afe_result && afe_result->ret_value == ESP_OK && afe_result->data) {
      processed_audio = afe_result->data;
      processed_frames = afe_result->data_size / sizeof(int16_t);
      vad_detected = (afe_result->vad_state == VAD_SPEECH);
    }
    // If fetch fails or returns no data, we fall back to raw audio_mono
  }

  // Rest of your code remains the same...
  // Calculate RMS
  double sum = 0.0;
  for (int i = 0; i < processed_frames; i++) {
    sum += (double)processed_audio[i] * (double)processed_audio[i];
  }
  double rms = std::sqrt(sum / processed_frames);
  double rms_db = 20.0 * std::log10(rms + 1e-9);

  // // Calculate features
  // float zcr = zcr_calculation(processed_audio, processed_frames);
  // zcr_history[zcr_history_index] = zcr;
  // zcr_history_index++;
  // if (zcr_history_index >= ZCR_HISTORY_SIZE) {
  //   zcr_history_index = 0;
  //   zcr_history_filled = true;
  // }

  // float zcr_variation = calculate_zcr_variation();
  // float pitch_period = find_pitch_period(processed_audio, processed_frames);
  // float pitch_freq = (float)SAMPLE_RATE / (float)pitch_period;

  // // Thresholds
  // float ZCR_CRY_MIN = 0.10;
  // float ZCR_CRY_MAX = 0.40;
  // float PITCH_CRY_MIN = 300;
  // float PITCH_CRY_MAX = 700;
  
  int pot = readPotAvg();
  double voice_db = map(pot, 0, 4095, 30, 90);
  double scream_db = voice_db + 20.0;

  // Detection logic
  bool loud_enough = rms_db > voice_db;
  bool screaming = rms_db > scream_db;

  // bool cry_zcr = (zcr > ZCR_CRY_MIN && zcr < ZCR_CRY_MAX);
  // bool cry_pitch = (pitch_freq > PITCH_CRY_MIN && pitch_freq < PITCH_CRY_MAX);
  // bool crying_detected = loud_enough && cry_zcr && cry_pitch;

  // Output control
  if (screaming) {
    digitalWrite(PIN_SCREAM, HIGH);
    digitalWrite(PIN_CRY, LOW);
    digitalWrite(PIN_DETECT, LOW);
  } 
  // else if (crying_detected) {
  //   digitalWrite(PIN_SCREAM, LOW);
  //   digitalWrite(PIN_CRY, HIGH);
  //   digitalWrite(PIN_DETECT, LOW);
  // } 
  else if (loud_enough) {
    digitalWrite(PIN_SCREAM, LOW);
    digitalWrite(PIN_CRY, LOW);
    digitalWrite(PIN_DETECT, HIGH);
  } else {
    digitalWrite(PIN_SCREAM, LOW);
    digitalWrite(PIN_CRY, LOW);
    digitalWrite(PIN_DETECT, LOW);
  }

  // Serial output for plotting
  printf("RMS:%.1f,Threshold:%.1f,Scream:%.1f,AFE:%d,VAD:%d\n",
         rms_db, voice_db, scream_db, 
         //zcr * 100, 
         //pitch_freq, 
         //crying_detected ? 80 : 0, 
         use_afe ? 1 : 0, 
         vad_detected ? 1 : 0);

  // Debug output every 20 loops
  loop_count++;
  if (loop_count % 20 == 0) {
    ESP_LOGI(TAG, "Stats - RMS: %.1f dB, AFE: %s, VAD: %s",
             rms_db, 
             use_afe ? "ON" : "OFF", vad_detected ? "SPEECH" : "SILENCE");
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
    
    // Cleanup (though this never runs)
    cleanupAFE();
}