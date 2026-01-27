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

// Pin definitions for both microphones
static const int PIN_BCLK = 12;
static const int PIN_WS   = 13;
static const int PIN_DIN_1  = 11;
static const int PIN_DIN_2  = 6;

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

static i2s_chan_handle_t rx_chan_1;  // First microphone
static i2s_chan_handle_t rx_chan_2;  // Second microphone

// AFE variables
static const esp_afe_sr_iface_t *afe_handle = NULL;
static esp_afe_sr_data_t *afe_data = NULL;
static int16_t *afe_feed_buf = NULL;  // Make this global so we can free it
static bool use_afe = false;
static int afe_feed_chunksize = 0;
static int afe_feed_channels = 0;

void setupI2S() {
  ESP_LOGI(TAG, "Setting up I2S...");
  
  // MIC 1
  i2s_chan_config_t chan_cfg_1 = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg_1, NULL, &rx_chan_1));

  i2s_std_config_t std_cfg_1 = {};
  std_cfg_1.clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE);
  std_cfg_1.slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO);

  std_cfg_1.gpio_cfg.mclk = I2S_GPIO_UNUSED;
  std_cfg_1.gpio_cfg.bclk = (gpio_num_t)PIN_BCLK;
  std_cfg_1.gpio_cfg.ws   = (gpio_num_t)PIN_WS;
  std_cfg_1.gpio_cfg.dout = I2S_GPIO_UNUSED;
  std_cfg_1.gpio_cfg.din  = (gpio_num_t)PIN_DIN_1;
  std_cfg_1.gpio_cfg.invert_flags.mclk_inv = false;
  std_cfg_1.gpio_cfg.invert_flags.bclk_inv = false;
  std_cfg_1.gpio_cfg.invert_flags.ws_inv   = false;

  ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_chan_1, &std_cfg_1));
  ESP_ERROR_CHECK(i2s_channel_enable(rx_chan_1));

  ESP_LOGI(TAG, "Microphone 1 initialized on I2S_NUM_0");

  // MIC 2
  i2s_chan_config_t chan_cfg_2 = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg_2, NULL, &rx_chan_2));

  i2s_std_config_t std_cfg_2 = {};
  std_cfg_2.clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE);
  std_cfg_2.slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO);

  std_cfg_2.gpio_cfg.mclk = I2S_GPIO_UNUSED;
  std_cfg_2.gpio_cfg.bclk = (gpio_num_t)PIN_BCLK;
  std_cfg_2.gpio_cfg.ws   = (gpio_num_t)PIN_WS;
  std_cfg_2.gpio_cfg.dout = I2S_GPIO_UNUSED;
  std_cfg_2.gpio_cfg.din  = (gpio_num_t)PIN_DIN_2;
  std_cfg_2.gpio_cfg.invert_flags.mclk_inv = false;
  std_cfg_2.gpio_cfg.invert_flags.bclk_inv = false;
  std_cfg_2.gpio_cfg.invert_flags.ws_inv   = false;

  ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_chan_2, &std_cfg_2));
  ESP_ERROR_CHECK(i2s_channel_enable(rx_chan_2));
  
  ESP_LOGI(TAG, "Microphone 2 initialized on I2S_NUM_1");
  ESP_LOGI(TAG, "Dual I2S setup complete!");
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
    "TM",
    models,
    AFE_TYPE_SR,
    AFE_MODE_LOW_COST
  );

// Configure for dual-mic with minimal memory usage
  afe_cfg->aec_init = false;      // No acoustic echo cancellation
  afe_cfg->se_init = true;        // Enable speech enhancement (benefits from 2 mics)
  afe_cfg->ns_init = true;        // Enable noise suppression (benefits from 2 mics)
  afe_cfg->vad_init = true;       // Voice activity detection
  afe_cfg->wakenet_init = false;  // No wake word detection
  afe_cfg->vad_mode = VAD_MODE_3; // Aggressive VAD
  afe_cfg->afe_ringbuf_size = 100; // Ring buffer size
  afe_cfg->memory_alloc_mode = AFE_MEMORY_ALLOC_MORE_PSRAM;
  
  // Dual-mic specific settings
  afe_cfg->pcm_config.total_ch_num = 2;  // 2 microphones
  afe_cfg->pcm_config.mic_num = 2;       // 2 microphones
  afe_cfg->pcm_config.ref_num = 0;       // No reference channel

  ESP_LOGI(TAG, "AFE config created (ringbuf_size=100, PSRAM mode)");
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
  static int32_t i2s_buf_1[FRAME_SAMPLES];  // Mic 1 buffer (mono)
  static int32_t i2s_buf_2[FRAME_SAMPLES];  // Mic 2 buffer (mono)
  static int16_t audio_stereo[FRAME_SAMPLES * 2];
  static int loop_count = 0;
  size_t bytes_read_1 = 0;
  size_t bytes_read_2 = 0;

  // Read from both I2S channels
  esp_err_t err1 = i2s_channel_read(rx_chan_1, i2s_buf_1, sizeof(i2s_buf_1), &bytes_read_1, pdMS_TO_TICKS(200));
  esp_err_t err2 = i2s_channel_read(rx_chan_2, i2s_buf_2, sizeof(i2s_buf_2), &bytes_read_2, pdMS_TO_TICKS(200));
  
  if (err1 != ESP_OK || bytes_read_1 == 0 || err2 != ESP_OK || bytes_read_2 == 0) {
    ESP_LOGW(TAG, "I2S read timeout/error on one or both channels");
    return;
  }

  int frames = min(bytes_read_1 / sizeof(int32_t), bytes_read_2 / sizeof(int32_t));
  
  // Convert 32-bit stereo to 16-bit stereo (keep both channels!)
  for (int i = 0; i < frames; i++) {
    int32_t sample1 = i2s_buf_1[i] >> 11;  // Mic 1
    int32_t sample2 = i2s_buf_2[i] >> 11;  // Mic 2
    audio_stereo[i * 2 + 0] = (int16_t)sample1;
    audio_stereo[i * 2 + 1] = (int16_t)sample2;
  }

  // Process with AFE if enabled
  int16_t *processed_audio = audio_stereo;
  int processed_frames = frames;
  bool vad_detected = false;
  
  if (use_afe && afe_handle && afe_data && afe_feed_buf) {
    // Prepare feed buffer - interleaved stereo format
    for (int i = 0; i < afe_feed_chunksize && i < frames; i++) {
      afe_feed_buf[i * 2 + 0] = audio_stereo[i * 2 + 0];  // Left mic
      afe_feed_buf[i * 2 + 1] = audio_stereo[i * 2 + 1];  // Right mic
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

  // Calculate RMS - handle both stereo input and mono AFE output
  double sum = 0.0;
  int sample_count = 0;
  
  if (use_afe && processed_audio != audio_stereo) {
    // AFE output is mono
    for (int i = 0; i < processed_frames; i++) {
      sum += (double)processed_audio[i] * (double)processed_audio[i];
      sample_count++;
    }
  } else {
    // Raw stereo - average both channels
    for (int i = 0; i < frames; i++) {
      int16_t avg = (audio_stereo[i * 2 + 0] + audio_stereo[i * 2 + 1]) / 2;
      sum += (double)avg * (double)avg;
      sample_count++;
    }
  }
  
  double rms = std::sqrt(sum / processed_frames);
  double rms_db = 20.0 * std::log10(rms + 1e-9);
  
  int pot = readPotAvg();
  double voice_db = map(pot, 0, 4095, 30, 90);
  double scream_db = voice_db + 20.0;

  // Detection logic
  bool loud_enough = rms_db > voice_db;
  bool screaming = rms_db > scream_db;

  // Output control
  if (screaming) {
    digitalWrite(PIN_SCREAM, HIGH);
    digitalWrite(PIN_CRY, LOW);
    digitalWrite(PIN_DETECT, LOW);
  } 
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
  printf("RMS:%.1f,Threshold:%.1f,Scream:%.1f, AFE:%d,VAD:%d\n",
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