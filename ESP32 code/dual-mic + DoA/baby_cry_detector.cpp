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


static const char* TAG = "BabyCry";

// Pin definitions for both microphones
static const int PIN_BCLK = 12;
static const int PIN_WS   = 13;
static const int PIN_DIN = 11;

// Control pin
static const int PIN_POT = 7;

// Output and status pins
static const int PIN_DETECT = 8;
static const int PIN_SCREAM = 9;
static const int PIN_CRY = 10;
static const int PIN_REAL = 6;  //PIN_REAL

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

void print_memory_info() {
  ESP_LOGI(TAG, "=== Memory Info ===");
  ESP_LOGI(TAG, "Free heap: %lu bytes", esp_get_free_heap_size());
  ESP_LOGI(TAG, "Free PSRAM: %lu bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  ESP_LOGI(TAG, "Largest free block PSRAM: %lu bytes", heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
  ESP_LOGI(TAG, "Min free heap: %lu bytes", esp_get_minimum_free_heap_size());
  ESP_LOGI(TAG, "==================");
}

// Initialize AFE for dual-mic input
void setupAFE() {
  ESP_LOGI(TAG, "========================================");
  ESP_LOGI(TAG, "Initializing AFE...");
  
  print_memory_info();
  
  // Load SR models
  srmodel_list_t *models = esp_srmodel_init("model");
  
  // Create AFE config - use PSRAM aggressively
  afe_config_t *afe_cfg = afe_config_init(
    "MM",
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
  
  // The pcm_config is automatically set from "MM" string, but verify:
  ESP_LOGI(TAG, "PCM Config - Total channels: %d, Mic channels: %d, Ref channels: %d",
           afe_cfg->pcm_config.total_ch_num,
           afe_cfg->pcm_config.mic_num,
           afe_cfg->pcm_config.ref_num);

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

// Read potentiometer with averaging
int readPotAvg() {
  long sum = 0;
  for (int i = 0; i < 8; i++) {
    sum += analogRead(PIN_POT);
    delayMicroseconds(200);
  }
  return sum / 8;
}

// Initialize Direction of Arrival (with AFE implementation) for dual-mic input
void setupDOA() {
  ESP_LOGI(TAG, "========================================");
  ESP_LOGI(TAG, "Initializing DOA (Direction of Arrival)...");
  
  // IMPORTANT: Measure the actual distance between your microphones!
  float mic_distance_meters = 0.046;  // Update this with your actual measurement!
  
  // DOA parameters
  // input_format: "MM" = two microphones, interleaved stereo
  // fs: 16000 Hz sample rate
  // resolution: 20 degrees (higher = more CPU, lower = less precise)
  // d_mics: microphone spacing in meters
  // input_samples: frame size for processing
  
  doa_handle = afe_doa_create(
    "MM",                    // Two mic input format (interleaved stereo)
    SAMPLE_RATE,            // 16000 Hz
    20.0,                   // 20 degree resolution
    mic_distance_meters,    // Mic spacing in meters
    FRAME_SAMPLES           // Frame size
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

void setup() {  
  Serial.begin(115200);
  delay(500);
  
  ESP_LOGI(TAG, "Serial initialized at 115200 baud");
  
  setupI2S();
  setupAFE();  // AFE after I2S
  setupDOA();  // DOA after AFE
  
  // Initialise
  pinMode(PIN_DETECT, OUTPUT);
  pinMode(PIN_SCREAM, OUTPUT);
  pinMode(PIN_CRY, OUTPUT);
  pinMode(PIN_REAL, OUTPUT);
  ESP_LOGI(TAG, "GPIO pins configured");

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  ESP_LOGI(TAG, "ADC configured");
  
  ESP_LOGI(TAG, "========================================");
  ESP_LOGI(TAG, "AFE Status: %s", use_afe ? "ENABLED" : "DISABLED");
  ESP_LOGI(TAG, "Setup complete! Starting detection...");
  ESP_LOGI(TAG, "========================================");
}

void loop() {
  static int32_t i2s_buf[FRAME_SAMPLES * 2];
  static int16_t audio_stereo[FRAME_SAMPLES * 2];
  static int loop_count = 0;
  static float last_sound_angle = 90.0;
  static int doa_call_counter = 0;
  static int cached_pot_value = 2048;  // Default to middle
  size_t bytes_read = 0;

  // Read stereo data from single I2S channel
  esp_err_t err = i2s_channel_read(rx_chan, i2s_buf, sizeof(i2s_buf), &bytes_read, pdMS_TO_TICKS(200));
  
  if (err != ESP_OK || bytes_read == 0) {
    ESP_LOGW(TAG, "I2S read timeout/error");
    return;
  }

  int frames = bytes_read / (sizeof(int32_t) * 2);
  
  // Convert 32-bit stereo to 16-bit stereo
  for (int i = 0; i < frames; i++) {
    int32_t sampleL = i2s_buf[i * 2 + 0] >> 11;  // Mic 1 (Left channel)
    int32_t sampleR = i2s_buf[i * 2 + 1] >> 11;  // Mic 2 (Right channel)
    audio_stereo[i * 2 + 0] = (int16_t)sampleL;
    audio_stereo[i * 2 + 1] = (int16_t)sampleR;
  }


  // Process with AFE if enabled
  int16_t *processed_audio = audio_stereo;
  int processed_frames = frames;
  bool vad_detected = false;
  
if (use_afe && afe_handle && afe_data && afe_feed_buf) {

    // Prepare feed buffer
    for (int i = 0; i < afe_feed_chunksize && i < frames; i++) {
      afe_feed_buf[i * 2 + 0] = audio_stereo[i * 2 + 0];
      afe_feed_buf[i * 2 + 1] = audio_stereo[i * 2 + 1];
    }
    
    // Feed the data
    afe_handle->feed(afe_data, afe_feed_buf);

    // Small delay to let AFE process
    vTaskDelay(pdMS_TO_TICKS(1));
    
    // First fetch
    afe_fetch_result_t *afe_result = afe_handle->fetch(afe_data);

    if (afe_result && afe_result->ret_value == ESP_OK && afe_result->data) {
      processed_audio = afe_result->data;
      processed_frames = afe_result->data_size / sizeof(int16_t);
      vad_detected = (afe_result->vad_state == VAD_SPEECH);
    }
    
    // Second fetch - drain remaining data
    afe_fetch_result_t *afe_result2 = afe_handle->fetch(afe_data);

    if (afe_result2 && afe_result2->ret_value == ESP_OK && afe_result2->data) {
      // Use the second fetch result (most recent processed data)
      processed_audio = afe_result2->data;
      processed_frames = afe_result2->data_size / sizeof(int16_t);
      vad_detected = (afe_result2->vad_state == VAD_SPEECH);
    }
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

  // DOA processing
  float sound_angle = last_sound_angle;
  bool sound_from_baby_zone = (last_sound_angle >= BABY_ZONE_MIN && 
                               last_sound_angle <= BABY_ZONE_MAX);

  if (use_doa && doa_handle) {
    // Process DOA with interleaved stereo audio
    sound_angle = afe_doa_process(doa_handle, audio_stereo);
    
    // Check if sound is from baby zone
    sound_from_baby_zone = (sound_angle >= BABY_ZONE_MIN && 
                            sound_angle <= BABY_ZONE_MAX);
    
    bool sound_from_ignore_zone = (sound_angle >= IGNORE_ZONE_MIN && 
                                    sound_angle <= IGNORE_ZONE_MAX);
    
    const char* zone_name = sound_from_baby_zone ? "BABY" : 
                              sound_from_ignore_zone ? "IGNORE" : "OTHER";
    ESP_LOGI(TAG, "DOA: %.1f° - Zone: %s", sound_angle, zone_name);
    
    last_sound_angle = sound_angle;

    // Early exit if sound is from ignore zone
    if (sound_from_ignore_zone) {
      loop_count++;
      delay(50);
      return;
    }
  }

  // Output control
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
  }
  else {
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
