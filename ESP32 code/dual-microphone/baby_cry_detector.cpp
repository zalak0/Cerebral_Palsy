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
#include "esp_dsp.h"


static const char* TAG = "BabyCry";

// Pin definitions for both microphones
static const int PIN_BCLK = 12;
static const int PIN_WS   = 13;
static const int PIN_DIN = 11;

static const int PIN_POT = 7;
static const int PIN_DETECT = 8;
static const int PIN_SCREAM = 9;
static const int PIN_CRY = 10;

static const int SAMPLE_RATE = 96000;
static const int FRAME_SAMPLES = 1024;

#define FFT_SIZE FRAME_SAMPLES
static float fft_l[FFT_SIZE * 2];
static float fft_r[FFT_SIZE * 2];
static float cross_spec[FFT_SIZE * 2];
static float corr[FFT_SIZE];

static i2s_chan_handle_t rx_chan;

// AFE variables
static const esp_afe_sr_iface_t *afe_handle = NULL;
static esp_afe_sr_data_t *afe_data = NULL;
static int16_t *afe_feed_buf = NULL;  // Make this global so we can free it
static bool use_afe = false;
static int afe_feed_chunksize = 0;
static int afe_feed_channels = 0;

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
  afe_cfg->afe_ringbuf_size = 250; // Ring buffer size
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

int estimate_delay(const int16_t* left,
                   const int16_t* right,
                   int samples) {
    int maxLag = 3;
    int bestLag = 0;
    int64_t bestCorr = 0;

    for (int lag = -maxLag; lag <= maxLag; lag++) {
        int64_t corr = 0;

        for (int i = maxLag; i < samples - maxLag; i++) {
            int j = i + lag;
            if (j < 0 || j >= samples) continue;
            corr += left[i] * right[j];
        }

        int64_t mag = llabs(corr);
        if (mag > bestCorr) {
            bestCorr = mag;
            bestLag = lag;
        }
    }
    return bestLag;
}

int estimate_delay_gcc_phat(int16_t *x, int16_t *y, int N)
{
    // Convert to float, real FFT input
    for (int i = 0; i < N; i++) {
        fft_l[2*i]   = (float)x[i];
        fft_l[2*i+1] = 0.0f;
        fft_r[2*i]   = (float)y[i];
        fft_r[2*i+1] = 0.0f;
    }

    dsps_fft2r_fc32(fft_l, N);
    dsps_fft2r_fc32(fft_r, N);
    dsps_bit_rev_fc32(fft_l, N);
    dsps_bit_rev_fc32(fft_r, N);

    // Cross power spectrum with PHAT
    for (int i = 0; i < N; i++) {
        float rl = fft_l[2*i];
        float il = fft_l[2*i+1];
        float rr = fft_r[2*i];
        float ir = fft_r[2*i+1];

        float real = rl*rr + il*ir;
        float imag = il*rr - rl*ir;

        float mag = sqrtf(real*real + imag*imag) + 1e-9f;

        cross_spec[2*i]   = real / mag;
        cross_spec[2*i+1] = imag / mag;
    }

    // IFFT
    dsps_fft2r_fc32(cross_spec, N);
    dsps_bit_rev_fc32(cross_spec, N);

    for (int i = 0; i < N; i++) {
        corr[i] = cross_spec[2*i];
    }

    // Find peak
    int max_idx = 0;
    float max_val = corr[0];
    for (int i = 1; i < N; i++) {
        if (corr[i] > max_val) {
            max_val = corr[i];
            max_idx = i;
        }
    }

    // Wrap negative delays
    int delay = max_idx;
    if (delay > N / 2) {
        delay -= N;
    }

    return delay;
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

  ESP_ERROR_CHECK(dsps_fft2r_init_fc32(NULL, FRAME_SAMPLES));
  
  ESP_LOGI(TAG, "========================================");
  ESP_LOGI(TAG, "AFE Status: %s", use_afe ? "ENABLED" : "DISABLED");
  ESP_LOGI(TAG, "Setup complete! Starting detection...");
  ESP_LOGI(TAG, "========================================");
}

void loop() {
  static int32_t i2s_buf[FRAME_SAMPLES * 2];  // Stereo buffer
  static int16_t audio_stereo[FRAME_SAMPLES * 2];
  static int loop_count = 0;
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

    // int samples_to_feed = min(afe_feed_chunksize, frames);
    // ESP_LOGI(TAG, "[FEED] Feeding %d samples to AFE", samples_to_feed);

    // Prepare feed buffer
    for (int i = 0; i < afe_feed_chunksize && i < frames; i++) {
      afe_feed_buf[i * 2 + 0] = audio_stereo[i * 2 + 0];
      afe_feed_buf[i * 2 + 1] = audio_stereo[i * 2 + 1];
    }

    // if (loop_count % 100 == 0) {
    //   ESP_LOGI(TAG, "Feeding AFE - samples[0]: L=%d R=%d, samples[10]: L=%d R=%d",
    //            afe_feed_buf[0], afe_feed_buf[1],
    //            afe_feed_buf[20], afe_feed_buf[21]);
    // }
    
    // Feed the data
    int feed_result = afe_handle->feed(afe_data, afe_feed_buf);
    
    // if (feed_result == 0) {
    //   ESP_LOGI(TAG, "[FEED] ✓ Feed successful (result=%d)", feed_result);
    // } else {
    //   ESP_LOGW(TAG, "[FEED] ✗ Feed failed or buffer full (result=%d)", feed_result);
    // }
    
    // Small delay to let AFE process
    vTaskDelay(pdMS_TO_TICKS(1));

    // ============================================
    // CRITICAL FIX: Fetch TWICE to drain buffer
    // ============================================
    
    // First fetch
    afe_fetch_result_t *afe_result = afe_handle->fetch(afe_data);
    
    // if (afe_result) {
    //   ESP_LOGI(TAG, "[FETCH #1] ret_value=%d, data_size=%d", 
    //            afe_result->ret_value, afe_result->data_size);
    // }
    
    if (afe_result && afe_result->ret_value == ESP_OK && afe_result->data) {
      processed_audio = afe_result->data;
      processed_frames = afe_result->data_size / sizeof(int16_t);
      vad_detected = (afe_result->vad_state == VAD_SPEECH);

      // ESP_LOGI(TAG, "[FETCH #1] ✓ Got %d frames, VAD=%s",
      //          processed_frames, vad_detected ? "SPEECH" : "SILENCE");
    }
    
    // Second fetch - drain remaining data
    afe_fetch_result_t *afe_result2 = afe_handle->fetch(afe_data);
    
    // if (afe_result2) {
    //   ESP_LOGI(TAG, "[FETCH #2] ret_value=%d, data_size=%d", 
    //            afe_result2->ret_value, afe_result2->data_size);
    // }
    
    if (afe_result2 && afe_result2->ret_value == ESP_OK && afe_result2->data) {
      // Use the second fetch result (most recent processed data)
      processed_audio = afe_result2->data;
      processed_frames = afe_result2->data_size / sizeof(int16_t);
      vad_detected = (afe_result2->vad_state == VAD_SPEECH);

      // ESP_LOGI(TAG, "[FETCH #2] ✓ Got %d frames, VAD=%s",
      //          processed_frames, vad_detected ? "SPEECH" : "SILENCE");
    }
    
    // ESP_LOGI(TAG, "[SUMMARY] Loop #%d: Fed=1024, Fetched (total from 2 fetches), VAD=%d",
    //          loop_count, vad_detected ? 1 : 0);
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

  static int16_t left[FRAME_SAMPLES];
  static int16_t right[FRAME_SAMPLES];

  for (int i = 0; i < frames; i++) {
      left[i]  = audio_stereo[i * 2];
      right[i] = audio_stereo[i * 2 + 1];
  }

  if (loud_enough){
    int delay_samples = estimate_delay_gcc_phat(left, right, frames);
    ESP_LOGI(TAG, "Estimated delay between mics: %d samples", delay_samples);

    float mic_distance = 0.023f; // meters (23 mm)
    float speed_of_sound = 343.0f;
    float sample_period = 1.0f / SAMPLE_RATE;

    float time_delay = delay_samples * sample_period;
    float ratio = (time_delay * speed_of_sound) / mic_distance;
    ratio = fminf(fmaxf(ratio, -1.0f), 1.0f);
    float angle_rad = asinf(ratio);

    float angle_deg = angle_rad * 180.0f / M_PI;

    ESP_LOGI(TAG, "Manual DOA: %.1f degrees", angle_deg);
  }

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