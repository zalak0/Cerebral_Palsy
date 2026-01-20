#include "driver/i2s_std.h"
#include <Arduino.h>
#include <cmath>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

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
  ESP_LOGI(TAG, "========================================");
  ESP_LOGI(TAG, "=== Baby Cry Detector Starting... ===");
  ESP_LOGI(TAG, "========================================");
  
  Serial.begin(115200);
  delay(500);
  
  ESP_LOGI(TAG, "Serial initialized at 115200 baud");
  
  setupI2S();
  ESP_LOGI(TAG, "I2S initialized");

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
  ESP_LOGI(TAG, "Setup complete! Starting detection...");
  ESP_LOGI(TAG, "========================================");
  
  // Flash LEDs to show we're ready
  digitalWrite(PIN_DETECT, HIGH);
  delay(200);
  digitalWrite(PIN_DETECT, LOW);
  digitalWrite(PIN_CRY, HIGH);
  delay(200);
  digitalWrite(PIN_CRY, LOW);
  digitalWrite(PIN_SCREAM, HIGH);
  delay(200);
  digitalWrite(PIN_SCREAM, LOW);
  
  ESP_LOGI(TAG, "Ready! Starting main loop...");
}

void loop() {
  static int32_t i2s_buf[FRAME_SAMPLES * 2];
  static int16_t audio_buf[FRAME_SAMPLES];
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
    audio_buf[i] = (int16_t)sampleL;
  }

  // Calculate RMS
  double sum = 0.0;
  for (int i = 0; i < frames; i++) {
    sum += (double)audio_buf[i] * (double)audio_buf[i];
  }
  double rms = std::sqrt(sum / frames);
  double rms_db = 20.0 * std::log10(rms + 1e-9);

  // Calculate features
  float zcr = zcr_calculation(audio_buf, frames);
  zcr_history[zcr_history_index] = zcr;
  zcr_history_index++;
  if (zcr_history_index >= ZCR_HISTORY_SIZE) {
    zcr_history_index = 0;
    zcr_history_filled = true;
  }

  float zcr_variation = calculate_zcr_variation();
  float pitch_period = find_pitch_period(audio_buf, frames);
  float pitch_freq = (float)SAMPLE_RATE / (float)pitch_period;

  // Thresholds
  float ZCR_CRY_MIN = 0.10;
  float ZCR_CRY_MAX = 0.40;
  float PITCH_CRY_MIN = 300;
  float PITCH_CRY_MAX = 700;
  
  int pot = readPotAvg();
  double voice_db = map(pot, 0, 4095, 30, 90);
  double scream_db = voice_db + 20.0;

  // Detection logic
  bool loud_enough = rms_db > voice_db;
  bool screaming = rms_db > scream_db;
  bool cry_zcr = (zcr > ZCR_CRY_MIN && zcr < ZCR_CRY_MAX);
  bool cry_pitch = (pitch_freq > PITCH_CRY_MIN && pitch_freq < PITCH_CRY_MAX);
  bool crying_detected = loud_enough && cry_zcr && cry_pitch;

  // Output control
  if (screaming) {
    digitalWrite(PIN_SCREAM, HIGH);
    digitalWrite(PIN_CRY, LOW);
    digitalWrite(PIN_DETECT, LOW);
  } else if (crying_detected) {
    digitalWrite(PIN_SCREAM, LOW);
    digitalWrite(PIN_CRY, HIGH);
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

  // Serial output for plotting
  printf("RMS:%.1f,Threshold:%.1f,Scream:%.1f,ZCR:%.1f,Pitch:%.1f,Cry:%d\n",
         rms_db, voice_db, scream_db, zcr * 100, pitch_freq, crying_detected ? 80 : 0);

  // Debug output every 20 loops (once per second at 50ms delay)
  loop_count++;
  if (loop_count % 20 == 0) {
    ESP_LOGI(TAG, "Stats - RMS: %.1f dB, ZCR: %.2f, Pitch: %.0f Hz, Cry: %s",
             rms_db, zcr, pitch_freq, crying_detected ? "YES" : "NO");
  }
  
  // Suppress warnings
  (void)zcr_variation;
  
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