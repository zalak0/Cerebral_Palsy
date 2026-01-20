#include <Arduino.h>
#include "driver/i2s_std.h"

static const int PIN_BCLK = 12;
static const int PIN_WS   = 13;
static const int PIN_DIN  = 11;

static const int PIN_POT = 7;      // ADC
static const int PIN_DETECT = 8;   // LED 1 / Audio detected output
static const int PIN_SCREAM = 9;   // LED 2 / Scream output
static const int PIN_CRY = 10;     // LED 3 / Cry output

static const int SAMPLE_RATE = 16000;
static const int FRAME_SAMPLES = 512;

// ZCR history for variation tracking
static const int ZCR_HISTORY_SIZE = 10;
static float zcr_history[ZCR_HISTORY_SIZE];
static int zcr_history_index = 0;
static bool zcr_history_filled = false;

static i2s_chan_handle_t rx_chan;

void setupI2S() {
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
}

int readPotAvg() {
  long sum = 0;
  for (int i = 0; i < 8; i++) {
    sum += analogRead(PIN_POT);
    delayMicroseconds(200);
  }
  return sum / 8;
}

float zcr_calculation(int32_t buf[], int frames) {
  int zeroCrossings = 0;
  
  for (int i = 1; i < frames; i++) {
    int32_t sL_prev = buf[(i-1) * 2 + 0] >> 11;
    int32_t sL_curr = buf[i * 2 + 0] >> 11;
    
    if ((sL_prev >= 0 && sL_curr < 0) || (sL_prev < 0 && sL_curr >= 0)) {
      zeroCrossings++;
    }
  }

  return (float)zeroCrossings / frames;
}

// Calculate ZCR variation over recent history
float calculate_zcr_variation() {
  if (!zcr_history_filled && zcr_history_index < 5) {
    return 0.0;  // Not enough data
  }
  
  int size = zcr_history_filled ? ZCR_HISTORY_SIZE : zcr_history_index;
  
  // Calculate mean
  float sum = 0.0;
  for (int i = 0; i < size; i++) {
    sum += zcr_history[i];
  }
  float mean = sum / size;
  
  // Calculate standard deviation
  float variance = 0.0;
  for (int i = 0; i < size; i++) {
    float diff = zcr_history[i] - mean;
    variance += diff * diff;
  }
  float std_dev = sqrt(variance / size);
  
  return std_dev;
}

int find_pitch_period(int32_t buf[], int frames) {
  // We're looking for pitch in range ~80Hz to 800Hz
  // At 16kHz sample rate:
  // 80 Hz = 200 samples period
  // 800 Hz = 20 samples period
  
  int min_period = 5;   // ~3200 Hz (high pitch limit)
  int max_period = 200;  // ~80 Hz (low pitch limit)
  
  // Make sure we don't exceed buffer
  if (max_period >= frames) {
    max_period = frames - 1;
  }
  
  float max_correlation = 0.0;
  int best_period = min_period;
  
  // Try each possible period
  for (int period = min_period; period <= max_period; period++) {
    float correlation = 0.0;
    int count = 0;
    
    // Calculate autocorrelation at this period
    for (int i = 0; i < frames - period; i++) {
      int32_t sample1 = buf[i * 2 + 0] >> 11;  // Left channel
      int32_t sample2 = buf[(i + period) * 2 + 0] >> 11;
      
      correlation += (float)sample1 * (float)sample2;
      count++;
    }
    
    // Normalize by number of samples
    if (count > 0) {
      correlation /= count;
    }
    
    // Track the period with maximum correlation
    if (correlation > max_correlation) {
      max_correlation = correlation;
      best_period = period;
    }
  }
  
  return best_period;
}


void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\nI2S Audio: RMS + ZCR + Fluctuation Detection");
  setupI2S();

  pinMode(PIN_DETECT, OUTPUT);
  pinMode(PIN_SCREAM, OUTPUT);
  pinMode(PIN_CRY, OUTPUT);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  
  // Initialize history
  for (int i = 0; i < ZCR_HISTORY_SIZE; i++) {
    zcr_history[i] = 0.0;
  }
}

void loop() {
  static int32_t buf[FRAME_SAMPLES * 2];
  size_t bytes_read = 0;

  esp_err_t err = i2s_channel_read(rx_chan, buf, sizeof(buf), &bytes_read, pdMS_TO_TICKS(200));
  if (err != ESP_OK || bytes_read == 0) {
    Serial.println("I2S read timeout / error");
    return;
  }

  int frames = bytes_read / (sizeof(int32_t) * 2);
  double sumL = 0.0, sumR = 0.0;

  for (int i = 0; i < frames; i++) {
    int32_t rawL = buf[i * 2 + 0];
    int32_t rawR = buf[i * 2 + 1];
    int32_t sL = rawL >> 11;
    int32_t sR = rawR >> 11;

    sumL += (double)sL * (double)sL;
    sumR += (double)sR * (double)sR;
  }

  double Ef = sumL / frames;
  double Er = sumR / frames;
  double rmsL = sqrt(Ef);
  double rmsR = sqrt(Er);
  double rmsL_db = 20.0 * log10(rmsL + 1e-9);
  double rmsR_db = 20.0 * log10(rmsR + 1e-9);

  // Store RMS in history for fluctuation analysis
  zcr_history[zcr_history_index] = rmsL;
  zcr_history_index++;
  if (zcr_history_index >= ZCR_HISTORY_SIZE) {
    zcr_history_index = 0;
    zcr_history_filled = true;
  }

  // Calculate features
    // Calculate ZCR and store in history
  float zcr = zcr_calculation(buf, frames);
  zcr_history[zcr_history_index] = zcr;
  zcr_history_index++;
  if (zcr_history_index >= ZCR_HISTORY_SIZE) {
    zcr_history_index = 0;
    zcr_history_filled = true;
  }

  // Calculate ZCR variation
  float zcr_variation = calculate_zcr_variation();
  float pitch_period = find_pitch_period(buf, frames);
  float pitch_freq = (float)SAMPLE_RATE / (float)pitch_period;

  // Thresholds (TUNE THESE!)
  float ZCR_CRY_MIN = 0.10;
  float ZCR_CRY_MAX = 0.40;
  float ZCR_VARIATION_MIN = 0.03;  // Crying ZCR jumps around
  float PITCH_CRY_MIN = 300;  // Crying has high variation
  float PITCH_CRY_MAX = 700;
  
  int pot = readPotAvg();
  double voice_db = map(pot, 0, 4095, 30, 90);
  double scream_db = voice_db + 20.0;

  // Detection logic with multiple features
  bool loud_enough = rmsL_db > voice_db;
  bool screaming = rmsL_db > scream_db;
  bool cry_zcr = (zcr > ZCR_CRY_MIN && zcr < ZCR_CRY_MAX);
  bool zcr_varying = (zcr_variation > ZCR_VARIATION_MIN);
  bool cry_pitch = (pitch_freq > PITCH_CRY_MIN && pitch_freq < PITCH_CRY_MAX);

  // Crying = loud + right ZCR + high fluctuation + rhythmic
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

  // Serial Plotter output
  Serial.print(rmsL_db);
  Serial.print(" ");
  Serial.print(voice_db);
  Serial.print(" ");
  Serial.print(scream_db);
  Serial.print(" ");
  Serial.print(zcr * 100);  // Scale ZCR
  Serial.print(" ");
  Serial.println(pitch_freq);  // Show actual pitch frequency
  
  // Debug output (uncomment to see details)
  // Serial.print("  Pitch=");
  // Serial.print(pitch_freq, 1);
  // Serial.print(" Hz  ZCR_var=");
  // Serial.print(zcr_variation, 3);
  // Serial.print("  Cry=");
  // Serial.println(crying_detected ? "YES" : "NO");
  
  delay(50);
}