/*
  openDOME ESP32-S3 firmware: 4-mic audio stream + servo commands

  Wiring (from the team's Cirkit diagram):
    GPIO 4  -> SCK  of all 4 mics   (clock, shared)
    GPIO 5  -> WS   of all 4 mics   (word select, shared)
    GPIO 6  -> SD   of mics 1 and 2 (data line 1)
    GPIO 7  -> SD   of mics 3 and 4 (data line 2)
    GPIO 1  -> pan servo signal
    GPIO 2  -> tilt servo signal
    Each data line carries two mics: the one with L/R -> GND talks in the
    "left" half of each frame, the one with L/R -> 3V3 in the "right" half.

  How the four mics stay in sync:
    The ESP32-S3 has two I2S (audio) units. I2S0 is the master: it drives the
    clock on GPIO 4/5 and reads GPIO 6. I2S1 is a listener: it reads the SAME
    GPIO 4/5 clock (connected inside the chip, no jumper wires) and reads
    GPIO 7. All four mics therefore sample on the same clock edge.

  What goes to the laptop (binary packets over the USB port):
    A5 5A | seq u16 | rate u16 | frames u16 | frames x 4 x int16 | crc16
    Samples are interleaved per frame: [line1 slot0, line1 slot1,
    line2 slot0, line2 slot1]. The laptop maps slots to mic numbers.

  What comes from the laptop (text lines):
    P,<pan>          pan angle 0..180 (90 = straight ahead)
    P,<pan>,<tilt>   pan and tilt
*/
#include <Arduino.h>
#include <ESP32Servo.h>
#include <driver/i2s.h>
#include <driver/gpio.h>
#include <soc/i2s_periph.h>
#include <esp_rom_gpio.h>

// ---------- settings ----------
const int SAMPLE_RATE = 32000;   // Hz. 48000 per brief if the USB link keeps up (check "gaps" on laptop)
const int FRAMES = 256;          // frames per packet (1 frame = 1 sample from each of the 4 mics)
const int PIN_SCK = 4, PIN_WS = 5, PIN_SD1 = 6, PIN_SD2 = 7;
const int PIN_PAN = 2, PIN_TILT = 1;

// ---------- packet ----------
const int HEADER = 8;                         // marker(2) seq(2) rate(2) frames(2)
const int PAYLOAD = FRAMES * 4 * 2;           // int16 samples
const int PACKET = HEADER + PAYLOAD + 2;      // + crc16
const int QUEUE_LEN = 8;

static uint8_t packets[QUEUE_LEN][PACKET];
static QueueHandle_t readyQ, freeQ;           // indexes of full / empty packet buffers
static int32_t raw1[FRAMES * 2], raw2[FRAMES * 2];

Servo panServo, tiltServo;
String line = "";

uint16_t crc16(const uint8_t *d, int n) {     // CRC-16/CCITT, init 0xFFFF
  uint16_t c = 0xFFFF;
  while (n--) {
    c ^= (uint16_t)(*d++) << 8;
    for (int i = 0; i < 8; i++) c = (c & 0x8000) ? (c << 1) ^ 0x1021 : (c << 1);
  }
  return c;
}

void setupI2S() {
  i2s_config_t cfg = {};
  cfg.sample_rate = SAMPLE_RATE;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;   // INMP441 sends 24-bit in 32-bit slots
  cfg.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;   // both slots (two mics per data line)
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = FRAMES;

  // I2S1: listener, reads the shared clock and data line 2
  cfg.mode = (i2s_mode_t)(I2S_MODE_SLAVE | I2S_MODE_RX);
  i2s_driver_install(I2S_NUM_1, &cfg, 0, NULL);
  i2s_pin_config_t p1 = {};
  p1.mck_io_num = I2S_PIN_NO_CHANGE;
  p1.bck_io_num = PIN_SCK; p1.ws_io_num = PIN_WS;
  p1.data_out_num = I2S_PIN_NO_CHANGE; p1.data_in_num = PIN_SD2;
  i2s_set_pin(I2S_NUM_1, &p1);

  // I2S0: master, drives the clock and reads data line 1
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
  i2s_pin_config_t p0 = p1;
  p0.data_in_num = PIN_SD1;
  i2s_set_pin(I2S_NUM_0, &p0);

  // Share SCK/WS inside the chip: the pins output I2S0's clock AND feed it
  // into I2S1 (setting up I2S0 switched the pins to output-only).
  gpio_set_direction((gpio_num_t)PIN_SCK, GPIO_MODE_INPUT_OUTPUT);
  gpio_set_direction((gpio_num_t)PIN_WS, GPIO_MODE_INPUT_OUTPUT);
  esp_rom_gpio_connect_out_signal(PIN_SCK, i2s_periph_signal[0].m_rx_bck_sig, false, false);
  esp_rom_gpio_connect_out_signal(PIN_WS, i2s_periph_signal[0].m_rx_ws_sig, false, false);
  esp_rom_gpio_connect_in_signal(PIN_SCK, i2s_periph_signal[1].s_rx_bck_sig, false);
  esp_rom_gpio_connect_in_signal(PIN_WS, i2s_periph_signal[1].s_rx_ws_sig, false);

  // Restart listener first, then master, so both begin on the same frame.
  i2s_stop(I2S_NUM_0); i2s_stop(I2S_NUM_1);
  i2s_zero_dma_buffer(I2S_NUM_0); i2s_zero_dma_buffer(I2S_NUM_1);
  i2s_start(I2S_NUM_1); i2s_start(I2S_NUM_0);
}

static inline int16_t to16(int32_t s) { return (int16_t)(s >> 16); }  // keep top 16 of 24 bits

// Core 0: read both lines, build packets
void audioTask(void *) {
  uint16_t seq = 0;
  for (;;) {
    size_t n1 = 0, n2 = 0;
    i2s_read(I2S_NUM_0, raw1, sizeof(raw1), &n1, portMAX_DELAY);
    i2s_read(I2S_NUM_1, raw2, sizeof(raw2), &n2, portMAX_DELAY);

    int idx;
    if (xQueueReceive(freeQ, &idx, 0) != pdTRUE) { seq++; continue; }  // laptop too slow: drop, seq shows the gap
    uint8_t *p = packets[idx];
    p[0] = 0xA5; p[1] = 0x5A;
    memcpy(p + 2, &seq, 2);
    uint16_t rate = SAMPLE_RATE; memcpy(p + 4, &rate, 2);
    uint16_t fr = FRAMES;        memcpy(p + 6, &fr, 2);
    int16_t *s = (int16_t *)(p + HEADER);
    for (int i = 0; i < FRAMES; i++) {
      s[4 * i + 0] = to16(raw1[2 * i]);
      s[4 * i + 1] = to16(raw1[2 * i + 1]);
      s[4 * i + 2] = to16(raw2[2 * i]);
      s[4 * i + 3] = to16(raw2[2 * i + 1]);
    }
    uint16_t c = crc16(p + 2, HEADER - 2 + PAYLOAD);
    memcpy(p + HEADER + PAYLOAD, &c, 2);
    xQueueSend(readyQ, &idx, portMAX_DELAY);
    seq++;
  }
}

void handleLine(const String &l) {
  if (!l.startsWith("P,")) return;
  int comma = l.indexOf(',', 2);
  float pan = l.substring(2, comma < 0 ? l.length() : comma).toFloat();
  panServo.write((int)constrain(pan, 0.0f, 180.0f));
  if (comma > 0) tiltServo.write((int)constrain(l.substring(comma + 1).toFloat(), 0.0f, 180.0f));
}

void setup() {
  Serial.setTxBufferSize(8192);
  Serial.setTxTimeoutMs(20);         // don't stall long if the laptop isn't reading
  Serial.begin(115200);

  panServo.attach(PIN_PAN, 500, 2500);
  tiltServo.attach(PIN_TILT, 500, 2500);
  panServo.write(90);
  tiltServo.write(90);

  readyQ = xQueueCreate(QUEUE_LEN, sizeof(int));
  freeQ = xQueueCreate(QUEUE_LEN, sizeof(int));
  for (int i = 0; i < QUEUE_LEN; i++) xQueueSend(freeQ, &i, 0);

  setupI2S();
  xTaskCreatePinnedToCore(audioTask, "audio", 8192, NULL, 5, NULL, 0);
}

// Core 1: send packets, read servo commands
void loop() {
  int idx;
  if (xQueueReceive(readyQ, &idx, pdMS_TO_TICKS(5)) == pdTRUE) {
    Serial.write(packets[idx], PACKET);
    xQueueSend(freeQ, &idx, 0);
  }
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n') { handleLine(line); line = ""; }
    else if (ch != '\r' && line.length() < 32) line += ch;
  }
}