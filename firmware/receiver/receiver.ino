#include <WiFi.h>
#include <esp_wifi.h>

#define CSI_CHANNEL 1

void _csi_cb(void *ctx, wifi_csi_info_t *data) {
  if (!data || !data->buf) {
    return;
  }

  Serial.printf("CSI_DATA,%d,%d,%d,%d,", 
                data->rx_ctrl.rssi, 
                data->rx_ctrl.noise_floor, 
                data->rx_ctrl.rate, 
                data->len);

  int8_t *csi_raw = (int8_t *)data->buf;
  for (int i = 0; i < data->len; i++) {
    Serial.printf("%d%c", csi_raw[i], (i == data->len - 1) ? '\n' : ',');
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(CSI_CHANNEL, WIFI_SECOND_CHAN_NONE);

  wifi_csi_config_t csi_config = {
    .lltf_en = true,
    .htltf_en = true,
    .stbc_htltf2_en = true,
    .ltf_merge_en = true,
    .channel_filter_en = false,
    .manu_scale = false,
    .shift = false,
  };

  ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
  ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(_csi_cb, NULL));
  ESP_ERROR_CHECK(esp_wifi_set_csi(true));

  Serial.println("Receiver started on Channel 1.");
}

void loop() {
  delay(1000);
}
