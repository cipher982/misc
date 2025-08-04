import os
import platform
import socket
import time

from Adafruit_DHT import common, read_retry

DHT_SENSOR = common.DHT22
DHT_PIN = 4

CARBON_SERVER = "192.168.1.215"
CARBON_PORT = 2003

node = platform.node().replace(".", "-")


if __name__ == "__main__":
    while True:
        # Read sensor data
        humidity, temperature = read_retry(DHT_SENSOR, DHT_PIN)
        if humidity is not None and temperature is not None:
            print(f"Temp={temperature:.1f} Humidity={humidity:.1f}")

            # Craft message
            timestamp = int(time.time())
            lines = [
                f"{node}.ambient_humidity {humidity} {timestamp}",
                f"{node}.ambient_temperature {temperature} {timestamp}",
            ]
            message = "\n".join(lines) + "\n"

            # Send to Graphite
            sock = socket.socket()
            sock.connect((CARBON_SERVER, CARBON_PORT))
            sock.sendall(str.encode(message))
            sock.close()

        else:
            print("Failed to retrieve data from humidity sensor")

        time.sleep(2)
