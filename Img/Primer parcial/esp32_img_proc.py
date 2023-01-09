import bluetooth as bt
import serial
import time

# Protocolo RFCOMM para asegurar integridad
# de datos enviados
#BTsocket = BluetoothSocket(RFCOMM)
#print(bt.discover_devices(duration=20 ,lookup_names=True))

esp = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=.1)
def write_read(x):
    esp.write(x)
    time.sleep(0.1)
    data = esp.readline()
    return data
while True:
    num = bytearray([2, 3, 1, 2, 4])#input("Enter a number: ") # Taking input from user
    value = write_read(num)
    print(value) # printing the value