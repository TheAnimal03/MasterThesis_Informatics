from tool.MobileNetV2 import MobileNetV2

if __name__ == '__main__':
    net = MobileNetV2.mobilenet_v2(True)
    print(net)