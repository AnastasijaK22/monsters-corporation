import sys
import cv2
import argparse
import logging as log
import numpy as np
from openvino.inference_engine import IENetwork, IECore

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to an .xml file with a trained model.    ', required=False, type=str, dest='model_xml')
    parser.add_argument('-w', '--weights', help='Path to an .bin file with a trained weights.', required=False, type=str, dest='model_bin')
    parser.add_argument('-i', '--input', help='Data for input                                ', required=False, type=str, nargs='+', dest='input')
    parser.add_argument('-l', '--extension', help='Path to MKLDNN (CPU, MYRIAD) custom layers', type=str, default=None, dest='extension')    
    parser.add_argument('--default_device', help='Default device for heterogeneous inference', 
                        choices=['CPU', 'GPU', 'MYRIAD', 'FGPA'], default=None, type=str, dest='default_device')
    parser.add_argument('--labels', help='Labels mapping file', default=None, type=str, dest='labels')  
    return parser

def prepare_input():
    pass

def recognize_smile(image, exec_net, input_blob, out_blob):
    input = cv2.resize(image, (64, 64))
    input = input.transpose(2, 0, 1)
    result = image

    output = exec_net.infer(inputs={input_blob : input})
    if np.argmax(output[out_blob]) == 1:
        cv2.putText(result, "Smile!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    return image

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
        level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    cap = cv2.VideoCapture(0)

    ie = IECore()
    net = ie.read_network(model="./models/emotions-recognition-retail-0003/FP16-INT8/emotions-recognition-retail-0003.xml")
    exec_net = ie.load_network(network=net, device_name="CPU")
    out_blob = next(iter(net.outputs))
    input_blob = next(iter(net.inputs))

    while True:
        ret, frame = cap.read()
        frame = recognize_smile(frame, exec_net, input_blob, out_blob)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    sys.exit(main() or 0)