import argparse
import train_methods

parser = argparse.ArgumentParser(description='Train Image Classifier')
parser.add_argument('data_dir', nargs='?', default="./flowers/")
parser.add_argument('--gpu', dest="gpu", default="gpu", help="gpu or cpu mode")
parser.add_argument('--save_dir', dest="save_dir", default="checkpoint.pth", help="checkpoint file to store model's weights and other info")
parser.add_argument('--arch', dest="arch", default="vgg16", help="checkpoint file to store model's weights and other info")
parser.add_argument('--learning_rate', dest="learning_rate", default="0.001")
parser.add_argument('--hidden_units ', dest="hidden_units", default="4480")
parser.add_argument('--epochs', dest="epochs", default="6")

args = parser.parse_args()
train_methods.train(args)