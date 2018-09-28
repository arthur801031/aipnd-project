import argparse
import predict_methods

parser = argparse.ArgumentParser(description='Use Image Classifier model to predict flower name')
parser.add_argument('path_to_image_file', nargs='?', default="flowers/test/1/image_06743.jpg")
parser.add_argument('checkpoint', nargs='?', default="checkpoint.pth", help="checkpoint file")
parser.add_argument('--top_k', dest="top_k", default="3", help="top k most likely classes")
parser.add_argument('--category_names', dest="category_names", default="cat_to_name.json", help="mapping of categories to real names")
parser.add_argument('--gpu', dest="gpu", default="gpu", help="gpu or cpu mode")

args = parser.parse_args()
probs, flower_names = predict_methods.run(args)
print("Top {} classes' probabilities: {}".format(args.top_k, probs))
print("Top {} classes' flower names: {}".format(args.top_k, flower_names))