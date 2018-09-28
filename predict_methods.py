import train_methods
import torch
import json
import numpy as np
from PIL import Image

def load_checkpoint(filepath, gpu):
    print("loading {}...".format(filepath))
    load_success = True
    model = None

    try:
        checkpoint = torch.load(filepath)
        model, _, _ = train_methods.NeuralNetwork(checkpoint['arch'],
                                                  checkpoint['hidden_units'],
                                                  checkpoint['learning_rate'],
                                                  gpu,
                                                  checkpoint['input_size'],
                                                  checkpoint['output_size'])
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
        print("{} loaded.".format(filepath))
    except:
        load_success = False
        print("{} cannot be loaded".format(filepath))
 
    return model, load_success


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    im.thumbnail((256, 256))
    im = im.resize((224, 224))
    np_image = np.array(im)
    np_image = (np_image - np_image.mean()) / np_image.std()
    np_image = np_image.transpose((2, 0, 1))
    return np_image


def process_classes(classes_int, inverted_class_to_idx):
    classes = []
    for class_num in classes_int:
        classes.append(inverted_class_to_idx[class_num])
    return classes


def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image_tensor = torch.from_numpy(image).float()
    
    if gpu == 'gpu' and torch.cuda.is_available():
        print("using gpu...")
        image_tensor = image_tensor.to('cuda', dtype=torch.float)
    else:
        print("using cpu...")

    image_tensor.unsqueeze_(0)
    with torch.no_grad():
        print("predicting...")
        outputs = model(image_tensor)
    probs_classes_topk = torch.exp(outputs).cpu().topk(topk)
    probs = np.squeeze(probs_classes_topk[0].data.numpy())
    inverted_class_to_idx = {model.class_to_idx[k]: k for k in model.class_to_idx}
    classes = process_classes(np.squeeze(probs_classes_topk[1].data.numpy()), inverted_class_to_idx)
    return probs, classes


def run(args):
    flower_names = []
    category_names = args.category_names
    checkpoint = args.checkpoint
    path_to_image_file = args.path_to_image_file
    top_k = int(args.top_k)
    gpu = args.gpu
    
    model, load_success = load_checkpoint(checkpoint, gpu)
    
    if load_success:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        
        probs, classes = predict(path_to_image_file, model, top_k, gpu)
        for class_num in classes:
            flower_names.append(cat_to_name[class_num])

        return probs, flower_names

    return [], []