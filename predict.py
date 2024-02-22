import argparse
import torch
from torchvision import transforms, models
import json
from PIL import Image
import re
import glob

def args_input():
    parser = argparse.ArgumentParser(description='Flower class prediction')

    parser.add_argument('--image_path', type=str, default='flowers/test/92/image_03052.jpg')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='filename for your checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='making list of top predictions, default is 5 lists')
    parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='filename cointaining flower\'s name mapping')
    parser.add_argument('--gpu', action='store_true', default=False)

    args = parser.parse_args()
    
    if not re.search("checkpoint", args.checkpoint):
        parser.error('ERROR: Checkpoint file must have "checkpoint" in the name')
        
    return args

def process_image(image_path):
    pil_image = Image.open(image_path)
    
    preprocess = transforms.Compose([transforms.Resize(255), 
                                     transforms.CenterCrop(224), 
                                     transforms.ToTensor(), 
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    tensor = preprocess(pil_image)
    
    return tensor

def predict(image_path, checkpoint, topk=5):

    with open('cat_to_name.json', 'r') as f:
       cat_to_name = json.load(f)
    
    checkpoint = None
    checkpoints = glob.glob('*.pth')
    for check in checkpoints:
        if 'checkpoint' in check:
            checkpoint = check 
            break

    checkp = torch.load(checkpoint)
    arch = checkp['arch']
    classifier =checkp['classifier']
    class_to_idx = checkp['class_to_idx']

    model = getattr(models, arch)(pretrained=True)
    model.classifier = classifier
    model.class_to_idx = class_to_idx

    # Move model to device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Process image
    image = process_image(image_path)  
    image = image.unsqueeze(0) 
    image = image.to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        
    # Get top k probabilities and class indices 
    top_probs, top_inds = probs.topk(topk)
    
    # Detach tensors from graph and move to CPU
    top_probs = top_probs.detach().cpu().numpy() 
    top_inds = top_inds.detach().cpu().numpy()
    
    # Convert indices to classes
    idx_to_class = {idx: class_ for class_, idx in model.class_to_idx.items()}
    top_classes = [idx_to_class[ind] for ind in top_inds]
    
    return top_probs, top_classes

def main():
    args = args_input()
    top_probs, top_classes = predict(args.image_path, args.checkpoint, args.top_k)

    with open(args.cat_to_name, 'r') as f:
        cat_to_name = json.load(f, strict=False)

    cat_names = [cat_to_name[cl] for cl in top_classes]
    true_flower = args.image_path.split('/')[2]
    name_flower = cat_to_name[true_flower]
    
    print(f'True class: {name_flower}')
    print('=============================')
    print('The Prediction:\n')
    for prob, class_, category_name in zip(top_probs, top_classes, cat_names):
        print(f"Class: {class_} | Category: {category_name} | Probability: {prob:.2f}")

if __name__ == "__main__":
    main()