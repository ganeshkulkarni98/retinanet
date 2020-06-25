import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import sys
import cv2
import torch
import model
import coco_eval
import json
import skimage.io
import skimage.transform
import skimage.color
import skimage

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main():

    images = list(sorted(os.listdir(images_folder)))

    results = []
    annotated_images = []
    
    # initialize the model
    retinanet, device = model_init(model_name)

    retinanet.to(device)
    retinanet = torch.nn.DataParallel(retinanet).to(device)

    retinanet.training = False
    retinanet.eval()

    def draw_caption(image, box, caption):

      b = np.array(box).astype(int)
      cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
      cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    idx = 0
    for original_image in images:

      with torch.no_grad():
        st = time.time()
        img_result = []
        
        original_image = skimage.io.imread(os.path.join(images_folder, original_image))

        original_image = original_image.astype(np.float32)/255.0

        # Transform and Resizer
        image = ResizerandNormalize(original_image)


        if torch.cuda.is_available():
          scores, classification, transformed_anchors = retinanet(image.unsqueeze(0).cuda().float())
        else:
          scores, classification, transformed_anchors = retinanet(image.unsqueeze(0).float())
        print('Elapsed time for {} image: {}'.format(idx, time.time()-st))

        idxs = np.where(scores.cpu()>0.5)

        # unnormalize image
        img = np.array(255 * unnormalize(image.unsqueeze(0)[0, :, :, :])).copy()

        img[img<0] = 0
        img[img>255] = 255

        img = np.transpose(img, (1, 2, 0))

        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

        for j in range(idxs[0].shape[0]):
          objects = {}
          bbox = transformed_anchors[idxs[0][j], :]
          x1 = int(bbox[0])
          y1 = int(bbox[1])
          x2 = int(bbox[2])
          y2 = int(bbox[3])

          label_name = rev_label_map[int(classification[idxs[0][j]])]
          draw_caption(img, (x1, y1, x2, y2), label_name)

          cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

          objects['xy_top_left'] = (bbox[0].item(), bbox[3].item())
          objects['xy_bot_right'] = (bbox[2].item(), bbox[1].item())
          objects['conf_level'] = scores[idxs[0][j]].item()
          objects['label']= label_name

          img_result.append(objects)

        # # save image and see results (create new folder named results and add images)
        # image_path = '/content/results/' + str(idx) + '.jpg'  
        # cv2.imwrite(image_path, img)
        results.append(img_result)
        annotated_images.append(np.array(img))
        idx += 1

    return annotated_images, results


def model_init(model_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == 'retinanet' :
      #weight_file_path = '/content/retinanet/resnet34-333f7ec4.pth'
      #weight_file_path = '/content/retinanet/CP_epoch5.pth'
      weight_file_path = '/content/retinanet/retinanet50_pretrained.pth'

    total_keys = len(list(torch.load(weight_file_path).keys()))

    # Create the model
    if total_keys >= 102 and total_keys < 182 :
        retinanet = model.resnet18(num_classes=num_classes, pretrained=False)

    elif total_keys >= 182 and total_keys < 267:
        retinanet = model.resnet34(num_classes=num_classes, pretrained=False)
        
    elif total_keys >= 267 and total_keys < 522:
        retinanet = model.resnet50(num_classes=num_classes, pretrained=False)
        
    elif total_keys >= 522 and total_keys < 777:
        retinanet = model.resnet101(num_classes=num_classes, pretrained=False)
        
    elif total_keys >= 777:
        retinanet = model.resnet152(num_classes=num_classes, pretrained=False)
        
    else:
        raise ValueError('Unsupported model backbone, must be one of resnet18, resnet34, resnet50, resnet101, resnet152')

    retinanet.load_state_dict(torch.load(weight_file_path, map_location=device), strict=False) # Initialisng Model with loaded weights
    print('model initialized..')

    return retinanet, device

def label_map_fn(json_file_path):
    anno = json.load(open(json_file_path))
    categories = anno["categories"]
    labels = []
    for i in range (0,len(categories)):
      labels.append(str(categories[i]["name"]))
    label_map = {k: v for v, k in enumerate(labels)}
    rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

    return rev_label_map

def ResizerandNormalize(image):
    """Convert ndarrays in sample to Tensors."""

    min_side=608
    max_side=1024

    if len(image.shape) == 2:
      image = skimage.color.gray2rgb(image)

    rows, cols, cns = image.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
    rows, cols, cns = image.shape

    pad_w = 32 - rows%32
    pad_h = 32 - cols%32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)

    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])

    new_image = (new_image.astype(np.float32)-mean)/std
    
    revised_image = np.transpose(new_image,(2,0,1))

    return torch.from_numpy(revised_image)

def unnormalize(tensor):
    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: Normalized image.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for t, m, s in zip(tensor, mean, std):
      t.mul_(s).add_(m)
    return tensor



if __name__ == '__main__':

    model_name = 'retinanet'

    images_folder = '/content/data/test_images'
    test_json_file = '/content/data/test_coco_dataset.json'

    # json file used for label mapping
    rev_label_map = label_map_fn(test_json_file)
    num_classes = len(rev_label_map)

    # Run test function
    annotated_images, results = main()
    print(annotated_images, results)
