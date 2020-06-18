import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time

import sys
import cv2

from dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import model
import coco_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main():

    results = []
    annotated_images = []
    # Dataloader
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=batch_size, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=num_workers, collate_fn=collater, batch_sampler=sampler_val)
    
    # initialize the model
    retinanet, device = model_init(model_name)

    retinanet.to(device)
    retinanet = torch.nn.DataParallel(retinanet).to(device)

    retinanet.training = False
    retinanet.eval()
    #retinanet.module.freeze_bn()

    #coco_eval.evaluate_coco(dataset_val, retinanet)
    unnormalize = UnNormalizer()

    def draw_caption(image, box, caption):

      b = np.array(box).astype(int)
      cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
      cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    for idx, data in enumerate(dataloader_val):

      with torch.no_grad():
        st = time.time()
        img_result = []
        if torch.cuda.is_available():
          scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
        else:
          scores, classification, transformed_anchors = retinanet(data['img'].float())
        print('Elapsed time: {}'.format(time.time()-st))

        idxs = np.where(scores.cpu()>0.5)
        img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

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
          label_name = dataset_val.labels[int(classification[idxs[0][j]])]
          draw_caption(img, (x1, y1, x2, y2), label_name)

          cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
          #print(label_name)
          objects['x_min'] = bbox[0]
          objects['y_min'] = bbox[1]
          objects['x_max'] = bbox[2]
          objects['y_max'] = bbox[3]
          objects['conf_level'] = scores[idxs[0][j]]
          objects['label']= label_name

          img_result.append(objects)

        results.append(img_result)
        annotated_images.append(np.array(img))

    return annotated_images, results


def model_init(model_name):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if model_name == 'retinanet' :
    weight_file_path = '/content/retinanet/resnet34-333f7ec4.pth'
    #weight_file_path = '/content/retinanet/CP_epoch5.pth'

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

if __name__ == '__main__':

    # Hyperparameters
    batch_size = 1
    num_workers = 1

    model_name = 'retinanet'

    # Load test image folder with corresponding coco json file to test_dataset
    dataset_val = CocoDataset('/content/data/images', '/content/data/output.json',
                            transform=transforms.Compose([Normalizer(), Resizer()]))

    num_classes = dataset_val.num_classes()

    # Run test function
    annotated_images, results = main()
    print(annotated_images, results)
