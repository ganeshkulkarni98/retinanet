import torch
from torchvision import transforms

import model
from dataloader import CocoDataset, Resizer, Normalizer
import coco_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main():

    # initialize the model
    retinanet, device = model_init(model_name)

    retinanet.to(device)
    retinanet = torch.nn.DataParallel(retinanet).to(device)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    coco_eval.evaluate_coco(dataset_val, retinanet)


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

if __name__ == '__main__':

    # Hyperparameters
    val_batch_size = 8
    num_workers = 3

    model_name = 'retinanet'

    images_folder = '/content/data/val2017'
    test_json_file = '/content/data/test_coco_dataset.json'

    # Load test image folder with corresponding coco json file to test_dataset
    dataset_val = CocoDataset(images_folder, test_json_file,
                            transform=transforms.Compose([Normalizer(), Resizer()]))

    num_classes = dataset_val.num_classes()

    # Run test function
    main()
