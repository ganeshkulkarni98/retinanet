import collections
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import model
from dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader
import coco_eval

def main():

    assert torch.__version__.split('.')[0] == '1'

    print('CUDA available: {}'.format(torch.cuda.is_available()))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=train_batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=num_workers, collate_fn=collater, batch_sampler=sampler)

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=val_batch_size, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=num_workers, collate_fn=collater, batch_sampler=sampler_val)

    # initialize the model
    retinanet, device = model_init(model_name)

    retinanet.to(device)
    retinanet = torch.nn.DataParallel(retinanet).to(device)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=verbose)

    loss_hist = collections.deque(maxlen=maxlen)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Number of training images: {} Number of validation images: {}'.format(len(dataset_train), len(dataset_val)))
    
    for epoch_num in range(epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []
        print('Training dataset')
        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        print('Evaluating dataset')

        coco_eval.evaluate_coco(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.state_dict(), f'CP_epoch{epoch_num + 1}.pth')
        

# def model_init(model_name, backbone):

#   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#   if model_name == 'retinanet' :
#     print("Loading model")
#     # Create the model
#     if backbone == 'resnet18':
#         retinanet = model.resnet18(num_classes=num_classes, pretrained=False)
#         weight_file_path = '/content/pytorch-retinanet/resnet18-5c106cde.pth'
#         retinanet.load_state_dict(torch.load(weight_file_path, map_location=device), strict=False) # Initialisng Model with loaded weights
#     elif backbone == 'resnet34':
#         retinanet = model.resnet34(num_classes=num_classes, pretrained=False)
#         weight_file_path = '/content/pytorch-retinanet/resnet34-333f7ec4.pth'
#         retinanet.load_state_dict(torch.load(weight_file_path, map_location=device), strict=False)
#     elif backbone == 'resnet50':
#         retinanet = model.resnet50(num_classes=num_classes, pretrained=False)
#         weight_file_path = '/content/pytorch-retinanet/resnet50-19c8e357.pth'
#         retinanet.load_state_dict(torch.load(weight_file_path, map_location=device), strict=False)        
#     elif backbone == 'resnet101':
#         retinanet = model.resnet101(num_classes=num_classes, pretrained=False)
#         weight_file_path = '/content/pytorch-retinanet/resnet101-5d3b4d8f.pth'
#         retinanet.load_state_dict(torch.load(weight_file_path, map_location=device), strict=False)        
#     elif backbone == 'resnet152':
#         retinanet = model.resnet152(num_classes=num_classes, pretrained=False)
#         weight_file_path = '/content/pytorch-retinanet/resnet152-b121ed2d.pth'
#         retinanet.load_state_dict(torch.load(weight_file_path, map_location=device), strict=False)        
#     else:
#         raise ValueError('Unsupported model backbone, must be one of resnet18, resnet34, resnet50, resnet101, resnet152')
  
#   print('model initialized..')

#   return retinanet, device

def model_init(model_name):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if model_name == 'retinanet' :
    weight_file_path = '/content/retinanet/resnet34-333f7ec4.pth'

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
    train_batch_size = 2
    val_batch_size = 1
    num_workers = 3
    lr=1e-5
    patience=3
    verbose=True
    maxlen=500

    model_name = 'retinanet'

    epochs = 5

    # Load train and validation dataset (for sake of example i have used same but use different dataset)
    # Load train image folder and corresponding coco json file to train dataset
    # Load validation image folder and corresponding json file to validation dataset

    dataset_train = CocoDataset('/content/data/images', '/content/data/output.json' ,
                            transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    dataset_val = CocoDataset('/content/data/images', '/content/data/output.json',
                            transform=transforms.Compose([Normalizer(), Resizer()])) 
 
    num_classes = dataset_train.num_classes()
                                                        
    # run main function
    main()
