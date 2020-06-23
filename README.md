### Google Colab Link for RetinaNet Implementation : https://colab.research.google.com/drive/12a9f0hKkI0mImPkVZFjEtTrIqWl_W3O2?usp=sharing

### RetinaNet model pretrained weight (Trained on COCO dataset) : https://drive.google.com/file/d/1-9UHb7cDcgiPdjrbhG93ooGHBvFYCu1F/view?usp=sharing

### ResNet (Backbone) pretrained weights (Trained on ImageNet Dataset)
- ResNet18 : https://download.pytorch.org/models/resnet18-5c106cde.pth
- ResNet34  : https://download.pytorch.org/models/resnet34-333f7ec4.pth
- ResNet50 : https://download.pytorch.org/models/resnet50-19c8e357.pth
- ResNet101 : https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
- ResNet152 : https://download.pytorch.org/models/resnet152-b121ed2d.pth



# pytorch-retinanet

![img3](https://github.com/yhenon/pytorch-retinanet/blob/master/images/3.jpg)
![img5](https://github.com/yhenon/pytorch-retinanet/blob/master/images/5.jpg)

Pytorch  implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

This implementation is primarily designed to be easy to read and simple to modify.

## Results
Currently, this repo achieves 33.5% mAP at 600px resolution with a Resnet-50 backbone. The published result is 34.0% mAP. The difference is likely due to the use of Adam optimizer instead of SGD with weight decay.

## Installation

1) Clone this repo

2) Install the required packages:

```
apt-get install tk-dev python-tk
```

3) Install the python packages:
	
```
pip install pandas
pip install opencv-python
pip install requests

cython==0.29.17
pillow==7.0.0
torch==1.5.0+cu101
torchvision==0.6.0+cu101
numpy==1.17.0

pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

## Training

The network can be trained using the `train.py` script. For training on coco, use

```
python train.py
```

## Pre-trained current model

A pre-trained model is available at: 
- https://drive.google.com/file/d/1-9UHb7cDcgiPdjrbhG93ooGHBvFYCu1F/view?usp=sharing

The state dict model can be loaded using:

```
retinanet = model.resnet50(num_classes=num_classes, pretrained=False)
retinanet.load_state_dict(torch.load(weight_file_path, map_location=device), strict=False) # Initialisng Model with loaded weights
```
## Pre-trained backbone model of Retinanet

A pre-trained model is available at: 
- https://download.pytorch.org/models/resnet18-5c106cde.pth
- https://download.pytorch.org/models/resnet34-333f7ec4.pth
- https://download.pytorch.org/models/resnet50-19c8e357.pth
- https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
- https://download.pytorch.org/models/resnet152-b121ed2d.pth

## Validation

Run `coco_validation.py` to validate the code on the COCO dataset. With the above model, run:

`python coco_validation.py

This produces the following results:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.335
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.499
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.357
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.167
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.369
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.466
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.282
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.429
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.458
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.255
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.508
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.597
```

## Visualization

To visualize the network detection, use `visualize.py`:

```
python visualize.py
```

## Model

The retinanet model uses a resnet backbone. You can set the depth of the resnet model using the --depth argument. Depth must be one of 18, 34, 50, 101 or 152. Note that deeper models are more accurate but are slower and use more memory.


### Annotations format

The COCO dataset format is annotation format 
For more details of COCO datset format, please visit http://cocodataset.org/#format-data

Indexing for classes starts at 0.
Do not include a background class as it is implicit.

For example:
```
cow:0
cat:1
bird:2
```

## Acknowledgements

- Code are borrowed from the https://github.com/yhenon/pytorch-retinanet 
- Significant amounts of code are borrowed from the [keras retinanet implementation](https://github.com/fizyr/keras-retinanet)
- The NMS module used is from the [pytorch faster-rcnn implementation](https://github.com/ruotianluo/pytorch-faster-rcnn)

