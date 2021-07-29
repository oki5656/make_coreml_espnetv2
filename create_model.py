import torch
import torchvision
import cv2
import json
import coremltools as ct
from torchvision import transforms
from argparse import ArgumentParser
from PIL import Image

from model.segmentation.espnetv2 import espnetv2_seg
from commons.general_details import segmentation_models, segmentation_datasets
from data_loader.segmentation.mydataset import MY_DATASET_CLASS_LIST

seg_classes = len(MY_DATASET_CLASS_LIST)
parser = ArgumentParser()

# mdoel details
parser.add_argument('--classes', default=seg_classes, help='number of segmentation classes')
parser.add_argument('--model', default="espnetv2", choices=segmentation_models, help='Model name')
parser.add_argument('--weights-test', default='', help='Pretrained weights directory.')
parser.add_argument('--s', default=2.0, type=float, help='scale')
# dataset details
parser.add_argument('--data-path', default="vision_datasets/mydataset/", help='Data directory')
parser.add_argument('--dataset', default='mydataset', choices=segmentation_datasets, help='Dataset name')
# input details
parser.add_argument('--im-size', type=int, nargs="+", default=[640, 480], help='Image size for testing (W x H)')
parser.add_argument('--split', default='val', choices=['val', 'test'], help='data split')
parser.add_argument('--model-width', default=640, type=int, help='Model width')
parser.add_argument('--model-height', default=480, type=int, help='Model height')
parser.add_argument('--channels', default=3, type=int, help='Input channels')
parser.add_argument('--num-classes', default=1000, type=int,
                    help='ImageNet classes. Required for loading the base network')
args = parser.parse_args()
args.weights = ''



if not args.weights_test:
    ##############################　　　　load　weight file         ###############################
    from model.weight_locations.segmentation import model_weight_map   
    model_key = '{}_{}'.format(args.model, args.s)
    dataset_key = '{}_{}x{}'.format(args.dataset, args.im_size[0], args.im_size[1])
    #print("model_key,dataset_key:",model_key,"     ",dataset_key)
    assert model_key in model_weight_map.keys(), '{} does not exist'.format(model_key)
    assert dataset_key in model_weight_map[model_key].keys(), '{} does not exist'.format(dataset_key)
    args.weights_test = model_weight_map[model_key][dataset_key]['weights']


model = espnetv2_seg(args)
model.eval()
example = torch.rand(1, 3, 640, 480)
traced_script_module = torch.jit.trace(model, example)

mlmodel = ct.convert(
    traced_script_module,
    inputs=[ct.TensorType(name="input", shape=example.shape)],
)
mlmodel.save('./espnetv2_640_480.mlmodel')


# load the model
mlmodel = ct.models.MLModel("espnetv2_640_480.mlmodel")

labels_json = {"labels": ["obstacle", "background"]}

mlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageSegmenter"
mlmodel.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps(labels_json)

mlmodel.save("espnet_640_480_with_metadata.mlmodel")