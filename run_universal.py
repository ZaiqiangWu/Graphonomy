from networks import deeplab_xception_universal, graph
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from dataloaders import custom_transforms as tr

label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]

def main():
    net = deeplab_xception_universal.deeplab_xception_end2end_3d(n_classes=20, os=16,
                                                                  hidden_layers=128,
                                                                  source_classes=7,
                                                                  middle_classes=18, )
    ckpt = './universal_trained.pth'
    x = torch.load(ckpt)
    net.load_state_dict_new(x)
    print('load pretrainedModel.')
    net.eval()
    net=net.cuda()
    # Get graphs
    train_graph, test_graph = get_graphs()
    adj1, adj2, adj3, adj4, adj5, adj6 = train_graph
    adj1_test, adj2_test, adj3_test, adj4_test, adj5_test, adj6_test = test_graph

    composed_transforms = transforms.Compose([
        tr.Normalize_xception_tf_only_img(),
        tr.ToTensor_only_img()])

    dataset_lbl=0
    img_path='./ichao_input.jpg'
    img = read_img(img_path)
    inputs = img_transform(img, composed_transforms)['image']
    inputs = inputs.unsqueeze(0)
    print(inputs.shape)


    if dataset_lbl == 0:
        # 0 is cihp -- target
        with torch.no_grad():
            _, outputs, _ = net.forward(None, input_target=inputs, input_middle=None, adj1_target=adj1, adj2_source=adj2,
                                    adj3_transfer_s2t=adj3, adj3_transfer_t2s=adj3.transpose(2, 3), adj4_middle=adj4,
                                    adj5_transfer_s2m=adj5.transpose(2, 3),
                                    adj6_transfer_t2m=adj6.transpose(2, 3), adj5_transfer_m2s=adj5,
                                    adj6_transfer_m2t=adj6, )
    elif dataset_lbl == 1:
        # pascal is source
        outputs, _, _ = net.forward(inputs, input_target=None, input_middle=None, adj1_target=adj1,
                                    adj2_source=adj2,
                                    adj3_transfer_s2t=adj3, adj3_transfer_t2s=adj3.transpose(2, 3),
                                    adj4_middle=adj4, adj5_transfer_s2m=adj5.transpose(2, 3),
                                    adj6_transfer_t2m=adj6.transpose(2, 3), adj5_transfer_m2s=adj5,
                                    adj6_transfer_m2t=adj6, )
    else:
        # atr
        _, _, outputs = net.forward(None, input_target=None, input_middle=inputs, adj1_target=adj1,
                                    adj2_source=adj2,
                                    adj3_transfer_s2t=adj3, adj3_transfer_t2s=adj3.transpose(2, 3),
                                    adj4_middle=adj4, adj5_transfer_s2m=adj5.transpose(2, 3),
                                    adj6_transfer_t2m=adj6.transpose(2, 3), adj5_transfer_m2s=adj5,
                                    adj6_transfer_m2t=adj6, )




def get_graphs():
    '''source is pascal; target is cihp; middle is atr'''
    gpus=1
    # target 1
    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj1_ = torch.from_numpy(cihp_adj).float()
    adj1 = adj1_.unsqueeze(0).unsqueeze(0).expand(gpus, 1, 20, 20).cuda()
    adj1_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20)
    #source 2
    adj2_ = torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float()
    adj2 = adj2_.unsqueeze(0).unsqueeze(0).expand(gpus, 1, 7, 7).cuda()
    adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7)
    # s to target 3
    adj3_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj3 = adj3_.unsqueeze(0).unsqueeze(0).expand(gpus, 1, 7, 20).transpose(2,3).cuda()
    adj3_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).transpose(2,3)
    # middle 4
    atr_adj = graph.preprocess_adj(graph.atr_graph)
    adj4_ = torch.from_numpy(atr_adj).float()
    adj4 = adj4_.unsqueeze(0).unsqueeze(0).expand(gpus, 1, 18, 18).cuda()
    adj4_test = adj4_.unsqueeze(0).unsqueeze(0).expand(1, 1, 18, 18)
    # source to middle 5
    adj5_ = torch.from_numpy(graph.pascal2atr_nlp_adj).float()
    adj5 = adj5_.unsqueeze(0).unsqueeze(0).expand(gpus, 1, 7, 18).cuda()
    adj5_test = adj5_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 18)
    # target to middle 6
    adj6_ = torch.from_numpy(graph.cihp2atr_nlp_adj).float()
    adj6 = adj6_.unsqueeze(0).unsqueeze(0).expand(gpus, 1, 20, 18).cuda()
    adj6_test = adj6_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 18)
    train_graph = [adj1, adj2, adj3, adj4, adj5, adj6]
    test_graph = [adj1_test, adj2_test, adj3_test, adj4_test, adj5_test, adj6_test]
    return train_graph, test_graph

def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs

def read_img(img_path):
    _img = Image.open(img_path).convert('RGB')  # return is RGB pic
    return _img

def img_transform(img, transform=None):
    sample = {'image': img, 'label': 0}

    sample = transform(sample)
    return sample

if __name__ == '__main__':
    main()
