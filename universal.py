from networks import deeplab_xception_universal, graph
import torch
import numpy as np

def main():
    net = deeplab_xception_universal.deeplab_xception_end2end_3d(n_classes=20, os=16,
                                                                  hidden_layers=128,
                                                                  source_classes=7,
                                                                  middle_classes=18, )
    ckpt = './universal_trained.pth'
    x = torch.load(ckpt)
    net.load_state_dict_new(x)
    print('load pretrainedModel.')

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

if __name__ == '__main__':
    main()
