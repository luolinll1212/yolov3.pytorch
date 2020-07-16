# -*- coding: utf-8 -*-  

class config:
    # 数据集参数
    train_list = "./data/2007_train.txt"
    test_list = "./data/2007_test.txt"
    num_workers = 4
    batch_size = 8
    manualseed = 0
    img_h = 416
    img_w = 416
    classes = 20
    anchors = [[[116, 90], [156, 198], [373, 326]],
                [[30, 61], [62, 45], [59, 119]],
                [[10, 13], [16, 30], [33, 23]]]
    voc_classes = ["aeroplane","bicycle","bird","boat","bottle",
                   "bus","car","cat","chair","cow",
                   "diningtable","dog","horse","motorbike","person",
                   "pottedplant","sheep","sofa","train","tvmonitor"]

    # 训练参数
    num_epochs = 200
    start_epoch = 1
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.999
    output = "output"
    pretrained = f"./{output}/yolov3.voc.180.pt"
    interval = 100
    valinterval = 20

    # detect
    confidence = 0.5
    num_thres = 0.3
    eval_pt = ""
    font = "./data/simhei.ttf"
    
