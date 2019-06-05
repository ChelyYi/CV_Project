import numpy as np

def mean_iu(predict,gt):
    """
    :param predict: predict result, 2-d np array
    :param gt: ground truth, 2-d np array
    :return: mean iu value
    """
    iu = 0
    predict_classes = np.unique(predict)# classes in predict result
    gt_classes = np.unique(gt)

    if predict_classes.shape[0] >= gt_classes.shape[0]:
        class_num = predict_classes.shape[0]
        classes = predict_classes
    else:
        class_num = gt_classes.shape[0]
        classes = gt_classes

    # if class_num == 1: # only background
    #     return 0.0
    for c in classes:
        # if c == 0: # skip background class
        #     continue
        p_ii =element_wise_same(predict,gt,c)
        t_i = np.count_nonzero(gt == c)
        p_ij = np.count_nonzero(predict == c)
        if t_i != 0 or p_ij != 0:
            iu += p_ii/(t_i + p_ij - p_ii)
    mean_iu = iu / class_num
    return mean_iu

def element_wise_same(predict,gt,c):
    size = gt.shape[0]
    true = 0
    for i in range(0,size):
        for j in range(0,size):
            if predict[i][j] == c and predict[i][j] == gt[i][j]:
                true += 1
    return true
