import torch
import torch.nn.functional as F
import numpy as np
import os
import copy


def negative_index_sampler(samp_num, seg_num_list):
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            negative_index += np.random.randint(low=sum(seg_num_list[:j]),
                                                high=sum(seg_num_list[:j+1]),
                                                size=int(samp_num[i, j])).tolist()
    return negative_index

# --------------------------------------------------------------------------------
# Define ReCo loss
# --------------------------------------------------------------------------------
def compute_reco_loss(rep, label, uncer, strong_threshold=0.5, temp=0.5, num_queries=256, num_negatives=256):
    batch_size, num_feat, im_w_, im_h = rep.shape
    cls_num = label.shape[1]
    device = rep.device

    # compute valid binary mask for each pixel
    # valid_pixel = label * mask

    # permute representation for indexing: batch x im_h x im_w x feature_channel
    rep = rep.permute(0, 2, 3, 1)
    # label [1024, 1, 48, 48] why? 
    # compute prototype (class mean representation) for each class across all valid pixels
    seg_feat_all_list = []
    seg_feat_hard_list = []
    seg_num_list = []
    seg_proto_list = []
    for i in range(cls_num):
        # valid_pixel_seg = valid_pixel[:, i]  # select binary mask for i-th class
        cls_pixels = label[:, i]  # select binary mask for i-th class
        if cls_pixels.sum() == 0:  # not all classes would be available in a mini-batch
            continue

        uncer_seg = uncer[:, i, :, :]
        rep_mask_hard = (uncer_seg > strong_threshold) * cls_pixels.bool()  # select hard queries

        # seg_proto_list.append(torch.mean(rep[valid_pixel_seg.bool()], dim=0, keepdim=True))
        seg_proto_list.append(torch.mean(rep[cls_pixels.bool()], dim=0, keepdim=True)) # --> proto type (각 class 마다의 representation의 mean) 을 저장하게 됨 
        seg_feat_all_list.append(rep[cls_pixels.bool()])
        seg_feat_hard_list.append(rep[rep_mask_hard])
        seg_num_list.append(int(cls_pixels.sum().item())) # 각 class마다의 pixel의 개수


    
    # compute regional contrastive loss
    if len(seg_num_list) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        return torch.tensor(0.0)
    else:
    
        reco_loss = torch.tensor(0.0)
        seg_proto = torch.cat(seg_proto_list)
        valid_seg = len(seg_num_list)
        # seg_len = torch.arange(valid_seg)

        for i in range(valid_seg):
            # sample hard queries
            if len(seg_feat_hard_list[i]) > 0:
                # query 만큼의 index를 추출
                seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
                anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]
                anchor_feat = anchor_feat_hard
            else:  # in some rare cases, all queries in the current query class are easy
                continue

            # apply negative key sampling (with no gradients)
            with torch.no_grad():
                # generate index mask for the current query class; e.g. [0, 1, 2] -> [1, 2, 0] -> [2, 0, 1]
                # seg_mask = torch.cat(([seg_len[i:], seg_len[:i]]))

                # compute similarity for each negative segment prototype (semantic class relation graph)
                # 이부분을 어떻게 해석해야하는지 in binary segmentation
                
                
                # proto_sim = torch.cosine_similarity(seg_proto[seg_mask[0]].unsqueeze(0), seg_proto[seg_mask[1:]], dim=1)
                # proto_prob = torch.softmax(proto_sim / temp, dim=0)

                # sampling negative keys based on the generated distribution [num_queries x num_negatives]
                # negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
                # samp_class = negative_dist.sample(sample_shape=[num_queries, num_negatives])
                # samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)

                # sample negative indices from each negative class
                # negative_num_list = seg_num_list[i+1:] + seg_num_list[:i]
                # negative_index = negative_index_sampler(samp_num, negative_num_list)
                
                
                ######################################
                # since binary, we just have to sample index from the num_negative 
                negative_num_list = seg_num_list[(i+1) % 2]
                # list of index
                negative_index = np.random.randint(low = 0, high = negative_num_list-1, size = (num_queries, num_negatives))
                
                # index negative keys (from other classes)
                # negative_feat_all = torch.cat(seg_feat_all_list[i+1:] + seg_feat_all_list[:i])
                negative_feat_all = seg_feat_all_list[(i+1) % 2]
                
                
                negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, num_feat)

                # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
                positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
                all_feat = torch.cat((positive_feat, negative_feat), dim=1)

            seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
            reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().to(device))
        return reco_loss / valid_seg
