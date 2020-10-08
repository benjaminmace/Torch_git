import torch
from creating_bounding_box import intersection_over_union

def nms(bboxes, iou_threshold, threshold, box_format='corners'):
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes_after_nms = []
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [box
                  for box in bboxes
                  if box[0] != chosen_box[0]
                  or intersection_over_union(torch.tensor(chosen_box[2:]),
                                             torch.tensor(box[2:]),
                                             box_format=box_format) < iou_threshold]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms



