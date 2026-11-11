"""Project Blender instance masks through the generator's recorded image transform."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from PIL import Image

from worlddraft_eval.common.spatial_mapping import TRANSFORMS, input_crop


def output_mask_gt(sparse, canonical, model, output_size, masks_dir):
    gt = deepcopy(canonical)
    w, h = output_size
    crop = input_crop(model, (gt['width'],gt['height']), output_size)
    transform = sparse['letterbox']
    sw,sh = sparse['source_size']
    rw,rh = transform['resized_size']
    ox,oy = transform['offset_xy']
    # PIL takes the inverse map, from output pixels back to the source mask.
    affine = (crop[2]/w*sw/rw, 0, (crop[0]-ox)*sw/rw,
              0, crop[3]/h*sh/rh, (crop[1]-oy)*sh/rh)
    entities = {e['entity_id']:e['instance_id'] for e in sparse['entities']}
    for frame in gt['frames']:
        with Image.open(masks_dir/f"mask_{frame['frame']:03d}.png") as im:
            if transform.get('stages'):
                for stage in transform['stages']:
                    resized = im.resize(tuple(stage['resized_size']), Image.Resampling.NEAREST)
                    im = Image.new(im.mode, tuple(stage['target_size']), 0)
                    im.paste(resized, tuple(stage['offset_xy']))
                output_affine = (crop[2]/w, 0, crop[0], 0, crop[3]/h, crop[1])
                labels = np.asarray(im.transform((w,h),Image.Transform.AFFINE,output_affine,
                                                resample=Image.Resampling.NEAREST))
            else:
                labels = np.asarray(im.transform((w,h),Image.Transform.AFFINE,affine,
                                                resample=Image.Resampling.NEAREST))
        for eid, iid in entities.items():
            ys,xs = np.where(labels == iid)
            area = int(xs.size)
            frame['boxes'][eid] = dict(visible=bool(area),visible_mask_area=area,
                box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if area else None,
                center=[float(xs.mean())+.5,float(ys.mean())+.5] if area else None)
    gt.update(width=w,height=h)
    gt['adaptation'].update(output_crop_xywh=crop,output_size=list(output_size),
        mask_mapping='nearest-neighbor source instance mask through input letterbox and generator crop',
        spatial_mapping=TRANSFORMS.get(model))
    return gt
