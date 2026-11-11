"""Source-canvas region each model sees of the 832x480 draft, from eval/configs/models.yaml."""
import math

from worlddraft_eval.common import model_config

TRANSFORMS = {m: c['transform'] for m, c in model_config.load().items()}


def input_crop(model, source_size, output_size):
    """Return source-pixel [left, top, width, height], before resizing to output."""
    w, h = source_size
    ow, oh = output_size
    if min(w, h, ow, oh) <= 0:
        raise ValueError('Image dimensions must be positive')
    config = model_config.get(model)
    transform = config['transform']
    if transform == 'pad':
        if [w, h] != list(config['input_size']):
            raise ValueError(f'{model} upload mapping requires {config["input_size"][0]}x{config["input_size"][1]} source')
        # The service does not document its crop: the whole padded upload is assumed to fill the output.
        return [0, -config['top'], w, h + config['top'] + config['bottom']]
    if transform == 'letterbox':
        scale=min(ow/w,oh/h)
        rw,rh=round(w*scale),round(h*scale)
        return [-((ow-rw)//2)*w/rw,-((oh-rh)//2)*h/rh,ow*w/rw,oh*h/rh]
    if transform == 'same_canvas':
        if (w,h)!=(ow,oh):raise ValueError(f'{model} spatial transform is unverified for changed canvas sizes')
        return [0,0,w,h]
    if transform == 'area_resize_topleft_crop':
        area = 480 * config['area_width']
        rh = math.sqrt(area / (w / h))
        rw, rh = round(rh * w / h), round(rh)
        if (ow, oh) != (rw // 16 * 16, rh // 16 * 16):
            raise ValueError(f'{model} output disagrees with pinned preprocessor')
        return [0, 0, ow * w / rw, oh * h / rh]
    if transform == 'center_crop':
        scale = max(ow / w, oh / h)
        cw, ch = ow / scale, oh / scale
        return [(w - cw) / 2, (h - ch) / 2, cw, ch]
    if transform == 'none':
        return [0, 0, w, h]
    raise ValueError(f'{model}: unknown transform {transform!r} in eval/configs/models.yaml')
