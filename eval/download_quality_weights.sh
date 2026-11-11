#!/usr/bin/env bash
set -euo pipefail
# Usage: bash eval/download_quality_weights.sh [target dir]  (default ~/.cache/worlddraft/vbench-quality;
# set quality_weights in eval/configs/envs.local.yaml when using another directory)
WEIGHTS=${1:-$HOME/.cache/worlddraft/vbench-quality}
mkdir -p "$WEIGHTS/clip_model" "$WEIGHTS/pyiqa_model" "$WEIGHTS/aesthetic_model"
# Dependencies are checked separately; downloading must not upgrade the evaluation environment.
[ "$(stat -c %s "$WEIGHTS/clip_model/ViT-B-32.pt" 2>/dev/null || echo 0)" = 353976522 ] || curl -fL -C - --connect-timeout 20 --max-time 600 --retry 5 -o "$WEIGHTS/clip_model/ViT-B-32.pt" 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt'
[ "$(stat -c %s "$WEIGHTS/clip_model/ViT-L-14.pt" 2>/dev/null || echo 0)" = 932768134 ] || curl -fL -C - --connect-timeout 20 --max-time 600 --retry 5 -o "$WEIGHTS/clip_model/ViT-L-14.pt" 'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt'
[ "$(stat -c %s "$WEIGHTS/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth" 2>/dev/null || echo 0)" = 108610983 ] || curl -fL -C - --connect-timeout 20 --max-time 600 --retry 5 -o "$WEIGHTS/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth" 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth?resume=1792779000'
[ "$(stat -c %s "$WEIGHTS/aesthetic_model/sa_0_4_vit_l_14_linear.pth" 2>/dev/null || echo 0)" = 4071 ] || curl -fL -C - --connect-timeout 20 --max-time 600 --retry 5 -o "$WEIGHTS/aesthetic_model/sa_0_4_vit_l_14_linear.pth" 'https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth'
WEIGHTS="$WEIGHTS" ${PYTHON:-python} - <<'CHECK'
import hashlib,json,os
from pathlib import Path
import torch
root=Path(os.environ['WEIGHTS'])
expected={'clip_model/ViT-B-32.pt':'40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af','clip_model/ViT-L-14.pt':'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836','pyiqa_model/musiq_spaq_ckpt-358bb6af.pth':'358bb6af','aesthetic_model/sa_0_4_vit_l_14_linear.pth':''}
manifest={}
for name,prefix in expected.items():
 p=root/name
 with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
 if not digest.startswith(prefix): raise ValueError(f'Checksum mismatch: {p}')
 if name.endswith('.pth'): torch.load(p,map_location='cpu',weights_only=True)
 manifest[name]={'sha256':digest,'bytes':p.stat().st_size}
 print('verified',name,digest,flush=True)
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(root/'DOWNLOAD_COMPLETE').write_text('verified\n')
CHECK
