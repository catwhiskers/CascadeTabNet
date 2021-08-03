wget https://download.openmmlab.com/pretrain/third_party/hrnetv2_w32-dc9eeb4f.pth
#mv hrnetv2_w32-dc9eeb4f.pth ~/.cache/torch/checkpoints/ 
mkdir -p /root/.cache/
mkdir -p /root/.cache/torch/
mkdir -p /root/.cache/torch/checkpoints/
mv hrnetv2_w32-dc9eeb4f.pth /root/.cache/torch/checkpoints/ 
