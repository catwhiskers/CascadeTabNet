from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import cv2
# Load model
config_file = './Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py'
checkpoint_file = './epoch_36.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Test a single image
img = "./Demo/demo.png"
cv2img = cv2.imread(img)
# Run Inference
result = inference_detector(model, img)

print(result)
# Visualization results
cv2img = show_result(cv2img, result,('Bordered', 'cell', 'Borderless'), score_thr=0.85, show=False)
cv2.imwrite('output.png', cv2img)
