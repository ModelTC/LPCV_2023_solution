# LPCV 2023 solution

## LPCV 2023 introduction
https://lpcv.ai/2023LPCVC/introduction

## Inference Environment
### Hardware
[NVIDIA Jetson Nano 2GB](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit)

### Software
refer to official sample: [23LPCVC_Segmentation_Track-Sample_Solution](https://github.com/lpcvai/23LPCVC_Segmentation_Track-Sample_Solution) 


## Our submission
- file `submit_pyz/submit_0803.pyz`
- SHA256 = f1db90947eebedc3229bd5dd70ce5af586893ed8e370acce4f1b9ce33c62c315
- Submitted at **2023-08-03 09:16:28 EST**
- Perfomance Score **75.608**

## Usage of our inference code
- train a segmentation model on **your server**
- export a '.onnx' file from your framework on **your server**
- Use `trtexec` on **Jetson Nano**, convert onnx model to tensorrt model: 
```shell
trtexec --workspace=4096 --onnx=xxx.onnx --saveEngine=xxx.trt --best --useSpinWait --outputIOFormats=fp16:chw --inputIOFormats=fp16:chw --verbose
```
- put the tensorrt model file into `inference_code` folder
- pack `inference_code` folder into pyz and submit(this step is same as the sample solution)

## Training 

Goto [Train readme](./train/README.md) for training details, log, codebase, trained weights and reproduction