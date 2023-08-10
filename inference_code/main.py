from argparse import ArgumentParser, Namespace
from typing import List, Tuple
import numpy as np
# from PIL import Image
import torch
import torch.nn.functional as F
# from torch import Tensor
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import gc
import os
import ctypes
import tensorrt as trt
import pkg_resources
import imageio
from concurrent.futures import ThreadPoolExecutor

libcudart = ctypes.CDLL('libcudart.so')

SIZE: List[int] = [512, 512]
input_size_x = 256
INPUT_SIZE = [input_size_x, input_size_x]

MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
STANDARD_DEVIATION: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def getArgs() -> Namespace:
    programName: str = "LPCVC 2023 Sample Solution"
    authors: List[str] = ["Benjamin Boardley", "Nicholas M. Synovic", "Ping Hu"]

    prog: str = programName
    usage: str = f"This is the {programName}"
    description: str = f"This {programName} does create a single segmentation map of arieal scenes of disaster environments captured by unmanned arieal vehicles (UAVs)"
    epilog: str = f"This {programName} was created by {''.join(authors)}"

    # NOTE: Do not change these flags
    parser: ArgumentParser = ArgumentParser(prog, usage, description, epilog)
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Filepath to an image to create a segmentation map of",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Filepath to the corresponding output segmentation map",
    )

    return parser.parse_args()


def write_image(outArray, output_image_path):
    imageio.imwrite(output_image_path, outArray, format='png')
    # outImage = Image.fromarray(outArray, mode='L')
    # outImage.save(output_image_path)


def gpu_free(data):
    libcudart.cudaFree(data)


def gpu_memcpy_htod(data, h_data):
    libcudart.cudaMemcpy(data, h_data.ctypes.data_as(ctypes.c_void_p), h_data.nbytes, 1)


def gpu_memcpy_dtoh(data, d_data):
    libcudart.cudaMemcpy(data.ctypes.data_as(ctypes.c_void_p), d_data, data.nbytes, 2)


def gpu_alloc(size):
    data = ctypes.c_void_p()
    libcudart.cudaMalloc(ctypes.byref(data), ctypes.c_size_t(size))
    return data


class SegmentationDataset(Dataset):
    __slots__ = ['root_dir', 'image_list', 'out_list', 'transform']

    def __init__(self, root_dir, out_dir):
        self.root_dir = root_dir
        self.image_list = [os.path.join(self.root_dir, i) for i in os.listdir(root_dir)]
        self.out_list = [os.path.join(out_dir, i) for i in os.listdir(root_dir)]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(tuple(INPUT_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=MEAN, std=STANDARD_DEVIATION)
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        return self.transform(imageio.imread(uri=self.image_list[idx])), self.out_list[idx]


def main() -> None:
    with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache()

        interp_mode = 'bicubic'

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        with pkg_resources.resource_stream(__name__, "256c_input256_step11_from_21k_bicubic_b4.trt") as model_file:
            engine = runtime.deserialize_cuda_engine(model_file.read())
        model_file.close()

        context = engine.create_execution_context()
        output_data = np.empty(engine.get_binding_shape(1), dtype=np.float16)
        output_volume = np.prod(engine.get_binding_shape(1))

        input_data = np.random.random((4, 3, input_size_x, input_size_x)).astype(np.float16)
        d_input = gpu_alloc(input_data.nbytes)
        d_output = gpu_alloc(output_volume * input_data.dtype.itemsize)
        args: Namespace = getArgs()
        test_dataset = SegmentationDataset(root_dir=args.input, out_dir=args.output)
        data_loader = DataLoader(test_dataset, persistent_workers=True, batch_size=4, shuffle=False,
            num_workers=2, prefetch_factor=2, pin_memory=True)

        gc.collect()
        torch.cuda.empty_cache()

        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        idx = 0

        # executor = ThreadPoolExecutor(max_workers=8)
        for input, filenames in data_loader:
            idx += 1
            if idx > 100//4:
                break
            gpu_memcpy_htod(d_input, input.data.numpy().astype(np.float16))
            torch.cuda.synchronize()
            start.record()
            context.execute_v2(bindings=[int(d_input.value), int(d_output.value)])
            end.record()
            gpu_memcpy_dtoh(output_data, d_output)
            outTensor = torch.from_numpy(output_data).cuda().float()
            n, c, h, w = outTensor.shape
            while h < 256:
                h *= 2
                w *= 2
                outTensor: torch.Tensor = F.interpolate(
                    outTensor, (h, w), mode=interp_mode, align_corners=True
                )
            outArray = F.interpolate(
                outTensor, SIZE, mode=interp_mode, align_corners=True
            ).data.max(1)[1]
            outArray = outArray.cpu().numpy().astype(np.uint8)
            # executor.submit(write_image, outArray[0], '/dev/null')
            # executor.submit(write_image, outArray[1], '/dev/null')
            # executor.submit(write_image, outArray[2], '/dev/null')
            # executor.submit(write_image, outArray[3], '/dev/null')
        # executor.shutdown(wait=True)
        # gc.collect()
        # torch.cuda.empty_cache()
        time = 0

        executor = ThreadPoolExecutor(max_workers=8)
        for input, filenames in data_loader:
            gpu_memcpy_htod(d_input, input.data.numpy().astype(np.float16))
            torch.cuda.synchronize()
            start.record()
            context.execute_v2(bindings=[int(d_input.value), int(d_output.value)])
            end.record()
            gpu_memcpy_dtoh(output_data, d_output)
            time += start.elapsed_time(end)
            outTensor = torch.from_numpy(output_data).cuda().float()
            n, c, h, w = outTensor.shape
            while h < 256:
                h *= 2
                w *= 2
                outTensor: torch.Tensor = F.interpolate(
                    outTensor, (h, w), mode=interp_mode, align_corners=True
                )
            outArray = F.interpolate(
                outTensor, SIZE, mode=interp_mode, align_corners=True
            ).data.max(1)[1]
            outArray = outArray.cpu().numpy().astype(np.uint8)
            executor.submit(write_image, outArray[0], filenames[0])
            executor.submit(write_image, outArray[1], filenames[1])
            executor.submit(write_image, outArray[2], filenames[2])
            executor.submit(write_image, outArray[3], filenames[3])
        executor.shutdown(wait=True)
    print(time / 1000)
    torch.cuda.empty_cache()
    gpu_free(d_input)
    gpu_free(d_output)
