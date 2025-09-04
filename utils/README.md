# model converter

* convert_to_tflite - convert .torchscript to .tflite without quantize.

Repeat run convert_to_tflite will get identical .tflite.
But every ncc-tflite will lead to different .dla.

> python ./convert_to_tflite
> ncc-tflite --arch=mdla3.0 --relax-fp32  yolov8s.tflite -d yolov8s.dla

* convert_to_tflite_quantized - convert .torchscript to .tflite with quantize.
> python ./convert_to_tflite_quantized

# Usage:
```
> python ./numpy_utils.py --load ../samples/face-640x640.npy  --as CHW --type int32 --save ./img.chw.i32.le --raw --little
Loaded NPY: path=../samples/face-640x640.npy, shape=(640, 640, 3), dtype=uint8
Layout -> CHW: shape=(3, 640, 640)
Dtype -> int32
Saving RAW (LE) -> ./img.chw.i32.le
```
```
> python ./numpy_utils.py --load ../samples/face-640x640.npy  --as CHW --type int32 --save ./img.chw.i32.be --raw
Loaded NPY: path=../samples/face-640x640.npy, shape=(640, 640, 3), dtype=uint8
Layout -> CHW: shape=(3, 640, 640)
Dtype -> >i4
Saving RAW (BE) -> ./img.chw.i32.be
```
```
> python ./numpy_utils.py --load ../samples/face-640x640.npy  --as CHW --type float32 --save ./img.chw.fp32.be --raw
```

## Loading non-3-channel tensors

Use `--as NONE` to disable layout checks when working with tensors that do not have three channels.

```
$ python ./utils/numpy_utils.py --load_raw ./face-640x640.raw.out  --shape 1,84,8400 --raw_dtype int32 --as NONE --type int32 --show 3
Loaded RAW: path=./face-640x640.raw.out, shape=(1, 84, 8400), dtype=>i4
Dtype -> >i4
First 3 elements: [0 0 0]
```

The `--help` message now reflects the new option:

```
usage: numpy_utils.py [-h] (--load LOAD | --load_raw LOAD_RAW) [--shape SHAPE] [--raw_dtype {int8,uint8,int16,uint16,int32,uint32,float32,float64}]
                      [--show SHOW] [--save SAVE] [--as {NONE,HWC,CHW}] [--type {int8,int32,uint8,float32}] [--raw] [--little]
```

> ./numpy_utils.py --load_raw  face_1x20x8400.bin --shape 1,20,8400 --raw_dtype float32 --type float32 --show 100 --little
