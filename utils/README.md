# Ussage:
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

## TODO:

**load raw binary if --share not included C(Channel)=3**
In this case, it load shape(1,84,8400), the code will get failure.
```
 $ python ./utils/numpy_utils.py --load_raw ./face-640x640.raw.out  --shape 1,84,8400 --raw_dtype int32 --type int32 --show 30
Loaded RAW: path=./face-640x640.raw.out, shape=(1, 84, 8400), dtype=>i4
usage: numpy_utils.py [-h] (--load LOAD | --load_raw LOAD_RAW) [--shape SHAPE] [--raw_dtype {int8,uint8,int16,uint16,int32,uint32,float32,float64}]
                      [--show SHOW] [--save SAVE] [--as {HWC,CHW}] [--type {int8,int32,uint8,float32}] [--raw] [--little]
numpy_utils.py: error: Expect CHW with C=3 before transposing to HWC.
```
## Issue:

**.npy packle should already included shape info.**

In this case, Cannot load packle *face-640x640.np*  to shown .npy without *--shape* and *--as*
```
> python  numpy_utils.py  --load ../samples/face-640x640.npy --shape 640,640,3 --as CHW --show 10
Loaded NPY: path=../samples/face-640x640.npy, shape=(640, 640, 3), dtype=uint8
Layout -> CHW: shape=(3, 640, 640)
Dtype -> int8
First 10 elements: [-40 -40 -40 -40 -40 -40 -40 -40 -40 -40]
```

working if given options of *--shape* and *--as*

```
> python  numpy_utils.py  --load ../samples/face-640x640.npy --show 10
Loaded NPY: path=../samples/face-640x640.npy, shape=(640, 640, 3), dtype=uint8
usage: numpy_utils.py [-h] (--load LOAD | --load_raw LOAD_RAW) [--shape SHAPE] [--raw_dtype {int8,uint8,int16,uint16,int32,uint32,float32,float64}]
                      [--show SHOW] [--save SAVE] [--as {HWC,CHW}] [--type {int8,int32,uint8,float32}] [--raw] [--little]
numpy_utils.py: error: Expect CHW with C=3 before transposing to HWC.
```



