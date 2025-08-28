
> cd utils/
> yolo export model=yolov8s.pt format=torchscript
> python ./convert_to_tflite_quantized
> python ./convert_to_tflite


# To inference
> python ./tflite_runner --model ./yolov8s_mtk.tflite --in_path ./samples/face-640x640.npy --out_path out.npy --auto_type --auto_reshape
