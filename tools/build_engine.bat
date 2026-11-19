@echo off
REM Сборка TRT-движка на ЭТОЙ карте. Движок непереносим.
REM Требует trtexec из TensorRT в PATH и model.onnx рядом.
REM ВАЖНО: --inputIOFormats/--outputIOFormats fp16 ОБЯЗАТЕЛЬНЫ — иначе движок
REM ждет FP32 (x2 памяти), а конвейер подает FP16: silent data corruption,
REM потом illegal access и смерть всего конвейера (проверено на Blackwell).
REM ONNX обязан быть нативно FP16 (см. export_onnx.py --dtype fp16),
REM иначе stronglyTyped сборка упадет (нужен был бы ICastLayer).
setlocal
if "%TRT_ROOT%"=="" echo Set TRT_ROOT & exit /b 1
"%TRT_ROOT%\bin\trtexec.exe" --onnx=model.onnx --saveEngine=model.engine ^
  --fp16 --stronglyTyped --useCudaGraph --builderOptimizationLevel=5 ^
  --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw ^
  --workspace=4096 --verbose
echo built model.engine
