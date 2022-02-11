# Getting Started

## Environment Setup

### Install Bazel

Install Bazel version 4.1.

### Install Compiler

```commandline
sudo apt install clang

export CC=clang
export CXX=clang++
```

## Clone IREE and MLPerf

A local copy of IREE is needed as a dependency for MLPerf. When cloning the repos, ensure that the project structure below is used:

```commandline
./iree
./mlperf/mobile_app_open
```

### Clone and Build IREE
```commandline
cd ${PROJECT_ROOT}
git clone https://github.com/google/iree.git

cd iree
git submodule update --init
python3 configure_bazel.py

bazel test -k iree/...
```

### Clone MLPerf
The IREE backend code lives in my fork at the moment on branch `linux_workspace`.
```commandline
cd ${PROJECT_ROOT}
mkdir mlperf
cd mlperf

git clone https://github.com/mariecwhite/mobile_app_open.git
git checkout linux_workspace
```

## Retrieving Datasets

Links to datasaets for each task can be found in this config file: https://github.com/mlcommons/mobile_models/blob/main/v1_0/assets/tasks_v4.pbtxt

Some of these datasets are trimmed down versions (e.g. Squad only contains ~130 samples). Please contact me for instructions on how to generate the full versions.

## Running a Benchmark

### TFLite Backend

```commandline
MODEL_PATH=<path to TFLite file>
SQUAD_ROOT=<path to SQUAD dataset>

OUTPUT_DIR=/tmp/mlperf
rm -rf ${OUTPUT_DIR}
mkdir ${OUTPUT_DIR}

bazel build -c opt --config=linux mobile_back_tflite/cpp/backend_tflite:libtflitebackend.so
bazel run -c opt --config=linux android/cpp/binary/main -- EXTERNAL SQUAD --mode=SubmissionRun  \
--output_dir=${OUTPUT_DIR}  \
--model_file=${MODEL_PATH} \
--input_file=${SQUAD_ROOT}/squad_eval_mini.tfrecord \
--groundtruth_file=${SQUAD_ROOT}/squad_groundtruth.tfrecord \
--lib_path=`pwd`/bazel-bin/mobile_back_tflite/cpp/backend_tflite/libtflitebackend.so \
--min_query_count=32 \
--min_duration=1000
```

### IREE Backend

```commandline
BASE_DIR=<path to directory of TFLite file>
MODEL_NAME=<name of model>

# First generate the IREE module.
iree-import-tflite ${BASE_DIR}/${MODEL_NAME}.tflite -o ${BASE_DIR}/${MODEL_NAME}.mlir
iree-translate --iree-input-type=tosa --iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=dylib-llvm-aot ${BASE_DIR}/${MODEL_NAME}.mlir --o ${BASE_DIR}/${MODEL_NAME}.vmfb

# Now run MLPerf with the vmfb as input.
SQUAD_ROOT=<path to SQUAD dataset>
MODULE_PATH=${BASE_DIR}/${MODEL_NAME}.vmfb

OUTPUT_DIR=/tmp/mlperf
rm -rf ${OUTPUT_DIR}
mkdir ${OUTPUT_DIR}

bazel build -c opt --config=linux mobile_back_iree/cpp/backend_iree:libireebackend.so

bazel run -c opt --config=linux android/cpp/binary/main -- IREE SQUAD --mode=SubmissionRun  \
--output_dir=${OUTPUT_DIR}  \
--model_file=${MODEL_PATH} \
--module=${MODULE_PATH} \
--input_file=${SQUAD_ROOT}/squad_eval_mini.tfrecord \
--groundtruth_file=${SQUAD_ROOT}/squad_groundtruth.tfrecord \
--lib_path=`pwd`/bazel-bin/mobile_back_iree/cpp/backend_iree/libireebackend.so \
--min_query_count=32 \
--min_duration=1000 \
--function_inputs="1x384xi32,1x384xi32,1x384xi32" \
--function_outputs="1x384xf32,1x384xf32"
```

Sample output should look like this:
```commandline
================================================
MLPerf Results Summary
================================================
SUT name : IREE
Scenario : SingleStream
Mode     : SubmissionRun
90th percentile latency (ns) : 1226613802
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes

================================================
Additional Stats
================================================
QPS w/ loadgen overhead         : 0.84
QPS w/o loadgen overhead        : 0.84

Min latency (ns)                : 1126466297
Max latency (ns)                : 1295413525
Mean latency (ns)               : 1193830595
50.00 percentile latency (ns)   : 1193365586
90.00 percentile latency (ns)   : 1226613802
95.00 percentile latency (ns)   : 1247309784
97.00 percentile latency (ns)   : 1295413525
99.00 percentile latency (ns)   : 1295413525
99.90 percentile latency (ns)   : 1295413525

================================================
Test Parameters Used
================================================
samples_per_query : 1
target_qps : 1000
target_latency (ns): 0
max_async_queries : 1
min_duration (ms): 1000
max_duration (ms): 0
min_query_count : 32
max_query_count : 0
qsl_rng_seed : 1624344308455410291
sample_index_rng_seed : 517984244576520566
schedule_rng_seed : 10051496985653635065
accuracy_log_rng_seed : 0
accuracy_log_probability : 0
accuracy_log_sampling_target : 0
print_timestamps : 0
performance_issue_unique : 0
performance_issue_same : 0
performance_issue_same_index : 0
performance_sample_count : 108506

No warnings encountered during test.

No errors encountered during test.
2022-02-11 13:11:43.282294: I android/cpp/binary/main.cc:386] 90 percentile latency: 1226.61 ms
2022-02-11 13:20:45.404771: I android/cpp/binary/main.cc:387] Accuracy: 87.9930
```

