import 'dart:io';

import 'package:collection/collection.dart';
import 'package:flutter_svg/svg.dart';
import 'package:path_provider/path_provider.dart';

import 'package:mlperfbench/app_constants.dart';
import 'package:mlperfbench/backend/bridge/ffi_config.dart';
import 'package:mlperfbench/backend/bridge/handle.dart';
import 'package:mlperfbench/backend/bridge/isolate.dart';
import 'package:mlperfbench/backend/list.dart';
import 'package:mlperfbench/backend/run_settings.dart';
import 'package:mlperfbench/benchmark/benchmark_result.dart';
import 'package:mlperfbench/icons.dart';
import 'package:mlperfbench/info.dart';
import 'package:mlperfbench/protos/backend_setting.pb.dart' as pb;
import 'package:mlperfbench/protos/mlperf_task.pb.dart' as pb;
import 'package:mlperfbench/resources/resource_manager.dart';

class Benchmark {
  final pb.BenchmarkSetting benchmarkSetting;
  final pb.TaskConfig taskConfig;
  final pb.ModelConfig modelConfig;

  double? score;
  String? accuracy;
  String backendDescription;

  Benchmark(this.benchmarkSetting, this.taskConfig, this.modelConfig)
      : backendDescription =
            '${benchmarkSetting.configuration} | ${benchmarkSetting.acceleratorDesc}';

  String get id => modelConfig.id;

  String get name => modelConfig.name;

  double get maxScore => MAX_SCORE[id]!;

  /// 'Object Detection', 'Image Classification (offline)', and so on.
  String get taskName => taskConfig.name;

  /// 'IC', 'OD', and so on.
  String get code => modelConfig.id.split('_').first;

  /// 'SingleStream' or 'Offline'.
  String get scenario => modelConfig.scenario;

  BenchmarkTypeEnum get type => _typeFromCode();

  SvgPicture get icon => BENCHMARK_ICONS[scenario]?[code] ?? Icons.logo;

  SvgPicture get iconWhite =>
      BENCHMARK_ICONS_WHITE[scenario]?[code] ?? Icons.logo;

  @override
  String toString() => 'Benchmark:$id';

  BenchmarkTypeEnum _typeFromCode() {
    switch (code) {
      case 'IC':
        return BenchmarkTypeEnum.imageClassification;
      case 'OD':
        return BenchmarkTypeEnum.objectDetection;
      case 'IS':
        return BenchmarkTypeEnum.imageSegmentation;
      case 'LU':
        return BenchmarkTypeEnum.languageUnderstanding;
      default:
        return BenchmarkTypeEnum.unknown;
    }
  }
}

class MiddleInterface {
  final List<Benchmark> benchmarks;

  MiddleInterface._(this.benchmarks);

  static Future<MiddleInterface> create(
      File configFile, BackendInfo backendInfo) async {
    final tasks = getMLPerfConfig(await configFile.readAsString());

    final benchmarks = <Benchmark>[];
    for (final task in tasks.task) {
      for (final model in task.model) {
        final benchmarkSetting = backendInfo.settings.benchmarkSetting
            .singleWhereOrNull((setting) => setting.benchmarkId == model.id);
        if (benchmarkSetting == null) continue;

        benchmarks.add(Benchmark(benchmarkSetting, task, model));
      }
    }

    return MiddleInterface._(benchmarks);
  }

  /// The list of URL or file names to download.
  List<String> data() {
    // TODO unify with listSelectedResources
    final result = <String>[];

    for (final b in benchmarks) {
      result.add(b.taskConfig.liteDataset.path);
      result.add(b.taskConfig.liteDataset.groundtruthSrc);

      result.add(b.taskConfig.dataset.path);
      result.add(b.taskConfig.dataset.groundtruthSrc);

      result.add(b.taskConfig.testDataset.path);
      result.add(b.taskConfig.testDataset.groundtruthSrc);

      result.add(b.benchmarkSetting.src);
    }

    result.sort();
    return result.where((element) => element.isNotEmpty).toList();
  }
}

enum BenchmarkTypeEnum {
  unknown,
  imageClassification,
  objectDetection,
  imageSegmentation,
  languageUnderstanding,
}

enum DatasetMode { lite, full, test }

class BenchmarkJob {
  final Benchmark benchmark;
  late final pb.DatasetConfig dataset;
  final bool accuracyMode;
  late final bool _fastMode;
  late final DatasetMode _datasetMode;
  final int threadsNumber;

  BenchmarkJob({
    required this.benchmark,
    required this.accuracyMode,
    required this.threadsNumber,
    required bool testMode,
  }) {
    if (testMode) {
      _datasetMode = DatasetMode.test;
      dataset = benchmark.taskConfig.testDataset;
    } else {
      _datasetMode = accuracyMode ? DatasetMode.full : DatasetMode.lite;
      dataset = accuracyMode
          ? benchmark.taskConfig.dataset
          : benchmark.taskConfig.liteDataset;
    }
    _fastMode = testMode || FAST_MODE;
  }

  Future<RunResult> run(ResourceManager resourceManager, BridgeIsolate backend,
      List<pb.Setting> commonSettings, String backendLibPath) async {
    final tmpDir = await getTemporaryDirectory();

    print(
        'Running $benchmark in ${accuracyMode ? 'accuracy' : 'performance'} mode...');
    final stopwatch = Stopwatch()..start();

    var minQueryCount = _fastMode ? 8 : benchmark.taskConfig.minQueryCount;
    var minDuration = _fastMode ? 10 : benchmark.taskConfig.minDurationMs;

    final settings = pb.SettingList(
      setting: commonSettings,
      benchmarkSetting: benchmark.benchmarkSetting,
    );

    final batchSizeValue = benchmark.benchmarkSetting.batchSize;

    if (benchmark.modelConfig.scenario == 'Offline') {
      var shardsNumSetting = benchmark.benchmarkSetting.customSetting
          .singleWhereOrNull((element) => element.id == 'shards_num');
      if (shardsNumSetting == null) {
        benchmark.benchmarkSetting.customSetting.add(pb.CustomSetting(
            id: 'shards_num', value: threadsNumber.toString()));
      } else {
        shardsNumSetting.value = threadsNumber.toString();
      }
      var batchSizeSetting = benchmark.benchmarkSetting.customSetting
          .singleWhereOrNull((element) => element.id == 'batch_size');
      if (batchSizeSetting == null) {
        benchmark.benchmarkSetting.customSetting.add(pb.CustomSetting(
            id: 'batch_size', value: batchSizeValue.toString()));
      } else {
        batchSizeSetting.value = batchSizeValue.toString();
      }

      benchmark.benchmarkSetting.batchSize *= threadsNumber;
      settings.setting.add(pb.Setting(
        id: 'shards_num',
        name: 'Number of threads for inference',
        value: pb.Setting_Value(
          name: threadsNumber.toString(),
          value: threadsNumber.toString(),
        ),
      ));
    }

    var backendNativeLibPath = '';
    if (Platform.isAndroid) {
      backendNativeLibPath = await getNativeLibraryPath();
    }
    final result = await backend.run(RunSettings(
      backend_model_path: resourceManager.get(benchmark.benchmarkSetting.src),
      backend_lib_path: backendLibPath,
      backend_settings: settings.writeToBuffer(),
      backend_native_lib_path: backendNativeLibPath,
      dataset_type: benchmark.taskConfig.dataset.type.value,
      dataset_data_path: resourceManager.get(dataset.path),
      dataset_groundtruth_path: resourceManager.get(dataset.groundtruthSrc),
      dataset_offset: benchmark.modelConfig.offset,
      scenario: benchmark.modelConfig.scenario,
      batch: benchmark.benchmarkSetting.batchSize,
      batch_size: batchSizeValue,
      threads_number: threadsNumber,
      mode: accuracyMode
          ? BenchmarkMode.backendAccuracy
          : BenchmarkMode.backendPerfomance,
      min_query_count: minQueryCount,
      min_duration: minDuration,
      output_dir: tmpDir.path,
      benchmark_id: benchmark.id,
      dataset_mode: _datasetMode,
    ));
    final elapsed = stopwatch.elapsed;

    print('Benchmark result: $result, elapsed: $elapsed');
    return result;
  }
}
