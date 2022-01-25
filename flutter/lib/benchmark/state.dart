import 'dart:io';
import 'dart:math';

import 'package:flutter/cupertino.dart';
import 'package:flutter/foundation.dart' show ChangeNotifier;
import 'package:flutter/material.dart' hide Icons;

import 'package:async/async.dart';
import 'package:collection/collection.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'package:path_provider/path_provider.dart';
import 'package:wakelock/wakelock.dart';

import 'package:mlperfbench/app_constants.dart';
import 'package:mlperfbench/backend/bridge/isolate.dart';
import 'package:mlperfbench/backend/list.dart';
import 'package:mlperfbench/benchmark/benchmark_result.dart';
import 'package:mlperfbench/info.dart';
import 'package:mlperfbench/protos/backend_setting.pb.dart' as pb;
import 'package:mlperfbench/resources/config_manager.dart';
import 'package:mlperfbench/resources/resource_manager.dart';
import 'package:mlperfbench/resources/utils.dart';
import 'package:mlperfbench/store.dart';
import 'benchmark.dart';

enum BenchmarkStateEnum {
  downloading,
  cooldown,
  waiting,
  running,
  aborting,
  done,
}

class BenchmarkState extends ChangeNotifier {
  final Store _store;
  final BridgeIsolate backendBridge;

  late final ResourceManager resourceManager;
  late final ConfigManager configManager;
  late final BackendInfo backendInfo;

  // null - downloading/waiting; false - running; true - done
  bool? _doneRunning;
  bool _cooling = false;

  // Only if [state] == [BenchmarkStateEnum.downloading]
  String get downloadingProgress => resourceManager.progress;

  // Only if [state] == [BenchmarkStateEnum.running]
  Benchmark? currentlyRunning;
  String runningProgress = '';

  num get result {
    final benchmarksCount =
        benchmarks.where((benchmark) => benchmark.performance != null).length;

    if (benchmarksCount == 0) return 0;

    final summaryScore = pow(
        benchmarks.fold<double>(1, (prev, i) {
          if (i.performance != null) return prev * i.performance!.score;
          return prev;
        }),
        1.0 / benchmarksCount);

    return summaryScore / _getSummaryMaxScore();
  }

  List<Benchmark> get benchmarks => _middle.benchmarks;

  Future<void> _cooldownFuture = Future.value();

  bool _aborting = false;

  late BenchmarkList _middle;

  BenchmarkState._(this._store, this.backendBridge) {
    resourceManager = ResourceManager(notifyListeners);
    backendInfo = BackendInfo.findMatching();
  }

  static double _getSummaryMaxScore() => MAX_SCORE['SUMMARY_MAX_SCORE']!;

  Future<String> validateExternalResourcesDirectory(
      String errorDescription) async {
    final resources = _middle.listResources(skipInactive: true, includeAccuracy: _store.submissionMode);
    final missing = await resourceManager.validateResourcesExist(resources);
    if (missing.isEmpty) return '';

    return errorDescription +
        missing.mapIndexed((i, element) => '\n${i + 1}) $element').join();
  }

  Future<String> validateOfflineMode(String errorDescription) async {
    final resources = _middle.listResources(skipInactive: true, includeAccuracy: _store.submissionMode);
    final internetResources = filterInternetResources(resources);
    if (internetResources.isEmpty) return '';

    return errorDescription +
        internetResources
            .mapIndexed((i, element) => '\n${i + 1}) $element')
            .join();
  }

  Future<void> clearCache() async {
    await resourceManager.cacheManager.deleteLoadedResources([], 0);
    await configManager.deleteDefaultConfig();
    await resetConfig();
    await loadResources();
  }

  Future<void> loadResources() async {
    _middle = BenchmarkList(
        await File(configManager.configPath).readAsString(),
        backendInfo.settings.benchmarkSetting,
        _store.testMode,
        resourceManager.getDefaultBatchPreset());
    await reset();

    final packageInfo = await PackageInfo.fromPlatform();
    final newAppVersion = packageInfo.version + '+' + packageInfo.buildNumber;
    var needToPurgeCache = false;
    if (_store.previousAppVersion != newAppVersion) {
      _store.previousAppVersion = newAppVersion;
      needToPurgeCache = true;
    }

    await Wakelock.enable();
    resourceManager.handleResources(_middle.listResources(), needToPurgeCache);
    await Wakelock.disable();
  }

  Future<bool> _handlePreviousResult() async {
    if (_doneRunning == null) {
      return resourceManager.resultManager
          .restoreResults(_store.previousResult, benchmarks);
    } else {
      await _store.deletePreviousResult();
      await resourceManager.resultManager.delete();
    }

    return false;
  }

  static Future<BenchmarkState> create(Store store) async {
    final result = BenchmarkState._(store, await BridgeIsolate.create());

    await result.resourceManager.initSystemPaths();
    result.configManager = ConfigManager(
        result.resourceManager.applicationDirectory,
        store.chosenConfigurationName,
        result.resourceManager);
    await result.resourceManager.loadBatchPresets();
    await result.resetConfig();
    await result.loadResources();
    return result;
  }

  Future<void> resetConfig({BenchmarksConfig? newConfig}) async {
    final config = newConfig ??
        await configManager.currentConfig ??
        configManager.defaultConfig;
    await configManager.setConfig(config);
    _store.chosenConfigurationName = config.name;
  }

  BenchmarkStateEnum get state {
    if (!resourceManager.done) return BenchmarkStateEnum.downloading;
    switch (_doneRunning) {
      case null:
        return BenchmarkStateEnum.waiting;
      case false:
        return _aborting
            ? BenchmarkStateEnum.aborting
            : _cooling
                ? BenchmarkStateEnum.cooldown
                : BenchmarkStateEnum.running;
      case true:
        return BenchmarkStateEnum.done;
    }
    throw StateError('unreachable');
  }

  List<BenchmarkJob> _getBenchmarkJobs() {
    final submissionMode = _store.submissionMode;
    final jobs = <BenchmarkJob>[];

    for (final benchmark in _middle.benchmarks) {
      if (!benchmark.config.active) continue;
      jobs.add(benchmark.createPerformanceJob());

      if (!submissionMode) continue;

      jobs.add(benchmark.createAccuracyJob());
    }
    return jobs;
  }

  void runBenchmarks() async {
    await reset();

    assert(resourceManager.done, 'Resource manager is not done.');
    assert(_doneRunning == null, '_doneRunning is not null');
    _doneRunning = false;

    // disable screen sleep when benchmarks is running
    await Wakelock.enable();

    final cooldown = _store.cooldown;
    final cooldownPause = FAST_MODE
        ? Duration(seconds: 1)
        : Duration(minutes: _store.cooldownPause);
    final jobs = _getBenchmarkJobs();

    var jobsDoneCounter = 0;
    var wasAccuracy = true;
    final results = <RunResult>[];

    for (final job in jobs) {
      if (_aborting) break;

      if (cooldown && !job.accuracyMode && !wasAccuracy) {
        _cooling = true;
        notifyListeners();
        await (_cooldownFuture = Future.delayed(cooldownPause));
        _cooling = false;
        notifyListeners();
      }
      if (_aborting) break;
      wasAccuracy = job.accuracyMode;

      final resultFuture = runBenchmark(
          job, backendInfo.settings.commonSetting, backendInfo.libPath);
      currentlyRunning = job.benchmark;
      runningProgress = '${(100 * (jobsDoneCounter / jobs.length)).round()}%';
      jobsDoneCounter++;
      notifyListeners();

      final result = await resultFuture;
      results.add(result);

      if (job.accuracyMode) {
        job.benchmark.accuracy =
            BenchmarkResult(0.0, result.accuracy, result.backendDescription);
      } else {
        job.benchmark.performance =
            BenchmarkResult(result.score, 'N/A', result.backendDescription);
      }
    }

    if (!_aborting) {
      await resourceManager.resultManager.writeResults(results);
      _store.previousResult =
          resourceManager.resultManager.serializeBriefResults(results);
    }

    currentlyRunning = null;
    _doneRunning = _aborting ? null : true;
    _aborting = false;
    notifyListeners();

    await Wakelock.disable();
  }

  Future<RunResult> runBenchmark(BenchmarkJob job,
      List<pb.Setting> commonSettings, String backendLibPath) async {
    final tmpDir = await getTemporaryDirectory();

    print(
        'Running ${job.benchmark.id} in ${job.accuracyMode ? 'accuracy' : 'performance'} mode...');
    final stopwatch = Stopwatch()..start();

    final runSettings = job.createRunSettings(
        resourceManager, commonSettings, backendLibPath, tmpDir);
    final result = await backendBridge.run(runSettings);
    final elapsed = stopwatch.elapsed;

    print('Benchmark result: $result, elapsed: $elapsed');
    return result;
  }

  Future<void> abortBenchmarks() async {
    if (_doneRunning == false) {
      _aborting = true;
      await CancelableOperation.fromFuture(_cooldownFuture).cancel();
      notifyListeners();
    }
  }

  Future<void> reset() async {
    final isPreviousResultUsed = await _handlePreviousResult();

    if (isPreviousResultUsed) {
      _doneRunning = true;
    } else {
      _middle.benchmarks.forEach(
          (benchmark) => benchmark.accuracy = benchmark.performance = null);
      _doneRunning = null;
    }

    _aborting = false;
  }
}
