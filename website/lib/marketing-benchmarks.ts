import { cache } from "react";
import path from "node:path";
import { promises as fs } from "node:fs";

import { BENCHMARK_SHOWCASE_CONFIG } from "@/content/benchmark-showcase";
import { formatNumber } from "@/lib/number-format";

const DEFAULT_BENCH_PATH =
  process.env.MARKETING_BENCH_PATH ??
  path.resolve(process.cwd(), "content", "benchmarks.json");

type ParamTarget = number | "max" | "min";

const DEFAULT_IMPL_LABELS: Record<string, string> = {
  runmat: "RunMat (GPU)",
  "python-torch": "PyTorch",
  "python-numpy": "NumPy",
  octave: "Octave",
};

interface ProviderInfo {
  backend?: string;
  device_id?: number;
  memory_bytes?: number;
  name?: string;
  vendor?: string;
}

type RawBenchmarkSuite = {
  suite?: {
    auto_offload_calibration?: {
      provider?: ProviderInfo;
    };
  };
  cases: BenchmarkCase[];
};

type BenchmarkCase = {
  id: string;
  label: string;
  sweep?: {
    param: string;
    values?: Array<number | string>;
  };
  results: BenchmarkResult[];
};

type BenchmarkResult = {
  impl: string;
  median_ms?: number;
  [key: string]: unknown;
};

export interface BenchmarkShowcaseConfig {
  caseId: string;
  heroLabel: string;
  description?: string;
  deviceLabel?: string;
  link?: string;
  chart: BenchmarkBarChartConfig | BenchmarkLineChartConfig;
  stat?: {
    referenceImpl?: string;
    compareImpl: string;
    param?: ParamTarget;
  };
}

interface BaseChartConfig {
  includeImpls?: string[];
  baselineImpl?: string;
  highlightImpl?: string;
  paramKey?: string;
  paramLabel?: string;
  labelOverrides?: Record<string, string>;
}

export interface BenchmarkBarChartConfig extends BaseChartConfig {
  type: "bar";
  focusParam?: number | string;
}

export interface BenchmarkLineChartConfig extends BaseChartConfig {
  type: "line";
}

export type BenchmarkChartData = BenchmarkBarChartData | BenchmarkLineChartData;

export interface BenchmarkBarChartData {
  type: "bar";
  baselineImpl: string;
  highlightImpl?: string;
  paramLabel?: string;
  paramValue?: number | string;
  paramValueLabel?: string;
  entries: BenchmarkBarEntry[];
}

interface BenchmarkBarEntry {
  impl: string;
  label: string;
  medianMs: number;
  speedup: number;
}

export interface BenchmarkLineChartData {
  type: "line";
  baselineImpl: string;
  highlightImpl?: string;
  paramKey: string;
  paramLabel?: string;
  series: BenchmarkLineSeries[];
}

interface BenchmarkLineSeries {
  impl: string;
  label: string;
  points: BenchmarkLinePoint[];
}

export interface BenchmarkLinePoint {
  param: number;
  rawParam: number | string;
  label: string;
  medianMs: number;
  speedup?: number;
}

export interface BenchmarkSlideStat {
  speedup: number;
  referenceLabel: string;
  compareLabel: string;
  paramLabel?: string;
  paramValueLabel?: string;
}

export interface BenchmarkShowcaseSlide {
  caseId: string;
  label: string;
  heroLabel: string;
  description?: string;
  deviceLabel?: string;
  link?: string;
  footnote?: string;
  chart: BenchmarkChartData;
  stat?: BenchmarkSlideStat;
  headlineRange?: string;
}

const readSuite = cache(async (): Promise<RawBenchmarkSuite> => {
  const buffer = await fs.readFile(DEFAULT_BENCH_PATH, "utf-8");
  const parsed = JSON.parse(buffer);
  if (!parsed?.cases) {
    throw new Error(`Benchmark suite file missing cases array: ${DEFAULT_BENCH_PATH}`);
  }
  return parsed as RawBenchmarkSuite;
});

export const getBenchmarkShowcaseSlides = cache(async (): Promise<BenchmarkShowcaseSlide[]> => {
  const suite = await readSuite();
  return BENCHMARK_SHOWCASE_CONFIG.map((config) => buildSlide(config, suite));
});

function buildSlide(config: BenchmarkShowcaseConfig, suite: RawBenchmarkSuite): BenchmarkShowcaseSlide {
  const targetCase = suite.cases.find((entry) => entry.id === config.caseId);
  if (!targetCase) {
    throw new Error(`Benchmark case "${config.caseId}" not found in marketing suite`);
  }

  const chart =
    config.chart.type === "bar"
      ? buildBarChart(targetCase, config.chart)
      : buildLineChart(targetCase, config.chart);

  return {
    caseId: targetCase.id,
    label: targetCase.label,
    heroLabel: config.heroLabel ?? targetCase.label,
    description: config.description,
    deviceLabel:
      config.deviceLabel ??
      formatDeviceLabel(suite.suite?.auto_offload_calibration?.provider) ??
      config.deviceLabel,
    link: config.link,
    chart,
    stat: config.stat ? deriveStat(chart, config.stat, config.chart.labelOverrides) : undefined,
    headlineRange: computeHeadlineRange(chart),
  };
}

function buildBarChart(caseData: BenchmarkCase, chartConfig: BenchmarkBarChartConfig): BenchmarkBarChartData {
  const baselineImpl = chartConfig.baselineImpl ?? "python-numpy";
  const highlightImpl = chartConfig.highlightImpl ?? "runmat";
  const paramKey = chartConfig.paramKey ?? caseData.sweep?.param;
  const includeImpls = chartConfig.includeImpls ?? collectImpls(caseData.results);

  const focusParam =
    chartConfig.focusParam ??
    (paramKey && caseData.sweep?.values && caseData.sweep.values.length
      ? caseData.sweep.values[caseData.sweep.values.length - 1]
      : undefined);

  const filtered = filterResults(caseData.results, includeImpls, paramKey, focusParam);

  const baselineEntry = filtered.find((entry) => entry.impl === baselineImpl);
  if (!baselineEntry?.median_ms) {
    throw new Error(
      `Baseline implementation "${baselineImpl}" missing median_ms for case "${caseData.id}".`
    );
  }

  const baselineMs = baselineEntry.median_ms;
  const entries: BenchmarkBarEntry[] = [];

  includeImpls.forEach((impl) => {
    const target = filtered.find((entry) => entry.impl === impl);
    if (!target?.median_ms) {
      return;
    }
    const label = labelForImpl(impl, chartConfig.labelOverrides);
    const speedup = baselineMs / target.median_ms;
    entries.push({
      impl,
      label,
      medianMs: target.median_ms,
      speedup,
    });
  });

  const paramValueLabel =
    focusParam !== undefined && paramKey
      ? `${chartConfig.paramLabel ?? paramKey} = ${formatNumber(focusParam)}`
      : undefined;

  return {
    type: "bar",
    baselineImpl,
    highlightImpl,
    paramLabel: chartConfig.paramLabel ?? paramKey,
    paramValue: focusParam,
    paramValueLabel,
    entries,
  };
}

function buildLineChart(caseData: BenchmarkCase, chartConfig: BenchmarkLineChartConfig): BenchmarkLineChartData {
  const paramKey = chartConfig.paramKey ?? caseData.sweep?.param;
  if (!paramKey) {
    throw new Error(`Line chart requires a sweep parameter for case "${caseData.id}".`);
  }

  const includeImpls = chartConfig.includeImpls ?? collectImpls(caseData.results);
  const baselineImpl = chartConfig.baselineImpl ?? "python-numpy";
  const highlightImpl = chartConfig.highlightImpl ?? "runmat";

  const grouped = new Map<string, BenchmarkLinePoint[]>();

  caseData.results.forEach((result) => {
    if (!includeImpls.includes(result.impl)) {
      return;
    }
    if (typeof result.median_ms !== "number" || Number.isNaN(result.median_ms)) {
      return;
    }
    const rawValue = result[paramKey];
    if (rawValue === undefined || rawValue === null) {
      return;
    }
    if (typeof rawValue !== "number" && typeof rawValue !== "string") {
      return;
    }
    const rawParam = rawValue;
    const paramValue = typeof rawParam === "number" ? rawParam : Number(rawParam);
    if (Number.isNaN(paramValue)) {
      return;
    }
    const points = grouped.get(result.impl) ?? [];
    points.push({
      param: paramValue,
      rawParam,
      label: formatNumber(rawParam),
      medianMs: result.median_ms,
    });
    grouped.set(result.impl, points);
  });

  const series: BenchmarkLineSeries[] = [];

  const baselineSeries = grouped.get(baselineImpl);
  const baselineMap = new Map<number, number>();
  baselineSeries?.forEach((point) => {
    baselineMap.set(point.param, point.medianMs);
  });

  grouped.forEach((points, impl) => {
    points.sort((a, b) => a.param - b.param);
    points.forEach((point) => {
      if (impl === baselineImpl) {
        point.speedup = 1;
        return;
      }
      const base = baselineMap.get(point.param);
      if (typeof base === "number" && base > 0) {
        point.speedup = base / point.medianMs;
      }
    });
    series.push({
      impl,
      label: labelForImpl(impl, chartConfig.labelOverrides),
      points,
    });
  });

  if (!series.length) {
    throw new Error(`No series data available for case "${caseData.id}".`);
  }

  return {
    type: "line",
    baselineImpl,
    highlightImpl,
    paramKey,
    paramLabel: chartConfig.paramLabel ?? paramKey,
    series,
  };
}

function deriveStat(
  chart: BenchmarkChartData,
  statConfig: NonNullable<BenchmarkShowcaseConfig["stat"]>,
  overrides?: Record<string, string>
): BenchmarkSlideStat | undefined {
  const referenceImpl = statConfig.referenceImpl ?? "runmat";
  const compareImpl = statConfig.compareImpl;

  if (chart.type === "bar") {
    const referenceEntry = chart.entries.find((entry) => entry.impl === referenceImpl);
    const compareEntry = chart.entries.find((entry) => entry.impl === compareImpl);
    if (!referenceEntry || !compareEntry) {
      return undefined;
    }
    const speedup = compareEntry.medianMs / referenceEntry.medianMs;
    return {
      speedup,
      referenceLabel: labelForImpl(referenceImpl, overrides),
      compareLabel: labelForImpl(compareImpl, overrides),
      paramLabel: chart.paramLabel,
      paramValueLabel: chart.paramValueLabel,
    };
  }

  const referenceSeries = chart.series.find((series) => series.impl === referenceImpl);
  const compareSeries = chart.series.find((series) => series.impl === compareImpl);
  if (!referenceSeries || !compareSeries) {
    return undefined;
  }

  const targetParamValue = resolveParamTarget(statConfig.param, referenceSeries.points);
  if (targetParamValue === undefined) {
    return undefined;
  }

  const referencePoint = referenceSeries.points.find((pt) => pt.param === targetParamValue);
  const comparePoint = compareSeries.points.find((pt) => pt.param === targetParamValue);
  if (!referencePoint || !comparePoint) {
    return undefined;
  }

  const speedup = comparePoint.medianMs / referencePoint.medianMs;

  return {
    speedup,
    referenceLabel: referenceSeries.label,
    compareLabel: compareSeries.label,
    paramLabel: chart.paramLabel,
    paramValueLabel: `${chart.paramLabel ?? chart.paramKey} = ${formatNumber(referencePoint.rawParam)}`,
  };
}

function resolveParamTarget(target: ParamTarget | undefined, points: BenchmarkLinePoint[]): number | undefined {
  if (!points.length) {
    return undefined;
  }
  if (typeof target === "number") {
    return target;
  }
  if (target === "min") {
    return points.reduce((min, point) => Math.min(min, point.param), Number.POSITIVE_INFINITY);
  }
  // Default and "max"
  return points.reduce((max, point) => Math.max(max, point.param), Number.NEGATIVE_INFINITY);
}

function collectImpls(results: BenchmarkResult[]): string[] {
  const set = new Set<string>();
  results.forEach((result) => {
    if (result.impl) {
      set.add(result.impl);
    }
  });
  return Array.from(set);
}

function filterResults(
  results: BenchmarkResult[],
  includeImpls: string[],
  paramKey?: string,
  focusParam?: number | string
): BenchmarkResult[] {
  if (!paramKey || focusParam === undefined || focusParam === null) {
    return results.filter((result) => includeImpls.includes(result.impl));
  }

  return results.filter((result) => {
    if (!includeImpls.includes(result.impl)) {
      return false;
    }
    return normalizeParam(result[paramKey]) === normalizeParam(focusParam);
  });
}

function normalizeParam(value: unknown): number | string | undefined {
  if (typeof value === "number" || typeof value === "string") {
    return value;
  }
  return undefined;
}

function labelForImpl(impl: string, overrides?: Record<string, string>): string {
  if (overrides?.[impl]) {
    return overrides[impl];
  }
  if (DEFAULT_IMPL_LABELS[impl]) {
    return DEFAULT_IMPL_LABELS[impl];
  }
  return impl
    .split(/[-_]/)
    .map((chunk) => chunk.charAt(0).toUpperCase() + chunk.slice(1))
    .join(" ");
}

function formatDeviceLabel(provider?: ProviderInfo): string | undefined {
  if (!provider?.name) {
    return undefined;
  }
  const parts = [provider.name];
  if (provider.memory_bytes) {
    const gb = provider.memory_bytes / 1024 / 1024 / 1024;
    parts.push(`${gb.toFixed(0)} GB`);
  }
  return parts.join(" · ");
}

function computeHeadlineRange(chart: BenchmarkChartData): string | undefined {
  let minSpeedup = Number.POSITIVE_INFINITY;
  let maxSpeedup = 0;

  if (chart.type === "bar") {
    chart.entries.forEach((entry) => {
      if (entry.impl === chart.baselineImpl) {
        return;
      }
      minSpeedup = Math.min(minSpeedup, entry.speedup);
      maxSpeedup = Math.max(maxSpeedup, entry.speedup);
    });
  } else {
    chart.series.forEach((series) => {
      if (series.impl === chart.baselineImpl) {
        return;
      }
      series.points.forEach((point) => {
        if (typeof point.speedup === "number" && point.speedup > 0) {
          minSpeedup = Math.min(minSpeedup, point.speedup);
          maxSpeedup = Math.max(maxSpeedup, point.speedup);
        }
      });
    });
  }

  if (!Number.isFinite(minSpeedup) || maxSpeedup <= 0) {
    return undefined;
  }

  const format = (value: number) => (value >= 9 ? value.toFixed(0) : value.toFixed(1));
  if (Math.abs(maxSpeedup - minSpeedup) < 0.1) {
    return `${format(maxSpeedup)}× faster`;
  }
  return `${format(minSpeedup)}×–${format(maxSpeedup)}× faster`;
}


