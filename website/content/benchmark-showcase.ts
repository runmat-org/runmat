import type { BenchmarkShowcaseConfig } from "@/lib/marketing-benchmarks";

export const BENCHMARK_SHOWCASE_CONFIG: BenchmarkShowcaseConfig[] = [
  {
    caseId: "4k-image-processing",
    heroLabel: "4K Image Processing",
    description: "Denoise + Tone Mapping 4K Frames",
    deviceLabel: "Apple M2 Max",
    link: "/benchmarks/4k-image-processing",
    chart: {
      type: "bar",
      focusParam: 64,
      paramLabel: "Batch size B",
      includeImpls: ["python-numpy", "python-torch", "runmat"],
      highlightImpl: "runmat",
      baselineImpl: "python-numpy",
      labelOverrides: {
        "python-numpy": "Python NumPy",
        "python-torch": "Python PyTorch (GPU)",
        runmat: "RunMat (GPU)",
      },
    },
    stat: {
      compareImpl: "python-numpy",
      referenceImpl: "runmat",
      param: 64,
    },
  },
  {
    caseId: "monte-carlo-analysis",
    heroLabel: "Monte Carlo Analysis",
    description: "GBM Monte Carlo Simulation",
    deviceLabel: "Apple M2 Max",
    link: "/benchmarks/monte-carlo-analysis",
    chart: {
      type: "line",
      paramLabel: "Simulations (M)",
      includeImpls: ["python-numpy", "python-torch", "runmat"],
      highlightImpl: "runmat",
      baselineImpl: "python-numpy",
      labelOverrides: {
        "python-numpy": "Python NumPy",
        "python-torch": "Python PyTorch (GPU)",
        runmat: "RunMat (GPU)",
      },
    },
    stat: {
      compareImpl: "python-numpy",
      referenceImpl: "runmat",
      param: "max",
    },
  },
  {
    caseId: "monte-carlo-analysis",
    heroLabel: "Elementwise Matrix Math",
    description: "Elementwise Matrix Math",
    deviceLabel: "Apple M2 Max",
    link: "/benchmarks/general-math",
    chart: {
      type: "line",
      paramLabel: "Operations (M)",
      includeImpls: ["python-numpy", "python-torch", "runmat"],
      highlightImpl: "runmat",
      baselineImpl: "python-numpy",
      labelOverrides: {
        "python-numpy": "Python NumPy",
        "python-torch": "Python PyTorch (GPU)",
        runmat: "RunMat (GPU)",
      },
    },
    stat: {
      compareImpl: "python-numpy",
      referenceImpl: "runmat",
      param: "max",
    },
  },
];