use runmat_time::Instant;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use log::info;
#[cfg(feature = "wgpu")]
use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};
#[cfg(feature = "wgpu")]
use runmat_accelerate::provider_cache_stats;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, HostTensorOwned, HostTensorView, ReductionFlavor,
};
use serde::Serialize;
#[cfg(feature = "wgpu")]
use wgpu::PowerPreference;
const VALUE_TOLERANCE: f64 = 1e-5;
const VALUE_REL_TOLERANCE: f64 = 1e-4;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut output_path = PathBuf::from("wgpu_profile.json");
    let mut do_reduce_sweep = false;
    let mut only_reduce_sweep = false;
    let mut quick = false;
    let mut sweep_max_secs: Option<u64> = None;
    let mut sweep_first = false;
    let mut wg_override: Option<u32> = None;
    let mut kernel_probe = false;
    let mut do_fused = false;
    let mut do_fused_wgsl = false;
    if let Some(pos) = args.iter().position(|a| a == "--output") {
        if pos + 1 < args.len() {
            output_path = PathBuf::from(args[pos + 1].clone());
        }
    }
    if args.iter().any(|a| a == "--reduce-sweep") {
        do_reduce_sweep = true;
    }
    if args.iter().any(|a| a == "--only-reduce-sweep") {
        only_reduce_sweep = true;
    }
    if args.iter().any(|a| a == "--sweep-first") {
        sweep_first = true;
    }
    if args.iter().any(|a| a == "--kernel-probe") {
        kernel_probe = true;
    }
    if args.iter().any(|a| a == "--fused-sweep") {
        do_fused = true;
    }
    if args.iter().any(|a| a == "--fused-wgsl") {
        do_fused_wgsl = true;
    }
    if args.iter().any(|a| a == "--quick") {
        quick = true;
    }
    if let Some(pos) = args.iter().position(|a| a == "--sweep-max-secs") {
        if pos + 1 < args.len() {
            if let Ok(v) = args[pos + 1].parse::<u64>() {
                sweep_max_secs = Some(v);
            }
        }
    }
    if let Some(pos) = args.iter().position(|a| a == "--wg-override") {
        if pos + 1 < args.len() {
            if let Ok(v) = args[pos + 1].parse::<u32>() {
                wg_override = Some(v);
            }
        }
    }

    if do_fused_wgsl {
        std::env::set_var("RUNMAT_TWO_PASS_THRESHOLD", "1000000000");
    }
    setup_wgpu_provider()?;
    let provider = runmat_accelerate_api::provider()
        .ok_or_else(|| anyhow!("No acceleration provider registered"))?;
    #[cfg(feature = "wgpu")]
    {
        if kernel_probe {
            info!("kernel-probe: compiling minimal kernels to triage Metal pipeline creation");
            let prov = runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider()?;
            if let Some(p) = prov {
                // no bindings
                let _ = p.try_compile_kernel("noop-1", "@compute @workgroup_size(1) fn main() {}");
                // with bindings, no barrier
                let src1 = "struct T{data:array<f32>}; @group(0)@binding(0)var<storage,read> a:T; @group(0)@binding(1)var<storage,read_write> o:T; @compute @workgroup_size(1) fn main(){ o.data[0u]=a.data[0u]; }";
                let _ = p.probe_kernel_with_buffers("acc-no-barrier", src1, 1);
                // wg=64
                let src2 = "struct T{data:array<f32>}; @group(0)@binding(0)var<storage,read> a:T; @group(0)@binding(1)var<storage,read_write> o:T; @compute @workgroup_size(64) fn main(@builtin(local_invocation_id) lid:vec3<u32>){ if(lid.x==0u){ o.data[0u]=a.data[0u]; }}";
                let _ = p.probe_kernel_with_buffers("acc-wg64", src2, 64);
                // tile+barrier
                let src3 = "struct T{data:array<f32>}; var<workgroup> t:array<f32,64>; @group(0)@binding(0)var<storage,read> a:T; @group(0)@binding(1)var<storage,read_write> o:T; @compute @workgroup_size(64) fn main(@builtin(local_invocation_id) lid:vec3<u32>){ t[lid.x]=1.0; workgroupBarrier(); if(lid.x==0u){ o.data[0u]=a.data[0u]+t[0u]; }}";
                let _ = p.probe_kernel_with_buffers("acc-tile-barrier", src3, 64);
            }
            info!("kernel-probe: done");
            return Ok(());
        }
    }
    #[cfg(not(feature = "wgpu"))]
    {
        if kernel_probe {
            info!("kernel-probe requested but binary built without 'wgpu' feature; skipping");
            return Ok(());
        }
    }

    let info = provider.device_info_struct();
    info!(
        "Using WGPU device '{}' (backend {:?})",
        info.name, info.backend
    );

    let mut reports = Vec::new();

    if !only_reduce_sweep {
        reports.push(run_elementwise_case(
            provider,
            "elementwise_add_256",
            ElementwiseOp::Add,
            256,
            256,
            3,
        )?);
        reports.push(run_elementwise_case(
            provider,
            "elementwise_add_4096",
            ElementwiseOp::Add,
            4096,
            4096,
            3,
        )?);
        reports.push(run_elementwise_case(
            provider,
            "elementwise_mul_engineering",
            ElementwiseOp::Mul,
            2048,
            128,
            3,
        )?);

        reports.push(run_matmul_case(provider, "matmul_256", 256, 256, 256, 3)?);
        reports.push(run_matmul_case(
            provider,
            "matmul_1024",
            1024,
            1024,
            1024,
            3,
        )?);
        reports.push(run_matmul_case(
            provider,
            "matmul_tall_skinny",
            2048,
            64,
            256,
            3,
        )?);
        reports.push(run_matmul_case(
            provider,
            "matmul_wide_short",
            128,
            2048,
            128,
            3,
        )?);

        reports.push(run_transpose_case(
            provider,
            "transpose_4096",
            4096,
            4096,
            3,
        )?);

        reports.push(run_reduction_case(
            provider,
            "reduction_sum_all",
            ReductionKind::SumAll,
            1_048_576,
            1,
            3,
        )?);
        reports.push(run_reduction_case(
            provider,
            "reduction_sum_row",
            ReductionKind::SumDim(1),
            1024,
            256,
            3,
        )?);
        reports.push(run_reduction_case(
            provider,
            "reduction_mean_col",
            ReductionKind::MeanDim(2),
            1024,
            256,
            3,
        )?);

        reports.push(run_minmax_case(
            provider,
            "min_indices_dim1",
            MinMaxKind::Min,
            1024,
            256,
            1,
            3,
        )?);
        reports.push(run_minmax_case(
            provider,
            "max_indices_dim2",
            MinMaxKind::Max,
            1024,
            256,
            2,
            3,
        )?);

        reports.push(run_composite_atda_case(
            provider,
            "composite_atda",
            1536,
            256,
            3,
        )?);
    }

    // Optionally append reduction sweep reports
    if do_reduce_sweep {
        info!(
            "starting reduction sweep (quick={}, max_secs={:?})",
            quick, sweep_max_secs
        );
        let sweep_reports =
            run_reduction_sweep(provider, quick, sweep_max_secs, sweep_first, wg_override)?;
        info!("reduction sweep produced {} reports", sweep_reports.len());
        reports.extend(sweep_reports);

        // Cache stats after reduction sweep
        #[cfg(feature = "wgpu")]
        {
            if let Some((hits, misses)) = provider_cache_stats() {
                info!(
                    "fused pipeline cache stats after sweep - hits: {}, misses: {}",
                    hits, misses
                );
            }
        }
    }

    if do_fused {
        info!("starting fused elementwise→reduction sweep");
        let fused_reports = run_fused_elementwise_reduction_sweep(
            provider,
            quick,
            sweep_max_secs,
            sweep_first,
            wg_override,
        )?;
        info!("fused sweep produced {} reports", fused_reports.len());
        reports.extend(fused_reports);
    }

    if do_fused_wgsl {
        info!("starting fused WGSL (single-kernel sin→sum) sweep");
        let fused_wgsl_reports =
            run_fused_wgsl_sweep(provider, quick, sweep_max_secs, sweep_first, wg_override)?;
        info!(
            "fused WGSL sweep produced {} reports",
            fused_wgsl_reports.len()
        );
        reports.extend(fused_wgsl_reports);
    }

    let mut file = File::create(&output_path)
        .with_context(|| format!("Failed to create {:?}", output_path))?;
    serde_json::to_writer_pretty(&mut file, &reports)
        .with_context(|| format!("Failed to write benchmark results to {:?}", output_path))?;
    file.write_all(b"\n")?;

    info!(
        "WGPU benchmark suite complete; results written to {:?}",
        output_path
    );

    Ok(())
}
fn run_fused_elementwise_reduction_sweep(
    provider: &'static dyn AccelProvider,
    quick: bool,
    sweep_max_secs: Option<u64>,
    sweep_first: bool,
    wg_override: Option<u32>,
) -> Result<Vec<CaseReport>> {
    let sizes_quick = [256usize, 2048];
    let slices_quick = [1usize, 64];
    let wgs_quick = [128u32, 256];

    let sizes_full = [64usize, 128, 256, 512, 1024, 2048, 4096, 8192];
    let slices_full = [1usize, 8, 64, 256];
    let wgs_full = [128u32, 256, 512];

    let sizes: &[usize] = if quick { &sizes_quick } else { &sizes_full };
    let slices: &[usize] = if quick { &slices_quick } else { &slices_full };
    let wgs: &[u32] = if quick { &wgs_quick } else { &wgs_full };

    let mut out = Vec::new();
    let start = Instant::now();
    'outer: for &wg in wgs {
        for &rows in sizes {
            for &cols in slices {
                if let Some(maxs) = sweep_max_secs {
                    if start.elapsed() > Duration::from_secs(maxs) {
                        info!(
                            "fused-sweep: time budget hit ({}s), returning {} reports",
                            maxs,
                            out.len()
                        );
                        return Ok(out);
                    }
                }
                let iters = if quick { 3 } else { 6 };
                let wg_use = wg_override.unwrap_or(wg);
                info!(
                    "fused-sweep enqueue rows={} cols={} wg={} iters={}",
                    rows, cols, wg_use, iters
                );
                out.push(run_fused_elementwise_reduction_case(
                    provider, rows, cols, wg_use, iters,
                )?);
                if sweep_first {
                    break 'outer;
                }
            }
        }
    }
    Ok(out)
}

fn run_fused_elementwise_reduction_case(
    provider: &'static dyn AccelProvider,
    rows: usize,
    cols: usize,
    _wg_size: u32,
    iterations: usize,
) -> Result<CaseReport> {
    info!(
        "fused case start rows={} cols={} iters={}",
        rows, cols, iterations
    );
    // Build inputs on host and upload once per iter to measure full path
    let base = Matrix::generate(rows, cols, 0.45, -0.25);
    let mut upload_stats = TimeStats::new();
    let mut compute_stats = TimeStats::new();
    let mut download_stats = TimeStats::new();
    let mut cpu_stats = TimeStats::new();

    for iter in 0..iterations {
        let warmup = iter == 0;
        let (handle_a, upload_time) = upload_matrix(provider, &base)?;

        // CPU fused reference: sin(A) then sum along dim=1
        let cpu_start = Instant::now();
        let cpu_sin = Matrix {
            rows,
            cols,
            data: base.data.iter().map(|v| v.sin()).collect(),
        };
        let _cpu_ref = cpu_reduce_sum_dim(&cpu_sin, 1);
        let cpu_time = cpu_start.elapsed();

        // GPU fused: unary_sin then reduce_sum_dim
        let compute_start = Instant::now();
        let sin_handle = provider.unary_sin(&handle_a)?;
        let reduced_handle = provider.reduce_sum_dim(&sin_handle, 0)?; // dim=1 -> index 0
        let compute_time = compute_start.elapsed();

        let (_out_matrix, download_time) = download_matrix(provider, &reduced_handle)?;
        provider.free(&handle_a)?;
        provider.free(&sin_handle)?;
        provider.free(&reduced_handle)?;

        if !warmup {
            upload_stats.record(upload_time);
            compute_stats.record(compute_time);
            download_stats.record(download_time);
            cpu_stats.record(cpu_time);
        }
    }

    upload_stats.finalize();
    compute_stats.finalize();
    download_stats.finalize();

    let samples = iterations.saturating_sub(1).max(1);
    let total_stats = TimeStats {
        total: upload_stats.total + compute_stats.total + download_stats.total,
        min: upload_stats
            .min
            .min(compute_stats.min.min(download_stats.min)),
        max: upload_stats.max + compute_stats.max + download_stats.max,
    };

    Ok(CaseReport {
        name: format!("fused_sin_sum_{}x{}", rows, cols),
        category: "fused_elementwise_reduction".to_string(),
        detail: format!("sin -> sum rows {}x{}", rows, cols),
        input_shapes: vec![(rows, cols)],
        iterations: samples,
        upload_ms: DurationSummary::from_stats(&upload_stats, samples),
        compute_ms: DurationSummary::from_stats(&compute_stats, samples),
        download_ms: DurationSummary::from_stats(&download_stats, samples),
        total_ms: DurationSummary::from_stats(&total_stats, samples),
        notes: vec![],
        cpu_ms: Some(DurationSummary::from_stats(&cpu_stats, samples)),
    })
}

#[cfg(not(feature = "wgpu"))]
fn setup_wgpu_provider() -> Result<()> {
    Err(anyhow!("wgpu_profile requires the 'wgpu' feature"))
}

#[cfg(feature = "wgpu")]
fn setup_wgpu_provider() -> Result<()> {
    let high_perf = WgpuProviderOptions {
        power_preference: PowerPreference::HighPerformance,
        force_fallback_adapter: false,
    };

    match provider::register_wgpu_provider(high_perf) {
        Ok(_) => Ok(()),
        Err(err_hp) => {
            let fallback_opts = WgpuProviderOptions {
                power_preference: PowerPreference::LowPower,
                force_fallback_adapter: true,
            };
            match provider::register_wgpu_provider(fallback_opts) {
                Ok(_) => {
                    info!("Using fallback GPU adapter after {:?}", err_hp);
                    Ok(())
                }
                Err(err_fb) => Err(anyhow!(
                    "Failed to initialize WGPU provider: {:?}; fallback attempt: {:?}",
                    err_hp,
                    err_fb
                )),
            }
        }
    }
}

#[derive(Clone, Copy, Serialize, Debug)]
struct DurationSummary {
    avg_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

impl DurationSummary {
    fn from_stats(stats: &TimeStats, samples: usize) -> Self {
        DurationSummary {
            avg_ms: stats.total.as_secs_f64() * 1000.0 / samples as f64,
            min_ms: stats.min.as_secs_f64() * 1000.0,
            max_ms: stats.max.as_secs_f64() * 1000.0,
        }
    }
}

fn run_fused_wgsl_sweep(
    provider: &'static dyn AccelProvider,
    quick: bool,
    sweep_max_secs: Option<u64>,
    sweep_first: bool,
    wg_override: Option<u32>,
) -> Result<Vec<CaseReport>> {
    let sizes_quick = [256usize, 2048];
    let slices_quick = [1usize, 64];
    let wgs_quick = [128u32, 256];

    let sizes_full = [64usize, 128, 256, 512, 1024, 2048, 4096, 8192];
    let slices_full = [1usize, 8, 64, 256];
    let wgs_full = [128u32, 256, 512];

    let sizes: &[usize] = if quick { &sizes_quick } else { &sizes_full };
    let slices: &[usize] = if quick { &slices_quick } else { &slices_full };
    let wgs: &[u32] = if quick { &wgs_quick } else { &wgs_full };

    let mut out = Vec::new();
    let start = Instant::now();
    'outer: for &wg in wgs {
        for &rows in sizes {
            for &cols in slices {
                if let Some(maxs) = sweep_max_secs {
                    if start.elapsed() > Duration::from_secs(maxs) {
                        info!(
                            "fused-wgsl: time budget hit ({}s), returning {} reports",
                            maxs,
                            out.len()
                        );
                        return Ok(out);
                    }
                }
                let iters = if quick { 3 } else { 6 };
                let wg_use = wg_override.unwrap_or(wg);
                info!(
                    "fused-wgsl enqueue rows={} cols={} wg={} iters={}",
                    rows, cols, wg_use, iters
                );
                out.push(run_fused_wgsl_case(provider, rows, cols, wg_use, iters)?);
                if sweep_first {
                    break 'outer;
                }
            }
        }
    }
    Ok(out)
}

fn run_fused_wgsl_case(
    provider: &'static dyn AccelProvider,
    rows: usize,
    cols: usize,
    wg_size: u32,
    iterations: usize,
) -> Result<CaseReport> {
    let matrix = Matrix::generate(rows, cols, 0.45, -0.25);
    // CPU reference: sin then sum along dim=1 (rows)
    let mut upload_stats = TimeStats::new();
    let mut compute_stats = TimeStats::new();
    let mut download_stats = TimeStats::new();
    let mut cpu_stats = TimeStats::new();

    for iter in 0..iterations {
        let warmup = iter == 0;
        let (handle_matrix, upload_time) = upload_matrix(provider, &matrix)?;

        let cpu_start = Instant::now();
        let cpu_sin = Matrix {
            rows,
            cols,
            data: matrix.data.iter().map(|v| v.sin()).collect(),
        };
        let _cpu_ref = cpu_reduce_sum_dim(&cpu_sin, 1);
        let cpu_time = cpu_start.elapsed();

        // WGSL single-kernel: apply sin then reduce along rows per column
        // Column-major indexing
        let wg = wg_size;
        let shader = format!(
            "struct Tensor {{ data: array<f32> }};\nstruct MParams {{ nrows:u32,ncols:u32,ld:u32,flags:u32 }}\n@group(0) @binding(0) var<storage,read> input0: Tensor;\n@group(0) @binding(1) var<storage,read_write> output: Tensor;\n@group(0) @binding(2) var<uniform> params: MParams;\n@compute @workgroup_size({wg})\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{ let slice = wid.x; if (slice >= params.ncols) {{ return; }} var acc: f32 = 0.0; var r: u32 = 0u; loop {{ if (r >= params.nrows) {{ break; }} let v = input0.data[(slice * params.ld) + r]; acc = acc + sin(v); r += 1u; }} if (lid.x == 0u) {{ output.data[slice] = acc; }} }}",
            wg = wg
        );

        let compute_start = Instant::now();
        let _ = provider.fused_reduction(
            &shader,
            std::slice::from_ref(&handle_matrix),
            &[cols],
            rows,
            cols,
            wg_size,
            ReductionFlavor::Sum,
        )?;
        let compute_time = compute_start.elapsed();
        let (_out_matrix, download_time) = download_matrix(provider, &handle_matrix)?;
        provider.free(&handle_matrix)?;

        if !warmup {
            upload_stats.record(upload_time);
            compute_stats.record(compute_time);
            download_stats.record(download_time);
            cpu_stats.record(cpu_time);
        }
    }

    upload_stats.finalize();
    compute_stats.finalize();
    download_stats.finalize();

    let samples = iterations.saturating_sub(1).max(1);
    let total_stats = TimeStats {
        total: upload_stats.total + compute_stats.total + download_stats.total,
        min: upload_stats
            .min
            .min(compute_stats.min.min(download_stats.min)),
        max: upload_stats.max + compute_stats.max + download_stats.max,
    };

    Ok(CaseReport {
        name: format!("fused_wgsl_sin_sum_{}x{}_wg{}", rows, cols, wg_size),
        category: "fused_wgsl".to_string(),
        detail: format!(
            "single-kernel WGSL sin→sum rows {}x{} wg {}",
            rows, cols, wg_size
        ),
        input_shapes: vec![(rows, cols)],
        iterations: samples,
        upload_ms: DurationSummary::from_stats(&upload_stats, samples),
        compute_ms: DurationSummary::from_stats(&compute_stats, samples),
        download_ms: DurationSummary::from_stats(&download_stats, samples),
        total_ms: DurationSummary::from_stats(&total_stats, samples),
        notes: vec![],
        cpu_ms: Some(DurationSummary::from_stats(&cpu_stats, samples)),
    })
}

#[derive(Default)]
struct TimeStats {
    total: Duration,
    min: Duration,
    max: Duration,
}

impl TimeStats {
    fn new() -> Self {
        TimeStats {
            total: Duration::ZERO,
            min: Duration::MAX,
            max: Duration::ZERO,
        }
    }

    fn record(&mut self, duration: Duration) {
        self.total += duration;
        if duration < self.min {
            self.min = duration;
        }
        if duration > self.max {
            self.max = duration;
        }
    }

    fn finalize(&mut self) {
        if self.min == Duration::MAX {
            self.min = Duration::ZERO;
        }
    }
}

#[derive(Serialize, Debug)]
struct CaseReport {
    name: String,
    category: String,
    detail: String,
    input_shapes: Vec<(usize, usize)>,
    iterations: usize,
    upload_ms: DurationSummary,
    compute_ms: DurationSummary,
    download_ms: DurationSummary,
    total_ms: DurationSummary,
    notes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cpu_ms: Option<DurationSummary>,
}

#[derive(Clone, Copy, Debug)]
enum ElementwiseOp {
    Add,
    Mul,
}

#[derive(Clone, Copy, Debug)]
enum ReductionKind {
    SumAll,
    SumDim(usize),
    MeanDim(usize),
}

#[derive(Clone, Copy, Debug)]
enum MinMaxKind {
    Min,
    Max,
}

fn run_elementwise_case(
    provider: &'static dyn AccelProvider,
    name: &str,
    op: ElementwiseOp,
    rows: usize,
    cols: usize,
    iterations: usize,
) -> Result<CaseReport> {
    let matrix_a = Matrix::generate(rows, cols, 1.0, 0.5);
    let matrix_b = match op {
        ElementwiseOp::Add => Matrix::generate(rows, cols, -0.4, 0.3),
        ElementwiseOp::Mul => Matrix::generate(rows, cols, 0.8, -0.2),
    };

    generic_single_output_case(
        provider,
        name,
        "elementwise",
        format!("{:?} {}x{}", op, rows, cols),
        vec![matrix_a, matrix_b],
        iterations,
        VALUE_TOLERANCE,
        move |prov, handles| match op {
            ElementwiseOp::Add => prov.elem_add(&handles[0], &handles[1]),
            ElementwiseOp::Mul => prov.elem_mul(&handles[0], &handles[1]),
        },
        move |inputs| cpu_elementwise(&inputs[0], &inputs[1], op),
    )
}

fn run_matmul_case(
    provider: &'static dyn AccelProvider,
    name: &str,
    m: usize,
    k: usize,
    n: usize,
    iterations: usize,
) -> Result<CaseReport> {
    let matrix_a = Matrix::generate(m, k, 0.7, -0.1);
    let matrix_b = Matrix::generate(k, n, -0.6, 0.15);

    generic_single_output_case(
        provider,
        name,
        "matmul",
        format!("{}x{} * {}x{}", m, k, k, n),
        vec![matrix_a, matrix_b],
        iterations,
        VALUE_TOLERANCE,
        move |prov, handles| prov.matmul(&handles[0], &handles[1]),
        move |inputs| cpu_matmul(&inputs[0], &inputs[1]),
    )
}

fn run_transpose_case(
    provider: &'static dyn AccelProvider,
    name: &str,
    rows: usize,
    cols: usize,
    iterations: usize,
) -> Result<CaseReport> {
    let matrix = Matrix::generate(rows, cols, 0.3, 0.9);

    generic_single_output_case(
        provider,
        name,
        "transpose",
        format!("{}x{}", rows, cols),
        vec![matrix],
        iterations,
        VALUE_TOLERANCE,
        move |prov, handles| prov.transpose(&handles[0]),
        move |inputs| cpu_transpose(&inputs[0]),
    )
}

fn run_reduction_case(
    provider: &'static dyn AccelProvider,
    name: &str,
    kind: ReductionKind,
    rows: usize,
    cols: usize,
    iterations: usize,
) -> Result<CaseReport> {
    let matrix = Matrix::generate(rows, cols, 0.45, -0.25);

    generic_single_output_case(
        provider,
        name,
        "reduction",
        format!("{:?} {}x{}", kind, rows, cols),
        vec![matrix],
        iterations,
        VALUE_TOLERANCE,
        move |prov, handles| match kind {
            ReductionKind::SumAll => prov.reduce_sum(&handles[0]),
            ReductionKind::SumDim(dim) => prov.reduce_sum_dim(&handles[0], dim - 1),
            ReductionKind::MeanDim(dim) => prov.reduce_mean_dim(&handles[0], dim - 1),
        },
        move |inputs| match kind {
            ReductionKind::SumAll => Matrix::from_scalar(cpu_reduce_sum_all(&inputs[0])),
            ReductionKind::SumDim(dim) => cpu_reduce_sum_dim(&inputs[0], dim),
            ReductionKind::MeanDim(dim) => cpu_reduce_mean_dim(&inputs[0], dim),
        },
    )
}

fn run_reduction_sweep_case(
    provider: &'static dyn AccelProvider,
    rows: usize,
    cols: usize,
    wg_size: u32,
    iterations: usize,
) -> Result<CaseReport> {
    info!(
        "reduce-sweep case start rows={} cols={} wg={} iters={}",
        rows, cols, wg_size, iterations
    );
    let matrix = Matrix::generate(rows, cols, 0.45, -0.25);
    // Only timing; compute a reference to keep structure similar
    let _cpu_result = cpu_reduce_sum_dim(&matrix, 1);
    let mut upload_stats = TimeStats::new();
    let mut compute_stats = TimeStats::new();
    let mut download_stats = TimeStats::new();
    let mut cpu_stats = TimeStats::new();

    for iter in 0..iterations {
        let warmup = iter == 0;
        info!(
            "  iter {} warmup={} (rows={} cols={} wg={})",
            iter, warmup, rows, cols, wg_size
        );
        let (handle_matrix, upload_time) = upload_matrix(provider, &matrix)?;
        info!(
            "    upload done: {:.3} ms",
            upload_time.as_secs_f64() * 1000.0
        );

        // Dispatch via fused_reduction; we directly use provider for fair timing.
        // Minimal single-pass WGSL to test pipeline creation on Metal: no workgroup memory/barriers.
        // Correctness is not validated in this sweep; we just need a compilable kernel.
        let wg = wg_size;
        let shader = format!(
            "struct Tensor {{ data: array<f32> }};\nstruct MParams {{ nrows:u32,ncols:u32,ld:u32,flags:u32 }}\n@group(0) @binding(0) var<storage,read> input0: Tensor;\n@group(0) @binding(1) var<storage,read_write> output: Tensor;\n@group(0) @binding(2) var<uniform> params: MParams;\n@compute @workgroup_size({wg})\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{ let slice = wid.x; if (slice >= params.ncols) {{ return; }} var acc: f32 = 0.0; var r: u32 = 0u; loop {{ if (r >= params.nrows) {{ break; }} let v = input0.data[(slice * params.ld) + r]; acc = acc + v; r += 1u; }} if (lid.x == 0u) {{ output.data[slice] = acc; }} }}",
            wg = wg
        );
        // GPU compute timing
        let compute_start = Instant::now();
        let _ = provider.fused_reduction(
            &shader,
            std::slice::from_ref(&handle_matrix),
            &[cols],
            rows,
            cols,
            wg_size,
            ReductionFlavor::Sum,
        )?;
        let compute_time = compute_start.elapsed();
        info!(
            "    compute done: {:.3} ms",
            compute_time.as_secs_f64() * 1000.0
        );
        let (_, download_time) = download_matrix(provider, &handle_matrix)?; // just to measure path; not used
        provider.free(&handle_matrix)?;
        info!(
            "    download done: {:.3} ms",
            download_time.as_secs_f64() * 1000.0
        );

        // CPU timing (sum along dim=1 rows -> 1 x cols)
        let cpu_start = Instant::now();
        let _cpu_ref = cpu_reduce_sum_dim(&matrix, 1);
        let cpu_time = cpu_start.elapsed();
        info!(
            "    cpu sum(dim=1) done: {:.3} ms",
            cpu_time.as_secs_f64() * 1000.0
        );

        if !warmup {
            upload_stats.record(upload_time);
            compute_stats.record(compute_time);
            download_stats.record(download_time);
            cpu_stats.record(cpu_time);
        }
    }

    upload_stats.finalize();
    compute_stats.finalize();
    download_stats.finalize();

    let samples = iterations.saturating_sub(1).max(1);
    let total_stats = TimeStats {
        total: upload_stats.total + compute_stats.total + download_stats.total,
        min: upload_stats
            .min
            .min(compute_stats.min.min(download_stats.min)),
        max: upload_stats.max + compute_stats.max + download_stats.max,
    };

    Ok(CaseReport {
        name: format!("reduction_sweep_{}x{}_wg{}", rows, cols, wg_size),
        category: "reduction_sweep".to_string(),
        detail: format!("sum rows {}x{} wg {}", rows, cols, wg_size),
        input_shapes: vec![(rows, cols)],
        iterations: samples,
        upload_ms: DurationSummary::from_stats(&upload_stats, samples),
        compute_ms: DurationSummary::from_stats(&compute_stats, samples),
        download_ms: DurationSummary::from_stats(&download_stats, samples),
        total_ms: DurationSummary::from_stats(&total_stats, samples),
        notes: vec![],
        cpu_ms: Some(DurationSummary::from_stats(&cpu_stats, samples)),
    })
}

fn run_reduction_sweep(
    provider: &'static dyn AccelProvider,
    quick: bool,
    sweep_max_secs: Option<u64>,
    sweep_first: bool,
    wg_override: Option<u32>,
) -> Result<Vec<CaseReport>> {
    let sizes_quick = [256usize, 2048];
    let slices_quick = [1usize, 64];
    let wgs_quick = [128u32, 256];

    let sizes_full = [64usize, 128, 256, 512, 1024, 2048, 4096, 8192];
    let slices_full = [1usize, 8, 64, 256];
    let wgs_full = [128u32, 256, 512];

    let sizes: &[usize] = if quick { &sizes_quick } else { &sizes_full };
    let slices: &[usize] = if quick { &slices_quick } else { &slices_full };
    let wgs: &[u32] = if quick { &wgs_quick } else { &wgs_full };

    let mut out = Vec::new();
    let start = Instant::now();
    'outer: for &wg in wgs {
        for &rows in sizes {
            for &cols in slices {
                if let Some(maxs) = sweep_max_secs {
                    if start.elapsed() > Duration::from_secs(maxs) {
                        info!(
                            "reduce-sweep: time budget hit ({}s), returning {} reports",
                            maxs,
                            out.len()
                        );
                        return Ok(out);
                    }
                }
                // 3 iterations in quick mode (1 warmup + 2 samples), 6 in full
                let iters = if quick { 3 } else { 6 };
                let wg_use = wg_override.unwrap_or(wg);
                info!(
                    "reduce-sweep enqueue rows={} cols={} wg={} iters={}",
                    rows, cols, wg_use, iters
                );
                out.push(run_reduction_sweep_case(
                    provider, rows, cols, wg_use, iters,
                )?);
                if sweep_first {
                    break 'outer;
                }
            }
        }
    }
    Ok(out)
}

fn run_minmax_case(
    provider: &'static dyn AccelProvider,
    name: &str,
    kind: MinMaxKind,
    rows: usize,
    cols: usize,
    dim: usize,
    iterations: usize,
) -> Result<CaseReport> {
    let matrix = Matrix::generate(rows, cols, 0.25, 0.05);
    let (cpu_values, cpu_indices) = cpu_minmax_dim(&matrix, dim, kind);

    let mut upload_stats = TimeStats::new();
    let mut compute_stats = TimeStats::new();
    let mut download_stats = TimeStats::new();

    for iter in 0..iterations {
        let warmup = iter == 0;
        let (handle_matrix, upload_time) = upload_matrix(provider, &matrix)?;

        let compute_start = Instant::now();
        let result = match kind {
            MinMaxKind::Min => provider.reduce_min_dim(&handle_matrix, dim - 1)?,
            MinMaxKind::Max => provider.reduce_max_dim(&handle_matrix, dim - 1)?,
        };
        let compute_time = compute_start.elapsed();

        let (values_matrix, download_values) = download_matrix(provider, &result.values)?;
        let (indices_matrix, download_indices) = download_matrix(provider, &result.indices)?;

        provider.free(&handle_matrix)?;
        provider.free(&result.values)?;
        provider.free(&result.indices)?;

        verify_matrix(&cpu_values, &values_matrix, VALUE_TOLERANCE)?;
        verify_indices(&cpu_indices, &indices_matrix)?;

        if !warmup {
            upload_stats.record(upload_time);
            compute_stats.record(compute_time);
            download_stats.record(download_values + download_indices);
        }
    }

    upload_stats.finalize();
    compute_stats.finalize();
    download_stats.finalize();

    let samples = iterations.saturating_sub(1).max(1);
    let total_stats = TimeStats {
        total: upload_stats.total + compute_stats.total + download_stats.total,
        min: upload_stats
            .min
            .min(compute_stats.min.min(download_stats.min)),
        max: upload_stats.max + compute_stats.max + download_stats.max,
    };

    Ok(CaseReport {
        name: name.to_string(),
        category: "minmax".to_string(),
        detail: format!("{:?} {}x{} dim {}", kind, rows, cols, dim),
        input_shapes: vec![(rows, cols)],
        iterations: samples,
        upload_ms: DurationSummary::from_stats(&upload_stats, samples),
        compute_ms: DurationSummary::from_stats(&compute_stats, samples),
        download_ms: DurationSummary::from_stats(&download_stats, samples),
        total_ms: DurationSummary::from_stats(&total_stats, samples),
        notes: vec!["Indices verified against CPU reference".to_string()],
        cpu_ms: None,
    })
}

fn run_composite_atda_case(
    provider: &'static dyn AccelProvider,
    name: &str,
    rows: usize,
    cols: usize,
    iterations: usize,
) -> Result<CaseReport> {
    let matrix_a = Matrix::generate(rows, cols, 0.65, -0.3);
    let diag_vec = Vector::generate(cols, 1.2, 0.4);
    let diag_matrix = Matrix::from_diag_vector(rows, &diag_vec);
    let cpu_report = cpu_composite_atda(&matrix_a, &diag_vec);

    let mut upload_stats = TimeStats::new();
    let mut compute_stats = TimeStats::new();
    let mut download_stats = TimeStats::new();

    for iter in 0..iterations {
        let warmup = iter == 0;
        let (handle_a, upload_a) = upload_matrix(provider, &matrix_a)?;
        let (handle_d, upload_d) = upload_matrix(provider, &diag_matrix)?;

        let compute_start = Instant::now();
        let scaled_handle = provider.elem_mul(&handle_a, &handle_d)?;
        let step1 = compute_start.elapsed();

        let transpose_start = Instant::now();
        let a_t_handle = provider.transpose(&handle_a)?;
        let step2 = transpose_start.elapsed();

        let matmul_start = Instant::now();
        let result_handle = provider.matmul(&a_t_handle, &scaled_handle)?;
        let step3 = matmul_start.elapsed();

        let (result_matrix, download_time) = download_matrix(provider, &result_handle)?;

        provider.free(&handle_a)?;
        provider.free(&handle_d)?;
        provider.free(&scaled_handle)?;
        provider.free(&a_t_handle)?;
        provider.free(&result_handle)?;

        verify_matrix(&cpu_report.result, &result_matrix, VALUE_TOLERANCE)?;

        if !warmup {
            upload_stats.record(upload_a + upload_d);
            compute_stats.record(step1 + step2 + step3);
            download_stats.record(download_time);
        }
    }

    upload_stats.finalize();
    compute_stats.finalize();
    download_stats.finalize();

    let samples = iterations.saturating_sub(1).max(1);
    let total_stats = TimeStats {
        total: upload_stats.total + compute_stats.total + download_stats.total,
        min: upload_stats
            .min
            .min(compute_stats.min.min(download_stats.min)),
        max: upload_stats.max + compute_stats.max + download_stats.max,
    };

    Ok(CaseReport {
        name: name.to_string(),
        category: "composite".to_string(),
        detail: format!("At^T * diag(D) * A {}x{}", rows, cols),
        input_shapes: vec![(rows, cols), (cols, cols)],
        iterations: samples,
        upload_ms: DurationSummary::from_stats(&upload_stats, samples),
        compute_ms: DurationSummary::from_stats(&compute_stats, samples),
        download_ms: DurationSummary::from_stats(&download_stats, samples),
        total_ms: DurationSummary::from_stats(&total_stats, samples),
        notes: vec!["Composite engineering-style workload".to_string()],
        cpu_ms: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn generic_single_output_case<FGpu, FCpu>(
    provider: &'static dyn AccelProvider,
    name: &str,
    category: &str,
    detail: String,
    inputs: Vec<Matrix>,
    iterations: usize,
    tolerance: f64,
    gpu_op: FGpu,
    cpu_op: FCpu,
) -> Result<CaseReport>
where
    FGpu: Fn(&'static dyn AccelProvider, &[GpuTensorHandle]) -> Result<GpuTensorHandle>,
    FCpu: Fn(&[Matrix]) -> Matrix,
{
    let mut upload_stats = TimeStats::new();
    let mut compute_stats = TimeStats::new();
    let mut download_stats = TimeStats::new();
    let mut cpu_stats = TimeStats::new();

    for iter in 0..iterations {
        let warmup = iter == 0;
        let mut handles = Vec::new();
        let mut upload_total = Duration::ZERO;

        for matrix in &inputs {
            let (handle, upload_time) = upload_matrix(provider, matrix)?;
            handles.push(handle);
            upload_total += upload_time;
        }

        // CPU timing
        let cpu_start = Instant::now();
        let cpu_result = cpu_op(&inputs);
        let cpu_time = cpu_start.elapsed();

        let compute_start = Instant::now();
        let result_handle = gpu_op(provider, &handles)?;
        let compute_time = compute_start.elapsed();

        let (result_matrix, download_time) = download_matrix(provider, &result_handle)?;

        for handle in &handles {
            provider.free(handle)?;
        }
        provider.free(&result_handle)?;

        verify_matrix(&cpu_result, &result_matrix, tolerance)?;

        if !warmup {
            upload_stats.record(upload_total);
            compute_stats.record(compute_time);
            download_stats.record(download_time);
            cpu_stats.record(cpu_time);
        }
    }

    upload_stats.finalize();
    compute_stats.finalize();
    download_stats.finalize();

    let samples = iterations.saturating_sub(1).max(1);
    let total_stats = TimeStats {
        total: upload_stats.total + compute_stats.total + download_stats.total,
        min: upload_stats
            .min
            .min(compute_stats.min.min(download_stats.min)),
        max: upload_stats.max + compute_stats.max + download_stats.max,
    };

    Ok(CaseReport {
        name: name.to_string(),
        category: category.to_string(),
        detail,
        input_shapes: inputs.iter().map(|m| (m.rows, m.cols)).collect(),
        iterations: samples,
        upload_ms: DurationSummary::from_stats(&upload_stats, samples),
        compute_ms: DurationSummary::from_stats(&compute_stats, samples),
        download_ms: DurationSummary::from_stats(&download_stats, samples),
        total_ms: DurationSummary::from_stats(&total_stats, samples),
        notes: Vec::new(),
        cpu_ms: Some(DurationSummary::from_stats(&cpu_stats, samples)),
    })
}

fn upload_matrix(
    provider: &'static dyn AccelProvider,
    matrix: &Matrix,
) -> Result<(GpuTensorHandle, Duration)> {
    let view = HostTensorView {
        data: &matrix.data,
        shape: &[matrix.rows, matrix.cols],
    };
    let start = Instant::now();
    let handle = provider.upload(&view)?;
    Ok((handle, start.elapsed()))
}

fn download_matrix(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> Result<(Matrix, Duration)> {
    let start = Instant::now();
    let host = provider.download(handle)?;
    let elapsed = start.elapsed();
    let matrix = Matrix::from_host(&host)?;
    Ok((matrix, elapsed))
}

#[derive(Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    fn generate(rows: usize, cols: usize, base: f64, delta: f64) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for c in 0..cols {
            for r in 0..rows {
                let idx = r + c * rows;
                let value = base + delta * ((idx % 128) as f64) / 127.0;
                data.push(value);
            }
        }
        Self { rows, cols, data }
    }

    fn from_diag_vector(rows: usize, vector: &Vector) -> Self {
        let cols = vector.len();
        let mut data = Vec::with_capacity(rows * cols);
        for c in 0..cols {
            for _ in 0..rows {
                data.push(vector.values[c]);
            }
        }
        Self { rows, cols, data }
    }

    fn from_host(host: &HostTensorOwned) -> Result<Self> {
        match host.shape.as_slice() {
            [rows, cols] => Ok(Self {
                rows: *rows,
                cols: *cols,
                data: host.data.clone(),
            }),
            [len] => Ok(Self {
                rows: *len,
                cols: 1,
                data: host.data.clone(),
            }),
            _ => Err(anyhow!("Unsupported tensor shape {:?}", host.shape)),
        }
    }

    fn from_scalar(value: f64) -> Self {
        Self {
            rows: 1,
            cols: 1,
            data: vec![value],
        }
    }
}

#[derive(Clone)]
struct Vector {
    values: Vec<f64>,
}

impl Vector {
    fn generate(len: usize, base: f64, delta: f64) -> Self {
        let mut values = Vec::with_capacity(len);
        for i in 0..len {
            let value = base + delta * (i % 97) as f64 / 96.0;
            values.push(value.abs() + 0.05);
        }
        Self { values }
    }

    fn len(&self) -> usize {
        self.values.len()
    }
}

fn cpu_elementwise(a: &Matrix, b: &Matrix, op: ElementwiseOp) -> Matrix {
    let mut data = Vec::with_capacity(a.rows * a.cols);
    for (av, bv) in a.data.iter().zip(&b.data) {
        let value = match op {
            ElementwiseOp::Add => av + bv,
            ElementwiseOp::Mul => av * bv,
        };
        data.push(value);
    }
    Matrix {
        rows: a.rows,
        cols: a.cols,
        data,
    }
}

fn cpu_matmul(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.rows);
    let m = a.rows;
    let k = a.cols;
    let n = b.cols;
    let mut data = vec![0.0; m * n];
    for col in 0..n {
        for row in 0..m {
            let mut acc = 0.0;
            for kk in 0..k {
                let a_idx = row + kk * m;
                let b_idx = kk + col * k;
                acc += a.data[a_idx] * b.data[b_idx];
            }
            data[row + col * m] = acc;
        }
    }
    Matrix {
        rows: m,
        cols: n,
        data,
    }
}

fn cpu_transpose(matrix: &Matrix) -> Matrix {
    let mut data = vec![0.0; matrix.rows * matrix.cols];
    for c in 0..matrix.cols {
        for r in 0..matrix.rows {
            let src = r + c * matrix.rows;
            let dst = c + r * matrix.cols;
            data[dst] = matrix.data[src];
        }
    }
    Matrix {
        rows: matrix.cols,
        cols: matrix.rows,
        data,
    }
}

fn cpu_reduce_sum_all(matrix: &Matrix) -> f64 {
    matrix.data.iter().sum()
}

fn cpu_reduce_sum_dim(matrix: &Matrix, dim: usize) -> Matrix {
    match dim {
        1 => {
            let mut data = vec![0.0; matrix.cols];
            for (c, out) in data.iter_mut().enumerate().take(matrix.cols) {
                let mut acc = 0.0;
                for r in 0..matrix.rows {
                    acc += matrix.data[r + c * matrix.rows];
                }
                *out = acc;
            }
            Matrix {
                rows: 1,
                cols: matrix.cols,
                data,
            }
        }
        2 => {
            let mut data = vec![0.0; matrix.rows];
            for (r, out) in data.iter_mut().enumerate().take(matrix.rows) {
                let mut acc = 0.0;
                for c in 0..matrix.cols {
                    acc += matrix.data[r + c * matrix.rows];
                }
                *out = acc;
            }
            Matrix {
                rows: matrix.rows,
                cols: 1,
                data,
            }
        }
        _ => panic!("Invalid reduction dim: {}", dim),
    }
}

fn cpu_reduce_mean_dim(matrix: &Matrix, dim: usize) -> Matrix {
    let sum = cpu_reduce_sum_dim(matrix, dim);
    let scale = if dim == 1 {
        matrix.rows as f64
    } else {
        matrix.cols as f64
    };
    Matrix {
        rows: sum.rows,
        cols: sum.cols,
        data: sum.data.iter().map(|v| v / scale).collect(),
    }
}

fn cpu_minmax_dim(matrix: &Matrix, dim: usize, kind: MinMaxKind) -> (Matrix, Matrix) {
    match dim {
        1 => {
            let mut values = vec![
                match kind {
                    MinMaxKind::Min => f64::INFINITY,
                    MinMaxKind::Max => f64::NEG_INFINITY,
                };
                matrix.cols
            ];
            let mut indices = vec![1.0; matrix.cols];
            for c in 0..matrix.cols {
                for r in 0..matrix.rows {
                    let v = matrix.data[r + c * matrix.rows];
                    let better = match kind {
                        MinMaxKind::Min => v < values[c],
                        MinMaxKind::Max => v > values[c],
                    };
                    if better {
                        values[c] = v;
                        indices[c] = (r + 1) as f64;
                    }
                }
            }
            (
                Matrix {
                    rows: 1,
                    cols: matrix.cols,
                    data: values,
                },
                Matrix {
                    rows: 1,
                    cols: matrix.cols,
                    data: indices,
                },
            )
        }
        2 => {
            let mut values = vec![
                match kind {
                    MinMaxKind::Min => f64::INFINITY,
                    MinMaxKind::Max => f64::NEG_INFINITY,
                };
                matrix.rows
            ];
            let mut indices = vec![1.0; matrix.rows];
            for r in 0..matrix.rows {
                for c in 0..matrix.cols {
                    let v = matrix.data[r + c * matrix.rows];
                    let better = match kind {
                        MinMaxKind::Min => v < values[r],
                        MinMaxKind::Max => v > values[r],
                    };
                    if better {
                        values[r] = v;
                        indices[r] = (c + 1) as f64;
                    }
                }
            }
            (
                Matrix {
                    rows: matrix.rows,
                    cols: 1,
                    data: values,
                },
                Matrix {
                    rows: matrix.rows,
                    cols: 1,
                    data: indices,
                },
            )
        }
        _ => panic!("Invalid dim for min/max: {}", dim),
    }
}

struct CompositeAtdAReport {
    result: Matrix,
}

fn cpu_composite_atda(a: &Matrix, diag: &Vector) -> CompositeAtdAReport {
    let diag_matrix = Matrix::from_diag_vector(a.rows, diag);
    let scaled = cpu_elementwise(a, &diag_matrix, ElementwiseOp::Mul);
    let a_t = cpu_transpose(a);
    let result = cpu_matmul(&a_t, &scaled);
    CompositeAtdAReport { result }
}

fn verify_matrix(expected: &Matrix, actual: &Matrix, abs_tol: f64) -> Result<()> {
    if expected.rows != actual.rows || expected.cols != actual.cols {
        return Err(anyhow!(
            "Shape mismatch: expected {}x{}, got {}x{}",
            expected.rows,
            expected.cols,
            actual.rows,
            actual.cols
        ));
    }
    for idx in 0..expected.data.len() {
        let expected_val = expected.data[idx];
        let actual_val = actual.data[idx];
        let diff = (expected_val - actual_val).abs();
        let allowed = abs_tol.max(VALUE_REL_TOLERANCE * expected_val.abs());
        if diff > allowed {
            return Err(anyhow!(
                "Mismatch at idx {} (value {} vs {}), abs tol {}, rel tol {} (allowed {})",
                idx,
                expected_val,
                actual_val,
                abs_tol,
                VALUE_REL_TOLERANCE,
                allowed
            ));
        }
    }
    Ok(())
}

fn verify_indices(expected: &Matrix, actual: &Matrix) -> Result<()> {
    if expected.rows != actual.rows || expected.cols != actual.cols {
        return Err(anyhow!(
            "Index shape mismatch: expected {}x{}, got {}x{}",
            expected.rows,
            expected.cols,
            actual.rows,
            actual.cols
        ));
    }
    for idx in 0..expected.data.len() {
        let diff = (expected.data[idx] - actual.data[idx]).abs();
        if diff > 1e-5 {
            return Err(anyhow!(
                "Index mismatch at {} ({} vs {})",
                idx,
                expected.data[idx],
                actual.data[idx]
            ));
        }
    }
    Ok(())
}
