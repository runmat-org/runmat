use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::{auto_offload_options, fusion::active_fusion, fusion_residency, AutoOffloadLogLevel};
use anyhow::{anyhow, Result};
use log::{debug, info, trace};
use once_cell::sync::OnceCell;
use runmat_accelerate_api::{AccelProvider, HostTensorView};
use runmat_builtins::{builtin_functions, AccelTag, Tensor, Value};
use runmat_runtime::gather_if_needed;
use serde::Deserialize;

const DEFAULT_CPU_ELEM_PER_ELEM: f64 = 1.0e-7;
const DEFAULT_CPU_REDUCTION_PER_ELEM: f64 = 1.2e-7;
const DEFAULT_CPU_MATMUL_PER_FLOP: f64 = 2.5e-11;

#[derive(Clone, Copy, Debug)]
pub enum BinaryOp {
    Elementwise,
    MatMul,
}

#[derive(Clone, Copy, Debug)]
pub enum UnaryOp {
    Generic,
    Transpose,
}

#[derive(Clone, Copy, Debug)]
pub enum ReductionOp {
    Sum,
    Mean,
    Min,
    Max,
}

#[derive(Debug, Clone)]
struct ThresholdConfig {
    unary_min_elems: usize,
    binary_min_elems: usize,
    reduction_min_elems: usize,
    matmul_min_flops: usize,
    cpu_elem_per_elem: f64,
    cpu_reduction_per_elem: f64,
    cpu_matmul_per_flop: f64,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            unary_min_elems: 4_096,
            binary_min_elems: 4_096,
            reduction_min_elems: 256,
            matmul_min_flops: 1_000_000, // roughly 100x100x100
            cpu_elem_per_elem: DEFAULT_CPU_ELEM_PER_ELEM,
            cpu_reduction_per_elem: DEFAULT_CPU_REDUCTION_PER_ELEM,
            cpu_matmul_per_flop: DEFAULT_CPU_MATMUL_PER_FLOP,
        }
    }
}

pub struct NativeAutoOffload {
    provider: &'static dyn AccelProvider,
    thresholds: ThresholdConfig,
    enabled: bool,
}

static GLOBAL: OnceCell<Option<NativeAutoOffload>> = OnceCell::new();
static PROFILE_MODEL: OnceCell<Option<ProfileCostModel>> = OnceCell::new();

fn env_bool(key: &str) -> Option<bool> {
    env::var(key).ok().and_then(|v| parse_bool(&v))
}

fn parse_bool(s: &str) -> Option<bool> {
    match s.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn log_promotion<F>(builder: F)
where
    F: FnOnce() -> String,
{
    match auto_offload_options().log_level {
        AutoOffloadLogLevel::Off => {}
        AutoOffloadLogLevel::Info => info!("{}", builder()),
        AutoOffloadLogLevel::Trace => trace!("{}", builder()),
    }
}

fn update_cpu_cost(slot: &mut f64, candidate: f64) {
    if candidate.is_finite() && candidate > 0.0 && candidate < *slot {
        *slot = candidate;
    }
}

fn value_len(value: &Value) -> Option<usize> {
    match value {
        Value::Tensor(t) => Some(t.data.len()),
        Value::GpuTensor(handle) => Some(handle.shape.iter().product()),
        Value::Num(_) | Value::Bool(_) | Value::Int(_) => Some(1),
        Value::Complex(_, _) => Some(1),
        _ => None,
    }
}

fn element_count_pair(a: &Value, b: &Value) -> Option<usize> {
    let la = value_len(a)?;
    let lb = value_len(b)?;
    Some(la.max(lb))
}

pub fn global() -> Option<&'static NativeAutoOffload> {
    GLOBAL.get_or_init(|| initialize()).as_ref()
}

fn initialize() -> Option<NativeAutoOffload> {
    if !auto_enabled() {
        return None;
    }
    let provider = runmat_accelerate_api::provider()?;
    let mut config = ThresholdConfig::default();
    apply_env_overrides(&mut config);
    if calibrate_enabled() {
        if let Err(err) = auto_calibrate(provider, &mut config) {
            debug!("Native auto-offload calibration failed: {err}");
        }
    }
    let model_status = if profile_cost_model().is_some() {
        "profile"
    } else {
        "fallback"
    };
    info!(
        "Native auto-offload thresholds: unary={} binary={} reduction={} matmul_flops={} (model: {})",
        config.unary_min_elems,
        config.binary_min_elems,
        config.reduction_min_elems,
        config.matmul_min_flops,
        model_status
    );
    Some(NativeAutoOffload::new(provider, config))
}

impl NativeAutoOffload {
    fn new(provider: &'static dyn AccelProvider, thresholds: ThresholdConfig) -> Self {
        let enabled = true;
        Self {
            provider,
            thresholds,
            enabled,
        }
    }

    fn promote_tensor_if_large(&self, value: &Value, threshold: usize) -> Result<Value> {
        match value {
            Value::GpuTensor(_) => Ok(value.clone()),
            Value::Tensor(t) => {
                if t.data.len() >= threshold && threshold > 0 {
                    log_promotion(|| {
                        format!(
                            "Promoting tensor to GPU (len={}, threshold={})",
                            t.data.len(),
                            threshold
                        )
                    });
                    self.tensor_to_gpu(t)
                } else {
                    Ok(value.clone())
                }
            }
            _ => Ok(value.clone()),
        }
    }

    fn tensor_to_gpu(&self, tensor: &Tensor) -> Result<Value> {
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = self
            .provider
            .upload(&view)
            .map_err(|e| anyhow!(e.to_string()))?;
        Ok(Value::GpuTensor(handle))
    }

    fn promote_binary(&self, op: BinaryOp, a: &Value, b: &Value) -> Result<(Value, Value)> {
        if !self.enabled {
            return Ok((a.clone(), b.clone()));
        }
        match op {
            BinaryOp::Elementwise => {
                let mut elems = element_count_pair(a, b).unwrap_or(0);
                if let Some(active) = active_fusion() {
                    if active.kind.is_elementwise() {
                        if let Some(ec) = active.element_count {
                            elems = ec;
                        }
                    }
                }
                let should_gpu = self
                    .should_gpu_elementwise(elems)
                    .unwrap_or(elems >= self.thresholds.binary_min_elems);
                if should_gpu {
                    log_promotion(|| format!("Elementwise offload accepted ({} elems)", elems));
                }
                let threshold = if should_gpu {
                    1
                } else {
                    self.thresholds.binary_min_elems
                };
                let a_p = self.promote_tensor_if_large(a, threshold)?;
                let b_p = self.promote_tensor_if_large(b, threshold)?;
                Ok((a_p, b_p))
            }
            BinaryOp::MatMul => {
                if let (Some((ra, ca)), Some((rb, cb))) = (tensor_rows_cols(a), tensor_rows_cols(b))
                {
                    if ca != rb {
                        return Ok((a.clone(), b.clone()));
                    }
                    let flops = ra * ca * cb;
                    let should_gpu = self
                        .should_gpu_matmul(flops)
                        .unwrap_or(flops >= self.thresholds.matmul_min_flops);
                    if should_gpu {
                        log_promotion(|| {
                            format!(
                                "Promoting matmul operands (flops={}, threshold={})",
                                flops, self.thresholds.matmul_min_flops
                            )
                        });
                        let a_p = self.promote_tensor_if_large(a, 1)?;
                        let b_p = self.promote_tensor_if_large(b, 1)?;
                        return Ok((a_p, b_p));
                    }
                }
                Ok((a.clone(), b.clone()))
            }
        }
    }

    fn promote_unary(&self, op: UnaryOp, v: &Value) -> Result<Value> {
        if !self.enabled {
            return Ok(v.clone());
        }
        let elems = value_len(v).unwrap_or(0);
        let base = self.thresholds.unary_min_elems;
        let use_gpu = match op {
            UnaryOp::Transpose => self.should_gpu_transpose(elems).unwrap_or(elems >= base),
            UnaryOp::Generic => elems >= base,
        };
        if use_gpu {
            log_promotion(|| format!("Unary offload accepted ({:?}, {} elems)", op, elems));
        }
        let threshold = if use_gpu { 1 } else { base };
        self.promote_tensor_if_large(v, threshold)
    }

    fn promote_reduction(&self, _op: ReductionOp, args: &[Value]) -> Result<Vec<Value>> {
        if !self.enabled || args.is_empty() {
            return Ok(args.to_vec());
        }
        let elems = value_len(&args[0]).unwrap_or(0);
        let should_gpu = self
            .should_gpu_reduction(elems)
            .unwrap_or(elems >= self.thresholds.reduction_min_elems);
        if should_gpu {
            log_promotion(|| format!("Reduction offload accepted ({} elems)", elems));
        }
        let threshold = if should_gpu {
            1
        } else {
            self.thresholds.reduction_min_elems
        };
        let mut out = Vec::with_capacity(args.len());
        if let Some(first) = args.first() {
            out.push(self.promote_tensor_if_large(first, threshold)?);
            for rest in &args[1..] {
                out.push(rest.clone());
            }
        }
        Ok(out)
    }

    fn prepare_builtin(&self, name: &str, args: &[Value]) -> Result<Vec<Value>> {
        if !self.enabled {
            return Ok(args.to_vec());
        }
        // Do not attempt to promote 'double' on providers that cannot store f64.
        // Offloading a cast to double requires device-side f64; otherwise keep host.
        if name.eq_ignore_ascii_case("double") {
            if self.provider.precision() != runmat_accelerate_api::ProviderPrecision::F64 {
                return Ok(args.to_vec());
            }
        }
        if let Some(policy) = builtin_policy(name) {
            if policy.is_sink {
                return gather_args(args);
            }

            let mut processed = args.to_vec();

            if policy
                .accel_tags
                .iter()
                .any(|tag| matches!(tag, AccelTag::Reduction))
            {
                log_promotion(|| format!("Promoting builtin '{}' as reduction", name));
                return self.promote_reduction(ReductionOp::Sum, args);
            }

            if policy
                .accel_tags
                .iter()
                .any(|tag| matches!(tag, AccelTag::MatMul))
                && processed.len() >= 2
            {
                log_promotion(|| format!("Promoting builtin '{}' as matmul", name));
                let (a_p, b_p) =
                    self.promote_binary(BinaryOp::MatMul, &processed[0], &processed[1])?;
                processed[0] = a_p;
                processed[1] = b_p;
                return Ok(processed);
            }

            if policy
                .accel_tags
                .iter()
                .any(|tag| matches!(tag, AccelTag::Elementwise))
                && processed.len() >= 2
            {
                log_promotion(|| format!("Promoting builtin '{}' as elementwise", name));
                let (a_p, b_p) =
                    self.promote_binary(BinaryOp::Elementwise, &processed[0], &processed[1])?;
                processed[0] = a_p;
                processed[1] = b_p;
                return Ok(processed);
            }

            if let Some(first) = processed.first_mut() {
                if policy
                    .accel_tags
                    .iter()
                    .any(|tag| matches!(tag, AccelTag::Transpose))
                {
                    log_promotion(|| format!("Promoting builtin '{}' as transpose", name));
                    *first = self.promote_unary(UnaryOp::Transpose, first)?;
                    return Ok(processed);
                }

                if policy
                    .accel_tags
                    .iter()
                    .any(|tag| matches!(tag, AccelTag::Unary))
                {
                    log_promotion(|| format!("Promoting builtin '{}' as unary", name));
                    *first = self.promote_unary(UnaryOp::Generic, first)?;
                    return Ok(processed);
                }
            }
        }
        Ok(args.to_vec())
    }

    fn should_gpu_elementwise(&self, elements: usize) -> Option<bool> {
        if let Some(active) = active_fusion() {
            if active.kind.is_elementwise() {
                return Some(true);
            }
        }
        if elements == 0 {
            return Some(false);
        }
        let cpu_cost = self.thresholds.cpu_elem_per_elem * elements as f64;
        profile_cost_model()
            .and_then(|model| model.estimate_elemwise(elements))
            .map(|gpu| gpu.as_secs_f64() * 0.95 < cpu_cost)
    }

    fn should_gpu_reduction(&self, elements: usize) -> Option<bool> {
        if elements == 0 {
            return Some(false);
        }
        let cpu_cost = self.thresholds.cpu_reduction_per_elem * elements as f64;
        profile_cost_model()
            .and_then(|model| model.estimate_reduction(elements))
            .map(|gpu| gpu.as_secs_f64() * 0.95 < cpu_cost)
    }

    fn should_gpu_matmul(&self, flops: usize) -> Option<bool> {
        if flops == 0 {
            return Some(false);
        }
        let cpu_cost = self.thresholds.cpu_matmul_per_flop * flops as f64;
        profile_cost_model()
            .and_then(|model| model.estimate_matmul_flops(flops))
            .map(|gpu| gpu.as_secs_f64() * 0.95 < cpu_cost)
    }

    fn should_gpu_transpose(&self, elements: usize) -> Option<bool> {
        if elements == 0 {
            return Some(false);
        }
        let cpu_cost = self.thresholds.cpu_elem_per_elem * elements as f64;
        profile_cost_model()
            .and_then(|model| model.estimate_transpose(elements))
            .map(|gpu| gpu.as_secs_f64() * 0.95 < cpu_cost)
    }
}

fn tensor_rows_cols(value: &Value) -> Option<(usize, usize)> {
    match value {
        Value::Tensor(t) => Some((t.rows(), t.cols())),
        Value::GpuTensor(handle) => {
            if handle.shape.len() == 2 {
                Some((handle.shape[0], handle.shape[1]))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn gather_args(args: &[Value]) -> Result<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        if let Value::GpuTensor(handle) = value {
            fusion_residency::clear(handle);
        }
        out.push(gather_if_needed(value).map_err(|e| anyhow!(e))?);
    }
    Ok(out)
}

#[derive(Clone, Copy)]
struct BuiltinPolicy {
    accel_tags: &'static [AccelTag],
    is_sink: bool,
}

static BUILTIN_POLICIES: OnceCell<HashMap<String, BuiltinPolicy>> = OnceCell::new();

fn builtin_policy(name: &str) -> Option<BuiltinPolicy> {
    let map = BUILTIN_POLICIES.get_or_init(|| {
        let mut map = HashMap::new();
        for func in builtin_functions() {
            map.insert(
                func.name.to_ascii_lowercase(),
                BuiltinPolicy {
                    accel_tags: func.accel_tags,
                    is_sink: func.is_sink,
                },
            );
        }
        map
    });
    map.get(&name.to_ascii_lowercase()).copied()
}

fn auto_enabled() -> bool {
    if let Some(flag) = env_bool("RUNMAT_ACCEL_AUTO_OFFLOAD") {
        return flag;
    }
    auto_offload_options().enabled
}

fn calibrate_enabled() -> bool {
    if let Some(flag) = env_bool("RUNMAT_ACCEL_CALIBRATE") {
        return flag;
    }
    auto_offload_options().calibrate
}

fn apply_env_overrides(cfg: &mut ThresholdConfig) {
    if let Some(val) = env_usize("RUNMAT_ACCEL_THRESHOLD_UNARY") {
        cfg.unary_min_elems = val;
    }
    if let Some(val) = env_usize("RUNMAT_ACCEL_THRESHOLD_ELEMWISE") {
        cfg.binary_min_elems = val;
    }
    if let Some(val) = env_usize("RUNMAT_ACCEL_THRESHOLD_REDUCTION") {
        cfg.reduction_min_elems = val;
    }
    if let Some(val) = env_usize("RUNMAT_ACCEL_THRESHOLD_MATMUL") {
        cfg.matmul_min_flops = val;
    }
    if let Some(val) = env_usize("RUNMAT_ACCEL_THRESHOLD_ALL") {
        cfg.unary_min_elems = val;
        cfg.binary_min_elems = val;
        cfg.reduction_min_elems = val;
    }
}

fn env_usize(key: &str) -> Option<usize> {
    env::var(key).ok().and_then(|v| v.parse::<usize>().ok())
}

fn auto_calibrate(provider: &'static dyn AccelProvider, cfg: &mut ThresholdConfig) -> Result<()> {
    if let Some(elem_threshold) = calibrate_elemwise(provider, cfg).transpose()? {
        if elem_threshold != usize::MAX {
            cfg.binary_min_elems = elem_threshold;
            cfg.unary_min_elems = cfg.unary_min_elems.min(elem_threshold);
        }
    }
    if let Some(red_threshold) = calibrate_reduction(provider, cfg).transpose()? {
        if red_threshold != usize::MAX {
            cfg.reduction_min_elems = red_threshold;
        }
    }
    if let Some(matmul_threshold) = calibrate_matmul(provider, cfg).transpose()? {
        if matmul_threshold != usize::MAX {
            cfg.matmul_min_flops = matmul_threshold;
        }
    }
    Ok(())
}

fn calibrate_elemwise(
    provider: &'static dyn AccelProvider,
    cfg: &mut ThresholdConfig,
) -> Option<Result<usize>> {
    let sizes = [256usize, 1_024, 4_096, 16_384, 65_536];
    for size in sizes {
        match compare_elemwise(provider, size, &mut cfg.cpu_elem_per_elem) {
            Ok(Some(true)) => return Some(Ok(size)),
            Ok(Some(false)) => continue,
            Ok(None) => return None,
            Err(e) => return Some(Err(e)),
        }
    }
    Some(Ok(usize::MAX))
}

fn compare_elemwise(
    provider: &'static dyn AccelProvider,
    elements: usize,
    cpu_cost_slot: &mut f64,
) -> Result<Option<bool>> {
    if elements == 0 {
        return Ok(Some(false));
    }
    let data: Vec<f64> = (0..elements).map(|i| i as f64).collect();
    let tensor = Tensor::new(data.clone(), vec![elements, 1]).map_err(|e| anyhow!(e))?;
    let a = Value::Tensor(tensor.clone());
    let b = Value::Tensor(tensor.clone());
    let cpu_time = time(|| runmat_runtime::call_builtin("plus", &[a.clone(), b.clone()]))?;
    let cpu_per_elem = cpu_time.as_secs_f64() / elements as f64;
    update_cpu_cost(cpu_cost_slot, cpu_per_elem);
    if let Some(model) = profile_cost_model() {
        if let Some(gpu_time) = model.estimate_elemwise(elements) {
            trace!(
                "Elemwise calibration ({} elems): cpu={:?}, gpu_est={:?}",
                elements,
                cpu_time,
                gpu_time
            );
            return Ok(Some(gpu_time < cpu_time));
        }
    }
    let view = HostTensorView {
        data: &data,
        shape: &[elements, 1],
    };
    let ha = provider.upload(&view).map_err(|e| anyhow!(e.to_string()))?;
    let hb = provider.upload(&view).map_err(|e| anyhow!(e.to_string()))?;
    let start = Instant::now();
    let hc = match provider.elem_add(&ha, &hb) {
        Ok(h) => h,
        Err(_) => {
            let _ = provider.free(&ha);
            let _ = provider.free(&hb);
            return Ok(None);
        }
    };
    let gpu_time = start.elapsed();
    let _ = provider.free(&ha);
    let _ = provider.free(&hb);
    let _ = provider.free(&hc);
    Ok(Some(gpu_time < cpu_time))
}

fn calibrate_reduction(
    provider: &'static dyn AccelProvider,
    cfg: &mut ThresholdConfig,
) -> Option<Result<usize>> {
    let sizes = [256usize, 1_024, 4_096, 16_384, 65_536];
    for size in sizes {
        match compare_reduction(provider, size, &mut cfg.cpu_reduction_per_elem) {
            Ok(Some(true)) => return Some(Ok(size)),
            Ok(Some(false)) => continue,
            Ok(None) => return None,
            Err(e) => return Some(Err(e)),
        }
    }
    Some(Ok(usize::MAX))
}

fn compare_reduction(
    provider: &'static dyn AccelProvider,
    elements: usize,
    cpu_cost_slot: &mut f64,
) -> Result<Option<bool>> {
    let data: Vec<f64> = (0..elements).map(|i| i as f64).collect();
    let tensor = Tensor::new(data.clone(), vec![elements, 1]).map_err(|e| anyhow!(e))?;
    let value = Value::Tensor(tensor.clone());
    let cpu_time = time(|| runmat_runtime::call_builtin("sum", &[value.clone()]))?;
    let cpu_per_elem = cpu_time.as_secs_f64() / elements as f64;
    update_cpu_cost(cpu_cost_slot, cpu_per_elem);
    if let Some(model) = profile_cost_model() {
        if let Some(gpu_time) = model.estimate_reduction(elements) {
            trace!(
                "Reduction calibration ({} elems): cpu={:?}, gpu_est={:?}",
                elements,
                cpu_time,
                gpu_time
            );
            return Ok(Some(gpu_time < cpu_time));
        }
    }
    let view = HostTensorView {
        data: &data,
        shape: &[elements, 1],
    };
    let h = provider.upload(&view).map_err(|e| anyhow!(e.to_string()))?;
    let start = Instant::now();
    let out = match provider.reduce_sum(&h) {
        Ok(hc) => hc,
        Err(_) => {
            provider.free(&h).ok();
            return Ok(None);
        }
    };
    let gpu_time = start.elapsed();
    let _ = provider.free(&h);
    let _ = provider.free(&out);
    Ok(Some(gpu_time < cpu_time))
}

fn calibrate_matmul(
    provider: &'static dyn AccelProvider,
    cfg: &mut ThresholdConfig,
) -> Option<Result<usize>> {
    let dims = [32usize, 64, 96, 128, 192];
    for n in dims {
        match compare_matmul(provider, n, &mut cfg.cpu_matmul_per_flop) {
            Ok(Some(true)) => {
                let flops = n * n * n;
                return Some(Ok(flops));
            }
            Ok(Some(false)) => continue,
            Ok(None) => return None,
            Err(e) => return Some(Err(e)),
        }
    }
    Some(Ok(usize::MAX))
}

fn compare_matmul(
    provider: &'static dyn AccelProvider,
    n: usize,
    cpu_cost_slot: &mut f64,
) -> Result<Option<bool>> {
    if n == 0 {
        return Ok(Some(false));
    }
    let total = n * n;
    let data_a: Vec<f64> = (0..total).map(|i| (i % 13) as f64).collect();
    let data_b: Vec<f64> = (0..total).map(|i| (i % 7) as f64).collect();
    let ta = Tensor::new(data_a.clone(), vec![n, n]).map_err(|e| anyhow!(e))?;
    let tb = Tensor::new(data_b.clone(), vec![n, n]).map_err(|e| anyhow!(e))?;
    let a = Value::Tensor(ta.clone());
    let b = Value::Tensor(tb.clone());
    let cpu_time = time(|| runmat_runtime::matrix::value_matmul(&a, &b))?;
    let flops = (n * n * n) as f64;
    update_cpu_cost(cpu_cost_slot, cpu_time.as_secs_f64() / flops);
    if let Some(model) = profile_cost_model() {
        if let Some(gpu_time) = model.estimate_matmul(n, n, n) {
            trace!(
                "Matmul calibration ({}^3 flops): cpu={:?}, gpu_est={:?}",
                n,
                cpu_time,
                gpu_time
            );
            return Ok(Some(gpu_time < cpu_time));
        }
    }
    let view_a = HostTensorView {
        data: &data_a,
        shape: &[n, n],
    };
    let view_b = HostTensorView {
        data: &data_b,
        shape: &[n, n],
    };
    let ha = provider
        .upload(&view_a)
        .map_err(|e| anyhow!(e.to_string()))?;
    let hb = provider
        .upload(&view_b)
        .map_err(|e| anyhow!(e.to_string()))?;
    let start = Instant::now();
    let hc = match provider.matmul(&ha, &hb) {
        Ok(h) => h,
        Err(_) => {
            let _ = provider.free(&ha);
            let _ = provider.free(&hb);
            return Ok(None);
        }
    };
    let gpu_time = start.elapsed();
    let _ = provider.free(&ha);
    let _ = provider.free(&hb);
    let _ = provider.free(&hc);
    Ok(Some(gpu_time < cpu_time))
}

fn time<F, T>(mut f: F) -> Result<Duration>
where
    F: FnMut() -> Result<T, String>,
{
    let start = Instant::now();
    let _ = f().map_err(|e| anyhow!(e))?;
    Ok(start.elapsed())
}

#[derive(Clone, Deserialize, Debug)]
struct ProfileDurationSummary {
    #[serde(default)]
    avg_ms: f64,
}

#[derive(Clone, Deserialize, Debug)]
struct ProfileReport {
    category: String,
    #[serde(default)]
    input_shapes: Vec<Vec<usize>>,
    total_ms: ProfileDurationSummary,
}

#[derive(Clone, Copy, Default, Debug)]
struct LinearModel {
    slope: f64,
    intercept: f64,
}

impl LinearModel {
    fn estimate(&self, x: f64) -> Option<Duration> {
        if !self.slope.is_finite() || self.slope <= 0.0 {
            return None;
        }
        let total = self.intercept + self.slope * x;
        if total.is_finite() && total > 0.0 {
            Some(Duration::from_secs_f64(total))
        } else {
            None
        }
    }
}

#[derive(Default)]
struct ProfileCostModel {
    elem: Option<LinearModel>,
    reduction: Option<LinearModel>,
    transpose: Option<LinearModel>,
    matmul: Option<LinearModel>,
}

impl ProfileCostModel {
    fn from_reports(reports: &[ProfileReport]) -> Self {
        let mut elem_samples = Vec::<(f64, f64)>::new();
        let mut reduction_samples = Vec::<(f64, f64)>::new();
        let mut transpose_samples = Vec::<(f64, f64)>::new();
        let mut matmul_samples = Vec::<(f64, f64)>::new();

        for report in reports {
            let total_secs = report.total_ms.avg_ms / 1_000.0;
            match report.category.as_str() {
                "elementwise" | "reduction" | "transpose" => {
                    if let Some(shape) = report.input_shapes.first() {
                        let elems: usize = shape.iter().copied().product();
                        if elems == 0 {
                            continue;
                        }
                        let sample = (elems as f64, total_secs);
                        match report.category.as_str() {
                            "elementwise" => elem_samples.push(sample),
                            "reduction" => reduction_samples.push(sample),
                            "transpose" => transpose_samples.push(sample),
                            _ => {}
                        }
                    }
                }
                "matmul" => {
                    if report.input_shapes.len() >= 2 {
                        let a = &report.input_shapes[0];
                        let b = &report.input_shapes[1];
                        if a.len() == 2 && b.len() == 2 {
                            let m = a[0];
                            let k = a[1];
                            let n = b[1];
                            let flops = m.checked_mul(k).and_then(|val| val.checked_mul(n));
                            if let Some(flops) = flops {
                                matmul_samples.push((flops as f64, total_secs));
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        ProfileCostModel {
            elem: fit_linear_model(&elem_samples),
            reduction: fit_linear_model(&reduction_samples),
            transpose: fit_linear_model(&transpose_samples),
            matmul: fit_linear_model(&matmul_samples),
        }
    }

    fn estimate_elemwise(&self, elements: usize) -> Option<Duration> {
        self.elem.and_then(|model| model.estimate(elements as f64))
    }

    fn estimate_reduction(&self, elements: usize) -> Option<Duration> {
        self.reduction
            .and_then(|model| model.estimate(elements as f64))
    }

    fn estimate_matmul(&self, m: usize, k: usize, n: usize) -> Option<Duration> {
        let flops = m.checked_mul(k)?.checked_mul(n)?;
        self.matmul.and_then(|model| model.estimate(flops as f64))
    }

    fn estimate_matmul_flops(&self, flops: usize) -> Option<Duration> {
        self.matmul.and_then(|model| model.estimate(flops as f64))
    }

    fn estimate_transpose(&self, elements: usize) -> Option<Duration> {
        self.transpose
            .and_then(|model| model.estimate(elements as f64))
    }
}

fn fit_linear_model(samples: &[(f64, f64)]) -> Option<LinearModel> {
    if samples.is_empty() {
        return None;
    }
    if samples.len() == 1 {
        let (x, y) = samples[0];
        if x > 0.0 {
            return Some(LinearModel {
                slope: (y / x).max(0.0),
                intercept: 0.0,
            });
        }
        return None;
    }

    let sum_x: f64 = samples.iter().map(|(x, _)| *x).sum();
    let sum_y: f64 = samples.iter().map(|(_, y)| *y).sum();
    let sum_xx: f64 = samples.iter().map(|(x, _)| x * x).sum();
    let sum_xy: f64 = samples.iter().map(|(x, y)| x * y).sum();
    let n = samples.len() as f64;
    let denom = (n * sum_xx) - (sum_x * sum_x);
    if denom.abs() < f64::EPSILON {
        return None;
    }
    let slope = ((n * sum_xy) - (sum_x * sum_y)) / denom;
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;
    let mut intercept = mean_y - slope * mean_x;
    if intercept < 0.0 {
        intercept = 0.0;
    }
    if !slope.is_finite() || slope <= 0.0 {
        return None;
    }
    Some(LinearModel { slope, intercept })
}

fn profile_cost_model() -> Option<&'static ProfileCostModel> {
    PROFILE_MODEL
        .get_or_init(|| load_profile_cost_model())
        .as_ref()
}

fn load_profile_cost_model() -> Option<ProfileCostModel> {
    let mut candidates = Vec::new();
    if let Ok(path) = env::var("RUNMAT_ACCEL_PROFILE") {
        candidates.push(PathBuf::from(path));
    }
    if let Some(path) = auto_offload_options().profile_path.clone() {
        candidates.push(path);
    }
    candidates.push(PathBuf::from("benchmarks/wgpu_profile/mac_m2.json"));
    candidates.push(PathBuf::from("wgpu_profile.json"));

    for path in candidates {
        if !path.exists() {
            continue;
        }
        match fs::read_to_string(&path) {
            Ok(contents) => match serde_json::from_str::<Vec<ProfileReport>>(&contents) {
                Ok(reports) => {
                    debug!(
                        "Loaded {} GPU profile reports from {}",
                        reports.len(),
                        path.display()
                    );
                    return Some(ProfileCostModel::from_reports(&reports));
                }
                Err(err) => {
                    debug!("Failed to parse GPU profile {}: {err}", path.display());
                }
            },
            Err(err) => {
                debug!("Failed to read GPU profile {}: {err}", path.display());
            }
        }
    }
    None
}

pub fn promote_binary(op: BinaryOp, a: &Value, b: &Value) -> Result<(Value, Value)> {
    if !auto_enabled() {
        return Ok((a.clone(), b.clone()));
    }
    if let Some(auto) = global() {
        auto.promote_binary(op, a, b)
    } else {
        Ok((a.clone(), b.clone()))
    }
}

pub fn promote_unary(op: UnaryOp, value: &Value) -> Result<Value> {
    if !auto_enabled() {
        return Ok(value.clone());
    }
    if let Some(auto) = global() {
        auto.promote_unary(op, value)
    } else {
        Ok(value.clone())
    }
}

pub fn prepare_builtin_args(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if let Some(policy) = builtin_policy(name) {
        if policy.is_sink {
            return gather_args(args);
        }
    }
    if !auto_enabled() {
        return Ok(args.to_vec());
    }
    if let Some(auto) = global() {
        auto.prepare_builtin(name, args)
    } else {
        Ok(args.to_vec())
    }
}

pub fn is_sink(name: &str) -> bool {
    builtin_policy(name).map(|p| p.is_sink).unwrap_or(false)
}

pub fn promote_reduction_args(op: ReductionOp, args: &[Value]) -> Result<Vec<Value>> {
    if !auto_enabled() {
        return Ok(args.to_vec());
    }
    if let Some(auto) = global() {
        auto.promote_reduction(op, args)
    } else {
        Ok(args.to_vec())
    }
}
