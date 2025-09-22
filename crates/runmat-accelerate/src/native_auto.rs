use std::collections::HashMap;
use std::env;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use once_cell::sync::OnceCell;
use runmat_accelerate_api::{AccelProvider, HostTensorView};
use runmat_builtins::{builtin_functions, AccelTag, Tensor, Value};
use runmat_runtime::gather_if_needed;

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
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            unary_min_elems: 4_096,
            binary_min_elems: 4_096,
            reduction_min_elems: 4_096,
            matmul_min_flops: 1_000_000, // roughly 100x100x100
        }
    }
}

pub struct NativeAutoOffload {
    provider: &'static dyn AccelProvider,
    thresholds: ThresholdConfig,
    enabled: bool,
}

static GLOBAL: OnceCell<Option<NativeAutoOffload>> = OnceCell::new();

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
            // Best-effort: fallback to env/default thresholds
            let _ = err; // suppress unused warning without logging
        }
    }
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
                let threshold = self.thresholds.binary_min_elems;
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
                    if flops >= self.thresholds.matmul_min_flops {
                        let a_p = self.promote_tensor_if_large(a, 1)?;
                        let b_p = self.promote_tensor_if_large(b, 1)?;
                        return Ok((a_p, b_p));
                    }
                }
                Ok((a.clone(), b.clone()))
            }
        }
    }

    fn promote_unary(&self, _op: UnaryOp, v: &Value) -> Result<Value> {
        if !self.enabled {
            return Ok(v.clone());
        }
        let threshold = self.thresholds.unary_min_elems;
        self.promote_tensor_if_large(v, threshold)
    }

    fn promote_reduction(&self, _op: ReductionOp, args: &[Value]) -> Result<Vec<Value>> {
        if !self.enabled || args.is_empty() {
            return Ok(args.to_vec());
        }
        let threshold = self.thresholds.reduction_min_elems;
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
                return self.promote_reduction(ReductionOp::Sum, args);
            }

            if policy
                .accel_tags
                .iter()
                .any(|tag| matches!(tag, AccelTag::MatMul))
                && processed.len() >= 2
            {
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
                    *first = self.promote_unary(UnaryOp::Transpose, first)?;
                    return Ok(processed);
                }

                if policy
                    .accel_tags
                    .iter()
                    .any(|tag| matches!(tag, AccelTag::Unary))
                {
                    *first = self.promote_unary(UnaryOp::Generic, first)?;
                    return Ok(processed);
                }
            }
        }
        Ok(args.to_vec())
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
    args.iter()
        .map(|v| gather_if_needed(v).map_err(|e| anyhow!(e)))
        .collect()
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
    env::var("RUNMAT_ACCEL_AUTO_OFFLOAD")
        .map(|v| {
            let v = v.to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        })
        .unwrap_or(true)
}

fn calibrate_enabled() -> bool {
    env::var("RUNMAT_ACCEL_CALIBRATE")
        .map(|v| {
            let v = v.to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        })
        .unwrap_or(true)
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
    if let Some(elem_threshold) = calibrate_elemwise(provider).transpose()? {
        cfg.binary_min_elems = elem_threshold;
        cfg.unary_min_elems = cfg.unary_min_elems.min(elem_threshold);
    }
    if let Some(red_threshold) = calibrate_reduction(provider).transpose()? {
        cfg.reduction_min_elems = red_threshold;
    }
    if let Some(matmul_threshold) = calibrate_matmul(provider).transpose()? {
        cfg.matmul_min_flops = matmul_threshold;
    }
    Ok(())
}

fn calibrate_elemwise(provider: &'static dyn AccelProvider) -> Option<Result<usize>> {
    let sizes = [256usize, 1_024, 4_096, 16_384, 65_536];
    for size in sizes {
        match compare_elemwise(provider, size) {
            Ok(Some(true)) => return Some(Ok(size)),
            Ok(Some(false)) => continue,
            Ok(None) => return None,
            Err(e) => return Some(Err(e)),
        }
    }
    Some(Ok(usize::MAX))
}

fn compare_elemwise(provider: &'static dyn AccelProvider, elements: usize) -> Result<Option<bool>> {
    if elements == 0 {
        return Ok(Some(false));
    }
    let data: Vec<f64> = (0..elements).map(|i| i as f64).collect();
    let tensor = Tensor::new(data.clone(), vec![elements, 1]).map_err(|e| anyhow!(e))?;
    let a = Value::Tensor(tensor.clone());
    let b = Value::Tensor(tensor.clone());
    let cpu_time = time(|| runmat_runtime::elementwise_add(&a, &b))?;
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

fn calibrate_reduction(provider: &'static dyn AccelProvider) -> Option<Result<usize>> {
    let sizes = [256usize, 1_024, 4_096, 16_384, 65_536];
    for size in sizes {
        match compare_reduction(provider, size) {
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
) -> Result<Option<bool>> {
    let data: Vec<f64> = (0..elements).map(|i| i as f64).collect();
    let tensor = Tensor::new(data.clone(), vec![elements, 1]).map_err(|e| anyhow!(e))?;
    let value = Value::Tensor(tensor.clone());
    let cpu_time = time(|| runmat_runtime::call_builtin("sum", &[value.clone()]))?;
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

fn calibrate_matmul(provider: &'static dyn AccelProvider) -> Option<Result<usize>> {
    let dims = [32usize, 64, 96, 128, 192];
    for n in dims {
        match compare_matmul(provider, n) {
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

fn compare_matmul(provider: &'static dyn AccelProvider, n: usize) -> Result<Option<bool>> {
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
