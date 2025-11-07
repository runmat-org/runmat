use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::{
    auto_offload_options,
    fusion::{active_fusion, FusionKind},
    fusion_residency, AutoOffloadLogLevel,
};
use anyhow::{anyhow, Result};
use log::{debug, info, trace, warn};
use once_cell::sync::{Lazy, OnceCell};
use runmat_accelerate_api::{AccelProvider, ApiDeviceInfo, HostTensorView};
use runmat_builtins::{builtin_functions, AccelTag, Tensor, Value};
use runmat_runtime::gather_if_needed;
use serde::{Deserialize, Serialize};

const DEFAULT_CPU_ELEM_PER_ELEM: f64 = 1.0e-7;
const DEFAULT_CPU_REDUCTION_PER_ELEM: f64 = 1.2e-7;
const DEFAULT_CPU_MATMUL_PER_FLOP: f64 = 2.5e-11;
const SMALL_BATCH_DEFAULT_MAX_DIM: usize = 8;
const SMALL_BATCH_DEFAULT_MIN_ELEMS: usize = 1_048_576;
const DECISION_LOG_CAPACITY: usize = 128;
const CALIBRATION_VERSION: u32 = 1;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ThresholdConfig {
    unary_min_elems: usize,
    binary_min_elems: usize,
    reduction_min_elems: usize,
    matmul_min_flops: usize,
    cpu_elem_per_elem: f64,
    cpu_reduction_per_elem: f64,
    cpu_matmul_per_flop: f64,
    small_batch_max_dim: usize,
    small_batch_min_elems: usize,
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
            small_batch_max_dim: SMALL_BATCH_DEFAULT_MAX_DIM,
            small_batch_min_elems: SMALL_BATCH_DEFAULT_MIN_ELEMS,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AutoOffloadDecisionEntry {
    pub timestamp_ms: u128,
    pub operation: String,
    pub elements: Option<usize>,
    pub flops: Option<usize>,
    pub batch: Option<usize>,
    pub decision: AutoOffloadDisposition,
    pub reason: DecisionReason,
    pub cpu_estimate_ms: Option<f64>,
    pub gpu_estimate_ms: Option<f64>,
    pub threshold: Option<usize>,
    pub fusion_kind: Option<FusionKind>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum AutoOffloadDisposition {
    Gpu,
    Cpu,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum DecisionReason {
    FusionOverride,
    SmallBatchGuard,
    ProfileModel,
    Threshold,
    Disabled,
}

#[derive(Debug, Clone, Serialize)]
pub struct ThresholdSnapshot {
    pub unary_min_elems: usize,
    pub binary_min_elems: usize,
    pub reduction_min_elems: usize,
    pub matmul_min_flops: usize,
    pub cpu_elem_per_elem: f64,
    pub cpu_reduction_per_elem: f64,
    pub cpu_matmul_per_flop: f64,
    pub small_batch_max_dim: usize,
    pub small_batch_min_elems: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct AutoOffloadCalibrationSummary {
    pub previous: ThresholdSnapshot,
    pub delta: ThresholdDelta,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ThresholdDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_elem_per_elem: Option<ThresholdDeltaEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_reduction_per_elem: Option<ThresholdDeltaEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_matmul_per_flop: Option<ThresholdDeltaEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ThresholdDeltaEntry {
    pub before: f64,
    pub after: f64,
    pub absolute: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ratio: Option<f64>,
}

impl ThresholdDeltaEntry {
    fn new(before: f64, after: f64) -> Self {
        let absolute = after - before;
        let ratio = if before.abs() > f64::EPSILON {
            Some(after / before)
        } else {
            None
        };
        Self {
            before,
            after,
            absolute,
            ratio,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AutoOffloadReport {
    pub provider: Option<CachedProviderInfo>,
    pub thresholds: ThresholdSnapshot,
    pub base_source: ThresholdBase,
    pub env_overrides_applied: bool,
    pub cache_path: Option<String>,
    pub calibrate_duration_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration: Option<AutoOffloadCalibrationSummary>,
    pub decisions: Vec<AutoOffloadDecisionEntry>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ThresholdBase {
    BuiltInDefault,
    LoadedFromCache,
    Calibrated,
}

impl ThresholdBase {
    pub fn as_str(&self) -> &'static str {
        match self {
            ThresholdBase::BuiltInDefault => "built-in-default",
            ThresholdBase::LoadedFromCache => "loaded-from-cache",
            ThresholdBase::Calibrated => "calibrated",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CachedProviderInfo {
    pub name: String,
    pub vendor: String,
    pub backend: Option<String>,
    pub device_id: u32,
}

#[derive(Debug, Clone)]
struct AutoOffloadState {
    provider: Option<CachedProviderInfo>,
    thresholds: ThresholdConfig,
    base_source: ThresholdBase,
    env_overrides_applied: bool,
    cache_path: Option<String>,
    calibrate_duration_ms: Option<u128>,
    previous_thresholds: Option<ThresholdConfig>,
    calibration_delta: Option<ThresholdDelta>,
}

#[derive(Clone)]
struct DecisionEvaluation {
    recommend_gpu: bool,
    reason: DecisionReason,
    cpu_secs: Option<f64>,
    gpu_secs: Option<f64>,
    threshold: Option<usize>,
    fusion_kind: Option<FusionKind>,
    batch: Option<usize>,
}

struct DecisionLog {
    entries: Vec<AutoOffloadDecisionEntry>,
}

impl DecisionLog {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn push(&mut self, entry: AutoOffloadDecisionEntry) {
        self.entries.push(entry);
        if self.entries.len() > DECISION_LOG_CAPACITY {
            let overflow = self.entries.len() - DECISION_LOG_CAPACITY;
            self.entries.drain(0..overflow);
        }
    }

    fn snapshot(&self) -> Vec<AutoOffloadDecisionEntry> {
        self.entries.clone()
    }

    fn clear(&mut self) {
        self.entries.clear();
    }
}

static DECISION_LOG: Lazy<Mutex<DecisionLog>> = Lazy::new(|| Mutex::new(DecisionLog::new()));
static AUTO_STATE: OnceCell<Mutex<AutoOffloadState>> = OnceCell::new();

fn record_decision(entry: AutoOffloadDecisionEntry) {
    if let Ok(mut log) = DECISION_LOG.lock() {
        log.push(entry);
    }
}

fn snapshot_decisions() -> Vec<AutoOffloadDecisionEntry> {
    DECISION_LOG
        .lock()
        .map(|log| log.snapshot())
        .unwrap_or_default()
}

fn clear_decisions() {
    if let Ok(mut log) = DECISION_LOG.lock() {
        log.clear();
    }
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis()
}

fn threshold_snapshot(cfg: &ThresholdConfig) -> ThresholdSnapshot {
    ThresholdSnapshot {
        unary_min_elems: cfg.unary_min_elems,
        binary_min_elems: cfg.binary_min_elems,
        reduction_min_elems: cfg.reduction_min_elems,
        matmul_min_flops: cfg.matmul_min_flops,
        cpu_elem_per_elem: cfg.cpu_elem_per_elem,
        cpu_reduction_per_elem: cfg.cpu_reduction_per_elem,
        cpu_matmul_per_flop: cfg.cpu_matmul_per_flop,
        small_batch_max_dim: cfg.small_batch_max_dim,
        small_batch_min_elems: cfg.small_batch_min_elems,
    }
}

fn compute_delta(before: &ThresholdConfig, after: &ThresholdConfig) -> ThresholdDelta {
    let mut delta = ThresholdDelta::default();

    if (before.cpu_elem_per_elem - after.cpu_elem_per_elem).abs() > f64::EPSILON {
        delta.cpu_elem_per_elem = Some(ThresholdDeltaEntry::new(
            before.cpu_elem_per_elem,
            after.cpu_elem_per_elem,
        ));
    }

    if (before.cpu_reduction_per_elem - after.cpu_reduction_per_elem).abs() > f64::EPSILON {
        delta.cpu_reduction_per_elem = Some(ThresholdDeltaEntry::new(
            before.cpu_reduction_per_elem,
            after.cpu_reduction_per_elem,
        ));
    }

    if (before.cpu_matmul_per_flop - after.cpu_matmul_per_flop).abs() > f64::EPSILON {
        delta.cpu_matmul_per_flop = Some(ThresholdDeltaEntry::new(
            before.cpu_matmul_per_flop,
            after.cpu_matmul_per_flop,
        ));
    }

    delta
}

#[derive(Debug, Deserialize)]
struct CalibrationFile {
    #[serde(default)]
    suite: Option<CalibrationSuiteSection>,
    #[serde(default)]
    auto_offload_calibration: Option<CalibrationSample>,
}

#[derive(Debug, Deserialize)]
struct CalibrationSuiteSection {
    #[serde(default)]
    auto_offload_calibration: Option<CalibrationSample>,
}

#[derive(Debug, Clone, Deserialize)]
struct CalibrationSample {
    #[serde(default)]
    runs: usize,
    #[serde(default, rename = "cpu_time_ms")]
    cpu_time: CalibrationTimes,
    #[serde(default)]
    units: CalibrationUnits,
    #[serde(default)]
    provider: Option<CalibrationProviderInfo>,
    #[serde(default)]
    provider_conflict: bool,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct CalibrationTimes {
    #[serde(default)]
    elementwise: f64,
    #[serde(default)]
    reduction: f64,
    #[serde(default)]
    matmul: f64,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct CalibrationUnits {
    #[serde(default)]
    elementwise: f64,
    #[serde(default)]
    reduction: f64,
    #[serde(default, rename = "matmul_flops")]
    matmul_flops: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct CalibrationProviderInfo {
    name: String,
    vendor: String,
    #[serde(default)]
    backend: Option<String>,
    device_id: u32,
}

#[derive(Debug, Serialize)]
pub struct AutoOffloadCalibrationOutcome {
    pub runs: usize,
    pub before: ThresholdSnapshot,
    pub after: ThresholdSnapshot,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<ThresholdDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persisted_to: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<CachedProviderInfo>,
    pub commit: bool,
}

fn load_calibration_sample(path: &Path) -> Result<CalibrationSample> {
    let payload = fs::read_to_string(path).map_err(|e| anyhow!(e.to_string()))?;
    let file: CalibrationFile = serde_json::from_str(&payload)
        .map_err(|e| anyhow!(format!("failed to parse calibration file: {e}")))?;
    if let Some(suite) = file.suite {
        if let Some(sample) = suite.auto_offload_calibration {
            return Ok(sample);
        }
    }
    if let Some(sample) = file.auto_offload_calibration {
        return Ok(sample);
    }
    Err(anyhow!(
        "calibration file does not contain an auto_offload_calibration section"
    ))
}

fn apply_calibration_sample(
    cfg: &mut ThresholdConfig,
    sample: &CalibrationSample,
) -> Option<ThresholdDelta> {
    let mut delta = ThresholdDelta::default();
    let mut changed = false;

    if sample.units.elementwise > 0.0 && sample.cpu_time.elementwise > 0.0 {
        let secs_per_elem = (sample.cpu_time.elementwise / 1_000.0) / sample.units.elementwise;
        if secs_per_elem.is_finite() && secs_per_elem > 0.0 {
            if (cfg.cpu_elem_per_elem - secs_per_elem).abs() > f64::EPSILON {
                delta.cpu_elem_per_elem = Some(ThresholdDeltaEntry::new(
                    cfg.cpu_elem_per_elem,
                    secs_per_elem,
                ));
                cfg.cpu_elem_per_elem = secs_per_elem;
                changed = true;
            }
        }
    }

    if sample.units.reduction > 0.0 && sample.cpu_time.reduction > 0.0 {
        let secs_per_elem = (sample.cpu_time.reduction / 1_000.0) / sample.units.reduction;
        if secs_per_elem.is_finite() && secs_per_elem > 0.0 {
            if (cfg.cpu_reduction_per_elem - secs_per_elem).abs() > f64::EPSILON {
                delta.cpu_reduction_per_elem = Some(ThresholdDeltaEntry::new(
                    cfg.cpu_reduction_per_elem,
                    secs_per_elem,
                ));
                cfg.cpu_reduction_per_elem = secs_per_elem;
                changed = true;
            }
        }
    }

    if sample.units.matmul_flops > 0.0 && sample.cpu_time.matmul > 0.0 {
        let secs_per_flop = (sample.cpu_time.matmul / 1_000.0) / sample.units.matmul_flops;
        if secs_per_flop.is_finite() && secs_per_flop > 0.0 {
            if (cfg.cpu_matmul_per_flop - secs_per_flop).abs() > f64::EPSILON {
                delta.cpu_matmul_per_flop = Some(ThresholdDeltaEntry::new(
                    cfg.cpu_matmul_per_flop,
                    secs_per_flop,
                ));
                cfg.cpu_matmul_per_flop = secs_per_flop;
                changed = true;
            }
        }
    }

    if changed {
        Some(delta)
    } else {
        None
    }
}

pub fn apply_auto_offload_calibration_from_file(
    path: &Path,
    commit: bool,
) -> Result<AutoOffloadCalibrationOutcome> {
    let sample = load_calibration_sample(path)?;
    if sample.runs == 0 {
        return Err(anyhow!("calibration sample contains zero runs"));
    }

    let provider = runmat_accelerate_api::provider()
        .ok_or_else(|| anyhow!("no acceleration provider registered"))?;
    let device_info = provider.device_info_struct();

    if let Some(ref prov) = sample.provider {
        if prov.name != device_info.name
            || prov.vendor != device_info.vendor
            || prov.backend.as_deref() != device_info.backend.as_deref()
            || prov.device_id != device_info.device_id
        {
            warn!(
                "Calibration provider mismatch: sample='{} ({})' device='{} ({})'",
                prov.name, prov.vendor, device_info.name, device_info.vendor
            );
        }
        if sample.provider_conflict {
            warn!("Calibration sample reported provider conflict across cases");
        }
    }

    let (mut cfg, _) = load_cached_thresholds(&device_info)
        .unwrap_or_else(|| (ThresholdConfig::default(), PathBuf::new()));
    let before_cfg = cfg.clone();

    let delta = apply_calibration_sample(&mut cfg, &sample)
        .ok_or_else(|| anyhow!("calibration sample did not produce coefficient updates"))?;

    let mut persisted_to: Option<PathBuf> = None;
    if commit {
        persisted_to = Some(persist_thresholds(&device_info, &cfg)?);
    }

    if let Some(state_mutex) = AUTO_STATE.get() {
        if let Ok(mut state) = state_mutex.lock() {
            state.previous_thresholds = Some(before_cfg.clone());
            state.calibration_delta = Some(delta.clone());
            if commit {
                state.thresholds = cfg.clone();
                state.base_source = ThresholdBase::Calibrated;
                if let Some(ref path_buf) = persisted_to {
                    state.cache_path = Some(path_buf.to_string_lossy().into_owned());
                }
                state.calibrate_duration_ms = None;
            }
        }
    }

    Ok(AutoOffloadCalibrationOutcome {
        runs: sample.runs,
        before: threshold_snapshot(&before_cfg),
        after: threshold_snapshot(&cfg),
        delta: Some(delta),
        persisted_to: persisted_to.map(|p| p.to_string_lossy().into_owned()),
        provider: Some(cached_provider_info(&device_info)),
        commit,
    })
}

fn cached_provider_info(info: &ApiDeviceInfo) -> CachedProviderInfo {
    CachedProviderInfo {
        name: info.name.clone(),
        vendor: info.vendor.clone(),
        backend: info.backend.clone(),
        device_id: info.device_id,
    }
}

fn cpu_estimate(per_unit: f64, units: usize) -> Option<f64> {
    if per_unit.is_finite() && per_unit > 0.0 {
        Some(per_unit * units as f64)
    } else {
        None
    }
}

fn value_shape(value: &Value) -> Option<&[usize]> {
    match value {
        Value::Tensor(t) => Some(&t.shape),
        Value::GpuTensor(handle) => Some(&handle.shape),
        _ => None,
    }
}

fn batch_dimension_from_value(value: &Value) -> Option<usize> {
    let shape = value_shape(value)?;
    if shape.len() < 3 {
        return None;
    }
    shape.last().copied()
}

fn batch_dimension_from_values(values: &[&Value]) -> Option<usize> {
    values
        .iter()
        .filter_map(|value| batch_dimension_from_value(value))
        .min()
}

fn decision_entry(
    operation: &str,
    elements: Option<usize>,
    flops: Option<usize>,
    eval: &DecisionEvaluation,
) -> AutoOffloadDecisionEntry {
    AutoOffloadDecisionEntry {
        timestamp_ms: now_millis(),
        operation: operation.to_string(),
        elements,
        flops,
        batch: eval.batch,
        decision: if eval.recommend_gpu {
            AutoOffloadDisposition::Gpu
        } else {
            AutoOffloadDisposition::Cpu
        },
        reason: eval.reason,
        cpu_estimate_ms: eval.cpu_secs.map(|secs| secs * 1_000.0),
        gpu_estimate_ms: eval.gpu_secs.map(|secs| secs * 1_000.0),
        threshold: eval.threshold,
        fusion_kind: eval.fusion_kind.clone(),
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
        clear_decisions();
        return None;
    }
    let provider = runmat_accelerate_api::provider()?;
    let device_info = provider.device_info_struct();
    let mut config = ThresholdConfig::default();
    let mut base_source = ThresholdBase::BuiltInDefault;
    let mut cache_path: Option<PathBuf> = None;
    let mut calibrate_duration_ms: Option<u128> = None;
    let refresh_calibration = calibrate_refresh_enabled();

    if !refresh_calibration {
        if let Some((cached, path)) = load_cached_thresholds(&device_info) {
            info!(
                "Native auto-offload: loaded cached calibration for '{}' from {}",
                device_info.name,
                path.display()
            );
            config = cached;
            cache_path = Some(path);
            base_source = ThresholdBase::LoadedFromCache;
        }
    }

    let needs_calibration = calibrate_enabled() && (refresh_calibration || cache_path.is_none());
    if needs_calibration {
        let start = Instant::now();
        match auto_calibrate(provider, &mut config) {
            Ok(()) => {
                calibrate_duration_ms = Some(start.elapsed().as_millis());
                base_source = ThresholdBase::Calibrated;
                match persist_thresholds(&device_info, &config) {
                    Ok(path) => {
                        cache_path = Some(path.clone());
                        info!(
                            "Native auto-offload: persisted calibration for '{}' to {}",
                            device_info.name,
                            path.display()
                        );
                    }
                    Err(err) => {
                        debug!("Native auto-offload: failed to persist calibration: {err}");
                    }
                }
            }
            Err(err) => {
                debug!("Native auto-offload calibration failed: {err}");
            }
        }
    }

    let env_overrides_applied = apply_env_overrides(&mut config);
    let model_status = if profile_cost_model().is_some() {
        "profile"
    } else {
        "fallback"
    };
    info!(
        "Native auto-offload thresholds: unary={} binary={} reduction={} matmul_flops={} small_batch_dim={} small_batch_min_elems={} (model: {}, source: {}, env_overrides={})",
        config.unary_min_elems,
        config.binary_min_elems,
        config.reduction_min_elems,
        config.matmul_min_flops,
        config.small_batch_max_dim,
        config.small_batch_min_elems,
        model_status,
        base_source.as_str(),
        env_overrides_applied
    );

    let cache_path_str = cache_path
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned());
    let state = AutoOffloadState {
        provider: Some(cached_provider_info(&device_info)),
        thresholds: config.clone(),
        base_source,
        env_overrides_applied,
        cache_path: cache_path_str,
        calibrate_duration_ms,
        previous_thresholds: None,
        calibration_delta: None,
    };
    let _ = AUTO_STATE.set(Mutex::new(state));

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

    fn small_batch_guard(&self, elements: usize, batch: Option<usize>) -> bool {
        if !self.enabled {
            return false;
        }
        let Some(batch) = batch else {
            return false;
        };
        if batch == 0 {
            return false;
        }
        let thresholds = &self.thresholds;
        thresholds.small_batch_max_dim > 0
            && thresholds.small_batch_min_elems > 0
            && batch <= thresholds.small_batch_max_dim
            && elements >= thresholds.small_batch_min_elems
    }

    fn promote_binary(&self, op: BinaryOp, a: &Value, b: &Value) -> Result<(Value, Value)> {
        if !self.enabled {
            return Ok((a.clone(), b.clone()));
        }
        match op {
            BinaryOp::Elementwise => {
                let elems = element_count_pair(a, b).unwrap_or(0);
                let eval = self.evaluate_elementwise(elems, &[a, b]);
                record_decision(decision_entry("elementwise", Some(elems), None, &eval));
                if eval.recommend_gpu {
                    log_promotion(|| format!("Elementwise offload accepted ({} elems)", elems));
                    let a_p = self.promote_tensor_if_large(a, 1)?;
                    let b_p = self.promote_tensor_if_large(b, 1)?;
                    Ok((a_p, b_p))
                } else {
                    Ok((a.clone(), b.clone()))
                }
            }
            BinaryOp::MatMul => {
                if let (Some((ra, ca)), Some((rb, cb))) = (tensor_rows_cols(a), tensor_rows_cols(b))
                {
                    if ca != rb {
                        return Ok((a.clone(), b.clone()));
                    }
                    let flops = ra.saturating_mul(ca).saturating_mul(cb);
                    let eval = self.evaluate_matmul(flops);
                    record_decision(decision_entry("matmul", None, Some(flops), &eval));
                    if eval.recommend_gpu {
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
        let eval = self.evaluate_unary(elems, op, v);
        let op_label = match op {
            UnaryOp::Transpose => "transpose",
            UnaryOp::Generic => "unary",
        };
        record_decision(decision_entry(op_label, Some(elems), None, &eval));
        if eval.recommend_gpu {
            log_promotion(|| format!("Unary offload accepted ({:?}, {} elems)", op, elems));
            self.promote_tensor_if_large(v, 1)
        } else {
            Ok(v.clone())
        }
    }

    fn promote_reduction(&self, _op: ReductionOp, args: &[Value]) -> Result<Vec<Value>> {
        if !self.enabled || args.is_empty() {
            return Ok(args.to_vec());
        }
        let elems = value_len(&args[0]).unwrap_or(0);
        let eval = self.evaluate_reduction(elems);
        record_decision(decision_entry("reduction", Some(elems), None, &eval));
        if !eval.recommend_gpu {
            return Ok(args.to_vec());
        }
        log_promotion(|| format!("Reduction offload accepted ({} elems)", elems));
        let mut out = Vec::with_capacity(args.len());
        if let Some(first) = args.first() {
            out.push(self.promote_tensor_if_large(first, 1)?);
            out.extend(args.iter().skip(1).cloned());
        }
        Ok(out)
    }

    fn evaluate_elementwise(&self, elements: usize, values: &[&Value]) -> DecisionEvaluation {
        let fusion = active_fusion();
        let fusion_kind = fusion.as_ref().map(|f| f.kind.clone());
        let batch = batch_dimension_from_values(values);
        let cpu_secs = cpu_estimate(self.thresholds.cpu_elem_per_elem, elements);

        if let Some(active) = fusion.as_ref() {
            if active.kind.is_elementwise() && active.supported {
                return DecisionEvaluation {
                    recommend_gpu: true,
                    reason: DecisionReason::FusionOverride,
                    cpu_secs,
                    gpu_secs: None,
                    threshold: Some(self.thresholds.binary_min_elems),
                    fusion_kind,
                    batch,
                };
            }
        }

        if self.small_batch_guard(elements, batch) {
            return DecisionEvaluation {
                recommend_gpu: false,
                reason: DecisionReason::SmallBatchGuard,
                cpu_secs,
                gpu_secs: None,
                threshold: Some(self.thresholds.binary_min_elems),
                fusion_kind,
                batch,
            };
        }

        if let Some(model) = profile_cost_model() {
            if let Some(gpu_duration) = model.estimate_elemwise(elements) {
                let gpu_secs = Some(gpu_duration.as_secs_f64());
                let cpu = cpu_secs.unwrap_or(f64::INFINITY);
                let recommend = gpu_duration.as_secs_f64() * 0.95 < cpu;
                return DecisionEvaluation {
                    recommend_gpu: recommend,
                    reason: DecisionReason::ProfileModel,
                    cpu_secs,
                    gpu_secs,
                    threshold: Some(self.thresholds.binary_min_elems),
                    fusion_kind,
                    batch,
                };
            }
        }

        DecisionEvaluation {
            recommend_gpu: elements >= self.thresholds.binary_min_elems,
            reason: DecisionReason::Threshold,
            cpu_secs,
            gpu_secs: None,
            threshold: Some(self.thresholds.binary_min_elems),
            fusion_kind,
            batch,
        }
    }

    fn evaluate_matmul(&self, flops: usize) -> DecisionEvaluation {
        let cpu_secs = cpu_estimate(self.thresholds.cpu_matmul_per_flop, flops);
        if let Some(model) = profile_cost_model() {
            if let Some(gpu_duration) = model.estimate_matmul_flops(flops) {
                let gpu_secs = Some(gpu_duration.as_secs_f64());
                let cpu = cpu_secs.unwrap_or(f64::INFINITY);
                let recommend = gpu_duration.as_secs_f64() * 0.95 < cpu;
                return DecisionEvaluation {
                    recommend_gpu: recommend,
                    reason: DecisionReason::ProfileModel,
                    cpu_secs,
                    gpu_secs,
                    threshold: Some(self.thresholds.matmul_min_flops),
                    fusion_kind: None,
                    batch: None,
                };
            }
        }

        DecisionEvaluation {
            recommend_gpu: flops >= self.thresholds.matmul_min_flops,
            reason: DecisionReason::Threshold,
            cpu_secs,
            gpu_secs: None,
            threshold: Some(self.thresholds.matmul_min_flops),
            fusion_kind: None,
            batch: None,
        }
    }

    fn evaluate_reduction(&self, elements: usize) -> DecisionEvaluation {
        let fusion_kind = active_fusion().map(|f| f.kind.clone());
        let cpu_secs = cpu_estimate(self.thresholds.cpu_reduction_per_elem, elements);
        if let Some(model) = profile_cost_model() {
            if let Some(gpu_duration) = model.estimate_reduction(elements) {
                let gpu_secs = Some(gpu_duration.as_secs_f64());
                let cpu = cpu_secs.unwrap_or(f64::INFINITY);
                let recommend = gpu_duration.as_secs_f64() * 0.95 < cpu;
                return DecisionEvaluation {
                    recommend_gpu: recommend,
                    reason: DecisionReason::ProfileModel,
                    cpu_secs,
                    gpu_secs,
                    threshold: Some(self.thresholds.reduction_min_elems),
                    fusion_kind,
                    batch: None,
                };
            }
        }

        DecisionEvaluation {
            recommend_gpu: elements >= self.thresholds.reduction_min_elems,
            reason: DecisionReason::Threshold,
            cpu_secs,
            gpu_secs: None,
            threshold: Some(self.thresholds.reduction_min_elems),
            fusion_kind,
            batch: None,
        }
    }

    fn evaluate_unary(&self, elements: usize, op: UnaryOp, value: &Value) -> DecisionEvaluation {
        let fusion_kind = active_fusion().map(|f| f.kind.clone());
        let batch = batch_dimension_from_values(&[value]);
        if matches!(op, UnaryOp::Generic) && self.small_batch_guard(elements, batch) {
            return DecisionEvaluation {
                recommend_gpu: false,
                reason: DecisionReason::SmallBatchGuard,
                cpu_secs: cpu_estimate(self.thresholds.cpu_elem_per_elem, elements),
                gpu_secs: None,
                threshold: Some(self.thresholds.unary_min_elems),
                fusion_kind,
                batch,
            };
        }

        let cpu_secs = cpu_estimate(self.thresholds.cpu_elem_per_elem, elements);
        if let Some(model) = profile_cost_model() {
            let gpu_duration = match op {
                UnaryOp::Transpose => model.estimate_transpose(elements),
                UnaryOp::Generic => model.estimate_elemwise(elements),
            };
            if let Some(gpu_duration) = gpu_duration {
                let gpu_secs = Some(gpu_duration.as_secs_f64());
                let cpu = cpu_secs.unwrap_or(f64::INFINITY);
                let recommend = gpu_duration.as_secs_f64() * 0.95 < cpu;
                return DecisionEvaluation {
                    recommend_gpu: recommend,
                    reason: DecisionReason::ProfileModel,
                    cpu_secs,
                    gpu_secs,
                    threshold: Some(self.thresholds.unary_min_elems),
                    fusion_kind,
                    batch,
                };
            }
        }

        DecisionEvaluation {
            recommend_gpu: elements >= self.thresholds.unary_min_elems,
            reason: DecisionReason::Threshold,
            cpu_secs,
            gpu_secs: None,
            threshold: Some(self.thresholds.unary_min_elems),
            fusion_kind,
            batch,
        }
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

fn calibrate_refresh_enabled() -> bool {
    env_bool("RUNMAT_ACCEL_CALIBRATE_REFRESH").unwrap_or(false)
}

fn apply_env_overrides(cfg: &mut ThresholdConfig) -> bool {
    let mut applied = false;
    if let Some(val) = env_usize("RUNMAT_ACCEL_THRESHOLD_UNARY") {
        cfg.unary_min_elems = val;
        applied = true;
    }
    if let Some(val) = env_usize("RUNMAT_ACCEL_THRESHOLD_ELEMWISE") {
        cfg.binary_min_elems = val;
        applied = true;
    }
    if let Some(val) = env_usize("RUNMAT_ACCEL_THRESHOLD_REDUCTION") {
        cfg.reduction_min_elems = val;
        applied = true;
    }
    if let Some(val) = env_usize("RUNMAT_ACCEL_THRESHOLD_MATMUL") {
        cfg.matmul_min_flops = val;
        applied = true;
    }
    if let Some(val) = env_usize("RUNMAT_ACCEL_THRESHOLD_ALL") {
        cfg.unary_min_elems = val;
        cfg.binary_min_elems = val;
        cfg.reduction_min_elems = val;
        applied = true;
    }
    if let Some(val) = env_usize("RUNMAT_ACCEL_SMALL_BATCH_MAX_DIM") {
        cfg.small_batch_max_dim = val;
        applied = true;
    }
    if let Some(val) = env_usize("RUNMAT_ACCEL_SMALL_BATCH_MIN_ELEMS") {
        cfg.small_batch_min_elems = val;
        applied = true;
    }
    applied
}

fn env_usize(key: &str) -> Option<usize> {
    env::var(key).ok().and_then(|v| v.parse::<usize>().ok())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CalibrationRecord {
    version: u32,
    recorded_at: u64,
    provider: CalibrationProviderDetails,
    thresholds: ThresholdConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CalibrationProviderDetails {
    name: String,
    vendor: String,
    backend: Option<String>,
    device_id: u32,
}

fn load_cached_thresholds(info: &ApiDeviceInfo) -> Option<(ThresholdConfig, PathBuf)> {
    let path = calibration_cache_file(info)?;
    let contents = fs::read_to_string(&path).ok()?;
    match serde_json::from_str::<CalibrationRecord>(&contents) {
        Ok(record) => {
            if record.version != CALIBRATION_VERSION {
                debug!(
                    "Native auto-offload calibration cache version mismatch (found {}, expected {})",
                    record.version,
                    CALIBRATION_VERSION
                );
                None
            } else {
                Some((record.thresholds, path))
            }
        }
        Err(err) => {
            debug!(
                "Native auto-offload failed to parse cached calibration for '{}': {err}",
                info.name
            );
            None
        }
    }
}

fn persist_thresholds(info: &ApiDeviceInfo, cfg: &ThresholdConfig) -> Result<PathBuf> {
    let path = calibration_cache_file(info)
        .ok_or_else(|| anyhow!("unable to determine calibration cache directory"))?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| anyhow!(e.to_string()))?;
    }
    let record = CalibrationRecord {
        version: CALIBRATION_VERSION,
        recorded_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_secs(0))
            .as_secs(),
        provider: CalibrationProviderDetails {
            name: info.name.clone(),
            vendor: info.vendor.clone(),
            backend: info.backend.clone(),
            device_id: info.device_id,
        },
        thresholds: cfg.clone(),
    };
    let payload = serde_json::to_string_pretty(&record).map_err(|e| anyhow!(e.to_string()))?;
    fs::write(&path, payload).map_err(|e| anyhow!(e.to_string()))?;
    Ok(path)
}

fn calibration_cache_file(info: &ApiDeviceInfo) -> Option<PathBuf> {
    let mut dir = calibration_cache_dir()?;
    let vendor = slugify(&info.vendor);
    let name = slugify(&info.name);
    let backend = slugify(info.backend.as_deref().unwrap_or("unknown"));
    let file = format!("{}-{}-{}-{}.json", vendor, name, backend, info.device_id);
    dir.push(file);
    Some(dir)
}

fn calibration_cache_dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|base| base.join("runmat").join("auto_offload"))
}

fn slugify(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut last_underscore = false;
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_underscore = false;
        } else if !last_underscore {
            out.push('_');
            last_underscore = true;
        }
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        "device".to_string()
    } else {
        trimmed.to_string()
    }
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

pub fn auto_offload_report() -> Option<AutoOffloadReport> {
    let state_guard = AUTO_STATE.get()?;
    let state = state_guard.lock().ok()?;
    let calibration = state.previous_thresholds.as_ref().map(|prev| {
        let delta = state
            .calibration_delta
            .clone()
            .unwrap_or_else(|| compute_delta(prev, &state.thresholds));
        AutoOffloadCalibrationSummary {
            previous: threshold_snapshot(prev),
            delta,
        }
    });
    Some(AutoOffloadReport {
        provider: state.provider.clone(),
        thresholds: threshold_snapshot(&state.thresholds),
        base_source: state.base_source,
        env_overrides_applied: state.env_overrides_applied,
        cache_path: state.cache_path.clone(),
        calibrate_duration_ms: state.calibrate_duration_ms,
        calibration,
        decisions: snapshot_decisions(),
    })
}

pub fn reset_auto_offload_log() {
    clear_decisions();
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
