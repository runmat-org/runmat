use runmat_time::{duration_ns_saturating, system_time_now, Instant};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, UNIX_EPOCH};

use crate::{
    auto_offload_options,
    fusion::{active_fusion, FusionKind},
    fusion_residency,
    placement::{
        plan_local_vm_cpu_with_provider, LocalCpuAdmission, LocalCpuCandidate,
        LocalPlacementOutcome, LocalPlacementRequest, LocalProviderAdmission, PlacementAttribute,
        PlacementEventKind, PlacementPolicy, PlacementTraceContext, PlacementVariant,
    },
    precision::ensure_provider_supports_dtype,
    AutoOffloadLogLevel,
};
use anyhow::{anyhow, Result};
use futures::lock::Mutex as AsyncMutex;
use log::{debug, info, trace};
use once_cell::sync::{Lazy, OnceCell};
use runmat_accelerate_api::{
    AccelProvider, ApiDeviceInfo, GpuTensorHandle, HostTensorView, ProviderOperationFamily,
    ProviderPrecision, ProviderRejectionCode, ProviderWorkload,
};
use runmat_accelerate_api::{ProviderExecutionError, ProviderExecutionResult};
use runmat_builtins::{builtin_functions, AccelTag};
use runmat_execution::{
    CandidatePreparationState, EstimateConfidence, EstimateSource, ExecutionCandidateKind,
    ExecutionCostComponents, ExecutionCostEstimate,
};
use runmat_runtime::builtins::common::{
    spec::{builtin_residency_policy, ResidencyPolicy},
    tensor::tensor_element_len,
};
use runmat_runtime::gather_if_needed_async;
use runmat_value::{Tensor, Value};
use serde::{Deserialize, Serialize};

mod builtin_call;
mod calibration_cleanup;
mod calibration_identity;
mod refresh;
pub use calibration_identity::{CalibrationSource, CalibrationUnitPolicy};
pub use refresh::refresh_auto_offload_calibration;
mod host_route;
mod state;
mod original_work;
pub use builtin_call::{install_preparation_service, AdmittedAcceleration};
pub use host_route::{
    promote_binary_in_context, promote_builtin_binary_in_context, promote_builtin_unary_in_context,
};
pub use state::AutoOffloadHandle;

const DEFAULT_CPU_ELEM_PER_ELEM: f64 = 1.0e-7;
const DEFAULT_CPU_REDUCTION_PER_ELEM: f64 = 1.2e-7;
const DEFAULT_CPU_MATMUL_PER_FLOP: f64 = 2.5e-11;
const SMALL_BATCH_DEFAULT_MAX_DIM: usize = 8;
const SMALL_BATCH_DEFAULT_MIN_ELEMS: usize = 1_048_576;
const DECISION_LOG_CAPACITY: usize = 128;
const CALIBRATION_VERSION: u32 = 2;

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
    Residency,
    SmallBatchGuard,
    Threshold,
    Disabled,
    ProviderRejected,
    CostModel,
    CostUnavailable,
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

fn record_decision(entry: AutoOffloadDecisionEntry) -> PlacementTraceContext {
    #[cfg(test)]
    tests::capture_decision(&entry);
    let trace = crate::placement::begin_trace(&entry.operation, None);
    let attributes = decision_attributes(&entry);
    trace.event(
        PlacementEventKind::Candidate,
        Some(PlacementVariant::SharedRuntime),
        Some("legacy_cpu_candidate"),
        entry.cpu_estimate_ms.and_then(milliseconds_to_nanoseconds),
        None,
        &attributes,
    );
    trace.event(
        PlacementEventKind::Candidate,
        Some(PlacementVariant::ProviderOperation),
        Some("legacy_provider_candidate"),
        entry.gpu_estimate_ms.and_then(milliseconds_to_nanoseconds),
        None,
        &attributes,
    );
    let selected = if entry.decision == AutoOffloadDisposition::Gpu {
        PlacementVariant::ProviderOperation
    } else {
        PlacementVariant::SharedRuntime
    };
    trace.event(
        PlacementEventKind::Selected,
        Some(selected),
        Some(decision_reason_token(entry.reason)),
        None,
        None,
        &attributes,
    );
    if let Ok(mut log) = DECISION_LOG.lock() {
        log.push(entry);
    }
    trace
}

fn decision_attributes(entry: &AutoOffloadDecisionEntry) -> Vec<PlacementAttribute> {
    [
        (
            "elements",
            entry.elements.and_then(|value| u64::try_from(value).ok()),
        ),
        (
            "flops",
            entry.flops.and_then(|value| u64::try_from(value).ok()),
        ),
        (
            "batch",
            entry.batch.and_then(|value| u64::try_from(value).ok()),
        ),
        (
            "legacy_threshold",
            entry.threshold.and_then(|value| u64::try_from(value).ok()),
        ),
    ]
    .into_iter()
    .filter_map(|(key, value)| {
        value.map(|value| PlacementAttribute {
            key: key.to_string(),
            value,
        })
    })
    .collect()
}

fn decision_reason_token(reason: DecisionReason) -> &'static str {
    match reason {
        DecisionReason::FusionOverride => "legacy_fusion_override",
        DecisionReason::Residency => "legacy_residency",
        DecisionReason::SmallBatchGuard => "legacy_small_batch_guard",
        DecisionReason::Threshold => "legacy_threshold",
        DecisionReason::Disabled => "legacy_disabled",
        DecisionReason::ProviderRejected => "provider_feasibility_rejected",
        DecisionReason::CostModel => "unified_cost_model",
        DecisionReason::CostUnavailable => "comparison_cost_unavailable",
    }
}

fn milliseconds_to_nanoseconds(milliseconds: f64) -> Option<u64> {
    let nanoseconds = milliseconds * 1_000_000.0;
    nanoseconds
        .is_finite()
        .then(|| nanoseconds.max(0.0).min(u64::MAX as f64) as u64)
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
    system_time_now()
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
    schema_version: u32,
    #[serde(default)]
    precision: Option<ProviderPrecision>,
    #[serde(default)]
    unit_policy: Option<calibration_identity::CalibrationUnitPolicy>,
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
    pub schema_version: u32,
    pub precision: ProviderPrecision,
    pub runs: Option<usize>,
    pub source: CalibrationSource,
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
        if secs_per_elem.is_finite()
            && secs_per_elem > 0.0
            && (cfg.cpu_elem_per_elem - secs_per_elem).abs() > f64::EPSILON
        {
            delta.cpu_elem_per_elem = Some(ThresholdDeltaEntry::new(
                cfg.cpu_elem_per_elem,
                secs_per_elem,
            ));
            cfg.cpu_elem_per_elem = secs_per_elem;
            changed = true;
        }
    }

    if sample.units.reduction > 0.0 && sample.cpu_time.reduction > 0.0 {
        let secs_per_elem = (sample.cpu_time.reduction / 1_000.0) / sample.units.reduction;
        if secs_per_elem.is_finite()
            && secs_per_elem > 0.0
            && (cfg.cpu_reduction_per_elem - secs_per_elem).abs() > f64::EPSILON
        {
            delta.cpu_reduction_per_elem = Some(ThresholdDeltaEntry::new(
                cfg.cpu_reduction_per_elem,
                secs_per_elem,
            ));
            cfg.cpu_reduction_per_elem = secs_per_elem;
            changed = true;
        }
    }

    if sample.units.matmul_flops > 0.0 && sample.cpu_time.matmul > 0.0 {
        let secs_per_flop = (sample.cpu_time.matmul / 1_000.0) / sample.units.matmul_flops;
        if secs_per_flop.is_finite()
            && secs_per_flop > 0.0
            && (cfg.cpu_matmul_per_flop - secs_per_flop).abs() > f64::EPSILON
        {
            delta.cpu_matmul_per_flop = Some(ThresholdDeltaEntry::new(
                cfg.cpu_matmul_per_flop,
                secs_per_flop,
            ));
            cfg.cpu_matmul_per_flop = secs_per_flop;
            changed = true;
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

    let (mut cfg, _) = load_cached_thresholds(&device_info, provider.precision())
        .unwrap_or_else(|| (ThresholdConfig::default(), PathBuf::new()));
    let before_cfg = cfg.clone();
    if sample.unit_policy != Some(CalibrationUnitPolicy::OriginalScanMacV1) {
        if commit {
            return Err(anyhow!("residual telemetry is diagnostic, not original CPU calibration; use accel-calibrate --refresh to measure the current provider's authoritative templates"));
        }
        return Ok(AutoOffloadCalibrationOutcome {
            schema_version: CALIBRATION_VERSION,
            precision: provider.precision(),
            runs: Some(sample.runs),
            source: CalibrationSource::ResidualDiagnostic,
            before: threshold_snapshot(&cfg),
            after: threshold_snapshot(&cfg),
            delta: None,
            persisted_to: None,
            provider: Some(cached_provider_info(&device_info)),
            commit: false,
        });
    }
    sample.require_current(&device_info, provider.precision())?;

    let delta = apply_calibration_sample(&mut cfg, &sample)
        .ok_or_else(|| anyhow!("calibration sample did not produce coefficient updates"))?;

    let mut persisted_to: Option<PathBuf> = None;
    if commit {
        persisted_to = Some(persist_thresholds(
            &device_info,
            &cfg,
            provider.precision(),
        )?);
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
        schema_version: CALIBRATION_VERSION,
        precision: provider.precision(),
        runs: Some(sample.runs),
        source: CalibrationSource::ImportedOriginalTemplates,
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

fn seconds_to_ns(seconds: f64) -> Option<u64> {
    // Preserve a known positive stage below clock granularity as one tick,
    // rather than round it into a claimed zero-cost nonempty operation.
    milliseconds_to_nanoseconds(seconds * 1_000.0)
        .map(|ns| if seconds > 0.0 { ns.max(1) } else { ns })
}

fn ns_to_seconds(nanoseconds: u64) -> f64 {
    nanoseconds as f64 / 1_000_000_000.0
}

fn value_shape(value: &Value) -> Option<&[usize]> {
    match value {
        Value::Int(_) | Value::Num(_) | Value::Complex(_, _) | Value::Bool(_) => Some(&[1, 1]),
        Value::Tensor(t) => Some(t.shape()),
        Value::ComplexTensor(t) => Some(t.shape()),
        Value::LogicalArray(array) => Some(&array.shape),
        Value::CharArray(array) => Some(array.shape()),
        Value::GpuTensor(handle) => Some(handle.shape()),
        _ => None,
    }
}

fn batch_dimension_from_value(value: &Value) -> Option<usize> {
    // Geometry is not numeric admission. Preserve the existing batch heuristic
    // boundary when other admitted callers borrow logical/complex/text shapes.
    if !matches!(value, Value::Tensor(_) | Value::GpuTensor(_)) {
        return None;
    }
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

static GLOBAL_INIT_LOCK: Lazy<AsyncMutex<()>> = Lazy::new(|| AsyncMutex::new(()));

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
        Value::Tensor(t) => Some(tensor_element_len(t)),
        Value::GpuTensor(handle) => Some(handle.shape().iter().product()),
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

fn values_are_provider_resident(values: &[&Value]) -> bool {
    values
        .iter()
        .all(|value| matches!(value, Value::GpuTensor(_)))
}

fn promoted_binary_variant(a: &Value, b: &Value) -> PlacementVariant {
    if values_are_provider_resident(&[a, b]) {
        PlacementVariant::ProviderOperation
    } else {
        PlacementVariant::SharedRuntime
    }
}

fn promoted_unary_variant(value: &Value) -> PlacementVariant {
    if matches!(value, Value::GpuTensor(_)) {
        PlacementVariant::ProviderOperation
    } else {
        PlacementVariant::SharedRuntime
    }
}

fn record_provider_rejection(
    trace: &PlacementTraceContext,
    rejection: Option<ProviderRejectionCode>,
) {
    if let Some(rejection) = rejection {
        trace.event(
            PlacementEventKind::Fallback,
            Some(PlacementVariant::SharedRuntime),
            Some(crate::placement::provider_rejection_token(rejection)),
            None,
            None,
            &[],
        );
    }
}

pub async fn global() -> Option<AutoOffloadHandle> {
    if !auto_enabled() {
        return None;
    }
    let active_provider = runmat_accelerate_api::provider()?;
    if let Some(existing) = state::cached() {
        return existing.filter(|offload| {
            runmat_accelerate_api::same_provider_instance(offload.provider, active_provider)
        });
    }
    // The early checks above must not cache an absent provider. Initialization
    // can suspend while another caller registers one. Serialize initialization
    // and retain configuration in the same process/worker scope as its provider.
    let _guard = GLOBAL_INIT_LOCK.lock().await;
    if let Some(existing) = state::cached() {
        return existing.filter(|offload| {
            runmat_accelerate_api::same_provider_instance(offload.provider, active_provider)
        });
    }
    let initialized = initialize_async(active_provider).await;
    state::initialize(initialized);
    state::cached().flatten()
}

async fn initialize_async(provider: &'static dyn AccelProvider) -> Option<NativeAutoOffload> {
    if !auto_enabled() {
        clear_decisions();
        return None;
    }
    let device_info = provider.device_info_struct();
    let mut config = ThresholdConfig::default();
    let mut base_source = ThresholdBase::BuiltInDefault;
    let mut cache_path: Option<String> = None;
    let mut calibrate_duration_ms: Option<u128> = None;
    let refresh_calibration = calibrate_refresh_enabled();

    if !refresh_calibration {
        if let Some((cached, path)) =
            load_cached_thresholds_async(&device_info, provider.precision()).await
        {
            info!(
                "Native auto-offload: loaded cached calibration for '{}' from {}",
                device_info.name, path
            );
            config = cached;
            cache_path = Some(path);
            base_source = ThresholdBase::LoadedFromCache;
        }
    }

    let needs_calibration = calibrate_enabled() && (refresh_calibration || cache_path.is_none());
    if needs_calibration {
        let start = Instant::now();
        match auto_calibrate(provider, &mut config).await {
            Ok(()) => {
                calibrate_duration_ms = Some(start.elapsed().as_millis());
                base_source = ThresholdBase::Calibrated;
                match persist_thresholds_async(&device_info, &config, provider.precision()).await {
                    Ok(path) => {
                        cache_path = Some(path.clone());
                        info!(
                            "Native auto-offload: persisted calibration for '{}' to {}",
                            device_info.name, path
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
    let model_status = "canonical-placement-session; cold-priors";
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

    let cache_path_str = cache_path.clone();
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

    fn promote_tensor_if_large_observed(
        &self,
        value: &Value,
        threshold: usize,
        trace: Option<&PlacementTraceContext>,
    ) -> ProviderExecutionResult<Value> {
        self.promote_tensor_protecting(value, threshold, trace, &[])
    }

    fn promote_tensor_protecting(
        &self,
        value: &Value,
        threshold: usize,
        trace: Option<&PlacementTraceContext>,
        protected: &[&GpuTensorHandle],
    ) -> ProviderExecutionResult<Value> {
        match value {
            Value::GpuTensor(_) => Ok(value.clone()),
            Value::Tensor(t) => {
                // Exact integer buffers are a residency/transfer capability, not a
                // floating-kernel precision. Try the provider's native numeric
                // transfer below; an older provider that cannot represent the class
                // leaves the value on the host rather than coercing it.
                if t.integer_storage().is_none()
                    && ensure_provider_supports_dtype(self.provider, t.numeric_dtype()).is_err()
                {
                    return Ok(value.clone());
                }
                let element_count = tensor_element_len(t);
                if element_count >= threshold && threshold > 0 {
                    log_promotion(|| {
                        format!(
                            "Promoting tensor to GPU (len={}, threshold={})",
                            element_count, threshold
                        )
                    });
                    let started = Instant::now();
                    match self.tensor_to_gpu(t, protected) {
                        Ok(value) => {
                            if let Some(trace) = trace {
                                trace.event(
                                    PlacementEventKind::Upload,
                                    Some(PlacementVariant::ProviderOperation),
                                    None,
                                    Some(duration_ns_saturating(started.elapsed())),
                                    t.len()
                                        .checked_mul(t.numeric_dtype().byte_size())
                                        .and_then(|bytes| u64::try_from(bytes).ok()),
                                    &[],
                                );
                            }
                            Ok(value)
                        }
                        Err(error) if t.integer_storage().is_some() && error.is_unsupported() => {
                            if let Some(trace) = trace {
                                trace.event(
                                    PlacementEventKind::Fallback,
                                    Some(PlacementVariant::SharedRuntime),
                                    Some("provider_integer_upload_unsupported"),
                                    Some(duration_ns_saturating(started.elapsed())),
                                    None,
                                    &[],
                                );
                            }
                            debug!(
                                "Auto-offload retaining integer tensor on host: provider has no exact integer upload ({error})"
                            );
                            Ok(value.clone())
                        }
                        Err(error) => {
                            if let Some(trace) = trace {
                                trace.event(
                                    PlacementEventKind::Fallback,
                                    Some(PlacementVariant::SharedRuntime),
                                    Some("provider_upload_failed"),
                                    Some(duration_ns_saturating(started.elapsed())),
                                    None,
                                    &[],
                                );
                            }
                            Err(error)
                        }
                    }
                } else {
                    Ok(value.clone())
                }
            }
            _ => Ok(value.clone()),
        }
    }

    fn tensor_to_gpu(
        &self,
        tensor: &Tensor,
        protected: &[&GpuTensorHandle],
    ) -> ProviderExecutionResult<Value> {
        let handle = runmat_runtime::builtins::common::gpu_helpers::upload_tensor_protecting_typed(
            self.provider,
            tensor,
            protected,
        )
        .map_err(ProviderExecutionError::from_source)?;
        let handle = handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .map_err(ProviderExecutionError::from_source)?;
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

    fn promote_binary(
        &self,
        op: BinaryOp,
        a: &Value,
        b: &Value,
    ) -> ProviderExecutionResult<(Value, Value)> {
        if !self.enabled {
            return Ok((a.clone(), b.clone()));
        }
        match op {
            BinaryOp::Elementwise => {
                let elems = element_count_pair(a, b).unwrap_or(0);
                let eval = self.evaluate_elementwise(elems, &[a, b]);
                let (eval, rejection) = self.enforce_provider_feasibility(
                    eval,
                    "legacy.elementwise",
                    ProviderOperationFamily::Elementwise,
                    &[a, b],
                    ProviderWorkload {
                        elements: u64::try_from(elems).ok(),
                        ..ProviderWorkload::default()
                    },
                )?;
                let trace =
                    record_decision(decision_entry("elementwise", Some(elems), None, &eval));
                record_provider_rejection(&trace, rejection);
                if eval.recommend_gpu {
                    log_promotion(|| format!("Elementwise offload accepted ({} elems)", elems));
                    let a_p = self.promote_tensor_if_large_observed(a, 1, Some(&trace))?;
                    let b_p = self.promote_tensor_if_large_observed(b, 1, Some(&trace))?;
                    trace.complete(Some(promoted_binary_variant(&a_p, &b_p)), None);
                    Ok((a_p, b_p))
                } else {
                    trace.complete(Some(PlacementVariant::SharedRuntime), None);
                    Ok((a.clone(), b.clone()))
                }
            }
            BinaryOp::MatMul => {
                if let (Some((ra, ca)), Some((rb, cb))) = (tensor_rows_cols(a), tensor_rows_cols(b))
                {
                    if ca != rb {
                        return Ok((a.clone(), b.clone()));
                    }
                    let Some(flops) = ra.checked_mul(ca).and_then(|n| n.checked_mul(cb)) else {
                        return Ok((a.clone(), b.clone()));
                    };
                    let mut eval = self.evaluate_matmul(flops);
                    eval.cpu_secs = original_work::matrix(self, ra, ca, cb);
                    let (eval, rejection) = self.enforce_provider_feasibility(
                        eval,
                        "legacy.matmul",
                        ProviderOperationFamily::MatrixMultiply,
                        &[a, b],
                        ProviderWorkload {
                            flops: u64::try_from(flops).ok(),
                            ..ProviderWorkload::default()
                        },
                    )?;
                    let trace = record_decision(decision_entry("matmul", None, Some(flops), &eval));
                    record_provider_rejection(&trace, rejection);
                    if eval.recommend_gpu {
                        log_promotion(|| {
                            format!(
                                "Promoting matmul operands (flops={}, threshold={})",
                                flops, self.thresholds.matmul_min_flops
                            )
                        });
                        let a_p = self.promote_tensor_if_large_observed(a, 1, Some(&trace))?;
                        let b_p = self.promote_tensor_if_large_observed(b, 1, Some(&trace))?;
                        let variant = if values_are_provider_resident(&[&a_p, &b_p]) {
                            PlacementVariant::ProviderLibrary
                        } else {
                            PlacementVariant::SharedRuntime
                        };
                        trace.complete(Some(variant), None);
                        return Ok((a_p, b_p));
                    }
                    trace.complete(Some(PlacementVariant::SharedRuntime), None);
                }
                Ok((a.clone(), b.clone()))
            }
        }
    }

    fn promote_unary(&self, op: UnaryOp, v: &Value) -> ProviderExecutionResult<Value> {
        if !self.enabled {
            return Ok(v.clone());
        }
        let elems = value_len(v).unwrap_or(0);
        let eval = self.evaluate_unary(elems, op, v);
        let op_label = match op {
            UnaryOp::Transpose => "transpose",
            UnaryOp::Generic => "unary",
        };
        let (eval, rejection) = self.enforce_provider_feasibility(
            eval,
            "legacy.elementwise",
            ProviderOperationFamily::Elementwise,
            &[v],
            ProviderWorkload {
                elements: u64::try_from(elems).ok(),
                ..ProviderWorkload::default()
            },
        )?;
        let trace = record_decision(decision_entry(op_label, Some(elems), None, &eval));
        record_provider_rejection(&trace, rejection);
        if eval.recommend_gpu {
            log_promotion(|| format!("Unary offload accepted ({:?}, {} elems)", op, elems));
            let value = self.promote_tensor_if_large_observed(v, 1, Some(&trace))?;
            trace.complete(Some(promoted_unary_variant(&value)), None);
            Ok(value)
        } else {
            trace.complete(Some(PlacementVariant::SharedRuntime), None);
            Ok(v.clone())
        }
    }

    fn promote_reduction(
        &self,
        _op: ReductionOp,
        args: &[Value],
    ) -> ProviderExecutionResult<Vec<Value>> {
        if !self.enabled || args.is_empty() {
            return Ok(args.to_vec());
        }
        let elems = value_len(&args[0]).unwrap_or(0);
        let eval = self.evaluate_reduction(elems);
        let (eval, rejection) = self.enforce_provider_feasibility(
            eval,
            "legacy.reduction",
            ProviderOperationFamily::Reduction,
            &[&args[0]],
            ProviderWorkload {
                elements: u64::try_from(elems).ok(),
                ..ProviderWorkload::default()
            },
        )?;
        let trace = record_decision(decision_entry("reduction", Some(elems), None, &eval));
        record_provider_rejection(&trace, rejection);
        if !eval.recommend_gpu {
            trace.complete(Some(PlacementVariant::SharedRuntime), None);
            return Ok(args.to_vec());
        }
        log_promotion(|| format!("Reduction offload accepted ({} elems)", elems));
        let mut out = Vec::with_capacity(args.len());
        if let Some(first) = args.first() {
            out.push(self.promote_tensor_if_large_observed(first, 1, Some(&trace))?);
            out.extend(args.iter().skip(1).cloned());
        }
        let variant = out
            .first()
            .map(promoted_unary_variant)
            .unwrap_or(PlacementVariant::SharedRuntime);
        trace.complete(Some(variant), None);
        Ok(out)
    }

    fn enforce_provider_feasibility(
        &self,
        evaluation: DecisionEvaluation,
        operation: &'static str,
        family: ProviderOperationFamily,
        values: &[&Value],
        workload: ProviderWorkload,
    ) -> ProviderExecutionResult<(DecisionEvaluation, Option<ProviderRejectionCode>)> {
        let (evaluation, rejection, _) = self.plan_operation(
            evaluation,
            operation,
            family,
            values,
            workload,
            Default::default(),
        )?;
        Ok((evaluation, rejection))
    }

    fn plan_operation(
        &self,
        mut evaluation: DecisionEvaluation,
        operation: &str,
        family: ProviderOperationFamily,
        values: &[&Value],
        workload: ProviderWorkload,
        planning: builtin_call::observation::BuiltinPlanningContext<'_>,
    ) -> ProviderExecutionResult<(
        DecisionEvaluation,
        Option<ProviderRejectionCode>,
        builtin_call::observation::BuiltinObservationPlan,
    )> {
        let host = planning.host;
        let cpu_cost = evaluation
            .cpu_secs
            .and_then(seconds_to_ns)
            .map(|cpu_ns| ExecutionCostEstimate {
                components: ExecutionCostComponents {
                    execution_ns: cpu_ns,
                    ..ExecutionCostComponents::default()
                },
                scratch_bytes: Some(host.map_or(0, |(resources, _)| resources.scratch_bytes)),
                confidence: EstimateConfidence::Low,
                source: EstimateSource::StaticPrior,
            })
            .map_or(
                runmat_execution::CandidateCostEvidence::Unavailable(
                    runmat_execution::CandidateCostUnavailable::NoEstimate,
                ),
                runmat_execution::CandidateCostEvidence::Estimated,
            );
        let mut cpu = LocalCpuCandidate::host(format!("cpu.shared.{operation}"), cpu_cost);
        if let Some((host, _)) = host {
            cpu = cpu.with_presentation(host.presentation.clone());
        }
        let provider_kind = match family {
            ProviderOperationFamily::MatrixMultiply => ExecutionCandidateKind::ProviderLibrary,
            _ if evaluation.fusion_kind.is_some() => ExecutionCandidateKind::ProviderFusion,
            _ => ExecutionCandidateKind::ProviderOperation,
        };
        let provider_fallback = runmat_execution::CandidateCostEvidence::Unavailable(
            runmat_execution::CandidateCostUnavailable::NoEstimate,
        );
        let sum_preparation_unknown = workload.sum.is_some();
        // These are the operation's data leaves, never unrelated controls.
        // Synchronous callers have no admitted resident-to-host route. The
        // async preparation owner may supply exact private readback evidence;
        // VM-certified overlays already arrive here as host values.
        let cpu_admission = if let Some((host, _)) = host {
            LocalCpuAdmission::ExactReadbacks {
                resources: Box::new(host.resources),
                download_bytes: host.download_bytes,
                readback_count: host.readback_count,
            }
        } else if values
            .iter()
            .any(|value| runmat_runtime::dispatcher::value_contains_gpu(value))
        {
            LocalCpuAdmission::Unavailable
        } else {
            LocalCpuAdmission::Legacy
        };
        let original_cast_decline = matches!(&cpu_admission, LocalCpuAdmission::Unavailable)
            && planning.source_decline == crate::placement::LocalSourceDecline::PlainFloatingCast;
        Ok(
            match plan_local_vm_cpu_with_provider(
                self.provider,
                LocalPlacementRequest {
                    compound: planning.compound,
                    operation,
                    semantic_identity: host.map(|(_, identity)| identity),
                    observation_contract: if planning.prepared_authority.is_some() {
                        crate::placement::TotalCandidateObservationContract::FullCall
                    } else {
                        crate::placement::TotalCandidateObservationContract::PreparationBodyCleanup
                    },
                    prepared_authority: planning.prepared_authority,
                    source_decline: planning.source_decline,
                    family,
                    provider_kind,
                    inputs: values,
                    outputs: Vec::new(),
                    workload,
                    preparation: if sum_preparation_unknown {
                        // No exact cache snapshot for the ordinary sum's kernel
                        // inventory yet; charging Warm would assert an unseen hit.
                        CandidatePreparationState::Cold
                    } else {
                        CandidatePreparationState::Warm
                    },
                    preparation_program: planning.preparation_program,
                    shader_requirements: None,
                    cpu,
                    provider_fallback,
                    required_upload_bytes: None,
                    required_download_bytes: 0,
                    downstream_materialization: false,
                    compare_profitability: true,
                    runtime: planning.runtime,
                },
                PlacementPolicy::default(),
                None,
                None,
                cpu_admission,
                LocalProviderAdmission::Available,
            ) {
                Ok(LocalPlacementOutcome::ProviderRejected {
                    code,
                    cpu_cost,
                    feedback,
                }) => {
                    evaluation.recommend_gpu = false;
                    evaluation.reason = DecisionReason::ProviderRejected;
                    evaluation.cpu_secs = cpu_cost
                        .estimate()
                        .and_then(|cost| cost.checked_total_ns())
                        .map(ns_to_seconds);
                    (
                        evaluation,
                        Some(code),
                        builtin_call::observation::BuiltinObservationPlan {
                            feedback,
                            source_declined: original_cast_decline,
                            ..Default::default()
                        },
                    )
                }
                Ok(LocalPlacementOutcome::Selected {
                    selected,
                    cpu_cost,
                    provider_cost,
                    provider_rejection,
                    decision_kind,
                    predicted_total_ns,
                    feedback,
                    provider_query,
                    provider_observation,
                    ..
                }) => {
                    evaluation.recommend_gpu = selected.kind.is_provider();
                    evaluation.reason = if provider_rejection.is_some() {
                        DecisionReason::ProviderRejected
                    } else if decision_kind == runmat_execution::WholeWorkDecisionKind::Compared {
                        DecisionReason::CostModel
                    } else {
                        DecisionReason::CostUnavailable
                    };
                    evaluation.cpu_secs = cpu_cost
                        .estimate()
                        .and_then(|cost| cost.checked_total_ns())
                        .map(ns_to_seconds);
                    evaluation.gpu_secs = if selected.kind.is_provider() {
                        predicted_total_ns
                    } else {
                        provider_cost.and_then(|cost| cost.checked_total_ns())
                    }
                    .map(ns_to_seconds);
                    (
                        evaluation,
                        provider_rejection,
                        builtin_call::observation::BuiltinObservationPlan {
                            feedback,
                            provider_query: Some(provider_query),
                            provider_observation,
                            source_declined: false,
                        },
                    )
                }
                Err(error) => {
                    let mut error = *error;
                    error.gpu_gather_retry = runmat_runtime::GpuGatherRetry::Never;
                    return Err(ProviderExecutionError::from_source(error).with_unproven_effects());
                }
            },
        )
    }

    fn evaluate_elementwise(&self, elements: usize, values: &[&Value]) -> DecisionEvaluation {
        let fusion = active_fusion();
        let fusion_kind = fusion.as_ref().map(|f| f.kind.clone());
        let batch = batch_dimension_from_values(values);
        let cpu_secs = cpu_estimate(self.thresholds.cpu_elem_per_elem, elements);

        // Chain-aware residency: if any input is already on GPU, keep the op on GPU
        if values.iter().any(|v| matches!(v, Value::GpuTensor(_))) {
            return DecisionEvaluation {
                recommend_gpu: true,
                reason: DecisionReason::Residency,
                cpu_secs,
                gpu_secs: None,
                threshold: Some(self.thresholds.binary_min_elems),
                fusion_kind,
                batch,
            };
        }

        if let Some(active) = fusion.as_ref() {
            // If an elementwise chain is actively fused OR this elementwise op
            // participates in a fused reduction group, force GPU to keep the
            // whole chain resident and avoid host round-trips.
            if (active.kind.is_elementwise() || active.kind.is_reduction()) && active.supported {
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
        // Input cardinality alone cannot tell whether an empty reduction
        // produces empty output or a nonempty fill. Admitted Sum supplies its
        // checked publication geometry; unadmitted legacy calls stay ordinary.
        let cpu_secs = (elements != 0)
            .then(|| cpu_estimate(self.thresholds.cpu_reduction_per_elem, elements))
            .flatten();

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
        // Chain-aware residency for unary ops: if operand is already on GPU, keep it on GPU
        if matches!(value, Value::GpuTensor(_)) {
            return DecisionEvaluation {
                recommend_gpu: true,
                reason: DecisionReason::Residency,
                cpu_secs: cpu_estimate(self.thresholds.cpu_elem_per_elem, elements),
                gpu_secs: None,
                threshold: Some(self.thresholds.unary_min_elems),
                fusion_kind,
                batch,
            };
        }
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
}

fn tensor_rows_cols(value: &Value) -> Option<(usize, usize)> {
    match value {
        Value::Tensor(t) => Some((t.rows(), t.cols())),
        Value::GpuTensor(handle) => {
            if handle.shape().len() == 2 {
                Some((handle.shape()[0], handle.shape()[1]))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn should_skip_reduction_promotion(name: &str, args: &[Value]) -> bool {
    (name.eq_ignore_ascii_case("max") || name.eq_ignore_ascii_case("min"))
        && !max_or_min_reduction_call(args)
}

fn max_or_min_reduction_call(args: &[Value]) -> bool {
    if args.len() <= 1 {
        return true;
    }
    args.get(1).map(is_empty_placeholder_value).unwrap_or(false)
}

fn is_empty_placeholder_value(value: &Value) -> bool {
    match value {
        Value::Tensor(t) => tensor_element_len(t) == 0,
        Value::LogicalArray(l) => l.data.is_empty(),
        Value::StringArray(sa) => sa.data().is_empty(),
        Value::CharArray(ca) => ca.data().is_empty(),
        Value::Cell(cell) => cell.data().is_empty(),
        Value::String(_) => false,
        _ => false,
    }
}

async fn gather_args(args: &[Value]) -> ProviderExecutionResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for (idx, value) in args.iter().enumerate() {
        if let Value::GpuTensor(handle) = value {
            trace!(
                "auto-offload: gather_args arg[{}]=GpuTensor device_id={} buffer_id={} shape={:?}",
                idx,
                handle.device_id(),
                handle.buffer_id(),
                handle.shape()
            );
        } else {
            trace!(
                "auto-offload: gather_args arg[{}]={:?}",
                idx,
                value_kind(value)
            );
        }
        let gathered = gather_if_needed_async(value)
            .await
            .map_err(ProviderExecutionError::from_source)?;
        trace!(
            "auto-offload: gather_args arg[{}] -> {:?}",
            idx,
            value_kind(&gathered)
        );
        out.push(gathered);
    }
    Ok(out)
}

fn should_gather_sink_args(name: &str) -> bool {
    matches!(
        builtin_residency_policy(name),
        Some(ResidencyPolicy::GatherImmediately) | None
    )
}

fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::GpuTensor(_) => "GpuTensor",
        Value::Tensor(_) => "Tensor",
        Value::SparseTensor(_) => "SparseTensor",
        Value::Num(_) => "Num",
        Value::Int(_) => "Int",
        Value::Bool(_) => "Bool",
        Value::LogicalArray(_) => "LogicalArray",
        Value::CharArray(_) => "CharArray",
        Value::String(_) => "String",
        Value::StringArray(_) => "StringArray",
        Value::Symbolic(_) => "Symbolic",
        Value::SymbolicArray(_) => "SymbolicArray",
        Value::Cell(_) => "Cell",
        Value::Struct(_) => "Struct",
        Value::StructArray(_) => "StructArray",
        Value::ObjectArray(_) => "ObjectArray",
        Value::Object(_) => "Object",
        Value::HandleObject(_) => "HandleObject",
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_) => "FunctionHandle",
        Value::BoundFunctionHandle(_) | Value::ScopedFunctionHandle(_) => "FunctionHandle",
        Value::Closure(_) => "Closure",
        Value::ClassRef(_) => "ClassRef",
        Value::Complex(_, _) => "Complex",
        Value::ComplexTensor(_) => "ComplexTensor",
        Value::Listener(_) => "Listener",
        Value::MException(_) => "MException",
        Value::OutputList(_) => "OutputList",
        Value::Future(_) => "Future",
        Value::Task(_) => "Task",
        Value::Pool(_) => "Pool",
        Value::Job(_) => "Job",
        Value::Distributed(_) => "Distributed",
        Value::Composite(_) => "Composite",
        Value::Foreign(_) => "Foreign",
    }
}

#[cfg(test)]
#[path = "native_auto/upload_tests.rs"]
mod upload_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::test_support::TestAllocationOwner;
    use runmat_accelerate_api::{
        GpuTensorHandle, HostNumericDataOwned, NumericElementType, ProviderCostEstimate,
        ProviderCostQuery, ProviderFeasibility, ProviderFeasibilityQuery, ProviderResourceEstimate,
    };
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::OnceLock;

    thread_local! {
        // These promotion APIs execute synchronously. Capture only the caller's
        // invocation; the public diagnostic log remains process-wide.
        static CAPTURED_DECISIONS: RefCell<Option<Vec<AutoOffloadDecisionEntry>>> = const { RefCell::new(None) };
    }

    pub(super) fn capture_decision(entry: &AutoOffloadDecisionEntry) {
        CAPTURED_DECISIONS.with(|captured| {
            if let Some(entries) = captured.borrow_mut().as_mut() {
                entries.push(entry.clone());
            }
        });
    }

    fn capture_promotion<T>(body: impl FnOnce() -> T) -> (T, Vec<AutoOffloadDecisionEntry>) {
        struct CaptureGuard;
        impl Drop for CaptureGuard {
            fn drop(&mut self) {
                CAPTURED_DECISIONS.with(|captured| *captured.borrow_mut() = None);
            }
        }
        CAPTURED_DECISIONS.with(|captured| {
            let mut captured = captured.borrow_mut();
            assert!(captured.is_none(), "promotion capture must not be nested");
            *captured = Some(Vec::new());
        });
        let _guard = CaptureGuard;
        let result = body();
        let entries = CAPTURED_DECISIONS.with(|captured| captured.borrow_mut().take().unwrap());
        (result, entries)
    }

    struct RejectingProvider {
        uploads: AtomicUsize,
    }

    impl AccelProvider for RejectingProvider {
        fn upload(
            &self,
            _host: &HostTensorView<'_>,
        ) -> Result<runmat_accelerate_api::GpuTensorHandle> {
            self.uploads.fetch_add(1, Ordering::Relaxed);
            Err(anyhow!(
                "upload must not execute after feasibility rejection"
            ))
        }

        fn download<'a>(
            &'a self,
            _handle: &'a runmat_accelerate_api::GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            Box::pin(async { Err(anyhow!("unused")) })
        }

        fn device_info(&self) -> String {
            "rejecting-test-provider".to_string()
        }
    }

    static REJECTING_PROVIDER: RejectingProvider = RejectingProvider {
        uploads: AtomicUsize::new(0),
    };

    struct CostedProvider {
        uploads: AtomicUsize,
        owner: OnceLock<TestAllocationOwner>,
        quote: bool,
    }

    impl CostedProvider {
        fn owner(&self) -> &TestAllocationOwner {
            self.owner.get_or_init(TestAllocationOwner::new)
        }
    }

    impl AccelProvider for CostedProvider {
        fn upload(&self, host: &HostTensorView<'_>) -> Result<GpuTensorHandle> {
            self.uploads.fetch_add(1, Ordering::Relaxed);
            Ok(self
                .owner()
                .allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
                    data: HostNumericDataOwned::F64(host.data.to_vec()),
                    shape: host.shape.to_vec(),
                    storage: runmat_accelerate_api::GpuTensorStorage::Real,
                })?
                .handle)
        }

        fn download<'a>(
            &'a self,
            _handle: &'a GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            Box::pin(async { Err(anyhow!("unused")) })
        }

        fn device_info(&self) -> String {
            "costed-test-provider".to_string()
        }

        fn device_id(&self) -> u32 {
            self.owner().device_id()
        }

        fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
            ProviderFeasibility::supported(ProviderResourceEstimate {
                output_bytes: Some(
                    query
                        .workload
                        .elements
                        .expect("fixture element count")
                        .checked_mul(8)
                        .expect("F64 fixture output bytes"),
                ),
                ..Default::default()
            })
        }

        fn estimate_cost(&self, query: &ProviderCostQuery) -> Option<ProviderCostEstimate> {
            if !self.quote {
                return None;
            }
            Some(ProviderCostEstimate {
                cost: ExecutionCostEstimate {
                    components: ExecutionCostComponents {
                        upload_ns: query.required_upload_bytes.saturating_mul(1_000),
                        execution_ns: 1,
                        ..ExecutionCostComponents::default()
                    },
                    scratch_bytes: Some(0),
                    confidence: EstimateConfidence::Exact,
                    source: EstimateSource::Provider,
                },
            })
        }
    }

    static COSTED_PROVIDER: CostedProvider = CostedProvider {
        uploads: AtomicUsize::new(0),
        owner: OnceLock::new(),
        quote: true,
    };

    static UNQUOTED_PROVIDER: CostedProvider = CostedProvider {
        uploads: AtomicUsize::new(0),
        owner: OnceLock::new(),
        quote: false,
    };

    #[test]
    fn missing_provider_quotes_ignore_threshold_prices_and_preserve_explicit_intent() {
        for threshold in [1, usize::MAX] {
            for cpu_price in [DEFAULT_CPU_ELEM_PER_ELEM, f64::NAN] {
                let auto = NativeAutoOffload::new(
                    &UNQUOTED_PROVIDER,
                    ThresholdConfig {
                        binary_min_elems: threshold,
                        cpu_elem_per_elem: cpu_price,
                        ..ThresholdConfig::default()
                    },
                );
                let tensor = Value::Tensor(Tensor::new(vec![1.; 8], vec![1, 8]).unwrap());
                let ((left, right), decisions) = capture_promotion(|| {
                    auto.promote_binary(BinaryOp::Elementwise, &tensor, &tensor)
                        .unwrap()
                });
                assert!(matches!(left, Value::Tensor(_)) && matches!(right, Value::Tensor(_)));
                assert_eq!(decisions[0].reason, DecisionReason::CostUnavailable);
                assert!(decisions[0].gpu_estimate_ms.is_none());
                assert_eq!(
                    decisions[0].cpu_estimate_ms.is_some(),
                    cpu_price.is_finite()
                );

                let allocation = UNQUOTED_PROVIDER
                    .owner()
                    .allocate_zeroed(
                        vec![1, 8],
                        NumericElementType::F64,
                        runmat_accelerate_api::GpuTensorStorage::Real,
                    )
                    .unwrap();
                let explicit = allocation
                    .handle
                    .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                    .unwrap();
                let value = Value::GpuTensor(explicit.clone());
                let (result, decisions) = capture_promotion(|| {
                    auto.promote_binary(BinaryOp::Elementwise, &value, &value)
                });
                // This promotion-only fixture declares no transactional body
                // capability. Explicit intent cannot turn that absence into a
                // legal execution by returning unchanged GPU operands.
                let error =
                    result.expect_err("mandatory resident route requires physical admission");
                let mut cause: &(dyn std::error::Error + 'static) = &error;
                let original = loop {
                    if let Some(runtime) = cause.downcast_ref::<runmat_runtime::RuntimeError>() {
                        break runtime;
                    }
                    cause = cause
                        .source()
                        .expect("original typed planning failure retained");
                };
                assert_eq!(
                    original.identifier(),
                    Some("RunMat:Placement:NoLegalCandidate")
                );
                assert_eq!(
                    original.gpu_gather_retry,
                    runmat_runtime::GpuGatherRetry::Never
                );
                assert!(
                    decisions.is_empty(),
                    "no Disabled/success decision disguises refusal"
                );
                assert!(runmat_accelerate_api::handle_is_explicit(&explicit));
                explicit.allocation_lease().validate_live().unwrap();
            }
        }
        assert_eq!(UNQUOTED_PROVIDER.uploads.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn max_detection_handles_placeholders() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let placeholder = Tensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap();
        let data = Value::Tensor(tensor);
        let empty = Value::Tensor(placeholder);

        assert!(max_or_min_reduction_call(std::slice::from_ref(&data)));
        assert!(max_or_min_reduction_call(&[
            data.clone(),
            empty.clone(),
            Value::Num(1.0)
        ]));
        assert!(!max_or_min_reduction_call(&[data.clone(), Value::Num(0.0)]));
    }

    #[test]
    fn auto_promotion_uploads_native_numeric_storage_without_conversion() {
        crate::simple_provider::register_inprocess_provider();
        let provider = runmat_accelerate_api::provider().expect("in-process provider");
        let auto = NativeAutoOffload::new(provider, ThresholdConfig::default());
        let expected = runmat_value::IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]);
        let tensor = Tensor::new_integer(expected.clone(), vec![1, 2]).expect("integer tensor");

        let promoted = auto
            .promote_tensor_if_large_observed(&Value::Tensor(tensor), 1, None)
            .expect("exact integer promotion");
        let Value::GpuTensor(handle) = promoted else {
            panic!("expected resident integer gpuArray");
        };
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&handle),
            Some(runmat_accelerate_api::IntegerElementType::U64)
        );
        assert_eq!(
            futures::executor::block_on(provider.download_numeric(&handle))
                .expect("exact integer download")
                .data,
            HostNumericDataOwned::U64(vec![1_u64 << 63, u64::MAX])
        );
        drop(handle);

        let single = Tensor::from_f32(vec![1.25, -2.5], vec![1, 2]).expect("single tensor");
        let promoted = auto
            .promote_tensor_if_large_observed(&Value::Tensor(single), 1, None)
            .expect("native single promotion");
        let Value::GpuTensor(handle) = promoted else {
            panic!("expected resident single gpuArray");
        };
        let downloaded = futures::executor::block_on(provider.download_numeric(&handle))
            .expect("native single download");
        assert_eq!(downloaded.data.element_type(), NumericElementType::F32);
        assert_eq!(downloaded.data, HostNumericDataOwned::F32(vec![1.25, -2.5]));
        drop(handle);
    }

    #[test]
    fn quoted_nontransactional_provider_keeps_legal_host_work_without_upload() {
        COSTED_PROVIDER.uploads.store(0, Ordering::Relaxed);
        let auto = NativeAutoOffload::new(&COSTED_PROVIDER, ThresholdConfig::default());
        let tensor = Value::Tensor(Tensor::new(vec![1.0; 128], vec![1, 128]).unwrap());
        let ((left, right), decisions) = capture_promotion(|| {
            auto.promote_binary(BinaryOp::Elementwise, &tensor, &tensor)
                .unwrap()
        });
        assert!(matches!(left, Value::Tensor(_)) && matches!(right, Value::Tensor(_)));
        assert_eq!(COSTED_PROVIDER.uploads.load(Ordering::Relaxed), 0);
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].operation, "elementwise");
        assert_eq!(decisions[0].decision, AutoOffloadDisposition::Cpu);
        assert_eq!(decisions[0].reason, DecisionReason::CostUnavailable);
    }

    #[test]
    fn decision_capture_unwinds_without_leaking_into_the_next_invocation() {
        assert!(std::panic::catch_unwind(|| {
            capture_promotion(|| panic!("interrupted test invocation"));
        })
        .is_err());
        let ((), decisions) = capture_promotion(|| ());
        assert!(decisions.is_empty());
    }

    #[test]
    fn quoted_nontransactional_resident_route_is_rejected_without_retiring_inputs() {
        let left = COSTED_PROVIDER
            .owner()
            .allocate_zeroed(
                vec![1, 128],
                NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let right = COSTED_PROVIDER
            .owner()
            .allocate_zeroed(
                vec![1, 128],
                NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let left_weak = left.handle.downgrade();
        let right_weak = right.handle.downgrade();
        let left_probe = left.probe;
        let right_probe = right.probe;
        let auto = NativeAutoOffload::new(&COSTED_PROVIDER, ThresholdConfig::default());
        let left = Value::GpuTensor(left.handle);
        let right = Value::GpuTensor(right.handle);
        let (result, decisions) =
            capture_promotion(|| auto.promote_binary(BinaryOp::Elementwise, &left, &right));
        let error = result.expect_err("a quote does not prove transactional execution");
        let mut cause: &(dyn std::error::Error + 'static) = &error;
        let original = loop {
            if let Some(runtime) = cause.downcast_ref::<runmat_runtime::RuntimeError>() {
                break runtime;
            }
            cause = cause
                .source()
                .expect("original typed planning failure retained");
        };
        assert_eq!(
            original.identifier(),
            Some("RunMat:Placement:NoLegalCandidate")
        );
        assert_eq!(
            original.gpu_gather_retry,
            runmat_runtime::GpuGatherRetry::Never
        );
        assert!(
            matches!(left, Value::GpuTensor(ref handle) if handle.shares_allocation_with(&left_weak.acquire().unwrap()))
        );
        assert!(
            matches!(right, Value::GpuTensor(ref handle) if handle.shares_allocation_with(&right_weak.acquire().unwrap()))
        );
        assert!(decisions.is_empty());
        assert_eq!(left_probe.retirements(), 0);
        assert_eq!(right_probe.retirements(), 0);
        drop((left, right));
        assert_eq!(left_probe.retirements(), 1);
        assert_eq!(right_probe.retirements(), 1);
    }

    #[test]
    fn selected_elementwise_builtin_retains_resident_inputs_for_its_async_owner() {
        use runmat_accelerate_api::GpuHandleProvenance;

        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        for provenance in [
            GpuHandleProvenance::Automatic,
            GpuHandleProvenance::Explicit,
        ] {
            let allocation = COSTED_PROVIDER
                .owner()
                .allocate_zeroed(
                    vec![1, 2],
                    NumericElementType::F32,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .unwrap();
            let input = allocation.handle.with_provenance(provenance).unwrap();
            let original = Value::GpuTensor(input.clone());
            let (result, decisions) = capture_promotion(|| {
                futures::executor::block_on(promote_builtin_binary_in_context(
                    &runtime,
                    BinaryOp::Elementwise,
                    &original,
                    &Value::Num(2.0),
                ))
                .expect("selected builtin owns the exact resident route")
            });
            let (returned, scalar) = result;
            let Value::GpuTensor(returned) = returned else {
                panic!("automatic promotion must not materialize the builtin input");
            };
            assert!(returned.shares_allocation_with(&input));
            assert_eq!(
                runmat_accelerate_api::handle_provenance(&returned),
                Some(provenance)
            );
            assert!(matches!(scalar, Value::Num(2.0)));
            assert!(decisions.is_empty(), "no generic route may be selected");
        }
    }

    #[test]
    fn native_auto_uses_integer_storage_for_lengths_and_empty_placeholders() {
        let wide = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![u64::MAX, 7]),
            vec![1, 2],
        )
        .expect("wide integer tensor");
        assert_eq!(value_len(&Value::Tensor(wide.clone())), Some(2));
        assert!(!is_empty_placeholder_value(&Value::Tensor(wide)));

        let signed = Tensor::new_integer(
            runmat_value::IntegerStorage::I64(vec![i64::MIN, -1, i64::MAX]),
            vec![1, 3],
        )
        .expect("signed integer tensor");
        assert_eq!(value_len(&Value::Tensor(signed.clone())), Some(3));
        assert!(!is_empty_placeholder_value(&Value::Tensor(signed)));

        let empty = Tensor::new_integer(runmat_value::IntegerStorage::U64(Vec::new()), vec![0, 0])
            .expect("empty integer placeholder");
        assert_eq!(value_len(&Value::Tensor(empty.clone())), Some(0));
        assert!(is_empty_placeholder_value(&Value::Tensor(empty)));
    }

    #[test]
    fn feasibility_rejection_prevents_provider_execution_and_is_observable() {
        let _recorder = crate::placement::ScopedRecorder::new();
        REJECTING_PROVIDER.uploads.store(0, Ordering::Relaxed);
        let thresholds = ThresholdConfig {
            binary_min_elems: 1,
            ..ThresholdConfig::default()
        };
        let auto = NativeAutoOffload::new(&REJECTING_PROVIDER, thresholds);
        let tensor = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());

        let (left, right) = auto
            .promote_binary(BinaryOp::Elementwise, &tensor, &tensor)
            .unwrap();

        assert!(matches!(left, Value::Tensor(_)));
        assert!(matches!(right, Value::Tensor(_)));
        assert_eq!(REJECTING_PROVIDER.uploads.load(Ordering::Relaxed), 0);
        let report = crate::placement::report();
        assert_eq!(
            report.traces.len(),
            1,
            "this invocation owns exactly one trace"
        );
        let trace = &report.traces[0];
        assert!(trace.events.iter().any(|event| {
            event.kind == PlacementEventKind::Fallback
                && event.reason.as_deref() == Some("provider_operation_unsupported")
        }));
        assert!(trace.complete);
    }
}

#[derive(Clone, Copy)]
struct BuiltinPolicy {
    accel_tags: &'static [AccelTag],
    is_sink: bool,
}

static BUILTIN_POLICIES: OnceCell<HashMap<String, BuiltinPolicy>> = OnceCell::new();

fn build_builtin_policy_map() -> HashMap<String, BuiltinPolicy> {
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
}

fn builtin_policy(name: &str) -> Option<BuiltinPolicy> {
    let map = BUILTIN_POLICIES.get_or_init(build_builtin_policy_map);
    map.get(&name.to_ascii_lowercase()).copied()
}

pub(crate) fn auto_enabled() -> bool {
    if let Some(policy) = runmat_runtime::context::current_acceleration_policy() {
        return policy.automatic_offload_enabled();
    }
    if let Some(flag) = env_bool("RUNMAT_ACCEL_AUTO_OFFLOAD") {
        return flag;
    }
    auto_offload_options().enabled
}

fn calibrate_enabled() -> bool {
    if let Some(policy) = runmat_runtime::context::current_acceleration_policy() {
        return policy.calibrate;
    }
    if let Some(flag) = env_bool("RUNMAT_ACCEL_CALIBRATE") {
        return flag;
    }
    auto_offload_options().calibrate
}

fn calibrate_refresh_enabled() -> bool {
    if runmat_runtime::context::current_acceleration_policy().is_some() {
        // A resolved host/worker policy must not acquire a second ambient
        // calibration authority while the invocation is running.
        return false;
    }
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
    precision: ProviderPrecision,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CalibrationProviderDetails {
    name: String,
    vendor: String,
    backend: Option<String>,
    device_id: u32,
}

#[cfg(target_arch = "wasm32")]
fn calibration_cache_key(info: &ApiDeviceInfo, precision: ProviderPrecision) -> String {
    let vendor = slugify(&info.vendor);
    let name = slugify(&info.name);
    let backend = slugify(info.backend.as_deref().unwrap_or("unknown"));
    format!(
        "v2-{}-{}-{}-{}-{:?}.json",
        vendor, name, backend, info.device_id, precision
    )
}

async fn load_cached_thresholds_async(
    info: &ApiDeviceInfo,
    precision: ProviderPrecision,
) -> Option<(ThresholdConfig, String)> {
    #[cfg(target_arch = "wasm32")]
    {
        let key = calibration_cache_key(info, precision);
        let contents = crate::web_auto_offload_store::load(&key).await?;
        match serde_json::from_str::<CalibrationRecord>(&contents) {
            Ok(record) => {
                if !record.matches(info, precision) {
                    debug!(
                        "Native auto-offload calibration cache version mismatch (found {}, expected {})",
                        record.version,
                        CALIBRATION_VERSION
                    );
                    None
                } else {
                    Some((record.thresholds, key))
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
    #[cfg(not(target_arch = "wasm32"))]
    {
        load_cached_thresholds(info, precision).map(|(cfg, path)| (cfg, path.display().to_string()))
    }
}

async fn persist_thresholds_async(
    info: &ApiDeviceInfo,
    cfg: &ThresholdConfig,
    precision: ProviderPrecision,
) -> Result<String> {
    #[cfg(target_arch = "wasm32")]
    {
        let key = calibration_cache_key(info, precision);
        let record = CalibrationRecord {
            precision,
            version: CALIBRATION_VERSION,
            recorded_at: system_time_now()
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
        crate::web_auto_offload_store::save(&key, &payload)
            .await
            .map_err(|e| anyhow!(format!("indexeddb persist failed: {e:?}")))?;
        Ok(key)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        persist_thresholds(info, cfg, precision).map(|path| path.display().to_string())
    }
}

fn load_cached_thresholds(
    info: &ApiDeviceInfo,
    precision: ProviderPrecision,
) -> Option<(ThresholdConfig, PathBuf)> {
    let path = calibration_cache_file(info, precision)?;
    let contents = fs::read_to_string(&path).ok()?;
    match serde_json::from_str::<CalibrationRecord>(&contents) {
        Ok(record) => {
            if !record.matches(info, precision) {
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

fn persist_thresholds(
    info: &ApiDeviceInfo,
    cfg: &ThresholdConfig,
    precision: ProviderPrecision,
) -> Result<PathBuf> {
    let path = calibration_cache_file(info, precision)
        .ok_or_else(|| anyhow!("unable to determine calibration cache directory"))?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| anyhow!(e.to_string()))?;
    }
    let record = CalibrationRecord {
        precision,
        version: CALIBRATION_VERSION,
        recorded_at: system_time_now()
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

fn calibration_cache_file(info: &ApiDeviceInfo, precision: ProviderPrecision) -> Option<PathBuf> {
    let mut dir = calibration_cache_dir()?;
    let vendor = slugify(&info.vendor);
    let name = slugify(&info.name);
    let backend = slugify(info.backend.as_deref().unwrap_or("unknown"));
    let file = format!(
        "v2-{}-{}-{}-{}-{:?}.json",
        vendor, name, backend, info.device_id, precision
    );
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

async fn auto_calibrate(
    provider: &'static dyn AccelProvider,
    cfg: &mut ThresholdConfig,
) -> ProviderExecutionResult<()> {
    if let Some(elem_threshold) = calibrate_elemwise(provider, cfg).await.transpose()? {
        if elem_threshold != usize::MAX {
            cfg.binary_min_elems = elem_threshold;
            cfg.unary_min_elems = cfg.unary_min_elems.min(elem_threshold);
        }
    }
    if let Some(red_threshold) = calibrate_reduction(provider, cfg).await.transpose()? {
        if red_threshold != usize::MAX {
            cfg.reduction_min_elems = red_threshold;
        }
    }
    if let Some(matmul_threshold) = calibrate_matmul(provider, cfg).await.transpose()? {
        if matmul_threshold != usize::MAX {
            cfg.matmul_min_flops = matmul_threshold;
        }
    }
    Ok(())
}

async fn calibrate_elemwise(
    provider: &'static dyn AccelProvider,
    cfg: &mut ThresholdConfig,
) -> Option<ProviderExecutionResult<usize>> {
    let sizes = [256usize, 1_024, 4_096, 16_384, 65_536];
    for size in sizes {
        match compare_elemwise(provider, size, &mut cfg.cpu_elem_per_elem).await {
            Ok(Some(true)) => return Some(Ok(size)),
            Ok(Some(false)) => continue,
            Ok(None) => return None,
            Err(e) => return Some(Err(e)),
        }
    }
    Some(Ok(usize::MAX))
}

async fn compare_elemwise(
    provider: &'static dyn AccelProvider,
    elements: usize,
    cpu_cost_slot: &mut f64,
) -> ProviderExecutionResult<Option<bool>> {
    if elements == 0 {
        return Ok(Some(false));
    }
    let shape = vec![elements, 1];
    let template = match provider.precision() {
        ProviderPrecision::F64 => {
            Tensor::new((0..elements).map(|i| i as f64).collect(), shape.clone())
                .map_err(|e| anyhow!(e))?
        }
        ProviderPrecision::F32 => {
            Tensor::from_f32((0..elements).map(|i| i as f32).collect(), shape.clone())
                .map_err(|e| anyhow!(e))?
        }
    };
    let cpu_data = template.materialize_f64();
    let cpu_time = time_cpu(|| {
        std::hint::black_box(
            cpu_data
                .iter()
                .zip(&cpu_data)
                .map(|(lhs, rhs)| lhs + rhs)
                .collect::<Vec<_>>(),
        )
    });
    let cpu_per_elem = cpu_time.as_secs_f64() / elements as f64;
    update_cpu_cost(cpu_cost_slot, cpu_per_elem);
    let template_data = template.materialize_f64();
    let view = HostTensorView {
        data: template_data.as_slice(),
        shape: template.shape(),
    };
    let ha = provider.upload(&view)?;
    let hb = provider.upload(&view)?;
    let start = Instant::now();
    let hc = match provider.elem_add(&ha, &hb).await {
        Ok(h) => h,
        Err(error) => return calibration_cleanup::operation_failed(error, [ha, hb]),
    };
    let gpu_time = start.elapsed();
    calibration_cleanup::release([ha, hb, hc])?;
    Ok(Some(gpu_time < cpu_time))
}

async fn calibrate_reduction(
    provider: &'static dyn AccelProvider,
    cfg: &mut ThresholdConfig,
) -> Option<ProviderExecutionResult<usize>> {
    let sizes = [256usize, 1_024, 4_096, 16_384, 65_536];
    for size in sizes {
        match compare_reduction(provider, size, &mut cfg.cpu_reduction_per_elem).await {
            Ok(Some(true)) => return Some(Ok(size)),
            Ok(Some(false)) => continue,
            Ok(None) => return None,
            Err(e) => return Some(Err(e)),
        }
    }
    Some(Ok(usize::MAX))
}

async fn compare_reduction(
    provider: &'static dyn AccelProvider,
    elements: usize,
    cpu_cost_slot: &mut f64,
) -> ProviderExecutionResult<Option<bool>> {
    let shape = vec![elements, 1];
    let template = match provider.precision() {
        ProviderPrecision::F64 => {
            Tensor::new((0..elements).map(|i| i as f64).collect(), shape.clone())
                .map_err(|e| anyhow!(e))?
        }
        ProviderPrecision::F32 => {
            Tensor::from_f32((0..elements).map(|i| i as f32).collect(), shape.clone())
                .map_err(|e| anyhow!(e))?
        }
    };
    let cpu_data = template.materialize_f64();
    let cpu_time = time_cpu(|| std::hint::black_box(cpu_data.iter().sum::<f64>()));
    let cpu_per_elem = cpu_time.as_secs_f64() / elements as f64;
    update_cpu_cost(cpu_cost_slot, cpu_per_elem);
    let template_data = template.materialize_f64();
    let view = HostTensorView {
        data: template_data.as_slice(),
        shape: template.shape(),
    };
    let h = provider.upload(&view)?;
    let start = Instant::now();
    let out = match provider.reduce_sum(&h).await {
        Ok(hc) => hc,
        Err(error) => return calibration_cleanup::operation_failed(error, [h]),
    };
    let gpu_time = start.elapsed();
    calibration_cleanup::release([h, out])?;
    Ok(Some(gpu_time < cpu_time))
}

async fn calibrate_matmul(
    provider: &'static dyn AccelProvider,
    cfg: &mut ThresholdConfig,
) -> Option<ProviderExecutionResult<usize>> {
    let dims = [32usize, 64, 96, 128, 192];
    for n in dims {
        match compare_matmul(provider, n, &mut cfg.cpu_matmul_per_flop).await {
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

async fn compare_matmul(
    provider: &'static dyn AccelProvider,
    n: usize,
    cpu_cost_slot: &mut f64,
) -> ProviderExecutionResult<Option<bool>> {
    if n == 0 {
        return Ok(Some(false));
    }
    let total = n * n;
    let shape = vec![n, n];
    let (ta, tb) = match provider.precision() {
        ProviderPrecision::F64 => {
            let data_a: Vec<f64> = (0..total).map(|i| (i % 13) as f64).collect();
            let data_b: Vec<f64> = (0..total).map(|i| (i % 7) as f64).collect();
            let ta = Tensor::new(data_a, shape.clone()).map_err(|e| anyhow!(e))?;
            let tb = Tensor::new(data_b, shape.clone()).map_err(|e| anyhow!(e))?;
            (ta, tb)
        }
        ProviderPrecision::F32 => {
            let data_a: Vec<f32> = (0..total).map(|i| (i % 13) as f32).collect();
            let data_b: Vec<f32> = (0..total).map(|i| (i % 7) as f32).collect();
            let ta = Tensor::from_f32(data_a, shape.clone()).map_err(|e| anyhow!(e))?;
            let tb = Tensor::from_f32(data_b, shape.clone()).map_err(|e| anyhow!(e))?;
            (ta, tb)
        }
    };
    let cpu_a = ta.materialize_f64();
    let cpu_b = tb.materialize_f64();
    let cpu_time = time_cpu(|| {
        let mut output = vec![0.0; total];
        for col in 0..n {
            for row in 0..n {
                let mut value = 0.0;
                for inner in 0..n {
                    value += cpu_a[row + inner * n] * cpu_b[inner + col * n];
                }
                output[row + col * n] = value;
            }
        }
        std::hint::black_box(output)
    });
    let flops = (n * n * n) as f64;
    update_cpu_cost(cpu_cost_slot, cpu_time.as_secs_f64() / flops);
    let data_a = ta.materialize_f64();
    let data_b = tb.materialize_f64();
    let view_a = HostTensorView {
        data: data_a.as_slice(),
        shape: ta.shape(),
    };
    let view_b = HostTensorView {
        data: data_b.as_slice(),
        shape: tb.shape(),
    };
    let ha = provider.upload(&view_a)?;
    let hb = provider.upload(&view_b)?;
    let start = Instant::now();
    let hc = match provider.matmul(&ha, &hb).await {
        Ok(h) => h,
        Err(error) => return calibration_cleanup::operation_failed(error, [ha, hb]),
    };
    let gpu_time = start.elapsed();
    calibration_cleanup::release([ha, hb, hc])?;
    Ok(Some(gpu_time < cpu_time))
}

fn time_cpu<F, T>(mut f: F) -> Duration
where
    F: FnMut() -> T,
{
    let start = Instant::now();
    std::hint::black_box(f());
    start.elapsed()
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

pub fn sequence_threshold_hint() -> Option<usize> {
    AUTO_STATE
        .get()
        .and_then(|state| state.lock().ok())
        .map(|state| state.thresholds.unary_min_elems)
}

pub fn reset_auto_offload_log() {
    clear_decisions();
}

pub async fn promote_binary(
    op: BinaryOp,
    a: &Value,
    b: &Value,
) -> ProviderExecutionResult<(Value, Value)> {
    if !auto_enabled() {
        return Ok((a.clone(), b.clone()));
    }
    if let Some(auto) = global().await {
        auto.promote_binary(op, a, b)
    } else {
        Ok((a.clone(), b.clone()))
    }
}

pub async fn promote_unary(op: UnaryOp, value: &Value) -> ProviderExecutionResult<Value> {
    if !auto_enabled() {
        return Ok(value.clone());
    }
    if let Some(auto) = global().await {
        auto.promote_unary(op, value)
    } else {
        Ok(value.clone())
    }
}

pub async fn prepare_builtin_args(
    name: &str,
    args: &[Value],
) -> ProviderExecutionResult<Vec<Value>> {
    Ok(builtin_call::prepare(name, args, None).await?.into_args())
}

pub fn is_sink(name: &str) -> bool {
    builtin_policy(name).map(|p| p.is_sink).unwrap_or(false)
}

pub async fn promote_reduction_args(
    op: ReductionOp,
    args: &[Value],
) -> ProviderExecutionResult<Vec<Value>> {
    if !auto_enabled() {
        return Ok(args.to_vec());
    }
    if let Some(auto) = global().await {
        auto.promote_reduction(op, args)
    } else {
        Ok(args.to_vec())
    }
}
