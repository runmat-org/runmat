//! MATLAB-compatible `tic` builtin with precise stopwatch semantics for RunMat.

use once_cell::sync::Lazy;
use runmat_macros::runtime_builtin;
use runmat_time::Instant;
use std::sync::Mutex;
use std::time::Duration;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::timing::tic")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tic",
    op_kind: GpuOpKind::Custom("timer"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Stopwatch state lives on the host. Providers are never consulted for tic/toc.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::timing::tic")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tic",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Timing builtins are executed eagerly on the host and do not participate in fusion.",
};

static MONOTONIC_ORIGIN: Lazy<Instant> = Lazy::new(Instant::now);
static STOPWATCH: Lazy<Mutex<StopwatchState>> = Lazy::new(|| Mutex::new(StopwatchState::default()));

#[cfg(test)]
pub(crate) static TEST_GUARD: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[derive(Default)]
struct StopwatchState {
    stack: Vec<Instant>,
}

impl StopwatchState {
    fn push(&mut self, instant: Instant) {
        self.stack.push(instant);
    }

    fn pop(&mut self) -> Option<Instant> {
        self.stack.pop()
    }
}

const BUILTIN_NAME: &str = "tic";
const LOCK_ERR: &str = "tic: failed to acquire stopwatch state";

fn stopwatch_error(builtin: &str, message: impl Into<String>) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_builtin(builtin)
        .build()
}

/// Start a stopwatch timer and return a handle suitable for `toc`.
#[runtime_builtin(
    name = "tic",
    category = "timing",
    summary = "Start a stopwatch timer and optionally return a handle for toc.",
    keywords = "tic,timing,profiling,benchmark",
    sink = true,
    builtin_path = "crate::builtins::timing::tic"
)]
pub async fn tic_builtin() -> crate::BuiltinResult<f64> {
    record_tic(BUILTIN_NAME)
}

/// Record a `tic` start time and return the encoded handle.
pub(crate) fn record_tic(builtin: &str) -> Result<f64, crate::RuntimeError> {
    let now = Instant::now();
    {
        let mut guard = STOPWATCH
            .lock()
            .map_err(|_| stopwatch_error(builtin, LOCK_ERR))?;
        guard.push(now);
    }
    Ok(encode_instant(now))
}

/// Remove and return the most recently recorded `tic`, if any.
pub(crate) fn take_latest_start(builtin: &str) -> Result<Option<Instant>, crate::RuntimeError> {
    let mut guard = STOPWATCH
        .lock()
        .map_err(|_| stopwatch_error(builtin, LOCK_ERR))?;
    Ok(guard.pop())
}

/// Encode an `Instant` into the scalar handle returned by `tic`.
pub(crate) fn encode_instant(instant: Instant) -> f64 {
    instant.duration_since(*MONOTONIC_ORIGIN).as_secs_f64()
}

/// Decode a scalar handle into an `Instant`.
pub(crate) fn decode_handle(handle: f64, builtin: &str) -> Result<Instant, crate::RuntimeError> {
    if !handle.is_finite() || handle.is_sign_negative() {
        return Err(crate::build_runtime_error("toc: invalid timer handle")
            .with_builtin(builtin)
            .with_identifier("MATLAB:toc:InvalidTimerHandle")
            .build());
    }
    let duration = Duration::from_secs_f64(handle);
    Ok((*MONOTONIC_ORIGIN) + duration)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use std::thread;
    use std::time::Duration;

    fn reset_stopwatch() {
        let mut guard = STOPWATCH.lock().unwrap();
        guard.stack.clear();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tic_returns_monotonic_handle() {
        let _guard = TEST_GUARD.lock().unwrap();
        reset_stopwatch();
        let handle = block_on(tic_builtin()).expect("tic");
        assert!(handle >= 0.0);
        assert!(take_latest_start(BUILTIN_NAME).expect("take").is_some());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tic_handles_increase_over_time() {
        let _guard = TEST_GUARD.lock().unwrap();
        reset_stopwatch();
        let first = block_on(tic_builtin()).expect("tic");
        thread::sleep(Duration::from_millis(5));
        let second = block_on(tic_builtin()).expect("tic");
        assert!(second > first);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn decode_roundtrip_matches_handle() {
        let _guard = TEST_GUARD.lock().unwrap();
        reset_stopwatch();
        let handle = block_on(tic_builtin()).expect("tic");
        let decoded = decode_handle(handle, "toc").expect("decode");
        let round_trip = encode_instant(decoded);
        let delta = (round_trip - handle).abs();
        assert!(delta < 1e-9, "delta {delta}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn take_latest_start_pops_stack() {
        let _guard = TEST_GUARD.lock().unwrap();
        reset_stopwatch();
        block_on(tic_builtin()).expect("tic");
        assert!(take_latest_start(BUILTIN_NAME).expect("take").is_some());
        assert!(take_latest_start(BUILTIN_NAME)
            .expect("second take")
            .is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn decode_handle_rejects_invalid_values() {
        let _guard = TEST_GUARD.lock().unwrap();
        assert!(decode_handle(f64::NAN, "toc").is_err());
        assert!(decode_handle(-1.0, "toc").is_err());
    }
}
