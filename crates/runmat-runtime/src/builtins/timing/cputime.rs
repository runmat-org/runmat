//! MATLAB-compatible legacy `cputime` builtin.

#[cfg(target_arch = "wasm32")]
use once_cell::sync::Lazy;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;
#[cfg(target_arch = "wasm32")]
use runmat_time::Instant;
#[cfg(windows)]
use std::ffi::c_void;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::timing::type_resolvers::cputime_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "cputime";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::timing::cputime")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cputime",
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
    notes: "Host-side process timing helper. GPU providers are never consulted.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::timing::cputime")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cputime",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Timing builtins execute eagerly on the host and do not participate in fusion.",
};

const CPUTIME_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Total CPU time used by the current RunMat process in seconds.",
}];

const CPUTIME_INPUTS: [BuiltinParamDescriptor; 0] = [];

const CPUTIME_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "t = cputime()",
    inputs: &CPUTIME_INPUTS,
    outputs: &CPUTIME_OUTPUT,
}];

const CPUTIME_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CPUTIME.TOO_MANY_INPUTS",
    identifier: Some("RunMat:cputime:TooManyInputs"),
    when: "Any input arguments are supplied.",
    message: "cputime: too many input arguments",
};

const CPUTIME_ERROR_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CPUTIME.UNAVAILABLE",
    identifier: Some("RunMat:cputime:Unavailable"),
    when: "The host platform reports that process CPU time is unavailable.",
    message: "cputime: process CPU time is unavailable",
};

const CPUTIME_ERRORS: [BuiltinErrorDescriptor; 2] =
    [CPUTIME_ERROR_TOO_MANY_INPUTS, CPUTIME_ERROR_UNAVAILABLE];

pub const CPUTIME_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CPUTIME_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CPUTIME_ERRORS,
};

#[cfg(target_arch = "wasm32")]
static FALLBACK_ORIGIN: Lazy<Instant> = Lazy::new(Instant::now);

#[cfg(windows)]
#[repr(C)]
#[derive(Clone, Copy)]
struct FileTime {
    low_date_time: u32,
    high_date_time: u32,
}

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn GetCurrentProcess() -> *mut c_void;
    fn GetProcessTimes(
        process: *mut c_void,
        creation_time: *mut FileTime,
        exit_time: *mut FileTime,
        kernel_time: *mut FileTime,
        user_time: *mut FileTime,
    ) -> i32;
}

#[runtime_builtin(
    name = "cputime",
    category = "timing",
    summary = "Return total CPU time used by the current process.",
    keywords = "cputime,cpu time,timing,profiling,legacy",
    accel = "metadata",
    type_resolver(cputime_type),
    descriptor(crate::builtins::timing::cputime::CPUTIME_DESCRIPTOR),
    builtin_path = "crate::builtins::timing::cputime"
)]
fn cputime_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(cputime_error(
            CPUTIME_ERROR_TOO_MANY_INPUTS.message,
            &CPUTIME_ERROR_TOO_MANY_INPUTS,
        ));
    }
    Ok(Value::Num(process_cpu_seconds()?))
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
fn process_cpu_seconds() -> BuiltinResult<f64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: `usage` points to valid writable memory for libc to initialize.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return Err(cputime_error(
            CPUTIME_ERROR_UNAVAILABLE.message,
            &CPUTIME_ERROR_UNAVAILABLE,
        ));
    }
    // SAFETY: getrusage returned success, so the structure has been initialized.
    let usage = unsafe { usage.assume_init() };
    Ok(timeval_seconds(usage.ru_utime) + timeval_seconds(usage.ru_stime))
}

#[cfg(windows)]
fn process_cpu_seconds() -> BuiltinResult<f64> {
    let mut creation_time = std::mem::MaybeUninit::<FileTime>::uninit();
    let mut exit_time = std::mem::MaybeUninit::<FileTime>::uninit();
    let mut kernel_time = std::mem::MaybeUninit::<FileTime>::uninit();
    let mut user_time = std::mem::MaybeUninit::<FileTime>::uninit();
    // SAFETY: GetCurrentProcess returns a valid pseudo-handle for the current
    // process, and all FILETIME pointers reference writable uninitialized
    // storage for GetProcessTimes to fill.
    let ok = unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            creation_time.as_mut_ptr(),
            exit_time.as_mut_ptr(),
            kernel_time.as_mut_ptr(),
            user_time.as_mut_ptr(),
        )
    };
    if ok == 0 {
        return Err(cputime_error(
            CPUTIME_ERROR_UNAVAILABLE.message,
            &CPUTIME_ERROR_UNAVAILABLE,
        ));
    }
    // SAFETY: GetProcessTimes returned success, so kernel/user times are initialized.
    let kernel_time = unsafe { kernel_time.assume_init() };
    let user_time = unsafe { user_time.assume_init() };
    Ok((filetime_ticks(kernel_time) + filetime_ticks(user_time)) as f64 / 10_000_000.0)
}

#[cfg(target_arch = "wasm32")]
fn process_cpu_seconds() -> BuiltinResult<f64> {
    // Browsers do not expose process CPU accounting. Preserve a monotonic
    // scalar clock so compatibility code can still take differences, and
    // document the wall-clock fallback.
    Ok(Instant::now()
        .checked_duration_since(*FALLBACK_ORIGIN)
        .unwrap_or_default()
        .as_secs_f64())
}

#[cfg(windows)]
fn filetime_ticks(value: FileTime) -> u64 {
    ((value.high_date_time as u64) << 32) | value.low_date_time as u64
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
fn timeval_seconds(value: libc::timeval) -> f64 {
    value.tv_sec as f64 + value.tv_usec as f64 / 1_000_000.0
}

fn cputime_error(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cputime_returns_finite_nonnegative_scalar() {
        let value = cputime_builtin(Vec::new()).expect("cputime");
        let Value::Num(time) = value else {
            panic!("expected numeric scalar")
        };
        assert!(time.is_finite());
        assert!(time >= 0.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cputime_is_monotonic_for_current_process() {
        let Value::Num(first) = cputime_builtin(Vec::new()).expect("first") else {
            panic!("expected numeric scalar")
        };
        for value in 0_u64..10_000 {
            std::hint::black_box(value.wrapping_mul(value));
        }
        let Value::Num(second) = cputime_builtin(Vec::new()).expect("second") else {
            panic!("expected numeric scalar")
        };
        assert!(second >= first);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cputime_rejects_inputs() {
        let err = cputime_builtin(vec![Value::Num(1.0)]).unwrap_err();
        assert_eq!(err.identifier(), CPUTIME_ERROR_TOO_MANY_INPUTS.identifier);
    }
}
