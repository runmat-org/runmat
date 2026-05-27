//! MATLAB-compatible `timeit` builtin for RunMat.
//!
//! Measures the execution time of zero-input function handles by running them
//! repeatedly and returning the median per-invocation runtime in seconds.

use runmat_time::Instant;
use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::timing::type_resolvers::timeit_type;

const TARGET_BATCH_SECONDS: f64 = 0.005;
const MAX_BATCH_SECONDS: f64 = 0.25;
const LOOP_COUNT_LIMIT: usize = 1 << 20;
const MIN_SAMPLE_COUNT: usize = 7;
const MAX_SAMPLE_COUNT: usize = 21;
const BUILTIN_NAME: &str = "timeit";

const TIMEIT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Median execution time per invocation in seconds.",
}];

const TIMEIT_INPUTS_ONE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "f",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zero-input function handle to benchmark.",
}];

const TIMEIT_INPUTS_TWO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero-input function handle to benchmark.",
    },
    BuiltinParamDescriptor {
        name: "numOutputs",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Requested output count for invoking the benchmarked handle.",
    },
];

const TIMEIT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "t = timeit(f)",
        inputs: &TIMEIT_INPUTS_ONE,
        outputs: &TIMEIT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "t = timeit(f, numOutputs)",
        inputs: &TIMEIT_INPUTS_TWO,
        outputs: &TIMEIT_OUTPUT,
    },
];

const TIMEIT_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMEIT.TOO_MANY_INPUTS",
    identifier: Some("RunMat:timeit:TooManyInputs"),
    when: "More than two input arguments are supplied.",
    message: "timeit: too many input arguments",
};

const TIMEIT_ERROR_NUM_OUTPUTS_SCALAR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMEIT.NUM_OUTPUTS_SCALAR",
    identifier: Some("RunMat:timeit:NumOutputsScalar"),
    when: "numOutputs is not a scalar numeric/integer value.",
    message: "timeit: numOutputs must be a scalar numeric value",
};

const TIMEIT_ERROR_NUM_OUTPUTS_FINITE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMEIT.NUM_OUTPUTS_FINITE",
    identifier: Some("RunMat:timeit:NumOutputsFinite"),
    when: "numOutputs is NaN or infinite.",
    message: "timeit: numOutputs must be finite",
};

const TIMEIT_ERROR_NUM_OUTPUTS_NONNEG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMEIT.NUM_OUTPUTS_NONNEGATIVE",
    identifier: Some("RunMat:timeit:NumOutputsNonnegative"),
    when: "numOutputs is negative.",
    message: "timeit: numOutputs must be a nonnegative integer",
};

const TIMEIT_ERROR_NUM_OUTPUTS_INTEGER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMEIT.NUM_OUTPUTS_INTEGER",
    identifier: Some("RunMat:timeit:NumOutputsInteger"),
    when: "numOutputs has a non-integer numeric value.",
    message: "timeit: numOutputs must be an integer value",
};

const TIMEIT_ERROR_EMPTY_HANDLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMEIT.EMPTY_HANDLE",
    identifier: Some("RunMat:timeit:EmptyFunctionHandle"),
    when: "A function-handle string or payload is empty after trimming.",
    message: "timeit: empty function handle string",
};

const TIMEIT_ERROR_EXPECTS_AT_HANDLE_STRING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMEIT.EXPECTS_AT_HANDLE_STRING",
    identifier: Some("RunMat:timeit:ExpectedAtHandleString"),
    when: "A string/char function handle does not begin with '@'.",
    message: "timeit: expected a function handle string beginning with '@'",
};

const TIMEIT_ERROR_HANDLE_KIND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMEIT.HANDLE_KIND",
    identifier: Some("RunMat:timeit:HandleKind"),
    when: "Function handle argument is not a scalar string/char or callable handle value.",
    message: "timeit: function handle must be a string scalar or function handle",
};

const TIMEIT_ERROR_FIRST_ARG_KIND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMEIT.FIRST_ARG_KIND",
    identifier: Some("RunMat:timeit:FirstArgKind"),
    when: "First argument is not a function handle value.",
    message: "timeit: first argument must be a function handle",
};

const TIMEIT_ERRORS: [BuiltinErrorDescriptor; 9] = [
    TIMEIT_ERROR_TOO_MANY_INPUTS,
    TIMEIT_ERROR_NUM_OUTPUTS_SCALAR,
    TIMEIT_ERROR_NUM_OUTPUTS_FINITE,
    TIMEIT_ERROR_NUM_OUTPUTS_NONNEG,
    TIMEIT_ERROR_NUM_OUTPUTS_INTEGER,
    TIMEIT_ERROR_EMPTY_HANDLE,
    TIMEIT_ERROR_EXPECTS_AT_HANDLE_STRING,
    TIMEIT_ERROR_HANDLE_KIND,
    TIMEIT_ERROR_FIRST_ARG_KIND,
];

pub const TIMEIT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TIMEIT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TIMEIT_ERRORS,
};

fn timeit_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut builder = crate::build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::timing::timeit")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "timeit",
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
    notes: "Host-side helper; GPU kernels execute only if invoked by the timed function.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::timing::timeit")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "timeit",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Timing helper; excluded from fusion planning.",
};

#[runtime_builtin(
    name = "timeit",
    category = "timing",
    summary = "Measure the execution time of a zero-argument function handle.",
    keywords = "timeit,benchmark,timing,performance,gpu",
    accel = "helper",
    type_resolver(timeit_type),
    descriptor(crate::builtins::timing::timeit::TIMEIT_DESCRIPTOR),
    builtin_path = "crate::builtins::timing::timeit"
)]
async fn timeit_builtin(func: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let requested_outputs = parse_num_outputs(&rest)?;
    let callable = prepare_callable(func, requested_outputs)?;

    // Warm-up once to catch early errors and pay one-time JIT costs.
    callable.invoke().await?;

    let loop_count = determine_loop_count(&callable).await?;
    let samples = collect_samples(&callable, loop_count).await?;
    if samples.is_empty() {
        return Ok(Value::Num(0.0));
    }

    Ok(Value::Num(compute_median(samples)))
}

fn parse_num_outputs(rest: &[Value]) -> Result<Option<usize>, crate::RuntimeError> {
    match rest.len() {
        0 => Ok(None),
        1 => parse_non_negative_integer(&rest[0]).map(Some),
        _ => Err(timeit_error_with_message(
            TIMEIT_ERROR_TOO_MANY_INPUTS.message,
            &TIMEIT_ERROR_TOO_MANY_INPUTS,
        )),
    }
}

fn parse_non_negative_integer(value: &Value) -> Result<usize, crate::RuntimeError> {
    match value {
        Value::Int(iv) => {
            let raw = iv.to_i64();
            if raw < 0 {
                Err(timeit_error_with_message(
                    TIMEIT_ERROR_NUM_OUTPUTS_NONNEG.message,
                    &TIMEIT_ERROR_NUM_OUTPUTS_NONNEG,
                ))
            } else {
                Ok(raw as usize)
            }
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(timeit_error_with_message(
                    TIMEIT_ERROR_NUM_OUTPUTS_FINITE.message,
                    &TIMEIT_ERROR_NUM_OUTPUTS_FINITE,
                ));
            }
            if *n < 0.0 {
                return Err(timeit_error_with_message(
                    TIMEIT_ERROR_NUM_OUTPUTS_NONNEG.message,
                    &TIMEIT_ERROR_NUM_OUTPUTS_NONNEG,
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(timeit_error_with_message(
                    TIMEIT_ERROR_NUM_OUTPUTS_INTEGER.message,
                    &TIMEIT_ERROR_NUM_OUTPUTS_INTEGER,
                ));
            }
            Ok(rounded as usize)
        }
        _ => Err(timeit_error_with_message(
            TIMEIT_ERROR_NUM_OUTPUTS_SCALAR.message,
            &TIMEIT_ERROR_NUM_OUTPUTS_SCALAR,
        )),
    }
}

async fn determine_loop_count(callable: &TimeitCallable) -> Result<usize, crate::RuntimeError> {
    let mut loops = 1usize;
    loop {
        let elapsed = run_batch(callable, loops).await?;
        if elapsed >= TARGET_BATCH_SECONDS
            || elapsed >= MAX_BATCH_SECONDS
            || loops >= LOOP_COUNT_LIMIT
        {
            return Ok(loops);
        }
        loops = loops.saturating_mul(2);
        if loops == 0 {
            return Ok(LOOP_COUNT_LIMIT);
        }
    }
}

async fn collect_samples(
    callable: &TimeitCallable,
    loop_count: usize,
) -> Result<Vec<f64>, crate::RuntimeError> {
    let mut samples = Vec::with_capacity(MIN_SAMPLE_COUNT);
    while samples.len() < MIN_SAMPLE_COUNT {
        let elapsed = run_batch(callable, loop_count).await?;
        let per_iter = elapsed / loop_count as f64;
        samples.push(per_iter);
        if samples.len() >= MAX_SAMPLE_COUNT || elapsed >= MAX_BATCH_SECONDS {
            break;
        }
    }
    Ok(samples)
}

async fn run_batch(
    callable: &TimeitCallable,
    loop_count: usize,
) -> Result<f64, crate::RuntimeError> {
    let start = Instant::now();
    for _ in 0..loop_count {
        let value = callable.invoke().await?;
        drop(value);
    }
    Ok(start.elapsed().as_secs_f64())
}

fn compute_median(mut samples: Vec<f64>) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap_or_else(|| {
            if a < b {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }),
    });
    let mid = samples.len() / 2;
    if samples.len() % 2 == 1 {
        samples[mid]
    } else {
        (samples[mid - 1] + samples[mid]) * 0.5
    }
}

#[derive(Clone, Debug)]
struct TimeitCallable {
    handle: Value,
    num_outputs: Option<usize>,
}

impl TimeitCallable {
    async fn invoke(&self) -> Result<Value, crate::RuntimeError> {
        let requested_outputs = self.num_outputs.unwrap_or(1);
        let value =
            crate::call_feval_async_with_outputs(self.handle.clone(), &[], requested_outputs)
                .await?;
        drop(value);
        Ok(Value::Num(0.0))
    }
}

fn prepare_callable(
    func: Value,
    num_outputs: Option<usize>,
) -> Result<TimeitCallable, crate::RuntimeError> {
    fn normalize_name(name: &str) -> Result<String, crate::RuntimeError> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            Err(timeit_error_with_message(
                TIMEIT_ERROR_EMPTY_HANDLE.message,
                &TIMEIT_ERROR_EMPTY_HANDLE,
            ))
        } else {
            Ok(trimmed.to_string())
        }
    }

    fn canonicalize_text_handle(handle: String) -> Value {
        let name = handle.strip_prefix('@').unwrap_or(handle.as_str());
        handle_for_name(name).unwrap_or(Value::String(handle))
    }

    match func {
        Value::String(text) => parse_handle_string(&text).map(|handle| TimeitCallable {
            handle: canonicalize_text_handle(handle),
            num_outputs,
        }),
        Value::CharArray(arr) => {
            if arr.rows != 1 {
                Err(timeit_error_with_message(
                    TIMEIT_ERROR_HANDLE_KIND.message,
                    &TIMEIT_ERROR_HANDLE_KIND,
                ))
            } else {
                let text: String = arr.data.iter().collect();
                parse_handle_string(&text).map(|handle| TimeitCallable {
                    handle: canonicalize_text_handle(handle),
                    num_outputs,
                })
            }
        }
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                parse_handle_string(&sa.data[0]).map(|handle| TimeitCallable {
                    handle: canonicalize_text_handle(handle),
                    num_outputs,
                })
            } else {
                Err(timeit_error_with_message(
                    TIMEIT_ERROR_HANDLE_KIND.message,
                    &TIMEIT_ERROR_HANDLE_KIND,
                ))
            }
        }
        Value::FunctionHandle(name) => {
            let normalized = normalize_name(&name)?;
            Ok(TimeitCallable {
                handle: handle_for_name(&normalized)
                    .unwrap_or_else(|| Value::String(format!("@{normalized}"))),
                num_outputs,
            })
        }
        Value::ExternalFunctionHandle(name) => {
            let normalized = normalize_name(&name)?;
            Ok(TimeitCallable {
                handle: if crate::is_well_formed_qualified_name(&normalized) {
                    handle_for_name(&normalized)
                        .unwrap_or_else(|| Value::ExternalFunctionHandle(normalized))
                } else {
                    Value::ExternalFunctionHandle(normalized)
                },
                num_outputs,
            })
        }
        Value::BoundFunctionHandle { name, function } => {
            let normalized = normalize_name(&name)?;
            Ok(TimeitCallable {
                handle: Value::BoundFunctionHandle {
                    name: normalized,
                    function,
                },
                num_outputs,
            })
        }
        Value::Closure(mut closure) => Ok(TimeitCallable {
            handle: {
                if closure.bound_function.is_none() {
                    if let Some(function) = crate::user_functions::resolve_semantic_function_by_name(
                        &closure.function_name,
                    ) {
                        closure.bound_function = Some(function);
                    }
                }
                Value::Closure(closure)
            },
            num_outputs,
        }),
        other => Err(timeit_error_with_message(
            format!("timeit: first argument must be a function handle, got {other:?}"),
            &TIMEIT_ERROR_FIRST_ARG_KIND,
        )),
    }
}

fn handle_for_name(name: &str) -> Option<Value> {
    let function = crate::user_functions::resolve_semantic_function_by_name(name)?;
    Some(Value::BoundFunctionHandle {
        name: name.to_string(),
        function,
    })
}

fn parse_handle_string(text: &str) -> Result<String, crate::RuntimeError> {
    let trimmed = text.trim();
    if let Some(rest) = trimmed.strip_prefix('@') {
        if rest.trim().is_empty() {
            Err(timeit_error_with_message(
                TIMEIT_ERROR_EMPTY_HANDLE.message,
                &TIMEIT_ERROR_EMPTY_HANDLE,
            ))
        } else {
            Ok(format!("@{}", rest.trim()))
        }
    } else {
        Err(timeit_error_with_message(
            TIMEIT_ERROR_EXPECTS_AT_HANDLE_STRING.message,
            &TIMEIT_ERROR_EXPECTS_AT_HANDLE_STRING,
        ))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{Closure, IntValue};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    const TIMEIT_HELPER_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Helper scalar return value.",
    }];

    const TIMEIT_HELPER_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
        [BuiltinSignatureDescriptor {
            label: "y = __timeit_helper()",
            inputs: &[],
            outputs: &TIMEIT_HELPER_OUTPUT,
        }];

    const TIMEIT_HELPER_ERRORS: [BuiltinErrorDescriptor; 0] = [];

    pub const TIMEIT_TEST_HELPER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
        signatures: &TIMEIT_HELPER_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::HiddenInternal,
        errors: &TIMEIT_HELPER_ERRORS,
    };

    static COUNTER_DEFAULT: AtomicUsize = AtomicUsize::new(0);
    static COUNTER_NUM_OUTPUTS: AtomicUsize = AtomicUsize::new(0);
    static COUNTER_INVALID: AtomicUsize = AtomicUsize::new(0);
    static COUNTER_ZERO_OUTPUTS: AtomicUsize = AtomicUsize::new(0);

    #[runtime_builtin(
        name = "__timeit_helper_counter_default",
        type_resolver(crate::builtins::timing::type_resolvers::timeit_type),
        descriptor(crate::builtins::timing::timeit::tests::TIMEIT_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::timing::timeit::tests"
    )]
    async fn helper_counter_default() -> crate::BuiltinResult<Value> {
        COUNTER_DEFAULT.fetch_add(1, Ordering::SeqCst);
        Ok(Value::Num(1.0))
    }

    #[runtime_builtin(
        name = "__timeit_helper_counter_outputs",
        type_resolver(crate::builtins::timing::type_resolvers::timeit_type),
        descriptor(crate::builtins::timing::timeit::tests::TIMEIT_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::timing::timeit::tests"
    )]
    async fn helper_counter_outputs() -> crate::BuiltinResult<Value> {
        COUNTER_NUM_OUTPUTS.fetch_add(1, Ordering::SeqCst);
        Ok(Value::Num(1.0))
    }

    #[runtime_builtin(
        name = "__timeit_helper_counter_invalid",
        type_resolver(crate::builtins::timing::type_resolvers::timeit_type),
        descriptor(crate::builtins::timing::timeit::tests::TIMEIT_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::timing::timeit::tests"
    )]
    async fn helper_counter_invalid() -> crate::BuiltinResult<Value> {
        COUNTER_INVALID.fetch_add(1, Ordering::SeqCst);
        Ok(Value::Num(1.0))
    }

    #[runtime_builtin(
        name = "__timeit_helper_zero_outputs",
        type_resolver(crate::builtins::timing::type_resolvers::timeit_type),
        descriptor(crate::builtins::timing::timeit::tests::TIMEIT_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::timing::timeit::tests"
    )]
    async fn helper_counter_zero_outputs() -> crate::BuiltinResult<Value> {
        COUNTER_ZERO_OUTPUTS.fetch_add(1, Ordering::SeqCst);
        Ok(Value::Num(0.0))
    }

    fn default_handle() -> Value {
        Value::String("@__timeit_helper_counter_default".to_string())
    }

    fn assert_timeit_error_contains(err: &crate::RuntimeError, needle: &str) {
        let message = err.message().to_ascii_lowercase();
        assert!(
            message.contains(&needle.to_ascii_lowercase()),
            "unexpected error text: {}",
            err.message()
        );
    }

    fn assert_timeit_error_identifier(err: &crate::RuntimeError, identifier: &'static str) {
        assert_eq!(err.identifier(), Some(identifier), "{}", err.message());
    }

    fn outputs_handle() -> Value {
        Value::String("@__timeit_helper_counter_outputs".to_string())
    }

    fn invalid_handle() -> Value {
        Value::String("@__timeit_helper_counter_invalid".to_string())
    }

    fn zero_outputs_handle() -> Value {
        Value::String("@__timeit_helper_zero_outputs".to_string())
    }

    #[test]
    fn timeit_test_helper_descriptor_is_attached_shape() {
        assert_eq!(
            TIMEIT_TEST_HELPER_DESCRIPTOR.signatures[0].label,
            "y = __timeit_helper()"
        );
    }

    #[test]
    fn timeit_accepts_external_function_handle() {
        let callable = prepare_callable(
            Value::ExternalFunctionHandle("pkg.callback".to_string()),
            Some(2),
        )
        .expect("timeit should accept external function handle");
        assert_eq!(
            callable.handle,
            Value::ExternalFunctionHandle("pkg.callback".to_string())
        );
        assert_eq!(callable.num_outputs, Some(2));
    }

    #[test]
    fn timeit_rejects_empty_function_handle_name_value() {
        let err = prepare_callable(Value::FunctionHandle("   ".to_string()), None)
            .expect_err("timeit should reject empty function-handle payload name");
        assert_timeit_error_contains(&err, "empty function handle");
        assert_timeit_error_identifier(&err, TIMEIT_ERROR_EMPTY_HANDLE.identifier.unwrap());
    }

    #[test]
    fn timeit_rejects_empty_external_function_handle_name_value() {
        let err = prepare_callable(Value::ExternalFunctionHandle("   ".to_string()), None)
            .expect_err("timeit should reject empty external function-handle payload name");
        assert_timeit_error_contains(&err, "empty function handle");
        assert_timeit_error_identifier(&err, TIMEIT_ERROR_EMPTY_HANDLE.identifier.unwrap());
    }

    #[test]
    fn timeit_trims_function_handle_name_for_semantic_resolution() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "__timeit_helper_counter_default").then_some(188)
            })));
        let callable = prepare_callable(
            Value::FunctionHandle("  __timeit_helper_counter_default  ".to_string()),
            None,
        )
        .expect("timeit should normalize function-handle payload name");
        assert_eq!(
            callable.handle,
            Value::BoundFunctionHandle {
                name: "__timeit_helper_counter_default".to_string(),
                function: 188,
            }
        );
    }

    #[test]
    fn timeit_callable_invoke_honors_multi_requested_outputs() {
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 612);
                assert!(args.is_empty());
                assert_eq!(requested_outputs, 3);
                Box::pin(async {
                    Ok(Value::OutputList(vec![
                        Value::Num(1.0),
                        Value::Num(2.0),
                        Value::Num(3.0),
                    ]))
                })
            }),
        ));

        let callable = prepare_callable(
            Value::BoundFunctionHandle {
                name: "function_target".to_string(),
                function: 612,
            },
            Some(3),
        )
        .expect("timeit should accept semantic callback handles");

        let invoked = block_on(callable.invoke()).expect("timeit callable invoke should succeed");
        assert_eq!(invoked, Value::Num(0.0));
    }

    #[test]
    fn timeit_string_handle_prefers_semantic_resolver_identity() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "__timeit_helper_counter_default").then_some(87)
            })));
        let callable = prepare_callable(
            Value::String("@__timeit_helper_counter_default".to_string()),
            None,
        )
        .expect("timeit should accept string function handle");
        assert_eq!(
            callable.handle,
            Value::BoundFunctionHandle {
                name: "__timeit_helper_counter_default".to_string(),
                function: 87,
            }
        );
    }

    #[test]
    fn timeit_char_handle_prefers_semantic_resolver_identity() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "__timeit_helper_counter_default").then_some(88)
            })));
        let callable = prepare_callable(
            Value::CharArray(runmat_builtins::CharArray::new_row(
                "@__timeit_helper_counter_default",
            )),
            None,
        )
        .expect("timeit should accept char function handle");
        assert_eq!(
            callable.handle,
            Value::BoundFunctionHandle {
                name: "__timeit_helper_counter_default".to_string(),
                function: 88,
            }
        );
    }

    #[test]
    fn timeit_external_function_handle_prefers_semantic_resolver_identity() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.callback").then_some(86)
            })));
        let callable = prepare_callable(
            Value::ExternalFunctionHandle("pkg.callback".to_string()),
            Some(2),
        )
        .expect("timeit should accept external function handle");
        assert_eq!(
            callable.handle,
            Value::BoundFunctionHandle {
                name: "pkg.callback".to_string(),
                function: 86,
            }
        );
        assert_eq!(callable.num_outputs, Some(2));
    }

    #[test]
    fn timeit_accepts_semantic_function_handle() {
        let callable = prepare_callable(
            Value::BoundFunctionHandle {
                name: "function_target".to_string(),
                function: 41,
            },
            Some(1),
        )
        .expect("timeit should accept semantic function handle");
        assert_eq!(
            callable.handle,
            Value::BoundFunctionHandle {
                name: "function_target".to_string(),
                function: 41,
            }
        );
        assert_eq!(callable.num_outputs, Some(1));
    }

    #[test]
    fn timeit_name_only_closure_prefers_semantic_resolver_identity() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "__timeit_helper_counter_default").then_some(89)
            })));
        let callable = prepare_callable(
            Value::Closure(Closure {
                function_name: "__timeit_helper_counter_default".to_string(),
                bound_function: None,
                captures: vec![Value::Num(9.0)],
            }),
            None,
        )
        .expect("timeit should accept closure callback");
        assert_eq!(
            callable.handle,
            Value::Closure(Closure {
                function_name: "__timeit_helper_counter_default".to_string(),
                bound_function: Some(89),
                captures: vec![Value::Num(9.0)],
            })
        );
    }

    #[test]
    fn timeit_name_only_closure_without_resolver_keeps_name_shaped_identity() {
        let callable = prepare_callable(
            Value::Closure(Closure {
                function_name: "__timeit_helper_counter_default".to_string(),
                bound_function: None,
                captures: vec![Value::Num(9.0)],
            }),
            None,
        )
        .expect("timeit should accept closure callback");
        assert_eq!(
            callable.handle,
            Value::Closure(Closure {
                function_name: "__timeit_helper_counter_default".to_string(),
                bound_function: None,
                captures: vec![Value::Num(9.0)],
            })
        );
    }

    #[test]
    fn timeit_external_function_handle_surfaces_undefined_function() {
        let err = block_on(timeit_builtin(
            Value::ExternalFunctionHandle("pkg.missing_callback".to_string()),
            Vec::new(),
        ))
        .expect_err("unresolved external callback should fail");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn timeit_measures_time() {
        COUNTER_DEFAULT.store(0, Ordering::SeqCst);
        let result = block_on(timeit_builtin(default_handle(), Vec::new())).expect("timeit");
        match result {
            Value::Num(v) => assert!(v >= 0.0),
            other => panic!("expected numeric result, got {other:?}"),
        }
        assert!(
            COUNTER_DEFAULT.load(Ordering::SeqCst) >= MIN_SAMPLE_COUNT,
            "expected at least {} invocations",
            MIN_SAMPLE_COUNT
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn timeit_accepts_num_outputs_argument() {
        COUNTER_NUM_OUTPUTS.store(0, Ordering::SeqCst);
        let args = vec![Value::Int(IntValue::I32(3))];
        let _ = block_on(timeit_builtin(outputs_handle(), args)).expect("timeit numOutputs");
        assert!(
            COUNTER_NUM_OUTPUTS.load(Ordering::SeqCst) >= MIN_SAMPLE_COUNT,
            "expected at least {} invocations",
            MIN_SAMPLE_COUNT
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn timeit_supports_zero_outputs() {
        COUNTER_ZERO_OUTPUTS.store(0, Ordering::SeqCst);
        let args = vec![Value::Int(IntValue::I32(0))];
        let _ = block_on(timeit_builtin(zero_outputs_handle(), args)).expect("timeit zero outputs");
        assert!(
            COUNTER_ZERO_OUTPUTS.load(Ordering::SeqCst) >= MIN_SAMPLE_COUNT,
            "expected at least {} invocations",
            MIN_SAMPLE_COUNT
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn timeit_runs_with_wgpu_provider_registered() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let result =
            block_on(timeit_builtin(default_handle(), Vec::new())).expect("timeit with wgpu");
        match result {
            Value::Num(v) => assert!(v >= 0.0),
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn timeit_rejects_non_function_input() {
        let err = block_on(timeit_builtin(Value::Num(1.0), Vec::new())).unwrap_err();
        assert_timeit_error_contains(&err, "function");
        assert_timeit_error_identifier(&err, TIMEIT_ERROR_FIRST_ARG_KIND.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn timeit_rejects_invalid_num_outputs() {
        COUNTER_INVALID.store(0, Ordering::SeqCst);
        let err = block_on(timeit_builtin(invalid_handle(), vec![Value::Num(-1.0)])).unwrap_err();
        assert_timeit_error_contains(&err, "nonnegative");
        assert_timeit_error_identifier(&err, TIMEIT_ERROR_NUM_OUTPUTS_NONNEG.identifier.unwrap());
        assert_eq!(COUNTER_INVALID.load(Ordering::SeqCst), 0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn timeit_rejects_extra_arguments() {
        let err = block_on(timeit_builtin(
            default_handle(),
            vec![Value::from(1.0), Value::from(2.0)],
        ))
        .unwrap_err();
        assert_timeit_error_contains(&err, "too many");
        assert_timeit_error_identifier(&err, TIMEIT_ERROR_TOO_MANY_INPUTS.identifier.unwrap());
    }
}
