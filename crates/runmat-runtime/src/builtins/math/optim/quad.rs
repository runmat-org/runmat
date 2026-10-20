//! MATLAB-compatible legacy `quad` builtin for finite scalar quadrature.

#[cfg(test)]
use runmat_builtins::QUAD_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, QUAD_ERROR_INVALID_ARGUMENT,
    QUAD_ERROR_INVALID_INPUT, QUAD_ERROR_TOO_MANY_OUTPUTS,
    QUAD_INTEGER_BOUND_EXTENSION as INTEGER_BOUND_EXTENSION,
    QUAD_INTEGER_CALLBACK_EXTENSION as INTEGER_CALLBACK_EXTENSION,
    QUAD_INTEGER_TOLERANCE_EXTENSION as INTEGER_TOLERANCE_EXTENSION,
    QUAD_LOGICAL_NUMERIC_EXTENSION as LOGICAL_NUMERIC_EXTENSION,
    QUAD_RESIDENT_INPUT_EXTENSION as RESIDENT_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{LogicalArray, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::optim::common::{call_function, value_to_scalar};
use crate::builtins::math::optim::quadrature::{
    integrate, midpoint, AdaptiveSimpsonResult, QuadratureArithmeticError, QuadratureCooperation,
    QuadratureEvaluator, QuadratureIdentity, ScaledEstimate,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "quad";
const DEFAULT_TOL: f64 = 1.0e-6;
const MAX_DEPTH: usize = 30;
const MAX_FUN_EVALS: usize = 100_000;

fn quad_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("quad:") {
        detail.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn quad_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        let mut mapped = quad_error_with_detail(fallback, err.message());
        mapped.span = err.span;
        mapped.context = err.context.clone();
        mapped.context.builtin = Some(NAME.to_string());
        mapped.gpu_gather_retry = err.gpu_gather_retry;
        mapped.source = Some(Box::new(err));
        mapped
    }
}

fn quad_arithmetic_error(error: QuadratureArithmeticError) -> RuntimeError {
    match error {
        QuadratureArithmeticError::Resource(source) => {
            build_runtime_error("quad: could not reserve quadrature workspace")
                .with_identifier("RunMat:Internal")
                .with_builtin(NAME)
                .with_source(source)
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
        }
        QuadratureArithmeticError::InconsistentComponents => quad_error_with_detail(
            &QUAD_ERROR_INVALID_INPUT,
            "callback changed its output size",
        ),
        QuadratureArithmeticError::Execution(source) => *source,
    }
}

fn validate_requested_outputs() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 2) {
        return Err(quad_error_with_detail(
            &QUAD_ERROR_TOO_MANY_OUTPUTS,
            "quad: too many output arguments; maximum is 2",
        ));
    }
    Ok(())
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::quad")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "quad",
    op_kind: GpuOpKind::Custom("legacy-adaptive-simpson"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host adaptive Simpson solver. Callback computations may use GPU-aware builtins, but the adaptive integration loop runs on the CPU.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::quad")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "quad",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Legacy adaptive quadrature repeatedly invokes user code and terminates fusion planning.",
};

#[runtime_builtin(
    name = "quad",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::quad"
)]
async fn quad_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    function: Value,
    a: Value,
    b: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    quad_in_runtime(runtime.as_ref(), function, a, b, rest).await
}

#[cfg(test)]
pub(super) async fn quad_builtin(
    function: Value,
    a: Value,
    b: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    quad_in_runtime(None, function, a, b, rest).await
}

async fn quad_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    function: Value,
    a: Value,
    b: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let mut cooperation = QuadratureCooperation::new(QuadratureIdentity::Quad, runtime);
    cooperation.completed(0).await?;
    validate_requested_outputs()?;
    ensure_quad_extensions(&a, &b, rest.view())?;
    preflight_quad_inputs(&a, &b, rest.view())?;
    crate::call::admission::admit_feval_target(&function)?;
    authenticate_quad_residents(&a, &b, rest.view())?;
    let options = QuadOptions::parse(rest, &mut cooperation)
        .await
        .map_err(|err| quad_map_error(err, &QUAD_ERROR_INVALID_ARGUMENT))?;
    let a = scalar_real("lower bound", a, &mut cooperation)
        .await
        .map_err(|err| quad_map_error(err, &QUAD_ERROR_INVALID_INPUT))?;
    let b = scalar_real("upper bound", b, &mut cooperation)
        .await
        .map_err(|err| quad_map_error(err, &QUAD_ERROR_INVALID_INPUT))?;

    let result = if a == b {
        QuadResult {
            q: 0.0,
            func_count: 0,
        }
    } else {
        let sign = if b < a { -1.0 } else { 1.0 };
        let lo = a.min(b);
        let hi = a.max(b);
        let mut result = {
            let mut evaluator = QuadEvaluator::new(&function, options, &mut cooperation)?;
            let AdaptiveSimpsonResult { q, evaluations } = integrate(&mut evaluator, lo, hi)
                .await
                .map_err(|err| quad_map_error(err, &QUAD_ERROR_INVALID_INPUT))?;
            QuadResult {
                q,
                func_count: evaluations,
            }
        };
        result.q *= sign;
        result
    };

    cooperation.completed(0).await?;
    finalize(result)
}

fn preflight_quad_inputs(
    a: &Value,
    b: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    preflight_quad_scalar(QuadScalarRole::LowerBound, a)?;
    preflight_quad_scalar(QuadScalarRole::UpperBound, b)?;
    if let Some(tol) = rest.first() {
        preflight_quad_scalar(QuadScalarRole::Tolerance, tol)?;
    }
    if let Some(trace) = rest.get(1) {
        preflight_quad_scalar(QuadScalarRole::Trace, trace)?;
    }
    Ok(())
}

fn authenticate_quad_residents(
    a: &Value,
    b: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    for (role, value) in [
        (QuadScalarRole::LowerBound, Some(a)),
        (QuadScalarRole::UpperBound, Some(b)),
        (QuadScalarRole::Tolerance, rest.first()),
        (QuadScalarRole::Trace, rest.get(1)),
    ] {
        let Some(Value::GpuTensor(handle)) = value else {
            continue;
        };
        handle
            .allocation_lease()
            .validate_live()
            .map_err(|source| {
                build_runtime_error(format!(
                    "quad: {} has an invalid resident lease",
                    role.label()
                ))
                .with_builtin(NAME)
                .with_source(source)
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
            })?;
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum QuadScalarRole {
    LowerBound,
    UpperBound,
    Tolerance,
    Trace,
}

impl QuadScalarRole {
    const fn label(self) -> &'static str {
        match self {
            Self::LowerBound => "lower bound",
            Self::UpperBound => "upper bound",
            Self::Tolerance => "tolerance",
            Self::Trace => "trace",
        }
    }

    const fn error(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::LowerBound | Self::UpperBound => &QUAD_ERROR_INVALID_INPUT,
            Self::Tolerance | Self::Trace => &QUAD_ERROR_INVALID_ARGUMENT,
        }
    }

    const fn allows_empty(self) -> bool {
        matches!(self, Self::Tolerance | Self::Trace)
    }

    const fn needs_exact_float(self) -> bool {
        !matches!(self, Self::Trace)
    }
}

fn preflight_quad_scalar(role: QuadScalarRole, value: &Value) -> BuiltinResult<()> {
    let label = role.label();
    let error = role.error();
    let count = match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Some(1),
        Value::Tensor(tensor) => Some(tensor.len()),
        Value::LogicalArray(array) => Some(array.data.len()),
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_storage(handle)
                != runmat_accelerate_api::GpuTensorStorage::Real
            {
                return Err(quad_error_with_detail(
                    error,
                    format!("{label} must be real"),
                ));
            }
            handle
                .shape()
                .iter()
                .try_fold(1_usize, |total, dim| total.checked_mul(*dim))
        }
        _ => None,
    };
    if count != Some(1) && !(role.allows_empty() && count == Some(0)) {
        return Err(quad_error_with_detail(
            error,
            format!(
                "{label} must be a finite real scalar{}",
                if role.allows_empty() { " or empty" } else { "" }
            ),
        ));
    }
    if count == Some(0) || matches!(value, Value::GpuTensor(_)) {
        return Ok(());
    }
    if role.needs_exact_float()
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64(value)
    {
        return Err(quad_error_with_detail(
            error,
            format!("integer {label} must be exactly representable as double"),
        ));
    }
    let finite = match value {
        Value::Num(number) => number.is_finite(),
        Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::Tensor(tensor) => tensor::tensor_value_f64(tensor, 0).is_finite(),
        Value::GpuTensor(_) => true,
        _ => false,
    };
    if !finite {
        return Err(quad_error_with_detail(
            error,
            format!("{label} must be finite"),
        ));
    }
    if role == QuadScalarRole::Tolerance {
        let positive = match value {
            Value::Num(number) => *number > 0.0,
            Value::Int(integer) => integer.to_f64() > 0.0,
            Value::Bool(flag) => *flag,
            Value::LogicalArray(array) => array.data[0] != 0,
            Value::Tensor(tensor) => tensor::tensor_value_f64(tensor, 0) > 0.0,
            Value::GpuTensor(_) => true,
            _ => false,
        };
        if !positive {
            return Err(quad_error_with_detail(
                error,
                "tolerance must be a positive finite scalar",
            ));
        }
    }
    Ok(())
}

fn ensure_quad_extensions(
    a: &Value,
    b: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    for bound in [a, b] {
        if crate::builtins::common::validation::value_contains_native_integer_class(bound) {
            crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_BOUND_EXTENSION, NAME)?;
        }
        if is_logical_numeric(bound) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &LOGICAL_NUMERIC_EXTENSION,
                NAME,
            )?;
        }
    }
    if let Some(tol) = rest.first() {
        if crate::builtins::common::validation::value_contains_native_integer_class(tol) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTEGER_TOLERANCE_EXTENSION,
                NAME,
            )?;
        }
        if is_logical_numeric(tol) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &LOGICAL_NUMERIC_EXTENSION,
                NAME,
            )?;
        }
    }
    if crate::builtins::common::validation::value_contains_explicit_gpu(a)
        || crate::builtins::common::validation::value_contains_explicit_gpu(b)
        || rest
            .iter()
            .any(|value| crate::builtins::common::validation::value_contains_explicit_gpu(value))
    {
        crate::compatibility::ensure_builtin_extension_enabled(&RESIDENT_INPUT_EXTENSION, NAME)?;
    }
    Ok(())
}

fn is_logical_numeric(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

async fn prepare_quad_floating_value(
    label: &str,
    value: Value,
    integer_extension: &'static BuiltinExtensionDescriptor,
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<Value> {
    let has_integer =
        crate::builtins::common::validation::value_contains_native_integer_class(&value);
    if has_integer {
        crate::compatibility::ensure_builtin_extension_enabled(integer_extension, NAME)?;
    }
    let prepared = if has_integer {
        let prepared = cooperation.prepare_integer_boundary(&value).await?;
        if !prepared.is_exact() {
            return Err(quad_error_with_detail(
                &QUAD_ERROR_INVALID_ARGUMENT,
                format!("integer {label} must be exactly representable as double"),
            ));
        }
        Some(prepared)
    } else {
        None
    };
    if is_logical_numeric(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(&LOGICAL_NUMERIC_EXTENSION, NAME)?;
    }
    if crate::builtins::common::validation::value_contains_explicit_gpu(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(&RESIDENT_INPUT_EXTENSION, NAME)?;
    }
    if let Some(prepared) = prepared {
        if let Some(payload) = prepared.into_resident_payload() {
            return match payload {
                crate::builtins::common::validation::PreparedIntegerPayload::Real(gathered) => {
                    Tensor::new_integer(
                        crate::builtins::common::gpu_helpers::integer_storage_from_owned(
                            gathered.data,
                        ),
                        gathered.shape,
                    )
                    .map(Value::Tensor)
                    .map_err(|detail| quad_error_with_detail(&QUAD_ERROR_INVALID_INPUT, detail))
                }
                crate::builtins::common::validation::PreparedIntegerPayload::Complex(_) => {
                    Err(quad_error_with_detail(
                        &QUAD_ERROR_INVALID_INPUT,
                        format!("{label} must be real"),
                    ))
                }
            };
        }
    }
    crate::dispatcher::gather_if_needed_async(&value).await
}

struct QuadOptions {
    tol: f64,
    trace: bool,
    extra_args: Vec<Value>,
}

impl QuadOptions {
    async fn parse(
        rest: crate::call::BuiltinArguments,
        cooperation: &mut QuadratureCooperation,
    ) -> BuiltinResult<Self> {
        let mut values = rest.into_iter();
        let tol = match values.next() {
            Some(value) => parse_optional_tol(value, cooperation).await?,
            None => DEFAULT_TOL,
        };
        let trace = match values.next() {
            Some(value) => parse_optional_trace(value).await?,
            None => false,
        };
        Ok(Self {
            tol,
            trace,
            extra_args: values.collect(),
        })
    }
}

async fn parse_optional_tol(
    value: Value,
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<f64> {
    let value = prepare_quad_floating_value(
        "tolerance",
        value,
        &INTEGER_TOLERANCE_EXTENSION,
        cooperation,
    )
    .await?;
    if is_empty_value(&value) {
        return Ok(DEFAULT_TOL);
    }
    let tol = scalar_real_sync("tolerance", value, &QUAD_ERROR_INVALID_ARGUMENT)?;
    if tol > 0.0 {
        Ok(tol)
    } else {
        Err(quad_error_with_detail(
            &QUAD_ERROR_INVALID_ARGUMENT,
            "tolerance must be a positive finite scalar",
        ))
    }
}

async fn parse_optional_trace(value: Value) -> BuiltinResult<bool> {
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    if is_empty_value(&value) {
        return Ok(false);
    }
    scalar_nonzero_sync("trace", value, &QUAD_ERROR_INVALID_ARGUMENT)
}

fn is_empty_value(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.is_empty(),
        Value::LogicalArray(LogicalArray { data, .. }) => data.is_empty(),
        _ => false,
    }
}

async fn scalar_real(
    label: &str,
    value: Value,
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<f64> {
    let value =
        prepare_quad_floating_value(label, value, &INTEGER_BOUND_EXTENSION, cooperation).await?;
    scalar_real_sync(label, value, &QUAD_ERROR_INVALID_INPUT)
}

fn scalar_nonzero_sync(
    label: &str,
    value: Value,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<bool> {
    if let Some(integer) = tensor::scalar_integer_value(&value) {
        return Ok(!integer.is_zero());
    }
    match value {
        Value::Bool(flag) => Ok(flag),
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => Ok(data[0] != 0),
        other => Ok(scalar_real_sync(label, other, error)? != 0.0),
    }
}

fn scalar_real_sync(
    label: &str,
    value: Value,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<f64> {
    let parsed = match value {
        Value::Num(n) => n,
        Value::Int(i) => i.to_f64(),
        Value::Bool(flag) => {
            if flag {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(&tensor) => {
            tensor::tensor_value_f64(&tensor, 0)
        }
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => {
            if data[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        other => {
            return Err(quad_error_with_detail(
                error,
                format!("{label} must be a finite real scalar, got {other:?}"),
            ))
        }
    };
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(quad_error_with_detail(
            error,
            format!("{label} must be finite"),
        ))
    }
}

#[derive(Clone, Copy)]
struct QuadResult {
    q: f64,
    func_count: usize,
}

struct QuadEvaluator<'a> {
    function: &'a Value,
    tolerance: f64,
    arguments: Vec<Value>,
    cooperation: &'a mut QuadratureCooperation,
    trace: Option<QuadTrace>,
}

impl<'a> QuadEvaluator<'a> {
    fn new(
        function: &'a Value,
        options: QuadOptions,
        cooperation: &'a mut QuadratureCooperation,
    ) -> BuiltinResult<Self> {
        let capacity = options.extra_args.len().checked_add(1).ok_or_else(|| {
            build_runtime_error("quad: callback argument count overflow")
                .with_identifier("RunMat:Internal")
                .with_builtin(NAME)
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
        })?;
        let mut arguments = Vec::new();
        arguments.try_reserve_exact(capacity).map_err(|source| {
            build_runtime_error("quad: could not reserve callback arguments")
                .with_identifier("RunMat:Internal")
                .with_builtin(NAME)
                .with_source(source)
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
        })?;
        arguments.push(Value::Num(0.0));
        arguments.extend(options.extra_args);
        Ok(Self {
            function,
            tolerance: options.tol,
            arguments,
            cooperation,
            trace: options.trace.then_some(QuadTrace),
        })
    }
}

impl QuadratureEvaluator for QuadEvaluator<'_> {
    type Sample = f64;

    async fn checkpoint(&mut self) -> BuiltinResult<()> {
        self.cooperation.completed(0).await
    }

    async fn evaluate(&mut self, x: f64) -> BuiltinResult<f64> {
        call_integrand(self.function, x, &mut self.arguments, self.cooperation).await
    }

    fn midpoint(&self, a: f64, b: f64) -> f64 {
        midpoint(a, b)
    }

    fn tolerances(&self) -> (f64, f64) {
        (self.tolerance, 0.0)
    }

    fn max_evaluations(&self) -> usize {
        MAX_FUN_EVALS
    }

    fn max_depth(&self) -> usize {
        MAX_DEPTH
    }

    fn evaluation_limit_error(&self) -> RuntimeError {
        quad_error_with_detail(
            &QUAD_ERROR_INVALID_INPUT,
            "exceeded maximum function evaluations",
        )
    }

    fn convergence_error(&self) -> RuntimeError {
        quad_error_with_detail(
            &QUAD_ERROR_INVALID_INPUT,
            "adaptive Simpson quadrature did not converge",
        )
    }

    fn nonrepresentable_error(&self) -> RuntimeError {
        quad_error_with_detail(
            &QUAD_ERROR_INVALID_INPUT,
            "quadrature estimate is not representable as a finite double",
        )
    }

    fn arithmetic_error(&self, error: QuadratureArithmeticError) -> RuntimeError {
        quad_arithmetic_error(error)
    }

    fn refined(
        &mut self,
        evals: usize,
        a: f64,
        b: f64,
        estimate: &ScaledEstimate,
        error: &ScaledEstimate,
    ) -> BuiltinResult<()> {
        if let Some(trace) = self.trace.as_mut() {
            trace.record(evals, a, b, estimate.trace_f64(), error.trace_f64())?;
        }
        Ok(())
    }
}

async fn call_integrand(
    function: &Value,
    x: f64,
    arguments: &mut [Value],
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<f64> {
    cooperation.completed(0).await?;
    arguments[0] = Value::Num(x);
    let value = call_function(function, &*arguments).await?;
    cooperation.completed(1).await?;
    value
        .with_owned_async(|value| async move {
            let value = prepare_quad_floating_value(
                "integrand result",
                value,
                &INTEGER_CALLBACK_EXTENSION,
                cooperation,
            )
            .await?;
            value_to_scalar(NAME, value)
        })
        .await
}

struct QuadTrace;

impl QuadTrace {
    fn record(
        &mut self,
        func_count: usize,
        a: f64,
        b: f64,
        q: f64,
        _err: f64,
    ) -> BuiltinResult<()> {
        crate::console::record_console_line(
            crate::console::ConsoleStream::Stdout,
            format!(
                "    {func_count:>5}    {a:13.6e} {width:13.6e} {q:13.6e}",
                width = b - a,
            ),
        )
    }
}

fn finalize(result: QuadResult) -> BuiltinResult<Value> {
    let q = Value::Num(result.q);
    let fcnt = Value::Num(result.func_count as f64);
    match crate::output_count::current_output_count() {
        None => Ok(q),
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(crate::output_count::output_list_with_padding(1, vec![q])),
        Some(2) => Ok(crate::output_count::output_list_with_padding(
            2,
            vec![q, fcnt],
        )),
        Some(_) => Err(quad_error_with_detail(
            &QUAD_ERROR_TOO_MANY_OUTPUTS,
            "quad: too many output arguments; maximum is 2",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{IntegerStorage, Tensor};
    use std::sync::Arc;

    fn selected_integrand(
        function: usize,
        name: &str,
    ) -> (
        crate::context::RuntimeContext,
        Value,
        crate::user_functions::BoundFunctionInvokerGuard,
    ) {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, function);
        let entry = runtime.enter();
        let forwarder =
            crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        drop(entry);
        (
            runtime,
            crate::user_functions::test_support::exact_local_value(name, binding),
            forwarder,
        )
    }

    #[test]
    fn quad_integrates_sine_with_default_tolerance() {
        let result = block_on(quad_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(std::f64::consts::PI),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("quad");
        match result {
            Value::Num(value) => assert!((value - 2.0).abs() < 1.0e-6),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn quad_respects_tighter_tolerance_on_polynomial() {
        let (runtime, integrand, _bound) = selected_integrand(7, "square");
        let _entry = runtime.enter();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, requested_outputs| {
                assert_eq!(function, 7);
                assert_eq!(requested_outputs, 1);
                let x = match &args[0] {
                    Value::Num(value) => *value,
                    other => panic!("expected x, got {other:?}"),
                };
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        x * x,
                    ))
                })
            },
        )))
        .expect("call-port scope");

        let result = block_on(runtime.scope(quad_builtin(
            integrand,
            Value::Num(0.0),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0e-10)]),
        )))
        .expect("quad");
        match result {
            Value::Num(value) => assert!((value - (1.0 / 3.0)).abs() < 1.0e-10),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn quad_two_outputs_include_function_count() {
        let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let result = block_on(quad_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(std::f64::consts::PI),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("quad");
        match result {
            Value::OutputList(outputs) => {
                assert_eq!(outputs.len(), 2);
                assert!(matches!(&outputs[0], Value::Num(value) if (value - 2.0).abs() < 1.0e-6));
                assert!(matches!(&outputs[1], Value::Num(fcnt) if *fcnt >= 5.0));
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn quad_trace_records_rows() {
        crate::console::reset_thread_buffer();
        let result = block_on(quad_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(std::f64::consts::PI),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1.0e-6),
                Value::Num(1.0),
            ]),
        ))
        .expect("quad");
        assert!(matches!(result, Value::Num(_)));

        let joined = crate::console::take_thread_buffer().into_iter().fold(
            String::new(),
            |mut text, entry| {
                text.push_str(entry.text.as_str());
                text
            },
        );
        let first_row: Vec<&str> = joined
            .lines()
            .next()
            .expect("expected at least one trace row")
            .split_whitespace()
            .collect();
        assert_eq!(first_row.len(), 4, "{joined}");
        assert_eq!(first_row[0], "5", "{joined}");
        assert!((first_row[1].parse::<f64>().unwrap() - 0.0).abs() < 1.0e-12);
        assert!(
            (first_row[2].parse::<f64>().unwrap() - std::f64::consts::PI).abs() < 1.0e-6,
            "{joined}"
        );
    }

    #[test]
    fn quad_tol_and_trace_read_typed_integer_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        crate::console::reset_thread_buffer();
        let tol = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("tol");
        let trace = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("trace");

        let result = block_on(quad_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(tol),
                Value::Tensor(trace),
            ]),
        ))
        .expect("quad");
        assert!(matches!(result, Value::Num(_)));
        assert!(
            !crate::console::take_thread_buffer().is_empty(),
            "typed trace value should enable trace output"
        );
    }

    #[test]
    fn quad_strict_mode_rejects_integer_tolerance() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let tolerance = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).unwrap();

        let error = block_on(quad_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
                tolerance,
            )]),
        ))
        .expect_err("integer tolerance is a RunMat-only extension");

        assert_eq!(
            error.identifier(),
            INTEGER_TOLERANCE_EXTENSION.error_identifier
        );
    }

    #[test]
    fn quad_rejects_wide_integer_bound_before_float_conversion() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let bound =
            Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]).unwrap();

        let error = block_on(quad_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Tensor(bound),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("wide integer bound cannot cross exactly");

        assert!(error.message().contains("exactly representable"));
    }

    #[test]
    fn quad_integer_trace_uses_exact_zero_test_in_strict_mode() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        crate::console::reset_thread_buffer();
        let trace = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap();

        let result = block_on(quad_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(Tensor::zeros(vec![0, 0])),
                Value::Tensor(trace),
            ]),
        ))
        .expect("documented integer trace selector");

        assert!(matches!(result, Value::Num(_)));
        assert!(!crate::console::take_thread_buffer().is_empty());
    }

    #[test]
    fn quad_passes_wide_integer_extra_parameter_exactly() {
        let (runtime, integrand, _bound) = selected_integrand(902, "parameterized");
        let _entry = runtime.enter();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let expected = u64::MAX;
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |function, args, _requested_outputs| {
                assert_eq!(function, 902);
                let Value::Tensor(parameter) = &args[1] else {
                    panic!("expected integer parameter")
                };
                assert!(matches!(
                    parameter.numeric_value_at(0),
                    Some(runmat_value::NumericScalar::U64(value)) if value == expected
                ));
                let Value::Num(x) = args[0] else {
                    panic!("expected scalar quadrature point")
                };
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(x))
                })
            },
        )))
        .expect("call-port scope");
        let parameter =
            Tensor::new_integer(IntegerStorage::U64(vec![expected]), vec![1, 1]).unwrap();

        let result = block_on(runtime.scope(quad_builtin(
            integrand,
            Value::Num(0.0),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(Tensor::zeros(vec![0, 0])),
                Value::Num(0.0),
                Value::Tensor(parameter),
            ]),
        )))
        .expect("exact callback parameter");

        assert!(matches!(result, Value::Num(value) if (value - 0.5).abs() < 1.0e-6));
    }

    #[test]
    fn quad_automatic_resident_bound_gathers_but_explicit_bound_is_gated() {
        test_support::with_test_provider(|provider| {
            let values = [1.0];
            let shape = [1, 1];
            let automatic = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &shape,
                })
                .expect("automatic upload");
            let automatic = automatic
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .expect("admit provenance view");
            let result = block_on(quad_builtin(
                Value::FunctionHandle("sin".into()),
                Value::Num(0.0),
                Value::GpuTensor(automatic),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect("automatic resident bound gathers");
            assert!(matches!(result, Value::Num(_)));

            let explicit = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &shape,
                })
                .expect("explicit upload");
            let explicit = explicit
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("admit provenance view");
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(quad_builtin(
                Value::FunctionHandle("sin".into()),
                Value::Num(0.0),
                Value::GpuTensor(explicit),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect_err("explicit resident bound is gated before fallback");
            assert_eq!(
                error.identifier(),
                RESIDENT_INPUT_EXTENSION.error_identifier
            );
        });
    }

    #[test]
    fn quad_forwards_extra_arguments_after_tol_and_trace() {
        let (runtime, integrand, _bound) = selected_integrand(42, "scaled_line");
        let _entry = runtime.enter();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, requested_outputs| {
                assert_eq!(function, 42);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args.len(), 2);
                let x = match &args[0] {
                    Value::Num(value) => *value,
                    other => panic!("expected x, got {other:?}"),
                };
                let scale = match &args[1] {
                    Value::Num(value) => *value,
                    other => panic!("expected scale, got {other:?}"),
                };
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        scale * x,
                    ))
                })
            },
        )))
        .expect("call-port scope");

        let result = block_on(runtime.scope(quad_builtin(
            integrand,
            Value::Num(0.0),
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(Tensor::zeros(vec![0, 0])),
                Value::Tensor(Tensor::zeros(vec![0, 0])),
                Value::Num(3.0),
            ]),
        )))
        .expect("quad");
        match result {
            Value::Num(value) => assert!((value - 6.0).abs() < 1.0e-8),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn quad_handles_oscillatory_integrand() {
        let result = block_on(quad_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(2.0 * std::f64::consts::PI),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0e-8)]),
        ))
        .expect("quad");
        match result {
            Value::Num(value) => assert!(value.abs() < 1.0e-8),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn quad_handles_integrable_endpoint_shape() {
        let (runtime, integrand, _bound) = selected_integrand(9, "sqrt_fn");
        let _entry = runtime.enter();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, _requested_outputs| {
                assert_eq!(function, 9);
                let x = match &args[0] {
                    Value::Num(value) => *value,
                    other => panic!("expected x, got {other:?}"),
                };
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        x.sqrt(),
                    ))
                })
            },
        )))
        .expect("call-port scope");

        let result = block_on(runtime.scope(quad_builtin(
            integrand,
            Value::Num(0.0),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0e-7)]),
        )))
        .expect("quad");
        match result {
            Value::Num(value) => assert!((value - (2.0 / 3.0)).abs() < 1.0e-6),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn quad_reversed_bounds_negate_result() {
        let result = block_on(quad_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(std::f64::consts::PI),
            Value::Num(0.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("quad");
        match result {
            Value::Num(value) => assert!((value + 2.0).abs() < 1.0e-6),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn quad_rejects_more_than_two_outputs() {
        let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let err = block_on(quad_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("too many outputs should fail");
        assert_eq!(err.identifier(), Some("RunMat:quad:TooManyOutputs"));
    }

    #[test]
    fn quad_descriptor_signatures_cover_legacy_forms() {
        let labels: Vec<&str> = QUAD_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(
            labels,
            vec![
                "q = quad(fun, a, b)",
                "q = quad(fun, a, b, tol, trace, p1, p2, ...)",
                "[q, fcnt] = quad(fun, a, b)",
                "[q, fcnt] = quad(fun, a, b, tol, trace, p1, p2, ...)",
            ]
        );

        let codes: Vec<&str> = QUAD_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert_eq!(
            codes,
            vec![
                "RM.QUAD.INVALID_ARGUMENT",
                "RM.QUAD.INVALID_INPUT",
                "RM.QUAD.TOO_MANY_OUTPUTS",
            ]
        );
    }

    #[test]
    fn invalid_host_tolerance_is_rejected_before_callback() {
        let error = block_on(quad_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(-1.0)]),
        ))
        .expect_err("negative tolerance");
        assert_eq!(error.identifier(), QUAD_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn invalid_callback_is_rejected_during_admission() {
        let error = block_on(quad_builtin(
            Value::Num(7.0),
            Value::Num(0.0),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("invalid callback");
        assert_eq!(
            error.identifier(),
            runmat_builtins::FEVAL_ERROR_FUNCTION_VALUE_UNSUPPORTED.identifier,
        );
    }

    #[test]
    fn registered_quad_observes_preexisting_cancellation_before_callback() {
        use runmat_execution::cancellation::CancellationSource;
        use runmat_execution::CancellationReason;
        use std::rc::Rc;

        let source = CancellationSource::new();
        source.cancel(CancellationReason::User);
        let runtime = crate::context::RuntimeContext::with_cancellation(
            Rc::new(crate::execution::RuntimeExecutionService::new()),
            source.token(),
        );
        let error = block_on(quad_builtin_registered(
            Some(runtime),
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("cancelled invocation");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        assert!(error.source.is_some());
    }
}
