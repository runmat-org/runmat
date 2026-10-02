//! MATLAB-compatible `integral` builtin for finite scalar numerical integration.

#[cfg(test)]
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, INTEGRAL_DESCRIPTOR,
};
use runmat_builtins::{
    BuiltinErrorDescriptor, IntegralCallbackMode, IntegralOptionName,
    INTEGRAL_ERROR_INVALID_ARGUMENT, INTEGRAL_ERROR_INVALID_INPUT, INTEGRAL_ERROR_TOO_MANY_OUTPUTS,
    INTEGRAL_INTEGER_BOUND_EXTENSION, INTEGRAL_INTEGER_OPTION_EXTENSION,
    INTEGRAL_LOGICAL_BOUND_EXTENSION, INTEGRAL_LOGICAL_NUMERIC_OPTION_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{LogicalArray, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::optim::common::call_function;
use crate::builtins::math::optim::quadrature::{
    integrate, AdaptiveSimpsonResult, QuadratureArithmeticError, QuadratureCooperation,
    QuadratureEvaluator, QuadratureIdentity,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod array;

const NAME: &str = "integral";
const DEFAULT_MAX_FUN_EVALS: usize = 10_000;
const MAX_DEPTH: usize = 30;

fn integral_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("integral:") {
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

fn integral_map_error(
    err: RuntimeError,
    fallback: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        let mut mapped = integral_error_with_detail(fallback, err.message());
        mapped.span = err.span.clone();
        mapped.context = err.context.clone();
        mapped.context.builtin = Some(NAME.to_string());
        mapped.gpu_gather_retry = err.gpu_gather_retry;
        mapped.source = Some(Box::new(err));
        mapped
    }
}

fn quadrature_arithmetic_error(error: QuadratureArithmeticError) -> RuntimeError {
    match error {
        QuadratureArithmeticError::Resource(source) => {
            build_runtime_error("integral: could not reserve array-valued quadrature workspace")
                .with_identifier("RunMat:Internal")
                .with_builtin(NAME)
                .with_source(source)
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
        }
        QuadratureArithmeticError::InconsistentComponents => integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            "array-valued callback changed its output size",
        ),
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::integral")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "integral",
    op_kind: GpuOpKind::Custom("adaptive-quadrature"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host adaptive quadrature solver. Callback computations may use GPU-aware builtins, but the adaptive integration loop runs on the CPU.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::integral")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "integral",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Adaptive integration repeatedly invokes user code and terminates fusion planning.",
};

#[runtime_builtin(
    name = "integral",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::integral"
)]
async fn integral_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    function: Value,
    a: Value,
    b: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    integral_in_runtime(runtime.as_ref(), function, a, b, rest).await
}

#[cfg(test)]
async fn integral_builtin(
    function: Value,
    a: Value,
    b: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    integral_in_runtime(None, function, a, b, rest).await
}

async fn integral_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    function: Value,
    a: Value,
    b: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let mut cooperation = QuadratureCooperation::new(QuadratureIdentity::Integral, runtime);
    cooperation.completed(0).await?;
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(integral_error_with_detail(
            &INTEGRAL_ERROR_TOO_MANY_OUTPUTS,
            "integral: too many output arguments; maximum is 1",
        ));
    }
    preflight_integral_bound_extensions(&a, &b)?;
    let admitted_options = admit_option_headers(&rest)?;
    preflight_integral_option_extensions(&admitted_options)?;
    crate::call::admission::admit_feval_target(&function)?;
    for (label, value) in [("lower bound", &a), ("upper bound", &b)] {
        preflight_integral_bound(label, value)?;
    }
    IntegralOptions::preflight_host(&admitted_options)?;
    authenticate_integral_residents(&a, &b, &admitted_options)?;
    let options = IntegralOptions::parse(&admitted_options)
        .await
        .map_err(|err| integral_map_error(err, &INTEGRAL_ERROR_INVALID_ARGUMENT))?;
    let a = scalar_bound("lower bound", a)
        .await
        .map_err(|err| integral_map_error(err, &INTEGRAL_ERROR_INVALID_INPUT))?;
    let b = scalar_bound("upper bound", b)
        .await
        .map_err(|err| integral_map_error(err, &INTEGRAL_ERROR_INVALID_INPUT))?;
    let sign = if b < a { -1.0 } else { 1.0 };
    let lo = a.min(b);
    let hi = a.max(b);
    if options.array_valued {
        let result =
            array::integrate_array_valued(&function, lo, hi, sign, &options, &mut cooperation)
                .await?;
        cooperation.completed(0).await?;
        return Ok(result);
    }
    if a == b {
        cooperation.completed(0).await?;
        return Ok(Value::Num(0.0));
    }
    let result = {
        let mut evaluator = IntegralEvaluator {
            function: &function,
            options: &options,
            cooperation: &mut cooperation,
        };
        let AdaptiveSimpsonResult { q, .. } = integrate(&mut evaluator, lo, hi)
            .await
            .map_err(|err| integral_map_error(err, &INTEGRAL_ERROR_INVALID_INPUT))?;
        q
    };
    cooperation.completed(0).await?;
    Ok(Value::Num(sign * result))
}

fn preflight_integral_bound_extensions(a: &Value, b: &Value) -> BuiltinResult<()> {
    use crate::builtins::common::validation::{
        value_has_logical_class, value_has_native_integer_class,
    };
    for bound in [a, b] {
        if value_has_native_integer_class(bound) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTEGRAL_INTEGER_BOUND_EXTENSION,
                NAME,
            )?;
        }
        if value_has_logical_class(bound) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTEGRAL_LOGICAL_BOUND_EXTENSION,
                NAME,
            )?;
        }
    }
    Ok(())
}

fn preflight_integral_option_extensions(
    options: &[(IntegralOptionName, &Value)],
) -> BuiltinResult<()> {
    use crate::builtins::common::validation::{
        value_has_logical_class, value_has_native_integer_class,
    };
    for &(name, value) in options {
        if value_has_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTEGRAL_INTEGER_OPTION_EXTENSION,
                NAME,
            )?;
        }
        if name.is_numeric() && value_has_logical_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTEGRAL_LOGICAL_NUMERIC_OPTION_EXTENSION,
                NAME,
            )?;
        }
    }
    Ok(())
}

fn preflight_integral_bound(label: &str, value: &Value) -> BuiltinResult<()> {
    if let Value::GpuTensor(handle) = value {
        use runmat_accelerate_api::{handle_storage, GpuTensorStorage};
        let scalar = handle
            .shape()
            .iter()
            .try_fold(1_usize, |count, axis| count.checked_mul(*axis))
            == Some(1);
        if !scalar || handle_storage(handle) != GpuTensorStorage::Real {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_INPUT,
                format!("{label} must be a finite real scalar"),
            ));
        }
        return Ok(());
    }
    scalar_bound_host(label, value).map(|_| ())
}

fn authenticate_integral_residents(
    a: &Value,
    b: &Value,
    options: &[(IntegralOptionName, &Value)],
) -> BuiltinResult<()> {
    for (label, value) in [("lower bound", a), ("upper bound", b)].into_iter().chain(
        options
            .iter()
            .map(|(name, value)| (name.display_name(), *value)),
    ) {
        let Value::GpuTensor(handle) = value else {
            continue;
        };
        handle
            .allocation_lease()
            .validate_live()
            .map_err(|source| {
                build_runtime_error(format!("integral: {label} has an invalid resident lease"))
                    .with_builtin(NAME)
                    .with_source(source)
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build()
            })?;
    }
    Ok(())
}

fn admit_option_headers(rest: &[Value]) -> BuiltinResult<Vec<(IntegralOptionName, &Value)>> {
    if let [Value::Struct(fields)] = rest {
        let mut admitted = Vec::new();
        admitted
            .try_reserve_exact(fields.fields.len())
            .map_err(|source| {
                integral_error_with_detail(&INTEGRAL_ERROR_INVALID_ARGUMENT, source.to_string())
            })?;
        for (name, value) in &fields.fields {
            admitted.push((decode_option_name(name)?, value));
        }
        return Ok(admitted);
    }
    if !rest.len().is_multiple_of(2) {
        return Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            "expected option name/value pairs",
        ));
    }
    let mut admitted = Vec::new();
    admitted
        .try_reserve_exact(rest.len() / 2)
        .map_err(|source| {
            integral_error_with_detail(&INTEGRAL_ERROR_INVALID_ARGUMENT, source.to_string())
        })?;
    for pair in rest.chunks_exact(2) {
        admitted.push((option_name(&pair[0])?, &pair[1]));
    }
    Ok(admitted)
}

#[derive(Clone, Copy)]
struct IntegralOptions {
    abs_tol: f64,
    rel_tol: f64,
    max_fun_evals: usize,
    callback_mode: IntegralCallbackMode,
    array_valued: bool,
}

impl IntegralOptions {
    fn defaults() -> Self {
        Self {
            abs_tol: IntegralOptionName::DEFAULT_ABS_TOL,
            rel_tol: IntegralOptionName::DEFAULT_REL_TOL,
            max_fun_evals: DEFAULT_MAX_FUN_EVALS,
            callback_mode: IntegralCallbackMode::Vectorized,
            array_valued: false,
        }
    }

    fn preflight_host(rest: &[(IntegralOptionName, &Value)]) -> BuiltinResult<()> {
        let mut check = Self::defaults();
        let mut known_abs_tol = Some(check.abs_tol);
        let mut known_rel_tol = Some(check.rel_tol);
        for &(name, value) in rest {
            if matches!(value, Value::GpuTensor(_)) {
                match name {
                    IntegralOptionName::AbsTol => known_abs_tol = None,
                    IntegralOptionName::RelTol => known_rel_tol = None,
                    IntegralOptionName::MaxFunEvals
                    | IntegralOptionName::MaxIntervalCount
                    | IntegralOptionName::ArrayValued
                    | IntegralOptionName::Vectorized => {}
                }
                continue;
            }
            check.apply_option(name, value)?;
            match name {
                IntegralOptionName::AbsTol => known_abs_tol = Some(check.abs_tol),
                IntegralOptionName::RelTol => known_rel_tol = Some(check.rel_tol),
                IntegralOptionName::MaxFunEvals
                | IntegralOptionName::MaxIntervalCount
                | IntegralOptionName::ArrayValued
                | IntegralOptionName::Vectorized => {}
            }
        }
        if known_abs_tol == Some(0.0) && known_rel_tol == Some(0.0) {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "AbsTol and RelTol cannot both be zero",
            ));
        }
        if known_abs_tol.is_some_and(|value| value < 0.0) {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "AbsTol must be nonnegative",
            ));
        }
        if known_rel_tol.is_some_and(|value| value < 0.0) {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "RelTol must be nonnegative",
            ));
        }
        Ok(())
    }

    async fn parse(rest: &[(IntegralOptionName, &Value)]) -> BuiltinResult<Self> {
        let mut options = Self::defaults();
        for &(name, value) in rest {
            if matches!(value, Value::GpuTensor(_)) {
                let gathered = crate::dispatcher::gather_if_needed_async(value).await?;
                options.apply_option(name, &gathered)?;
            } else {
                options.apply_option(name, value)?;
            }
        }
        options.validate()?;
        Ok(options)
    }

    fn apply_option(&mut self, name: IntegralOptionName, value: &Value) -> BuiltinResult<()> {
        match name {
            IntegralOptionName::AbsTol => {
                self.abs_tol = numeric_option(name.display_name(), value)?
            }
            IntegralOptionName::RelTol => {
                self.rel_tol = numeric_option(name.display_name(), value)?
            }
            IntegralOptionName::MaxFunEvals | IntegralOptionName::MaxIntervalCount => {
                let parsed = integer_option(name.display_name(), value)?;
                if parsed < 5 {
                    return Err(integral_error_with_detail(
                        &INTEGRAL_ERROR_INVALID_ARGUMENT,
                        "MaxFunEvals must be an integer scalar >= 5",
                    ));
                }
                self.max_fun_evals = parsed;
            }
            IntegralOptionName::ArrayValued => {
                self.array_valued = bool_option("ArrayValued", value)?;
            }
            IntegralOptionName::Vectorized => {
                self.callback_mode =
                    IntegralCallbackMode::from_vectorized(bool_option("Vectorized", value)?);
            }
        }
        Ok(())
    }

    fn validate(&self) -> BuiltinResult<()> {
        if self.abs_tol < 0.0 {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "AbsTol must be nonnegative",
            ));
        }
        if self.rel_tol < 0.0 {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "RelTol must be nonnegative",
            ));
        }
        if self.abs_tol == 0.0 && self.rel_tol == 0.0 {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "AbsTol and RelTol cannot both be zero",
            ));
        }
        Ok(())
    }
}

fn option_name(value: &Value) -> BuiltinResult<IntegralOptionName> {
    let text = match value {
        Value::String(s) => s.as_str(),
        Value::StringArray(sa) if sa.data.len() == 1 => sa.data[0].as_str(),
        Value::CharArray(chars) if chars.rows == 1 => {
            return decode_option_name(&chars.data.iter().collect::<String>());
        }
        other => Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("option names must be strings, got {other:?}"),
        ))?,
    };
    decode_option_name(text)
}

fn decode_option_name(text: &str) -> BuiltinResult<IntegralOptionName> {
    IntegralOptionName::parse(text).ok_or_else(|| {
        integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("unsupported option {text}"),
        )
    })
}

async fn scalar_bound(label: &str, value: Value) -> BuiltinResult<f64> {
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    scalar_bound_host(label, &value)
}

fn scalar_bound_host(label: &str, value: &Value) -> BuiltinResult<f64> {
    if !crate::builtins::common::validation::native_integer_value_is_exact_f64(&value) {
        return Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            format!("{label} must be exactly representable as double"),
        ));
    }
    let parsed = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => scalar_tensor_value(tensor)
            .ok_or_else(|| {
                integral_error_with_detail(
                    &INTEGRAL_ERROR_INVALID_INPUT,
                    format!("{label} must be a finite real scalar"),
                )
            })?,
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => {
            if data[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        other => {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_INPUT,
                format!("{label} must be a finite real scalar, got {other:?}"),
            ))
        }
    };
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            format!("{label} must be finite"),
        ))
    }
}

fn numeric_option(name: &str, value: &Value) -> BuiltinResult<f64> {
    if !crate::builtins::common::validation::native_integer_value_is_exact_f64(value) {
        return Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("option {name} must be exactly representable as double"),
        ));
    }
    let parsed = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => {
            if data[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        other => {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                format!("option {name} must be numeric, got {other:?}"),
            ))
        }
    };
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("option {name} must be finite"),
        ))
    }
}

fn integer_option(name: &str, value: &Value) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer.try_to_usize().ok_or_else(|| {
            integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                format!("option {name} must be nonnegative"),
            )
        });
    }
    let parsed = numeric_option(name, value)?;
    if parsed < 0.0 {
        return Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("option {name} must be nonnegative"),
        ));
    }
    if parsed.fract() != 0.0 {
        return Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("option {name} must be an integer scalar"),
        ));
    }
    if parsed > usize::MAX as f64 || (usize::BITS == 64 && parsed == usize::MAX as f64) {
        return Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("option {name} exceeds maximum supported size"),
        ));
    }
    Ok(parsed as usize)
}

fn bool_option(name: &str, value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Num(n) if *n == 0.0 || *n == 1.0 => Ok(*n != 0.0),
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw == 0 || raw == 1 {
                Ok(raw != 0)
            } else {
                Err(integral_error_with_detail(
                    &INTEGRAL_ERROR_INVALID_ARGUMENT,
                    format!("option {name} must be logical scalar"),
                ))
            }
        }
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => Ok(data[0] != 0),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            if let Some(integer) = tensor::scalar_integer_value(value) {
                match integer.try_to_usize() {
                    Some(0) => Ok(false),
                    Some(1) => Ok(true),
                    _ => Err(integral_error_with_detail(
                        &INTEGRAL_ERROR_INVALID_ARGUMENT,
                        format!("option {name} must be logical scalar"),
                    )),
                }
            } else {
                match tensor::tensor_value_f64(tensor, 0) {
                    0.0 => Ok(false),
                    1.0 => Ok(true),
                    _ => Err(integral_error_with_detail(
                        &INTEGRAL_ERROR_INVALID_ARGUMENT,
                        format!("option {name} must be logical scalar"),
                    )),
                }
            }
        }
        other => Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("option {name} must be logical scalar, got {other:?}"),
        )),
    }
}

struct IntegralEvaluator<'a> {
    function: &'a Value,
    options: &'a IntegralOptions,
    cooperation: &'a mut QuadratureCooperation,
}

impl QuadratureEvaluator for IntegralEvaluator<'_> {
    type Sample = f64;

    async fn checkpoint(&mut self) -> BuiltinResult<()> {
        self.cooperation.completed(0).await
    }

    async fn evaluate(&mut self, x: f64) -> BuiltinResult<f64> {
        call_integrand(self.function, x, self.cooperation).await
    }

    async fn evaluate_many<const N: usize>(&mut self, points: [f64; N]) -> BuiltinResult<[f64; N]> {
        match self.options.callback_mode {
            IntegralCallbackMode::Vectorized => {
                call_integrand_batch(self.function, points, self.cooperation).await
            }
            IntegralCallbackMode::Scalar => {
                let mut values = [0.0; N];
                for (index, point) in points.into_iter().enumerate() {
                    values[index] = self.evaluate(point).await?;
                }
                Ok(values)
            }
        }
    }

    fn midpoint(&self, a: f64, b: f64) -> f64 {
        crate::builtins::math::optim::quadrature::midpoint(a, b)
    }

    fn tolerances(&self) -> (f64, f64) {
        (self.options.abs_tol, self.options.rel_tol)
    }

    fn max_evaluations(&self) -> usize {
        self.options.max_fun_evals
    }

    fn max_depth(&self) -> usize {
        MAX_DEPTH
    }

    fn evaluation_limit_error(&self) -> RuntimeError {
        integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            "exceeded maximum function evaluations",
        )
    }

    fn convergence_error(&self) -> RuntimeError {
        integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            "adaptive quadrature did not converge",
        )
    }

    fn nonrepresentable_error(&self) -> RuntimeError {
        integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            "quadrature estimate is not representable as a finite double",
        )
    }

    fn arithmetic_error(&self, error: QuadratureArithmeticError) -> RuntimeError {
        quadrature_arithmetic_error(error)
    }
}

fn scalar_tensor_value(tensor: &Tensor) -> Option<f64> {
    tensor::is_scalar_tensor(tensor).then(|| tensor::tensor_value_f64(tensor, 0))
}

async fn call_integrand(
    function: &Value,
    x: f64,
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<f64> {
    cooperation.completed(0).await?;
    let value = call_function(function, vec![Value::Num(x)]).await?;
    cooperation.completed(1).await?;
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    match value {
        Value::Num(n) if n.is_finite() => Ok(n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if b { 1.0 } else { 0.0 }),
        Value::Tensor(tensor)
            if tensor::is_scalar_tensor(&tensor)
                && scalar_tensor_value(&tensor)
                    .map(|value| value.is_finite())
                    .unwrap_or(false) =>
        {
            Ok(scalar_tensor_value(&tensor).expect("finite scalar tensor value"))
        }
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
        }
        Value::Num(_) | Value::Tensor(_) => Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            "function value must be a finite real scalar",
        )),
        other => Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            format!("function value must be real numeric scalar, got {other:?}"),
        )),
    }
}

async fn call_integrand_batch<const N: usize>(
    function: &Value,
    points: [f64; N],
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<[f64; N]> {
    cooperation.completed(0).await?;
    let samples = Tensor::new(points.to_vec(), vec![1, N]).map_err(|error| {
        integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            format!("could not construct callback sample batch: {error}"),
        )
    })?;
    let value = call_function(function, [Value::Tensor(samples)]).await?;
    cooperation.completed(N as u64).await?;
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    let mut result = [0.0; N];
    match value {
        Value::Tensor(tensor) if tensor.shape.as_slice() == [1, N] => {
            for (index, result) in result.iter_mut().enumerate() {
                *result = tensor::tensor_value_f64(&tensor, index);
            }
        }
        Value::LogicalArray(logical) if logical.shape.as_slice() == [1, N] => {
            for (index, result) in result.iter_mut().enumerate() {
                *result = if logical.data[index] != 0 { 1.0 } else { 0.0 };
            }
        }
        _ => {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_INPUT,
                "vectorized function must return one real value for each row-vector sample",
            ))
        }
    }
    if result.iter().any(|value| !value.is_finite()) {
        return Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            "function values must be finite real numbers",
        ));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::IntegerStorage;

    async fn mapped_helper(x: Value, map: impl Fn(f64) -> f64) -> crate::BuiltinResult<Value> {
        match x {
            Value::Tensor(samples) => {
                let values = (0..samples.len())
                    .map(|index| map(tensor::tensor_value_f64(&samples, index)))
                    .collect();
                Ok(Value::Tensor(
                    Tensor::new(values, samples.shape).expect("fixed test sample shape"),
                ))
            }
            other => Ok(Value::Num(map(scalar_bound("x", other).await?))),
        }
    }

    const INTEGRAL_HELPER_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
        name: "fx",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integrand values for scalar or vector sample input.",
    }];

    const INTEGRAL_HELPER_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar or row-vector integrand sample locations.",
    }];

    const INTEGRAL_HELPER_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
        [BuiltinSignatureDescriptor {
            label: "fx = __integral_helper(x)",
            inputs: &INTEGRAL_HELPER_INPUTS,
            outputs: &INTEGRAL_HELPER_OUTPUT,
        }];

    const INTEGRAL_HELPER_ERRORS: [BuiltinErrorDescriptor; 0] = [];

    pub const INTEGRAL_TEST_HELPER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
        signatures: &INTEGRAL_HELPER_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::HiddenInternal,
        errors: &INTEGRAL_HELPER_ERRORS,
    };

    #[runtime_builtin(
        name = "__integral_square",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn square_helper(x: Value) -> crate::BuiltinResult<Value> {
        mapped_helper(x, |value| value * value).await
    }

    #[runtime_builtin(
        name = "__integral_one",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn one_helper(_x: Value) -> crate::BuiltinResult<Value> {
        mapped_helper(_x, |_| 1.0).await
    }

    #[runtime_builtin(
        name = "__integral_scaled_identity",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn scaled_identity_helper(x: Value) -> crate::BuiltinResult<Value> {
        mapped_helper(x, |value| value / 1.0e308).await
    }

    #[runtime_builtin(
        name = "__integral_identity",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn identity_helper(x: Value) -> crate::BuiltinResult<Value> {
        Ok(x)
    }

    #[runtime_builtin(
        name = "__integral_scalar_only",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn scalar_only_helper(x: Value) -> crate::BuiltinResult<Value> {
        let Value::Num(x) = x else {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_INPUT,
                "scalar-only callback received a vector batch",
            ));
        };
        Ok(Value::Num(x * x))
    }

    #[runtime_builtin(
        name = "__integral_vector",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn vector_helper(_x: Value) -> crate::BuiltinResult<Value> {
        Ok(Value::Tensor(
            Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(),
        ))
    }

    #[runtime_builtin(
        name = "__integral_array_polynomial",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn array_polynomial_helper(x: Value) -> crate::BuiltinResult<Value> {
        let Value::Num(x) = x else {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_INPUT,
                "array callback did not receive scalar sample",
            ));
        };
        Ok(Value::Tensor(
            Tensor::new(vec![x, x * x, 2.0 * x], vec![1, 3]).expect("fixed shape"),
        ))
    }

    #[runtime_builtin(
        name = "__integral_array_changing_shape",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn array_changing_shape_helper(x: Value) -> crate::BuiltinResult<Value> {
        let Value::Num(x) = x else {
            unreachable!("array callback receives scalar")
        };
        let values = if x == 0.0 { vec![1.0] } else { vec![1.0, 2.0] };
        Ok(Value::Tensor(
            Tensor::new(values.clone(), vec![1, values.len()]).expect("shape"),
        ))
    }

    #[runtime_builtin(
        name = "__integral_array_nd",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn array_nd_helper(x: Value) -> crate::BuiltinResult<Value> {
        let Value::Num(x) = x else {
            unreachable!("array callback receives scalar")
        };
        Ok(Value::Tensor(
            Tensor::new(vec![x, x * x, x + 1.0, 2.0 * x], vec![2, 1, 2]).expect("fixed N-D shape"),
        ))
    }

    #[runtime_builtin(
        name = "__integral_array_empty",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn array_empty_helper(_x: Value) -> crate::BuiltinResult<Value> {
        Ok(Value::Tensor(
            Tensor::new(vec![], vec![0, 2]).expect("empty array shape"),
        ))
    }

    #[runtime_builtin(
        name = "__integral_nan",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn nan_helper(_x: Value) -> crate::BuiltinResult<Value> {
        mapped_helper(_x, |_| f64::NAN).await
    }

    #[runtime_builtin(
        name = "__integral_integer_tensor",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn integer_tensor_helper(_x: Value) -> crate::BuiltinResult<Value> {
        let (len, shape) = match _x {
            Value::Tensor(samples) => (samples.len(), samples.shape),
            _ => (1, vec![1, 1]),
        };
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![3; len]), shape).expect("tensor");
        Ok(Value::Tensor(tensor))
    }

    fn run(function: Value, a: f64, b: f64) -> crate::BuiltinResult<Value> {
        block_on(integral_builtin(
            function,
            Value::Num(a),
            Value::Num(b),
            Vec::new(),
        ))
    }

    #[test]
    fn integral_test_helper_descriptor_is_attached_shape() {
        assert_eq!(
            INTEGRAL_TEST_HELPER_DESCRIPTOR.signatures[0].label,
            "fx = __integral_helper(x)"
        );
    }

    #[test]
    fn integral_bounds_read_typed_integer_storage_exactly() {
        let bounds = || {
            (
                Tensor::new_integer(IntegerStorage::I16(vec![1]), vec![1, 1]).expect("lower"),
                Tensor::new_integer(IntegerStorage::U16(vec![3]), vec![1, 1]).expect("upper"),
            )
        };
        let (lower, upper) = bounds();
        let error = block_on(integral_builtin(
            Value::FunctionHandle("__integral_square".to_string()),
            Value::Tensor(lower),
            Value::Tensor(upper),
            Vec::new(),
        ))
        .expect_err("compatible mode rejects integer bounds");
        assert_eq!(
            error.identifier(),
            INTEGRAL_INTEGER_BOUND_EXTENSION.error_identifier
        );

        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let (lower, upper) = bounds();
        let result = block_on(integral_builtin(
            Value::FunctionHandle("__integral_square".to_string()),
            Value::Tensor(lower),
            Value::Tensor(upper),
            Vec::new(),
        ))
        .expect("integral");

        match result {
            Value::Num(value) => assert!((value - (26.0 / 3.0)).abs() < 1.0e-6),
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[test]
    fn integral_integrand_reads_typed_integer_scalar_storage_exactly() {
        let result = run(
            Value::FunctionHandle("__integral_integer_tensor".to_string()),
            0.0,
            2.0,
        )
        .expect("integral");

        match result {
            Value::Num(value) => assert!((value - 6.0).abs() < 1.0e-9),
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[test]
    fn integrates_named_sine_function() {
        let result = run(
            Value::FunctionHandle("sin".into()),
            0.0,
            std::f64::consts::PI,
        )
        .expect("integral");
        match result {
            Value::Num(value) => assert!((value - 2.0).abs() < 1.0e-7),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn integrates_polynomial_helper() {
        let result =
            run(Value::FunctionHandle("__integral_square".into()), 0.0, 1.0).expect("integral");
        match result {
            Value::Num(value) => assert!((value - (1.0 / 3.0)).abs() < 1.0e-9),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn vectorized_default_batches_samples_and_scalar_option_honors_current_pin() {
        let default_error = run(
            Value::FunctionHandle("__integral_scalar_only".into()),
            0.0,
            1.0,
        )
        .expect_err("the default callback receives a sample row vector");
        assert!(default_error.message().contains("vector batch"));

        let scalar_form = || {
            block_on(integral_builtin(
                Value::FunctionHandle("__integral_scalar_only".into()),
                Value::Num(0.0),
                Value::Num(1.0),
                vec![Value::from("Vectorized"), Value::Bool(false)],
            ))
        };
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let strict_result = scalar_form().expect("pinned scalar-callback selector is compatible");
        assert!(matches!(strict_result, Value::Num(value) if (value - 1.0/3.0).abs() < 1.0e-8));
        drop(_strict);
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = scalar_form().expect("RunMat scalar callback form");
        assert!(matches!(result, Value::Num(value) if (value - 1.0/3.0).abs() < 1.0e-8));
    }

    #[test]
    fn finite_extreme_intervals_keep_representable_results() {
        let constant = run(Value::FunctionHandle("__integral_one".into()), 0.0, 1.0e308)
            .expect("finite constant integral");
        assert!(
            matches!(constant, Value::Num(value) if (value - 1.0e308).abs() / 1.0e308 < 1.0e-12)
        );

        let symmetric_linear = run(
            Value::FunctionHandle("__integral_scaled_identity".into()),
            -1.0e308,
            1.0e308,
        )
        .expect("opposite-sign interval with cancelling integrand");
        assert!(matches!(symmetric_linear, Value::Num(value) if value == 0.0));

        let cancelling_large_subintervals = run(
            Value::FunctionHandle("__integral_identity".into()),
            -1.0e308,
            1.0e308,
        )
        .expect("nonrepresentable half-integrals cancel to zero");
        assert!(matches!(cancelling_large_subintervals, Value::Num(value) if value == 0.0));

        let nonrepresentable = run(
            Value::FunctionHandle("__integral_one".into()),
            -f64::MAX,
            f64::MAX,
        )
        .expect_err("nonrepresentable integral must fail deliberately");
        assert_eq!(
            nonrepresentable.identifier(),
            INTEGRAL_ERROR_INVALID_INPUT.identifier
        );
        assert!(nonrepresentable.message().contains("not representable"));
    }

    #[test]
    fn reversed_bounds_negate_result() {
        let result = run(
            Value::FunctionHandle("sin".into()),
            std::f64::consts::PI,
            0.0,
        )
        .expect("integral");
        match result {
            Value::Num(value) => assert!((value + 2.0).abs() < 1.0e-7),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn zero_width_interval_returns_zero_without_callback() {
        let result =
            run(Value::FunctionHandle("__integral_nan".into()), 1.0, 1.0).expect("integral");
        assert!(matches!(result, Value::Num(0.0)));
    }

    #[test]
    fn rejects_vector_valued_integrand_for_initial_scope() {
        let err = run(Value::FunctionHandle("__integral_vector".into()), 0.0, 1.0).unwrap_err();
        assert!(err.message().contains("one real value"));
    }

    #[test]
    fn array_valued_callback_preserves_shape_and_component_integrals() {
        let result = block_on(integral_builtin(
            Value::FunctionHandle("__integral_array_polynomial".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::from("ArrayValued"), Value::Bool(true)],
        ))
        .expect("array integral");
        let Value::Tensor(result) = result else {
            panic!("expected array output")
        };
        assert_eq!(result.shape, vec![1, 3]);
        for (index, expected) in [0.5, 1.0 / 3.0, 1.0].into_iter().enumerate() {
            assert!((tensor::tensor_value_f64(&result, index) - expected).abs() < 1.0e-7);
        }
    }

    #[test]
    fn array_valued_equal_bounds_retains_output_geometry() {
        let result = block_on(integral_builtin(
            Value::FunctionHandle("__integral_array_polynomial".into()),
            Value::Num(2.0),
            Value::Num(2.0),
            vec![Value::from("ArrayValued"), Value::Bool(true)],
        ))
        .expect("zero-width array integral");
        let Value::Tensor(result) = result else {
            panic!("expected array output")
        };
        assert_eq!(result.shape, vec![1, 3]);
        assert!((0..3).all(|index| tensor::tensor_value_f64(&result, index) == 0.0));
    }

    #[test]
    fn array_valued_callback_rejects_shape_changes() {
        let error = block_on(integral_builtin(
            Value::FunctionHandle("__integral_array_changing_shape".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::from("ArrayValued"), Value::Bool(true)],
        ))
        .expect_err("fixed callback shape");
        assert_eq!(error.identifier(), INTEGRAL_ERROR_INVALID_INPUT.identifier);
        assert!(error.message().contains("changed its output size"));
    }

    #[test]
    fn array_valued_nd_and_empty_outputs_keep_callback_geometry() {
        for (helper, shape, expected) in [
            (
                "__integral_array_nd",
                vec![2, 1, 2],
                vec![0.5, 1.0 / 3.0, 1.5, 1.0],
            ),
            ("__integral_array_empty", vec![0, 2], vec![]),
        ] {
            let result = block_on(integral_builtin(
                Value::FunctionHandle(helper.into()),
                Value::Num(0.0),
                Value::Num(1.0),
                vec![Value::from("ArrayValued"), Value::Bool(true)],
            ))
            .expect("array integral");
            let Value::Tensor(result) = result else {
                panic!("expected array output")
            };
            assert_eq!(result.shape, shape);
            for (index, expected) in expected.into_iter().enumerate() {
                assert!((tensor::tensor_value_f64(&result, index) - expected).abs() < 1.0e-7);
            }
        }
    }

    #[test]
    fn array_valued_overrides_vectorized_in_either_option_order() {
        for rest in [
            vec![
                Value::from("ArrayValued"),
                Value::Bool(true),
                Value::from("Vectorized"),
                Value::Bool(true),
            ],
            vec![
                Value::from("Vectorized"),
                Value::Bool(false),
                Value::from("ArrayValued"),
                Value::Bool(true),
            ],
        ] {
            let result = block_on(integral_builtin(
                Value::FunctionHandle("__integral_array_polynomial".into()),
                Value::Num(0.0),
                Value::Num(1.0),
                rest,
            ))
            .expect("array form ignores Vectorized");
            let Value::Tensor(result) = result else {
                panic!("expected array output")
            };
            assert_eq!(result.shape, vec![1, 3]);
        }
    }

    #[test]
    fn scalar_array_flag_values_match_typed_logical_and_numeric_admission() {
        for flag in [
            Value::LogicalArray(LogicalArray::new(vec![1], vec![1, 1]).expect("logical scalar")),
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).expect("numeric scalar")),
        ] {
            let result = block_on(integral_builtin(
                Value::FunctionHandle("__integral_array_polynomial".into()),
                Value::Num(0.0),
                Value::Num(1.0),
                vec![Value::from("ArrayValued"), flag],
            ))
            .expect("array-valued flag");
            assert!(matches!(result, Value::Tensor(array) if array.shape == [1, 3]));
        }
    }

    #[test]
    fn rejects_nonfinite_integrand_values() {
        let err = run(Value::FunctionHandle("__integral_nan".into()), 0.0, 1.0).unwrap_err();
        assert!(err.message().contains("finite real numbers"));
    }

    #[test]
    fn accepts_tolerance_name_value_options() {
        let result = block_on(integral_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(std::f64::consts::PI),
            vec![
                Value::from("AbsTol"),
                Value::Num(1.0e-12),
                Value::from("RelTol"),
                Value::Num(1.0e-8),
            ],
        ))
        .expect("integral");
        match result {
            Value::Num(value) => assert!((value - 2.0).abs() < 1.0e-8),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn rejects_too_small_max_fun_evals() {
        let err = block_on(integral_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::from("MaxFunEvals"), Value::Num(4.0)],
        ))
        .unwrap_err();
        assert!(err.message().contains("integer scalar >= 5"));
    }

    #[test]
    fn rejects_fractional_max_fun_evals() {
        let err = block_on(integral_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::from("MaxFunEvals"), Value::Num(5.5)],
        ))
        .unwrap_err();
        assert!(err.message().contains("integer scalar"));
    }

    #[test]
    fn max_fun_evals_option_reads_typed_integer_storage_exactly() {
        let option = || {
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U16(vec![50]), vec![1, 1])
                    .expect("MaxFunEvals"),
            )
        };

        let error = block_on(integral_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::from("MaxFunEvals"), option()],
        ))
        .expect_err("compatible mode rejects typed integer options");
        assert_eq!(
            error.identifier(),
            INTEGRAL_INTEGER_OPTION_EXTENSION.error_identifier
        );
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = block_on(integral_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::from("MaxFunEvals"), option()],
        ))
        .expect("RunMat integer option");
        assert!(matches!(result, Value::Num(_)));
    }

    #[test]
    fn resident_integer_tolerance_option_gathers_after_compatibility_admission() {
        use crate::builtins::common::test_support;
        use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let value = [1_u64];
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&value),
                    shape: &[1, 1],
                })
                .expect("upload resident tolerance");
            let result = block_on(integral_builtin(
                Value::FunctionHandle("sin".into()),
                Value::Num(0.0),
                Value::Num(1.0),
                vec![Value::from("AbsTol"), Value::GpuTensor(handle.clone())],
            ))
            .expect("resident integer tolerance");
            assert!(matches!(result, Value::Num(_)));
            handle
                .allocation_lease()
                .validate_live()
                .expect("borrowed tolerance remains live");
            let preserved = test_support::gather(Value::GpuTensor(handle.clone())).unwrap();
            assert_eq!(
                preserved.integer_storage(),
                Some(&IntegerStorage::U64(vec![1]))
            );
            let physical = handle.allocation_lease().downgrade();
            handle.release().expect("release resident tolerance");
            assert!(physical.acquire().is_none());
        });
    }

    #[test]
    fn resident_options_are_authenticated_before_the_first_bound_download() {
        use runmat_accelerate_api::{
            test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
            AccelProvider, GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned,
            HostNumericTensorOwned, HostTensorView, ThreadProviderGuard,
        };
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct RejectDownloadProvider {
            owner: TestAllocationOwner,
            downloads: AtomicUsize,
        }

        impl AccelProvider for RejectDownloadProvider {
            fn device_id(&self) -> u32 {
                self.owner.device_id()
            }

            fn device_info(&self) -> String {
                "integral admission fixture".into()
            }

            fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
                panic!("admission must not upload")
            }

            fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
                self.downloads.fetch_add(1, Ordering::SeqCst);
                Box::pin(async { Err(std::io::Error::other("unexpected download").into()) })
            }

            fn download_numeric<'a>(
                &'a self,
                _: &'a GpuTensorHandle,
            ) -> AccelNumericDownloadFuture<'a> {
                self.downloads.fetch_add(1, Ordering::SeqCst);
                Box::pin(async { Err(std::io::Error::other("unexpected download").into()) })
            }
        }

        let _serial = crate::builtins::common::test_support::accel_test_lock();
        let provider = Box::leak(Box::new(RejectDownloadProvider {
            owner: TestAllocationOwner::new(),
            downloads: AtomicUsize::new(0),
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let allocated = |number| {
            provider
                .owner
                .allocate_numeric(HostNumericTensorOwned {
                    data: HostNumericDataOwned::F64(vec![number]),
                    shape: vec![1, 1],
                    storage: GpuTensorStorage::Real,
                })
                .expect("allocate fixture tensor")
        };
        let bound = allocated(0.0);
        let valid_option = allocated(1.0e-8);
        let stale_option = allocated(1.0e-7);
        stale_option.probe.mark_unavailable();
        let error = block_on(integral_builtin(
            Value::FunctionHandle("sin".into()),
            Value::GpuTensor(bound.handle.clone()),
            Value::Num(1.0),
            vec![
                Value::from("AbsTol"),
                Value::GpuTensor(valid_option.handle.clone()),
                Value::from("RelTol"),
                Value::GpuTensor(stale_option.handle.clone()),
            ],
        ))
        .expect_err("later stale option must fail before an earlier resident download");
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        assert!(error.source.is_some());
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert_eq!(bound.probe.retirements(), 0);
        assert_eq!(valid_option.probe.retirements(), 0);
        bound.handle.allocation_lease().validate_live().unwrap();
        valid_option
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();

        let zero_tolerances = block_on(integral_builtin(
            Value::FunctionHandle("sin".into()),
            Value::GpuTensor(bound.handle.clone()),
            Value::Num(1.0),
            vec![
                Value::from("AbsTol"),
                Value::GpuTensor(valid_option.handle.clone()),
                Value::from("AbsTol"),
                Value::Num(0.0),
                Value::from("RelTol"),
                Value::Num(0.0),
            ],
        ))
        .expect_err("known effective zero tolerances fail before resident access");
        assert_eq!(
            zero_tolerances.identifier(),
            INTEGRAL_ERROR_INVALID_ARGUMENT.identifier
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert_eq!(bound.probe.retirements(), 0);
        assert_eq!(valid_option.probe.retirements(), 0);
    }

    #[test]
    fn max_fun_evals_option_rejects_negative_typed_integer_storage_exactly() {
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let max_fun_evals =
            Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("MaxFunEvals");

        let err = block_on(integral_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::from("MaxFunEvals"), Value::Tensor(max_fun_evals)],
        ))
        .unwrap_err();
        assert!(err.message().contains("MaxFunEvals"));
    }

    #[test]
    fn max_fun_evals_option_rejects_unrepresentable_double_boundary() {
        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        let err = block_on(integral_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::from("MaxFunEvals"), Value::Num(boundary)],
        ))
        .unwrap_err();
        assert!(err.message().contains("MaxFunEvals"));
    }

    #[test]
    fn integral_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = INTEGRAL_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(
            labels,
            vec![
                "q = integral(fun, xmin, xmax)",
                "q = integral(fun, xmin, xmax, options)",
                "q = integral(fun, xmin, xmax, name, value, ...)",
            ]
        );

        let codes: Vec<&str> = INTEGRAL_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert_eq!(
            codes,
            vec![
                "RM.INTEGRAL.INVALID_ARGUMENT",
                "RM.INTEGRAL.INVALID_INPUT",
                "RM.INTEGRAL.TOO_MANY_OUTPUTS",
            ]
        );
    }

    #[test]
    fn integral_bad_name_value_pairs_use_stable_identifier() {
        let err = block_on(integral_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::from("AbsTol")],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:integral:InvalidArgument"));
    }

    #[test]
    fn invalid_host_bound_is_rejected_during_admission() {
        let error = block_on(integral_builtin(
            Value::FunctionHandle("sin".into()),
            Value::from("invalid lower bound"),
            Value::Num(1.0),
            vec![Value::from("AbsTol"), Value::Num(1.0e-8)],
        ))
        .expect_err("invalid bound");
        assert_eq!(error.identifier(), INTEGRAL_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn invalid_callback_is_rejected_before_numerical_execution() {
        let error = block_on(integral_builtin(
            Value::Num(7.0),
            Value::Num(0.0),
            Value::Num(1.0),
            Vec::new(),
        ))
        .expect_err("invalid callback");
        assert_eq!(
            error.identifier(),
            runmat_builtins::FEVAL_ERROR_FUNCTION_VALUE_UNSUPPORTED.identifier
        );
    }

    #[test]
    fn registered_invocation_observes_preexisting_cancellation_before_callback() {
        use runmat_execution::cancellation::CancellationSource;
        use runmat_execution::CancellationReason;
        use std::rc::Rc;

        let source = CancellationSource::new();
        source.cancel(CancellationReason::User);
        let runtime = crate::context::RuntimeContext::with_cancellation(
            Rc::new(crate::execution::RuntimeExecutionService::new()),
            source.token(),
        );
        let error = block_on(integral_builtin_registered(
            Some(runtime),
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            Vec::new(),
        ))
        .expect_err("cancelled invocation");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        assert!(error.source.is_some());
    }
}
