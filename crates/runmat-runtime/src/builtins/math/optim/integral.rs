//! `integral` numerical integration over real finite and improper domains.

#[cfg(test)]
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, INTEGRAL_DESCRIPTOR,
};
use runmat_builtins::{
    BuiltinErrorDescriptor, IntegralOptionName, INTEGRAL_ERROR_INVALID_ARGUMENT,
    INTEGRAL_ERROR_INVALID_INPUT, INTEGRAL_ERROR_TOO_MANY_OUTPUTS,
    INTEGRAL_INTEGER_BOUND_EXTENSION, INTEGRAL_INTEGER_OPTION_EXTENSION,
    INTEGRAL_LOGICAL_BOUND_EXTENSION, INTEGRAL_LOGICAL_NUMERIC_OPTION_EXTENSION,
    INTEGRAL_RENAMED_TOLERANCE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{LogicalArray, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::optim::quadrature::{
    QuadratureArithmeticError, QuadratureCooperation, QuadratureIdentity,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod contour;
mod options;
mod path;
mod waypoints;

use options::IntegralOptions;

const NAME: &str = "integral";
const DEFAULT_MAX_FUN_EVALS: usize = 10_000;
// The open rule is iterative. Let representable interior geometry and the
// independent callback/interval budgets, rather than an arbitrary recursion
// depth, determine when further refinement is impossible.
const MAX_DEPTH: usize = usize::MAX;

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
        mapped.span = err.span;
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
        QuadratureArithmeticError::Execution(source) => *source,
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
    rest: crate::call::BuiltinArguments,
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
    integral_in_runtime(
        None,
        function,
        a,
        b,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
}

async fn integral_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    function: Value,
    a: Value,
    b: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let mut cooperation = QuadratureCooperation::new(QuadratureIdentity::Integral, runtime);
    integral_with_cooperation(function, a, b, rest, &mut cooperation).await
}

async fn integral_with_cooperation(
    function: Value,
    a: Value,
    b: Value,
    rest: crate::call::BuiltinArguments,
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<Value> {
    cooperation.completed(0).await?;
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(integral_error_with_detail(
            &INTEGRAL_ERROR_TOO_MANY_OUTPUTS,
            "integral: too many output arguments; maximum is 1",
        ));
    }
    preflight_integral_bound_extensions(&a, &b)?;
    let admitted_options = admit_option_headers(rest.view())?;
    preflight_integral_option_extensions(&admitted_options)?;
    crate::call::admission::admit_feval_target(&function)?;
    for (label, value) in [("lower bound", &a), ("upper bound", &b)] {
        preflight_integral_bound(label, value)?;
    }
    let waypoint_receipts = IntegralOptions::preflight_host(&admitted_options, cooperation).await?;
    let complex_path = contour::has_complex_class(&a)
        || contour::has_complex_class(&b)
        || admitted_options
            .iter()
            .rev()
            .find(|(name, _)| *name == IntegralOptionName::Waypoints)
            .is_some_and(|(_, value)| contour::has_complex_class(value));
    let prepared_path = if complex_path {
        None
    } else {
        path::preflight_host_path(&a, &b, &admitted_options, &waypoint_receipts, cooperation)
            .await?
    };
    authenticate_integral_residents(&a, &b, &admitted_options)?;
    let options = IntegralOptions::parse(&admitted_options, waypoint_receipts, cooperation)
        .await
        .map_err(|err| integral_map_error(err, &INTEGRAL_ERROR_INVALID_ARGUMENT))?;
    if complex_path {
        let lower = contour::scalar_bound("lower bound", &a).await?;
        let upper = contour::scalar_bound("upper bound", &b).await?;
        let segments =
            contour::ordered_segments(lower, upper, &options.waypoints, cooperation).await?;
        let result = contour::integrate_contour(
            &function,
            &options,
            &segments,
            1.0,
            false,
            lower,
            cooperation,
        )
        .await?;
        cooperation.completed(0).await?;
        return Ok(result);
    }
    let a = scalar_bound("lower bound", a)
        .await
        .map_err(|err| integral_map_error(err, &INTEGRAL_ERROR_INVALID_INPUT))?;
    let b = scalar_bound("upper bound", b)
        .await
        .map_err(|err| integral_map_error(err, &INTEGRAL_ERROR_INVALID_INPUT))?;
    let segments = match prepared_path {
        Some(segments) => segments,
        None => path::real_segments(a, b, &options.waypoints, cooperation).await?,
    };
    if options
        .max_interval_count
        .is_some_and(|limit| segments.len() > limit.get())
    {
        return Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            "exceeded MaxIntervalCount adaptive subintervals",
        ));
    }
    // The numerical rule owns its minimum callback count and checks it before
    // any callback effect. Do not mirror a fixed batch-size policy here.
    let sign = if b < a { -1.0 } else { 1.0 };
    let result = contour::integrate_contour(
        &function,
        &options,
        &segments,
        sign,
        true,
        num_complex::Complex::new(a, 0.0),
        cooperation,
    )
    .await?;
    cooperation.completed(0).await?;
    Ok(result)
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
        if !scalar
            || !matches!(
                handle_storage(handle),
                GpuTensorStorage::Real | GpuTensorStorage::ComplexInterleaved
            )
        {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_INPUT,
                format!("{label} must be a real scalar"),
            ));
        }
        return Ok(());
    }
    if contour::has_complex_class(value) {
        contour::scalar_bound_host(label, value).map(|_| ())
    } else {
        scalar_bound_host(label, value).map(|_| ())
    }
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

fn admit_option_headers(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Vec<(IntegralOptionName, &Value)>> {
    if let Some(Value::Struct(fields)) = rest.first().filter(|_| rest.len() == 1) {
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
    for index in (0..rest.len()).step_by(2) {
        let name = rest.get(index).expect("bounded integral option name");
        let value = rest.get(index + 1).expect("bounded integral option value");
        admitted.push((option_name(name)?, value));
    }
    Ok(admitted)
}

fn option_name(value: &Value) -> BuiltinResult<IntegralOptionName> {
    let text = match value {
        Value::String(runmat_value::StringElement::Text(s)) => s.as_str(),
        Value::StringArray(sa) if sa.data().len() == 1 => {
            sa.data()[0].as_str().ok_or_else(|| {
                integral_error_with_detail(
                    &INTEGRAL_ERROR_INVALID_ARGUMENT,
                    "option names must be nonmissing text",
                )
            })?
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            return decode_option_name(
                &runmat_value::decode_utf16_code_units(chars.data()).map_err(|source| {
                    let mut error = integral_error_with_detail(
                        &INTEGRAL_ERROR_INVALID_ARGUMENT,
                        source.to_string(),
                    );
                    error.source = Some(Box::new(source));
                    error
                })?,
            );
        }
        other => Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("option names must be strings, got {other:?}"),
        ))?,
    };
    decode_option_name(text)
}

fn decode_option_name(text: &str) -> BuiltinResult<IntegralOptionName> {
    let identity = IntegralOptionName::parse(text).ok_or_else(|| {
        integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("unsupported option {text}"),
        )
    })?;
    if identity.is_versioned_tolerance_alias() {
        crate::compatibility::ensure_builtin_feature_enabled(
            &INTEGRAL_RENAMED_TOLERANCE_EXTENSION,
            NAME,
        )?;
    }
    Ok(identity)
}

async fn scalar_bound(label: &str, value: Value) -> BuiltinResult<f64> {
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    scalar_bound_host(label, &value)
}

fn scalar_bound_host(label: &str, value: &Value) -> BuiltinResult<f64> {
    if !crate::builtins::common::validation::native_integer_value_is_exact_f64(value) {
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
                    format!("{label} must be a real scalar"),
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
                format!("{label} must be a real scalar, got {other:?}"),
            ));
        }
    };
    if !parsed.is_nan() {
        Ok(parsed)
    } else {
        Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            format!("{label} must not be NaN"),
        ))
    }
}

fn scalar_tensor_value(tensor: &Tensor) -> Option<f64> {
    tensor::is_scalar_tensor(tensor).then(|| tensor::tensor_value_f64(tensor, 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::IntegerStorage;

    struct PendingIntegralTurn(std::rc::Rc<std::cell::Cell<bool>>);

    impl std::future::Future for PendingIntegralTurn {
        type Output = Result<(), crate::execution::ExecutionServiceError>;

        fn poll(
            self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Self::Output> {
            std::task::Poll::Pending
        }
    }

    impl Drop for PendingIntegralTurn {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }

    async fn mapped_helper(x: Value, map: impl Fn(f64) -> f64) -> crate::BuiltinResult<Value> {
        match x {
            Value::Tensor(samples) => {
                let values = (0..samples.len())
                    .map(|index| map(tensor::tensor_value_f64(&samples, index)))
                    .collect();
                let shape = samples.shape().to_vec();
                Ok(Value::Tensor(
                    Tensor::new(values, shape).expect("fixed test sample shape"),
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

    thread_local! {
        static COUNTED_CALLBACKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    #[runtime_builtin(
        name = "__integral_counted_one",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn counted_one_helper(x: Value) -> crate::BuiltinResult<Value> {
        COUNTED_CALLBACKS.with(|count| count.set(count.get() + 1));
        mapped_helper(x, |_| 1.0).await
    }

    #[runtime_builtin(
        name = "__integral_complex_one",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn complex_one_helper(x: Value) -> crate::BuiltinResult<Value> {
        COUNTED_CALLBACKS.with(|count| count.set(count.get() + 1));
        let (len, shape) = match x {
            Value::Tensor(samples) => {
                let len = samples.len();
                let shape = samples.shape().to_vec();
                (len, shape)
            }
            Value::ComplexTensor(samples) => {
                let len = samples.len();
                let (_, shape) = samples.into_complex_storage_and_shape();
                (len, shape.to_vec())
            }
            _ => return Ok(Value::Complex(1.0, 1.0)),
        };
        let values = vec![(1.0, 1.0); len];
        let tensor = runmat_value::ComplexTensor::from_complex_storage(
            runmat_value::ComplexStorage::F64(values.into()),
            shape,
        )
        .expect("fixed callback geometry");
        Ok(Value::ComplexTensor(tensor))
    }

    #[runtime_builtin(
        name = "__integral_complex_array",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn complex_array_helper(_x: Value) -> crate::BuiltinResult<Value> {
        let tensor = runmat_value::ComplexTensor::from_complex_storage(
            runmat_value::ComplexStorage::F64(vec![(1.0, 1.0), (2.0, -1.0)].into()),
            vec![1, 2],
        )
        .expect("fixed callback geometry");
        Ok(Value::ComplexTensor(tensor))
    }

    #[runtime_builtin(
        name = "__integral_single_array",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn single_array_helper(_x: Value) -> crate::BuiltinResult<Value> {
        Ok(Value::Tensor(
            Tensor::from_numeric_storage(
                runmat_value::NumericStorage::F32(vec![1.0]),
                vec![1, 1, 1],
            )
            .expect("fixed single callback geometry"),
        ))
    }

    #[runtime_builtin(
        name = "__integral_single_complex_array",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn single_complex_array_helper(_x: Value) -> crate::BuiltinResult<Value> {
        Ok(Value::ComplexTensor(
            runmat_value::ComplexTensor::from_complex_storage(
                runmat_value::ComplexStorage::F32(vec![(1.0_f32, 1.0_f32)].into()),
                vec![1, 1, 1],
            )
            .expect("fixed single complex callback geometry"),
        ))
    }

    #[runtime_builtin(
        name = "__integral_complex_exp_tail",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn complex_exp_tail_helper(x: Value) -> crate::BuiltinResult<Value> {
        COUNTED_CALLBACKS.with(|count| count.set(count.get() + 1));
        match x {
            Value::Tensor(samples) => {
                let values = (0..samples.len())
                    .map(|index| {
                        let tail = (-tensor::tensor_value_f64(&samples, index)).exp();
                        (tail, tail)
                    })
                    .collect::<Vec<_>>();
                let shape = samples.shape().to_vec();
                Ok(Value::ComplexTensor(
                    runmat_value::ComplexTensor::from_complex_storage(
                        runmat_value::ComplexStorage::F64(values.into()),
                        shape,
                    )
                    .expect("callback batch geometry"),
                ))
            }
            other => {
                let tail = (-scalar_bound("x", other).await?).exp();
                Ok(Value::Complex(tail, tail))
            }
        }
    }

    #[runtime_builtin(
        name = "__integral_endpoint_sqrt",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn endpoint_sqrt_helper(x: Value) -> crate::BuiltinResult<Value> {
        mapped_helper(x, |value| 1.0 / value.sqrt()).await
    }

    #[runtime_builtin(
        name = "__integral_exp_tail",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn exp_tail_helper(x: Value) -> crate::BuiltinResult<Value> {
        mapped_helper(x, |value| (-value).exp()).await
    }

    #[runtime_builtin(
        name = "__integral_gaussian",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn gaussian_helper(x: Value) -> crate::BuiltinResult<Value> {
        mapped_helper(x, |value| (-value * value).exp()).await
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
        name = "__integral_array_sine",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn array_sine_helper(x: Value) -> crate::BuiltinResult<Value> {
        let Value::Num(x) = x else {
            unreachable!("array callback receives scalar")
        };
        Ok(Value::Tensor(
            Tensor::new(vec![x.sin(), x.cos()], vec![1, 2]).expect("fixed shape"),
        ))
    }

    #[runtime_builtin(
        name = "__integral_array_exp_tails",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn array_exp_tails_helper(x: Value) -> crate::BuiltinResult<Value> {
        let Value::Num(x) = x else {
            unreachable!("array callback receives scalar")
        };
        Ok(Value::Tensor(
            Tensor::new(vec![(-x).exp(), (-2.0 * x).exp()], vec![1, 2]).expect("fixed shape"),
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
        let values = if x < 0.5 { vec![1.0] } else { vec![1.0, 2.0] };
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
        if x < 0.0 {
            return Ok(Value::Tensor(
                Tensor::new(vec![x], vec![1, 1, 1]).expect("singleton N-D shape"),
            ));
        }
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
        name = "__integral_array_opposite_constants",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        descriptor(crate::builtins::math::optim::integral::tests::INTEGRAL_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn array_opposite_constants_helper(_x: Value) -> crate::BuiltinResult<Value> {
        Ok(Value::Tensor(
            Tensor::new(vec![1.0, -1.0], vec![1, 2]).expect("fixed shape"),
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
            Value::Tensor(samples) => {
                let len = samples.len();
                let shape = samples.shape().to_vec();
                (len, shape)
            }
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
            Value::FunctionHandle("__integral_square".into()),
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
            Value::FunctionHandle("__integral_square".into()),
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
            Value::FunctionHandle("__integral_integer_tensor".into()),
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
    fn finite_real_bounds_integrate_complex_callback_once_per_vector_batch() {
        COUNTED_CALLBACKS.with(|count| count.set(0));
        let result = run(
            Value::FunctionHandle("__integral_complex_one".into()),
            0.0,
            1.0,
        )
        .expect("complex callback on real interval");
        assert!(
            matches!(result, Value::Complex(real, imaginary) if (real - 1.0).abs() < 1.0e-10 && (imaginary - 1.0).abs() < 1.0e-10)
        );
        assert_eq!(COUNTED_CALLBACKS.with(|count| count.get()), 1);
    }

    #[test]
    fn ordered_complex_contour_and_array_values_share_one_mesh() {
        let result = block_on(integral_builtin(
            Value::FunctionHandle("__integral_complex_one".into()),
            Value::Complex(0.0, 0.0),
            Value::Complex(0.0, 1.0),
            vec![Value::from("Waypoints"), Value::Complex(1.0, 1.0)],
        ))
        .expect("ordered complex contour");
        assert!(
            matches!(result, Value::Complex(real, imaginary) if (real + 1.0).abs() < 1.0e-9 && (imaginary - 1.0).abs() < 1.0e-9)
        );

        let array = block_on(integral_builtin(
            Value::FunctionHandle("__integral_complex_array".into()),
            Value::Complex(0.0, 0.0),
            Value::Complex(0.0, 1.0),
            vec![Value::from("ArrayValued"), Value::Bool(true)],
        ))
        .expect("complex array contour");
        let Value::ComplexTensor(array) = array else {
            panic!("expected complex tensor")
        };
        assert_eq!(array.shape(), vec![1, 2]);
        assert!(
            matches!(array.numeric_value_at(0), Some((runmat_value::NumericScalar::F64(real), runmat_value::NumericScalar::F64(imaginary))) if (real + 1.0).abs() < 1.0e-9 && (imaginary - 1.0).abs() < 1.0e-9)
        );
        assert!(
            matches!(array.numeric_value_at(1), Some((runmat_value::NumericScalar::F64(real), runmat_value::NumericScalar::F64(imaginary))) if (real - 1.0).abs() < 1.0e-9 && (imaginary - 2.0).abs() < 1.0e-9)
        );
    }

    #[test]
    fn single_array_callbacks_publish_double_with_original_singleton_nd_shape() {
        for (helper, complex) in [
            ("__integral_single_array", false),
            ("__integral_single_complex_array", true),
        ] {
            for (lower, upper, expected) in [(0.0, 1.0, 1.0), (2.0, 2.0, 0.0)] {
                let result = block_on(integral_builtin(
                    Value::FunctionHandle(helper.into()),
                    Value::Num(lower),
                    Value::Num(upper),
                    vec![Value::from("ArrayValued"), Value::Bool(true)],
                ))
                .expect("single callback array integral");
                if complex {
                    let Value::ComplexTensor(array) = result else {
                        panic!("expected complex double array")
                    };
                    assert_eq!(array.shape(), vec![1, 1, 1]);
                    assert!(matches!(
                        array.complex_storage(),
                        runmat_value::ComplexStorage::F64(_)
                    ));
                    assert!(
                        matches!(array.numeric_value_at(0), Some((runmat_value::NumericScalar::F64(real), runmat_value::NumericScalar::F64(imaginary))) if (real-expected).abs()<1.0e-9 && (imaginary-expected).abs()<1.0e-9)
                    );
                } else {
                    let Value::Tensor(array) = result else {
                        panic!("expected real double array")
                    };
                    assert_eq!(array.shape(), vec![1, 1, 1]);
                    assert_eq!(array.numeric_dtype(), runmat_value::NumericDType::F64);
                    assert!((tensor::tensor_value_f64(&array, 0) - expected).abs() < 1.0e-9);
                }
            }
        }
    }

    #[test]
    fn improper_real_bounds_admit_complex_callbacks_on_one_rule_path() {
        COUNTED_CALLBACKS.with(|count| count.set(0));
        let result = block_on(integral_builtin(
            Value::FunctionHandle("__integral_complex_exp_tail".into()),
            Value::Num(0.0),
            Value::Num(f64::INFINITY),
            Vec::new(),
        ))
        .expect("complex callback on positive infinite real tail");
        assert!(
            matches!(result, Value::Complex(real, imaginary) if (real-1.0).abs()<1.0e-7 && (imaginary-1.0).abs()<1.0e-7)
        );
        assert!(COUNTED_CALLBACKS.with(|count| count.get()) > 0);
    }

    #[test]
    fn nonfinite_complex_contour_rejects_before_callback_and_real_infinity_remains_legal() {
        COUNTED_CALLBACKS.with(|count| count.set(0));
        for (lower, upper) in [
            (Value::Complex(f64::INFINITY, 0.0), Value::Complex(1.0, 1.0)),
            (Value::Complex(0.0, 0.0), Value::Complex(1.0, f64::NAN)),
        ] {
            let error = block_on(integral_builtin(
                Value::FunctionHandle("__integral_complex_one".into()),
                lower,
                upper,
                Vec::new(),
            ))
            .expect_err("nonfinite complex contour must reject");
            assert_eq!(error.identifier(), INTEGRAL_ERROR_INVALID_INPUT.identifier);
            assert_eq!(COUNTED_CALLBACKS.with(|count| count.get()), 0);
        }
        let legal = block_on(integral_builtin(
            Value::FunctionHandle("__integral_exp_tail".into()),
            Value::Num(0.0),
            Value::Num(f64::INFINITY),
            Vec::new(),
        ))
        .expect("real +Inf bound remains legal");
        assert!(matches!(legal, Value::Num(value) if (value-1.0).abs()<1.0e-7));
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
        .expect_err("representable cancellation does not certify the requested tolerance");
        assert_eq!(
            symmetric_linear.identifier(),
            INTEGRAL_ERROR_INVALID_INPUT.identifier
        );
        assert!(
            symmetric_linear.message().contains("function evaluations")
                || symmetric_linear.message().contains("did not converge")
        );

        let cancelling_large_subintervals = run(
            Value::FunctionHandle("__integral_identity".into()),
            -1.0e308,
            1.0e308,
        )
        .expect_err("nonrepresentable half-integrals cannot certify absolute tolerance");
        assert_eq!(
            cancelling_large_subintervals.identifier(),
            INTEGRAL_ERROR_INVALID_INPUT.identifier
        );
        assert!(
            cancelling_large_subintervals
                .message()
                .contains("function evaluations")
                || cancelling_large_subintervals
                    .message()
                    .contains("did not converge")
        );

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
        assert!(err.message().contains("one value"));
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
        assert_eq!(result.shape(), vec![1, 3]);
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
        assert_eq!(result.shape(), vec![1, 3]);
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
            assert_eq!(result.shape(), shape);
            for (index, expected) in expected.into_iter().enumerate() {
                assert!((tensor::tensor_value_f64(&result, index) - expected).abs() < 1.0e-7);
            }
        }
    }

    #[test]
    fn unordered_real_waypoints_share_one_scalar_and_array_integration_path() {
        let waypoints = Tensor::new(vec![0.75, 0.25, 0.25], vec![1, 3]).expect("waypoints");
        let scalar = block_on(integral_builtin(
            Value::FunctionHandle("__integral_square".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::from("Waypoints"), Value::Tensor(waypoints.clone())],
        ))
        .expect("scalar waypoint integral");
        let Value::Num(scalar) = scalar else {
            panic!("scalar integral must return a number")
        };
        assert!((scalar - 1.0 / 3.0).abs() < 1.0e-7);

        let array = block_on(integral_builtin(
            Value::FunctionHandle("__integral_array_polynomial".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![
                Value::from("ArrayValued"),
                Value::Bool(true),
                Value::from("Waypoints"),
                Value::Tensor(waypoints),
            ],
        ))
        .expect("array waypoint integral");
        let Value::Tensor(array) = array else {
            panic!("array integral must return a tensor")
        };
        assert_eq!(array.shape(), vec![1, 3]);
        for (index, expected) in [0.5, 1.0 / 3.0, 1.0].into_iter().enumerate() {
            assert!((tensor::tensor_value_f64(&array, index) - expected).abs() < 1.0e-7);
        }
    }

    #[test]
    fn array_valued_singleton_nd_output_keeps_callback_geometry() {
        for (lo, hi, expected) in [(-2.0, -1.0, -1.5), (-2.0, -2.0, 0.0)] {
            let result = block_on(integral_builtin(
                Value::FunctionHandle("__integral_array_nd".into()),
                Value::Num(lo),
                Value::Num(hi),
                vec![Value::from("ArrayValued"), Value::Bool(true)],
            ))
            .expect("singleton N-D array integral");
            let Value::Tensor(result) = result else {
                panic!("singleton N-D callback must retain its tensor result")
            };
            assert_eq!(result.shape(), vec![1, 1, 1]);
            assert!((tensor::tensor_value_f64(&result, 0) - expected).abs() < 1.0e-7);
        }
    }

    #[test]
    fn array_valued_scaled_estimates_preserve_large_finite_components() {
        let result = block_on(integral_builtin(
            Value::FunctionHandle("__integral_array_opposite_constants".into()),
            Value::Num(0.0),
            Value::Num(1.0e308),
            vec![Value::from("ArrayValued"), Value::Bool(true)],
        ))
        .expect("large finite componentwise integral");
        let Value::Tensor(result) = result else {
            panic!("expected array output")
        };
        assert_eq!(result.shape(), vec![1, 2]);
        assert!((tensor::tensor_value_f64(&result, 0) / 1.0e308 - 1.0).abs() < 1.0e-12);
        assert!((tensor::tensor_value_f64(&result, 1) / 1.0e308 + 1.0).abs() < 1.0e-12);
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
            assert_eq!(result.shape(), vec![1, 3]);
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
            assert!(matches!(result, Value::Tensor(array) if array.shape() == [1, 3]));
        }
    }

    #[test]
    fn rejects_nonfinite_integrand_values() {
        let err = run(Value::FunctionHandle("__integral_nan".into()), 0.0, 1.0).unwrap_err();
        assert!(err.message().contains("finite values"));
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
    fn max_interval_count_is_distinct_from_callback_evaluation_limit() {
        let zero = block_on(integral_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::from("MaxIntervalCount"), Value::Num(0.0)],
        ))
        .expect_err("zero is not a positive interval count");
        assert!(zero
            .message()
            .contains("MaxIntervalCount must be a positive"));

        let split_mesh = block_on(integral_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![
                Value::from("Waypoints"),
                Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).expect("waypoint")),
                Value::from("MaxIntervalCount"),
                Value::Num(1.0),
                Value::from("MaxFunEvals"),
                Value::Num(100.0),
            ],
        ))
        .expect_err("two initial intervals exceed a one-interval cap");
        assert!(split_mesh
            .message()
            .contains("MaxIntervalCount adaptive subintervals"));

        let zero_width = block_on(integral_builtin(
            Value::FunctionHandle("__integral_nan".into()),
            Value::Num(1.0),
            Value::Num(1.0),
            vec![Value::from("MaxIntervalCount"), Value::Num(1.0)],
        ))
        .expect("equal bounds do not invoke the scalar callback");
        assert!(matches!(zero_width, Value::Num(value) if value == 0.0));
    }

    #[test]
    fn scalar_and_array_refinement_obey_interval_cap_independently_of_evaluation_cap() {
        let scalar_polynomial = block_on(integral_builtin(
            Value::FunctionHandle("__integral_scalar_only".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![
                Value::from("Vectorized"),
                Value::Bool(false),
                Value::from("MaxFunEvals"),
                Value::Num(15.0),
                Value::from("MaxIntervalCount"),
                Value::Num(1.0),
            ],
        ))
        .expect("a polynomial accepts within one open-rule subinterval and fifteen evaluations");
        assert!(
            matches!(scalar_polynomial, Value::Num(value) if (value - 1.0 / 3.0).abs() < 1.0e-12)
        );

        let scalar_sine = block_on(integral_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(100.0),
            vec![
                Value::from("AbsTol"),
                Value::Num(1.0e-12),
                Value::from("RelTol"),
                Value::Num(0.0),
                Value::from("MaxFunEvals"),
                Value::Num(1000.0),
                Value::from("MaxIntervalCount"),
                Value::Num(1.0),
            ],
        ))
        .expect_err("oscillatory sine requires refinement beyond one interval");
        assert!(scalar_sine
            .message()
            .contains("MaxIntervalCount adaptive subintervals"));

        let array_polynomial = block_on(integral_builtin(
            Value::FunctionHandle("__integral_array_polynomial".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![
                Value::from("ArrayValued"),
                Value::Bool(true),
                Value::from("MaxFunEvals"),
                Value::Num(15.0),
                Value::from("MaxIntervalCount"),
                Value::Num(1.0),
            ],
        ))
        .expect("array polynomial accepts within one interval");
        assert!(matches!(array_polynomial, Value::Tensor(value) if value.shape() == [1, 3]));

        let array_sine = block_on(integral_builtin(
            Value::FunctionHandle("__integral_array_sine".into()),
            Value::Num(0.0),
            Value::Num(100.0),
            vec![
                Value::from("ArrayValued"),
                Value::Bool(true),
                Value::from("AbsTol"),
                Value::Num(1.0e-12),
                Value::from("RelTol"),
                Value::Num(0.0),
                Value::from("MaxFunEvals"),
                Value::Num(1000.0),
                Value::from("MaxIntervalCount"),
                Value::Num(1.0),
            ],
        ))
        .expect_err("array components require further refinement");
        assert!(array_sine
            .message()
            .contains("MaxIntervalCount adaptive subintervals"));
    }

    #[test]
    fn open_rule_checks_its_initial_evaluation_budget_before_callback_effects() {
        COUNTED_CALLBACKS.with(|count| count.set(0));
        let insufficient = block_on(integral_builtin(
            Value::FunctionHandle("__integral_counted_one".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::from("MaxFunEvals"), Value::Num(5.0)],
        ))
        .expect_err("the selected rule requires fifteen sample evaluations");
        assert!(insufficient
            .message()
            .contains("exceeded maximum function evaluations"));
        COUNTED_CALLBACKS.with(|count| assert_eq!(count.get(), 0));

        let enough = block_on(integral_builtin(
            Value::FunctionHandle("__integral_counted_one".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![
                Value::from("MaxFunEvals"),
                Value::Num(15.0),
                Value::from("MaxIntervalCount"),
                Value::Num(1.0),
            ],
        ))
        .expect("one vectorized batch is sufficient for a constant");
        assert!(matches!(enough, Value::Num(value) if (value - 1.0).abs() < 1.0e-12));
        COUNTED_CALLBACKS.with(|count| assert_eq!(count.get(), 1));
    }

    #[test]
    fn interior_rule_integrates_endpoint_singularity_and_improper_real_domains() {
        let endpoint = block_on(integral_builtin(
            Value::FunctionHandle("__integral_endpoint_sqrt".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![
                Value::from("AbsTol"),
                Value::Num(1.0e-10),
                Value::from("RelTol"),
                Value::Num(1.0e-8),
            ],
        ))
        .expect("the callback is never evaluated at singular endpoint zero");
        assert!(matches!(endpoint, Value::Num(value) if (value - 2.0).abs() < 1.0e-7));

        let right = block_on(integral_builtin(
            Value::FunctionHandle("__integral_exp_tail".into()),
            Value::Num(0.0),
            Value::Num(f64::INFINITY),
            Vec::new(),
        ))
        .expect("positive infinite tail");
        assert!(matches!(right, Value::Num(value) if (value - 1.0).abs() < 1.0e-7));

        let reversed = block_on(integral_builtin(
            Value::FunctionHandle("__integral_exp_tail".into()),
            Value::Num(f64::INFINITY),
            Value::Num(0.0),
            Vec::new(),
        ))
        .expect("reversed positive infinite tail");
        assert!(matches!(reversed, Value::Num(value) if (value + 1.0).abs() < 1.0e-7));

        let both = block_on(integral_builtin(
            Value::FunctionHandle("__integral_gaussian".into()),
            Value::Num(f64::NEG_INFINITY),
            Value::Num(f64::INFINITY),
            Vec::new(),
        ))
        .expect("both infinite tails share one global error budget");
        assert!(
            matches!(both, Value::Num(value) if (value - std::f64::consts::PI.sqrt()).abs() < 1.0e-6)
        );

        let array = block_on(integral_builtin(
            Value::FunctionHandle("__integral_array_exp_tails".into()),
            Value::Num(0.0),
            Value::Num(f64::INFINITY),
            vec![Value::from("ArrayValued"), Value::Bool(true)],
        ))
        .expect("array-valued infinite tail uses the same numerical rule");
        let Value::Tensor(array) = array else {
            panic!("expected fixed-shape array")
        };
        assert_eq!(array.shape(), [1, 2]);
        assert!((tensor::tensor_value_f64(&array, 0) - 1.0).abs() < 1.0e-6);
        assert!((tensor::tensor_value_f64(&array, 1) - 0.5).abs() < 1.0e-6);
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
    fn renamed_tolerance_names_follow_typed_release_gate_and_ordered_overrides() {
        let call = || {
            block_on(integral_builtin(
                Value::FunctionHandle("__integral_square".into()),
                Value::Num(0.0),
                Value::Num(1.0),
                vec![
                    Value::from("AbsTol"),
                    Value::Num(0.0),
                    Value::from("RelTol"),
                    Value::Num(0.0),
                    Value::from("AbsoluteTolerance"),
                    Value::Num(1.0e-10),
                    Value::from("RelativeTolerance"),
                    Value::Num(1.0e-6),
                ],
            ))
        };
        let strict_gate = crate::compatibility::ensure_builtin_feature_enabled(
            &INTEGRAL_RENAMED_TOLERANCE_EXTENSION,
            NAME,
        );
        if strict_gate.is_ok() {
            assert!(matches!(call(), Ok(Value::Num(value)) if (value - 1.0 / 3.0).abs() < 1.0e-7));
        } else {
            let strict = call().expect_err("feature is newer than the central pin");
            assert_eq!(
                strict.identifier(),
                INTEGRAL_RENAMED_TOLERANCE_EXTENSION.error_identifier
            );
        }
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = call().expect("RunMat mode accepts both renamed overrides");
        assert!(matches!(result, Value::Num(value) if (value - 1.0 / 3.0).abs() < 1.0e-7));
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

        let invalid_host_waypoint = block_on(integral_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![
                Value::from("AbsTol"),
                Value::GpuTensor(valid_option.handle.clone()),
                Value::from("Waypoints"),
                Value::Num(2.0),
            ],
        ))
        .expect_err("known out-of-range waypoint precedes an unrelated resident option");
        assert_eq!(
            invalid_host_waypoint.identifier(),
            INTEGRAL_ERROR_INVALID_ARGUMENT.identifier
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert_eq!(valid_option.probe.retirements(), 0);
        valid_option
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();
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
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("cancelled invocation");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        assert!(error.source.is_some());
    }

    #[test]
    fn active_array_integral_turn_cancels_or_drops_without_publication() {
        use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
        use futures::task::noop_waker;
        use runmat_execution::cancellation::CancellationSource;
        use runmat_execution::CancellationReason;
        use std::cell::Cell;
        use std::future::Future;
        use std::num::NonZeroU64;
        use std::rc::Rc;
        use std::task::{Context, Poll};

        for cancel in [false, true] {
            let source = CancellationSource::new();
            let dropped = Rc::new(Cell::new(false));
            let posted = Rc::new(Cell::new(0));
            let dropped_turn = dropped.clone();
            let posted_turn = posted.clone();
            let budget = CpuCooperationBudget::with_test_turn(
                CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
                source.token(),
                move || {
                    posted_turn.set(posted_turn.get() + 1);
                    Box::pin(PendingIntegralTurn(dropped_turn.clone()))
                },
            );
            let mut cooperation =
                QuadratureCooperation::with_budget(QuadratureIdentity::Integral, budget);
            let mut invocation = Box::pin(integral_with_cooperation(
                Value::FunctionHandle("__integral_array_opposite_constants".into()),
                Value::Num(0.0),
                Value::Num(1.0),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::from("ArrayValued"),
                    Value::Bool(true),
                ]),
                &mut cooperation,
            ));
            let waker = noop_waker();
            let mut context = Context::from_waker(&waker);
            assert!(invocation.as_mut().poll(&mut context).is_pending());
            assert_eq!(posted.get(), 1, "callback work reached the owning turn");
            if cancel {
                source.cancel(CancellationReason::User);
                let Poll::Ready(Err(error)) = invocation.as_mut().poll(&mut context) else {
                    panic!("cancelled integral cannot publish an array")
                };
                assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
                assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
                assert!(matches!(
                    error
                        .source
                        .as_deref()
                        .and_then(|cause| cause.downcast_ref::<ExecutionServiceError>()),
                    Some(ExecutionServiceError::Cancelled)
                ));
            }
            drop(invocation);
            assert!(dropped.get(), "abandoned turn was released");
        }
    }
}
