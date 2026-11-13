//! MATLAB-compatible `linspace` builtin with GPU-aware semantics for RunMat.
use runmat_builtins::{
    LINSPACE_ERROR_ARG_COUNT, LINSPACE_ERROR_COUNT_NOT_FINITE, LINSPACE_ERROR_COUNT_NOT_SCALAR,
    LINSPACE_ERROR_COUNT_TOO_LARGE,
};

use log::trace;
use runmat_accelerate_api::ProviderPrecision;
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, IntValue, NumericDType, Tensor, Value};

use crate::build_runtime_error;
use crate::builtins::common::residency::{sequence_gpu_preference, SequenceIntent};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::linspace")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "linspace",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("sequence")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "The typed sequence hook owns exact class and endpoints; pre-attempt absence permits class-preserving host generation and upload.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin("linspace")
        .build()
}

fn provider_sequence_error(
    stage: &str,
    error: runmat_accelerate_api::ProviderExecutionError,
) -> crate::RuntimeError {
    let mut result = builtin_error(format!("linspace: {stage} failed: {error}"));
    result.gpu_gather_retry = crate::GpuGatherRetry::Never;
    crate::runtime_error::with_provider_execution_source(result, error)
}

fn linspace_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    linspace_error_with_message(error.message, error)
}

fn linspace_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> crate::RuntimeError {
    linspace_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn linspace_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("linspace");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::linspace")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "linspace",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Sequence generation is treated as a sink and is not fused with other operations.",
};

#[runtime_builtin(
    name = "linspace",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::linspace"
)]
async fn linspace_builtin(
    start: Value,
    stop: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(linspace_error(&LINSPACE_ERROR_ARG_COUNT));
    }

    let (start_scalar, start_source) = parse_scalar("linspace", start).await?;
    let (stop_scalar, stop_source) = parse_scalar("linspace", stop).await?;
    let protected = start_source
        .iter()
        .chain(stop_source.iter())
        .collect::<Vec<_>>();
    let source = gpu_helpers::select_resident_output_source(
        protected.iter().map(|handle| (*handle).clone()),
        "linspace",
    )?;
    let source = source.as_ref();

    let count = if rest.is_empty() {
        Count::Length(100)
    } else {
        parse_count(&rest[0]).await?
    };
    if matches!(count, Count::Nan) {
        let single = start_scalar.single || stop_scalar.single;
        let tensor = if single {
            Tensor::from_f32(vec![f32::NAN], vec![1, 1])
        } else {
            Tensor::new(vec![f64::NAN], vec![1, 1])
        }
        .map_err(|error| builtin_error(format!("linspace: {error}")))?;
        return match source {
            Some(source) => restore_real_sequence(source, tensor, &protected),
            None => Ok(Value::Tensor(tensor)),
        };
    }
    let Count::Length(count) = count else {
        unreachable!("NaN count returned above")
    };

    let residency = sequence_gpu_preference(count, SequenceIntent::Linspace, source.is_some());
    if log::log_enabled!(log::Level::Trace) {
        trace!(
            "linspace: len={} prefer_gpu={} reason={:?}",
            count,
            residency.prefer_gpu,
            residency.reason
        );
    }
    let prefer_gpu = residency.prefer_gpu;
    build_sequence(
        start_scalar,
        stop_scalar,
        count,
        prefer_gpu,
        source,
        &protected,
    )
}

#[derive(Clone, Copy)]
enum Scalar {
    Real(f64),
    Complex { re: f64, im: f64 },
}

#[derive(Clone, Copy)]
struct Endpoint {
    scalar: Scalar,
    single: bool,
}

impl Endpoint {
    fn parts(&self) -> (f64, f64) {
        match self.scalar {
            Scalar::Real(r) => (r, 0.0),
            Scalar::Complex { re, im } => (re, im),
        }
    }

    fn is_complex(&self) -> bool {
        matches!(self.scalar, Scalar::Complex { .. })
    }
}

async fn parse_scalar(
    name: &str,
    value: Value,
) -> crate::BuiltinResult<(Endpoint, Option<runmat_accelerate_api::GpuTensorHandle>)> {
    match value {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_integer_type(&handle).is_some()
                || runmat_accelerate_api::handle_is_logical(&handle)
            {
                return Err(builtin_error(format!(
                    "{name}: endpoints must be single or double scalars"
                )));
            }
            runmat_accelerate_api::numeric_readback_for_handle(&handle)
                .map_err(|_| builtin_error(format!("{name}: no provider owns the endpoint")))?;
            let gathered = {
                let _residency = gpu_helpers::ResidencyRestoreGuard::new(&handle);
                gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await?
            };
            match gathered {
                Value::Tensor(tensor) => {
                    tensor_scalar(name, &tensor).map(|scalar| (scalar, Some(handle)))
                }
                Value::ComplexTensor(tensor) => {
                    complex_tensor_scalar(name, &tensor).map(|scalar| (scalar, Some(handle)))
                }
                _ => Err(builtin_error(format!(
                    "{name}: endpoints must be single or double scalars"
                ))),
            }
        }
        other => parse_scalar_host(name, other),
    }
}

fn parse_scalar_host(
    name: &str,
    value: Value,
) -> crate::BuiltinResult<(Endpoint, Option<runmat_accelerate_api::GpuTensorHandle>)> {
    match value {
        Value::Num(n) => Ok((
            Endpoint {
                scalar: Scalar::Real(n),
                single: false,
            },
            None,
        )),
        Value::Complex(re, im) => Ok((
            Endpoint {
                scalar: Scalar::Complex { re, im },
                single: false,
            },
            None,
        )),
        Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => Err(builtin_error(format!(
            "{name}: endpoints must be single or double scalars"
        ))),
        Value::Tensor(t) => tensor_scalar(name, &t).map(|scalar| (scalar, None)),
        Value::ComplexTensor(t) => complex_tensor_scalar(name, &t).map(|scalar| (scalar, None)),
        Value::GpuTensor(_) => unreachable!("GpuTensor handled by parse_scalar"),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => Err(builtin_error(
            format!("{name}: endpoints must be numeric scalars; received a string-like value"),
        )),
        other => Err(builtin_error(format!(
            "{name}: endpoints must be numeric scalars; received {other:?}"
        ))),
    }
}

fn tensor_scalar(name: &str, tensor: &Tensor) -> crate::BuiltinResult<Endpoint> {
    if !tensor::is_scalar_tensor(tensor) {
        return Err(builtin_error(format!("{name}: expected scalar input")));
    }
    if !matches!(
        tensor.numeric_dtype(),
        runmat_value::NumericDType::F32 | runmat_value::NumericDType::F64
    ) {
        return Err(builtin_error(format!(
            "{name}: endpoints must be single or double scalars"
        )));
    }
    Ok(Endpoint {
        scalar: Scalar::Real(tensor::tensor_value_f64(tensor, 0)),
        single: tensor.numeric_dtype() == NumericDType::F32,
    })
}

fn complex_tensor_scalar(name: &str, tensor: &ComplexTensor) -> crate::BuiltinResult<Endpoint> {
    if tensor.integer_storage().is_some() {
        return Err(builtin_error(format!(
            "{name}: endpoints must be single or double scalars"
        )));
    }
    let values = tensor.materialize_f64();
    if values.len() != 1 {
        return Err(builtin_error(format!("{name}: expected scalar input")));
    }
    let (re, im) = values[0];
    Ok(Endpoint {
        scalar: Scalar::Complex { re, im },
        single: tensor.numeric_dtype() == NumericDType::F32,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Count {
    Length(usize),
    Nan,
}

async fn parse_count(value: &Value) -> crate::BuiltinResult<Count> {
    match value {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_is_logical(handle) {
                return Err(linspace_error(&LINSPACE_ERROR_COUNT_NOT_SCALAR));
            }
            let tensor = gpu_helpers::gather_tensor_async(handle).await?;
            if !tensor::is_scalar_tensor(&tensor) {
                return Err(linspace_error(&LINSPACE_ERROR_COUNT_NOT_SCALAR));
            }
            parse_tensor_count(&tensor)
        }
        other => parse_count_host(other),
    }
}

fn parse_count_host(value: &Value) -> crate::BuiltinResult<Count> {
    match value {
        Value::Int(i) => parse_integer_count(i),
        Value::Num(n) => parse_numeric_count(*n),
        Value::Tensor(t) => {
            if !tensor::is_scalar_tensor(t) {
                return Err(linspace_error(&LINSPACE_ERROR_COUNT_NOT_SCALAR));
            }
            parse_tensor_count(t)
        }
        Value::GpuTensor(_) => unreachable!("GpuTensor handled by parse_count"),
        other => Err(linspace_error_with_detail(
            &LINSPACE_ERROR_COUNT_NOT_SCALAR,
            format!("got {other:?}"),
        )),
    }
}

fn parse_tensor_count(tensor: &Tensor) -> crate::BuiltinResult<Count> {
    if let Some(storage) = tensor.integer_storage() {
        let value = storage
            .value_at(0)
            .expect("scalar integer tensor has one storage value");
        return parse_integer_count(&value);
    }
    parse_numeric_count(tensor::tensor_value_f64(tensor, 0))
}

fn parse_integer_count(value: &IntValue) -> crate::BuiltinResult<Count> {
    if value.try_to_i64().is_some_and(|value| value < 0) {
        return Ok(Count::Length(0));
    }
    value
        .try_to_usize()
        .map(Count::Length)
        .ok_or_else(|| linspace_error(&LINSPACE_ERROR_COUNT_TOO_LARGE))
}

fn parse_numeric_count(raw: f64) -> crate::BuiltinResult<Count> {
    if raw.is_nan() {
        return Ok(Count::Nan);
    }
    if raw.is_infinite() {
        return Err(linspace_error(&LINSPACE_ERROR_COUNT_NOT_FINITE));
    }
    let floored = raw.floor();
    if floored <= 0.0 {
        return Ok(Count::Length(0));
    }
    if floored > usize::MAX as f64 || (usize::BITS == 64 && floored == usize::MAX as f64) {
        return Err(linspace_error(&LINSPACE_ERROR_COUNT_TOO_LARGE));
    }
    Ok(Count::Length(floored as usize))
}

fn build_sequence(
    start: Endpoint,
    stop: Endpoint,
    count: usize,
    prefer_gpu: bool,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
    protected: &[&runmat_accelerate_api::GpuTensorHandle],
) -> crate::BuiltinResult<Value> {
    let (start_re, start_im) = start.parts();
    let (stop_re, stop_im) = stop.parts();
    let complex = start.is_complex() || stop.is_complex();
    let single = start.single || stop.single;

    if complex {
        let data = generate_complex_sequence(start_re, start_im, stop_re, stop_im, count)?;
        let tensor = if single {
            ComplexTensor::from_f32(
                data.into_iter()
                    .map(|(real, imag)| (real as f32, imag as f32))
                    .collect(),
                vec![1, count],
            )
        } else {
            ComplexTensor::new(data, vec![1, count])
        }
        .map_err(|e| builtin_error(format!("linspace: {e}")))?;
        let value = Value::ComplexTensor(tensor);
        return match source {
            Some(source) => gpu_helpers::restore_class_preserving_value_with_operands(
                source,
                value,
                "linspace",
                gpu_helpers::RestoreOperandContext::new(protected, protected),
            ),
            None => Ok(value),
        };
    }

    if prefer_gpu {
        #[cfg(all(test, feature = "wgpu"))]
        {
            if runmat_accelerate_api::provider().is_none() {
                let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                    runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                );
            }
        }
        let provider = source
            .and_then(gpu_helpers::exact_provider_for_handle)
            .or_else(runmat_accelerate_api::provider);
        if let Some(provider) = provider {
            if count > 0 {
                if log::log_enabled!(log::Level::Trace) {
                    trace!(
                        "linspace: attempting provider.sequence start={} stop={} count={}",
                        start_re,
                        stop_re,
                        count
                    );
                }
                let request = runmat_accelerate_api::ProviderSequenceRequest::linear(
                    start_re,
                    stop_re,
                    count,
                    requested_precision(single),
                );
                match provider.sequence(&request) {
                    Ok(Some(handle)) => {
                        trace!("linspace: provider.sequence succeeded");
                        if let Some(source) = source {
                            return validated_sequence_output(
                                source,
                                provider,
                                handle,
                                protected,
                                vec![1, count],
                                if single {
                                    NumericDType::F32
                                } else {
                                    NumericDType::F64
                                },
                                "linspace",
                            );
                        }
                        return validated_unowned_sequence_output(
                            provider,
                            handle,
                            vec![1, count],
                            if single {
                                NumericDType::F32
                            } else {
                                NumericDType::F64
                            },
                            "linspace",
                        );
                    }
                    Ok(None) => trace!("linspace: provider.sequence unavailable"),
                    Err(error) => return Err(provider_sequence_error("provider sequence", error)),
                }
            }
        }
    }

    let data = generate_real_sequence(start_re, stop_re, count)?;
    let tensor = if single {
        Tensor::from_f32(
            data.into_iter().map(|value| value as f32).collect(),
            vec![1, count],
        )
    } else {
        Tensor::new(data, vec![1, count])
    }
    .map_err(|e| builtin_error(format!("linspace: {e}")))?;
    if let Some(source) = source {
        return restore_real_sequence(source, tensor, protected);
    }
    if prefer_gpu {
        #[cfg(all(test, feature = "wgpu"))]
        {
            if runmat_accelerate_api::provider().is_none() {
                let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                    runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                );
            }
        }
        let provider = source
            .and_then(gpu_helpers::exact_provider_for_handle)
            .or_else(runmat_accelerate_api::provider);
        if let Some(provider) = provider {
            if !gpu_helpers::real_sequence_upload_feasible(
                provider,
                requested_precision(single),
                tensor.shape(),
            ) {
                return Ok(Value::Tensor(tensor));
            }
            match gpu_helpers::upload_tensor_protecting_typed(provider, &tensor, &[]) {
                Ok(handle) => {
                    return validated_unowned_sequence_output(
                        provider,
                        handle,
                        vec![1, count],
                        if single {
                            NumericDType::F32
                        } else {
                            NumericDType::F64
                        },
                        "linspace",
                    );
                }
                Err(runmat_accelerate_api::NumericUploadError::Provider(error))
                    if error.is_unsupported() => {}
                Err(error) => {
                    return Err(provider_sequence_error(
                        "typed sequence upload",
                        error.into_execution_error(),
                    ))
                }
            }
        }
    }
    Ok(Value::Tensor(tensor))
}

fn restore_real_sequence(
    source: &runmat_accelerate_api::GpuTensorHandle,
    tensor: Tensor,
    protected: &[&runmat_accelerate_api::GpuTensorHandle],
) -> crate::BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(source)
        .ok_or_else(|| builtin_error("linspace: no provider owns the endpoint"))?;
    if !gpu_helpers::real_sequence_upload_feasible(
        provider,
        requested_precision(tensor.numeric_dtype() == NumericDType::F32),
        tensor.shape(),
    ) {
        if runmat_accelerate_api::handle_is_explicit(source) {
            let mut error = builtin_error("linspace: owner cannot preserve requested GPU class");
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            return Err(error);
        }
        return Ok(Value::Tensor(tensor));
    }
    gpu_helpers::restore_class_preserving_value_with_operands(
        source,
        Value::Tensor(tensor),
        "linspace",
        gpu_helpers::RestoreOperandContext::new(protected, protected),
    )
}

fn requested_precision(single: bool) -> ProviderPrecision {
    if single {
        ProviderPrecision::F32
    } else {
        ProviderPrecision::F64
    }
}

fn validated_sequence_output(
    source: &runmat_accelerate_api::GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    output: runmat_accelerate_api::GpuTensorHandle,
    protected: &[&runmat_accelerate_api::GpuTensorHandle],
    shape: Vec<usize>,
    dtype: NumericDType,
    builtin: &str,
) -> crate::BuiltinResult<Value> {
    let expected_precision = match dtype {
        NumericDType::F32 => runmat_accelerate_api::ProviderPrecision::F32,
        _ => runmat_accelerate_api::ProviderPrecision::F64,
    };
    let valid = protected
        .iter()
        .all(|input| !input.shares_allocation_with(&output))
        && output.shape() == shape
        && output.device_id() == source.device_id()
        && gpu_helpers::exact_provider_for_handle(&output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && runmat_accelerate_api::handle_storage(&output)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(&output) == Some(expected_precision)
        && runmat_accelerate_api::handle_integer_type(&output).is_none()
        && !runmat_accelerate_api::handle_is_logical(&output);
    if !valid {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            builtin_error(format!(
                "{builtin}: provider returned malformed sequence output"
            )),
        ));
    }
    let output = gpu_helpers::propagate_owned_output_provenance(output, protected.iter().copied())?;
    gpu_helpers::resident_gpu_value(output)
}

fn validated_unowned_sequence_output(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    output: runmat_accelerate_api::GpuTensorHandle,
    shape: Vec<usize>,
    dtype: NumericDType,
    builtin: &str,
) -> crate::BuiltinResult<Value> {
    let expected_precision = match dtype {
        NumericDType::F32 => runmat_accelerate_api::ProviderPrecision::F32,
        _ => runmat_accelerate_api::ProviderPrecision::F64,
    };
    let valid = output.shape() == shape
        && output.device_id() == provider.device_id()
        && gpu_helpers::exact_provider_for_handle(&output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && runmat_accelerate_api::handle_storage(&output)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(&output) == Some(expected_precision)
        && runmat_accelerate_api::handle_integer_type(&output).is_none()
        && !runmat_accelerate_api::handle_is_logical(&output);
    if !valid {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            builtin_error(format!(
                "{builtin}: provider returned malformed sequence output"
            )),
        ));
    }
    gpu_helpers::resident_gpu_value(output)
}

fn generate_real_sequence(start: f64, stop: f64, count: usize) -> crate::BuiltinResult<Vec<f64>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if count == 1 {
        return Ok(vec![stop]);
    }
    let mut data = Vec::new();
    data.try_reserve_exact(count)
        .map_err(|error| builtin_error(format!("linspace: cannot reserve sequence: {error}")))?;
    let step = (stop - start) / ((count - 1) as f64);
    for idx in 0..count {
        data.push(start + (idx as f64) * step);
    }
    if let Some(first) = data.first_mut() {
        *first = start;
    }
    if let Some(last) = data.last_mut() {
        *last = stop;
    }
    Ok(data)
}

fn generate_complex_sequence(
    start_re: f64,
    start_im: f64,
    stop_re: f64,
    stop_im: f64,
    count: usize,
) -> crate::BuiltinResult<Vec<(f64, f64)>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if count == 1 {
        return Ok(vec![(stop_re, stop_im)]);
    }
    let mut data = Vec::new();
    data.try_reserve_exact(count)
        .map_err(|error| builtin_error(format!("linspace: cannot reserve sequence: {error}")))?;
    let step_re = (stop_re - start_re) / ((count - 1) as f64);
    let step_im = (stop_im - start_im) / ((count - 1) as f64);
    for idx in 0..count {
        let re = start_re + (idx as f64) * step_re;
        let im = start_im + (idx as f64) * step_im;
        data.push((re, im));
    }
    if let Some(first) = data.first_mut() {
        *first = (start_re, start_im);
    }
    if let Some(last) = data.last_mut() {
        *last = (stop_re, stop_im);
    }
    Ok(data)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_accelerate_api::HostTensorView;

    #[test]
    fn count_parser_preserves_representable_uint64() {
        assert_eq!(
            parse_count_host(&Value::Int(IntValue::U64(u64::MAX))).ok(),
            usize::try_from(u64::MAX).ok().map(Count::Length)
        );
        assert_eq!(
            parse_count_host(&Value::Int(IntValue::I64(-1))).unwrap(),
            Count::Length(0)
        );
        assert!(parse_count_host(&Value::Num(usize::MAX as f64)).is_err());
        assert!(parse_count_host(&Value::Num((usize::MAX as f64) + 1.0)).is_err());
    }
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};
    use runmat_value::{IntValue, IntegerComplexStorage, IntegerStorage, Tensor};

    fn linspace_builtin(
        start: Value,
        stop: Value,
        rest: Vec<Value>,
    ) -> crate::BuiltinResult<Value> {
        block_on(super::linspace_builtin(
            start,
            stop,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_basic() {
        let result = linspace_builtin(
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Int(IntValue::I32(5))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), vec![1, 5]);
                let expected = [0.0, 0.25, 0.5, 0.75, 1.0];
                for (idx, expected_val) in expected.iter().enumerate() {
                    assert!((t.materialize_f64()[idx] - expected_val).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn double_sequence_declines_f32_preferred_owner_without_losing_class() {
        test_support::with_f32_test_provider(|_| {
            let output = build_sequence(
                Endpoint {
                    scalar: Scalar::Real(0.0),
                    single: false,
                },
                Endpoint {
                    scalar: Scalar::Real(1.0),
                    single: false,
                },
                5,
                true,
                None,
                &[],
            )
            .expect("double host fallback");
            match output {
                Value::Tensor(tensor) => {
                    assert_eq!(tensor.numeric_dtype(), NumericDType::F64);
                    assert_eq!(tensor.shape(), &[1, 5]);
                    assert_eq!(tensor.materialize_f64()[4], 1.0);
                }
                other => panic!("expected class-preserving host result, got {other:?}"),
            }
        });
    }

    #[test]
    fn host_sequence_retains_negative_zero_start() {
        assert_eq!(
            generate_real_sequence(-0.0, 1.0, 3).unwrap()[0].to_bits(),
            (-0.0f64).to_bits()
        );
        assert_eq!(
            generate_complex_sequence(-0.0, -0.0, 1.0, 1.0, 3).unwrap()[0]
                .0
                .to_bits(),
            (-0.0f64).to_bits()
        );
    }

    #[test]
    fn host_sequence_rejects_unreservable_count() {
        assert!(generate_real_sequence(0.0, 1.0, usize::MAX).is_err());
    }

    #[test]
    fn single_sequence_can_use_native_typed_hook_on_f64_preferred_owner() {
        test_support::with_test_provider(|_| {
            let output = build_sequence(
                Endpoint {
                    scalar: Scalar::Real(0.0),
                    single: true,
                },
                Endpoint {
                    scalar: Scalar::Real(1.0),
                    single: false,
                },
                5,
                true,
                None,
                &[],
            )
            .expect("single typed upload");
            let Value::GpuTensor(handle) = output else {
                panic!("expected resident single")
            };
            assert_eq!(
                runmat_accelerate_api::handle_precision(&handle),
                Some(ProviderPrecision::F32)
            );
        });
    }

    #[test]
    fn sequence_rejects_distinct_view_of_borrowed_allocation_without_retiring_source() {
        test_support::with_test_provider(|provider| {
            let source = gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(),
            )
            .unwrap();
            let alias = source.with_view(source.view().clone()).unwrap();
            assert_ne!(alias.view_identity(), source.view_identity());
            assert!(super::validated_sequence_output(
                &source,
                provider,
                alias,
                &[&source],
                vec![1, 2],
                NumericDType::F64,
                "linspace",
            )
            .is_err());
            let host = block_on(gpu_helpers::gather_tensor_async(&source)).unwrap();
            assert_eq!(host.materialize_f64(), vec![1.0, 2.0]);
        });
    }

    #[test]
    fn linspace_count_parser_preserves_typed_integer_tensors_exactly() {
        let large = 9_007_199_254_740_993_u64;
        let count = Tensor::new_integer(IntegerStorage::U64(vec![large]), vec![1, 1]).unwrap();

        assert_eq!(
            parse_count_host(&Value::Tensor(count)).unwrap(),
            Count::Length(large as usize)
        );
    }

    #[test]
    fn linspace_count_parser_ignores_poisoned_f64_mirrors_for_all_integer_classes() {
        let storages = [
            IntegerStorage::I8(vec![2]),
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I64(vec![2]),
            IntegerStorage::U8(vec![2]),
            IntegerStorage::U16(vec![2]),
            IntegerStorage::U32(vec![2]),
            IntegerStorage::U64(vec![2]),
        ];

        for storage in storages {
            let count = Tensor::new_integer(storage, vec![1, 1]).expect("integer count");
            assert_eq!(
                parse_count_host(&Value::Tensor(count)).unwrap(),
                Count::Length(2)
            );
        }
    }

    #[test]
    fn linspace_count_parser_maps_negative_typed_integer_tensors_to_empty() {
        let count = Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1]).unwrap();

        assert_eq!(
            parse_count_host(&Value::Tensor(count)).unwrap(),
            Count::Length(0)
        );
    }

    #[test]
    fn linspace_rejects_real_integer_tensor_endpoints() {
        let start =
            Tensor::new_integer(IntegerStorage::I64(vec![-3]), vec![1, 1]).expect("start tensor");
        let stop =
            Tensor::new_integer(IntegerStorage::U64(vec![9]), vec![1, 1]).expect("stop tensor");

        let error = linspace_builtin(
            Value::Tensor(start),
            Value::Tensor(stop),
            vec![Value::Int(IntValue::I32(3))],
        )
        .expect_err("integer endpoints must be rejected");
        assert!(error.message().contains("single or double"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_default_count() {
        let result =
            linspace_builtin(Value::Num(-1.0), Value::Num(1.0), Vec::new()).expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), vec![1, 100]);
                assert!((t.materialize_f64().first().copied().unwrap() + 1.0).abs() < 1e-12);
                assert!((t.materialize_f64().last().copied().unwrap() - 1.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_zero_count() {
        let result = linspace_builtin(
            Value::Num(0.0),
            Value::Num(10.0),
            vec![Value::Int(IntValue::I32(0))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), vec![1, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_single_point() {
        let result = linspace_builtin(
            Value::Num(5.0),
            Value::Num(9.0),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), vec![1, 1]);
                assert!((t.materialize_f64()[0] - 9.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_fractional_count_rounds_down() {
        let result = linspace_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Num(3.5)])
            .expect("fractional counts floor");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor")
        };
        assert_eq!(tensor.shape(), vec![1, 3]);
        assert_eq!(tensor.materialize_f64(), vec![0.0, 0.5, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_negative_count_returns_empty() {
        let result = linspace_builtin(
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Int(IntValue::I32(-2))],
        )
        .expect("negative count");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor")
        };
        assert_eq!(tensor.shape(), vec![1, 0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_infinite_count_errors() {
        let err = linspace_builtin(
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Num(f64::INFINITY)],
        )
        .expect_err("expected error");
        assert!(err.message().contains("finite"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_nan_count_returns_scalar_nan() {
        let result = linspace_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Num(f64::NAN)])
            .expect("NaN count");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor")
        };
        assert_eq!(tensor.shape(), vec![1, 1]);
        assert!(tensor.materialize_f64()[0].is_nan());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_non_scalar_count_errors() {
        let sz = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let err = linspace_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Tensor(sz)])
            .expect_err("expected error");
        assert!(err.message().contains("scalar"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_complex_sequence() {
        let result = linspace_builtin(
            Value::Complex(1.0, 1.0),
            Value::Complex(-3.0, 2.0),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("linspace");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape(), vec![1, 4]);
                let expected = [
                    (1.0, 1.0),
                    (-0.3333333333333333, 1.3333333333333333),
                    (-1.6666666666666667, 1.6666666666666667),
                    (-3.0, 2.0),
                ];
                for (idx, &(re, im)) in expected.iter().enumerate() {
                    let (r, i) = t.materialize_f64()[idx];
                    assert!((r - re).abs() < 1e-9);
                    assert!((i - im).abs() < 1e-9);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn linspace_rejects_complex_integer_tensor_endpoints() {
        let start_storage =
            IntegerComplexStorage::new(IntegerStorage::I16(vec![1]), IntegerStorage::I16(vec![1]))
                .expect("start storage");
        let start = ComplexTensor::new_integer(start_storage, vec![1, 1]).expect("start tensor");
        let stop_storage =
            IntegerComplexStorage::new(IntegerStorage::I16(vec![-3]), IntegerStorage::I16(vec![2]))
                .expect("stop storage");
        let stop = ComplexTensor::new_integer(stop_storage, vec![1, 1]).expect("stop tensor");

        let error = linspace_builtin(
            Value::ComplexTensor(start),
            Value::ComplexTensor(stop),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect_err("integer endpoints must be rejected");
        assert!(error.message().contains("single or double"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_rejects_logical_endpoints() {
        let error = linspace_builtin(
            Value::Bool(true),
            Value::Bool(false),
            vec![Value::Int(IntValue::I32(3))],
        )
        .expect_err("logical endpoints must be rejected");
        assert!(error.message().contains("single or double"));
    }

    #[test]
    fn linspace_rejects_scalar_integer_endpoints() {
        let error = linspace_builtin(
            Value::Int(IntValue::I8(0)),
            Value::Num(1.0),
            vec![Value::Int(IntValue::I32(3))],
        )
        .expect_err("integer endpoints must be rejected");
        assert!(error.message().contains("single or double"));
    }

    #[test]
    fn linspace_rejects_resident_integer_endpoints_from_metadata() {
        test_support::with_test_provider(|provider| {
            let endpoint = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&[9_007_199_254_740_993]),
                    shape: &[1, 1],
                })
                .expect("integer endpoint upload");
            let error = linspace_builtin(
                Value::GpuTensor(endpoint.clone()),
                Value::Num(1.0),
                vec![Value::Int(IntValue::I32(3))],
            )
            .expect_err("resident integer endpoint must be rejected");
            assert!(error.message().contains("single or double"));
            drop(endpoint);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_rejects_logical_count() {
        let error = linspace_builtin(Value::Num(3.0), Value::Num(7.0), vec![Value::Bool(true)])
            .expect_err("logical count must be rejected");
        assert!(error.message().contains("scalar"));
    }

    #[test]
    fn linspace_rejects_resident_logical_count_before_gathering() {
        test_support::with_test_provider(|provider| {
            let count = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &[1],
                    shape: &[1, 1],
                })
                .expect("count upload");
            let error = linspace_builtin(
                Value::Num(3.0),
                Value::Num(7.0),
                vec![Value::GpuTensor(count.clone())],
            )
            .expect_err("resident logical count must be rejected");
            assert!(error.message().contains("scalar"));
            drop(count);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_tensor_scalar_arguments() {
        let start = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let stop = Tensor::new(vec![4.0], vec![1, 1]).unwrap();
        let result = linspace_builtin(
            Value::Tensor(start),
            Value::Tensor(stop),
            vec![Value::Int(IntValue::I32(3))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), vec![1, 3]);
                let expected = [2.0, 3.0, 4.0];
                for (idx, expected_val) in expected.iter().enumerate() {
                    assert!((t.materialize_f64()[idx] - expected_val).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn linspace_preserves_single_real_and_complex_endpoint_class() {
        let start = Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap();
        let stop = Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap();
        let result = linspace_builtin(
            Value::Tensor(start),
            Value::Tensor(stop),
            vec![Value::Num(3.9)],
        )
        .unwrap();
        let Value::Tensor(result) = result else {
            panic!("expected real single tensor")
        };
        assert_eq!(result.numeric_dtype(), NumericDType::F32);
        assert_eq!(result.as_f32_slice(), Some(&[0.0, 0.5, 1.0][..]));

        let start = ComplexTensor::from_f32(vec![(0.0, 0.0)], vec![1, 1]).unwrap();
        let stop = ComplexTensor::from_f32(vec![(1.0, 0.0)], vec![1, 1]).unwrap();
        let result = linspace_builtin(
            Value::ComplexTensor(start),
            Value::ComplexTensor(stop),
            vec![Value::Int(IntValue::I32(2))],
        )
        .unwrap();
        let Value::ComplexTensor(result) = result else {
            panic!("expected complex single tensor")
        };
        assert_eq!(result.numeric_dtype(), NumericDType::F32);
        assert_eq!(
            result.as_f32_slice().map(|values| values
                .iter()
                .copied()
                .map(<(f32, f32)>::from)
                .collect::<Vec<_>>()),
            Some(vec![(0.0, 0.0), (1.0, 0.0)])
        );
    }

    #[test]
    fn linspace_nan_count_preserves_single_class() {
        let start = Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap();
        let stop = Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap();
        let result = linspace_builtin(
            Value::Tensor(start),
            Value::Tensor(stop),
            vec![Value::Num(f64::NAN)],
        )
        .unwrap();
        let Value::Tensor(result) = result else {
            panic!("expected single NaN tensor")
        };
        assert_eq!(result.numeric_dtype(), NumericDType::F32);
        assert!(result.as_f32_slice().unwrap()[0].is_nan());
    }

    #[test]
    fn linspace_rejects_typed_integer_tensor_endpoints() {
        let start = Tensor::new_integer(IntegerStorage::I16(vec![-2]), vec![1, 1]).expect("start");
        let stop = Tensor::new_integer(IntegerStorage::U16(vec![4]), vec![1, 1]).expect("stop");

        let error = linspace_builtin(
            Value::Tensor(start),
            Value::Tensor(stop),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect_err("integer endpoints must be rejected");
        assert!(error.message().contains("single or double"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_equal_endpoints_fill_with_endpoint() {
        let result = linspace_builtin(
            Value::Num(5.0),
            Value::Num(5.0),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), vec![1, 4]);
                assert!(t.materialize_f64().iter().all(|v| (*v - 5.0).abs() < 1e-12));
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let start = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let stop = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let start_view = runmat_accelerate_api::HostTensorView {
                data: &start.materialize_f64(),
                shape: start.shape(),
            };
            let stop_view = runmat_accelerate_api::HostTensorView {
                data: &stop.materialize_f64(),
                shape: stop.shape(),
            };
            let start_handle = provider.upload(&start_view).expect("upload start");
            let stop_handle = provider.upload(&stop_view).expect("upload stop");
            let result = linspace_builtin(
                Value::GpuTensor(start_handle),
                Value::GpuTensor(stop_handle),
                vec![Value::Int(IntValue::I32(5))],
            )
            .expect("linspace");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected = [0.0, 0.25, 0.5, 0.75, 1.0];
                    assert_eq!(gathered.shape(), vec![1, 5]);
                    for (idx, expected_val) in expected.iter().enumerate() {
                        assert!((gathered.materialize_f64()[idx] - expected_val).abs() < 1e-12);
                    }
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn linspace_gpu_nan_count_preserves_residency() {
        test_support::with_test_provider(|provider| {
            let start = provider
                .upload(&HostTensorView {
                    data: &[0.0],
                    shape: &[1, 1],
                })
                .unwrap();
            let result = linspace_builtin(
                Value::GpuTensor(start.clone()),
                Value::Num(1.0),
                vec![Value::Num(f64::NAN)],
            )
            .unwrap();
            assert!(matches!(result, Value::GpuTensor(_)));
            let gathered = test_support::gather(result).unwrap();
            assert_eq!(gathered.shape(), vec![1, 1]);
            assert!(gathered.materialize_f64()[0].is_nan());
            drop(start);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_gpu_zero_count_produces_gpu_empty_vector() {
        test_support::with_test_provider(|provider| {
            let start = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let stop = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let start_view = runmat_accelerate_api::HostTensorView {
                data: &start.materialize_f64(),
                shape: start.shape(),
            };
            let stop_view = runmat_accelerate_api::HostTensorView {
                data: &stop.materialize_f64(),
                shape: stop.shape(),
            };
            let start_handle = provider.upload(&start_view).expect("upload start");
            let stop_handle = provider.upload(&stop_view).expect("upload stop");
            let result = linspace_builtin(
                Value::GpuTensor(start_handle),
                Value::GpuTensor(stop_handle),
                vec![Value::Int(IntValue::I32(0))],
            )
            .expect("linspace");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape(), vec![1, 0]);
                    assert!(gathered.materialize_f64().is_empty());
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn linspace_wgpu_matches_cpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };
        use runmat_accelerate_api::{AccelProvider, HostTensorView, ProviderPrecision};

        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };

        let start = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let stop = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let start_view = HostTensorView {
            data: &start.materialize_f64(),
            shape: start.shape(),
        };
        let stop_view = HostTensorView {
            data: &stop.materialize_f64(),
            shape: stop.shape(),
        };
        let start_handle = provider.upload(&start_view).expect("upload start");
        let stop_handle = provider.upload(&stop_view).expect("upload stop");

        let result = linspace_builtin(
            Value::GpuTensor(start_handle),
            Value::GpuTensor(stop_handle),
            vec![Value::Int(IntValue::I32(9))],
        )
        .expect("linspace");
        let gathered = test_support::gather(result).expect("gather");
        let expected = generate_real_sequence(0.0, 1.0, 9).unwrap();

        let precision = runmat_accelerate_api::provider()
            .expect("provider")
            .precision();
        let tol = match precision {
            ProviderPrecision::F64 => 1e-12,
            ProviderPrecision::F32 => 1e-5,
        };

        assert_eq!(gathered.shape(), vec![1, 9]);
        for (idx, expected_value) in expected.iter().enumerate() {
            let actual = gathered.materialize_f64()[idx];
            assert!(
                (actual - expected_value).abs() <= tol,
                "mismatch at {idx}: gpu={} expected={}",
                actual,
                expected_value
            );
        }
    }
}
