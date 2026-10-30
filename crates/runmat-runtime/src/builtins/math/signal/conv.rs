//! MATLAB-compatible `conv` builtin with GPU-aware semantics for RunMat.

#[cfg(test)]
#[path = "conv/placement_tests.rs"]
mod placement_tests;

use num_complex::Complex;
use runmat_accelerate_api::{ProviderConv1dOptions, ProviderConvMode, ProviderConvOrientation};
use runmat_builtins::{
    BuiltinErrorDescriptor, CONV_ERROR_ARG_COUNT, CONV_ERROR_BUILD_COMPLEX_OUTPUT,
    CONV_ERROR_BUILD_OUTPUT, CONV_ERROR_GATHER_FAILED, CONV_ERROR_INVALID_INPUT,
    CONV_ERROR_SHAPE_INVALID,
};
use runmat_macros::runtime_builtin;
use runmat_types::ConvolutionMode as ConvMode;
use runmat_value::{ComplexStorage, ComplexTensor, NumericDType, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::signal::common::SignalCooperation;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::conv")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "conv",
    op_kind: GpuOpKind::Custom("conv1d"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("conv1d")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses a native hook only for same-provider, same-device real floating inputs and accepts its result only when the handle has the required precision, real storage, input device, and provider ownership. Other supported inputs gather through their owners and restore only when class and complexity can be preserved.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::conv")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "conv",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Convolution boundaries terminate fusion plans; intermediate expressions run on the host today.",
};

const BUILTIN_NAME: &str = "conv";

fn conv_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    conv_error_with_message(error.message, error)
}

fn conv_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    conv_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn conv_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn conv_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "conv",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::conv"
)]
async fn conv_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    a: Value,
    b: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let mode = parse_mode(rest.view())?;
    admit_conv_operand(&a)?;
    admit_conv_operand(&b)?;
    let mut cooperation = SignalCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    cooperation.completed(0).await?;
    let single_output = value_uses_single(&a) || value_uses_single(&b);
    let complex_output = value_is_complex(&a) || value_is_complex(&b);
    let resident_inputs = resident_inputs([&a, &b]);
    let (device_value, mut placement) =
        try_conv_gpu(runtime.as_ref(), &a, &b, mode, single_output)?;
    if let Some(device_value) = device_value {
        return Ok(device_value);
    }
    if placement.is_none() {
        if let Some(shape) = host_output_shape(&a, &b, mode) {
            if let Some(outputs) =
                super::common::convolution::placement::host_output_representations(
                    &[shape],
                    single_output,
                    complex_output,
                )
            {
                placement = super::common::convolution::placement::select_host_variant(
                    runtime.as_ref(),
                    BUILTIN_NAME,
                    "vector.host",
                    Some(mode),
                    &[&a, &b],
                    &resident_inputs,
                    outputs,
                    host_scratch_bound(&a, &b),
                )?;
            }
        }
    }
    let restore_recorder = placement.as_ref().and_then(|selected| selected.recorder());
    let result = async move {
        let lhs = normalize_input(a, &mut cooperation).await?;
        let rhs = normalize_input(b, &mut cooperation).await?;
        if matches!(lhs.hint, OrientationHint::General)
            || matches!(rhs.hint, OrientationHint::General)
        {
            return Err(conv_error_with_detail(
                &CONV_ERROR_INVALID_INPUT,
                "inputs must be vectors",
            ));
        }
        let orientation = output_orientation(&lhs, &rhs, mode);

        if lhs.len == 0 || rhs.len == 0 {
            let data = if matches!(mode, ConvMode::Valid) && rhs.len == 0 {
                let mut zeros = Vec::new();
                zeros
                    .try_reserve_exact(lhs.len)
                    .map_err(conv_allocation_error)?;
                zeros.resize(lhs.len, Complex::new(0.0, 0.0));
                zeros
            } else {
                Vec::new()
            };
            let output = convert_output(data, orientation, single_output, complex_output)?;
            return restore_to_owner(output, &resident_inputs, restore_recorder.clone());
        }

        let full = convolve(&lhs.data, &rhs.data, single_output, &mut cooperation).await?;
        let shaped = apply_mode(full, mode, lhs.len, rhs.len);
        let output = convert_output(shaped, orientation, single_output, complex_output)?;
        restore_to_owner(output, &resident_inputs, restore_recorder)
    }
    .await;
    if let Some(placement) = placement {
        placement.finish_value(&result);
    }
    result
}

fn host_output_shape(a: &Value, b: &Value, mode: ConvMode) -> Option<Vec<usize>> {
    fn shape(value: &Value) -> Option<&[usize]> {
        match value {
            Value::Tensor(tensor) => Some(tensor.shape()),
            Value::ComplexTensor(tensor) => Some(tensor.shape()),
            Value::LogicalArray(array) => Some(&array.shape),
            Value::GpuTensor(handle) => Some(handle.shape()),
            Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Some(&[1, 1]),
            _ => None,
        }
    }
    let lhs = conv_meta_from_shape(shape(a)?);
    let rhs = conv_meta_from_shape(shape(b)?);
    let len = if lhs.len == 0 || rhs.len == 0 {
        if mode == ConvMode::Valid && rhs.len == 0 {
            lhs.len
        } else {
            0
        }
    } else {
        match mode {
            ConvMode::Full => lhs.len.checked_add(rhs.len)?.checked_sub(1)?,
            ConvMode::Same => lhs.len,
            ConvMode::Valid => lhs
                .len
                .checked_sub(rhs.len)
                .and_then(|n| n.checked_add(1))
                .unwrap_or(0),
        }
    };
    let orientation = if mode == ConvMode::Full {
        if lhs.hint == OrientationHint::Column && rhs.hint == OrientationHint::Column {
            Orientation::Column
        } else {
            Orientation::Row
        }
    } else {
        orientation_from_hints(lhs.hint, rhs.hint)
    };
    Some(match orientation {
        Orientation::Row => vec![1, len],
        Orientation::Column => vec![len, 1],
    })
}

fn host_scratch_bound(a: &Value, b: &Value) -> runmat_execution::ResourceByteBound {
    let bound = (|| {
        let len = |value: &Value| -> Option<u64> {
            let shape = match value {
                Value::Tensor(tensor) => tensor.shape(),
                Value::ComplexTensor(tensor) => tensor.shape(),
                Value::LogicalArray(array) => &array.shape,
                Value::GpuTensor(handle) => {
                    return u64::try_from(super::common::convolution::checked_vector_elements(
                        handle.shape(),
                    )?)
                    .ok()
                }
                Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => {
                    return Some(1)
                }
                _ => return None,
            };
            u64::try_from(super::common::convolution::checked_vector_elements(shape)?).ok()
        };
        let left = len(a)?;
        let right = len(b)?;
        let full = if left == 0 || right == 0 {
            0
        } else {
            left.checked_add(right)?.checked_sub(1)?
        };
        // Both materialized Complex<f64> inputs; the single path can hold a
        // Complex<f32> accumulator and converted Complex<f64> full result at
        // once. Output construction can coexist with the latter. Readback may
        // hold raw, decoded, and value payloads before coefficient conversion.
        let coefficients = left.checked_add(right)?.checked_mul(16)?;
        let work = full.checked_mul(24)?.checked_add(full.checked_mul(16)?)?;
        let gather = [a, b].into_iter().try_fold(0_u64, |total, value| {
            let bytes = match value {
                Value::GpuTensor(handle) => handle.physical().requested_bytes().checked_mul(3)?,
                _ => 0,
            };
            total.checked_add(bytes)
        })?;
        coefficients.checked_add(work)?.checked_add(gather)
    })();
    bound
        .map(runmat_execution::ResourceByteBound::Known)
        .unwrap_or(runmat_execution::ResourceByteBound::Unavailable(
            runmat_execution::ResourceBoundUnavailable::NoScratchBound,
        ))
}

fn admit_conv_operand(value: &Value) -> BuiltinResult<()> {
    if matches!(value, Value::GpuTensor(_)) {
        super::common::convolution::admit_resident_operand(value, || {
            conv_error(&CONV_ERROR_INVALID_INPUT)
        })?;
    }
    let shape = match value {
        Value::Tensor(tensor) => Some(tensor.shape()),
        Value::ComplexTensor(tensor) => Some(tensor.shape()),
        Value::LogicalArray(array) => Some(array.shape.as_slice()),
        Value::GpuTensor(handle) => Some(handle.shape()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => None,
        _ => return Err(conv_error(&CONV_ERROR_INVALID_INPUT)),
    };
    if shape
        .is_some_and(|shape| super::common::convolution::checked_vector_elements(shape).is_none())
    {
        return Err(conv_error_with_detail(
            &CONV_ERROR_INVALID_INPUT,
            "inputs must be vectors",
        ));
    }
    Ok(())
}

fn resident_inputs<'a>(
    values: impl IntoIterator<Item = &'a Value>,
) -> Vec<runmat_accelerate_api::GpuTensorHandle> {
    values
        .into_iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        })
        .collect()
}

fn restore_to_owner(
    value: Value,
    resident_inputs: &[runmat_accelerate_api::GpuTensorHandle],
    recorder: Option<runmat_accelerate_api::observation::ProviderOperationRecorder>,
) -> BuiltinResult<Value> {
    let Some(source) = resident_inputs.first() else {
        return Ok(value);
    };
    let protected = resident_inputs.iter().collect::<Vec<_>>();
    runmat_accelerate_api::observation::with_provider_observation(recorder, || {
        gpu_helpers::restore_class_preserving_value_protecting(
            source,
            value,
            BUILTIN_NAME,
            &protected,
        )
    })
}

fn value_uses_single(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::ComplexTensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_precision(handle) == Some(runmat_accelerate_api::ProviderPrecision::F32))
}

fn value_is_complex(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_storage(handle) == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrientationHint {
    Row,
    Column,
    Scalar,
    General,
    Empty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Orientation {
    Row,
    Column,
}

#[derive(Clone, Copy, Debug)]
struct ConvInputMeta {
    len: usize,
    hint: OrientationHint,
}

#[derive(Clone)]
struct ConvInput {
    data: Vec<Complex<f64>>,
    len: usize,
    hint: OrientationHint,
}

fn parse_mode(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<ConvMode> {
    match args.len() {
        0 => Ok(ConvMode::Full),
        1 => {
            let super::common::convolution::ConvolutionSelectorText::Text(text) =
                super::common::convolution::convolution_selector_text(&args[0]).map_err(
                    |source| {
                        let mut error = conv_error(&CONV_ERROR_SHAPE_INVALID);
                        error.source = Some(Box::new(source));
                        error
                    },
                )?
            else {
                return Err(conv_error(&CONV_ERROR_SHAPE_INVALID));
            };
            ConvMode::decode(&text).ok_or_else(|| conv_error(&CONV_ERROR_SHAPE_INVALID))
        }
        _ => Err(conv_error(&CONV_ERROR_ARG_COUNT)),
    }
}

fn try_conv_gpu(
    runtime: Option<&crate::context::RuntimeContext>,
    a: &Value,
    b: &Value,
    mode: ConvMode,
    single_output: bool,
) -> BuiltinResult<(
    Option<Value>,
    Option<super::common::convolution::placement::SelectedConvolution>,
)> {
    let (lhs_handle, rhs_handle) = match (a, b) {
        (Value::GpuTensor(lhs), Value::GpuTensor(rhs)) => (lhs, rhs),
        _ => return Ok((None, None)),
    };
    if runmat_accelerate_api::handle_integer_type(lhs_handle).is_some()
        || runmat_accelerate_api::handle_integer_type(rhs_handle).is_some()
        || runmat_accelerate_api::handle_is_logical(lhs_handle)
        || runmat_accelerate_api::handle_is_logical(rhs_handle)
        || value_is_complex(a)
        || value_is_complex(b)
    {
        return Ok((None, None));
    }
    if lhs_handle.device_id() != rhs_handle.device_id() {
        return Ok((None, None));
    }
    let provider = match runmat_accelerate_api::provider_for_handle(lhs_handle) {
        Some(p) => p,
        None => return Ok((None, None)),
    };
    let Some(rhs_provider) = runmat_accelerate_api::provider_for_handle(rhs_handle) else {
        return Ok((None, None));
    };
    if !runmat_accelerate_api::same_provider_instance(provider, rhs_provider) {
        return Ok((None, None));
    }

    let lhs_meta = conv_meta_from_shape(lhs_handle.shape());
    let rhs_meta = conv_meta_from_shape(rhs_handle.shape());

    if lhs_meta.len == 0 || rhs_meta.len == 0 {
        return Ok((None, None));
    }

    let supported = |meta: &ConvInputMeta| {
        matches!(
            meta.hint,
            OrientationHint::Row | OrientationHint::Column | OrientationHint::Scalar
        )
    };

    if !supported(&lhs_meta) || !supported(&rhs_meta) {
        return Ok((None, None));
    }

    if matches!(mode, ConvMode::Valid) && lhs_meta.len < rhs_meta.len {
        return Ok((None, None));
    }

    let orientation = if matches!(mode, ConvMode::Full) {
        if matches!(lhs_meta.hint, OrientationHint::Column)
            && matches!(rhs_meta.hint, OrientationHint::Column)
        {
            Orientation::Column
        } else {
            Orientation::Row
        }
    } else {
        match lhs_meta.hint {
            OrientationHint::Column | OrientationHint::General => Orientation::Column,
            OrientationHint::Row => Orientation::Row,
            OrientationHint::Scalar | OrientationHint::Empty => {
                orientation_from_hints(lhs_meta.hint, rhs_meta.hint)
            }
        }
    };
    let provider_orientation = match orientation {
        Orientation::Row => ProviderConvOrientation::Row,
        Orientation::Column => ProviderConvOrientation::Column,
    };
    let provider_mode = match mode {
        ConvMode::Full => ProviderConvMode::Full,
        ConvMode::Same => ProviderConvMode::Same,
        ConvMode::Valid => ProviderConvMode::Valid,
    };

    let options = ProviderConv1dOptions {
        mode: provider_mode,
        orientation: provider_orientation,
    };

    let expected = if single_output {
        runmat_accelerate_api::ProviderPrecision::F32
    } else {
        runmat_accelerate_api::ProviderPrecision::F64
    };
    let output_len = match mode {
        ConvMode::Full => lhs_meta
            .len
            .checked_add(rhs_meta.len)
            .and_then(|len| len.checked_sub(1)),
        ConvMode::Same => Some(lhs_meta.len),
        ConvMode::Valid => lhs_meta
            .len
            .checked_sub(rhs_meta.len)
            .and_then(|len| len.checked_add(1)),
    }
    .ok_or_else(|| conv_error(&CONV_ERROR_BUILD_OUTPUT))?;
    let shape = match orientation {
        Orientation::Row => vec![1, output_len],
        Orientation::Column => vec![output_len, 1],
    };
    let contract = gpu_helpers::BinaryGpuOutputContract {
        shape,
        storage: runmat_accelerate_api::GpuTensorStorage::Real,
        precision: Some(expected),
        integer: None,
        logical: false,
        alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
    };
    if !matches!(
        runmat_accelerate_api::ProviderConvolutionInputBackings::from_full_live_handles([
            lhs_handle, rhs_handle,
        ]),
        Ok(Some(_))
    ) {
        return Ok((None, None));
    }
    let placement = super::common::convolution::placement::select_pair(
        runtime,
        BUILTIN_NAME,
        [lhs_handle, rhs_handle],
        mode,
        runmat_accelerate_api::ProviderConvolutionForm::Vector {
            orientation: provider_orientation,
        },
        &contract.shape,
        host_scratch_bound(a, b),
    )?;
    if placement
        .as_ref()
        .is_some_and(|selected| !selected.provider_route())
    {
        return Ok((None, placement));
    }
    if let Some(selected) = placement.as_ref() {
        if !selected.revalidate([lhs_handle, rhs_handle])? {
            return Err(conv_error(&CONV_ERROR_INVALID_INPUT));
        }
    }
    let recorder = placement.as_ref().and_then(|selected| selected.recorder());
    let hook = runmat_accelerate_api::observation::with_provider_observation(recorder, || {
        provider.conv1d(lhs_handle, rhs_handle, options)
    });
    let result = match hook {
        Ok(handle) => super::common::convolution::admit_binary_output(
            handle,
            lhs_handle,
            rhs_handle,
            provider,
            &contract,
            || conv_error(&CONV_ERROR_BUILD_OUTPUT),
        )
        .map(Some),
        Err(err)
            if gpu_helpers::provider_hook_allows_host_fallback(&err) && placement.is_none() =>
        {
            Ok(None)
        }
        Err(err) => {
            let mut failure = crate::runtime_error::with_provider_execution_source(
                conv_error(&CONV_ERROR_BUILD_OUTPUT),
                err.into(),
            );
            failure.gpu_gather_retry = GpuGatherRetry::Never;
            Err(failure)
        }
    };
    if let Some(placement) = placement {
        placement.finish_hook(&result);
    }
    result.map(|value| (value, None))
}

async fn normalize_input(
    value: Value,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<ConvInput> {
    match value {
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                .await
                .map_err(|flow| {
                    let message = flow.message().to_owned();
                    conv_error_with_source(
                        &CONV_ERROR_GATHER_FAILED,
                        message,
                        map_control_flow_with_builtin(flow, BUILTIN_NAME),
                    )
                })?;
            normalize_host_input(gathered, cooperation).await
        }
        other => normalize_host_input(other, cooperation).await,
    }
}

async fn normalize_host_input(
    value: Value,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<ConvInput> {
    let (len, hint) = match &value {
        Value::Tensor(tensor) => {
            let (rows, cols) = shape_rows_cols(tensor.shape());
            (tensor.len(), classify_orientation(rows, cols, tensor.len()))
        }
        Value::ComplexTensor(tensor) => (
            tensor.len(),
            classify_orientation(tensor.rows(), tensor.cols(), tensor.len()),
        ),
        Value::LogicalArray(array) => {
            let (rows, cols) = shape_rows_cols(&array.shape);
            (
                array.data.len(),
                classify_orientation(rows, cols, array.data.len()),
            )
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => {
            (1, OrientationHint::Scalar)
        }
        other => {
            return Err(conv_error_with_detail(
                &CONV_ERROR_INVALID_INPUT,
                format!("{other:?}; expected numeric or logical values"),
            ));
        }
    };
    let data = super::common::convolution::materialize_host_coefficients(
        &value,
        cooperation,
        || conv_error(&CONV_ERROR_INVALID_INPUT),
        conv_allocation_error,
    )
    .await?;
    Ok(ConvInput { data, len, hint })
}

fn classify_orientation(rows: usize, cols: usize, len: usize) -> OrientationHint {
    if rows == 1 && cols != 1 {
        OrientationHint::Row
    } else if cols == 1 && rows != 1 {
        OrientationHint::Column
    } else if rows == 1 && cols == 1 {
        OrientationHint::Scalar
    } else if len == 0 {
        OrientationHint::Empty
    } else {
        OrientationHint::General
    }
}

fn output_orientation(lhs: &ConvInput, rhs: &ConvInput, mode: ConvMode) -> Orientation {
    if matches!(mode, ConvMode::Full) {
        return if matches!(lhs.hint, OrientationHint::Column)
            && matches!(rhs.hint, OrientationHint::Column)
        {
            Orientation::Column
        } else {
            Orientation::Row
        };
    }
    match lhs.hint {
        OrientationHint::Column | OrientationHint::General => Orientation::Column,
        OrientationHint::Row => Orientation::Row,
        OrientationHint::Scalar | OrientationHint::Empty => {
            orientation_from_hints(lhs.hint, rhs.hint)
        }
    }
}

fn orientation_from_hints(lhs: OrientationHint, rhs: OrientationHint) -> Orientation {
    match lhs {
        OrientationHint::Row => Orientation::Row,
        OrientationHint::Column => Orientation::Column,
        OrientationHint::General => Orientation::Column,
        OrientationHint::Scalar | OrientationHint::Empty => match rhs {
            OrientationHint::Column | OrientationHint::General => Orientation::Column,
            OrientationHint::Row => Orientation::Row,
            OrientationHint::Scalar | OrientationHint::Empty => Orientation::Row,
        },
    }
}

fn conv_meta_from_shape(shape: &[usize]) -> ConvInputMeta {
    let len = tensor::element_count(shape);
    let (rows, cols) = shape_rows_cols(shape);
    let hint = classify_orientation(rows, cols, len);
    ConvInputMeta { len, hint }
}

fn shape_rows_cols(shape: &[usize]) -> (usize, usize) {
    if shape.is_empty() {
        (1, 1)
    } else if shape.len() == 1 {
        (1, shape[0])
    } else {
        (shape[0], shape[1])
    }
}

async fn convolve(
    a: &[Complex<f64>],
    b: &[Complex<f64>],
    single_output: bool,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<Complex<f64>>> {
    let len = a
        .len()
        .checked_add(b.len())
        .and_then(|total| total.checked_sub(1))
        .ok_or_else(|| conv_error(&CONV_ERROR_BUILD_OUTPUT))?;
    let mut work = 0_u64;
    if single_output {
        let mut out = Vec::new();
        out.try_reserve_exact(len).map_err(conv_allocation_error)?;
        out.resize(len, Complex::<f32>::new(0.0, 0.0));
        for (i, ai) in a.iter().enumerate() {
            let ai = Complex::<f32>::new(ai.re as f32, ai.im as f32);
            for (j, bj) in b.iter().enumerate() {
                let bj = Complex::<f32>::new(bj.re as f32, bj.im as f32);
                out[i + j] += ai * bj;
                work += 1;
                if work == super::common::SIGNAL_WORK_CHUNK {
                    cooperation.completed(work).await?;
                    work = 0;
                }
            }
        }
        cooperation.completed(work).await?;
        let mut converted = Vec::new();
        converted
            .try_reserve_exact(len)
            .map_err(conv_allocation_error)?;
        for (index, value) in out.into_iter().enumerate() {
            converted.push(Complex::new(f64::from(value.re), f64::from(value.im)));
            if (index + 1).is_multiple_of(super::common::SIGNAL_WORK_CHUNK as usize) {
                cooperation
                    .completed(super::common::SIGNAL_WORK_CHUNK)
                    .await?;
            }
        }
        cooperation
            .completed((len % super::common::SIGNAL_WORK_CHUNK as usize) as u64)
            .await?;
        return Ok(converted);
    }
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(conv_allocation_error)?;
    out.resize(len, Complex::new(0.0, 0.0));
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
            work += 1;
            if work == super::common::SIGNAL_WORK_CHUNK {
                cooperation.completed(work).await?;
                work = 0;
            }
        }
    }
    cooperation.completed(work).await?;
    Ok(out)
}

fn conv_allocation_error(cause: std::collections::TryReserveError) -> RuntimeError {
    let mut builder = build_runtime_error(CONV_ERROR_BUILD_OUTPUT.message)
        .with_builtin(BUILTIN_NAME)
        .with_source(cause)
        .with_gpu_gather_retry(GpuGatherRetry::Never);
    if let Some(identifier) = CONV_ERROR_BUILD_OUTPUT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn apply_mode(
    mut full: Vec<Complex<f64>>,
    mode: ConvMode,
    len_a: usize,
    len_b: usize,
) -> Vec<Complex<f64>> {
    match mode {
        ConvMode::Full => full,
        ConvMode::Same => {
            if len_a == 0 {
                return Vec::new();
            }
            let start = len_b / 2;
            let end = (start + len_a).min(full.len());
            full.copy_within(start..end, 0);
            full.truncate(end - start);
            full
        }
        ConvMode::Valid => {
            if len_a == 0 || len_b == 0 || len_a < len_b {
                return Vec::new();
            }
            let start = len_b - 1;
            let valid_len = len_a - len_b + 1;
            let end = (start + valid_len).min(full.len());
            full.copy_within(start..end, 0);
            full.truncate(end - start);
            full
        }
    }
}

fn convert_output(
    data: Vec<Complex<f64>>,
    orientation: Orientation,
    single_output: bool,
    complex_output: bool,
) -> BuiltinResult<Value> {
    let len = data.len();
    let shape = match (orientation, len) {
        (Orientation::Row, 0) => vec![1, 0],
        (Orientation::Column, 0) => vec![0, 1],
        (Orientation::Row, _) => vec![1, len],
        (Orientation::Column, _) => vec![len, 1],
    };

    let all_real = data.iter().all(|c| c.im == 0.0);
    if all_real && !complex_output {
        if single_output {
            let real_data: Vec<f32> = data.into_iter().map(|c| c.re as f32).collect();
            let tensor = Tensor::from_f32(real_data, shape)
                .map_err(|e| conv_error_with_detail(&CONV_ERROR_BUILD_OUTPUT, &e))?;
            return Ok(tensor::tensor_into_value(tensor));
        }
        let real_data: Vec<f64> = data.into_iter().map(|c| c.re).collect();
        let tensor = Tensor::new(real_data, shape)
            .map_err(|e| conv_error_with_detail(&CONV_ERROR_BUILD_OUTPUT, &e))?;
        return Ok(tensor::tensor_into_value(tensor));
    }

    let storage = if single_output {
        ComplexStorage::F32(
            data.into_iter()
                .map(|c| (c.re as f32, c.im as f32))
                .collect(),
        )
    } else {
        ComplexStorage::F64(data.into_iter().map(|c| (c.re, c.im)).collect())
    };
    let tensor = ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| conv_error_with_detail(&CONV_ERROR_BUILD_COMPLEX_OUTPUT, &e))?;
    if !single_output && tensor.materialize_f64().len() == 1 {
        let (re, im) = tensor.materialize_f64()[0];
        return Ok(Value::Complex(re, im));
    }
    Ok(Value::ComplexTensor(tensor))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use runmat_accelerate_api::{AccelProvider, HostTensorView};
    use runmat_value::{ComplexStorage, IntValue, IntegerStorage, LogicalArray, Tensor};

    fn error_message(error: RuntimeError) -> String {
        error.message().to_string()
    }

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
        Tensor::new_integer(storage, shape).expect("typed integer tensor")
    }

    #[test]
    fn later_invalid_host_vector_rejects_before_resident_download() {
        let _serial = test_support::accel_test_lock();
        let provider =
            super::super::common::convolution::test_support::RejectTransferProvider::new();
        let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let input = provider.input(vec![1.0, 2.0], vec![1, 2]);
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let error = conv_builtin(
            Value::GpuTensor(input.handle.clone()),
            Value::Tensor(matrix),
            Vec::new(),
        )
        .expect_err("matrix cannot be the second vector");
        assert_eq!(error.identifier(), CONV_ERROR_INVALID_INPUT.identifier);
        assert_eq!(provider.downloads(), 0);
        input.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input.probe.retirements(), 0);
    }

    #[test]
    fn later_stale_resident_vector_rejects_before_first_download() {
        let _serial = test_support::accel_test_lock();
        let provider =
            super::super::common::convolution::test_support::RejectTransferProvider::new();
        let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let left = provider.input(vec![1.0, 2.0], vec![1, 2]);
        let right = provider.input(vec![1.0], vec![1, 1]);
        right.probe.mark_unavailable();
        let error = conv_builtin(
            Value::GpuTensor(left.handle.clone()),
            Value::GpuTensor(right.handle.clone()),
            Vec::new(),
        )
        .expect_err("stale second operand");
        assert_eq!(error.gpu_gather_retry(), GpuGatherRetry::Never);
        assert!(error.source.is_some());
        assert_eq!(provider.downloads(), 0);
        left.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(left.probe.retirements(), 0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_full_row_vectors() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let b = Tensor::new(vec![1.0, 1.0, 1.0], vec![1, 3]).unwrap();
        let result = conv_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).expect("conv");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 5]);
                assert_eq!(t.materialize_f64(), vec![1.0, 3.0, 6.0, 5.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_reads_typed_integer_vectors_exactly() {
        let a = integer_tensor(IntegerStorage::I16(vec![1, 2, 3]), vec![1, 3]);
        let b = integer_tensor(IntegerStorage::U16(vec![1, 1, 1]), vec![1, 3]);

        let result = conv_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).expect("conv");

        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 5]);
                assert_eq!(t.materialize_f64(), vec![1.0, 3.0, 6.0, 5.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn conv_accepts_all_integer_classes_and_returns_double() {
        let cases = [
            IntegerStorage::I8(vec![1, 2]),
            IntegerStorage::I16(vec![1, 2]),
            IntegerStorage::I32(vec![1, 2]),
            IntegerStorage::I64(vec![1, 2]),
            IntegerStorage::U8(vec![1, 2]),
            IntegerStorage::U16(vec![1, 2]),
            IntegerStorage::U32(vec![1, 2]),
            IntegerStorage::U64(vec![1, 2]),
        ];
        for storage in cases {
            let input = integer_tensor(storage, vec![1, 2]);
            let output = conv_builtin(
                Value::Tensor(input),
                Value::Int(IntValue::U8(2)),
                Vec::new(),
            )
            .expect("integer conv");
            let Value::Tensor(output) = output else {
                panic!("expected double tensor")
            };
            assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F64);
            assert_eq!(output.materialize_f64(), vec![2.0, 4.0]);
        }
    }

    #[test]
    fn conv_full_orientation_uses_compatibility_target_both_columns_rule() {
        let row = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let col = Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap();
        let Value::Tensor(mixed) =
            conv_builtin(Value::Tensor(row), Value::Tensor(col.clone()), Vec::new()).unwrap()
        else {
            panic!("tensor")
        };
        assert_eq!(mixed.shape(), [1, 3]);
        let Value::Tensor(columns) =
            conv_builtin(Value::Tensor(col.clone()), Value::Tensor(col), Vec::new()).unwrap()
        else {
            panic!("tensor")
        };
        assert_eq!(columns.shape(), [3, 1]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn conv_preserves_rank_zero_and_rank_one_dense_orientation() {
        let kernel = || Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap());
        let Value::Tensor(empty) = conv_builtin(
            Value::Tensor(Tensor::new(vec![2.0], vec![]).unwrap()),
            kernel(),
            vec![Value::from("valid")],
        )
        .expect("rank-zero input is admitted") else {
            panic!("expected empty tensor");
        };
        // A rank-zero scalar has no column orientation; the row kernel owns
        // the empty valid-result orientation.
        assert_eq!(empty.shape(), [1, 0]);
        assert!(empty.materialize_f64().is_empty());

        let Value::Tensor(row) = conv_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2]).unwrap()),
            kernel(),
            vec![Value::from("same")],
        )
        .expect("rank-one input is admitted") else {
            panic!("expected row tensor");
        };
        assert_eq!(row.shape(), [1, 2]);
        assert_eq!(row.materialize_f64(), vec![3.0, 2.0]);
    }

    #[test]
    fn conv_typed_complex_integer_crosses_to_complex_double() {
        let storage = runmat_value::IntegerComplexStorage::new(
            IntegerStorage::I16(vec![1, 2]),
            IntegerStorage::I16(vec![1, -1]),
        )
        .unwrap();
        let tensor = ComplexTensor::from_complex_storage(
            runmat_value::ComplexStorage::Integer(storage),
            vec![1, 2],
        )
        .unwrap();
        let output = conv_builtin(
            Value::ComplexTensor(tensor),
            Value::Int(IntValue::I8(2)),
            Vec::new(),
        )
        .expect("complex integer conv");
        let Value::ComplexTensor(output) = output else {
            panic!("complex tensor")
        };
        assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F64);
        assert_eq!(output.materialize_f64(), vec![(2.0, 2.0), (4.0, -2.0)]);
    }

    #[test]
    fn conv_single_dominates_integer_output_class() {
        let single = Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let integer = integer_tensor(IntegerStorage::U64(vec![2, 3]), vec![1, 2]);
        let Value::Tensor(output) =
            conv_builtin(Value::Tensor(integer), Value::Tensor(single), Vec::new()).unwrap()
        else {
            panic!("tensor")
        };
        assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
    }

    #[test]
    fn conv_resident_integer_gathers_exactly_and_restores_double_residency() {
        test_support::with_test_provider(|provider| {
            let input = integer_tensor(
                IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
                vec![1, 2],
            );
            let handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
            let output = conv_builtin(
                Value::GpuTensor(handle),
                Value::Int(IntValue::U8(1)),
                Vec::new(),
            )
            .unwrap();
            let Value::GpuTensor(handle) = output else {
                panic!("resident output")
            };
            assert_eq!(runmat_accelerate_api::handle_integer_type(&handle), None);
            let gathered =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle))).unwrap();
            let Value::Tensor(gathered) = gathered else {
                panic!("tensor")
            };
            assert_eq!(gathered.numeric_dtype(), runmat_value::NumericDType::F64);
            assert_eq!(
                gathered.materialize_f64(),
                vec![9_007_199_254_740_992.0, 18_446_744_073_709_551_616.0]
            );
        });
    }

    #[test]
    fn conv_complex_single_retains_class_and_explicit_complexity() {
        let input = ComplexTensor::from_complex_storage(
            ComplexStorage::F32(vec![(1.0, 0.0), (2.0, 0.0)].into()),
            vec![1, 2],
        )
        .unwrap();
        let Value::ComplexTensor(output) = conv_builtin(
            Value::ComplexTensor(input),
            Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap()),
            Vec::new(),
        )
        .unwrap() else {
            panic!("explicitly complex tensor")
        };
        assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(output.materialize_f64(), vec![(1.0, 0.0), (2.0, 0.0)]);
    }

    #[test]
    fn conv_resident_complex_gathers_computes_and_restores() {
        test_support::with_test_provider(|provider| {
            let input = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -1.0)], vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &input).unwrap();
            let output = conv_builtin(Value::GpuTensor(handle), Value::Num(2.0), Vec::new())
                .expect("resident complex conv");
            let Value::GpuTensor(handle) = output else {
                panic!("resident complex output")
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&handle),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let Value::ComplexTensor(gathered) =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle))).unwrap()
            else {
                panic!("complex tensor")
            };
            assert_eq!(gathered.materialize_f64(), vec![(2.0, 4.0), (6.0, -2.0)]);
        });
    }

    #[test]
    fn conv_mixed_provider_inputs_gather_independently() {
        let _guard = test_support::accel_test_lock();
        let provider_a: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
            Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
        );
        let provider_b: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
            Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
        );
        unsafe {
            runmat_accelerate_api::register_provider(provider_a);
            runmat_accelerate_api::register_provider(provider_b);
        }
        let a = provider_a
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[1, 2],
            })
            .unwrap();
        let b = provider_b
            .upload(&HostTensorView {
                data: &[1.0, 1.0],
                shape: &[1, 2],
            })
            .unwrap();
        let Value::GpuTensor(output) =
            conv_builtin(Value::GpuTensor(a), Value::GpuTensor(b), Vec::new()).unwrap()
        else {
            panic!("owner-restored output")
        };
        assert_eq!(output.device_id(), provider_a.device_id());
        let gathered = block_on(provider_a.download(&output)).unwrap();
        assert_eq!(gathered.data, vec![1.0, 3.0, 2.0]);
    }

    #[test]
    fn conv_rejects_native_output_with_wrong_single_precision() {
        test_support::with_test_provider(|provider| {
            let a = gpu_helpers::upload_tensor(
                provider,
                &runmat_value::Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap(),
            )
            .unwrap();
            let b = provider
                .upload(&HostTensorView {
                    data: &[1.0, 1.0],
                    shape: &[1, 2],
                })
                .unwrap();
            let wrong_class = provider
                .upload(&HostTensorView {
                    data: &[1.0, 3.0, 2.0],
                    shape: &[1, 3],
                })
                .unwrap();
            let contract = gpu_helpers::BinaryGpuOutputContract {
                shape: vec![1, 3],
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
                precision: Some(runmat_accelerate_api::ProviderPrecision::F32),
                integer: None,
                logical: false,
                alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
            };
            let error = super::super::common::convolution::admit_binary_output(
                wrong_class,
                &a,
                &b,
                provider,
                &contract,
                || conv_error(&CONV_ERROR_BUILD_OUTPUT),
            )
            .expect_err("wrong-class provider output must be rejected");
            assert_eq!(error.identifier(), CONV_ERROR_BUILD_OUTPUT.identifier);
            assert_eq!(error.gpu_gather_retry(), GpuGatherRetry::Never);
        });
    }

    #[test]
    fn conv_rejects_native_result_owned_by_another_provider() {
        let _guard = test_support::accel_test_lock();
        let provider_a: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
            Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
        );
        let provider_b: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
            Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
        );
        unsafe {
            runmat_accelerate_api::register_provider(provider_a);
            runmat_accelerate_api::register_provider(provider_b);
        }
        let wrong_owner = provider_b
            .upload(&HostTensorView {
                data: &[1.0],
                shape: &[1, 1],
            })
            .unwrap();
        let input = provider_a
            .upload(&HostTensorView {
                data: &[1.0],
                shape: &[1, 1],
            })
            .unwrap();
        let contract = gpu_helpers::BinaryGpuOutputContract {
            shape: vec![1, 1],
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
            precision: Some(runmat_accelerate_api::ProviderPrecision::F64),
            integer: None,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
        };
        assert!(!gpu_helpers::binary_gpu_output_matches(
            &wrong_owner,
            &input,
            &input,
            provider_a,
            &contract,
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_same_matches_length() {
        let a = Tensor::new(vec![3.0, 4.0, 5.0, 6.0, 7.0], vec![1, 5]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, -1.0], vec![1, 3]).unwrap();
        let result = conv_builtin(
            Value::Tensor(a),
            Value::Tensor(b.clone()),
            vec![Value::from("same")],
        )
        .expect("conv");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 5]);
                assert_eq!(t.materialize_f64(), vec![4.0, 2.0, 2.0, 2.0, -6.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        let result_column = conv_builtin(
            Value::Tensor(Tensor::new(vec![3.0, 4.0, 5.0, 6.0, 7.0], vec![5, 1]).unwrap()),
            Value::Tensor(b),
            vec![Value::from("same")],
        )
        .expect("conv");
        match result_column {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [5, 1]);
                assert_eq!(t.materialize_f64(), vec![4.0, 2.0, 2.0, 2.0, -6.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn conv_same_even_kernel_uses_matlab_alignment() {
        let signal = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]).unwrap();
        let kernel = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let result = conv_builtin(
            Value::Tensor(signal),
            Value::Tensor(kernel),
            vec![Value::from("same")],
        )
        .expect("conv");
        let Value::Tensor(result) = result else {
            panic!("expected tensor result");
        };
        assert_eq!(result.shape(), [1, 5]);
        assert_eq!(result.materialize_f64(), vec![10.0, 20.0, 30.0, 34.0, 31.0]);
    }

    #[test]
    fn conv_rejects_malformed_native_result_without_replaying_on_host() {
        test_support::with_rejecting_native_result_provider(|provider| {
            let signal = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0, 3.0],
                    shape: &[1, 3],
                })
                .unwrap();
            let kernel = provider
                .upload(&HostTensorView {
                    data: &[1.0, 1.0],
                    shape: &[1, 2],
                })
                .unwrap();
            let error = conv_builtin(
                Value::GpuTensor(signal.clone()),
                Value::GpuTensor(kernel.clone()),
                Vec::new(),
            )
            .expect_err("malformed provider output is terminal");
            assert_eq!(error.identifier(), CONV_ERROR_BUILD_OUTPUT.identifier);
            assert_eq!(error.gpu_gather_retry(), GpuGatherRetry::Never);
            assert_eq!(provider.free_count(), 1);
            signal.allocation_lease().validate_live().unwrap();
            kernel.allocation_lease().validate_live().unwrap();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_valid_empty_when_kernel_longer() {
        let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 1.0, 1.0], vec![1, 3]).unwrap();
        let result = conv_builtin(
            Value::Tensor(a),
            Value::Tensor(b),
            vec![Value::from("valid")],
        )
        .unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn conv_valid_empty_second_vector_returns_zeros_like_first() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let b = Tensor::from_f32(Vec::new(), vec![0, 1]).unwrap();
        let Value::Tensor(output) = conv_builtin(
            Value::Tensor(a),
            Value::Tensor(b),
            vec![Value::from("valid")],
        )
        .unwrap() else {
            panic!("tensor")
        };
        assert_eq!(output.shape(), [3, 1]);
        assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(output.materialize_f64(), vec![0.0, 0.0, 0.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_complex_inputs() {
        let a = Value::Complex(1.0, 2.0);
        let b = Value::Complex(3.0, -1.0);
        let result = conv_builtin(a, b, Vec::new()).expect("conv");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 5.0).abs() < 1e-12);
                assert!((im - 5.0).abs() < 1e-12);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_scalar_times_vector_follows_other_orientation() {
        let scalar = Value::Num(2.0);
        let vec = Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3]).unwrap();
        let result = conv_builtin(scalar, Value::Tensor(vec), Vec::new()).expect("conv");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 3]);
                assert_eq!(t.materialize_f64(), vec![8.0, 10.0, 12.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_handles_empty_inputs_with_row_orientation() {
        let empty_row = Tensor::new(Vec::<f64>::new(), vec![1, 0]).unwrap();
        let kernel = Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap();
        let result =
            conv_builtin(Value::Tensor(empty_row), Value::Tensor(kernel), Vec::new()).unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_handles_empty_inputs_with_column_orientation() {
        let empty_col = Tensor::new(Vec::<f64>::new(), vec![0, 1]).unwrap();
        let kernel = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result =
            conv_builtin(Value::Tensor(empty_col), Value::Tensor(kernel), Vec::new()).unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [0, 1]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_promotes_logical_inputs_to_double() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let kernel = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let result = conv_builtin(
            Value::LogicalArray(logical.clone()),
            Value::Tensor(kernel),
            Vec::new(),
        )
        .expect("conv");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 4]);
                assert_eq!(t.materialize_f64(), vec![1.0, 1.0, 1.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        // ensure logical inputs on RHS follow the same promotion path
        let lhs = Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap();
        let result = conv_builtin(Value::Tensor(lhs), Value::LogicalArray(logical), Vec::new())
            .expect("conv");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 4]);
                assert_eq!(t.materialize_f64(), vec![2.0, 2.0, 2.0, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_rejects_invalid_shape_keyword() {
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let b = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = error_message(
            conv_builtin(
                Value::Tensor(a),
                Value::Tensor(b),
                vec![Value::from("diagonal")],
            )
            .unwrap_err(),
        );
        assert!(err.contains("third argument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_rejects_non_numeric_input() {
        let err = error_message(
            conv_builtin(Value::from("hi"), Value::Num(1.0), Vec::new()).unwrap_err(),
        );
        assert!(err.contains("unsupported input type"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let signal = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
            let kernel = Tensor::new(vec![1.0, 0.0, -1.0], vec![1, 3]).unwrap();
            let host_expected = conv_builtin(
                Value::Tensor(signal.clone()),
                Value::Tensor(kernel.clone()),
                Vec::new(),
            )
            .expect("host conv");

            let sig_view = HostTensorView {
                data: &signal.materialize_f64(),
                shape: signal.shape(),
            };
            let ker_view = HostTensorView {
                data: &kernel.materialize_f64(),
                shape: kernel.shape(),
            };
            let sig_handle = provider.upload(&sig_view).expect("upload signal");
            let ker_handle = provider.upload(&ker_view).expect("upload kernel");
            let gpu_result = conv_builtin(
                Value::GpuTensor(sig_handle),
                Value::GpuTensor(ker_handle),
                Vec::new(),
            )
            .expect("gpu conv");

            let gathered = test_support::gather(gpu_result).expect("gather gpu");
            let expected = test_support::gather(host_expected).expect("gather host");
            assert_eq!(gathered.shape(), expected.shape());
            assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn conv_wgpu_matches_cpu_same_mode() {
        register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu provider");
        let provider = runmat_accelerate_api::provider().expect("provider registry");
        let signal = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]).unwrap();
        let kernel = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();

        let sig_view = HostTensorView {
            data: &signal.materialize_f64(),
            shape: signal.shape(),
        };
        let ker_view = HostTensorView {
            data: &kernel.materialize_f64(),
            shape: kernel.shape(),
        };
        let sig_handle = provider.upload(&sig_view).expect("upload signal");
        let ker_handle = provider.upload(&ker_view).expect("upload kernel");

        let gpu_value = conv_builtin(
            Value::GpuTensor(sig_handle),
            Value::GpuTensor(ker_handle),
            vec![Value::from("same")],
        )
        .expect("gpu conv");

        let gathered_gpu = test_support::gather(gpu_value).expect("gather gpu");
        assert_eq!(gathered_gpu.shape(), [1, 5]);
        assert_eq!(
            gathered_gpu.materialize_f64(),
            vec![10.0, 20.0, 30.0, 34.0, 31.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_same_with_integer_dimension_argument() {
        let signal = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let kernel = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let result = conv_builtin(
            Value::Tensor(signal),
            Value::Tensor(kernel),
            vec![Value::Int(IntValue::I32(1))],
        );
        assert!(result.is_err());
    }

    #[test]
    fn registered_conv_binding_honors_invocation_cancellation() {
        let source = runmat_execution::cancellation::CancellationSource::new();
        let runtime = crate::context::RuntimeContext::with_cancellation(
            std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
            source.token(),
        );
        source.cancel(runmat_execution::CancellationReason::User);
        let error = block_on(super::conv_builtin(
            Some(runtime),
            Value::Num(1.0),
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("cancelled invocation");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(error.gpu_gather_retry(), GpuGatherRetry::Never);
        assert!(error.source.is_some(), "preserve execution-service cause");
    }

    fn conv_builtin(a: Value, b: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::conv_builtin(
            None,
            a,
            b,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }
}
