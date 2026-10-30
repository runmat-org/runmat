//! MATLAB-compatible `conv2` builtin with GPU-aware semantics for RunMat.

#[cfg(test)]
#[path = "conv2/placement_tests.rs"]
mod placement_tests;

use num_complex::Complex;
use runmat_accelerate_api::ProviderConvMode;
use runmat_builtins::{
    BuiltinErrorDescriptor, CONV2_ERROR_ARG_COUNT, CONV2_ERROR_BUILD_COMPLEX_OUTPUT,
    CONV2_ERROR_BUILD_OUTPUT, CONV2_ERROR_GATHER_FAILED, CONV2_ERROR_INVALID_INPUT,
    CONV2_ERROR_MATRIX_REQUIRED, CONV2_ERROR_SHAPE_INVALID, CONV2_ERROR_VECTOR_REQUIRED,
};
use runmat_macros::runtime_builtin;
use runmat_types::ConvolutionMode as Conv2Mode;
use runmat_value::{
    ComplexElement, ComplexStorage, ComplexTensor, HostComplexBuffer, NumericDType, Tensor, Value,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::signal::common::SignalCooperation;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry, RuntimeError};

mod separable;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::conv2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "conv2",
    op_kind: GpuOpKind::Custom("conv2d"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("conv2d")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses a native hook only for same-provider, same-device real floating inputs and accepts its result only when the handle has the required precision, real storage, input device, and provider ownership. Other supported inputs gather through their owners and restore only when class and complexity can be preserved.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::conv2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "conv2",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Currently implemented as a standalone op; future work may add FFT-backed or fused variants.",
};

const BUILTIN_NAME: &str = "conv2";

fn conv2_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    conv2_error_with_message(error.message, error)
}

fn conv2_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    conv2_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn conv2_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn conv2_error_with_source(
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

fn conv2_allocation_error(cause: std::collections::TryReserveError) -> RuntimeError {
    let mut builder = build_runtime_error(CONV2_ERROR_BUILD_OUTPUT.message)
        .with_builtin(BUILTIN_NAME)
        .with_source(cause)
        .with_gpu_gather_retry(GpuGatherRetry::Never);
    if let Some(identifier) = CONV2_ERROR_BUILD_OUTPUT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "conv2",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::conv2"
)]
async fn conv2_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    a: Value,
    b: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let mut extras = rest;
    let mode = extract_mode(&mut extras)?;
    if extras.len() > 1 {
        return Err(conv2_error(&CONV2_ERROR_ARG_COUNT));
    }
    let separable = !extras.is_empty();
    admit_conv2_operand(&a, separable)?;
    admit_conv2_operand(&b, separable)?;
    if let Some(signal) = extras.first() {
        admit_conv2_operand(signal, false)?;
    }
    let mut cooperation = SignalCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    cooperation.completed(0).await?;
    let single_output =
        value_uses_single(&a) || value_uses_single(&b) || extras.iter().any(value_uses_single);
    let complex_output =
        value_is_complex(&a) || value_is_complex(&b) || extras.iter().any(value_is_complex);
    let resident_inputs = std::iter::once(&a)
        .chain(std::iter::once(&b))
        .chain(extras.iter())
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();

    match extras.len() {
        0 => {
            let (device_value, mut placement) =
                try_conv2_gpu(runtime.as_ref(), &a, &b, mode, single_output)?;
            if let Some(device_value) = device_value {
                return Ok(device_value);
            }
            if placement.is_none() {
                if let Some(shape) = host_output_shape(&a, &b, None, mode) {
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
                            "matrix.host",
                            Some(mode),
                            &[&a, &b],
                            &resident_inputs,
                            outputs,
                            host_scratch_bound(&a, &b, None),
                        )?;
                    }
                }
            }
            let restore_recorder = placement.as_ref().and_then(|selected| selected.recorder());
            let result = async move {
                let left = convert_matrix(a, "A", &mut cooperation).await?;
                let right = convert_matrix(b, "B", &mut cooperation).await?;
                if mode == Conv2Mode::Full
                    && !left.is_empty()
                    && !right.is_empty()
                    && left.cols == 1
                    && right.rows == 1
                {
                    let result =
                        outer_product(&left.data, &right.data, single_output, &mut cooperation)
                            .await?;
                    let output =
                        matrix_to_value(result, single_output, complex_output, &mut cooperation)
                            .await?;
                    return restore_to_owner(output, &resident_inputs, restore_recorder.clone());
                }
                let result =
                    conv2_matrices(&left, &right, mode, single_output, &mut cooperation).await?;
                let output =
                    matrix_to_value(result, single_output, complex_output, &mut cooperation)
                        .await?;
                restore_to_owner(output, &resident_inputs, restore_recorder)
            }
            .await;
            if let Some(placement) = placement {
                placement.finish_value(&result);
            }
            result
        }
        1 => {
            let placement = if let Some(shape) = host_output_shape(&a, &b, extras.first(), mode) {
                if let Some(outputs) =
                    super::common::convolution::placement::host_output_representations(
                        &[shape],
                        single_output,
                        complex_output,
                    )
                {
                    super::common::convolution::placement::select_host_variant(
                        runtime.as_ref(),
                        BUILTIN_NAME,
                        "separable.host",
                        Some(mode),
                        &[&a, &b, &extras[0]],
                        &resident_inputs,
                        outputs,
                        host_scratch_bound(&a, &b, extras.first()),
                    )?
                } else {
                    None
                }
            } else {
                None
            };
            let restore_recorder = placement.as_ref().and_then(|selected| selected.recorder());
            let result = async move {
                let signal = convert_matrix(
                    extras.pop_front().expect("validated argument prefix"),
                    "A",
                    &mut cooperation,
                )
                .await?;
                let column = convert_vector(a, "H column", &mut cooperation).await?;
                let row = convert_vector(b, "H row", &mut cooperation).await?;
                let result = separable::convolve_separable(
                    &signal,
                    &column,
                    &row,
                    mode,
                    single_output,
                    &mut cooperation,
                )
                .await?;
                let output =
                    matrix_to_value(result, single_output, complex_output, &mut cooperation)
                        .await?;
                restore_to_owner(output, &resident_inputs, restore_recorder)
            }
            .await;
            if let Some(placement) = placement {
                placement.finish_value(&result);
            }
            result
        }
        _ => Err(conv2_error(&CONV2_ERROR_ARG_COUNT)),
    }
}

fn host_output_shape(
    a: &Value,
    b: &Value,
    signal: Option<&Value>,
    mode: Conv2Mode,
) -> Option<Vec<usize>> {
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
    let (rows, cols, kernel_rows, kernel_cols) = if let Some(signal) = signal {
        let (rows, cols) = conv2_dimensions(shape(signal)?)?;
        let kernel_rows = super::common::convolution::checked_vector_elements(shape(a)?)?;
        let kernel_cols = super::common::convolution::checked_vector_elements(shape(b)?)?;
        (rows, cols, kernel_rows, kernel_cols)
    } else {
        let (rows, cols) = conv2_dimensions(shape(a)?)?;
        let (kernel_rows, kernel_cols) = conv2_dimensions(shape(b)?)?;
        (rows, cols, kernel_rows, kernel_cols)
    };
    let empty = rows == 0 || cols == 0 || kernel_rows == 0 || kernel_cols == 0;
    let result = match mode {
        Conv2Mode::Full if empty => (0, 0),
        Conv2Mode::Full => (
            rows.checked_add(kernel_rows)?.checked_sub(1)?,
            cols.checked_add(kernel_cols)?.checked_sub(1)?,
        ),
        Conv2Mode::Same => (rows, cols),
        Conv2Mode::Valid if empty => (0, 0),
        Conv2Mode::Valid => (
            rows.checked_sub(kernel_rows)
                .and_then(|n| n.checked_add(1))
                .unwrap_or(0),
            cols.checked_sub(kernel_cols)
                .and_then(|n| n.checked_add(1))
                .unwrap_or(0),
        ),
    };
    result.0.checked_mul(result.1)?;
    Some(vec![result.0, result.1])
}

fn host_scratch_bound(
    a: &Value,
    b: &Value,
    signal: Option<&Value>,
) -> runmat_execution::ResourceByteBound {
    let bound = (|| {
        fn shape(value: &Value) -> Option<&[usize]> {
            match value {
                Value::Tensor(tensor) => Some(tensor.shape()),
                Value::ComplexTensor(tensor) => Some(tensor.shape()),
                Value::LogicalArray(array) => Some(&array.shape),
                Value::GpuTensor(handle) => Some(handle.shape()),
                Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => {
                    Some(&[1, 1])
                }
                _ => None,
            }
        }
        let (rows, cols, kr, kc) = if let Some(signal) = signal {
            let (rows, cols) = conv2_dimensions(shape(signal)?)?;
            (
                rows,
                cols,
                super::common::convolution::checked_vector_elements(shape(a)?)?,
                super::common::convolution::checked_vector_elements(shape(b)?)?,
            )
        } else {
            let (rows, cols) = conv2_dimensions(shape(a)?)?;
            let (kr, kc) = conv2_dimensions(shape(b)?)?;
            (rows, cols, kr, kc)
        };
        let (rows, cols, kr, kc) = (
            u64::try_from(rows).ok()?,
            u64::try_from(cols).ok()?,
            u64::try_from(kr).ok()?,
            u64::try_from(kc).ok()?,
        );
        let signal_count = rows.checked_mul(cols)?;
        let kernel_count = kr.checked_mul(kc)?;
        let full_rows = if rows == 0 || kr == 0 {
            0
        } else {
            rows.checked_add(kr)?.checked_sub(1)?
        };
        let full_cols = if cols == 0 || kc == 0 {
            0
        } else {
            cols.checked_add(kc)?.checked_sub(1)?
        };
        let full = full_rows.checked_mul(full_cols)?;
        let input_coefficients = if signal.is_some() {
            signal_count.checked_add(kr)?.checked_add(kc)?
        } else {
            signal_count.checked_add(kernel_count)?
        }
        .checked_mul(16)?;
        // Conservative sum of owned Matrix buffers: full result, optional
        // sliced result, converted output, separable vertical intermediate,
        // balanced factors, and general-kernel outer product.
        let intermediates = full
            .checked_mul(48)?
            .checked_add(kernel_count.checked_mul(16)?)?
            .checked_add(full_rows.checked_mul(cols)?.checked_mul(16)?)?
            .checked_add(kr.checked_add(kc)?.checked_mul(16)?)?;
        let gather =
            [Some(a), Some(b), signal]
                .into_iter()
                .flatten()
                .try_fold(0_u64, |total, value| {
                    let bytes = match value {
                        Value::GpuTensor(handle) => {
                            handle.physical().requested_bytes().checked_mul(3)?
                        }
                        _ => 0,
                    };
                    total.checked_add(bytes)
                })?;
        input_coefficients
            .checked_add(intermediates)?
            .checked_add(gather)
    })();
    bound
        .map(runmat_execution::ResourceByteBound::Known)
        .unwrap_or(runmat_execution::ResourceByteBound::Unavailable(
            runmat_execution::ResourceBoundUnavailable::NoScratchBound,
        ))
}

fn admit_conv2_operand(value: &Value, vector_required: bool) -> BuiltinResult<()> {
    if matches!(value, Value::GpuTensor(_)) {
        super::common::convolution::admit_resident_operand(value, || {
            conv2_error(&CONV2_ERROR_INVALID_INPUT)
        })?;
    }
    let shape = match value {
        Value::Tensor(tensor) => Some(tensor.shape()),
        Value::ComplexTensor(tensor) => Some(tensor.shape()),
        Value::LogicalArray(array) => Some(array.shape.as_slice()),
        Value::GpuTensor(handle) => Some(handle.shape()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => None,
        _ => return Err(conv2_error(&CONV2_ERROR_INVALID_INPUT)),
    };
    let admitted = shape.is_none_or(|shape| {
        if vector_required {
            super::common::convolution::checked_vector_elements(shape).is_some()
        } else {
            super::common::convolution::checked_matrix_dims(shape).is_some()
        }
    });
    if !admitted {
        return Err(conv2_error(if vector_required {
            &CONV2_ERROR_VECTOR_REQUIRED
        } else {
            &CONV2_ERROR_MATRIX_REQUIRED
        }));
    }
    Ok(())
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

fn try_conv2_gpu(
    runtime: Option<&crate::context::RuntimeContext>,
    a: &Value,
    b: &Value,
    mode: Conv2Mode,
    single_output: bool,
) -> BuiltinResult<(
    Option<Value>,
    Option<super::common::convolution::placement::SelectedConvolution>,
)> {
    let (lhs, rhs) = match (a, b) {
        (Value::GpuTensor(lhs), Value::GpuTensor(rhs)) => (lhs, rhs),
        _ => return Ok((None, None)),
    };
    if runmat_accelerate_api::handle_integer_type(lhs).is_some()
        || runmat_accelerate_api::handle_integer_type(rhs).is_some()
        || runmat_accelerate_api::handle_is_logical(lhs)
        || runmat_accelerate_api::handle_is_logical(rhs)
        || value_is_complex(a)
        || value_is_complex(b)
    {
        return Ok((None, None));
    }
    if lhs.device_id() != rhs.device_id() {
        return Ok((None, None));
    }
    let provider = match runmat_accelerate_api::provider_for_handle(lhs) {
        Some(p) => p,
        None => return Ok((None, None)),
    };
    let Some(rhs_provider) = runmat_accelerate_api::provider_for_handle(rhs) else {
        return Ok((None, None));
    };
    if !runmat_accelerate_api::same_provider_instance(provider, rhs_provider) {
        return Ok((None, None));
    }

    let lhs_dims = match conv2_dimensions(lhs.shape()) {
        Some(dims) => dims,
        None => return Ok((None, None)),
    };
    let rhs_dims = match conv2_dimensions(rhs.shape()) {
        Some(dims) => dims,
        None => return Ok((None, None)),
    };

    // Keep the explicit full-mode vector kernel path on the host so it can
    // return an outer-product kernel that matches MATLAB's separable identity.
    if mode == Conv2Mode::Full && lhs_dims.1 == 1 && rhs_dims.0 == 1 {
        return Ok((None, None));
    }

    let empty = lhs_dims.0 == 0 || lhs_dims.1 == 0 || rhs_dims.0 == 0 || rhs_dims.1 == 0;
    let provider_mode = match mode {
        Conv2Mode::Full => ProviderConvMode::Full,
        Conv2Mode::Same => ProviderConvMode::Same,
        Conv2Mode::Valid => ProviderConvMode::Valid,
    };

    let expected = if single_output {
        runmat_accelerate_api::ProviderPrecision::F32
    } else {
        runmat_accelerate_api::ProviderPrecision::F64
    };
    let shape = (|| match mode {
        Conv2Mode::Full if empty => Some((0, 0)),
        Conv2Mode::Full => Some((
            lhs_dims.0.checked_add(rhs_dims.0)?.checked_sub(1)?,
            lhs_dims.1.checked_add(rhs_dims.1)?.checked_sub(1)?,
        )),
        Conv2Mode::Same => Some(lhs_dims),
        Conv2Mode::Valid if empty => Some((0, 0)),
        Conv2Mode::Valid => Some((
            lhs_dims.0.checked_sub(rhs_dims.0)?.checked_add(1)?,
            lhs_dims.1.checked_sub(rhs_dims.1)?.checked_add(1)?,
        )),
    })();
    let Some((rows, cols)) = shape else {
        return Ok((None, None));
    };
    rows.checked_mul(cols)
        .ok_or_else(|| conv2_error(&CONV2_ERROR_BUILD_OUTPUT))?;
    let contract = gpu_helpers::BinaryGpuOutputContract {
        shape: vec![rows, cols],
        storage: runmat_accelerate_api::GpuTensorStorage::Real,
        precision: Some(expected),
        integer: None,
        logical: false,
        alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
    };
    if !matches!(
        runmat_accelerate_api::ProviderConvolutionInputBackings::from_full_live_handles([lhs, rhs]),
        Ok(Some(_))
    ) {
        return Ok((None, None));
    }
    let placement = super::common::convolution::placement::select_pair(
        runtime,
        BUILTIN_NAME,
        [lhs, rhs],
        mode,
        runmat_accelerate_api::ProviderConvolutionForm::Matrix,
        &contract.shape,
        host_scratch_bound(a, b, None),
    )?;
    if placement
        .as_ref()
        .is_some_and(|selected| !selected.provider_route())
    {
        return Ok((None, placement));
    }
    if let Some(selected) = placement.as_ref() {
        if !selected.revalidate([lhs, rhs])? {
            return Err(conv2_error(&CONV2_ERROR_INVALID_INPUT));
        }
    }
    let recorder = placement.as_ref().and_then(|selected| selected.recorder());
    let hook = runmat_accelerate_api::observation::with_provider_observation(recorder, || {
        provider.conv2d(lhs, rhs, provider_mode)
    });
    let result = match hook {
        Ok(handle) => super::common::convolution::admit_binary_output(
            handle,
            lhs,
            rhs,
            provider,
            &contract,
            || conv2_error(&CONV2_ERROR_BUILD_OUTPUT),
        )
        .map(Some),
        Err(err)
            if gpu_helpers::provider_hook_allows_host_fallback(&err) && placement.is_none() =>
        {
            Ok(None)
        }
        Err(err) => {
            let mut failure = crate::runtime_error::with_provider_execution_source(
                conv2_error(&CONV2_ERROR_BUILD_OUTPUT),
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

fn conv2_dimensions(shape: &[usize]) -> Option<(usize, usize)> {
    match shape.len() {
        0 => Some((1, 1)),
        1 => Some((shape[0], 1)),
        2 => Some((shape[0], shape[1])),
        _ => {
            if shape.iter().skip(2).all(|&dim| dim == 1) {
                Some((shape[0], shape[1]))
            } else {
                None
            }
        }
    }
}

#[derive(Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<Complex<f64>>,
}

impl Matrix {
    async fn zeros(
        rows: usize,
        cols: usize,
        cooperation: &mut SignalCooperation,
    ) -> BuiltinResult<Self> {
        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| conv2_error(&CONV2_ERROR_BUILD_OUTPUT))?;
        let mut data = Vec::new();
        data.try_reserve_exact(len)
            .map_err(conv2_allocation_error)?;
        while data.len() < len {
            let chunk = (len - data.len()).min(super::common::SIGNAL_WORK_CHUNK as usize);
            data.resize(data.len() + chunk, Complex::new(0.0, 0.0));
            cooperation.completed(chunk as u64).await?;
        }
        Ok(Self { rows, cols, data })
    }

    fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    #[inline]
    fn index(&self, row: usize, col: usize) -> usize {
        col * self.rows + row
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> Complex<f64> {
        self.data[self.index(row, col)]
    }

    #[inline]
    fn add_assign(&mut self, row: usize, col: usize, value: Complex<f64>) {
        let idx = self.index(row, col);
        self.data[idx] += value;
    }

    async fn slice(
        &self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
        cooperation: &mut SignalCooperation,
    ) -> BuiltinResult<Self> {
        let row_end = row_end.min(self.rows);
        let col_end = col_end.min(self.cols);
        if row_start >= row_end || col_start >= col_end {
            return Self::zeros(
                row_end.saturating_sub(row_start),
                col_end.saturating_sub(col_start),
                cooperation,
            )
            .await;
        }
        let rows = row_end - row_start;
        let cols = col_end - col_start;
        let mut sliced = Self::zeros(rows, cols, cooperation).await?;
        let mut work = 0_u64;
        for c in 0..cols {
            for r in 0..rows {
                let value = self.get(row_start + r, col_start + c);
                sliced.data[c * rows + r] = value;
                work += 1;
                if work == super::common::SIGNAL_WORK_CHUNK {
                    cooperation.completed(work).await?;
                    work = 0;
                }
            }
        }
        cooperation.completed(work).await?;
        Ok(sliced)
    }
}

fn extract_mode(extras: &mut crate::call::BuiltinArguments) -> BuiltinResult<Conv2Mode> {
    if let Some(last) = extras.last() {
        if let Some(mode) = parse_mode_value(last)? {
            extras.pop();
            return Ok(mode);
        }
    }
    if extras.len() == 2 {
        return Err(conv2_error(&CONV2_ERROR_SHAPE_INVALID));
    }
    Ok(Conv2Mode::Full)
}

fn parse_mode_value(value: &Value) -> BuiltinResult<Option<Conv2Mode>> {
    let text =
        match super::common::convolution::convolution_selector_text(value).map_err(|source| {
            let mut error = conv2_error(&CONV2_ERROR_SHAPE_INVALID);
            error.source = Some(Box::new(source));
            error
        })? {
            super::common::convolution::ConvolutionSelectorText::Text(text) => text,
            super::common::convolution::ConvolutionSelectorText::MalformedText => {
                return Err(conv2_error(&CONV2_ERROR_SHAPE_INVALID))
            }
            super::common::convolution::ConvolutionSelectorText::NotText => return Ok(None),
        };
    let mode = Conv2Mode::decode(&text).ok_or_else(|| conv2_error(&CONV2_ERROR_SHAPE_INVALID))?;
    Ok(Some(mode))
}

async fn convert_matrix(
    value: Value,
    arg: &str,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Matrix> {
    match value {
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                .await
                .map_err(|flow| {
                    let message = flow.message().to_owned();
                    conv2_error_with_source(
                        &CONV2_ERROR_GATHER_FAILED,
                        message,
                        map_control_flow_with_builtin(flow, BUILTIN_NAME),
                    )
                })?;
            convert_host_matrix(gathered, arg, cooperation).await
        }
        other => convert_host_matrix(other, arg, cooperation).await,
    }
}

async fn convert_host_matrix(
    value: Value,
    arg: &str,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Matrix> {
    let (rows, cols) = match &value {
        Value::Tensor(tensor) => super::common::convolution::checked_matrix_dims(tensor.shape())
            .ok_or_else(|| conv2_error(&CONV2_ERROR_MATRIX_REQUIRED))?,
        Value::ComplexTensor(tensor) => {
            super::common::convolution::checked_matrix_dims(tensor.shape())
                .ok_or_else(|| conv2_error(&CONV2_ERROR_MATRIX_REQUIRED))?
        }
        Value::LogicalArray(array) => super::common::convolution::checked_matrix_dims(&array.shape)
            .ok_or_else(|| conv2_error(&CONV2_ERROR_MATRIX_REQUIRED))?,
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => (1, 1),
        other => {
            return Err(conv2_error_with_detail(
                &CONV2_ERROR_INVALID_INPUT,
                format!("conv2/{arg}: expected numeric or logical values, got {other:?}"),
            ));
        }
    };
    let data = super::common::convolution::materialize_host_coefficients(
        &value,
        cooperation,
        || conv2_error(&CONV2_ERROR_INVALID_INPUT),
        conv2_allocation_error,
    )
    .await?;
    Ok(Matrix { rows, cols, data })
}

async fn convert_vector(
    value: Value,
    arg: &str,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<Complex<f64>>> {
    let matrix = convert_matrix(value, arg, cooperation).await?;
    if matrix.rows > 1 && matrix.cols > 1 {
        return Err(conv2_error_with_detail(
            &CONV2_ERROR_VECTOR_REQUIRED,
            format!("conv2: {arg} got {}x{}", matrix.rows, matrix.cols),
        ));
    }
    Ok(matrix.data)
}

async fn outer_product(
    column: &[Complex<f64>],
    row: &[Complex<f64>],
    single_output: bool,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Matrix> {
    let rows = column.len();
    let cols = row.len();
    let mut matrix = Matrix::zeros(rows, cols, cooperation).await?;
    let mut work = 0_u64;
    for c in 0..cols {
        for r in 0..rows {
            matrix.data[c * rows + r] = if single_output {
                let lhs = Complex::<f32>::new(column[r].re as f32, column[r].im as f32);
                let rhs = Complex::<f32>::new(row[c].re as f32, row[c].im as f32);
                let value = lhs * rhs;
                Complex::new(f64::from(value.re), f64::from(value.im))
            } else {
                column[r] * row[c]
            };
            work += 1;
            if work == super::common::SIGNAL_WORK_CHUNK {
                cooperation.completed(work).await?;
                work = 0;
            }
        }
    }
    cooperation.completed(work).await?;
    Ok(matrix)
}

async fn conv2_matrices(
    a: &Matrix,
    b: &Matrix,
    mode: Conv2Mode,
    single_output: bool,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Matrix> {
    if a.is_empty() || b.is_empty() {
        return empty_result(a, b, mode, cooperation).await;
    }
    if mode == Conv2Mode::Valid && (a.rows < b.rows || a.cols < b.cols) {
        return Matrix::zeros(0, 0, cooperation).await;
    }

    let rows = a
        .rows
        .checked_add(b.rows)
        .and_then(|n| n.checked_sub(1))
        .ok_or_else(|| conv2_error(&CONV2_ERROR_BUILD_OUTPUT))?;
    let cols = a
        .cols
        .checked_add(b.cols)
        .and_then(|n| n.checked_sub(1))
        .ok_or_else(|| conv2_error(&CONV2_ERROR_BUILD_OUTPUT))?;
    let mut full = Matrix::zeros(rows, cols, cooperation).await?;
    let mut work = 0_u64;
    for ac in 0..a.cols {
        for ar in 0..a.rows {
            let aval = a.get(ar, ac);
            for bc in 0..b.cols {
                let out_c = ac + bc;
                for br in 0..b.rows {
                    let out_r = ar + br;
                    let bval = b.get(br, bc);
                    if single_output {
                        let aval = Complex::<f32>::new(aval.re as f32, aval.im as f32);
                        let bval = Complex::<f32>::new(bval.re as f32, bval.im as f32);
                        let idx = full.index(out_r, out_c);
                        let current = full.data[idx];
                        let current = Complex::<f32>::new(current.re as f32, current.im as f32);
                        let value = current + aval * bval;
                        full.data[idx] = Complex::new(f64::from(value.re), f64::from(value.im));
                    } else {
                        full.add_assign(out_r, out_c, aval * bval);
                    }
                    work += 1;
                    if work == super::common::SIGNAL_WORK_CHUNK {
                        cooperation.completed(work).await?;
                        work = 0;
                    }
                }
            }
        }
    }
    cooperation.completed(work).await?;

    match mode {
        Conv2Mode::Full => Ok(full),
        Conv2Mode::Same => {
            if a.is_empty() {
                return Matrix::zeros(a.rows, a.cols, cooperation).await;
            }
            let row_start = b.rows / 2;
            let col_start = b.cols / 2;
            full.slice(
                row_start,
                row_start + a.rows,
                col_start,
                col_start + a.cols,
                cooperation,
            )
            .await
        }
        Conv2Mode::Valid => {
            let rows = a.rows - b.rows + 1;
            let cols = a.cols - b.cols + 1;
            let row_start = b.rows - 1;
            let col_start = b.cols - 1;
            full.slice(
                row_start,
                row_start + rows,
                col_start,
                col_start + cols,
                cooperation,
            )
            .await
        }
    }
}

async fn empty_result(
    a: &Matrix,
    _b: &Matrix,
    mode: Conv2Mode,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Matrix> {
    match mode {
        Conv2Mode::Full | Conv2Mode::Valid => Matrix::zeros(0, 0, cooperation).await,
        Conv2Mode::Same => Matrix::zeros(a.rows, a.cols, cooperation).await,
    }
}

async fn matrix_to_value(
    matrix: Matrix,
    single_output: bool,
    complex_output: bool,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Value> {
    let rows = matrix.rows;
    let cols = matrix.cols;
    let mut all_real = true;
    for chunk in matrix
        .data
        .chunks(super::common::SIGNAL_WORK_CHUNK as usize)
    {
        all_real &= chunk.iter().all(|c| c.im == 0.0);
        cooperation.completed(chunk.len() as u64).await?;
    }

    if all_real && !complex_output {
        if single_output {
            let real_data = map_matrix_data(matrix.data, cooperation, |c| c.re as f32).await?;
            let tensor = Tensor::from_f32(real_data, vec![rows, cols])
                .map_err(|e| conv2_error_with_detail(&CONV2_ERROR_BUILD_OUTPUT, &e))?;
            return Ok(tensor::tensor_into_value(tensor));
        }
        let real_data = map_matrix_data(matrix.data, cooperation, |c| c.re).await?;
        let tensor = Tensor::new(real_data, vec![rows, cols])
            .map_err(|e| conv2_error_with_detail(&CONV2_ERROR_BUILD_OUTPUT, &e))?;
        return Ok(tensor::tensor_into_value(tensor));
    }

    if !single_output && matrix.data.len() == 1 {
        let value = matrix.data[0];
        return Ok(Value::Complex(value.re, value.im));
    }
    let storage = if single_output {
        let elements = map_matrix_data(matrix.data, cooperation, |c| {
            ComplexElement(c.re as f32, c.im as f32)
        })
        .await?;
        ComplexStorage::F32(HostComplexBuffer::from_elements(elements))
    } else {
        let elements =
            map_matrix_data(matrix.data, cooperation, |c| ComplexElement(c.re, c.im)).await?;
        ComplexStorage::F64(HostComplexBuffer::from_elements(elements))
    };
    let tensor = ComplexTensor::from_complex_storage(storage, vec![rows, cols])
        .map_err(|e| conv2_error_with_detail(&CONV2_ERROR_BUILD_COMPLEX_OUTPUT, &e))?;
    Ok(Value::ComplexTensor(tensor))
}

async fn map_matrix_data<T, F>(
    input: Vec<Complex<f64>>,
    cooperation: &mut SignalCooperation,
    mut convert: F,
) -> BuiltinResult<Vec<T>>
where
    F: FnMut(Complex<f64>) -> T,
{
    let mut output = Vec::new();
    output
        .try_reserve_exact(input.len())
        .map_err(conv2_allocation_error)?;
    for chunk in input.chunks(super::common::SIGNAL_WORK_CHUNK as usize) {
        output.extend(chunk.iter().copied().map(&mut convert));
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(output)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::{tensor, test_support};
    use futures::executor::block_on;
    use runmat_accelerate_api::{AccelProvider, HostTensorView};
    use runmat_value::{ComplexStorage, IntegerStorage, LogicalArray};

    fn error_message(error: RuntimeError) -> String {
        error.message().to_string()
    }

    fn tensor_from_rows(rows: usize, cols: usize, data: &[f64]) -> Tensor {
        assert_eq!(rows * cols, data.len());
        // Convert from row-major (provided for readability) to column-major.
        let mut col_major = vec![0.0; data.len()];
        for r in 0..rows {
            for c in 0..cols {
                col_major[c * rows + r] = data[r * cols + c];
            }
        }
        Tensor::new(col_major, vec![rows, cols]).unwrap()
    }

    fn integer_tensor_from_rows(rows: usize, cols: usize, storage: IntegerStorage) -> Tensor {
        Tensor::new_integer(storage, vec![rows, cols]).unwrap()
    }

    #[test]
    fn later_invalid_matrix_rejects_before_resident_download() {
        let _serial = test_support::accel_test_lock();
        let provider =
            super::super::common::convolution::test_support::RejectTransferProvider::new();
        let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let input = provider.input(vec![1.0, 2.0], vec![2, 1]);
        let nd = Tensor::new(vec![1.0, 2.0], vec![1, 1, 2]).unwrap();
        let error = conv2_builtin(
            Value::GpuTensor(input.handle.clone()),
            Value::Tensor(nd),
            Vec::new(),
        )
        .expect_err("N-D kernel is not a matrix");
        assert_eq!(error.identifier(), CONV2_ERROR_MATRIX_REQUIRED.identifier);
        assert_eq!(provider.downloads(), 0);
        input.handle.allocation_lease().validate_live().unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_full_basic() {
        let a = tensor_from_rows(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = tensor_from_rows(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        let result = conv2_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).expect("conv2");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [3, 3]);
                let expected =
                    tensor_from_rows(3, 3, &[1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0]);
                assert_eq!(t.materialize_f64(), expected.materialize_f64());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_reads_typed_integer_matrices_exactly() {
        let a = integer_tensor_from_rows(2, 2, IntegerStorage::I16(vec![1, 3, 2, 4]));
        let b = integer_tensor_from_rows(1, 1, IntegerStorage::U16(vec![2]));

        let result = conv2_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).expect("conv2");

        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [2, 2]);
                assert_eq!(t.materialize_f64(), vec![2.0, 6.0, 4.0, 8.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn conv2_all_integer_classes_and_separable_form_return_double() {
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
            let a = Tensor::new_integer(storage, vec![2, 1]).unwrap();
            let output = conv2_builtin(
                Value::Tensor(a),
                Value::Int(runmat_value::IntValue::U8(2)),
                Vec::new(),
            )
            .expect("direct integer conv2");
            let Value::Tensor(output) = output else {
                panic!("tensor")
            };
            assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F64);
            assert_eq!(output.materialize_f64(), vec![2.0, 4.0]);
        }
        let u = Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![2, 1]).unwrap();
        let v = Tensor::new_integer(IntegerStorage::U32(vec![1, 3]), vec![1, 2]).unwrap();
        let a = Tensor::new_integer(IntegerStorage::U64(vec![1, 2, 3, 4]), vec![2, 2]).unwrap();
        let Value::Tensor(output) =
            conv2_builtin(Value::Tensor(u), Value::Tensor(v), vec![Value::Tensor(a)])
                .expect("separable integer conv2")
        else {
            panic!("tensor")
        };
        assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F64);
        assert_eq!(output.shape(), [3, 3]);
    }

    #[test]
    fn conv2_single_dominates_integer_output_class() {
        let single = Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let integer = Tensor::new_integer(IntegerStorage::U64(vec![2, 3]), vec![2, 1]).unwrap();
        let Value::Tensor(output) =
            conv2_builtin(Value::Tensor(integer), Value::Tensor(single), Vec::new()).unwrap()
        else {
            panic!("tensor")
        };
        assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
    }

    #[test]
    fn conv2_typed_complex_integer_crosses_to_complex_double() {
        let storage = runmat_value::IntegerComplexStorage::new(
            IntegerStorage::I16(vec![1, 2]),
            IntegerStorage::I16(vec![1, -1]),
        )
        .unwrap();
        let tensor = ComplexTensor::from_complex_storage(
            runmat_value::ComplexStorage::Integer(storage),
            vec![2, 1],
        )
        .unwrap();
        let output = conv2_builtin(
            Value::ComplexTensor(tensor),
            Value::Int(runmat_value::IntValue::I8(2)),
            Vec::new(),
        )
        .unwrap();
        let Value::ComplexTensor(output) = output else {
            panic!("complex tensor")
        };
        assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F64);
        assert_eq!(output.materialize_f64(), vec![(2.0, 2.0), (4.0, -2.0)]);
    }

    #[test]
    fn conv2_resident_integer_gathers_exactly_and_restores_double_residency() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new_integer(
                IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
                vec![2, 1],
            )
            .unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
            let output = conv2_builtin(
                Value::GpuTensor(handle),
                Value::Int(runmat_value::IntValue::U8(1)),
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
    fn conv2_separable_single_is_single_and_matches_exact_values() {
        let hcol = Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let hrow = Tensor::from_f32(vec![1.0, -1.0], vec![1, 2]).unwrap();
        let a = Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let Value::Tensor(output) = conv2_builtin(
            Value::Tensor(hcol),
            Value::Tensor(hrow),
            vec![Value::Tensor(a)],
        )
        .unwrap() else {
            panic!("tensor")
        };
        assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(output.shape(), [3, 2]);
        assert_eq!(
            output.materialize_f64(),
            vec![1.0, 4.0, 4.0, -1.0, -4.0, -4.0]
        );
    }

    #[test]
    fn conv2_complex_single_retains_class_and_explicit_complexity() {
        let input = ComplexTensor::from_complex_storage(
            ComplexStorage::F32(vec![(1.0, 0.0), (2.0, 0.0)].into()),
            vec![2, 1],
        )
        .unwrap();
        let Value::ComplexTensor(output) = conv2_builtin(
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
    fn conv2_resident_complex_gathers_computes_and_restores() {
        test_support::with_test_provider(|provider| {
            let input = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -1.0)], vec![2, 1]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &input).unwrap();
            let output = conv2_builtin(Value::GpuTensor(handle), Value::Num(2.0), Vec::new())
                .expect("resident complex conv2");
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
    fn conv2_mixed_provider_inputs_gather_independently() {
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
                shape: &[2, 1],
            })
            .unwrap();
        let b = provider_b
            .upload(&HostTensorView {
                data: &[2.0],
                shape: &[1, 1],
            })
            .unwrap();
        let Value::GpuTensor(output) =
            conv2_builtin(Value::GpuTensor(a), Value::GpuTensor(b), Vec::new()).unwrap()
        else {
            panic!("owner-restored output")
        };
        assert_eq!(output.device_id(), provider_a.device_id());
        let gathered = block_on(provider_a.download(&output)).unwrap();
        assert_eq!(gathered.data, vec![2.0, 4.0]);
    }

    #[test]
    fn conv2_rejects_native_output_with_wrong_single_precision() {
        test_support::with_test_provider(|provider| {
            let a = gpu_helpers::upload_tensor(
                provider,
                &runmat_value::Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap(),
            )
            .unwrap();
            let b = provider
                .upload(&HostTensorView {
                    data: &[2.0],
                    shape: &[1, 1],
                })
                .unwrap();
            let wrong_class = provider
                .upload(&HostTensorView {
                    data: &[2.0, 4.0],
                    shape: &[2, 1],
                })
                .unwrap();
            let contract = gpu_helpers::BinaryGpuOutputContract {
                shape: vec![2, 1],
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
                || conv2_error(&CONV2_ERROR_BUILD_OUTPUT),
            )
            .expect_err("wrong-class provider output must be rejected");
            assert_eq!(error.identifier(), CONV2_ERROR_BUILD_OUTPUT.identifier);
            assert_eq!(error.gpu_gather_retry(), GpuGatherRetry::Never);
        });
    }

    #[test]
    fn conv2_rejects_native_result_owned_by_another_provider() {
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
    fn conv2_same_matches_reference() {
        let a = tensor_from_rows(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = tensor_from_rows(3, 3, &[1.0; 9]);
        let result = conv2_builtin(
            Value::Tensor(a),
            Value::Tensor(b),
            vec![Value::from("same")],
        )
        .expect("conv2 same");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [3, 3]);
                let expected = tensor_from_rows(
                    3,
                    3,
                    &[12.0, 21.0, 16.0, 27.0, 45.0, 33.0, 24.0, 39.0, 28.0],
                );
                assert_eq!(t.materialize_f64(), expected.materialize_f64());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_same_implements_direct_convolution_formula() {
        let a = tensor_from_rows(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = tensor_from_rows(3, 3, &[1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0]);
        let result = conv2_builtin(
            Value::Tensor(a),
            Value::Tensor(b),
            vec![Value::from("same")],
        )
        .expect("conv2 same");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [3, 3]);
                let expected =
                    tensor_from_rows(3, 3, &[7.0, 4.0, -7.0, 15.0, 6.0, -15.0, 13.0, 4.0, -13.0]);
                assert_eq!(t.materialize_f64(), expected.materialize_f64());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_valid_returns_expected_sum() {
        let a = tensor_from_rows(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = tensor_from_rows(3, 3, &[1.0; 9]);
        let result = conv2_builtin(
            Value::Tensor(a),
            Value::Tensor(b),
            vec![Value::from("valid")],
        )
        .expect("conv2 valid");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 1]);
                assert_eq!(t.materialize_f64(), vec![45.0]);
            }
            Value::Num(n) => assert_eq!(n, 45.0),
            other => panic!("expected scalar 45, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_separable_matches_explicit_kernel() {
        let hcol = tensor_from_rows(3, 1, &[1.0, 2.0, 1.0]);
        let hrow = tensor_from_rows(1, 3, &[1.0, 0.0, -1.0]);
        let signal = tensor_from_rows(3, 3, &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
        let separable = conv2_builtin(
            Value::Tensor(hcol.clone()),
            Value::Tensor(hrow.clone()),
            vec![Value::Tensor(signal.clone()), Value::from("same")],
        )
        .expect("conv2 separable");

        // Build full kernel explicitly and compare.
        let kernel = {
            let h_matrix =
                conv2_builtin(Value::Tensor(hcol), Value::Tensor(hrow), Vec::new()).unwrap();
            match h_matrix {
                Value::Tensor(t) => Value::Tensor(t),
                other => panic!("expected tensor kernel, got {other:?}"),
            }
        };
        let explicit =
            conv2_builtin(kernel, Value::Tensor(signal), vec![Value::from("same")]).unwrap();
        assert_eq!(separable, explicit);
    }

    #[test]
    fn conv2_separable_numeric_and_logical_scalar_signal_are_not_shape_text() {
        let column = tensor_from_rows(2, 1, &[1.0, 2.0]);
        let row = tensor_from_rows(1, 2, &[1.0, 3.0]);
        for (signal, expected) in [
            (Value::Num(2.0), vec![2.0, 4.0, 6.0, 12.0]),
            (Value::Bool(true), vec![1.0, 2.0, 3.0, 6.0]),
        ] {
            let output = conv2_builtin(
                Value::Tensor(column.clone()),
                Value::Tensor(row.clone()),
                vec![signal],
            )
            .expect("scalar separable signal");
            let output = test_support::gather(output).expect("host result");
            assert_eq!(output.shape(), [2, 2]);
            assert_eq!(output.materialize_f64(), expected);
        }
        let error = conv2_builtin(
            Value::Tensor(column),
            Value::Tensor(row),
            vec![Value::from("circular")],
        )
        .expect_err("malformed text remains a shape error");
        assert_eq!(error.identifier(), CONV2_ERROR_SHAPE_INVALID.identifier);
        for invalid_shape in [Value::Num(7.0), Value::Bool(true)] {
            let error = conv2_builtin(
                Value::Tensor(tensor_from_rows(2, 1, &[1.0, 2.0])),
                Value::Tensor(tensor_from_rows(1, 2, &[1.0, 3.0])),
                vec![Value::Num(1.0), invalid_shape],
            )
            .expect_err("fourth input must be shape text");
            assert_eq!(error.identifier(), CONV2_ERROR_SHAPE_INVALID.identifier);
        }
        let nd_text = runmat_value::CharArray::from_unicode_scalars_with_shape(
            "same".chars().collect(),
            vec![1, 4, 1],
        )
        .expect("N-D character shape");
        let error = conv2_builtin(
            Value::Tensor(tensor_from_rows(2, 1, &[1.0, 2.0])),
            Value::Tensor(tensor_from_rows(1, 2, &[1.0, 3.0])),
            vec![Value::CharArray(nd_text)],
        )
        .expect_err("N-D character selector");
        assert_eq!(error.identifier(), CONV2_ERROR_SHAPE_INVALID.identifier);
    }

    #[test]
    fn conv2_empty_direct_full_does_not_take_outer_product_shortcut() {
        let left = Tensor::new(Vec::new(), vec![0, 1]).expect("empty column");
        let right = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).expect("row");
        let output = conv2_builtin(Value::Tensor(left), Value::Tensor(right), Vec::new())
            .expect("empty full result");
        let output = test_support::gather(output).expect("host result");
        assert_eq!(output.shape(), [0, 0]);
        assert!(output.materialize_f64().is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_complex_scaling() {
        let tensor = tensor_from_rows(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let expected_data = tensor.materialize_f64().clone();
        let result =
            conv2_builtin(Value::Tensor(tensor), Value::Complex(0.0, 2.0), Vec::new()).unwrap();
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape(), [2, 2]);
                for (idx, &(re, im)) in t.materialize_f64().iter().enumerate() {
                    assert_eq!(re, 0.0);
                    assert_eq!(im, 2.0 * expected_data[idx]);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_empty_inputs_follow_shape_rules() {
        let empty = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let kernel = tensor_from_rows(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        let result = conv2_builtin(
            Value::Tensor(empty.clone()),
            Value::Tensor(kernel.clone()),
            Vec::new(),
        )
        .unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [0, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }

        let same = conv2_builtin(
            Value::Tensor(empty.clone()),
            Value::Tensor(kernel),
            vec![Value::from("same")],
        )
        .unwrap();
        match same {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [0, 3]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_rejects_invalid_shape_keyword() {
        let a = tensor_from_rows(1, 1, &[1.0]);
        let b = tensor_from_rows(1, 1, &[1.0]);
        let err = error_message(
            conv2_builtin(
                Value::Tensor(a),
                Value::Tensor(b),
                vec![Value::from("diagonal")],
            )
            .unwrap_err(),
        );
        assert!(err.contains("shape argument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_promotes_logical_inputs() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let kernel = tensor_from_rows(1, 2, &[1.0, -1.0]);

        let logical_result = conv2_builtin(
            Value::LogicalArray(logical.clone()),
            Value::Tensor(kernel.clone()),
            Vec::new(),
        )
        .expect("conv2 logical");

        let numeric_tensor = tensor::logical_to_tensor(&logical).unwrap();
        let numeric_result = conv2_builtin(
            Value::Tensor(numeric_tensor),
            Value::Tensor(kernel),
            Vec::new(),
        )
        .expect("conv2 numeric");

        let logical_tensor = test_support::gather(logical_result).expect("gather logical");
        let numeric_tensor = test_support::gather(numeric_result).expect("gather numeric");

        assert_eq!(logical_tensor.shape(), numeric_tensor.shape());
        assert_eq!(
            logical_tensor.materialize_f64(),
            numeric_tensor.materialize_f64()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_same_even_kernel_alignment() {
        let a = tensor_from_rows(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = tensor_from_rows(2, 2, &[1.0, 2.0, 3.0, 4.0]);

        let result = conv2_builtin(
            Value::Tensor(a),
            Value::Tensor(b),
            vec![Value::from("same")],
        )
        .expect("conv2 same even");

        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [3, 3]);
                let expected = tensor_from_rows(
                    3,
                    3,
                    &[
                        23.0, 33.0, 24.0, //
                        53.0, 63.0, 42.0, //
                        52.0, 59.0, 36.0,
                    ],
                );
                assert_eq!(t.materialize_f64(), expected.materialize_f64());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let a = tensor_from_rows(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
            let b = tensor_from_rows(2, 2, &[1.0, 0.0, 0.0, -1.0]);

            let a_view = HostTensorView {
                data: &a.materialize_f64(),
                shape: a.shape(),
            };
            let b_view = HostTensorView {
                data: &b.materialize_f64(),
                shape: b.shape(),
            };

            let ah = provider.upload(&a_view).unwrap();
            let bh = provider.upload(&b_view).unwrap();

            let gpu_result = conv2_builtin(
                Value::GpuTensor(ah),
                Value::GpuTensor(bh),
                vec![Value::from("same")],
            )
            .unwrap();
            let gathered = test_support::gather(gpu_result).unwrap();

            let cpu_result = conv2_builtin(
                Value::Tensor(a),
                Value::Tensor(b),
                vec![Value::from("same")],
            )
            .unwrap();
            let cpu_tensor = test_support::gather(cpu_result).unwrap();

            assert_eq!(gathered.shape(), cpu_tensor.shape());
            assert_eq!(gathered.materialize_f64(), cpu_tensor.materialize_f64());
        });
    }

    #[test]
    fn conv2_rejects_malformed_native_result_without_replaying_on_host() {
        test_support::with_rejecting_native_result_provider(|provider| {
            let signal = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0, 3.0, 4.0],
                    shape: &[2, 2],
                })
                .unwrap();
            let kernel = provider
                .upload(&HostTensorView {
                    data: &[1.0],
                    shape: &[1, 1],
                })
                .unwrap();
            let error = conv2_builtin(
                Value::GpuTensor(signal.clone()),
                Value::GpuTensor(kernel.clone()),
                Vec::new(),
            )
            .expect_err("malformed provider output is terminal");
            assert_eq!(error.identifier(), CONV2_ERROR_BUILD_OUTPUT.identifier);
            assert_eq!(error.gpu_gather_retry(), GpuGatherRetry::Never);
            assert_eq!(provider.free_count(), 1);
            signal.allocation_lease().validate_live().unwrap();
            kernel.allocation_lease().validate_live().unwrap();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn conv2_wgpu_fallback_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let a = tensor_from_rows(
            4,
            4,
            &[
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0,
            ],
        );
        let b = tensor_from_rows(
            3,
            3,
            &[
                1.0, 0.0, -1.0, //
                2.0, 0.0, -2.0, //
                1.0, 0.0, -1.0,
            ],
        );

        let a_view = HostTensorView {
            data: &a.materialize_f64(),
            shape: a.shape(),
        };
        let b_view = HostTensorView {
            data: &b.materialize_f64(),
            shape: b.shape(),
        };

        let a_handle = provider.upload(&a_view).expect("upload A");
        let b_handle = provider.upload(&b_view).expect("upload B");

        let unsupported = provider
            .conv2d(&a_handle, &b_handle, ProviderConvMode::Valid)
            .expect_err("WGPU advertises no conv2d hook");
        assert!(runmat_accelerate_api::is_unsupported_provider_operation(
            &unsupported
        ));
        a_handle.allocation_lease().validate_live().unwrap();
        b_handle.allocation_lease().validate_live().unwrap();

        let gpu_value = conv2_builtin(
            Value::GpuTensor(a_handle),
            Value::GpuTensor(b_handle),
            vec![Value::from("valid")],
        )
        .expect("conv2 gpu");
        let gpu_tensor = test_support::gather(gpu_value).expect("gather gpu");

        let cpu_value = conv2_builtin(
            Value::Tensor(a.clone()),
            Value::Tensor(b.clone()),
            vec![Value::from("valid")],
        )
        .expect("conv2 cpu");
        let cpu_tensor = test_support::gather(cpu_value).expect("gather cpu");

        assert_eq!(gpu_tensor.shape(), cpu_tensor.shape());
        assert_eq!(gpu_tensor.materialize_f64(), cpu_tensor.materialize_f64());
    }

    fn conv2_builtin(a: Value, b: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::conv2_builtin(
            None,
            a,
            b,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    #[test]
    fn registered_conv2_binding_honors_invocation_cancellation() {
        let source = runmat_execution::cancellation::CancellationSource::new();
        let runtime = crate::context::RuntimeContext::with_cancellation(
            std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
            source.token(),
        );
        source.cancel(runmat_execution::CancellationReason::User);
        let error = block_on(super::conv2_builtin(
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
}
