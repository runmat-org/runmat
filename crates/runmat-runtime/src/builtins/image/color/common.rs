use std::borrow::Cow;

use runmat_accelerate_api::{
    AccelProvider, GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage, IntegerElementType,
    ProviderPrecision,
};
use runmat_builtins::{BuiltinIntegerClass, NumericDType, NumericScalar, Tensor, Type, Value};

use crate::builtins::common::{map_control_flow_with_builtin, tensor};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColorLayout {
    Truecolor { rows: usize, cols: usize },
    Colormap { rows: usize },
}

pub(crate) const DOCUMENTED_IMAGE_INTEGER_CLASSES: [BuiltinIntegerClass; 2] =
    [BuiltinIntegerClass::Uint8, BuiltinIntegerClass::Uint16];

pub(crate) const REJECTED_IMAGE_INTEGER_CLASSES: [BuiltinIntegerClass; 6] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Int64,
    BuiltinIntegerClass::Uint32,
    BuiltinIntegerClass::Uint64,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StackedColorLayout {
    plane_pixels: usize,
    stacks: usize,
    shape: Vec<usize>,
}

impl StackedColorLayout {
    pub(crate) fn pixels(&self) -> usize {
        self.plane_pixels.saturating_mul(self.stacks)
    }

    pub(crate) fn output_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    pub(crate) fn index(&self, pixel: usize, channel: usize) -> usize {
        let stack = pixel / self.plane_pixels;
        let within_stack = pixel % self.plane_pixels;
        within_stack + self.plane_pixels * (channel + 3 * stack)
    }
}

impl ColorLayout {
    pub(crate) fn pixels(self) -> usize {
        match self {
            ColorLayout::Truecolor { rows, cols } => rows * cols,
            ColorLayout::Colormap { rows } => rows,
        }
    }

    pub(crate) fn output_shape(self) -> Vec<usize> {
        match self {
            ColorLayout::Truecolor { rows, cols } => vec![rows, cols, 3],
            ColorLayout::Colormap { rows } => vec![rows, 3],
        }
    }

    pub(crate) fn index(self, pixel: usize, channel: usize) -> usize {
        pixel + self.pixels() * channel
    }
}

pub(crate) fn builtin_error(name: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

pub(crate) fn map_flow(name: &'static str, err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, name)
}

pub(crate) async fn gather_value(name: &'static str, value: &Value) -> BuiltinResult<Value> {
    gather_if_needed_async(value)
        .await
        .map_err(|err| map_flow(name, err))
}

#[derive(Clone)]
struct ResidentMetadata {
    logical: bool,
    transpose: Option<runmat_accelerate_api::TransposeInfo>,
    class_name: Option<String>,
}

pub(crate) struct ResidentInputGuard {
    entries: Vec<(GpuTensorHandle, ResidentMetadata)>,
}

impl ResidentInputGuard {
    pub(crate) fn restore(&self) {
        for (handle, metadata) in &self.entries {
            metadata.restore(handle);
        }
    }
}

impl Drop for ResidentInputGuard {
    fn drop(&mut self) {
        self.restore();
    }
}

pub(crate) fn protect_resident_inputs(sources: &[GpuTensorHandle]) -> ResidentInputGuard {
    ResidentInputGuard {
        entries: sources
            .iter()
            .cloned()
            .map(|handle| {
                let metadata = ResidentMetadata::capture(&handle);
                (handle, metadata)
            })
            .collect(),
    }
}

impl ResidentMetadata {
    fn capture(handle: &GpuTensorHandle) -> Self {
        Self {
            logical: runmat_accelerate_api::handle_is_logical(handle),
            transpose: runmat_accelerate_api::handle_transpose_info(handle),
            class_name: runmat_accelerate_api::handle_class_name(handle),
        }
    }

    fn restore(&self, handle: &GpuTensorHandle) {
        runmat_accelerate_api::set_handle_logical(handle, self.logical);
        match self.transpose {
            Some(info) => runmat_accelerate_api::record_handle_transpose(
                handle,
                info.base_rows,
                info.base_cols,
            ),
            None => runmat_accelerate_api::clear_handle_transpose(handle),
        }
        match self.class_name.as_ref() {
            Some(value) => runmat_accelerate_api::set_handle_class_name(handle, value.clone()),
            None => runmat_accelerate_api::clear_handle_class_name(handle),
        }
        runmat_accelerate_api::mark_residency(handle);
    }
}

fn exact_owner(
    handle: &GpuTensorHandle,
    name: &'static str,
) -> BuiltinResult<&'static dyn AccelProvider> {
    let owner = runmat_accelerate_api::provider_for_handle(handle).ok_or_else(|| {
        builtin_error(
            name,
            format!("{name}: no acceleration provider owns the input handle"),
        )
    })?;
    if owner.device_id() != handle.device_id {
        return Err(builtin_error(
            name,
            format!(
                "{name}: input handle device {} is not owned by the selected provider device {}",
                handle.device_id,
                owner.device_id()
            ),
        ));
    }
    Ok(owner)
}

pub(crate) fn restore_resident_numeric_result_for_sources(
    sources: &[GpuTensorHandle],
    value: Value,
    name: &'static str,
) -> BuiltinResult<Value> {
    let Some(first_source) = sources.first() else {
        return Ok(value);
    };
    let source = sources
        .iter()
        .find(|handle| runmat_accelerate_api::handle_is_explicit(handle))
        .unwrap_or(first_source);
    for input in sources {
        exact_owner(input, name)?;
    }
    let owner = exact_owner(source, name)?;
    let snapshots: Vec<_> = sources
        .iter()
        .map(|handle| ResidentMetadata::capture(handle))
        .collect();
    let tensor = match tensor::value_into_tensor_for(name, value.clone()) {
        Ok(tensor) => tensor,
        Err(_) => {
            if runmat_accelerate_api::handle_is_explicit(source) {
                return Err(builtin_error(
                    name,
                    format!("{name}: explicit gpuArray result cannot remain resident"),
                ));
            }
            return Ok(value);
        }
    };
    let expected_integer = tensor.integer_storage().map(|storage| match storage {
        runmat_builtins::IntegerStorage::I8(_) => IntegerElementType::I8,
        runmat_builtins::IntegerStorage::I16(_) => IntegerElementType::I16,
        runmat_builtins::IntegerStorage::I32(_) => IntegerElementType::I32,
        runmat_builtins::IntegerStorage::I64(_) => IntegerElementType::I64,
        runmat_builtins::IntegerStorage::U8(_) => IntegerElementType::U8,
        runmat_builtins::IntegerStorage::U16(_) => IntegerElementType::U16,
        runmat_builtins::IntegerStorage::U32(_) => IntegerElementType::U32,
        runmat_builtins::IntegerStorage::U64(_) => IntegerElementType::U64,
    });
    let expected_precision = expected_integer
        .is_none()
        .then_some(match tensor.numeric_dtype() {
            NumericDType::F32 => ProviderPrecision::F32,
            _ => ProviderPrecision::F64,
        });
    if expected_precision.is_some_and(|precision| precision != owner.precision()) {
        if runmat_accelerate_api::handle_is_explicit(source) {
            return Err(builtin_error(
                name,
                format!(
                    "{name}: explicit gpuArray result requires {expected_precision:?}, but its owner uses {:?}",
                    owner.precision()
                ),
            ));
        }
        return Ok(value);
    }
    let mut output = match crate::builtins::common::gpu_helpers::upload_tensor(owner, &tensor) {
        Ok(output) => output,
        Err(error) => {
            for (input, snapshot) in sources.iter().zip(&snapshots) {
                snapshot.restore(input);
            }
            return Err(builtin_error(
                name,
                format!("{name}: failed to restore GPU result: {error}"),
            ));
        }
    };
    let aliases_input = sources
        .iter()
        .any(|input| output.device_id == input.device_id && output.buffer_id == input.buffer_id);
    if aliases_input {
        for (input, snapshot) in sources.iter().zip(&snapshots) {
            snapshot.restore(input);
        }
        return Err(builtin_error(
            name,
            format!("{name}: provider aliased a protected input while restoring the result"),
        ));
    }
    let valid = output.shape == tensor.shape
        && output.device_id == owner.device_id()
        && runmat_accelerate_api::provider_for_handle(&output).is_some_and(|candidate| {
            candidate.device_id() == output.device_id && std::ptr::eq(candidate, owner)
        })
        && runmat_accelerate_api::handle_storage(&output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(&output) == expected_integer
        && expected_precision.is_none_or(|precision| {
            runmat_accelerate_api::handle_precision(&output) == Some(precision)
        })
        && !runmat_accelerate_api::handle_is_logical(&output)
        && crate::builtins::common::gpu_helpers::gpu_class_metadata_matches(
            &output,
            expected_precision,
            expected_integer,
            false,
        );
    if !valid {
        if let Some(actual_owner) = runmat_accelerate_api::provider_for_handle(&output)
            .filter(|candidate| candidate.device_id() == output.device_id)
        {
            if actual_owner.free(&output).is_ok() {
                runmat_accelerate_api::clear_handle_metadata(&output);
            }
        }
        for (input, snapshot) in sources.iter().zip(&snapshots) {
            snapshot.restore(input);
        }
        return Err(builtin_error(
            name,
            format!("{name}: provider returned an invalid restored result"),
        ));
    }
    for (input, snapshot) in sources.iter().zip(&snapshots) {
        snapshot.restore(input);
    }
    runmat_accelerate_api::set_handle_provenance(
        &mut output,
        runmat_accelerate_api::handle_provenance(source).unwrap_or(GpuHandleProvenance::Automatic),
    );
    runmat_accelerate_api::mark_residency(&output);
    Ok(Value::GpuTensor(output))
}

pub(crate) fn restore_resident_numeric_result(
    source: Option<&GpuTensorHandle>,
    value: Value,
    name: &'static str,
) -> BuiltinResult<Value> {
    match source {
        Some(source) => {
            restore_resident_numeric_result_for_sources(std::slice::from_ref(source), value, name)
        }
        None => Ok(value),
    }
}

pub(crate) fn resident_numeric_dtype(
    handle: &GpuTensorHandle,
    name: &'static str,
) -> BuiltinResult<NumericDType> {
    let owner = exact_owner(handle, name)?;
    if runmat_accelerate_api::handle_is_logical(handle)
        || runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real
    {
        return Err(builtin_error(
            name,
            format!("{name}: expected a real numeric gpuArray"),
        ));
    }
    if let Some(integer) = runmat_accelerate_api::handle_integer_type(handle) {
        let dtype = match integer {
            IntegerElementType::I8 => NumericDType::I8,
            IntegerElementType::I16 => NumericDType::I16,
            IntegerElementType::I32 => NumericDType::I32,
            IntegerElementType::I64 => NumericDType::I64,
            IntegerElementType::U8 => NumericDType::U8,
            IntegerElementType::U16 => NumericDType::U16,
            IntegerElementType::U32 => NumericDType::U32,
            IntegerElementType::U64 => NumericDType::U64,
        };
        validate_class_name(handle, dtype, name)?;
        return Ok(dtype);
    }
    let precision = runmat_accelerate_api::handle_precision(handle).ok_or_else(|| {
        builtin_error(
            name,
            format!("{name}: GPU precision metadata is unavailable"),
        )
    })?;
    if precision != owner.precision() {
        return Err(builtin_error(
            name,
            format!(
                "{name}: GPU precision metadata {precision:?} does not match owner precision {:?}",
                owner.precision()
            ),
        ));
    }
    let dtype = match precision {
        ProviderPrecision::F32 => NumericDType::F32,
        ProviderPrecision::F64 => NumericDType::F64,
    };
    validate_class_name(handle, dtype, name)?;
    Ok(dtype)
}

fn validate_class_name(
    handle: &GpuTensorHandle,
    dtype: NumericDType,
    name: &'static str,
) -> BuiltinResult<()> {
    if let Some(class_name) = runmat_accelerate_api::handle_class_name(handle) {
        if class_name != dtype.class_name() {
            return Err(builtin_error(
                name,
                format!(
                    "{name}: GPU class metadata {class_name} conflicts with {} storage",
                    dtype.class_name()
                ),
            ));
        }
    }
    Ok(())
}

pub(crate) async fn gather_tensor(name: &'static str, value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_value(name, &value).await?;
    tensor::value_into_tensor_for(name, gathered).map_err(|err| builtin_error(name, err))
}

pub(crate) fn tensor_values_f64(tensor: &Tensor) -> Cow<'_, [f64]> {
    tensor::tensor_values_f64_cow(tensor)
}

pub(crate) fn image_value_from_tensor(tensor: Tensor) -> Value {
    if tensor.len() == 1 {
        match tensor
            .numeric_value_at(0)
            .expect("one-element image tensor has one numeric value")
        {
            NumericScalar::F64(value) => Value::Num(value),
            NumericScalar::F32(value) => Value::Num(f64::from(value)),
            value => Value::Int(
                value
                    .into_int_value()
                    .expect("non-floating numeric scalar is integer"),
            ),
        }
    } else {
        Value::Tensor(tensor)
    }
}

pub(crate) fn tensor_with_dtype(
    data: Vec<f64>,
    shape: Vec<usize>,
    dtype: NumericDType,
    name: &'static str,
) -> BuiltinResult<Tensor> {
    let tensor = Tensor::new(data, shape).map_err(|err| builtin_error(name, err))?;
    Ok(tensor::coerce_tensor_dtype(tensor, dtype))
}

pub(crate) fn color_layout(tensor: &Tensor, name: &'static str) -> BuiltinResult<ColorLayout> {
    let shape = &tensor.shape;
    if shape.len() == 3 && shape[2] == 3 {
        let rows = shape[0];
        let cols = shape[1];
        if rows == 0 || cols == 0 {
            return Err(builtin_error(
                name,
                format!("{name}: RGB image must be non-empty"),
            ));
        }
        return Ok(ColorLayout::Truecolor { rows, cols });
    }
    if shape.len() == 2 && shape[1] == 3 {
        let rows = shape[0];
        if rows == 0 {
            return Err(builtin_error(
                name,
                format!("{name}: colormap must be non-empty"),
            ));
        }
        return Ok(ColorLayout::Colormap { rows });
    }
    Err(builtin_error(
        name,
        format!("{name}: expected an MxNx3 RGB image or an Nx3 colormap"),
    ))
}

pub(crate) fn stacked_color_layout(
    tensor: &Tensor,
    name: &'static str,
) -> BuiltinResult<StackedColorLayout> {
    if tensor.shape.len() == 2 && tensor.shape[1] == 3 && tensor.shape[0] > 0 {
        return Ok(StackedColorLayout {
            plane_pixels: tensor.shape[0],
            stacks: 1,
            shape: tensor.shape.clone(),
        });
    }
    if tensor.shape.len() >= 3 && tensor.shape[0] > 0 && tensor.shape[1] > 0 && tensor.shape[2] == 3
    {
        let plane_pixels = tensor.shape[0]
            .checked_mul(tensor.shape[1])
            .ok_or_else(|| builtin_error(name, format!("{name}: RGB image dimensions overflow")))?;
        let stacks = tensor.shape[3..]
            .iter()
            .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
            .ok_or_else(|| builtin_error(name, format!("{name}: RGB image dimensions overflow")))?;
        return Ok(StackedColorLayout {
            plane_pixels,
            stacks,
            shape: tensor.shape.clone(),
        });
    }
    Err(builtin_error(
        name,
        format!("{name}: expected an MxNx3xP RGB image stack or an Nx3 colormap"),
    ))
}

pub(crate) fn grayscale_shape(
    tensor: &Tensor,
    name: &'static str,
) -> BuiltinResult<(usize, usize)> {
    let shape = &tensor.shape;
    if shape.len() == 2 {
        return Ok((shape[0], shape[1]));
    }
    Err(builtin_error(
        name,
        format!("{name}: expected an MxN grayscale image"),
    ))
}

pub(crate) fn unit_value(value: f64, dtype: NumericDType) -> f64 {
    match dtype {
        NumericDType::I8 => (value - i8::MIN as f64) / (u8::MAX as f64),
        NumericDType::I16 => (value - i16::MIN as f64) / (u16::MAX as f64),
        NumericDType::I32 => (value - i32::MIN as f64) / (u32::MAX as f64),
        NumericDType::I64 => (value - i64::MIN as f64) / (u64::MAX as f64),
        NumericDType::U8 => value / 255.0,
        NumericDType::U16 => value / 65535.0,
        NumericDType::U32 => value / (u32::MAX as f64),
        NumericDType::U64 => value / (u64::MAX as f64),
        NumericDType::F32 | NumericDType::F64 => value,
    }
}

pub(crate) fn unit_to_dtype(value: f64, dtype: NumericDType) -> f64 {
    match dtype {
        NumericDType::I8 => (value * u8::MAX as f64 + i8::MIN as f64)
            .round()
            .clamp(i8::MIN as f64, i8::MAX as f64),
        NumericDType::I16 => (value * u16::MAX as f64 + i16::MIN as f64)
            .round()
            .clamp(i16::MIN as f64, i16::MAX as f64),
        NumericDType::I32 => (value * u32::MAX as f64 + i32::MIN as f64)
            .round()
            .clamp(i32::MIN as f64, i32::MAX as f64),
        NumericDType::I64 => (value * u64::MAX as f64 + i64::MIN as f64)
            .round()
            .clamp(i64::MIN as f64, i64::MAX as f64),
        NumericDType::U8 => clamp_round(value * 255.0, 255.0),
        NumericDType::U16 => clamp_round(value * 65535.0, 65535.0),
        NumericDType::U32 => clamp_round(value * (u32::MAX as f64), u32::MAX as f64),
        NumericDType::U64 => clamp_round(value * u64::MAX as f64, u64::MAX as f64),
        NumericDType::F32 => (value as f32) as f64,
        NumericDType::F64 => value,
    }
}

pub(crate) fn clamp01(value: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    }
}

pub(crate) fn clamp_round(value: f64, max: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else {
        value.round().clamp(0.0, max)
    }
}

pub(crate) fn image_output_dtype(input: NumericDType) -> NumericDType {
    match input {
        NumericDType::F32 => NumericDType::F32,
        NumericDType::F64
        | NumericDType::I8
        | NumericDType::I16
        | NumericDType::I32
        | NumericDType::I64
        | NumericDType::U8
        | NumericDType::U16
        | NumericDType::U32
        | NumericDType::U64 => NumericDType::F64,
    }
}

pub(crate) fn same_shape_type(args: &[Type]) -> Type {
    match args.first() {
        Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        Some(Type::Num) | Some(Type::Int) | Some(Type::Bool) => Type::Num,
        _ => Type::tensor(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::{gpu_helpers, test_support};
    use runmat_builtins::{IntValue, IntegerStorage};

    #[test]
    fn image_scalar_collapse_reads_authoritative_storage() {
        let single = Tensor::from_f32(vec![0.25], vec![1, 1]).unwrap();
        assert_eq!(image_value_from_tensor(single), Value::Num(0.25));

        let wide = u64::MAX - 1;
        let integer = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
        assert_eq!(
            image_value_from_tensor(integer),
            Value::Int(IntValue::U64(wide))
        );
    }

    #[test]
    fn image_arrays_remain_tensors() {
        let tensor = Tensor::from_f32(vec![0.25, 0.5], vec![1, 2]).unwrap();
        let Value::Tensor(output) = image_value_from_tensor(tensor) else {
            panic!("expected tensor output");
        };
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
        assert_eq!(output.materialize_f64(), vec![0.25, 0.5]);
    }

    #[test]
    fn rejects_legacy_global_provider_fallback_for_foreign_device_handle() {
        test_support::with_test_provider(|provider| {
            let handle = GpuTensorHandle {
                shape: vec![1, 1],
                device_id: provider.device_id().wrapping_add(1000),
                buffer_id: u64::MAX - 21,
                descriptor: Default::default(),
            }
            .with_numeric_descriptor(
                match provider.precision() {
                    runmat_accelerate_api::ProviderPrecision::F32 => {
                        runmat_accelerate_api::NumericElementType::F32
                    }
                    runmat_accelerate_api::ProviderPrecision::F64 => {
                        runmat_accelerate_api::NumericElementType::F64
                    }
                },
                GpuTensorStorage::Real,
            );
            let error = resident_numeric_dtype(&handle, "color-test")
                .expect_err("global fallback is not exact ownership");
            assert!(error.message().contains("not owned"));
        });
    }

    #[test]
    fn precision_mismatch_returns_host_only_for_automatic_residency() {
        test_support::with_test_provider(|provider| {
            let source_tensor =
                Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).unwrap();
            let source = gpu_helpers::upload_tensor(provider, &source_tensor).unwrap();
            let single = Value::Tensor(Tensor::from_f32(vec![0.5], vec![1, 1]).unwrap());

            let source =
                source.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic);
            assert!(matches!(
                restore_resident_numeric_result(Some(&source), single.clone(), "color-test")
                    .unwrap(),
                Value::Tensor(_)
            ));

            let source =
                source.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let error = restore_resident_numeric_result(Some(&source), single, "color-test")
                .expect_err("explicit residency cannot silently become host");
            assert!(error
                .message()
                .contains("explicit gpuArray result requires"));
            provider.free(&source).unwrap();
            runmat_accelerate_api::clear_handle_metadata(&source);
        });
    }
}
