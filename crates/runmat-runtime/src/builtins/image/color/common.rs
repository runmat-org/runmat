use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, IntegerElementType, ProviderPrecision,
};
use std::borrow::Cow;

use runmat_value::{NumericDType, NumericScalar, Tensor, Value};

use crate::builtins::common::{map_control_flow_with_builtin, tensor};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColorLayout {
    Truecolor { rows: usize, cols: usize },
    Colormap { rows: usize },
}

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

// View and physical facts are immutable. Retain borrowed views through await
// and restore only the external residency registration.
pub(crate) struct ResidentInputGuard {
    entries: Vec<GpuTensorHandle>,
}

impl ResidentInputGuard {
    pub(crate) fn restore(&self) {
        for handle in &self.entries {
            runmat_accelerate_api::mark_residency(handle);
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
        entries: sources.to_vec(),
    }
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
        // Host computation may have suspended while reading back an operand.
        // Revalidate every borrowed source before entering any upload hook.
        super::resident_owner::validate_live(input, name)?;
    }
    let _residency = protect_resident_inputs(sources);
    let restore = |value| {
        crate::builtins::common::gpu_helpers::restore_class_preserving_value_protecting(
            source,
            value,
            name,
            &sources.iter().collect::<Vec<_>>(),
        )
        .map_err(|error| map_flow(name, error))
    };
    match value {
        // Transfer the existing native payload, and let the shared scalar
        // adapter preserve Num when automatic restoration is unsupported.
        value @ (Value::Tensor(_) | Value::Num(_)) => restore(value),
        Value::Int(integer) => {
            let tensor = tensor::value_into_tensor_for(name, Value::Int(integer.clone()))
                .map_err(|error| builtin_error(name, error))?;
            match restore(Value::Tensor(tensor))? {
                Value::Tensor(_) => Ok(Value::Int(integer)),
                restored => Ok(restored),
            }
        }
        Value::Bool(bit) => match restore(Value::Num(if bit { 1.0 } else { 0.0 }))? {
            Value::Num(_) => Ok(Value::Bool(bit)),
            restored => Ok(restored),
        },
        Value::LogicalArray(logical) => {
            // This helper produces numeric color data, not a logical relabel.
            // Keep the original only for a legal automatic host fallback; the
            // converted floating allocation is necessary, not a payload clone.
            let tensor =
                tensor::logical_to_tensor(&logical).map_err(|error| builtin_error(name, error))?;
            match restore(Value::Tensor(tensor))? {
                Value::Tensor(_) => Ok(Value::LogicalArray(logical)),
                restored => Ok(restored),
            }
        }
        value => {
            if runmat_accelerate_api::handle_is_explicit(source) {
                return Err(builtin_error(
                    name,
                    format!("{name}: explicit gpuArray result cannot remain resident"),
                ));
            }
            Ok(value)
        }
    }
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
    super::resident_owner::validate_live(handle, name)?;
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
    if let Some(class_identity) = runmat_accelerate_api::handle_class_identity(handle) {
        if class_identity != dtype.class_identity() {
            return Err(builtin_error(
                name,
                format!(
                    "{name}: GPU class metadata {class_identity} conflicts with {} storage",
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
    let shape = tensor.shape();
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
    let shape = tensor.shape();
    if shape.len() == 2 && shape[1] == 3 && shape[0] > 0 {
        return Ok(StackedColorLayout {
            plane_pixels: shape[0],
            stacks: 1,
            shape: shape.to_vec(),
        });
    }
    if shape.len() >= 3 && shape[0] > 0 && shape[1] > 0 && shape[2] == 3 {
        let plane_pixels = shape[0]
            .checked_mul(shape[1])
            .ok_or_else(|| builtin_error(name, format!("{name}: RGB image dimensions overflow")))?;
        let stacks = shape[3..]
            .iter()
            .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
            .ok_or_else(|| builtin_error(name, format!("{name}: RGB image dimensions overflow")))?;
        return Ok(StackedColorLayout {
            plane_pixels,
            stacks,
            shape: shape.to_vec(),
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
    let shape = tensor.shape();
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

#[cfg(test)]
mod tests;
