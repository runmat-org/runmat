use std::borrow::Cow;

use runmat_builtins::{NumericDType, NumericScalar, Tensor, Type, Value};

use crate::builtins::common::{map_control_flow_with_builtin, tensor};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColorLayout {
    Truecolor { rows: usize, cols: usize },
    Colormap { rows: usize },
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

pub(crate) fn restore_resident_numeric_result(
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
    value: Value,
    name: &'static str,
) -> BuiltinResult<Value> {
    let Some(source) = source else {
        return Ok(value);
    };
    crate::builtins::common::gpu_helpers::restore_class_preserving_value(source, value, name)
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
}
