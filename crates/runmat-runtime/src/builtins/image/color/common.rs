use runmat_builtins::{IntValue, NumericDType, Tensor, Type, Value};

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

pub(crate) async fn gather_tensor(name: &'static str, value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_value(name, &value).await?;
    tensor::value_into_tensor_for(name, gathered).map_err(|err| builtin_error(name, err))
}

pub(crate) fn image_value_from_tensor(tensor: Tensor) -> Value {
    if tensor.data.len() == 1 {
        match tensor.dtype {
            NumericDType::U8 => Value::Int(IntValue::U8(clamp_round(tensor.data[0], 255.0) as u8)),
            NumericDType::U16 => {
                Value::Int(IntValue::U16(clamp_round(tensor.data[0], 65535.0) as u16))
            }
            NumericDType::F32 | NumericDType::F64 => Value::Num(tensor.data[0]),
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
    Tensor::new_with_dtype(data, shape, dtype).map_err(|err| builtin_error(name, err))
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

pub(crate) fn truecolor_layout(tensor: &Tensor, name: &'static str) -> BuiltinResult<ColorLayout> {
    let shape = &tensor.shape;
    if shape.len() == 3 && shape[2] == 3 && shape[0] > 0 && shape[1] > 0 {
        return Ok(ColorLayout::Truecolor {
            rows: shape[0],
            cols: shape[1],
        });
    }
    Err(builtin_error(
        name,
        format!("{name}: expected an MxNx3 RGB image"),
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

pub(crate) fn dtype_max(dtype: NumericDType) -> f64 {
    match dtype {
        NumericDType::U8 => 255.0,
        NumericDType::U16 => 65535.0,
        NumericDType::F32 | NumericDType::F64 => 1.0,
    }
}

pub(crate) fn unit_value(value: f64, dtype: NumericDType) -> f64 {
    match dtype {
        NumericDType::U8 => value / 255.0,
        NumericDType::U16 => value / 65535.0,
        NumericDType::F32 | NumericDType::F64 => value,
    }
}

pub(crate) fn unit_to_dtype(value: f64, dtype: NumericDType) -> f64 {
    match dtype {
        NumericDType::U8 => clamp_round(value * 255.0, 255.0),
        NumericDType::U16 => clamp_round(value * 65535.0, 65535.0),
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
        NumericDType::F64 | NumericDType::U8 | NumericDType::U16 => NumericDType::F64,
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
