use glam::Vec4;
use runmat_builtins::{Tensor, Value};
use std::collections::VecDeque;

use crate::gather_if_needed;

use super::style::{
    parse_color_value, parse_line_style_args, value_as_f64, value_as_string, LineStyleParseOptions,
    ParsedLineStyle,
};

#[derive(Clone)]
pub struct PointArgs {
    pub size: PointSizeArg,
    pub color: PointColorArg,
    pub filled: bool,
    pub style: ParsedLineStyle,
}

#[derive(Clone)]
pub enum PointSizeArg {
    Default,
    Scalar(f32),
    Values(Value),
}

#[derive(Clone)]
pub enum PointColorArg {
    Default,
    Uniform(Vec4),
    ScalarValues(Value),
    RgbMatrix(Value),
}

impl PointArgs {
    pub fn parse(rest: Vec<Value>, opts: LineStyleParseOptions) -> Result<Self, String> {
        let mut queue: VecDeque<Value> = VecDeque::from(rest);
        let mut size = PointSizeArg::Default;
        if queue.front().map_or(false, is_numeric_candidate) {
            let value = queue.pop_front().expect("queue peeked");
            size = PointSizeArg::from_value(value, opts.builtin_name)?;
        }

        let mut color = PointColorArg::Default;
        if let Some(token) = queue.front() {
            if !is_filled_token(token) && is_color_positional_candidate(token) {
                let value = queue.pop_front().expect("queue peeked");
                color = PointColorArg::from_value(value, &opts)?;
            }
        }

        let mut filled = false;
        while let Some(token) = queue.front() {
            if is_filled_token(token) {
                filled = true;
                queue.pop_front();
            } else {
                break;
            }
        }

        let remaining: Vec<Value> = queue.into();
        let style = parse_line_style_args(&remaining, &opts)?;
        Ok(Self {
            size,
            color,
            filled,
            style,
        })
    }

    pub fn requires_cpu(&self) -> bool {
        self.style.requires_cpu_fallback
            || matches!(self.size, PointSizeArg::Values(_))
            || matches!(
                self.color,
                PointColorArg::ScalarValues(_) | PointColorArg::RgbMatrix(_)
            )
    }
}

impl PointSizeArg {
    pub fn from_value(value: Value, builtin: &str) -> Result<Self, String> {
        if value_is_empty(&value) {
            return Ok(Self::Default);
        }
        if is_scalar_numeric(&value) {
            let scalar = extract_scalar_f32(&value)
                .ok_or_else(|| format!("{builtin}: marker sizes must be numeric"))?;
            return Ok(Self::Scalar(scalar.max(0.0)));
        }
        if matches!(value, Value::Tensor(_) | Value::GpuTensor(_)) {
            return Ok(Self::Values(value));
        }
        Err(format!(
            "{builtin}: marker sizes must be numeric vectors or scalars"
        ))
    }

    pub fn values(&self) -> Option<&Value> {
        match self {
            PointSizeArg::Values(value) => Some(value),
            _ => None,
        }
    }
}

impl PointColorArg {
    pub fn from_value(value: Value, opts: &LineStyleParseOptions) -> Result<Self, String> {
        if value_is_empty(&value) {
            return Ok(Self::Default);
        }
        if value_as_string(&value)
            .map(|text| text.trim().eq_ignore_ascii_case("auto"))
            .unwrap_or(false)
        {
            return Ok(Self::Default);
        }
        match &value {
            Value::CharArray(_) | Value::String(_) => {
                let color = parse_color_value(opts, &value)?;
                Ok(Self::Uniform(color))
            }
            Value::Tensor(tensor) => {
                if tensor.data.len() == 1 {
                    Ok(Self::ScalarValues(Value::Tensor(tensor.clone())))
                } else if tensor.cols == 3 {
                    Ok(Self::RgbMatrix(Value::Tensor(tensor.clone())))
                } else {
                    Ok(Self::ScalarValues(Value::Tensor(tensor.clone())))
                }
            }
            Value::GpuTensor(handle) => {
                if total_len(handle.shape.as_slice()) == 1 {
                    Ok(Self::ScalarValues(value))
                } else if handle.shape.last().copied().unwrap_or_default() == 3 {
                    Ok(Self::RgbMatrix(value))
                } else {
                    Ok(Self::ScalarValues(value))
                }
            }
            Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(Self::ScalarValues(value)),
            _ => Err(format!(
                "{}: color arguments must be numeric arrays or color strings",
                opts.builtin_name
            )),
        }
    }
}

fn is_numeric_candidate(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(_) | Value::GpuTensor(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_)
    )
}

fn is_color_positional_candidate(value: &Value) -> bool {
    match value {
        Value::Tensor(_) | Value::GpuTensor(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            true
        }
        Value::CharArray(_) | Value::String(_) => value_as_string(value)
            .map(|text| is_color_literal(&text))
            .unwrap_or(false),
        _ => false,
    }
}

fn is_color_literal(token: &str) -> bool {
    match token.trim().to_ascii_lowercase().as_str() {
        "r" | "red" | "g" | "green" | "b" | "blue" | "c" | "cyan" | "m" | "magenta" | "y"
        | "yellow" | "k" | "black" | "w" | "white" | "auto" => true,
        _ => false,
    }
}

fn is_filled_token(value: &Value) -> bool {
    value_as_string(value)
        .map(|text| text.trim().eq_ignore_ascii_case("filled"))
        .unwrap_or(false)
}

fn value_is_empty(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.data.is_empty(),
        Value::GpuTensor(handle) => total_len(handle.shape.as_slice()) == 0,
        Value::CharArray(chars) => chars.data.is_empty(),
        Value::String(s) => s.trim().is_empty(),
        _ => false,
    }
}

fn is_scalar_numeric(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => true,
        Value::Tensor(tensor) => tensor.data.len() == 1,
        Value::GpuTensor(handle) => total_len(handle.shape.as_slice()) == 1,
        _ => false,
    }
}

fn extract_scalar_f32(value: &Value) -> Option<f32> {
    value_as_f64(value).map(|v| v as f32)
}

fn total_len(shape: &[usize]) -> usize {
    if shape.is_empty() {
        0
    } else {
        shape.iter().product()
    }
}

pub(crate) fn tensor_from_value(value: &Value, context: &str) -> Result<Tensor, String> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.clone()),
        Value::GpuTensor(handle) => {
            let tmp = Value::GpuTensor(handle.clone());
            let gathered = gather_if_needed(&tmp)?;
            Tensor::try_from(&gathered).map_err(|e| format!("{context}: {e}"))
        }
        _ => Tensor::try_from(value).map_err(|e| format!("{context}: {e}")),
    }
}

pub(crate) fn convert_size_vector(
    value: &Value,
    point_count: usize,
    context: &str,
) -> Result<Vec<f32>, String> {
    let mut tensor = tensor_from_value(value, context)?;
    if tensor.data.is_empty() {
        return Err(format!("{context}: marker size array cannot be empty"));
    }
    if tensor.data.len() == 1 && point_count > 1 {
        tensor.data = vec![tensor.data[0]; point_count];
    } else if point_count > 0 && tensor.data.len() != point_count {
        return Err(format!(
            "{context}: marker size array must have {} elements or be scalar",
            point_count
        ));
    }
    Ok(tensor.data.into_iter().map(|v| v.max(0.1) as f32).collect())
}

pub(crate) fn convert_scalar_color_values(
    value: &Value,
    point_count: usize,
    context: &str,
) -> Result<Vec<f64>, String> {
    let mut tensor = tensor_from_value(value, context)?;
    if tensor.data.is_empty() {
        return Err(format!("{context}: color array cannot be empty"));
    }
    if tensor.data.len() == 1 && point_count > 1 {
        tensor.data = vec![tensor.data[0]; point_count];
    } else if point_count > 0 && tensor.data.len() != point_count {
        return Err(format!(
            "{context}: color array must have {} elements or be scalar",
            point_count
        ));
    }
    Ok(tensor.data)
}

pub(crate) fn convert_rgb_color_matrix(
    value: &Value,
    point_count: usize,
    context: &str,
) -> Result<Vec<Vec4>, String> {
    let tensor = tensor_from_value(value, context)?;
    if tensor.cols != 3 {
        return Err(format!(
            "{context}: RGB color matrices must have three columns"
        ));
    }
    if tensor.rows == 0 {
        return Err(format!("{context}: RGB color matrices cannot be empty"));
    }

    let rows = tensor.rows;
    if point_count > 0 && rows != point_count && rows != 1 {
        return Err(format!(
            "{context}: RGB color matrix must have {} rows or be a single color",
            point_count
        ));
    }

    let repeat = if rows == 1 && point_count > 1 {
        point_count
    } else {
        rows
    };
    let mut colors = Vec::with_capacity(repeat);
    for row in 0..repeat {
        let src_row = if rows == 1 { 0 } else { row };
        let r = tensor_value(&tensor, src_row, 0) as f32;
        let g = tensor_value(&tensor, src_row, 1) as f32;
        let b = tensor_value(&tensor, src_row, 2) as f32;
        colors.push(Vec4::new(r, g, b, 1.0));
    }
    Ok(colors)
}

fn tensor_value(tensor: &Tensor, row: usize, col: usize) -> f64 {
    let rows = tensor.rows.max(1);
    let idx = col * rows + row.min(rows - 1);
    tensor.data.get(idx).copied().unwrap_or(0.0)
}
