use glam::Vec4;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{Tensor, Value};
use runmat_plot::plots::surface::ColorMap;
use std::collections::VecDeque;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::{gather_if_needed, BuiltinResult};

use super::plotting_error;

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
    pub fn parse(rest: Vec<Value>, opts: LineStyleParseOptions) -> BuiltinResult<Self> {
        let mut queue: VecDeque<Value> = VecDeque::from(rest);
        let mut size = PointSizeArg::Default;
        if queue.front().is_some_and(is_numeric_candidate) {
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
}

impl PointSizeArg {
    pub fn from_value(value: Value, builtin: &'static str) -> BuiltinResult<Self> {
        if value_is_empty(&value) {
            return Ok(Self::Default);
        }
        if is_scalar_numeric(&value) {
            let scalar = extract_scalar_f32(&value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: marker sizes must be numeric"))
            })?;
            return Ok(Self::Scalar(scalar.max(0.0)));
        }
        if matches!(value, Value::Tensor(_) | Value::GpuTensor(_)) {
            return Ok(Self::Values(value));
        }
        Err(plotting_error(
            builtin,
            format!("{builtin}: marker sizes must be numeric vectors or scalars"),
        ))
    }

    pub fn values(&self) -> Option<&Value> {
        match self {
            PointSizeArg::Values(value) => Some(value),
            _ => None,
        }
    }

    pub fn value(&self) -> Option<&Value> {
        self.values()
    }
}

impl PointColorArg {
    pub fn from_value(value: Value, opts: &LineStyleParseOptions) -> BuiltinResult<Self> {
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
                } else if tensor.cols == 3 || tensor.cols == 4 {
                    Ok(Self::RgbMatrix(Value::Tensor(tensor.clone())))
                } else {
                    Ok(Self::ScalarValues(Value::Tensor(tensor.clone())))
                }
            }
            Value::GpuTensor(handle) => {
                if total_len(handle.shape.as_slice()) == 1 {
                    Ok(Self::ScalarValues(value))
                } else if matches!(handle.shape.last().copied().unwrap_or_default(), 3 | 4) {
                    Ok(Self::RgbMatrix(value))
                } else {
                    Ok(Self::ScalarValues(value))
                }
            }
            Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(Self::ScalarValues(value)),
            _ => Err(plotting_error(
                opts.builtin_name,
                format!(
                    "{}: color arguments must be numeric arrays or color strings",
                    opts.builtin_name
                ),
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
    matches!(
        token.trim().to_ascii_lowercase().as_str(),
        "r" | "red"
            | "g"
            | "green"
            | "b"
            | "blue"
            | "c"
            | "cyan"
            | "m"
            | "magenta"
            | "y"
            | "yellow"
            | "k"
            | "black"
            | "w"
            | "white"
            | "auto"
    )
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

pub(crate) fn tensor_from_value(value: &Value, context: &'static str) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.clone()),
        Value::GpuTensor(handle) => {
            let tmp = Value::GpuTensor(handle.clone());
            let gathered = gather_if_needed(&tmp)
                .map_err(|flow| map_control_flow_with_builtin(flow, context))?;
            Tensor::try_from(&gathered)
                .map_err(|e| plotting_error(context, format!("{context}: {e}")))
        }
        _ => Tensor::try_from(value).map_err(|e| plotting_error(context, format!("{context}: {e}"))),
    }
}

pub(crate) fn convert_size_vector(
    value: &Value,
    point_count: usize,
    context: &'static str,
) -> BuiltinResult<Vec<f32>> {
    let mut tensor = tensor_from_value(value, context)?;
    if tensor.data.is_empty() {
        return Err(plotting_error(
            context,
            format!("{context}: marker size array cannot be empty"),
        ));
    }
    if tensor.data.len() == 1 && point_count > 1 {
        tensor.data = vec![tensor.data[0]; point_count];
    } else if point_count > 0 && tensor.data.len() != point_count {
        return Err(plotting_error(
            context,
            format!(
                "{context}: marker size array must have {} elements or be scalar",
                point_count
            ),
        ));
    }
    Ok(tensor.data.into_iter().map(|v| v.max(0.1) as f32).collect())
}

pub(crate) fn convert_scalar_color_values(
    value: &Value,
    point_count: usize,
    context: &'static str,
) -> BuiltinResult<Vec<f64>> {
    let mut tensor = tensor_from_value(value, context)?;
    if tensor.data.is_empty() {
        return Err(plotting_error(
            context,
            format!("{context}: color array cannot be empty"),
        ));
    }
    if tensor.data.len() == 1 && point_count > 1 {
        tensor.data = vec![tensor.data[0]; point_count];
    } else if point_count > 0 && tensor.data.len() != point_count {
        return Err(plotting_error(
            context,
            format!(
                "{context}: color array must have {} elements or be scalar",
                point_count
            ),
        ));
    }
    Ok(tensor.data)
}

pub(crate) fn convert_rgb_color_matrix(
    value: &Value,
    point_count: usize,
    context: &'static str,
) -> BuiltinResult<Vec<Vec4>> {
    let tensor = tensor_from_value(value, context)?;
    if tensor.cols != 3 && tensor.cols != 4 {
        return Err(plotting_error(
            context,
            format!(
                "{context}: color matrices must have three (RGB) or four (RGBA) columns"
            ),
        ));
    }
    if tensor.rows == 0 {
        return Err(plotting_error(
            context,
            format!("{context}: RGB color matrices cannot be empty"),
        ));
    }

    let rows = tensor.rows;
    if point_count > 0 && rows != point_count && rows != 1 {
        return Err(plotting_error(
            context,
            format!(
                "{context}: RGB color matrix must have {} rows or be a single color",
                point_count
            ),
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
        let a = if tensor.cols > 3 {
            tensor_value(&tensor, src_row, 3) as f32
        } else {
            1.0
        };
        colors.push(Vec4::new(r, g, b, a));
    }
    Ok(colors)
}

fn tensor_value(tensor: &Tensor, row: usize, col: usize) -> f64 {
    let rows = tensor.rows.max(1);
    let idx = col * rows + row.min(rows - 1);
    tensor.data.get(idx).copied().unwrap_or(0.0)
}

#[derive(Clone, Debug)]
pub struct PointGpuColor {
    pub handle: GpuTensorHandle,
    pub components: PointColorComponents,
}

#[derive(Clone, Copy, Debug)]
pub enum PointColorComponents {
    Rgb,
    Rgba,
}

impl PointColorComponents {
    pub fn stride(&self) -> u32 {
        match self {
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
}

pub fn map_scalar_values_to_colors(values: &[f64], colormap: ColorMap) -> (Vec<Vec4>, (f64, f64)) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &value in values {
        if value.is_finite() {
            if value < lo {
                lo = value;
            }
            if value > hi {
                hi = value;
            }
        }
    }
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        lo = 0.0;
        hi = 1.0;
    }
    let denom = (hi - lo).max(f64::EPSILON);
    let colors = values
        .iter()
        .map(|&value| {
            let t = ((value - lo) / denom).clamp(0.0, 1.0) as f32;
            let rgb = colormap.map_value(t);
            Vec4::new(rgb.x, rgb.y, rgb.z, 1.0)
        })
        .collect::<Vec<_>>();
    (colors, (lo, hi))
}

pub fn validate_gpu_vector_length(
    handle: &GpuTensorHandle,
    point_count: usize,
    context: &'static str,
) -> BuiltinResult<()> {
    let len = total_len(handle.shape.as_slice());
    if len != point_count {
        return Err(plotting_error(
            context,
            format!(
                "{context}: gpuArray inputs must contain exactly {point_count} elements (got {len})"
            ),
        ));
    }
    Ok(())
}

pub fn validate_gpu_color_matrix(
    handle: &GpuTensorHandle,
    point_count: usize,
    context: &'static str,
) -> BuiltinResult<PointColorComponents> {
    if handle.shape.len() < 2 {
        return Err(plotting_error(
            context,
            format!("{context}: color gpuArray inputs must be at least 2-D"),
        ));
    }
    let cols = handle.shape.last().copied().unwrap_or_default();
    let components = match cols {
        3 => PointColorComponents::Rgb,
        4 => PointColorComponents::Rgba,
        _ => {
            return Err(plotting_error(
                context,
                format!(
                    "{context}: color gpuArray inputs must have three (RGB) or four (RGBA) columns"
                ),
            ))
        }
    };
    let rows = handle.shape[..handle.shape.len() - 1]
        .iter()
        .product::<usize>();
    if rows != point_count {
        return Err(plotting_error(
            context,
            format!(
                "{context}: color gpuArray inputs must have {point_count} rows (got {rows})"
            ),
        ));
    }
    Ok(components)
}
