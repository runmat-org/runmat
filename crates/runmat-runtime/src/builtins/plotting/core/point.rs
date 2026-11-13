use glam::Vec4;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_plot::plots::surface::ColorMap;
use runmat_value::{Tensor, Value};

use crate::builtins::common::{map_control_flow_with_builtin, tensor};
use crate::{gather_if_needed_async, BuiltinResult};

use super::plotting_error;

use super::style::{
    parse_color_value, value_as_f64, value_as_string, LineStyleParseOptions, ParsedLineStyle,
};

const POINTS_TO_PX: f32 = 96.0 / 72.0;
const DEFAULT_MARKER_AREA_POINTS2: f64 = 36.0;

/// Point colors are rows of RGB or RGBA components.
fn point_color_matrix_dimensions(tensor: &Tensor) -> (usize, usize) {
    match tensor.shape() {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

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
    /// Borrow segmented canonical arguments without collecting or dropping their
    /// root owner. The caller retains that owner through rendering suspension.
    pub fn parse_arguments(
        rest: crate::call::FunctionArgumentView<'_>,
        opts: LineStyleParseOptions,
    ) -> BuiltinResult<Self> {
        let mut index = 0;
        let size = if rest.get(index).is_some_and(is_numeric_candidate) {
            let value = rest[index].clone();
            index += 1;
            PointSizeArg::from_value(value, opts.builtin_name)?
        } else {
            PointSizeArg::Default
        };
        let color = if rest
            .get(index)
            .map(|token| {
                Ok::<_, crate::RuntimeError>(
                    !is_filled_token(token, opts.builtin_name)?
                        && is_color_positional_candidate(token, opts.builtin_name)?,
                )
            })
            .transpose()?
            .unwrap_or(false)
        {
            let value = rest[index].clone();
            index += 1;
            PointColorArg::from_value(value, &opts)?
        } else {
            PointColorArg::Default
        };
        let mut filled = false;
        while rest
            .get(index)
            .map(|token| is_filled_token(token, opts.builtin_name))
            .transpose()?
            .unwrap_or(false)
        {
            filled = true;
            index += 1;
        }
        let style = super::style::parse_line_style_arguments(rest.range(index..)?, &opts)?;
        Ok(Self {
            size,
            color,
            filled,
            style,
        })
    }

    pub fn parse(rest: Vec<Value>, opts: LineStyleParseOptions) -> BuiltinResult<Self> {
        Self::parse_arguments(crate::call::FunctionArgumentView::from_slice(&rest), opts)
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
            .map_err(|source| {
                crate::builtins::plotting::plotting_error_with_source(
                    opts.builtin_name,
                    source.to_string(),
                    source,
                )
            })?
            .map(|text| text.trim().eq_ignore_ascii_case("auto"))
            .unwrap_or(false)
        {
            return Ok(Self::Default);
        }
        match &value {
            Value::CharArray(_) | Value::String(runmat_value::StringElement::Text(_)) => {
                let color = parse_color_value(opts, &value)?;
                Ok(Self::Uniform(color))
            }
            Value::Tensor(tensor) => {
                let (rows, cols) = point_color_matrix_dimensions(tensor);
                if tensor::is_scalar_tensor(tensor) {
                    Ok(Self::ScalarValues(Value::Tensor(tensor.clone())))
                } else if rows == 1 && (cols == 3 || cols == 4) {
                    let r = tensor_value(tensor, 0, 0) as f32;
                    let g = tensor_value(tensor, 0, 1) as f32;
                    let b = tensor_value(tensor, 0, 2) as f32;
                    let a = if cols == 4 {
                        tensor_value(tensor, 0, 3) as f32
                    } else {
                        1.0
                    };
                    Ok(Self::Uniform(Vec4::new(r, g, b, a)))
                } else if cols == 3 || cols == 4 {
                    Ok(Self::RgbMatrix(Value::Tensor(tensor.clone())))
                } else {
                    Ok(Self::ScalarValues(Value::Tensor(tensor.clone())))
                }
            }
            Value::GpuTensor(handle) => {
                if total_len(handle.shape()) == 1 {
                    Ok(Self::ScalarValues(value))
                } else if matches!(handle.shape().last().copied().unwrap_or_default(), 3 | 4) {
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

fn is_color_positional_candidate(value: &Value, builtin: &str) -> crate::BuiltinResult<bool> {
    Ok({
        match value {
            Value::Tensor(_)
            | Value::GpuTensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_) => true,
            Value::CharArray(_) | Value::String(runmat_value::StringElement::Text(_)) => {
                value_as_string(value)
                    .map_err(|source| {
                        crate::builtins::plotting::plotting_error_with_source(
                            builtin,
                            source.to_string(),
                            source,
                        )
                    })?
                    .map(|text| is_color_literal(&text))
                    .unwrap_or(false)
            }
            _ => false,
        }
    })
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

fn is_filled_token(value: &Value, builtin: &str) -> crate::BuiltinResult<bool> {
    Ok({
        value_as_string(value)
            .map_err(|source| {
                crate::builtins::plotting::plotting_error_with_source(
                    builtin,
                    source.to_string(),
                    source,
                )
            })?
            .map(|text| text.trim().eq_ignore_ascii_case("filled"))
            .unwrap_or(false)
    })
}

fn value_is_empty(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor::tensor_element_len(tensor) == 0,
        Value::GpuTensor(handle) => total_len(handle.shape()) == 0,
        Value::CharArray(chars) => chars.data().is_empty(),
        Value::String(runmat_value::StringElement::Text(s)) => s.trim().is_empty(),
        _ => false,
    }
}

fn is_scalar_numeric(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => true,
        Value::Tensor(tensor) => tensor::is_scalar_tensor(tensor),
        Value::GpuTensor(handle) => total_len(handle.shape()) == 1,
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

pub(crate) async fn tensor_from_value_async(
    value: &Value,
    context: &'static str,
) -> BuiltinResult<Tensor> {
    match value {
        Value::GpuTensor(handle) => {
            let tmp = Value::GpuTensor(handle.clone());
            let gathered = gather_if_needed_async(&tmp)
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, context))?;
            Tensor::try_from(&gathered)
                .map_err(|e| plotting_error(context, format!("{context}: {e}")))
        }
        _ => tensor_from_value(value, context),
    }
}

pub(crate) fn tensor_from_value(value: &Value, context: &'static str) -> BuiltinResult<Tensor> {
    // Synchronous consumers have already materialized provider sources. Keep
    // host projection usable inside an active async property transaction;
    // provider readback belongs to the async conversion above.
    match value {
        Value::Tensor(tensor) => Ok(tensor.clone()),
        Value::GpuTensor(_) => Err(plotting_error(
            context,
            format!("{context}: GPU point data requires async conversion"),
        )),
        _ => {
            Tensor::try_from(value).map_err(|e| plotting_error(context, format!("{context}: {e}")))
        }
    }
}

pub(crate) fn convert_size_vector(
    value: &Value,
    point_count: usize,
    context: &'static str,
) -> BuiltinResult<Vec<f32>> {
    convert_size_tensor(tensor_from_value(value, context)?, point_count, context)
}

pub(crate) async fn convert_size_vector_async(
    value: &Value,
    point_count: usize,
    context: &'static str,
) -> BuiltinResult<Vec<f32>> {
    convert_size_tensor(
        tensor_from_value_async(value, context).await?,
        point_count,
        context,
    )
}

fn convert_size_tensor(
    tensor: Tensor,
    point_count: usize,
    context: &'static str,
) -> BuiltinResult<Vec<f32>> {
    let mut data = tensor::tensor_values_f64(&tensor);
    if data.is_empty() {
        return Err(plotting_error(
            context,
            format!("{context}: marker size array cannot be empty"),
        ));
    }
    if data.len() == 1 && point_count > 1 {
        data = vec![data[0]; point_count];
    } else if point_count > 0 && data.len() != point_count {
        return Err(plotting_error(
            context,
            format!(
                "{context}: marker size array must have {} elements or be scalar",
                point_count
            ),
        ));
    }
    Ok(data
        .into_iter()
        .map(marker_area_points2_to_diameter_px)
        .collect())
}

pub(crate) fn marker_area_points2_to_diameter_px(area_points2: f64) -> f32 {
    runmat_plot::plots::scatter::marker_area_points2_to_diameter_px(area_points2)
}

pub(crate) fn marker_diameter_px_to_area_points2(diameter_px: f32) -> f64 {
    let d_pt = (diameter_px.max(1.0) / POINTS_TO_PX) as f64;
    (d_pt * d_pt).max(0.1)
}

pub(crate) fn default_marker_diameter_px() -> f32 {
    marker_area_points2_to_diameter_px(DEFAULT_MARKER_AREA_POINTS2)
}

pub(crate) async fn convert_scalar_color_values_async(
    value: &Value,
    point_count: usize,
    context: &'static str,
) -> BuiltinResult<Vec<f64>> {
    convert_scalar_color_tensor(
        tensor_from_value_async(value, context).await?,
        point_count,
        context,
    )
}

fn convert_scalar_color_tensor(
    tensor: Tensor,
    point_count: usize,
    context: &'static str,
) -> BuiltinResult<Vec<f64>> {
    let mut data = tensor::tensor_values_f64(&tensor);
    if data.is_empty() {
        return Err(plotting_error(
            context,
            format!("{context}: color array cannot be empty"),
        ));
    }
    if data.len() == 1 && point_count > 1 {
        data = vec![data[0]; point_count];
    } else if point_count > 0 && data.len() != point_count {
        return Err(plotting_error(
            context,
            format!(
                "{context}: color array must have {} elements or be scalar",
                point_count
            ),
        ));
    }
    Ok(data)
}

pub(crate) fn convert_rgb_color_matrix(
    value: &Value,
    point_count: usize,
    context: &'static str,
) -> BuiltinResult<Vec<Vec4>> {
    convert_rgb_color_tensor(tensor_from_value(value, context)?, point_count, context)
}

pub(crate) async fn convert_rgb_color_matrix_async(
    value: &Value,
    point_count: usize,
    context: &'static str,
) -> BuiltinResult<Vec<Vec4>> {
    convert_rgb_color_tensor(
        tensor_from_value_async(value, context).await?,
        point_count,
        context,
    )
}

fn convert_rgb_color_tensor(
    tensor: Tensor,
    point_count: usize,
    context: &'static str,
) -> BuiltinResult<Vec<Vec4>> {
    let (rows, cols) = point_color_matrix_dimensions(&tensor);
    if cols != 3 && cols != 4 {
        return Err(plotting_error(
            context,
            format!("{context}: color matrices must have three (RGB) or four (RGBA) columns"),
        ));
    }
    if rows == 0 {
        return Err(plotting_error(
            context,
            format!("{context}: RGB color matrices cannot be empty"),
        ));
    }

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
        let a = if cols > 3 {
            tensor_value(&tensor, src_row, 3) as f32
        } else {
            1.0
        };
        colors.push(Vec4::new(r, g, b, a));
    }
    Ok(colors)
}

fn tensor_value(tensor: &Tensor, row: usize, col: usize) -> f64 {
    let rows = point_color_matrix_dimensions(tensor).0.max(1);
    let idx = col * rows + row.min(rows - 1);
    tensor::tensor_values_f64(tensor)
        .get(idx)
        .copied()
        .unwrap_or(0.0)
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
    let len = total_len(handle.shape());
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
    if handle.shape().len() < 2 {
        return Err(plotting_error(
            context,
            format!("{context}: color gpuArray inputs must be at least 2-D"),
        ));
    }
    let cols = handle.shape().last().copied().unwrap_or_default();
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
    let rows = handle.shape()[..handle.shape().len() - 1]
        .iter()
        .product::<usize>();
    if rows != point_count {
        return Err(plotting_error(
            context,
            format!("{context}: color gpuArray inputs must have {point_count} rows (got {rows})"),
        ));
    }
    Ok(components)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::IntegerStorage;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn point_rgb_colors_keep_rank_one_rows_and_reject_rank_zero_matrices() {
        let row = Tensor::new(vec![0.25, 0.5, 0.75], vec![3]).unwrap();
        let arg = PointColorArg::from_value(
            Value::Tensor(row.clone()),
            &LineStyleParseOptions::scatter(),
        )
        .unwrap();
        let expected = Vec4::new(0.25, 0.5, 0.75, 1.0);
        assert!(matches!(arg, PointColorArg::Uniform(color) if color == expected));
        assert_eq!(
            convert_rgb_color_tensor(row, 2, "scatter").unwrap(),
            vec![expected; 2]
        );

        let scalar = Tensor::new(vec![0.5], Vec::new()).unwrap();
        let error = convert_rgb_color_tensor(scalar, 1, "scatter")
            .expect_err("rank-zero color input has no RGB columns");
        assert!(error
            .message()
            .contains("three (RGB) or four (RGBA) columns"));

        let matrix = Tensor::new(vec![0.25, 0.5, 0.5, 0.75, 0.75, 1.0], vec![2, 3]).unwrap();
        assert_eq!(
            block_on(async {
                convert_rgb_color_matrix(&Value::Tensor(matrix), 2, "scatter").unwrap()
            }),
            vec![expected, Vec4::new(0.5, 0.75, 1.0, 1.0)]
        );
    }

    #[test]
    fn point_size_and_color_vectors_read_typed_integer_storage() {
        let sizes = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![4, 9]), vec![1, 2]).expect("sizes"),
        );
        let colors = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I16(vec![7, 8]), vec![1, 2]).expect("colors"),
        );

        // The sync host projection must also work while an async setter is
        // being polled, without entering another executor.
        let size_px = block_on(async {
            convert_size_vector(&sizes, 2, "scatter").expect("sizes inside async transaction")
        });
        let color_values =
            block_on(convert_scalar_color_values_async(&colors, 2, "scatter")).expect("colors");

        assert_eq!(size_px.len(), 2);
        assert_eq!(color_values, vec![7.0, 8.0]);
    }

    #[test]
    fn point_scalar_helpers_accept_typed_integer_without_double_mirror() {
        let scalar =
            Tensor::new_integer(IntegerStorage::U16(vec![12]), vec![1, 1]).expect("scalar");
        let value = Value::Tensor(scalar);

        assert!(is_scalar_numeric(&value));
        assert!(matches!(
            PointColorArg::from_value(value, &LineStyleParseOptions::scatter()).expect("color"),
            PointColorArg::ScalarValues(Value::Tensor(_))
        ));
    }
}
