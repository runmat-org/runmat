//! MATLAB-compatible `bar` builtin.

use futures::executor::block_on;
use glam::{Vec3, Vec4};
use log::warn;
use runmat_accelerate_api::{self, GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::core::BoundingBox;
use runmat_plot::gpu::bar::{BarGpuInputs, BarGpuParams, BarLayoutMode, BarOrientation};
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::BarChart;
use std::convert::TryFrom;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed_async;
use crate::{BuiltinResult, RuntimeError};

use super::common::numeric_vector;
use super::gpu_helpers::axis_bounds;
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_bar_style_args, BarLayout, BarStyle, BarStyleDefaults};
use crate::builtins::plotting::type_resolvers::string_type;

const BUILTIN_NAME: &str = "bar";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::bar")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "bar",
    op_kind: GpuOpKind::Custom("plot-render"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Single-precision gpuArray vectors render zero-copy when the shared renderer is active; other contexts gather first.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::bar")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "bar",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "bar performs I/O and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "bar",
    category = "plotting",
    summary = "Render a MATLAB-compatible bar chart.",
    keywords = "bar,barchart,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(string_type),
    builtin_path = "crate::builtins::plotting::bar"
)]
pub async fn bar_builtin(values: Value, rest: Vec<Value>) -> crate::BuiltinResult<String> {
    let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
    let style = parse_bar_style_args("bar", &rest, defaults)?;
    let mut input = Some(BarInput::from_value(values)?);
    let opts = PlotRenderOptions {
        title: "Bar Chart",
        x_label: "Category",
        y_label: "Value",
        ..Default::default()
    };
    let rendered = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let style = style.clone();
        let arg = input.take().expect("bar input consumed once");
        if !style.requires_cpu_path() {
            if let Some(handle) = arg.gpu_handle() {
                match build_bar_gpu_series(handle, &style) {
                    Ok(charts) if !charts.is_empty() => {
                        let total = charts.len();
                        for (idx, mut bar) in charts.into_iter().enumerate() {
                            let default_label = default_series_label(idx, total);
                            apply_bar_style(&mut bar, &style, &default_label);
                            figure.add_bar_chart_on_axes(bar, axes);
                        }
                        return Ok(());
                    }
                    Ok(_) => {}
                    Err(err) => {
                        warn!("bar GPU path unavailable: {err}");
                    }
                }
            }
        }
        let tensor = arg.into_tensor("bar")?;
        let charts = build_bar_series_from_tensor(tensor, &style)?;
        let total = charts.len();
        for (idx, mut bar) in charts.into_iter().enumerate() {
            let default_label = default_series_label(idx, total);
            apply_bar_style(&mut bar, &style, &default_label);
            figure.add_bar_chart_on_axes(bar, axes);
        }
        Ok(())
    })?;
    Ok(rendered)
}

const DEFAULT_BAR_WIDTH: f32 = 0.75;

fn bar_err(message: impl Into<String>) -> RuntimeError {
    plotting_error(BUILTIN_NAME, message)
}

const BAR_DEFAULT_LABEL: &str = "Series 1";
const MATLAB_COLOR_ORDER: [Vec4; 7] = [
    Vec4::new(0.0, 0.447, 0.741, 1.0),
    Vec4::new(0.85, 0.325, 0.098, 1.0),
    Vec4::new(0.929, 0.694, 0.125, 1.0),
    Vec4::new(0.494, 0.184, 0.556, 1.0),
    Vec4::new(0.466, 0.674, 0.188, 1.0),
    Vec4::new(0.301, 0.745, 0.933, 1.0),
    Vec4::new(0.635, 0.078, 0.184, 1.0),
];

fn default_bar_color() -> Vec4 {
    Vec4::new(0.2, 0.6, 0.9, 0.95)
}

fn build_bar_chart(values: Vec<f64>) -> BuiltinResult<BarChart> {
    if values.is_empty() {
        return Err(bar_err("bar: input cannot be empty"));
    }
    let labels: Vec<String> = (1..=values.len()).map(|idx| format!("{idx}")).collect();

    let bar = BarChart::new(labels, values).map_err(|err| bar_err(format!("bar: {err}")))?;
    Ok(bar)
}

fn build_bar_gpu_series(
    values: &GpuTensorHandle,
    style: &BarStyle,
) -> BuiltinResult<Vec<BarChart>> {
    let context = runmat_plot::shared_wgpu_context()
        .ok_or_else(|| bar_err("bar: plotting GPU context unavailable"))?;
    let exported = runmat_accelerate_api::export_wgpu_buffer(values)
        .ok_or_else(|| bar_err("bar: unable to export GPU values"))?;
    let shape = BarMatrixShape::from_handle(values)?;
    if shape.rows == 0 {
        return Err(bar_err("bar: input cannot be empty"));
    }
    if exported.len != shape.rows * shape.cols {
        return Err(bar_err("bar: gpuArray shape mismatch"));
    }
    let scalar = ScalarType::from_is_f64(exported.precision == ProviderPrecision::F64);
    let inputs = BarGpuInputs {
        values_buffer: exported.buffer.clone(),
        row_count: shape.rows as u32,
        scalar,
    };
    if shape.cols == 1 {
        let params = BarGpuParams {
            color: style.face_rgba(),
            bar_width: style.bar_width,
            series_index: 0,
            series_count: 1,
            group_index: 0,
            group_count: 1,
            orientation: BarOrientation::Vertical,
            layout: BarLayoutMode::Grouped,
        };
        let gpu_vertices = runmat_plot::gpu::bar::pack_vertices_from_values(
            &context.device,
            &context.queue,
            &inputs,
            &params,
        )
        .map_err(|e| bar_err(format!("bar: failed to build GPU vertices: {e}")))?;
        let labels: Vec<String> = (1..=shape.rows).map(|idx| format!("{idx}")).collect();
        let bounds = build_bar_gpu_bounds(values, shape.rows, params.bar_width)?;
        let vertex_count = gpu_vertices.vertex_count;
        let chart = BarChart::from_gpu_buffer(
            labels,
            shape.rows,
            gpu_vertices,
            vertex_count,
            bounds,
            params.color,
            params.bar_width,
        );
        return Ok(vec![chart]);
    }
    build_bar_gpu_matrix_charts(&context, values, &inputs, shape, style)
}

fn build_bar_gpu_matrix_charts(
    context: &runmat_plot::SharedWgpuContext,
    values: &GpuTensorHandle,
    inputs: &BarGpuInputs,
    shape: BarMatrixShape,
    style: &BarStyle,
) -> BuiltinResult<Vec<BarChart>> {
    let labels: Vec<String> = (1..=shape.rows).map(|idx| format!("{idx}")).collect();
    let layout_mode = match style.layout {
        BarLayout::Grouped => BarLayoutMode::Grouped,
        BarLayout::Stacked => BarLayoutMode::Stacked,
    };
    let bounds = if style.layout == BarLayout::Stacked {
        build_stacked_bar_gpu_bounds(values, shape.rows, shape.cols, style.bar_width)?
    } else {
        build_bar_gpu_bounds(values, shape.rows, style.bar_width)?
    };
    let mut charts = Vec::with_capacity(shape.cols);
    for col in 0..shape.cols {
        let params = BarGpuParams {
            color: style.face_rgba(),
            bar_width: style.bar_width,
            series_index: col as u32,
            series_count: shape.cols as u32,
            group_index: if style.layout == BarLayout::Stacked {
                0
            } else {
                col as u32
            },
            group_count: if style.layout == BarLayout::Stacked {
                1
            } else {
                shape.cols as u32
            },
            orientation: BarOrientation::Vertical,
            layout: layout_mode,
        };
        let gpu_vertices = runmat_plot::gpu::bar::pack_vertices_from_values(
            &context.device,
            &context.queue,
            inputs,
            &params,
        )
        .map_err(|e| bar_err(format!("bar: failed to build GPU vertices: {e}")))?;
        let vertex_count = gpu_vertices.vertex_count;
        let mut chart = BarChart::from_gpu_buffer(
            labels.clone(),
            shape.rows,
            gpu_vertices,
            vertex_count,
            bounds,
            params.color,
            params.bar_width,
        );
        chart = if style.layout == BarLayout::Stacked {
            chart.with_group(0, 1)
        } else {
            chart.with_group(col, shape.cols)
        };
        charts.push(chart);
    }
    Ok(charts)
}

#[derive(Clone, Copy)]
struct BarMatrixShape {
    rows: usize,
    cols: usize,
}

impl BarMatrixShape {
    fn from_handle(handle: &GpuTensorHandle) -> BuiltinResult<Self> {
        if handle.shape.is_empty() {
            return Err(bar_err("bar: input cannot be empty"));
        }
        if handle.shape.len() == 1 {
            return Ok(Self {
                rows: handle.shape[0],
                cols: 1,
            });
        }
        if handle.shape.len() != 2 {
            return Err(bar_err("bar: matrix inputs must be 2-D"));
        }
        let rows = handle.shape[0];
        let cols = handle.shape[1];
        if rows == 0 || cols == 0 {
            return Err(bar_err("bar: input cannot be empty"));
        }
        Ok(Self { rows, cols })
    }
}

fn build_bar_gpu_bounds(
    values: &GpuTensorHandle,
    rows: usize,
    bar_width: f32,
) -> BuiltinResult<BoundingBox> {
    let (min_y, max_y) = axis_bounds(values, "bar")?;
    let min_y = min_y.min(0.0);
    let max_y = max_y.max(0.0);
    let min_x = 1.0 - bar_width * 0.5;
    let max_x = rows as f32 + bar_width * 0.5;
    Ok(BoundingBox::new(
        Vec3::new(min_x, min_y, 0.0),
        Vec3::new(max_x, max_y, 0.0),
    ))
}

fn build_stacked_bar_gpu_bounds(
    values: &GpuTensorHandle,
    rows: usize,
    cols: usize,
    bar_width: f32,
) -> BuiltinResult<BoundingBox> {
    let tensor = gather_tensor_from_gpu(values.clone(), "bar")?;
    if tensor.data.len() != rows * cols {
        return Err(bar_err("bar: gpuArray shape mismatch"));
    }
    let mut pos = vec![0.0f64; rows];
    let mut neg = vec![0.0f64; rows];
    let mut max_pos = 0.0f64;
    let mut min_neg = 0.0f64;
    for col in 0..cols {
        for row in 0..rows {
            let value = tensor.data[row + col * rows];
            if !value.is_finite() {
                continue;
            }
            if value >= 0.0 {
                pos[row] += value;
                if pos[row] > max_pos {
                    max_pos = pos[row];
                }
            } else {
                neg[row] += value;
                if neg[row] < min_neg {
                    min_neg = neg[row];
                }
            }
        }
    }
    let min_y = min_neg.min(0.0) as f32;
    let max_y = max_pos.max(0.0) as f32;
    let min_x = 1.0 - bar_width * 0.5;
    let max_x = rows as f32 + bar_width * 0.5;
    Ok(BoundingBox::new(
        Vec3::new(min_x, min_y, 0.0),
        Vec3::new(max_x, max_y, 0.0),
    ))
}

enum BarInput {
    Host(Tensor),
    Gpu(GpuTensorHandle),
}

impl BarInput {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::GpuTensor(handle) => Ok(Self::Gpu(handle)),
            other => {
                let tensor = Tensor::try_from(&other).map_err(|e| bar_err(format!("bar: {e}")))?;
                Ok(Self::Host(tensor))
            }
        }
    }

    fn gpu_handle(&self) -> Option<&GpuTensorHandle> {
        match self {
            Self::Gpu(handle) => Some(handle),
            Self::Host(_) => None,
        }
    }

    fn into_tensor(self, context: &'static str) -> BuiltinResult<Tensor> {
        match self {
            Self::Host(tensor) => Ok(tensor),
            Self::Gpu(handle) => gather_tensor_from_gpu(handle, context),
        }
    }
}

async fn gather_tensor_from_gpu_async(
    handle: GpuTensorHandle,
    context: &'static str,
) -> BuiltinResult<Tensor> {
    let value = Value::GpuTensor(handle);
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, context))?;
    Tensor::try_from(&gathered).map_err(|e| bar_err(format!("{context}: {e}")))
}

fn gather_tensor_from_gpu(handle: GpuTensorHandle, context: &'static str) -> BuiltinResult<Tensor> {
    block_on(gather_tensor_from_gpu_async(handle, context))
}

pub(crate) fn apply_bar_style(bar: &mut BarChart, style: &BarStyle, default_label: &str) {
    if style.face_color_flat {
        let colors = generate_flat_colors(bar.bar_count(), style.face_alpha);
        bar.set_per_bar_colors(colors);
        bar.set_bar_width(style.bar_width);
    } else {
        bar.clear_per_bar_colors();
        let face = style.face_rgba();
        bar.apply_face_style(face, style.bar_width);
    }

    let outline = style.edge_rgba();
    bar.apply_outline_style(outline, style.line_width);

    if let Some(label) = &style.label {
        bar.label = Some(label.clone());
    } else if bar.label.is_none() {
        bar.label = Some(default_label.to_string());
    }
}

fn generate_flat_colors(count: usize, alpha: f32) -> Vec<Vec4> {
    let mut colors = Vec::with_capacity(count);
    if count == 0 {
        return colors;
    }
    for i in 0..count {
        let base = MATLAB_COLOR_ORDER[i % MATLAB_COLOR_ORDER.len()];
        colors.push(Vec4::new(base.x, base.y, base.z, alpha));
    }
    colors
}

fn default_series_label(index: usize, total: usize) -> String {
    if total <= 1 {
        BAR_DEFAULT_LABEL.to_string()
    } else {
        format!("Series {}", index + 1)
    }
}

enum BarTensorInput {
    Vector(Vec<f64>),
    Matrix(BarMatrixData),
}

struct BarMatrixData {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl BarMatrixData {
    fn value(&self, row: usize, col: usize) -> f64 {
        self.data[row + col * self.rows]
    }
}

fn tensor_to_bar_input(tensor: Tensor) -> BuiltinResult<BarTensorInput> {
    if tensor.shape.is_empty() {
        return Err(bar_err("bar: input cannot be empty"));
    }
    if tensor.shape.len() == 1 || tensor.cols <= 1 {
        return Ok(BarTensorInput::Vector(numeric_vector(tensor)));
    }
    if tensor.shape.len() != 2 {
        return Err(bar_err("bar: matrix inputs must be 2-D"));
    }
    let rows = tensor.shape[0];
    let cols = tensor.shape[1];
    if rows == 0 || cols == 0 {
        return Err(bar_err("bar: input cannot be empty"));
    }
    if rows * cols != tensor.data.len() {
        return Err(bar_err("bar: matrix inputs must be dense numeric arrays"));
    }
    Ok(BarTensorInput::Matrix(BarMatrixData {
        rows,
        cols,
        data: tensor.data,
    }))
}

fn build_bar_series_from_tensor(tensor: Tensor, style: &BarStyle) -> BuiltinResult<Vec<BarChart>> {
    match tensor_to_bar_input(tensor)? {
        BarTensorInput::Vector(values) => {
            let bar = build_bar_chart(values)?;
            Ok(vec![bar])
        }
        BarTensorInput::Matrix(matrix) => build_bar_series_from_matrix(matrix, style),
    }
}

fn build_bar_series_from_matrix(
    matrix: BarMatrixData,
    style: &BarStyle,
) -> BuiltinResult<Vec<BarChart>> {
    if matrix.cols == 0 {
        return Err(bar_err("bar: input cannot be empty"));
    }
    let labels: Vec<String> = (1..=matrix.rows).map(|idx| format!("{idx}")).collect();
    let mut charts = Vec::with_capacity(matrix.cols);
    let mut pos_offsets = vec![0.0f64; matrix.rows];
    let mut neg_offsets = vec![0.0f64; matrix.rows];
    for col in 0..matrix.cols {
        let mut values = Vec::with_capacity(matrix.rows);
        for row in 0..matrix.rows {
            values.push(matrix.value(row, col));
        }
        let mut chart = BarChart::new(labels.clone(), values.clone())
            .map_err(|err| bar_err(format!("bar: {err}")))?;
        if style.layout == BarLayout::Stacked {
            let offsets = compute_stack_offsets(&values, &mut pos_offsets, &mut neg_offsets);
            chart = chart.with_stack_offsets(offsets).with_group(0, 1);
        } else {
            chart = chart.with_group(col, matrix.cols);
        }
        charts.push(chart);
    }
    Ok(charts)
}

fn compute_stack_offsets(
    values: &[f64],
    pos_offsets: &mut [f64],
    neg_offsets: &mut [f64],
) -> Vec<f64> {
    let mut offsets = vec![0.0f64; values.len()];
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            offsets[idx] = 0.0;
            continue;
        }
        if value >= 0.0 {
            offsets[idx] = pos_offsets[idx];
            pos_offsets[idx] += value;
        } else {
            offsets[idx] = neg_offsets[idx];
            neg_offsets[idx] += value;
        }
    }
    offsets
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::plotting::tests::ensure_plot_test_env;
    use futures::executor::block_on;
    use runmat_builtins::Value;
    use runmat_builtins::{ResolveContext, Type};

    fn setup_plot_tests() {
        ensure_plot_test_env();
    }

    fn bar_builtin(values: Value, rest: Vec<Value>) -> BuiltinResult<String> {
        block_on(super::bar_builtin(values, rest))
    }

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor {
            data: data.to_vec(),
            shape: vec![data.len()],
            rows: data.len(),
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        }
    }

    fn matrix_tensor(data: &[f64], rows: usize, cols: usize) -> Tensor {
        Tensor {
            data: data.to_vec(),
            shape: vec![rows, cols],
            rows,
            cols,
            dtype: runmat_builtins::NumericDType::F64,
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bar_requires_non_empty_input() {
        setup_plot_tests();
        assert!(build_bar_chart(vec![]).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bar_builtin_matches_backend_contract() {
        setup_plot_tests();
        let out = bar_builtin(Value::Tensor(tensor_from(&[1.0, 2.0, 3.0])), Vec::new());
        if let Err(err) = out {
            let msg_lower = err.to_string().to_lowercase();
            assert!(
                msg_lower.contains("plotting is unavailable")
                    || msg_lower.contains("non-main thread"),
                "unexpected error: {err}"
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bar_parser_handles_stacked_flag() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style =
            parse_bar_style_args("bar", &[Value::String("stacked".into())], defaults).unwrap();
        assert_eq!(style.layout, BarLayout::Stacked);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bar_series_from_matrix_grouped() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let tensor = matrix_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let style = parse_bar_style_args("bar", &[], defaults).unwrap();
        let charts = build_bar_series_from_tensor(tensor, &style).unwrap();
        assert_eq!(charts.len(), 2);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bar_series_from_matrix_stacked() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style =
            parse_bar_style_args("bar", &[Value::String("stacked".into())], defaults).unwrap();
        let tensor = matrix_tensor(&[1.0, -2.0, 3.0, 4.0], 2, 2);
        let charts = build_bar_series_from_tensor(tensor, &style).unwrap();
        assert_eq!(charts.len(), 2);
    }

    #[test]
    fn bar_type_is_string() {
        assert_eq!(
            string_type(&[Type::tensor()], &ResolveContext::new(Vec::new())),
            Type::String
        );
    }
}
