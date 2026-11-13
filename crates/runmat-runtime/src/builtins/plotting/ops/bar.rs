//! MATLAB-compatible `bar` and `barh` builtins.

use runmat_builtins::BuiltinErrorDescriptor;
pub use runmat_builtins::{
    BARH_ERROR_INTERNAL, BARH_ERROR_INVALID_ARGUMENT, BAR_ERROR_INTERNAL,
    BAR_ERROR_INVALID_ARGUMENT,
};

use futures::executor::block_on;
use glam::{Vec3, Vec4};
use runmat_accelerate_api::{self, GpuTensorHandle, ProviderPrecision};
use runmat_macros::runtime_builtin;
use runmat_plot::core::BoundingBox;
use runmat_plot::gpu::bar::{BarGpuInputs, BarGpuParams, BarLayoutMode, BarOrientation};
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::bar::Orientation;
use runmat_plot::plots::BarChart;
use runmat_value::{NumericScalar, Tensor, Value};
use std::convert::TryFrom;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::gather_if_needed_async;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::common::numeric_vector;
use super::state::{
    continue_line_color_cycle_after, current_axes_state, current_hold_enabled,
    line_color_for_axes_series_index, next_line_color_for_axes, render_active_plot,
    PlotRenderOptions,
};
use super::style::{parse_bar_style_args, BarLayout, BarStyle, BarStyleDefaults};

const BUILTIN_NAME: &str = "bar";
const BARH_BUILTIN_NAME: &str = "barh";

fn bar_descriptor_error(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    detail: Option<impl AsRef<str>>,
) -> RuntimeError {
    let message = match detail {
        Some(detail) => format!("{}: {}", error.message, detail.as_ref()),
        None => error.message.to_string(),
    };
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[derive(Clone, Copy)]
struct BarRenderConfig {
    builtin_name: &'static str,
    orientation: Orientation,
    gpu_orientation: BarOrientation,
    invalid_error: &'static BuiltinErrorDescriptor,
    internal_error: &'static BuiltinErrorDescriptor,
}

impl BarRenderConfig {
    fn invalid(self, detail: impl AsRef<str>) -> RuntimeError {
        bar_descriptor_error(self.builtin_name, self.invalid_error, Some(detail))
    }

    fn internal(self, detail: impl AsRef<str>) -> RuntimeError {
        bar_descriptor_error(self.builtin_name, self.internal_error, Some(detail))
    }
}

const BAR_CONFIG: BarRenderConfig = BarRenderConfig {
    builtin_name: BUILTIN_NAME,
    orientation: Orientation::Vertical,
    gpu_orientation: BarOrientation::Vertical,
    invalid_error: &BAR_ERROR_INVALID_ARGUMENT,
    internal_error: &BAR_ERROR_INTERNAL,
};

const BARH_CONFIG: BarRenderConfig = BarRenderConfig {
    builtin_name: BARH_BUILTIN_NAME,
    orientation: Orientation::Horizontal,
    gpu_orientation: BarOrientation::Horizontal,
    invalid_error: &BARH_ERROR_INVALID_ARGUMENT,
    internal_error: &BARH_ERROR_INTERNAL,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::bar")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "bar",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    // Plotting is a sink, but can consume gpuArray inputs zero-copy when a shared WGPU context exists.
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Bar charts terminate fusion graphs; gpuArray inputs may remain on device when shared plotting context is installed.",
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::bar")]
pub const BARH_GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "barh",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Horizontal bar charts terminate fusion graphs; gpuArray inputs may remain on device through the shared bar vertex-packing path.",
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::bar")]
pub const BARH_FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "barh",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "barh performs I/O and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "bar",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::bar"
)]
pub async fn bar_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::BAR_DESCRIPTOR,
        &runmat_builtins::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
        "bar",
    )?;
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "bar")?;
    // Keep the original rooted owner alive even when helpers extract native data.
    let _arguments = args.fork();
    work.finish(bar_impl(args, BAR_CONFIG, &work).await)
}

#[runtime_builtin(
    name = "barh",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::bar"
)]
pub async fn barh_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::BARH_DESCRIPTOR,
        &runmat_builtins::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
        "barh",
    )?;
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "barh")?;
    // Keep the original rooted owner alive even when helpers extract native data.
    let _arguments = args.fork();
    work.finish(bar_impl(args, BARH_CONFIG, &work).await)
}

async fn bar_impl(
    args: crate::call::BuiltinArguments,
    config: BarRenderConfig,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<Value> {
    let (x_values, values, rest) = parse_bar_call_args(args, config)?;
    let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
    let style = parse_bar_style_args(config.builtin_name, &rest, defaults)?;
    let mut input = Some(BarInput::from_value(x_values, values, config).await?);
    let opts = PlotRenderOptions::default();
    let plot_indices = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let plot_indices_slot = std::rc::Rc::clone(&plot_indices);
    let series_total = std::rc::Rc::new(std::cell::RefCell::new(0usize));
    let series_total_slot = std::rc::Rc::clone(&series_total);
    let target_axes = current_axes_state().active_index;
    let hold_enabled = current_hold_enabled();
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    work.check()?;
    let render_result = render_active_plot(config.builtin_name, opts, move |figure, axes| {
        let style = style.clone();
        let arg = input.take().expect("bar input consumed once");
        if !style.requires_cpu_path() {
            if let Some(handle) = arg.gpu_handle() {
                match build_bar_gpu_series(arg.x_tensor().cloned(), handle, &style, config)? {
                    Some(charts) if !charts.is_empty() => {
                        let total = charts.len();
                        *series_total_slot.borrow_mut() = total;
                        for (idx, mut bar) in charts.into_iter().enumerate() {
                            let default_label = default_series_label(idx, total);
                            let style = series_bar_style(&style, axes, hold_enabled, idx);
                            apply_bar_style(&mut bar, &style, &default_label);
                            let plot_index = figure.add_bar_chart_on_axes(bar, axes);
                            plot_indices_slot.borrow_mut().push((axes, plot_index));
                        }
                        return Ok(());
                    }
                    Some(_) | None => {}
                }
            }
        }
        let (x_tensor, tensor) = arg.into_tensors(config)?;
        let charts = build_bar_series_from_tensor(x_tensor, tensor, &style, config)?;
        let total = charts.len();
        *series_total_slot.borrow_mut() = total;
        for (idx, mut bar) in charts.into_iter().enumerate() {
            let default_label = default_series_label(idx, total);
            let style = series_bar_style(&style, axes, hold_enabled, idx);
            apply_bar_style(&mut bar, &style, &default_label);
            let plot_index = figure.add_bar_chart_on_axes(bar, axes);
            plot_indices_slot.borrow_mut().push((axes, plot_index));
        }
        Ok(())
    });
    if !hold_enabled {
        // Rendering cleared the axes and reset the color order; resume it after these series.
        continue_line_color_cycle_after(target_axes, *series_total.borrow());
    }
    if let Err(err) = render_result {
        if !crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Err(err);
        }
    }
    let handles = plot_indices
        .borrow()
        .iter()
        .map(|(axes, plot_index)| {
            crate::builtins::plotting::state::register_bar_handle(figure_handle, *axes, *plot_index)
        })
        .collect();
    Ok(super::line::handles_value(handles))
}

const DEFAULT_BAR_WIDTH: f32 = 0.8;

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

fn parse_bar_call_args(
    args: crate::call::BuiltinArguments,
    config: BarRenderConfig,
) -> BuiltinResult<(Option<Value>, Value, Vec<Value>)> {
    if args.is_empty() {
        return Err(config.invalid("expected at least one input"));
    }
    let mut it = args.into_iter();
    let first = it.next().expect("first");
    let Some(second) = it.next() else {
        return Ok((None, first, Vec::new()));
    };
    let trailing: Vec<Value> = it.collect();
    if matches!(second, Value::String(_) | Value::CharArray(_)) {
        let mut rest = vec![second];
        rest.extend(trailing);
        return Ok((None, first, rest));
    }
    if is_positional_bar_width_candidate(&second, &trailing) {
        let mut rest = vec![second];
        rest.extend(trailing);
        return Ok((None, first, rest));
    }
    Ok((Some(first), second, trailing))
}

fn is_positional_bar_width_candidate(second: &Value, trailing: &[Value]) -> bool {
    if !is_numeric_scalar_value(second) {
        return false;
    }
    trailing
        .first()
        .is_none_or(|value| matches!(value, Value::String(_) | Value::CharArray(_)))
}

fn is_numeric_scalar_value(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) => true,
        Value::Tensor(tensor) => tensor_utils::is_scalar_tensor(tensor),
        _ => false,
    }
}

fn build_bar_chart_with_x(
    x_tensor: Option<Tensor>,
    values: Vec<f64>,
    config: BarRenderConfig,
) -> BuiltinResult<BarChart> {
    if values.is_empty() {
        return Err(config.invalid("input cannot be empty"));
    }
    let labels = category_labels_from_x(x_tensor, values.len(), config)?;
    BarChart::new(labels, values)
        .map(|chart| chart.with_orientation(config.orientation))
        .map_err(|err| config.internal(&err))
}

fn build_bar_gpu_series(
    x_tensor: Option<Tensor>,
    values: &GpuTensorHandle,
    style: &BarStyle,
    config: BarRenderConfig,
) -> BuiltinResult<Option<Vec<BarChart>>> {
    let x_count = x_tensor.as_ref().map(numeric_tensor_len);
    let shape = BarMatrixShape::from_handle(values, x_count, config)?;
    if shape.rows == 0 {
        return Err(config.invalid("input cannot be empty"));
    }
    for dimension in [shape.rows, shape.cols, shape.source_rows] {
        u32::try_from(dimension)
            .map_err(|_| config.invalid("GPU dimensions exceed supported range"))?;
    }
    let expected = shape
        .rows
        .checked_mul(shape.cols)
        .ok_or_else(|| config.invalid("GPU dimensions overflow"))?;
    if super::gpu_helpers::GpuBufferExports::element_count(values, config.builtin_name)? != expected
    {
        return Err(config.invalid("gpuArray shape mismatch"));
    }
    let labels = category_labels_from_x(x_tensor, shape.rows, config)?;
    let Some((context, mut gpu_exports)) =
        super::gpu_helpers::GpuBufferExports::prepare(&[values], config.builtin_name)?
    else {
        return Ok(None);
    };
    let result = (|| {
        let exported = gpu_exports
            .export(values, config.builtin_name)?
            .ok_or_else(|| config.internal("unable to export GPU values"))?;
        if exported.len != shape.rows * shape.cols {
            return Err(config.invalid("gpuArray shape mismatch"));
        }
        let scalar = ScalarType::from_is_f64(exported.precision == ProviderPrecision::F64);
        let inputs = BarGpuInputs {
            values_buffer: exported.clone().into(),
            row_count: shape.rows as u32,
            scalar,
        };
        if shape.cols == 1 {
            let params = BarGpuParams {
                color: style.face_rgba(),
                bar_width: style.bar_width,
                series_index: 0,
                series_count: 1,
                source_row_count: shape.source_rows as u32,
                transpose_source: shape.transposed,
                group_index: 0,
                group_count: 1,
                orientation: config.gpu_orientation,
                layout: BarLayoutMode::Grouped,
            };
            let gpu_vertices = context
                .access
                .with_shared(|emission| {
                    runmat_plot::gpu::bar::pack_vertices_from_values(emission, &inputs, &params)
                })
                .map_err(|source| {
                    super::plotting_error_with_source(
                        config.builtin_name,
                        format!(
                            "{}: GPU emission unavailable: {source}",
                            config.builtin_name
                        ),
                        source,
                    )
                })?
                .map_err(|e| config.internal(format!("failed to build GPU vertices: {e}")))?;
            gpu_exports.record_completed_effect();
            let bounds = build_bar_gpu_bounds(
                values,
                shape.rows,
                params.bar_width,
                config,
                &gpu_exports.protected_handles(),
            )?;
            let vertex_count = gpu_vertices.vertex_count;
            let chart = BarChart::from_gpu_buffer(
                labels,
                shape.rows,
                gpu_vertices,
                vertex_count,
                bounds,
                params.color,
                params.bar_width,
            )
            .with_orientation(config.orientation)
            .with_gpu_source_layout(
                inputs.clone(),
                0,
                1,
                shape.source_rows,
                shape.transposed,
            );
            return Ok(vec![chart]);
        }
        build_bar_gpu_matrix_charts(
            &context,
            values,
            &inputs,
            labels,
            shape,
            style,
            config,
            &mut gpu_exports,
        )
    })();
    gpu_exports.complete(result, config.builtin_name).map(Some)
}

fn build_bar_gpu_matrix_charts(
    context: &runmat_plot::SharedWgpuContext,
    values: &GpuTensorHandle,
    inputs: &BarGpuInputs,
    labels: Vec<String>,
    shape: BarMatrixShape,
    style: &BarStyle,
    config: BarRenderConfig,
    gpu_exports: &mut super::gpu_helpers::GpuBufferExports,
) -> BuiltinResult<Vec<BarChart>> {
    let layout_mode = match style.layout {
        BarLayout::Grouped => BarLayoutMode::Grouped,
        BarLayout::Stacked => BarLayoutMode::Stacked,
    };
    let bounds = if style.layout == BarLayout::Stacked {
        build_stacked_bar_gpu_bounds(values, shape, style.bar_width, config)?
    } else {
        build_bar_gpu_bounds(
            values,
            shape.rows,
            style.bar_width,
            config,
            &gpu_exports.protected_handles(),
        )?
    };
    gpu_exports.record_completed_effect();
    let mut charts = Vec::with_capacity(shape.cols);
    for col in 0..shape.cols {
        let params = BarGpuParams {
            color: style.face_rgba(),
            bar_width: style.bar_width,
            series_index: col as u32,
            series_count: shape.cols as u32,
            source_row_count: shape.source_rows as u32,
            transpose_source: shape.transposed,
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
            orientation: config.gpu_orientation,
            layout: layout_mode,
        };
        let gpu_vertices = context
            .access
            .with_shared(|emission| {
                runmat_plot::gpu::bar::pack_vertices_from_values(emission, inputs, &params)
            })
            .map_err(|source| {
                super::plotting_error_with_source(
                    config.builtin_name,
                    format!(
                        "{}: GPU emission unavailable: {source}",
                        config.builtin_name
                    ),
                    source,
                )
            })?
            .map_err(|e| config.internal(format!("failed to build GPU vertices: {e}")))?;
        let vertex_count = gpu_vertices.vertex_count;
        let mut chart = BarChart::from_gpu_buffer(
            labels.clone(),
            shape.rows,
            gpu_vertices,
            vertex_count,
            bounds,
            params.color,
            params.bar_width,
        )
        .with_orientation(config.orientation)
        .with_gpu_source_layout(
            inputs.clone(),
            col,
            shape.cols,
            shape.source_rows,
            shape.transposed,
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
    source_rows: usize,
    transposed: bool,
}

impl BarMatrixShape {
    fn from_handle(
        handle: &GpuTensorHandle,
        x_count: Option<usize>,
        config: BarRenderConfig,
    ) -> BuiltinResult<Self> {
        if handle.shape().is_empty() {
            return Err(config.invalid("input cannot be empty"));
        }
        if handle.shape().len() == 1 {
            return Ok(Self {
                rows: handle.shape()[0],
                cols: 1,
                source_rows: handle.shape()[0],
                transposed: false,
            });
        }
        if handle.shape().len() != 2 {
            return Err(config.invalid("matrix inputs must be 2-D"));
        }
        let rows = handle.shape()[0];
        let cols = handle.shape()[1];
        if rows == 0 || cols == 0 {
            return Err(config.invalid("input cannot be empty"));
        }
        if rows == 1 && cols > 1 {
            if x_count == Some(1) {
                return Ok(Self {
                    rows,
                    cols,
                    source_rows: rows,
                    transposed: false,
                });
            }
            return Ok(Self {
                rows: cols,
                cols: 1,
                source_rows: cols,
                transposed: false,
            });
        }
        if cols == 1 {
            return Ok(Self {
                rows,
                cols: 1,
                source_rows: rows,
                transposed: false,
            });
        }
        if let Some(count) = x_count {
            if count == rows {
                return Ok(Self {
                    rows,
                    cols,
                    source_rows: rows,
                    transposed: false,
                });
            }
            if count == cols {
                return Ok(Self {
                    rows: cols,
                    cols: rows,
                    source_rows: rows,
                    transposed: true,
                });
            }
        }
        Ok(Self {
            rows,
            cols,
            source_rows: rows,
            transposed: false,
        })
    }
}

fn numeric_tensor_len(tensor: &Tensor) -> usize {
    tensor_utils::tensor_element_len(tensor)
}

fn category_labels_from_x(
    x_tensor: Option<Tensor>,
    rows: usize,
    config: BarRenderConfig,
) -> BuiltinResult<Vec<String>> {
    let Some(x) = x_tensor else {
        return Ok((1..=rows).map(|idx| format!("{idx}")).collect());
    };
    if numeric_tensor_len(&x) != rows {
        return Err(config.invalid(format!(
            "X must contain {rows} category position{}",
            if rows == 1 { "" } else { "s" }
        )));
    }
    let labels = (0..rows)
        .map(|index| {
            x.numeric_value_at(index)
                .map(format_numeric_label)
                .ok_or_else(|| {
                    config.internal(format!(
                        "X {} storage is unavailable at element {index}",
                        x.numeric_dtype().class_name()
                    ))
                })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let mut unique = std::collections::HashSet::with_capacity(labels.len());
    if labels.iter().any(|label| !unique.insert(label.clone())) {
        return Err(config.invalid("X values must be unique"));
    }
    Ok(labels)
}

fn build_bar_gpu_bounds(
    values: &GpuTensorHandle,
    rows: usize,
    bar_width: f32,
    config: BarRenderConfig,
    protected: &[&GpuTensorHandle],
) -> BuiltinResult<BoundingBox> {
    let (min_value, max_value) = futures::executor::block_on(
        super::gpu_helpers::axis_bounds_protecting(values, protected, config.builtin_name),
    )?;
    let min_value = min_value.min(0.0);
    let max_value = max_value.max(0.0);
    let min_category = 1.0 - bar_width * 0.5;
    let max_category = rows as f32 + bar_width * 0.5;
    Ok(match config.orientation {
        Orientation::Vertical => BoundingBox::new(
            Vec3::new(min_category, min_value, 0.0),
            Vec3::new(max_category, max_value, 0.0),
        ),
        Orientation::Horizontal => BoundingBox::new(
            Vec3::new(min_value, min_category, 0.0),
            Vec3::new(max_value, max_category, 0.0),
        ),
    })
}

fn build_stacked_bar_gpu_bounds(
    values: &GpuTensorHandle,
    shape: BarMatrixShape,
    bar_width: f32,
    config: BarRenderConfig,
) -> BuiltinResult<BoundingBox> {
    let tensor = gather_tensor_from_gpu(values.clone(), config)?;
    let expected_len = shape.source_rows
        * if shape.transposed {
            shape.rows
        } else {
            shape.cols
        };
    if tensor_utils::tensor_element_len(&tensor) != expected_len {
        return Err(config.invalid("gpuArray shape mismatch"));
    }
    let tensor_values = tensor_utils::tensor_values_f64_cow(&tensor);
    let mut pos = vec![0.0f64; shape.rows];
    let mut neg = vec![0.0f64; shape.rows];
    let mut max_pos = 0.0f64;
    let mut min_neg = 0.0f64;
    for col in 0..shape.cols {
        for row in 0..shape.rows {
            let value_index = if shape.transposed {
                row * shape.source_rows + col
            } else {
                row + col * shape.source_rows
            };
            let value = tensor_values[value_index];
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
    let min_value = min_neg.min(0.0) as f32;
    let max_value = max_pos.max(0.0) as f32;
    let min_category = 1.0 - bar_width * 0.5;
    let max_category = shape.rows as f32 + bar_width * 0.5;
    Ok(match config.orientation {
        Orientation::Vertical => BoundingBox::new(
            Vec3::new(min_category, min_value, 0.0),
            Vec3::new(max_category, max_value, 0.0),
        ),
        Orientation::Horizontal => BoundingBox::new(
            Vec3::new(min_value, min_category, 0.0),
            Vec3::new(max_value, max_category, 0.0),
        ),
    })
}

enum BarInput {
    Host {
        x: Option<Tensor>,
        y: Tensor,
    },
    Gpu {
        x: Option<Tensor>,
        y: GpuTensorHandle,
    },
}

impl BarInput {
    async fn from_value(
        x: Option<Value>,
        value: Value,
        config: BarRenderConfig,
    ) -> BuiltinResult<Self> {
        match (x, value) {
            (x, Value::GpuTensor(handle)) => {
                let x = match x {
                    Some(Value::GpuTensor(handle)) => {
                        Some(gather_tensor_from_gpu_async(handle, config).await?)
                    }
                    Some(value) => Some(bar_host_coordinate(value, config)?),
                    None => None,
                };
                if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
                    let y = gather_tensor_from_gpu_async(handle, config).await?;
                    Ok(Self::Host { x, y })
                } else {
                    Ok(Self::Gpu { x, y: handle })
                }
            }
            (x, other) => {
                let y = bar_host_coordinate(other, config)?;
                let x = match x {
                    Some(value) => Some(bar_host_coordinate(value, config)?),
                    None => None,
                };
                Ok(Self::Host { x, y })
            }
        }
    }

    fn gpu_handle(&self) -> Option<&GpuTensorHandle> {
        match self {
            Self::Gpu { y, .. } => Some(y),
            Self::Host { .. } => None,
        }
    }

    fn x_tensor(&self) -> Option<&Tensor> {
        match self {
            Self::Gpu { x, .. } | Self::Host { x, .. } => x.as_ref(),
        }
    }

    fn into_tensors(self, config: BarRenderConfig) -> BuiltinResult<(Option<Tensor>, Tensor)> {
        match self {
            Self::Host { x, y } => Ok((x, y)),
            Self::Gpu { x, y } => Ok((x, gather_tensor_from_gpu(y, config)?)),
        }
    }
}

fn bar_host_coordinate(value: Value, config: BarRenderConfig) -> BuiltinResult<Tensor> {
    // Preserve the prior logical-array/scalar and host-Y/device-X restrictions.
    // Num/Int normalization is the same native owner as explicit chart pairs.
    if matches!(value, Value::Bool(_) | Value::GpuTensor(_)) {
        return Err(config.invalid("expected host real numeric coordinate data"));
    }
    match super::op_common::line_inputs::NumericInput::from_value(value, config.builtin_name) {
        Ok(super::op_common::line_inputs::NumericInput::Host(tensor)) => Ok(tensor),
        Ok(super::op_common::line_inputs::NumericInput::Gpu(_)) => {
            unreachable!("device input rejected")
        }
        Err(source) => Err(build_runtime_error(format!(
            "{}: {}",
            config.invalid_error.message,
            source.message()
        ))
        .with_builtin(config.builtin_name)
        .with_identifier(
            config
                .invalid_error
                .identifier
                .unwrap_or("RunMat:bar:InvalidArgument"),
        )
        .with_source(source)
        .build()),
    }
}

async fn gather_tensor_from_gpu_async(
    handle: GpuTensorHandle,
    config: BarRenderConfig,
) -> BuiltinResult<Tensor> {
    let value = Value::GpuTensor(handle);
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, config.builtin_name))?;
    Tensor::try_from(&gathered)
        .map_err(|e| config.internal(format!("{}: {e}", config.builtin_name)))
}

fn gather_tensor_from_gpu(
    handle: GpuTensorHandle,
    config: BarRenderConfig,
) -> BuiltinResult<Tensor> {
    block_on(gather_tensor_from_gpu_async(handle, config))
}

/// The style for one series: without a face color in the call, the series takes its
/// color from the axes color order (the next color under `hold on`), as in MATLAB.
pub(crate) fn series_bar_style(
    style: &BarStyle,
    axes: usize,
    hold_enabled: bool,
    series_index: usize,
) -> BarStyle {
    let mut style = style.clone();
    if !style.face_color_explicit && !style.face_color_flat {
        let color = if hold_enabled {
            next_line_color_for_axes(axes)
        } else {
            line_color_for_axes_series_index(axes, series_index)
        };
        style.face_color = Vec4::new(color.x, color.y, color.z, style.face_color.w);
    }
    style
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

    fn transpose(self) -> Self {
        let mut data = Vec::with_capacity(self.data.len());
        for col in 0..self.rows {
            for row in 0..self.cols {
                data.push(self.value(col, row));
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }
}

fn tensor_to_bar_input(tensor: Tensor, config: BarRenderConfig) -> BuiltinResult<BarTensorInput> {
    let shape = tensor.shape();
    if shape.is_empty() {
        return Err(config.invalid("input cannot be empty"));
    }
    if shape.len() == 1 || shape[0] <= 1 || shape[1] <= 1 {
        return Ok(BarTensorInput::Vector(numeric_vector(tensor)));
    }
    if shape.len() != 2 {
        return Err(config.invalid("matrix inputs must be 2-D"));
    }
    let rows = shape[0];
    let cols = shape[1];
    if rows == 0 || cols == 0 {
        return Err(config.invalid("input cannot be empty"));
    }
    if rows * cols != tensor_utils::tensor_element_len(&tensor) {
        return Err(config.invalid("matrix inputs must be dense numeric arrays"));
    }
    let data = tensor_utils::tensor_values_f64(&tensor);
    Ok(BarTensorInput::Matrix(BarMatrixData { rows, cols, data }))
}

fn build_bar_series_from_tensor(
    x_tensor: Option<Tensor>,
    tensor: Tensor,
    style: &BarStyle,
    config: BarRenderConfig,
) -> BuiltinResult<Vec<BarChart>> {
    match tensor_to_bar_input(tensor, config)? {
        BarTensorInput::Vector(values) => {
            if x_tensor
                .as_ref()
                .is_some_and(|x| numeric_tensor_len(x) == 1)
                && values.len() > 1
            {
                return build_bar_series_from_matrix(
                    x_tensor,
                    BarMatrixData {
                        rows: 1,
                        cols: values.len(),
                        data: values,
                    },
                    style,
                    config,
                );
            }
            let bar = build_bar_chart_with_x(x_tensor, values, config)?;
            Ok(vec![bar])
        }
        BarTensorInput::Matrix(matrix) => {
            build_bar_series_from_matrix(x_tensor, matrix, style, config)
        }
    }
}

fn build_bar_series_from_matrix(
    x_tensor: Option<Tensor>,
    matrix: BarMatrixData,
    style: &BarStyle,
    config: BarRenderConfig,
) -> BuiltinResult<Vec<BarChart>> {
    let x_len = x_tensor.as_ref().map(numeric_tensor_len);
    let matrix = if x_len == Some(matrix.cols) && matrix.rows != matrix.cols {
        matrix.transpose()
    } else {
        matrix
    };
    if matrix.cols == 0 {
        return Err(config.invalid("input cannot be empty"));
    }
    let labels = category_labels_from_x(x_tensor, matrix.rows, config)?;
    let mut charts = Vec::with_capacity(matrix.cols);
    let mut pos_offsets = vec![0.0f64; matrix.rows];
    let mut neg_offsets = vec![0.0f64; matrix.rows];
    for col in 0..matrix.cols {
        let mut values = Vec::with_capacity(matrix.rows);
        for row in 0..matrix.rows {
            values.push(matrix.value(row, col));
        }
        let mut chart = BarChart::new(labels.clone(), values.clone())
            .map(|chart| chart.with_orientation(config.orientation))
            .map_err(|err| config.internal(&err))?;
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

fn format_numeric_label(value: NumericScalar) -> String {
    match value {
        NumericScalar::F64(value) => format_floating_label(value),
        NumericScalar::F32(value) => format_floating_label(f64::from(value)),
        NumericScalar::I8(value) => value.to_string(),
        NumericScalar::I16(value) => value.to_string(),
        NumericScalar::I32(value) => value.to_string(),
        NumericScalar::I64(value) => value.to_string(),
        NumericScalar::U8(value) => value.to_string(),
        NumericScalar::U16(value) => value.to_string(),
        NumericScalar::U32(value) => value.to_string(),
        NumericScalar::U64(value) => value.to_string(),
    }
}

fn format_floating_label(value: f64) -> String {
    if value == 0.0 {
        "0".to_string()
    } else {
        value.to_string()
    }
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
    use runmat_builtins::{BuiltinIntegerInputAvailability, BuiltinParamType};
    use runmat_builtins::{
        BARH_DESCRIPTOR, BARH_INTEGER_CAPABILITIES, BAR_DESCRIPTOR, BAR_INPUTS_Y_WIDTH,
        BAR_INTEGER_CAPABILITIES, BAR_OUTPUT_HANDLE,
    };
    fn parse_bar_call_args(
        args: Vec<Value>,
        config: BarRenderConfig,
    ) -> BuiltinResult<(Option<Value>, Value, Vec<Value>)> {
        super::parse_bar_call_args(
            crate::builtins::common::test_support::builtin_arguments(args),
            config,
        )
    }
    use crate::builtins::common::{gpu_helpers, test_support};
    use crate::builtins::plotting::tests::ensure_plot_test_env;
    use futures::executor::block_on;
    use runmat_value::{IntegerStorage, Value};

    fn setup_plot_tests() {
        ensure_plot_test_env();
    }

    fn bar_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::bar_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
    }

    fn barh_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::barh_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
    }

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![data.len()]).expect("bar test vector")
    }

    fn matrix_tensor(data: &[f64], rows: usize, cols: usize) -> Tensor {
        Tensor::new(data.to_vec(), vec![rows, cols]).expect("bar test matrix")
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn bar_host_input_preserves_rank_zero_rejection_and_rank_one_vector() {
        let scalar = Tensor::new(vec![7.0], Vec::new()).unwrap();
        let error = match tensor_to_bar_input(scalar, BAR_CONFIG) {
            Ok(_) => panic!("rank-zero bar input is rejected"),
            Err(error) => error,
        };
        assert!(error.message().contains("input cannot be empty"));

        let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let BarTensorInput::Vector(values) = tensor_to_bar_input(row, BAR_CONFIG).unwrap() else {
            panic!("rank-one bar input is a vector")
        };
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    fn poisoned_integer_matrix(storage: IntegerStorage, rows: usize, cols: usize) -> Tensor {
        Tensor::new_integer(storage, vec![rows, cols]).expect("integer matrix")
    }

    fn all_integer_bar_storages() -> [IntegerStorage; 8] {
        [
            IntegerStorage::I8(vec![1, -2, 3, 4]),
            IntegerStorage::I16(vec![1, -2, 3, 4]),
            IntegerStorage::I32(vec![1, -2, 3, 4]),
            IntegerStorage::I64(vec![1, -2, 3, 4]),
            IntegerStorage::U8(vec![1, 2, 3, 4]),
            IntegerStorage::U16(vec![1, 2, 3, 4]),
            IntegerStorage::U32(vec![1, 2, 3, 4]),
            IntegerStorage::U64(vec![1, 2, 3, u64::MAX]),
        ]
    }

    fn row_vector_tensor(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![1, data.len()]).expect("bar test row vector")
    }

    fn gpu_handle_with_shape(shape: &[usize]) -> GpuTensorHandle {
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                shape.to_vec(),
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle
    }

    #[test]
    fn bar_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = BAR_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = bar(Y)"));
        assert!(labels.contains(&"h = bar(Y, width)"));
        assert!(labels.contains(&"h = bar(Y, width, color)"));
        assert!(labels.contains(&"h = bar(X, Y)"));
        assert!(labels.contains(&"h = bar(X, Y, width)"));
        assert!(labels.contains(&"h = bar(X, Y, Name, Value, ...)"));
        assert_eq!(BAR_OUTPUT_HANDLE[0].ty, BuiltinParamType::NumericArray);
        assert_eq!(BAR_INPUTS_Y_WIDTH[1].default, Some("0.8"));
        assert_eq!(DEFAULT_BAR_WIDTH, 0.8);
    }

    #[test]
    fn bar_integer_capabilities_cover_documented_forms_and_all_classes() {
        assert_eq!(BAR_INTEGER_CAPABILITIES.len(), 4);
        for capability in BAR_INTEGER_CAPABILITIES {
            for input in capability.inputs {
                assert_eq!(
                    input.classes,
                    crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES
                );
                assert_eq!(
                    input.availability,
                    BuiltinIntegerInputAvailability::Documented
                );
            }
        }
    }

    #[test]
    fn barh_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = BARH_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = barh(Y)"));
        assert!(labels.contains(&"h = barh(Y, width)"));
        assert!(labels.contains(&"h = barh(Y, width, color)"));
        assert!(labels.contains(&"h = barh(X, Y)"));
        assert!(labels.contains(&"h = barh(X, Y, width)"));
        assert!(labels.contains(&"h = barh(X, Y, Name, Value, ...)"));
    }

    #[test]
    fn barh_integer_capabilities_cover_documented_forms_and_all_classes() {
        assert_eq!(BARH_INTEGER_CAPABILITIES.len(), 4);
        for capability in BARH_INTEGER_CAPABILITIES {
            assert!(capability.form.starts_with("b = barh("));
            for input in capability.inputs {
                assert_eq!(
                    input.classes,
                    crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES
                );
                assert_eq!(
                    input.availability,
                    BuiltinIntegerInputAvailability::Documented
                );
            }
        }
    }

    #[test]
    fn bar_invalid_argument_uses_stable_identifier() {
        let err = block_on(super::bar_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("missing args should fail");
        assert_eq!(err.identifier(), BAR_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn barh_invalid_argument_uses_stable_identifier() {
        let err = block_on(super::barh_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("missing args should fail");
        assert_eq!(err.identifier(), BARH_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bar_requires_non_empty_input() {
        setup_plot_tests();
        assert!(build_bar_chart_with_x(None, vec![], BAR_CONFIG).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bar_builtin_matches_backend_contract() {
        let _guard = crate::builtins::plotting::tests::lock_plot_registry();
        setup_plot_tests();
        crate::builtins::plotting::state::reset_hold_state_for_run();
        crate::builtins::plotting::state::clear_figure(None).unwrap();
        let out = bar_builtin(vec![Value::Tensor(tensor_from(&[1.0, 2.0, 3.0]))]);
        if let Err(err) = out {
            assert!(
                crate::builtins::plotting::state::presentation_unavailable(&err),
                "unexpected error: {err}"
            );
        }
        let figure = crate::builtins::plotting::clone_figure(
            crate::builtins::plotting::current_figure_handle(),
        )
        .expect("bar figure exists");
        assert_eq!(figure.title.as_deref().unwrap_or_default(), "");
        assert_eq!(figure.x_label.as_deref().unwrap_or_default(), "");
        assert_eq!(figure.y_label.as_deref().unwrap_or_default(), "");
    }

    #[test]
    fn bar_matrix_returns_one_handle_per_series() {
        setup_plot_tests();
        let output = bar_builtin(vec![Value::Tensor(matrix_tensor(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            3,
            2,
        ))])
        .expect("matrix bar");
        match output {
            Value::Tensor(handles) => {
                assert_eq!(handles.shape(), [1, 2]);
                assert_eq!(handles.len(), 2);
            }
            other => panic!("expected a two-handle row vector, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn barh_builtin_matches_backend_contract() {
        setup_plot_tests();
        let out = barh_builtin(vec![Value::Tensor(tensor_from(&[1.0, 2.0, 3.0]))]);
        if let Err(err) = out {
            assert!(
                crate::builtins::plotting::state::presentation_unavailable(&err),
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

    #[test]
    fn bar_parser_treats_positional_width_as_style() {
        let (x, y, rest) = parse_bar_call_args(
            vec![
                Value::Tensor(row_vector_tensor(&[10.0, 22.0, 30.0, 42.0])),
                Value::from(0.4),
                Value::String("red".into()),
            ],
            BARH_CONFIG,
        )
        .unwrap();
        assert!(x.is_none());
        assert!(matches!(y, Value::Tensor(_)));
        assert_eq!(rest.len(), 2);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bar_series_from_matrix_grouped() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let tensor = matrix_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let style = parse_bar_style_args("bar", &[], defaults).unwrap();
        let charts = build_bar_series_from_tensor(None, tensor, &style, BAR_CONFIG).unwrap();
        assert_eq!(charts.len(), 2);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bar_series_from_matrix_reads_typed_integer_storage() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style = parse_bar_style_args("bar", &[], defaults).unwrap();
        let tensor = poisoned_integer_matrix(IntegerStorage::I16(vec![1, -2, 3, 4]), 2, 2);
        let charts = build_bar_series_from_tensor(None, tensor, &style, BAR_CONFIG).unwrap();
        assert_eq!(charts.len(), 2);
        assert_eq!(charts[0].values().unwrap(), &[1.0, -2.0]);
        assert_eq!(charts[1].values().unwrap(), &[3.0, 4.0]);
    }

    #[test]
    fn bar_accepts_all_integer_y_classes_at_the_graphics_boundary() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style = parse_bar_style_args("bar", &[], defaults).unwrap();
        for storage in all_integer_bar_storages() {
            let source = Tensor::new_integer(storage.clone(), vec![2, 2]).unwrap();
            let scalar = source
                .numeric_value_at(0)
                .unwrap()
                .into_int_value()
                .unwrap();
            for config in [BAR_CONFIG, BARH_CONFIG] {
                let input = block_on(BarInput::from_value(
                    None,
                    Value::Int(scalar.clone()),
                    config,
                ))
                .expect("native scalar Y");
                let BarInput::Host { y, .. } = input else {
                    panic!("host scalar")
                };
                assert_eq!(
                    y.integer_storage(),
                    Some(&IntegerStorage::from_scalar(scalar.clone()))
                );
                let charts =
                    build_bar_series_from_tensor(None, y, &style, config).expect("scalar chart");
                assert_eq!(charts.len(), 1);
                assert_eq!(charts[0].bar_count(), 1);
                assert_eq!(charts[0].orientation, config.orientation);
            }
            let tensor = Tensor::new_integer(storage, vec![2, 2]).expect("integer Y");
            let charts =
                build_bar_series_from_tensor(None, tensor, &style, BAR_CONFIG).expect("bar charts");
            assert_eq!(charts.len(), 2);
            assert_eq!(charts[0].bar_count(), 2);
        }
        for config in [BAR_CONFIG, BARH_CONFIG] {
            let input = block_on(BarInput::from_value(None, Value::Num(4.0), config))
                .expect("double scalar Y");
            let BarInput::Host { y, .. } = input else {
                panic!("host scalar")
            };
            assert_eq!(y.materialize_f64(), vec![4.0]);
        }
    }

    #[test]
    fn barh_accepts_all_integer_y_classes_at_the_graphics_boundary() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style = parse_bar_style_args("barh", &[], defaults).unwrap();
        for storage in all_integer_bar_storages() {
            let tensor = Tensor::new_integer(storage, vec![2, 2]).expect("integer Y");
            let charts = build_bar_series_from_tensor(None, tensor, &style, BARH_CONFIG)
                .expect("horizontal bar charts");
            assert_eq!(charts.len(), 2);
            assert_eq!(charts[0].bar_count(), 2);
            assert!(charts
                .iter()
                .all(|chart| chart.orientation == Orientation::Horizontal));
        }
    }

    #[test]
    fn bar_width_candidate_accepts_typed_integer_scalar_without_double_mirror() {
        let width =
            Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).expect("typed width");

        assert!(is_positional_bar_width_candidate(
            &Value::Tensor(width),
            &[Value::String("grouped".into())]
        ));
    }

    #[test]
    fn bar_width_and_line_width_accept_all_integer_scalar_classes() {
        let widths = [
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ];
        for storage in widths {
            let width = Value::Tensor(
                Tensor::new_integer(storage, vec![1, 1]).expect("typed integer width"),
            );
            let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
            let positional =
                parse_bar_style_args("bar", std::slice::from_ref(&width), defaults).unwrap();
            assert_eq!(positional.bar_width, 1.0);
            let line =
                parse_bar_style_args("bar", &[Value::String("LineWidth".into()), width], defaults)
                    .unwrap();
            assert_eq!(line.line_width, 1.0);
        }
    }

    #[test]
    fn bar_rejects_nonscalar_and_out_of_range_width_controls() {
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        assert!(parse_bar_style_args(
            "bar",
            &[
                Value::String("LineWidth".into()),
                Value::Tensor(tensor_from(&[1.0, 2.0]))
            ],
            defaults,
        )
        .is_err());
        assert!(parse_bar_style_args("bar", &[Value::Num(2.0)], defaults).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bar_series_from_matrix_stacked() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style =
            parse_bar_style_args("bar", &[Value::String("stacked".into())], defaults).unwrap();
        let tensor = matrix_tensor(&[1.0, -2.0, 3.0, 4.0], 2, 2);
        let charts = build_bar_series_from_tensor(None, tensor, &style, BAR_CONFIG).unwrap();
        assert_eq!(charts.len(), 2);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn barh_series_from_matrix_sets_horizontal_orientation() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style = parse_bar_style_args("barh", &[], defaults).unwrap();
        let tensor = matrix_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let charts = build_bar_series_from_tensor(None, tensor, &style, BARH_CONFIG).unwrap();
        assert_eq!(charts.len(), 2);
        assert!(charts
            .iter()
            .all(|chart| chart.orientation == Orientation::Horizontal));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn barh_stacked_series_preserves_horizontal_offsets() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style =
            parse_bar_style_args("barh", &[Value::String("stacked".into())], defaults).unwrap();
        let tensor = matrix_tensor(&[1.0, -2.0, 3.0, 4.0], 2, 2);
        let charts = build_bar_series_from_tensor(None, tensor, &style, BARH_CONFIG).unwrap();
        assert_eq!(charts.len(), 2);
        assert!(charts
            .iter()
            .all(|chart| chart.orientation == Orientation::Horizontal));
        assert_eq!(charts[1].stack_offsets().unwrap(), &[1.0, 0.0]);
    }

    #[test]
    fn bar_supports_explicit_x_values() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style = parse_bar_style_args("bar", &[], defaults).unwrap();
        let charts = build_bar_series_from_tensor(
            Some(tensor_from(&[10.0, 20.0])),
            tensor_from(&[1.0, 2.0]),
            &style,
            BAR_CONFIG,
        )
        .unwrap();
        assert_eq!(charts.len(), 1);
        assert_eq!(charts[0].labels[0], "10");
        assert_eq!(charts[0].labels[1], "20");
    }

    #[test]
    fn bar_preserves_wide_integer_x_labels_and_validates_uniqueness_exactly() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style = parse_bar_style_args("bar", &[], defaults).unwrap();
        let x = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![2, 1],
        )
        .expect("wide X");
        let charts =
            build_bar_series_from_tensor(Some(x), tensor_from(&[1.0, 2.0]), &style, BAR_CONFIG)
                .unwrap();
        assert_eq!(
            charts[0].labels,
            vec!["9007199254740993", "18446744073709551615"]
        );

        let duplicate =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, u64::MAX]), vec![2, 1])
                .expect("duplicate X");
        assert!(build_bar_series_from_tensor(
            Some(duplicate),
            tensor_from(&[1.0, 2.0]),
            &style,
            BAR_CONFIG,
        )
        .is_err());
    }

    #[test]
    fn barh_treats_row_vectors_as_vector_inputs() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style =
            parse_bar_style_args("barh", &[Value::String("stacked".into())], defaults).unwrap();
        let charts = build_bar_series_from_tensor(
            Some(row_vector_tensor(&[1980.0, 1990.0, 2000.0])),
            row_vector_tensor(&[10.0, 20.0, 30.0]),
            &style,
            BARH_CONFIG,
        )
        .unwrap();
        assert_eq!(charts.len(), 1);
        assert_eq!(charts[0].bar_count(), 3);
        assert_eq!(charts[0].orientation, Orientation::Horizontal);
    }

    #[test]
    fn barh_treats_scalar_x_vector_y_as_single_group() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style = parse_bar_style_args("barh", &[], defaults).unwrap();
        let charts = build_bar_series_from_tensor(
            Some(tensor_from(&[1990.0])),
            row_vector_tensor(&[10.0, 20.0, 30.0]),
            &style,
            BARH_CONFIG,
        )
        .unwrap();
        assert_eq!(charts.len(), 3);
        assert_eq!(charts[0].bar_count(), 1);
        assert_eq!(charts[0].labels[0], "1990");
        assert!(charts
            .iter()
            .all(|chart| chart.orientation == Orientation::Horizontal));
    }

    #[test]
    fn barh_supports_x_matching_matrix_columns() {
        setup_plot_tests();
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style = parse_bar_style_args("barh", &[], defaults).unwrap();
        let charts = build_bar_series_from_tensor(
            Some(row_vector_tensor(&[1980.0, 1990.0, 2000.0])),
            matrix_tensor(&[2.0, 11.0, 6.0, 22.0, 9.0, 32.0], 2, 3),
            &style,
            BARH_CONFIG,
        )
        .unwrap();
        assert_eq!(charts.len(), 2);
        assert_eq!(charts[0].bar_count(), 3);
        assert_eq!(charts[0].values().unwrap(), &[2.0, 6.0, 9.0]);
        assert_eq!(charts[1].values().unwrap(), &[11.0, 22.0, 32.0]);
        assert_eq!(charts[0].labels, vec!["1980", "1990", "2000"]);
    }

    #[test]
    fn gpu_shape_matches_host_row_vector_semantics() {
        let implicit =
            BarMatrixShape::from_handle(&gpu_handle_with_shape(&[1, 3]), None, BARH_CONFIG)
                .unwrap();
        assert_eq!(implicit.rows, 3);
        assert_eq!(implicit.cols, 1);

        let scalar_x =
            BarMatrixShape::from_handle(&gpu_handle_with_shape(&[1, 3]), Some(1), BARH_CONFIG)
                .unwrap();
        assert_eq!(scalar_x.rows, 1);
        assert_eq!(scalar_x.cols, 3);

        let vector_x =
            BarMatrixShape::from_handle(&gpu_handle_with_shape(&[1, 3]), Some(3), BARH_CONFIG)
                .unwrap();
        assert_eq!(vector_x.rows, 3);
        assert_eq!(vector_x.cols, 1);

        let x_matches_columns =
            BarMatrixShape::from_handle(&gpu_handle_with_shape(&[2, 3]), Some(3), BARH_CONFIG)
                .unwrap();
        assert_eq!(x_matches_columns.rows, 3);
        assert_eq!(x_matches_columns.cols, 2);
        assert_eq!(x_matches_columns.source_rows, 2);
        assert!(x_matches_columns.transposed);
    }

    #[test]
    fn explicit_x_with_gpu_y_preserves_gpu_input() {
        let input = block_on(BarInput::from_value(
            Some(Value::Tensor(tensor_from(&[1980.0, 1990.0, 2000.0]))),
            Value::GpuTensor(gpu_handle_with_shape(&[1, 3])),
            BARH_CONFIG,
        ))
        .unwrap();
        assert!(input.gpu_handle().is_some());
        assert_eq!(input.x_tensor().map(numeric_tensor_len), Some(3));
    }

    #[test]
    fn bar_and_barh_gather_all_resident_integer_classes_before_floating_gpu_geometry() {
        test_support::with_test_provider(|provider| {
            for config in [BAR_CONFIG, BARH_CONFIG] {
                for storage in all_integer_bar_storages() {
                    let expected = storage.clone();
                    let tensor = Tensor::new_integer(storage, vec![2, 2]).expect("integer Y");
                    let handle =
                        gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
                    let input =
                        block_on(BarInput::from_value(None, Value::GpuTensor(handle), config))
                            .expect("bar input");
                    match input {
                        BarInput::Host { y, .. } => {
                            assert_eq!(y.integer_storage(), Some(&expected));
                        }
                        BarInput::Gpu { .. } => {
                            panic!("resident integer Y must not enter floating GPU geometry")
                        }
                    }
                }
            }
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn bar_and_barh_wgpu_gather_all_resident_integer_classes_before_geometry() {
        let _accel_guard = test_support::accel_test_lock();
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        for config in [BAR_CONFIG, BARH_CONFIG] {
            for storage in all_integer_bar_storages() {
                let expected = storage.clone();
                let tensor = Tensor::new_integer(storage, vec![2, 2]).expect("integer Y");
                let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
                let input = block_on(BarInput::from_value(None, Value::GpuTensor(handle), config))
                    .expect("bar input");
                match input {
                    BarInput::Host { y, .. } => assert_eq!(y.integer_storage(), Some(&expected)),
                    BarInput::Gpu { .. } => {
                        panic!("resident integer Y must not enter floating WGPU geometry")
                    }
                }
            }
        }
    }

    #[test]
    fn bar_x_y_shorthand_builds_chart_with_explicit_labels() {
        setup_plot_tests();
        let out = bar_builtin(vec![
            Value::Tensor(tensor_from(&[10.0, 20.0])),
            Value::Tensor(tensor_from(&[1.0, 2.0])),
        ]);
        if let Err(err) = out {
            assert!(crate::builtins::plotting::state::presentation_unavailable(
                &err
            ));
        }
        let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
        let style = parse_bar_style_args("bar", &[], defaults).unwrap();
        let charts = build_bar_series_from_tensor(
            Some(tensor_from(&[10.0, 20.0])),
            tensor_from(&[1.0, 2.0]),
            &style,
            BAR_CONFIG,
        )
        .unwrap();
        assert_eq!(charts[0].labels[0], "10");
        assert_eq!(charts[0].labels[1], "20");
    }

    #[test]
    fn bar_catalog_infers_host_double_handle_array() {
        use runmat_types::*;
        let numeric = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
        let input = ValueFact::proven(
            numeric.clone(),
            ShapeFact::from(vec![Some(3), Some(2)]),
            StorageFact::Dense,
        );
        let request = CallRequest {
            arguments: vec![input; 1],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let result =
            runmat_builtins::infer_catalog_call(&runmat_builtins::BAR_CATALOG_ENTRY, &request);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs.len(), 1);
        let output = &result.outputs[0];
        assert_eq!(output.kind, numeric);
        assert_eq!(output.shape, ShapeFact::Unknown);
        assert_eq!(output.storage, StorageFact::Dense);
        assert_eq!(output.residency, ResidencyFact::Host);
    }
}
