use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::QuiverPlot;
use std::cell::RefCell;
use std::rc::Rc;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

use super::op_common::line_inputs::NumericInput;
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_line_style_args, value_as_f64, LineStyleParseOptions};

const BUILTIN_NAME: &str = "quiver";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::quiver")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "quiver",
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
    notes: "quiver is a plotting sink; GPU inputs may remain on device when a shared WGPU context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::quiver")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "quiver",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "quiver performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "quiver",
    category = "plotting",
    summary = "Render MATLAB-compatible quiver plots.",
    keywords = "quiver,plotting,vector field,arrows",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::quiver"
)]
pub async fn quiver_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (target_axes, x, y, u, v, rest) = parse_quiver_args(args)?;
    let parsed = parse_quiver_style_args(&rest)?;
    let mut x_in = Some(NumericInput::from_value(x, BUILTIN_NAME)?);
    let mut y_in = Some(NumericInput::from_value(y, BUILTIN_NAME)?);
    let mut u_in = Some(NumericInput::from_value(u, BUILTIN_NAME)?);
    let mut v_in = Some(NumericInput::from_value(v, BUILTIN_NAME)?);
    let opts = PlotRenderOptions {
        title: "Quiver",
        x_label: "X",
        y_label: "Y",
        ..Default::default()
    };
    let plot_index_out = Rc::new(RefCell::new(None));
    let plot_index_slot = Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let axes = target_axes.unwrap_or(axes);
        if let (Some(x_gpu), Some(y_gpu), Some(u_gpu), Some(v_gpu)) = (
            x_in.as_ref().and_then(NumericInput::gpu_handle),
            y_in.as_ref().and_then(NumericInput::gpu_handle),
            u_in.as_ref().and_then(NumericInput::gpu_handle),
            v_in.as_ref().and_then(NumericInput::gpu_handle),
        ) {
            if let Ok(plot) = build_quiver_gpu_plot(
                x_gpu,
                y_gpu,
                u_gpu,
                v_gpu,
                &parsed,
                parsed.label.as_deref().unwrap_or("Data"),
            ) {
                let plot_index = figure.add_quiver_plot_on_axes(plot, axes);
                *plot_index_slot.borrow_mut() = Some((axes, plot_index));
                return Ok(());
            }
        }
        let x_tensor = x_in.take().expect("x consumed").into_tensor(BUILTIN_NAME)?;
        let y_tensor = y_in.take().expect("y consumed").into_tensor(BUILTIN_NAME)?;
        let u_tensor = u_in.take().expect("u consumed").into_tensor(BUILTIN_NAME)?;
        let v_tensor = v_in.take().expect("v consumed").into_tensor(BUILTIN_NAME)?;
        let (x_vals, y_vals, u_vals, v_vals) =
            materialize_quiver_components(x_tensor, y_tensor, u_tensor, v_tensor, BUILTIN_NAME)?;
        let label = parsed.label.clone().unwrap_or_else(|| "Data".into());
        let plot = QuiverPlot::new(x_vals, y_vals, u_vals, v_vals)
            .map_err(|e| plotting_error(BUILTIN_NAME, format!("quiver: {e}")))?
            .with_style(
                parsed.color,
                parsed.line_width,
                parsed.scale,
                parsed.head_size,
            )
            .with_label(label);
        let plot_index = figure.add_quiver_plot_on_axes(plot, axes);
        *plot_index_slot.borrow_mut() = Some((axes, plot_index));
        Ok(())
    });
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = crate::builtins::plotting::state::register_quiver_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(err);
    }
    Ok(handle)
}

fn build_quiver_gpu_plot(
    x: &runmat_accelerate_api::GpuTensorHandle,
    y: &runmat_accelerate_api::GpuTensorHandle,
    u: &runmat_accelerate_api::GpuTensorHandle,
    v: &runmat_accelerate_api::GpuTensorHandle,
    parsed: &ParsedQuiverStyle,
    label: &str,
) -> crate::BuiltinResult<QuiverPlot> {
    let context = super::gpu_helpers::ensure_shared_wgpu_context(BUILTIN_NAME)?;
    let x_ref = runmat_accelerate_api::export_wgpu_buffer(x)
        .ok_or_else(|| plotting_error(BUILTIN_NAME, "quiver: unable to export GPU X data"))?;
    let y_ref = runmat_accelerate_api::export_wgpu_buffer(y)
        .ok_or_else(|| plotting_error(BUILTIN_NAME, "quiver: unable to export GPU Y data"))?;
    let u_ref = runmat_accelerate_api::export_wgpu_buffer(u)
        .ok_or_else(|| plotting_error(BUILTIN_NAME, "quiver: unable to export GPU U data"))?;
    let v_ref = runmat_accelerate_api::export_wgpu_buffer(v)
        .ok_or_else(|| plotting_error(BUILTIN_NAME, "quiver: unable to export GPU V data"))?;
    if u_ref.len != v_ref.len || u_ref.precision != v_ref.precision {
        return Err(plotting_error(BUILTIN_NAME, "quiver: U and V GPU inputs must match"));
    }
    let scalar = runmat_plot::gpu::ScalarType::from_is_f64(
        u_ref.precision == runmat_accelerate_api::ProviderPrecision::F64,
    );
    let rows = u_ref.shape.first().copied().unwrap_or(u_ref.len).max(1);
    let cols = u_ref.shape.get(1).copied().unwrap_or(1).max(1);
    let count = u_ref.len;
    let xy_mode = if x_ref.len == count && y_ref.len == count {
        0u32
    } else if x_ref.len == cols && y_ref.len == rows {
        1u32
    } else {
        return Err(plotting_error(
            BUILTIN_NAME,
            "quiver: GPU X/Y inputs must match U/V as full coordinates or meshgrid vectors",
        ));
    };
    let (min_x, max_x) = super::gpu_helpers::axis_bounds(u, BUILTIN_NAME)
        .map(|_| ())
        .err()
        .map(|_| (0.0, 0.0))
        .unwrap_or_else(|| if xy_mode == 0 { super::gpu_helpers::axis_bounds(x, BUILTIN_NAME).unwrap_or((0.0, 0.0)) } else { super::gpu_helpers::axis_bounds(x, BUILTIN_NAME).unwrap_or((0.0, 0.0)) });
    let (min_y, max_y) = super::gpu_helpers::axis_bounds(y, BUILTIN_NAME).unwrap_or((0.0, 0.0));
    let (min_u, max_u) = super::gpu_helpers::axis_bounds(u, BUILTIN_NAME).unwrap_or((0.0, 0.0));
    let (min_v, max_v) = super::gpu_helpers::axis_bounds(v, BUILTIN_NAME).unwrap_or((0.0, 0.0));
    let bounds = runmat_plot::core::BoundingBox::new(
        glam::Vec3::new(
            min_x + min_u.min(0.0) * parsed.scale,
            min_y + min_v.min(0.0) * parsed.scale,
            0.0,
        ),
        glam::Vec3::new(
            max_x + max_u.max(0.0) * parsed.scale,
            max_y + max_v.max(0.0) * parsed.scale,
            0.0,
        ),
    );
    let gpu_vertices = runmat_plot::gpu::quiver::pack_vertices(
        &context.device,
        &context.queue,
        &runmat_plot::gpu::quiver::QuiverGpuInputs {
            x_data: runmat_plot::gpu::axis::AxisData::Buffer(x_ref.buffer.clone()),
            y_data: runmat_plot::gpu::axis::AxisData::Buffer(y_ref.buffer.clone()),
            u_buffer: u_ref.buffer.clone(),
            v_buffer: v_ref.buffer.clone(),
            count: count as u32,
            rows: rows as u32,
            cols: cols as u32,
            xy_mode,
            scalar,
        },
        &runmat_plot::gpu::quiver::QuiverGpuParams {
            color: parsed.color,
            scale: parsed.scale,
            head_size: parsed.head_size,
        },
    )
    .map_err(|e| plotting_error(BUILTIN_NAME, format!("quiver: failed to build GPU vertices: {e}")))?;
    let mut plot = QuiverPlot::from_gpu_buffer(
        parsed.color,
        parsed.line_width,
        parsed.scale,
        parsed.head_size,
        gpu_vertices,
        count * 6,
        bounds,
    )
    .with_label(label);
    plot.x = Vec::new();
    plot.y = Vec::new();
    plot.u = Vec::new();
    plot.v = Vec::new();
    Ok(plot)
}

struct ParsedQuiverStyle {
    color: glam::Vec4,
    line_width: f32,
    label: Option<String>,
    scale: f32,
    head_size: f32,
}

fn parse_quiver_style_args(args: &[Value]) -> crate::BuiltinResult<ParsedQuiverStyle> {
    let mut filtered = Vec::new();
    let mut scale = 1.0f32;
    let mut head_size = 0.1f32;
    let mut idx = 0usize;
    while idx < args.len() {
        if let Some(key) = super::style::value_as_string(&args[idx]) {
            let key = key.trim().to_ascii_lowercase();
            if idx + 1 < args.len() {
                match key.as_str() {
                    "autoscalefactor" | "scale" => {
                        scale = value_as_f64(&args[idx + 1]).ok_or_else(|| {
                            plotting_error(BUILTIN_NAME, "quiver: AutoScaleFactor must be numeric")
                        })? as f32;
                        idx += 2;
                        continue;
                    }
                    "maxheadsize" | "headsize" => {
                        head_size = value_as_f64(&args[idx + 1]).ok_or_else(|| {
                            plotting_error(BUILTIN_NAME, "quiver: MaxHeadSize must be numeric")
                        })? as f32;
                        idx += 2;
                        continue;
                    }
                    _ => {}
                }
            }
        }
        filtered.push(args[idx].clone());
        idx += 1;
    }
    let parsed = parse_line_style_args(&filtered, &LineStyleParseOptions::generic(BUILTIN_NAME))?;
    Ok(ParsedQuiverStyle {
        color: parsed.appearance.color,
        line_width: parsed.appearance.line_width,
        label: parsed.label,
        scale,
        head_size,
    })
}

fn parse_quiver_args(
    args: Vec<Value>,
) -> crate::BuiltinResult<(Option<usize>, Value, Value, Value, Value, Vec<Value>)> {
    if args.len() < 2 {
        return Err(plotting_error(
            BUILTIN_NAME,
            "quiver: expected U,V or X,Y,U,V inputs",
        ));
    }
    let mut it = args.into_iter();
    let mut target_axes = None;
    let first = it.next().unwrap();
    let first = if let Ok(handle) =
        crate::builtins::plotting::properties::resolve_plot_handle(&first, BUILTIN_NAME)
    {
        if let crate::builtins::plotting::properties::PlotHandle::Axes(_, axes) = handle {
            target_axes = Some(axes);
            it.next().ok_or_else(|| {
                plotting_error(BUILTIN_NAME, "quiver: expected data after axes handle")
            })?
        } else {
            first
        }
    } else {
        first
    };
    let second = it.next().unwrap();
    let third = it.next();
    let fourth = it.next();
    match (third, fourth) {
        (None, _) => {
            let (x, y) = default_quiver_grid_from_values(&first, &second, BUILTIN_NAME)?;
            Ok((target_axes, Value::Tensor(x), Value::Tensor(y), first, second, Vec::new()))
        }
        (Some(third), Some(fourth)) => Ok((target_axes, first, second, third, fourth, it.collect())),
        _ => Err(plotting_error(
            BUILTIN_NAME,
            "quiver: expected U,V or X,Y,U,V inputs",
        )),
    }
}

fn default_quiver_grid_from_values(
    u: &Value,
    v: &Value,
    builtin: &'static str,
) -> crate::BuiltinResult<(Tensor, Tensor)> {
    let (u_rows, u_cols, u_len) = tensor_shape_from_value(u, builtin)?;
    let (v_rows, v_cols, v_len) = tensor_shape_from_value(v, builtin)?;
    if u_rows != v_rows || u_cols != v_cols || u_len != v_len {
        return Err(plotting_error(
            builtin,
            "quiver: U and V inputs must have identical size",
        ));
    }
    let rows = u_rows.max(1);
    let cols = u_cols.max(1);
    let mut x = Vec::with_capacity(rows * cols);
    let mut y = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            x.push((col + 1) as f64);
            y.push((row + 1) as f64);
        }
    }
    Ok((
        Tensor {
            data: x,
            shape: vec![rows * cols],
            rows: rows * cols,
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        },
        Tensor {
            data: y,
            shape: vec![rows * cols],
            rows: rows * cols,
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        },
    ))
}

fn tensor_shape_from_value(
    value: &Value,
    builtin: &'static str,
) -> crate::BuiltinResult<(usize, usize, usize)> {
    match value {
        Value::GpuTensor(handle) => {
            let rows = handle.shape.first().copied().unwrap_or(1).max(1);
            let cols = handle.shape.get(1).copied().unwrap_or(1).max(1);
            let len = handle.shape.iter().product::<usize>().max(1);
            Ok((rows, cols, len))
        }
        _ => {
            let tensor = Tensor::try_from(value)
                .map_err(|e| plotting_error(builtin, format!("quiver: {e}")))?;
            Ok((tensor.rows.max(1), tensor.cols.max(1), tensor.data.len()))
        }
    }
}

fn materialize_quiver_components(
    x: Tensor,
    y: Tensor,
    u: Tensor,
    v: Tensor,
    builtin: &'static str,
) -> crate::BuiltinResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    if u.rows != v.rows || u.cols != v.cols || u.data.len() != v.data.len() {
        return Err(plotting_error(
            builtin,
            "quiver: U and V inputs must have identical size",
        ));
    }

    let u_is_matrix = u.rows > 1 && u.cols > 1;
    let v_is_matrix = v.rows > 1 && v.cols > 1;
    if u_is_matrix != v_is_matrix {
        return Err(plotting_error(
            builtin,
            "quiver: U and V inputs must both be vectors or both be matrices",
        ));
    }

    if !u_is_matrix {
        let len = u.data.len();
        if x.data.len() != len || y.data.len() != len {
            return Err(plotting_error(
                builtin,
                "quiver: X, Y, U, and V vectors must have the same length",
            ));
        }
        return Ok((x.data, y.data, u.data, v.data));
    }

    let rows = u.rows;
    let cols = u.cols;
    if x.data.len() == rows * cols && y.data.len() == rows * cols {
        return Ok((x.data, y.data, u.data, v.data));
    }
    if x.data.len() == cols && y.data.len() == rows {
        let mut out_x = Vec::with_capacity(rows * cols);
        let mut out_y = Vec::with_capacity(rows * cols);
        for col in 0..cols {
            for row in 0..rows {
                out_x.push(x.data[col]);
                out_y.push(y.data[row]);
            }
        }
        return Ok((out_x, out_y, u.data, v.data));
    }
    Err(plotting_error(
        builtin,
        "quiver: X and Y must match U/V as vectors or meshgrid-style matrices",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
        subplot::subplot_builtin,
    };
    use runmat_plot::plots::PlotElement;

    fn vec_tensor(data: &[f64]) -> Tensor {
        Tensor {
            data: data.to_vec(),
            shape: vec![data.len()],
            rows: data.len(),
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        }
    }

    #[test]
    fn quiver_builds_plot_and_defaults_grid() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _ = futures::executor::block_on(quiver_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, -1.0])),
            Value::Tensor(vec_tensor(&[0.5, 0.5])),
        ]))
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Quiver(quiver) = fig.plots().next().unwrap() else {
            panic!("expected quiver");
        };
        assert_eq!(quiver.x, vec![1.0, 1.0]);
        assert_eq!(quiver.y, vec![1.0, 2.0]);
    }

    #[test]
    fn quiver_supports_axes_target_and_properties() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let ax = subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(2.0)).unwrap();
        let handle = futures::executor::block_on(quiver_builtin(vec![
            Value::Num(ax),
            Value::Tensor(vec_tensor(&[0.0, 1.0])),
            Value::Tensor(vec_tensor(&[0.0, 1.0])),
            Value::Tensor(vec_tensor(&[1.0, 0.0])),
            Value::Tensor(vec_tensor(&[0.0, 1.0])),
            Value::String("AutoScaleFactor".into()),
            Value::Num(2.5),
        ]))
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.plot_axes_indices()[0], 1);
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("AutoScaleFactor".into())]).unwrap(),
            Value::Num(2.5)
        );
        set_builtin(vec![
            Value::Num(handle),
            Value::String("MaxHeadSize".into()),
            Value::Num(0.3),
        ])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Quiver(quiver) = fig.plots().next().unwrap() else {
            panic!("expected quiver");
        };
        assert_eq!(quiver.head_size, 0.3);
    }
}
