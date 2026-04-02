use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::gpu::errorbar::{ErrorBarGpuInputs, ErrorBarGpuParams};
use runmat_plot::gpu::line::{
    self, LineGpuInputs as MarkerGpuInputs, LineGpuParams as MarkerGpuParams,
};
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::{ErrorBar, LineMarkerAppearance};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

use super::common::numeric_pair;
use super::gpu_helpers::gpu_errorbar_bounds;
use super::op_common::line_inputs::NumericInput;
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{marker_metadata_from_appearance, parse_line_style_args, LineStyleParseOptions};

const BUILTIN_NAME: &str = "errorbar";
type ErrorBarArgs = (
    Option<usize>,
    Value,
    Value,
    Option<Value>,
    Option<Value>,
    Value,
    Value,
    Vec<Value>,
);

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::errorbar")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "errorbar",
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
    notes: "errorbar is a plotting sink; GPU inputs may remain on device when shared plotting context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::errorbar")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "errorbar",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "errorbar performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "errorbar",
    category = "plotting",
    summary = "Render MATLAB-compatible error bars.",
    keywords = "errorbar,plotting,uncertainty",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::errorbar"
)]
pub fn errorbar_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (target_axes, x, y, x_neg, x_pos, y_neg, y_pos, rest) = parse_errorbar_args(args)?;
    let parsed = parse_errorbar_style_args(&rest)?;
    let mut x_in = Some(NumericInput::from_value(x, BUILTIN_NAME)?);
    let mut y_in = Some(NumericInput::from_value(y, BUILTIN_NAME)?);
    let mut xn_in = x_neg
        .map(|v| NumericInput::from_value(v, BUILTIN_NAME))
        .transpose()?;
    let mut xp_in = x_pos
        .map(|v| NumericInput::from_value(v, BUILTIN_NAME))
        .transpose()?;
    let mut n_in = Some(NumericInput::from_value(y_neg, BUILTIN_NAME)?);
    let mut p_in = Some(NumericInput::from_value(y_pos, BUILTIN_NAME)?);
    let opts = PlotRenderOptions {
        title: "Error Bars",
        x_label: "X",
        y_label: "Y",
        ..Default::default()
    };
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let axes = target_axes.unwrap_or(axes);
        let x_arg = x_in.take().expect("x consumed");
        let y_arg = y_in.take().expect("y consumed");
        let yn_arg = n_in.take().expect("yn consumed");
        let yp_arg = p_in.take().expect("yp consumed");
        let label = parsed.label.clone().unwrap_or_else(|| "Data".into());
        if let (Some(x_gpu), Some(y_gpu), Some(yn_gpu), Some(yp_gpu)) = (
            x_arg.gpu_handle(),
            y_arg.gpu_handle(),
            yn_arg.gpu_handle(),
            yp_arg.gpu_handle(),
        ) {
            match build_errorbar_gpu_plot(
                BUILTIN_NAME,
                x_gpu,
                y_gpu,
                xn_in.as_ref().and_then(|v| v.gpu_handle()),
                xp_in.as_ref().and_then(|v| v.gpu_handle()),
                yn_gpu,
                yp_gpu,
                &parsed,
                &label,
            ) {
                Ok(plot) => {
                    let plot_index = figure.add_errorbar_on_axes(plot, axes);
                    *plot_index_slot.borrow_mut() = Some((axes, plot_index));
                    return Ok(());
                }
                Err(err) => log::warn!("errorbar GPU path unavailable: {err}"),
            }
        }
        let x = x_arg.into_tensor(BUILTIN_NAME)?;
        let y = y_arg.into_tensor(BUILTIN_NAME)?;
        let xn = xn_in
            .take()
            .map(|v| v.into_tensor(BUILTIN_NAME))
            .transpose()?;
        let xp = xp_in
            .take()
            .map(|v| v.into_tensor(BUILTIN_NAME))
            .transpose()?;
        let yn = yn_arg.into_tensor(BUILTIN_NAME)?;
        let yp = yp_arg.into_tensor(BUILTIN_NAME)?;
        let (x, y) = numeric_pair(x, y, BUILTIN_NAME)?;
        let (yn, yp) = numeric_pair(yn, yp, BUILTIN_NAME)?;
        let mut plot = if let (Some(xn), Some(xp)) = (xn, xp) {
            let (xn, xp) = numeric_pair(xn, xp, BUILTIN_NAME)?;
            ErrorBar::new_both(x, y, xn, xp, yn, yp)
        } else {
            ErrorBar::new_vertical(x, y, yn, yp)
        }
        .map_err(|e| plotting_error(BUILTIN_NAME, format!("errorbar: {e}")))?
        .with_style(
            parsed.color,
            parsed.line_width,
            parsed.line_style,
            parsed.cap_size,
        )
        .with_label(label);
        if let Some(marker) = parsed.marker.clone() {
            plot.set_marker(Some(marker));
        }
        let plot_index = figure.add_errorbar_on_axes(plot, axes);
        *plot_index_slot.borrow_mut() = Some((axes, plot_index));
        Ok(())
    });
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle =
        crate::builtins::plotting::state::register_errorbar_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(err);
    }
    Ok(handle)
}

fn build_errorbar_gpu_plot(
    name: &'static str,
    x: &runmat_accelerate_api::GpuTensorHandle,
    y: &runmat_accelerate_api::GpuTensorHandle,
    x_neg: Option<&runmat_accelerate_api::GpuTensorHandle>,
    x_pos: Option<&runmat_accelerate_api::GpuTensorHandle>,
    y_neg: &runmat_accelerate_api::GpuTensorHandle,
    y_pos: &runmat_accelerate_api::GpuTensorHandle,
    parsed: &ParsedErrorBarStyle,
    label: &str,
) -> crate::BuiltinResult<ErrorBar> {
    let context = super::gpu_helpers::ensure_shared_wgpu_context(name)?;
    let x_ref = runmat_accelerate_api::export_wgpu_buffer(x)
        .ok_or_else(|| plotting_error(name, format!("{name}: unable to export GPU X data")))?;
    let y_ref = runmat_accelerate_api::export_wgpu_buffer(y)
        .ok_or_else(|| plotting_error(name, format!("{name}: unable to export GPU Y data")))?;
    let xn_ref = x_neg.and_then(runmat_accelerate_api::export_wgpu_buffer);
    let xp_ref = x_pos.and_then(runmat_accelerate_api::export_wgpu_buffer);
    let yn_ref = runmat_accelerate_api::export_wgpu_buffer(y_neg).ok_or_else(|| {
        plotting_error(
            name,
            format!("{name}: unable to export GPU negative error data"),
        )
    })?;
    let yp_ref = runmat_accelerate_api::export_wgpu_buffer(y_pos).ok_or_else(|| {
        plotting_error(
            name,
            format!("{name}: unable to export GPU positive error data"),
        )
    })?;
    if x_ref.len != y_ref.len
        || x_ref.len != yn_ref.len
        || x_ref.len != yp_ref.len
        || xn_ref.as_ref().map(|r| r.len).unwrap_or(x_ref.len) != x_ref.len
        || xp_ref.as_ref().map(|r| r.len).unwrap_or(x_ref.len) != x_ref.len
    {
        return Err(plotting_error(
            name,
            format!("{name}: X, Y, and error inputs must have identical lengths"),
        ));
    }
    if x_ref.precision != y_ref.precision
        || x_ref.precision != yn_ref.precision
        || x_ref.precision != yp_ref.precision
        || xn_ref
            .as_ref()
            .map(|r| r.precision)
            .unwrap_or(x_ref.precision)
            != x_ref.precision
        || xp_ref
            .as_ref()
            .map(|r| r.precision)
            .unwrap_or(x_ref.precision)
            != x_ref.precision
    {
        return Err(plotting_error(
            name,
            format!("{name}: gpuArray precision must match across all errorbar inputs"),
        ));
    }
    let scalar =
        ScalarType::from_is_f64(x_ref.precision == runmat_accelerate_api::ProviderPrecision::F64);
    let bounds = if let (Some(xn), Some(xp)) = (x_neg, x_pos) {
        let mut b = gpu_errorbar_bounds(x, y, y_neg, y_pos, name)?;
        let (min_xn, max_xn) = super::gpu_helpers::axis_bounds(xn, name)?;
        let (min_xp, max_xp) = super::gpu_helpers::axis_bounds(xp, name)?;
        b.min.x -= max_xn.max(min_xn.abs());
        b.max.x += max_xp.max(min_xp.abs());
        b
    } else {
        gpu_errorbar_bounds(x, y, y_neg, y_pos, name)?
    };
    let gpu_vertices = runmat_plot::gpu::errorbar::pack_vertical_vertices(
        &context.device,
        &context.queue,
        &ErrorBarGpuInputs {
            x_buffer: x_ref.buffer.clone(),
            y_buffer: y_ref.buffer.clone(),
            x_neg_buffer: xn_ref.as_ref().map(|r| r.buffer.clone()),
            x_pos_buffer: xp_ref.as_ref().map(|r| r.buffer.clone()),
            y_neg_buffer: yn_ref.buffer.clone(),
            y_pos_buffer: yp_ref.buffer.clone(),
            len: x_ref.len as u32,
            scalar,
        },
        &ErrorBarGpuParams {
            color: parsed.color,
            cap_size_data: parsed.cap_size * 0.01,
            line_style: parsed.line_style,
            orientation: if x_neg.is_some() && x_pos.is_some() {
                2
            } else {
                0
            },
        },
    )
    .map_err(|e| plotting_error(name, format!("{name}: failed to build GPU vertices: {e}")))?;
    let mut plot = ErrorBar::from_gpu_buffer(
        parsed.color,
        parsed.line_width,
        parsed.line_style,
        parsed.cap_size,
        if x_neg.is_some() && x_pos.is_some() {
            runmat_plot::plots::errorbar::ErrorBarOrientation::Both
        } else {
            runmat_plot::plots::errorbar::ErrorBarOrientation::Vertical
        },
        gpu_vertices,
        x_ref.len as usize * 6,
        bounds,
    )
    .with_label(label);
    if let Some(marker) = parsed.marker.clone() {
        let marker_gpu = line::pack_marker_vertices_from_xy(
            &context.device,
            &context.queue,
            &MarkerGpuInputs {
                x_buffer: x_ref.buffer.clone(),
                y_buffer: y_ref.buffer.clone(),
                len: x_ref.len as u32,
                scalar,
            },
            &MarkerGpuParams {
                color: marker.face_color,
                half_width_data: 0.0,
                thick: false,
                line_style: runmat_plot::plots::LineStyle::Solid,
                marker_size: marker.size,
            },
        )
        .map_err(|e| {
            plotting_error(
                name,
                format!("{name}: failed to build marker vertices: {e}"),
            )
        })?;
        plot.set_marker(Some(marker));
        plot.set_marker_gpu_vertices(Some(marker_gpu));
    }
    Ok(plot)
}

struct ParsedErrorBarStyle {
    color: glam::Vec4,
    line_width: f32,
    line_style: runmat_plot::plots::LineStyle,
    marker: Option<LineMarkerAppearance>,
    label: Option<String>,
    cap_size: f32,
}

fn parse_errorbar_style_args(args: &[Value]) -> crate::BuiltinResult<ParsedErrorBarStyle> {
    let mut filtered = Vec::new();
    let mut cap_size = 6.0;
    let mut idx = 0usize;
    while idx < args.len() {
        if let Some(key) = super::style::value_as_string(&args[idx]) {
            if key.trim().eq_ignore_ascii_case("CapSize") && idx + 1 < args.len() {
                cap_size = super::style::value_as_f64(&args[idx + 1]).ok_or_else(|| {
                    plotting_error(BUILTIN_NAME, "errorbar: CapSize must be numeric")
                })? as f32;
                idx += 2;
                continue;
            }
        }
        filtered.push(args[idx].clone());
        idx += 1;
    }
    let parsed = parse_line_style_args(&filtered, &LineStyleParseOptions::generic(BUILTIN_NAME))?;
    let marker = marker_metadata_from_appearance(&parsed.appearance);
    Ok(ParsedErrorBarStyle {
        color: parsed.appearance.color,
        line_width: parsed.appearance.line_width,
        line_style: parsed.appearance.line_style,
        marker,
        label: parsed.label,
        cap_size,
    })
}

fn parse_errorbar_args(args: Vec<Value>) -> crate::BuiltinResult<ErrorBarArgs> {
    if args.len() < 2 {
        return Err(plotting_error(
            BUILTIN_NAME,
            "errorbar: expected at least y and error inputs",
        ));
    }
    let mut it = args.into_iter();
    let mut target_axes = None;
    let first = it.next().unwrap();
    let first = if let Ok(crate::builtins::plotting::properties::PlotHandle::Axes(_, axes)) =
        crate::builtins::plotting::properties::resolve_plot_handle(&first, BUILTIN_NAME)
    {
        target_axes = Some(axes);
        it.next().ok_or_else(|| {
            plotting_error(BUILTIN_NAME, "errorbar: expected data after axes handle")
        })?
    } else {
        first
    };
    let second = it.next().unwrap();
    let third = it.next();
    let fourth = it.next();
    match (third, fourth) {
        (None, _) => {
            let y = first;
            let err = second;
            let len = Tensor::try_from(&y)
                .map_err(|e| plotting_error(BUILTIN_NAME, format!("errorbar: {e}")))?
                .data
                .len();
            let x = Value::Tensor(Tensor {
                data: (1..=len).map(|i| i as f64).collect(),
                shape: vec![len],
                rows: len,
                cols: 1,
                dtype: runmat_builtins::NumericDType::F64,
            });
            Ok((target_axes, x, y, None, None, err.clone(), err, Vec::new()))
        }
        (Some(third), None) => {
            if is_styleish(&third) {
                let y = first;
                let err = second;
                let len = Tensor::try_from(&y)
                    .map_err(|e| plotting_error(BUILTIN_NAME, format!("errorbar: {e}")))?
                    .data
                    .len();
                let x = Value::Tensor(Tensor {
                    data: (1..=len).map(|i| i as f64).collect(),
                    shape: vec![len],
                    rows: len,
                    cols: 1,
                    dtype: runmat_builtins::NumericDType::F64,
                });
                Ok((target_axes, x, y, None, None, err.clone(), err, vec![third]))
            } else {
                Ok((
                    target_axes,
                    first,
                    second,
                    None,
                    None,
                    third.clone(),
                    third,
                    Vec::new(),
                ))
            }
        }
        (Some(third), Some(fourth)) => {
            if is_styleish(&fourth) {
                let mut rest = vec![fourth];
                rest.extend(it);
                Ok((
                    target_axes,
                    first,
                    second,
                    None,
                    None,
                    third.clone(),
                    third,
                    rest,
                ))
            } else {
                let fifth = it.next();
                let sixth = it.next();
                match (fifth, sixth) {
                    (Some(fifth), Some(sixth)) => Ok((
                        target_axes,
                        first,
                        second,
                        Some(fifth),
                        Some(sixth),
                        third,
                        fourth,
                        it.collect(),
                    )),
                    _ => Ok((
                        target_axes,
                        first,
                        second,
                        None,
                        None,
                        third,
                        fourth,
                        it.collect(),
                    )),
                }
            }
        }
    }
}

fn is_styleish(value: &Value) -> bool {
    matches!(value, Value::String(_) | Value::CharArray(_))
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
    fn errorbar_builds_vertical_plot() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _ = errorbar_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[0.1, 0.2])),
        ]);
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert!(matches!(
            fig.plots().next().unwrap(),
            PlotElement::ErrorBar(_)
        ));
    }

    #[test]
    fn errorbar_supports_axes_target_and_capsize() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let ax = subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(2.0)).unwrap();
        let _ = errorbar_builtin(vec![
            Value::Num(ax),
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[3.0, 4.0])),
            Value::Tensor(vec_tensor(&[0.2, 0.3])),
            Value::String("CapSize".into()),
            Value::Num(10.0),
        ]);
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.plot_axes_indices()[0], 1);
    }

    #[test]
    fn errorbar_supports_both_direction_form() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _ = errorbar_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[3.0, 4.0])),
            Value::Tensor(vec_tensor(&[0.2, 0.3])),
            Value::Tensor(vec_tensor(&[0.2, 0.3])),
            Value::Tensor(vec_tensor(&[0.1, 0.2])),
            Value::Tensor(vec_tensor(&[0.1, 0.2])),
        ]);
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::ErrorBar(error) = fig.plots().next().unwrap() else {
            panic!("expected errorbar");
        };
        assert_eq!(
            error.orientation,
            runmat_plot::plots::errorbar::ErrorBarOrientation::Both
        );
        assert_eq!(error.x_neg, vec![0.1, 0.2]);
        assert_eq!(error.y_pos, vec![0.2, 0.3]);
    }

    #[test]
    fn errorbar_handle_exposes_runtime_properties() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle = errorbar_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[0.1, 0.2])),
        ])
        .unwrap();
        let props = get_builtin(vec![Value::Num(handle)]).unwrap();
        let Value::Struct(st) = props else {
            panic!("expected struct");
        };
        assert_eq!(
            st.fields.get("Type"),
            Some(&Value::String("errorbar".into()))
        );
        assert!(st.fields.contains_key("CapSize"));
    }

    #[test]
    fn errorbar_handle_set_updates_properties() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle = errorbar_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[0.1, 0.2])),
        ])
        .unwrap();
        set_builtin(vec![
            Value::Num(handle),
            Value::String("CapSize".into()),
            Value::Num(12.0),
        ])
        .unwrap();
        let cap = get_builtin(vec![Value::Num(handle), Value::String("CapSize".into())]).unwrap();
        assert_eq!(cap, Value::Num(12.0));
    }

    #[test]
    fn errorbar_accepts_scalar_point() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _ = errorbar_builtin(vec![Value::Num(1.0), Value::Num(2.0), Value::Num(0.3)]).unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::ErrorBar(plot) = fig.plots().next().unwrap() else {
            panic!("expected errorbar")
        };
        assert_eq!(plot.x, vec![1.0]);
        assert_eq!(plot.y, vec![2.0]);
    }
}
