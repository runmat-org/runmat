use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::gpu::line::{
    self, LineGpuInputs as MarkerGpuInputs, LineGpuParams as MarkerGpuParams,
};
use runmat_plot::gpu::stem::{StemGpuInputs, StemGpuParams};
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::{LineMarkerAppearance, StemPlot};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::type_resolvers::string_type;

use super::common::numeric_pair;
use super::gpu_helpers::gpu_xy_bounds;
use super::op_common::line_inputs::NumericInput;
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{
    marker_metadata_from_appearance, parse_line_style_args, LineAppearance, LineStyleParseOptions,
};

const BUILTIN_NAME: &str = "stem";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::stem")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "stem",
    op_kind: GpuOpKind::Custom("plot-render"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "stem is a plotting sink; GPU inputs may remain on device when shared plotting context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::stem")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "stem",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "stem performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "stem",
    category = "plotting",
    summary = "Render MATLAB-compatible stem plots.",
    keywords = "stem,plotting,discrete",
    sink = true,
    suppress_auto_output = true,
    type_resolver(string_type),
    builtin_path = "crate::builtins::plotting::stem"
)]
pub fn stem_builtin(args: Vec<Value>) -> crate::BuiltinResult<String> {
    let (target_axes, x, y, rest) = parse_stem_args(args)?;
    let parsed = parse_stem_style_args(&rest)?;
    let mut x_input = Some(NumericInput::from_value(x, BUILTIN_NAME)?);
    let mut y_input = Some(NumericInput::from_value(y, BUILTIN_NAME)?);
    let opts = PlotRenderOptions {
        title: "Stem",
        x_label: "X",
        y_label: "Y",
        ..Default::default()
    };
    render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let label = parsed.label.clone().unwrap_or_else(|| "Data".into());
        let x_arg = x_input.take().expect("stem x consumed once");
        let y_arg = y_input.take().expect("stem y consumed once");
        if let (Some(x_gpu), Some(y_gpu)) = (x_arg.gpu_handle(), y_arg.gpu_handle()) {
            match build_stem_gpu_plot(BUILTIN_NAME, x_gpu, y_gpu, &parsed, &label) {
                Ok(plot) => {
                    figure.add_stem_plot_on_axes(plot, target_axes.unwrap_or(axes));
                    return Ok(());
                }
                Err(err) => log::warn!("stem GPU path unavailable: {err}"),
            }
        }
        let x = x_arg.into_tensor(BUILTIN_NAME)?;
        let y = y_arg.into_tensor(BUILTIN_NAME)?;
        let (x, y) = numeric_pair(x, y, BUILTIN_NAME)?;
        let plot = build_stem_plot(x, y, &parsed, &label)?;
        figure.add_stem_plot_on_axes(plot, target_axes.unwrap_or(axes));
        Ok(())
    })
}

fn build_stem_plot(
    x: Vec<f64>,
    y: Vec<f64>,
    parsed: &ParsedStemStyle,
    label: &str,
) -> crate::BuiltinResult<StemPlot> {
    let mut plot = StemPlot::new(x, y)
        .map_err(|e| plotting_error(BUILTIN_NAME, format!("stem: {e}")))?
        .with_style(
            parsed.appearance.color,
            parsed.appearance.line_width,
            parsed.appearance.line_style,
            parsed.baseline,
        )
        .with_baseline_style(parsed.appearance.color, parsed.baseline_visible)
        .with_label(label);
    if let Some(marker) = parsed.marker.clone() {
        let mut marker = marker;
        if parsed.filled {
            marker.filled = true;
        }
        plot.set_marker(Some(marker));
    } else if parsed.filled {
        let mut marker = plot.marker.clone().unwrap_or(LineMarkerAppearance {
            kind: runmat_plot::plots::scatter::MarkerStyle::Circle,
            size: 6.0,
            edge_color: parsed.appearance.color,
            face_color: parsed.appearance.color,
            filled: true,
        });
        marker.filled = true;
        plot.set_marker(Some(marker));
    }
    Ok(plot)
}

fn build_stem_gpu_plot(
    name: &'static str,
    x: &runmat_accelerate_api::GpuTensorHandle,
    y: &runmat_accelerate_api::GpuTensorHandle,
    parsed: &ParsedStemStyle,
    label: &str,
) -> crate::BuiltinResult<StemPlot> {
    let context = super::gpu_helpers::ensure_shared_wgpu_context(name)?;
    let x_ref = runmat_accelerate_api::export_wgpu_buffer(x)
        .ok_or_else(|| plotting_error(name, format!("{name}: unable to export GPU X data")))?;
    let y_ref = runmat_accelerate_api::export_wgpu_buffer(y)
        .ok_or_else(|| plotting_error(name, format!("{name}: unable to export GPU Y data")))?;
    if x_ref.len != y_ref.len {
        return Err(plotting_error(
            name,
            format!("{name}: X and Y inputs must have identical lengths"),
        ));
    }
    if x_ref.precision != y_ref.precision {
        return Err(plotting_error(
            name,
            format!("{name}: X and Y gpuArrays must share the same precision"),
        ));
    }
    let scalar =
        ScalarType::from_is_f64(x_ref.precision == runmat_accelerate_api::ProviderPrecision::F64);
    let xy_bounds = gpu_xy_bounds(x, y, name)?;
    let min_x = xy_bounds.min.x;
    let max_x = xy_bounds.max.x;
    let bounds = runmat_plot::core::BoundingBox::new(
        glam::Vec3::new(
            xy_bounds.min.x,
            xy_bounds.min.y.min(parsed.baseline as f32),
            0.0,
        ),
        glam::Vec3::new(
            xy_bounds.max.x,
            xy_bounds.max.y.max(parsed.baseline as f32),
            0.0,
        ),
    );
    let gpu_vertices = runmat_plot::gpu::stem::pack_vertices_from_xy(
        &context.device,
        &context.queue,
        &StemGpuInputs {
            x_buffer: x_ref.buffer.clone(),
            y_buffer: y_ref.buffer.clone(),
            len: x_ref.len as u32,
            scalar,
        },
        &StemGpuParams {
            color: parsed.appearance.color,
            baseline_color: parsed.appearance.color,
            baseline: parsed.baseline as f32,
            baseline_visible: parsed.baseline_visible,
            min_x,
            max_x,
            line_style: parsed.appearance.line_style,
        },
    )
    .map_err(|e| plotting_error(name, format!("{name}: failed to build GPU vertices: {e}")))?;
    let mut plot = StemPlot::from_gpu_buffer(
        parsed.appearance.color,
        parsed.appearance.line_width,
        parsed.appearance.line_style,
        parsed.baseline,
        parsed.appearance.color,
        parsed.baseline_visible,
        gpu_vertices,
        (if parsed.baseline_visible { 2 } else { 0 }) + (x_ref.len as usize * 2),
        bounds,
    )
    .with_label(label);
    if let Some(marker) = parsed.marker.clone() {
        let mut marker = marker;
        if parsed.filled {
            marker.filled = true;
        }
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
    } else if parsed.filled {
        let marker = LineMarkerAppearance {
            kind: runmat_plot::plots::scatter::MarkerStyle::Circle,
            size: 6.0,
            edge_color: parsed.appearance.color,
            face_color: parsed.appearance.color,
            filled: true,
        };
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

struct ParsedStemStyle {
    appearance: LineAppearance,
    marker: Option<LineMarkerAppearance>,
    label: Option<String>,
    baseline: f64,
    baseline_visible: bool,
    filled: bool,
}

fn parse_stem_style_args(args: &[Value]) -> crate::BuiltinResult<ParsedStemStyle> {
    let mut filtered = Vec::new();
    let mut filled = false;
    let mut baseline = 0.0;
    let mut baseline_visible = true;
    let mut idx = 0usize;
    while idx < args.len() {
        if let Some(text) = super::style::value_as_string(&args[idx]) {
            let trimmed = text.trim();
            if trimmed.eq_ignore_ascii_case("filled") {
                filled = true;
                idx += 1;
                continue;
            }
            if (trimmed.eq_ignore_ascii_case("basevalue")
                || trimmed.eq_ignore_ascii_case("baseline"))
                && idx + 1 < args.len()
            {
                if trimmed.eq_ignore_ascii_case("basevalue") {
                    baseline = super::style::value_as_f64(&args[idx + 1]).ok_or_else(|| {
                        plotting_error(BUILTIN_NAME, "stem: BaseValue must be numeric")
                    })?;
                } else {
                    baseline_visible =
                        super::style::value_as_bool(&args[idx + 1]).ok_or_else(|| {
                            plotting_error(BUILTIN_NAME, "stem: BaseLine must be logical")
                        })?;
                }
                idx += 2;
                continue;
            }
        }
        filtered.push(args[idx].clone());
        idx += 1;
    }
    let parsed = parse_line_style_args(&filtered, &LineStyleParseOptions::generic(BUILTIN_NAME))?;
    let marker = marker_metadata_from_appearance(&parsed.appearance);
    Ok(ParsedStemStyle {
        appearance: parsed.appearance,
        marker,
        label: parsed.label,
        baseline,
        baseline_visible,
        filled,
    })
}

fn parse_stem_args(
    args: Vec<Value>,
) -> crate::BuiltinResult<(Option<usize>, Value, Value, Vec<Value>)> {
    if args.is_empty() {
        return Err(plotting_error(
            BUILTIN_NAME,
            "stem: expected at least Y data",
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
                plotting_error(BUILTIN_NAME, "stem: expected data after axes handle")
            })?
        } else {
            first
        }
    } else {
        first
    };
    let Some(second) = it.next() else {
        let y = first;
        let y_tensor =
            Tensor::try_from(&y).map_err(|e| plotting_error(BUILTIN_NAME, format!("stem: {e}")))?;
        let len = y_tensor.data.len();
        let x = Value::Tensor(Tensor {
            data: (1..=len).map(|i| i as f64).collect(),
            shape: vec![len],
            rows: len,
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        });
        return Ok((target_axes, x, y, Vec::new()));
    };
    if matches!(second, Value::String(_) | Value::CharArray(_)) {
        let y = first;
        let y_tensor =
            Tensor::try_from(&y).map_err(|e| plotting_error(BUILTIN_NAME, format!("stem: {e}")))?;
        let len = y_tensor.data.len();
        let x = Value::Tensor(Tensor {
            data: (1..=len).map(|i| i as f64).collect(),
            shape: vec![len],
            rows: len,
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        });
        let mut rest = vec![second];
        rest.extend(it);
        return Ok((target_axes, x, y, rest));
    }
    Ok((target_axes, first, second, it.collect()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::subplot::subplot_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_plot::plots::PlotElement;

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor {
            data: data.to_vec(),
            shape: vec![data.len()],
            rows: data.len(),
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        }
    }

    #[test]
    fn stem_builds_plot_from_y_only() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _ = stem_builtin(vec![Value::Tensor(tensor_from(&[1.0, 2.0, 3.0]))]);
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert!(matches!(fig.plots().next().unwrap(), PlotElement::Stem(_)));
    }

    #[test]
    fn stem_supports_axes_target_and_filled_option() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let ax = subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(2.0)).unwrap();
        let result = stem_builtin(vec![
            Value::Num(ax),
            Value::Tensor(tensor_from(&[1.0, 2.0])),
            Value::Tensor(tensor_from(&[3.0, 4.0])),
            Value::String("DisplayName".into()),
            Value::String("Impulse".into()),
            Value::String("BaseValue".into()),
            Value::Num(-1.0),
            Value::String("filled".into()),
        ]);
        if let Err(err) = &result {
            let msg = err.to_string().to_lowercase();
            assert!(msg.contains("plotting is unavailable") || msg.contains("non-main thread"));
        }
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.plot_axes_indices()[0], 1);
        let PlotElement::Stem(stem) = fig.plots().next().unwrap() else {
            panic!("expected stem");
        };
        assert_eq!(stem.baseline, -1.0);
        assert_eq!(stem.label.as_deref(), Some("Impulse"));
        assert!(stem.marker.as_ref().map(|m| m.filled).unwrap_or(false));
    }
}
