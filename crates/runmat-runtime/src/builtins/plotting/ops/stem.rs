use runmat_builtins::BuiltinErrorDescriptor;
pub use runmat_builtins::{STEM_ERROR_INTERNAL, STEM_ERROR_INVALID_ARGUMENT};
use runmat_macros::runtime_builtin;
use runmat_plot::gpu::line::{
    self, LineGpuInputs as MarkerGpuInputs, LineGpuParams as MarkerGpuParams,
};
use runmat_plot::gpu::stem::{StemGpuInputs, StemGpuParams};
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::NumericPlotData;
use runmat_plot::plots::{LineMarkerAppearance, StemPlot};
#[cfg(test)]
use runmat_value::Tensor;
use runmat_value::Value;
use std::cell::RefCell;
use std::rc::Rc;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, RuntimeError};

use super::common::numeric_plot_data_pair;
use super::gpu_helpers::gpu_xy_bounds_async;
use super::op_common::line_inputs::NumericInput;
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{
    marker_metadata_from_appearance, parse_line_style_args, LineAppearance, LineStyleParseOptions,
};

const BUILTIN_NAME: &str = "stem";

fn stem_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_stem_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    stem_error_with_detail(&STEM_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_stem_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    stem_error_with_detail(&STEM_ERROR_INTERNAL, err.message)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::stem")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "stem",
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
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::stem"
)]
pub async fn stem_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::STEM_DESCRIPTOR,
        &runmat_builtins::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
        "stem",
    )?;
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "stem")?;
    // Keep the original rooted owner alive even when helpers extract native data.
    let _arguments = args.fork();
    work.finish(stem_impl(args, &work).await)
}

async fn stem_impl(
    args: crate::call::BuiltinArguments,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<f64> {
    let (target_axes, x, y, rest) = parse_stem_args(args).map_err(map_stem_invalid_argument)?;
    let parsed = parse_stem_style_args(&rest).map_err(map_stem_invalid_argument)?;
    let x_input = NumericInput::from_value(x, BUILTIN_NAME).map_err(map_stem_invalid_argument)?;
    let y_input = NumericInput::from_value(y, BUILTIN_NAME).map_err(map_stem_invalid_argument)?;
    let label = parsed.label.clone().unwrap_or_else(|| "Data".into());
    // Complete provider work before entering the synchronous figure registry.
    // Pending/error/drop therefore cannot publish a partially prepared plot.
    let gpu_plot = if let (Some(x), Some(y)) = (x_input.gpu_handle(), y_input.gpu_handle()) {
        build_stem_gpu_plot(BUILTIN_NAME, x, y, &parsed, &label)
            .await
            .map_err(map_stem_internal)?
    } else {
        None
    };
    let plot = match gpu_plot {
        Some(plot) => plot,
        None => {
            let x = x_input
                .into_tensor_async(BUILTIN_NAME)
                .await
                .map_err(map_stem_invalid_argument)?;
            let y = y_input
                .into_tensor_async(BUILTIN_NAME)
                .await
                .map_err(map_stem_invalid_argument)?;
            let (x, y) =
                numeric_plot_data_pair(x, y, BUILTIN_NAME).map_err(map_stem_invalid_argument)?;
            build_stem_plot(x, y, &parsed, &label).map_err(map_stem_invalid_argument)?
        }
    };
    let mut plot = Some(plot);
    let opts = PlotRenderOptions::default();
    let plot_index_out = Rc::new(RefCell::new(None));
    let plot_index_slot = Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    work.check()?;
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let axes = target_axes.unwrap_or(axes);
        let plot = plot.take().expect("stem plot consumed once");
        let plot_index = figure.add_stem_plot_on_axes(plot, axes);
        *plot_index_slot.borrow_mut() = Some((axes, plot_index));
        Ok(())
    });
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle =
        crate::builtins::plotting::state::register_stem_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(handle);
        }
        return Err(map_stem_internal(err));
    }
    Ok(handle)
}

fn build_stem_plot(
    x: NumericPlotData,
    y: NumericPlotData,
    parsed: &ParsedStemStyle,
    label: &str,
) -> crate::BuiltinResult<StemPlot> {
    let mut plot = StemPlot::new_with_source(x, y)
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
    if let Some(visible) = parsed.appearance.visible {
        plot.set_visible(visible);
    }
    Ok(plot)
}

async fn build_stem_gpu_plot(
    name: &'static str,
    x: &runmat_accelerate_api::GpuTensorHandle,
    y: &runmat_accelerate_api::GpuTensorHandle,
    parsed: &ParsedStemStyle,
    label: &str,
) -> crate::BuiltinResult<Option<StemPlot>> {
    let mut gpu_exports = super::gpu_helpers::GpuBufferExports::default();
    // Only unavailable zero-copy rendering support permits host preparation.
    // Once admitted, invalid metadata, reduction and vertex failures propagate.
    let Ok(context) = super::gpu_helpers::ensure_shared_wgpu_context(name) else {
        return Ok(None);
    };
    let Some(x_ref) = gpu_exports.export(x, name)? else {
        return Ok(None);
    };
    let Some(y_ref) = gpu_exports.export(y, name)? else {
        return Ok(None);
    };
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
    let xy_bounds = gpu_xy_bounds_async(x, y, name).await?;
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
    let inputs = StemGpuInputs {
        x_buffer: x_ref.clone().into(),
        y_buffer: y_ref.clone().into(),
        len: x_ref.len as u32,
        scalar,
    };
    let gpu_vertices = context
        .access
        .with_shared(|emission| {
            runmat_plot::gpu::stem::pack_vertices_from_xy(
                emission,
                &inputs,
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
        })
        .map_err(|error| error.to_string())
        .and_then(|result| result)
        .map_err(|e| plotting_error(name, format!("{name}: failed to build GPU vertices: {e}")))?;
    let mut plot = StemPlot::from_gpu_buffer(
        parsed.appearance.color,
        parsed.appearance.line_width,
        parsed.appearance.line_style,
        parsed.baseline,
        parsed.appearance.color,
        parsed.baseline_visible,
        gpu_vertices,
        (if parsed.baseline_visible { 2 } else { 0 }) + (x_ref.len * 2),
        bounds,
    )
    .with_gpu_source_inputs(inputs)
    .with_label(label);
    if let Some(marker) = parsed.marker.clone() {
        let mut marker = marker;
        if parsed.filled {
            marker.filled = true;
        }
        let marker_gpu = context
            .access
            .with_shared(|emission| {
                line::pack_marker_vertices_from_xy(
                    emission,
                    &MarkerGpuInputs {
                        x_buffer: x_ref.clone().into(),
                        y_buffer: y_ref.clone().into(),
                        len: x_ref.len as u32,
                        scalar,
                    },
                    &MarkerGpuParams {
                        color: marker.face_color,
                        half_width_px: 0.0,
                        viewport_width_px: 1.0,
                        viewport_height_px: 1.0,
                        x_min: 0.0,
                        x_span: 1.0,
                        y_min: 0.0,
                        y_span: 1.0,
                        x_log: false,
                        y_log: false,
                        line_style: runmat_plot::plots::LineStyle::Solid,
                        marker_size: marker.size,
                    },
                )
            })
            .map_err(|error| error.to_string())
            .and_then(|result| result)
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
        let marker_gpu = context
            .access
            .with_shared(|emission| {
                line::pack_marker_vertices_from_xy(
                    emission,
                    &MarkerGpuInputs {
                        x_buffer: x_ref.clone().into(),
                        y_buffer: y_ref.clone().into(),
                        len: x_ref.len as u32,
                        scalar,
                    },
                    &MarkerGpuParams {
                        color: marker.face_color,
                        half_width_px: 0.0,
                        viewport_width_px: 1.0,
                        viewport_height_px: 1.0,
                        x_min: 0.0,
                        x_span: 1.0,
                        y_min: 0.0,
                        y_span: 1.0,
                        x_log: false,
                        y_log: false,
                        line_style: runmat_plot::plots::LineStyle::Solid,
                        marker_size: marker.size,
                    },
                )
            })
            .map_err(|error| error.to_string())
            .and_then(|result| result)
            .map_err(|e| {
                plotting_error(
                    name,
                    format!("{name}: failed to build marker vertices: {e}"),
                )
            })?;
        plot.set_marker(Some(marker));
        plot.set_marker_gpu_vertices(Some(marker_gpu));
    }
    if let Some(visible) = parsed.appearance.visible {
        plot.set_visible(visible);
    }
    Ok(Some(plot))
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
        if let Some(text) = super::style::value_as_string(&args[idx]).map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source(
                "stem",
                source.to_string(),
                source,
            )
        })? {
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
    args: crate::call::BuiltinArguments,
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
    let first = if let Ok(crate::builtins::plotting::properties::PlotHandle::Axes(_, axes)) =
        crate::builtins::plotting::properties::resolve_plot_handle(&first, BUILTIN_NAME)
    {
        target_axes = Some(axes);
        it.next()
            .ok_or_else(|| plotting_error(BUILTIN_NAME, "stem: expected data after axes handle"))?
    } else {
        first
    };
    let Some(second) = it.next() else {
        let y = first;
        let x = NumericInput::implicit_x(&y, BUILTIN_NAME, 0)?;
        return Ok((target_axes, x, y, Vec::new()));
    };
    if matches!(second, Value::String(_) | Value::CharArray(_)) {
        let y = first;
        let x = NumericInput::implicit_x(&y, BUILTIN_NAME, 0)?;
        let mut rest = vec![second];
        rest.extend(it);
        return Ok((target_axes, x, y, rest));
    }
    Ok((target_axes, first, second, it.collect()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::STEM_DESCRIPTOR;
    fn stem_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
        futures::executor::block_on(super::stem_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
    }
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::subplot::subplot_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_plot::plots::PlotElement;

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![data.len()]).expect("stem test vector")
    }

    #[test]
    fn stem_builds_plot_from_y_only() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle = stem_builtin(vec![Value::Tensor(tensor_from(&[1.0, 2.0, 3.0]))]).unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert!(matches!(fig.plots().next().unwrap(), PlotElement::Stem(_)));
        let ty = get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap();
        assert_eq!(ty, Value::String("stem".into()));
        for y in [
            Value::Num(4.0),
            Value::Int(runmat_value::IntValue::U64(9_007_199_254_740_993)),
            Value::Tensor(Tensor::from_f32(vec![4.0], vec![1, 1]).unwrap()),
        ] {
            let expected = NumericInput::from_value(y.clone(), "stem").unwrap();
            let NumericInput::Host(expected) = expected else {
                panic!("host scalar")
            };
            let handle = stem_builtin(vec![y]).expect("scalar stem");
            let Value::Tensor(x) =
                get_builtin(vec![Value::Num(handle), Value::from("XData")]).unwrap()
            else {
                panic!("XData")
            };
            let Value::Tensor(y) =
                get_builtin(vec![Value::Num(handle), Value::from("YData")]).unwrap()
            else {
                panic!("YData")
            };
            assert_eq!(x.materialize_f64(), vec![1.0]);
            assert_eq!(y.numeric_dtype(), expected.numeric_dtype());
            assert_eq!(y.integer_storage(), expected.integer_storage());
            assert_eq!(y.materialize_f64(), expected.materialize_f64());
        }
    }

    #[test]
    fn stem_graphics_properties_preserve_wide_integer_source_storage() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let wide = 9_007_199_254_740_993_u64;
        let x = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![wide, wide + 2]),
            vec![1, 2],
        )
        .expect("XData");
        let y = Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![-3, 4]), vec![2, 1])
            .expect("YData");
        let handle = stem_builtin(vec![Value::Tensor(x), Value::Tensor(y)]).expect("stem");
        let Value::Tensor(x) =
            get_builtin(vec![Value::Num(handle), Value::from("XData")]).expect("XData")
        else {
            panic!("expected XData tensor");
        };
        let Value::Tensor(y) =
            get_builtin(vec![Value::Num(handle), Value::from("YData")]).expect("YData")
        else {
            panic!("expected YData tensor");
        };
        assert_eq!(
            x.integer_storage(),
            Some(&runmat_value::IntegerStorage::U64(vec![wide, wide + 2]))
        );
        assert_eq!(
            y.integer_storage(),
            Some(&runmat_value::IntegerStorage::I16(vec![-3, 4]))
        );
        assert_eq!(x.shape(), [1, 2]);
        assert_eq!(y.shape(), [2, 1]);
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
            assert!(crate::builtins::plotting::state::presentation_unavailable(
                err
            ));
        }
        let handle = result.unwrap_or(-1.0);
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.plot_axes_indices()[0], 1);
        let PlotElement::Stem(stem) = fig.plots().next().unwrap() else {
            panic!("expected stem");
        };
        assert_eq!(stem.baseline, -1.0);
        assert_eq!(stem.label.as_deref(), Some("Impulse"));
        assert!(stem.marker.as_ref().map(|m| m.filled).unwrap_or(false));
        set_builtin(vec![
            Value::Num(handle),
            Value::String("Filled".into()),
            Value::Bool(false),
        ])
        .unwrap();
        let filled = get_builtin(vec![Value::Num(handle), Value::String("Filled".into())]).unwrap();
        assert_eq!(filled, Value::Bool(false));
    }

    #[test]
    fn stem_accepts_scalar_point() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _ = stem_builtin(vec![Value::Num(1.0), Value::Num(2.0)]).unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Stem(plot) = fig.plots().next().unwrap() else {
            panic!("expected stem")
        };
        assert_eq!(plot.x, vec![1.0]);
        assert_eq!(plot.y, vec![2.0]);
    }

    #[test]
    fn stem_descriptor_signatures_cover_supported_forms() {
        let labels: Vec<&str> = STEM_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = stem(Y)"));
        assert!(labels.contains(&"h = stem(Y, LineSpec)"));
        assert!(labels.contains(&"h = stem(X, Y)"));
        assert!(labels.contains(&"h = stem(ax, Y, Name, Value, ...)"));
        assert!(labels.contains(&"h = stem(ax, X, Y, Name, Value, ...)"));
    }

    #[test]
    fn stem_missing_input_uses_stable_identifier() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let err = stem_builtin(vec![]).expect_err("missing args should fail");
        assert_eq!(err.identifier(), STEM_ERROR_INVALID_ARGUMENT.identifier);
    }
}
