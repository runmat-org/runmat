use runmat_builtins::BuiltinErrorDescriptor;
use runmat_builtins::{HISTOGRAM_ERROR_INTERNAL, HISTOGRAM_ERROR_INVALID_ARGUMENT};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::BarChart;
use runmat_value::{Tensor, Value};
use std::cell::RefCell;
use std::rc::Rc;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

use crate::{build_runtime_error, RuntimeError};

use super::bar::{apply_bar_style, series_bar_style};
use super::state::{
    continue_line_color_cycle_after, current_axes_state, current_hold_enabled, render_active_plot,
    HistogramHandleMetadata, PlotRenderOptions,
};
use super::style::{parse_bar_style_args, BarStyleDefaults};

const BUILTIN_NAME: &str = "histogram";

const HIST_BAR_WIDTH: f32 = 0.95;
// MATLAB histograms draw with FaceAlpha 0.6 so overlapping series stay visible.
const HIST_DEFAULT_COLOR: glam::Vec4 = glam::Vec4::new(0.15, 0.5, 0.8, 0.6);
const HIST_DEFAULT_LABEL: &str = "Frequency";

fn histogram_descriptor_error(
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

fn map_histogram_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    let mapped = histogram_descriptor_error(&HISTOGRAM_ERROR_INVALID_ARGUMENT, err.message.clone());
    let mut builder = crate::build_runtime_error(mapped.message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = mapped.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_histogram_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    let mapped = histogram_descriptor_error(&HISTOGRAM_ERROR_INTERNAL, err.message.clone());
    let mut builder = crate::build_runtime_error(mapped.message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = mapped.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn histogram_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    histogram_descriptor_error(&HISTOGRAM_ERROR_INVALID_ARGUMENT, detail)
}

fn histogram_internal(detail: impl AsRef<str>) -> RuntimeError {
    histogram_descriptor_error(&HISTOGRAM_ERROR_INTERNAL, detail)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::histogram")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "histogram",
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
    notes: "Histogram rendering terminates fusion graphs; gpuArray inputs may remain on device when shared plotting context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::histogram")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "histogram",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "histogram terminates fusion graphs and produces plotting output.",
};

#[runtime_builtin(
    name = "histogram",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::histogram"
)]
pub async fn histogram_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 1) {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:plotting:TooManyOutputs",
            "histogram: too many output arguments",
        ));
    }
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "histogram")?;
    let _arguments = args.fork();
    work.finish(histogram_impl(args, &work).await)
}
async fn histogram_impl(
    args: crate::call::BuiltinArguments,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<f64> {
    let (target_axes, data, rest) = parse_histogram_call(args)?;
    super::op_common::apply_axes_target(target_axes, BUILTIN_NAME)?;
    let retained_data = match &data {
        Value::GpuTensor(handle) => Some(
            crate::builtins::common::gpu_helpers::gather_tensor_async(handle)
                .await
                .map_err(map_histogram_invalid_argument)?,
        ),
        _ => {
            crate::builtins::common::tensor::value_into_tensor_for(BUILTIN_NAME, data.clone()).ok()
        }
    };
    let evaluation_data = retained_data
        .as_ref()
        .map(|tensor| Value::Tensor(tensor.clone()))
        .unwrap_or_else(|| data.clone());
    let (histcounts_args, style_args) = split_histogram_args(rest.view())?;
    let raw_histcounts_args = histcounts_count_args(&histcounts_args);
    let eval = crate::builtins::stats::hist::histcounts::evaluate(
        evaluation_data,
        crate::call::FunctionArgumentView::from_slice(&raw_histcounts_args),
    )
    .await
    .map_err(map_histogram_invalid_argument)?;
    let (counts_value, edges_value) = eval.into_pair();
    let counts = Tensor::try_from(&counts_value)
        .map_err(|e| histogram_invalid_argument(format!("cannot convert counts tensor: {e}")))?;
    let edges = Tensor::try_from(&edges_value)
        .map_err(|e| histogram_invalid_argument(format!("cannot convert edge tensor: {e}")))?;
    let counts = counts
        .as_f64_slice()
        .ok_or_else(|| histogram_internal("histcounts returned non-double counts"))?
        .to_vec();
    let edges = edges
        .as_f64_slice()
        .ok_or_else(|| histogram_internal("histcounts returned non-double edges"))?
        .to_vec();

    let defaults = BarStyleDefaults::new(HIST_DEFAULT_COLOR, HIST_BAR_WIDTH);
    let style = parse_bar_style_args(BUILTIN_NAME, &style_args, defaults)
        .map_err(map_histogram_invalid_argument)?;
    let explicit_display_name = style.label.clone();
    let normalization = infer_normalization(&histcounts_args)?;
    crate::builtins::plotting::properties::validate_histogram_normalization(
        &normalization,
        BUILTIN_NAME,
    )
    .map_err(map_histogram_invalid_argument)?;
    let normalization_denominator = retained_data
        .as_ref()
        .map(|tensor| tensor.len() as f64)
        .unwrap_or_else(|| counts.iter().sum());
    let render_counts = crate::builtins::plotting::properties::apply_histogram_normalization(
        &counts,
        &edges,
        &normalization,
        normalization_denominator,
    );
    let labels = histogram_labels_from_edges(&edges);
    let mut chart = BarChart::new(labels, render_counts)
        .map_err(|e| histogram_internal(format!("chart construction failed: {e}")))?;
    chart.set_histogram_bin_edges(edges.clone());
    let color_axes = current_axes_state().active_index;
    let hold_enabled = current_hold_enabled();
    let series_style = series_bar_style(&style, color_axes, hold_enabled, 0);
    apply_bar_style(&mut chart, &series_style, HIST_DEFAULT_LABEL);

    let mut chart_opt = Some(chart);
    let plot_index_out = Rc::new(RefCell::new(None));
    let plot_index_slot = Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    work.check()?;
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions::default(),
        move |figure, axes| {
            let plot_index = figure.add_bar_chart_on_axes(
                chart_opt.take().expect("histogram chart consumed once"),
                axes,
            );
            *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            Ok(())
        },
    );
    if !hold_enabled {
        // Rendering cleared the axes and reset the color order; resume it after this series.
        continue_line_color_cycle_after(color_axes, 1);
    }
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = crate::builtins::plotting::state::register_histogram_handle(
        figure_handle,
        axes,
        plot_index,
        edges.clone(),
        counts.clone(),
        normalization.clone(),
        normalization_denominator,
        histogram_metadata(&edges, &style, retained_data, false, "bar"),
    );
    if let Some(display_name) = explicit_display_name {
        crate::builtins::plotting::state::set_histogram_handle_display_name(
            figure_handle,
            axes,
            plot_index,
            Some(display_name),
        )
        .map_err(|err| {
            histogram_internal(format!("failed to update histogram display name: {err}"))
        })?;
    }
    if let Err(err) = render_result {
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(handle);
        }
        return Err(map_histogram_internal(err));
    }
    Ok(handle)
}

fn parse_histogram_call(
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<(
    super::op_common::AxesTarget,
    Value,
    crate::call::BuiltinArguments,
)> {
    let (target, mut args) = super::op_common::split_leading_axes_arguments(args, BUILTIN_NAME)?;
    let data = args
        .pop_front()
        .ok_or_else(|| histogram_invalid_argument("expected data input"))?;
    Ok((target, data, args))
}

fn split_histogram_args(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<(Vec<Value>, Vec<Value>)> {
    Ok({
        let mut hist_args = Vec::new();
        let mut style_args = Vec::new();
        let mut idx = 0usize;
        while idx < args.len() {
            let Some(key) = value_as_string(&args[idx]).map_err(|source| {
                crate::builtins::plotting::plotting_error_with_source(
                    "histogram",
                    source.to_string(),
                    source,
                )
            })?
            else {
                if idx == 0 {
                    hist_args.push(args[idx].clone());
                    idx += 1;
                    continue;
                }
                style_args.extend(
                    args.range(idx..)
                        .expect("validated option tail")
                        .iter()
                        .cloned(),
                );
                break;
            };
            if idx + 1 >= args.len() {
                hist_args.push(args[idx].clone());
                break;
            }
            let lower = key.trim().to_ascii_lowercase();
            let target = match lower.as_str() {
                "binedges" | "numbins" | "binwidth" | "binlimits" | "normalization"
                | "binmethod" => &mut hist_args,
                _ => &mut style_args,
            };
            target.push(args[idx].clone());
            target.push(args[idx + 1].clone());
            idx += 2;
        }
        (hist_args, style_args)
    })
}

fn histogram_labels_from_edges(edges: &[f64]) -> Vec<String> {
    edges
        .windows(2)
        .map(|pair| format!("[{:.3}, {:.3})", pair[0], pair[1]))
        .collect()
}

fn infer_normalization(args: &[Value]) -> crate::BuiltinResult<String> {
    Ok({
        let mut idx = 0usize;
        while idx + 1 < args.len() {
            if let Some(key) = value_as_string(&args[idx]).map_err(|source| {
                crate::builtins::plotting::plotting_error_with_source(
                    "histogram",
                    source.to_string(),
                    source,
                )
            })? {
                if key.trim().eq_ignore_ascii_case("Normalization") {
                    if let Some(v) = value_as_string(&args[idx + 1]).map_err(|source| {
                        crate::builtins::plotting::plotting_error_with_source(
                            "histogram",
                            source.to_string(),
                            source,
                        )
                    })? {
                        return Ok(v.trim().to_ascii_lowercase());
                    }
                }
                idx += 2;
            } else {
                idx += 1;
            }
        }
        "count".to_string()
    })
}

fn histcounts_count_args(args: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    let mut idx = 0usize;
    while idx < args.len() {
        if crate::builtins::plotting::style::value_text_equals_ascii(
            &args[idx],
            "Normalization",
            true,
        ) {
            idx += 2;
        } else {
            out.push(args[idx].clone());
            idx += 1;
        }
    }
    out
}

pub(crate) fn histogram_metadata(
    edges: &[f64],
    style: &super::style::BarStyle,
    data: Option<Tensor>,
    is_polar: bool,
    display_style: &str,
) -> HistogramHandleMetadata {
    let mut metadata = HistogramHandleMetadata::new(edges);
    metadata.data = data;
    metadata.display_style = display_style.to_string();
    metadata.face_color = color_to_property_string(style.face_rgba());
    metadata.face_alpha = if (style.face_alpha - 1.0).abs() > f32::EPSILON {
        style.face_alpha as f64
    } else {
        style.face_rgba().w as f64
    };
    metadata.edge_color = style
        .edge_rgba()
        .map(color_to_property_string)
        .unwrap_or_else(|| "auto".to_string());
    metadata.is_polar = is_polar;
    metadata
}

pub(crate) fn color_to_property_string(color: glam::Vec4) -> String {
    let candidates = [
        (glam::Vec4::new(1.0, 0.0, 0.0, color.w), "r"),
        (glam::Vec4::new(0.0, 1.0, 0.0, color.w), "g"),
        (glam::Vec4::new(0.0, 0.0, 1.0, color.w), "b"),
        (glam::Vec4::new(0.0, 0.0, 0.0, color.w), "k"),
        (glam::Vec4::new(1.0, 1.0, 1.0, color.w), "w"),
        (glam::Vec4::new(1.0, 1.0, 0.0, color.w), "y"),
        (glam::Vec4::new(1.0, 0.0, 1.0, color.w), "m"),
        (glam::Vec4::new(0.0, 1.0, 1.0, color.w), "c"),
    ];
    for (candidate, name) in candidates {
        if (candidate - color).abs().max_element() < 1e-6 {
            return name.to_string();
        }
    }
    format!("[{:.3},{:.3},{:.3}]", color.x, color.y, color.z)
}

fn value_as_string(value: &Value) -> Result<Option<String>, runmat_value::Utf16DecodeError> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(s)) => Some(s.as_str().to_owned()),
        Value::CharArray(chars) => Some(runmat_value::decode_utf16_code_units(chars.data())?),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    async fn histogram_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
        super::histogram_builtin(
            crate::context::legacy::active(),
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .await
    }

    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_plot::plots::PlotElement;
    use runmat_value::IntegerStorage;

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![data.len()]).expect("histogram test vector")
    }

    #[test]
    fn histogram_accepts_edges_vector_semantics() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let out = futures::executor::block_on(histogram_builtin(vec![
            Value::Tensor(tensor_from(&[0.1, 0.2, 0.9, 1.1])),
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).expect("histogram edges")),
        ]));
        let out = out.unwrap();
        let counts = Tensor::try_from(
            &get_builtin(vec![Value::Num(out), Value::String("Values".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(counts.as_f64_slice(), Some(&[3.0, 1.0][..]));
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert!(matches!(fig.plots().next().unwrap(), PlotElement::Bar(_)));
    }

    #[test]
    fn histogram_accepts_typed_integer_samples_at_the_histcounts_boundary() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let samples =
            Tensor::new_integer(IntegerStorage::U16(vec![0, 1, 1, 2]), vec![4, 1]).unwrap();
        let out = futures::executor::block_on(histogram_builtin(vec![
            Value::Tensor(samples),
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]).unwrap()),
        ]))
        .unwrap();
        let counts = Tensor::try_from(
            &get_builtin(vec![Value::Num(out), Value::String("Values".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(counts.as_f64_slice(), Some(&[1.0, 2.0, 1.0][..]));
    }

    #[test]
    fn histogram_retains_typed_data_and_normalizes_raw_counts_once() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let samples =
            Tensor::new_integer(IntegerStorage::U16(vec![0, 1, 1, 2]), vec![4, 1]).unwrap();
        let handle = futures::executor::block_on(histogram_builtin(vec![
            Value::Tensor(samples),
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]).unwrap()),
            Value::String("Normalization".into()),
            Value::String("probability".into()),
        ]))
        .unwrap();
        let data = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("Data".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(data.numeric_dtype(), runmat_value::NumericDType::U16);
        assert_eq!(
            data.integer_storage(),
            Some(&IntegerStorage::U16(vec![0, 1, 1, 2]))
        );
        let values = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("Values".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(values.as_f64_slice(), Some(&[0.25, 0.5, 0.25][..]));
    }

    #[test]
    fn histogram_supports_extended_normalizations() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let out = futures::executor::block_on(histogram_builtin(vec![
            Value::Tensor(tensor_from(&[1.0, 2.0, 3.0])),
            Value::String("NumBins".into()),
            Value::Num(3.0),
            Value::String("Normalization".into()),
            Value::String("cdf".into()),
        ]));
        let out = out.unwrap();
        let counts = Tensor::try_from(
            &get_builtin(vec![Value::Num(out), Value::String("BinCounts".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            counts
                .as_f64_slice()
                .and_then(|values| values.last())
                .copied()
                .unwrap_or_default(),
            1.0
        );
    }

    #[test]
    fn histogram_probability_uses_all_input_elements_and_keeps_raw_bin_counts() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle = futures::executor::block_on(histogram_builtin(vec![
            Value::Tensor(tensor_from(&[0.25, 1.25, 99.0, f64::NAN])),
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap()),
            Value::String("Normalization".into()),
            Value::String("probability".into()),
        ]))
        .unwrap();
        let raw = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("BinCounts".into())]).unwrap(),
        )
        .unwrap();
        let values = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("Values".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(raw.as_f64_slice(), Some(&[1.0, 1.0][..]));
        assert_eq!(values.as_f64_slice(), Some(&[0.25, 0.25][..]));
        let figure = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Bar(chart) = figure.plots().next().unwrap() else {
            panic!("expected bar chart");
        };
        assert_eq!(
            futures::executor::block_on(chart.export_scene_values()).unwrap(),
            vec![0.25, 0.25]
        );
    }

    #[test]
    fn histogram_rejects_unknown_normalization_at_construction() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let error = futures::executor::block_on(histogram_builtin(vec![
            Value::Tensor(tensor_from(&[1.0, 2.0, 3.0])),
            Value::String("Normalization".into()),
            Value::String("bogus".into()),
        ]))
        .expect_err("unknown histogram normalization");
        assert_eq!(error.identifier(), Some("RunMat:histogram:InvalidArgument"));
    }

    #[test]
    fn histogram_supports_axes_target_and_histcounts_semantics() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();
        let out = futures::executor::block_on(histogram_builtin(vec![
            Value::Num(ax),
            Value::Tensor(tensor_from(&[
                0.0,
                1.0,
                2.0,
                100.0,
                f64::NAN,
                f64::INFINITY,
            ])),
            Value::Tensor(
                Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]).expect("histogram edges"),
            ),
        ]))
        .unwrap();
        let counts = Tensor::try_from(
            &get_builtin(vec![Value::Num(out), Value::String("BinCounts".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(counts.as_f64_slice(), Some(&[1.0, 1.0, 1.0][..]));
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.plot_axes_indices()[0], 1);
    }

    #[test]
    fn histogram_keeps_style_args_after_positional_edges() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle = futures::executor::block_on(histogram_builtin(vec![
            Value::Tensor(tensor_from(&[0.0, 0.2, 1.0, 2.0])),
            Value::Tensor(
                Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]).expect("histogram edges"),
            ),
            Value::String("DisplayName".into()),
            Value::String("angles".into()),
        ]))
        .unwrap();

        let display_name = get_builtin(vec![
            Value::Num(handle),
            Value::String("DisplayName".into()),
        ])
        .unwrap();
        assert_eq!(display_name, Value::String("angles".into()));
        let counts = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("BinCounts".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(counts.as_f64_slice(), Some(&[2.0, 1.0, 1.0][..]));
    }

    #[test]
    fn histogram_supports_binmethod_auto_and_countdensity() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let out = futures::executor::block_on(histogram_builtin(vec![
            Value::Tensor(tensor_from(&[1.0, 1.1, 2.0, 2.1, 2.2, 5.0])),
            Value::String("BinMethod".into()),
            Value::String("auto".into()),
            Value::String("Normalization".into()),
            Value::String("countdensity".into()),
        ]))
        .unwrap();
        let counts = Tensor::try_from(
            &get_builtin(vec![Value::Num(out), Value::String("BinCounts".into())]).unwrap(),
        )
        .unwrap();
        let counts = counts.as_f64_slice().expect("double counts");
        assert!(!counts.is_empty());
        assert!(counts.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn histogram_returns_object_handle_with_get_set_semantics() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle =
            futures::executor::block_on(histogram_builtin(vec![Value::Tensor(tensor_from(&[
                1.0, 2.0, 3.0,
            ]))]))
            .unwrap();
        let ty = get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap();
        assert_eq!(ty, Value::String("histogram".into()));
        set_builtin(vec![
            Value::Num(handle),
            Value::String("Normalization".into()),
            Value::String("cdf".into()),
        ])
        .unwrap();
        let norm = get_builtin(vec![
            Value::Num(handle),
            Value::String("Normalization".into()),
        ])
        .unwrap();
        assert_eq!(norm, Value::String("cdf".into()));
    }

    #[test]
    fn histogram_supports_displayname_property() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle =
            futures::executor::block_on(histogram_builtin(vec![Value::Tensor(tensor_from(&[
                1.0, 2.0, 3.0,
            ]))]))
            .unwrap();
        set_builtin(vec![
            Value::Num(handle),
            Value::String("DisplayName".into()),
            Value::String("Process A".into()),
        ])
        .unwrap();
        let display_name = get_builtin(vec![
            Value::Num(handle),
            Value::String("DisplayName".into()),
        ])
        .unwrap();
        assert_eq!(display_name, Value::String("Process A".into()));

        let props = get_builtin(vec![Value::Num(handle)]).unwrap();
        let Value::Struct(props) = props else {
            panic!("expected struct from get(handle)");
        };
        assert_eq!(
            props.fields.get("DisplayName"),
            Some(&Value::String("Process A".into()))
        );
    }

    #[test]
    fn histogram_descriptor_includes_axes_signature() {
        let labels: Vec<&str> = runmat_builtins::HISTOGRAM_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = histogram(X)"));
        assert!(labels.contains(&"h = histogram(X, bins)"));
        assert!(labels.contains(&"h = histogram(ax, X, Name, Value, ...)"));
    }

    #[test]
    fn histogram_missing_data_uses_stable_identifier() {
        let err = futures::executor::block_on(histogram_builtin(Vec::new()))
            .expect_err("expected histogram argument validation error");
        assert_eq!(
            err.identifier(),
            HISTOGRAM_ERROR_INVALID_ARGUMENT.identifier
        );
    }
}
