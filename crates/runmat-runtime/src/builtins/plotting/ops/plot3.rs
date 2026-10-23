use runmat_accelerate_api::ProviderPrecision;
use runmat_builtins::BuiltinErrorDescriptor;
pub use runmat_builtins::{PLOT3_ERROR_INTERNAL, PLOT3_ERROR_INVALID_ARGUMENT};
use runmat_macros::runtime_builtin;
use runmat_plot::gpu::{line3::Line3GpuInputs, ScalarType};
use runmat_plot::plots::NumericPlotData;
use runmat_plot::plots::{Line3Plot, LineStyle};
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

use super::common::numeric_plot_data_triplet;
use super::gpu_helpers::gpu_xyz_bounds_async;
use super::op_common::line_inputs::NumericInput;
use super::op_common::{apply_axes_target, split_leading_axes_arguments};
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_line_style_args, LineAppearance, LineStyleParseOptions};
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "plot3";

fn plot3_error_with_detail(
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

fn map_plot3_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    plot3_error_with_detail(&PLOT3_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_plot3_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    plot3_error_with_detail(&PLOT3_ERROR_INTERNAL, err.message)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::plot3")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "plot3",
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
    notes: "plot3 is a plotting sink; GPU inputs may remain on device when shared plotting context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::plot3")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "plot3",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "plot3 performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "plot3",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::plot3"
)]
pub async fn plot3_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::PLOT3_DESCRIPTOR,
        &runmat_builtins::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
        "plot3",
    )?;
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "plot3")?;
    // Keep the original rooted owner alive even when helpers extract native data.
    let _arguments = args.fork();
    work.finish(plot3_impl(args, &work).await)
}

async fn plot3_impl(
    args: crate::call::BuiltinArguments,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<f64> {
    let (axes_target, args) =
        split_leading_axes_arguments(args, BUILTIN_NAME).map_err(map_plot3_invalid_argument)?;
    let (mut plans, line_style_order) =
        parse_plot3_series_specs(args).map_err(map_plot3_invalid_argument)?;

    let opts = PlotRenderOptions {
        // MATLAB leaves the data aspect free; `axis equal` sets it.
        axis_equal: false,
        ..Default::default()
    };

    let mut plots = Vec::with_capacity(plans.len());
    let mut ordered_style_idx = 0usize;
    for (series_idx, mut plan) in plans.drain(..).enumerate() {
        if !plan.line_style_explicit {
            if let Some(order) = line_style_order.as_ref().filter(|order| !order.is_empty()) {
                plan.appearance.line_style = order[ordered_style_idx % order.len()];
                ordered_style_idx += 1;
            } else {
                plan.appearance.line_style = match series_idx % 4 {
                    0 => LineStyle::Solid,
                    1 => LineStyle::Dashed,
                    2 => LineStyle::Dotted,
                    _ => LineStyle::DashDot,
                };
            }
        }
        let label = plan
            .label
            .take()
            .unwrap_or_else(|| format!("Series {}", series_idx + 1));
        if let Some((xg, yg, zg)) = plan.data.gpu_handles() {
            if let Some(plot) =
                build_line3_gpu_plot_async(xg, yg, zg, &label, &plan.appearance).await?
            {
                plots.push(plot);
                continue;
            }
        }
        let (x, y, z) = plan
            .data
            .into_tensors_async(BUILTIN_NAME)
            .await
            .map_err(map_plot3_invalid_argument)?;
        let (x, y, z) =
            numeric_plot_data_triplet(x, y, z, BUILTIN_NAME).map_err(map_plot3_invalid_argument)?;
        plots.push(
            build_line3_plot_from_numeric_data(x, y, z, &label, &plan.appearance)
                .map_err(map_plot3_invalid_argument)?,
        );
    }
    let mut plots_opt = Some(plots);
    work.check()?;
    apply_axes_target(axes_target, BUILTIN_NAME).map_err(map_plot3_invalid_argument)?;
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    work.check()?;
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let plots = plots_opt.take().expect("plot3 consumed once");
        for (idx, plot) in plots.into_iter().enumerate() {
            let plot_index = figure.add_line3_plot_on_axes(plot, axes);
            if idx == 0 {
                *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            }
        }
        Ok(())
    });
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle =
        crate::builtins::plotting::state::register_line3_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(handle);
        }
        return Err(map_plot3_internal(err));
    }
    Ok(handle)
}

#[derive(Clone)]
struct Plot3SeriesInput {
    x: NumericInput,
    y: NumericInput,
    z: NumericInput,
}

impl Plot3SeriesInput {
    fn new(x: Value, y: Value, z: Value) -> crate::BuiltinResult<Self> {
        Ok(Self {
            x: NumericInput::from_value(x, BUILTIN_NAME)?,
            y: NumericInput::from_value(y, BUILTIN_NAME)?,
            z: NumericInput::from_value(z, BUILTIN_NAME)?,
        })
    }
    fn gpu_handles(
        &self,
    ) -> Option<(
        &runmat_accelerate_api::GpuTensorHandle,
        &runmat_accelerate_api::GpuTensorHandle,
        &runmat_accelerate_api::GpuTensorHandle,
    )> {
        Some((
            self.x.gpu_handle()?,
            self.y.gpu_handle()?,
            self.z.gpu_handle()?,
        ))
    }
    async fn into_tensors_async(
        self,
        name: &'static str,
    ) -> crate::BuiltinResult<(
        runmat_value::Tensor,
        runmat_value::Tensor,
        runmat_value::Tensor,
    )> {
        Ok((
            self.x.into_tensor_async(name).await?,
            self.y.into_tensor_async(name).await?,
            self.z.into_tensor_async(name).await?,
        ))
    }
}

struct Plot3SeriesPlan {
    data: Plot3SeriesInput,
    appearance: LineAppearance,
    line_style_explicit: bool,
    label: Option<String>,
}

fn parse_plot3_series_specs(
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<(Vec<Plot3SeriesPlan>, Option<Vec<LineStyle>>)> {
    if args.len() < 3 {
        return Err(plotting_error(
            BUILTIN_NAME,
            "plot3: expected at least one X,Y,Z series",
        ));
    }
    let mut plans = Vec::new();
    let mut line_style_order = None;
    let mut idx = 0usize;
    let opts = LineStyleParseOptions::generic(BUILTIN_NAME);
    while idx + 2 < args.len() {
        if !is_numeric_value(&args[idx])
            || !is_numeric_value(&args[idx + 1])
            || !is_numeric_value(&args[idx + 2])
        {
            return Err(plotting_error(
                BUILTIN_NAME,
                "plot3: expected X, Y, and Z data triplets",
            ));
        }
        let x = args[idx].clone();
        let y = args[idx + 1].clone();
        let z = args[idx + 2].clone();
        idx += 3;
        let mut style_tokens = Vec::new();
        loop {
            let should_consume =
                matches!(args.get(idx), Some(Value::String(_) | Value::CharArray(_)));
            if !should_consume {
                break;
            }
            let token = args[idx].clone();
            idx += 1;
            style_tokens.push(token.clone());
            if let Some(text) = super::style::value_as_string(&token).map_err(|source| {
                crate::builtins::plotting::plotting_error_with_source(
                    "plot3",
                    source.to_string(),
                    source,
                )
            })? {
                let lower = text.trim().to_ascii_lowercase();
                if super::style::looks_like_option_name(&lower) {
                    if idx >= args.len() {
                        return Err(plotting_error(
                            BUILTIN_NAME,
                            "plot3: name-value arguments must come in pairs",
                        ));
                    }
                    style_tokens.push(args[idx].clone());
                    idx += 1;
                }
            }
        }
        let parsed = parse_line_style_args(&style_tokens, &opts)?;
        if parsed.appearance.marker.is_some() {
            return Err(plotting_error(
                BUILTIN_NAME,
                "plot3: marker LineSpec and Marker options are not supported until Line3 marker rendering is implemented",
            ));
        }
        if let Some(order) = parsed.line_style_order.clone() {
            line_style_order = Some(order);
        }
        plans.push(Plot3SeriesPlan {
            data: Plot3SeriesInput::new(x, y, z)?,
            appearance: parsed.appearance,
            line_style_explicit: parsed.line_style_explicit,
            label: parsed.label,
        });
    }
    Ok((plans, line_style_order))
}

fn is_numeric_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(_) | Value::GpuTensor(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_)
    )
}

pub(super) fn build_line3_plot_for_builtin(
    builtin: &'static str,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    label: &str,
    appearance: &LineAppearance,
) -> crate::BuiltinResult<Line3Plot> {
    let mut plot = Line3Plot::new(x, y, z)
        .map_err(|e| plotting_error(builtin, format!("{builtin}: {e}")))?
        .with_style(
            appearance.color,
            appearance.line_width,
            appearance.line_style,
        )
        .with_label(label);
    if let Some(visible) = appearance.visible {
        plot.set_visible(visible);
    }
    Ok(plot)
}

pub(super) fn build_line3_plot_from_numeric_data_for_builtin(
    builtin: &'static str,
    x: NumericPlotData,
    y: NumericPlotData,
    z: NumericPlotData,
    label: &str,
    appearance: &LineAppearance,
) -> crate::BuiltinResult<Line3Plot> {
    let mut plot = Line3Plot::from_numeric_data(x, y, z)
        .map_err(|e| plotting_error(builtin, format!("{builtin}: {e}")))?
        .with_style(
            appearance.color,
            appearance.line_width,
            appearance.line_style,
        )
        .with_label(label);
    if let Some(visible) = appearance.visible {
        plot.set_visible(visible);
    }
    Ok(plot)
}

fn build_line3_plot_from_numeric_data(
    x: NumericPlotData,
    y: NumericPlotData,
    z: NumericPlotData,
    label: &str,
    appearance: &LineAppearance,
) -> crate::BuiltinResult<Line3Plot> {
    build_line3_plot_from_numeric_data_for_builtin(BUILTIN_NAME, x, y, z, label, appearance)
}

async fn build_line3_gpu_plot_async(
    x: &runmat_accelerate_api::GpuTensorHandle,
    y: &runmat_accelerate_api::GpuTensorHandle,
    z: &runmat_accelerate_api::GpuTensorHandle,
    label: &str,
    appearance: &LineAppearance,
) -> crate::BuiltinResult<Option<Line3Plot>> {
    let count = super::gpu_helpers::GpuBufferExports::element_count;
    let len = count(x, BUILTIN_NAME)?;
    if len == 0 || len != count(y, BUILTIN_NAME)? || len != count(z, BUILTIN_NAME)? {
        return Err(plotting_error(
            BUILTIN_NAME,
            "plot3: X, Y, and Z inputs must have identical non-empty lengths",
        ));
    }
    if len == 1
        || runmat_accelerate_api::handle_precision(x) != runmat_accelerate_api::handle_precision(y)
        || runmat_accelerate_api::handle_precision(x) != runmat_accelerate_api::handle_precision(z)
    {
        return Ok(None);
    }
    u32::try_from(len)
        .map_err(|_| plotting_error(BUILTIN_NAME, "plot3: point count exceeds supported range"))?;
    let Some((_context, mut gpu_exports)) =
        super::gpu_helpers::GpuBufferExports::prepare(&[x, y, z], BUILTIN_NAME)?
    else {
        return Ok(None);
    };
    let result = async {
        let x_ref = gpu_exports
            .export(x, BUILTIN_NAME)?
            .ok_or_else(|| plotting_error(BUILTIN_NAME, "plot3: unable to export GPU X data"))?;
        let y_ref = gpu_exports
            .export(y, BUILTIN_NAME)?
            .ok_or_else(|| plotting_error(BUILTIN_NAME, "plot3: unable to export GPU Y data"))?;
        let z_ref = gpu_exports
            .export(z, BUILTIN_NAME)?
            .ok_or_else(|| plotting_error(BUILTIN_NAME, "plot3: unable to export GPU Z data"))?;
        if x_ref.len == 0 || x_ref.len != y_ref.len || x_ref.len != z_ref.len {
            return Err(plotting_error(
                BUILTIN_NAME,
                "plot3: X, Y, and Z inputs must have identical non-empty lengths",
            ));
        }
        if x_ref.precision != y_ref.precision || x_ref.precision != z_ref.precision {
            return Err(plotting_error(
                BUILTIN_NAME,
                "plot3: gpuArray precision must match across X, Y, and Z",
            ));
        }
        let len = u32::try_from(x_ref.len).map_err(|_| {
            plotting_error(BUILTIN_NAME, "plot3: point count exceeds supported range")
        })?;
        let inputs = Line3GpuInputs {
            x_buffer: x_ref.clone().into(),
            y_buffer: y_ref.clone().into(),
            z_buffer: z_ref.clone().into(),
            len,
            scalar: ScalarType::from_is_f64(x_ref.precision == ProviderPrecision::F64),
        };
        let bounds = gpu_xyz_bounds_async(x, y, z, BUILTIN_NAME).await?;
        gpu_exports.record_completed_effect();
        let mut plot = runmat_plot::plots::Line3Plot::from_gpu_xyz(
            inputs,
            appearance.color,
            appearance.line_width,
            appearance.line_style,
            bounds,
        )
        .with_label(label);
        if let Some(visible) = appearance.visible {
            plot.set_visible(visible);
        }
        Ok(plot)
    }
    .await;
    gpu_exports.complete(result, BUILTIN_NAME).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::PLOT3_DESCRIPTOR;
    async fn plot3_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
        super::plot3_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .await
    }
    use crate::builtins::plotting::state::current_axes_handle_for_figure;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, configure_subplot, current_figure_handle,
        reset_hold_state_for_run,
    };
    use runmat_plot::plots::PlotElement;
    use runmat_value::Tensor;

    fn vec_tensor(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![data.len()]).expect("plot3 test vector")
    }

    #[test]
    fn plot3_builds_line3_plot() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _ = futures::executor::block_on(plot3_builtin(vec![
            Value::Tensor(vec_tensor(&[0.0, 1.0])),
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[2.0, 3.0])),
        ]));
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert!(matches!(fig.plots().next().unwrap(), PlotElement::Line3(_)));
    }

    #[test]
    fn plot3_supports_multiple_series_and_style_tokens() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _ = futures::executor::block_on(plot3_builtin(vec![
            Value::Tensor(vec_tensor(&[0.0, 1.0])),
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[2.0, 3.0])),
            Value::String("r--".into()),
            Value::Tensor(vec_tensor(&[0.0, 1.0])),
            Value::Tensor(vec_tensor(&[2.0, 3.0])),
            Value::Tensor(vec_tensor(&[4.0, 5.0])),
            Value::String("DisplayName".into()),
            Value::String("Series B".into()),
        ]));
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.plots().count(), 2);
    }

    #[test]
    fn plot3_rejects_mismatched_lengths() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let err = futures::executor::block_on(plot3_builtin(vec![
            Value::Tensor(vec_tensor(&[0.0, 1.0])),
            Value::Tensor(vec_tensor(&[1.0])),
            Value::Tensor(vec_tensor(&[2.0, 3.0])),
        ]))
        .unwrap_err();
        assert!(
            err.to_string().contains("same number of elements")
                || err.to_string().contains("Data length mismatch")
        );
    }

    #[test]
    fn plot3_accepts_leading_axes_handle() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        configure_subplot(1, 2, 1).unwrap();
        let fig_handle = current_figure_handle();
        let ax = current_axes_handle_for_figure(fig_handle).unwrap();
        let _ = futures::executor::block_on(plot3_builtin(vec![
            Value::Num(ax),
            Value::Tensor(vec_tensor(&[0.0, 1.0])),
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[2.0, 3.0])),
        ]));
        let fig = clone_figure(fig_handle).unwrap();
        assert_eq!(fig.plot_axes_indices(), &[1]);
    }

    #[test]
    fn plot3_accepts_scalar_point() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _ = futures::executor::block_on(plot3_builtin(vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
        ]));
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Line3(line) = fig.plots().next().unwrap() else {
            panic!("expected line3")
        };
        let (x, y, z) = line.host_xyz_f64().unwrap().unwrap();
        assert_eq!(x, vec![1.0]);
        assert_eq!(y, vec![2.0]);
        assert_eq!(z, vec![3.0]);
        assert_eq!(line.line_style, LineStyle::Solid);
    }

    #[test]
    fn plot3_rejects_marker_linespec_until_line3_markers_exist() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let err = futures::executor::block_on(plot3_builtin(vec![
            Value::Tensor(vec_tensor(&[0.0, 1.0])),
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[2.0, 3.0])),
            Value::String("o".into()),
        ]))
        .expect_err("plot3 marker specs should fail until rendered");
        assert!(err.message.contains("marker LineSpec"));
    }

    #[test]
    fn plot3_applies_line_style_order_to_implicit_series_styles() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _ = futures::executor::block_on(plot3_builtin(vec![
            Value::Tensor(vec_tensor(&[0.0, 1.0])),
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[2.0, 3.0])),
            Value::String("LineStyleOrder".into()),
            Value::StringArray(
                runmat_value::StringArray::new(vec!["--".into(), ":".into()], vec![1, 2]).unwrap(),
            ),
            Value::Tensor(vec_tensor(&[0.0, 1.0])),
            Value::Tensor(vec_tensor(&[2.0, 3.0])),
            Value::Tensor(vec_tensor(&[4.0, 5.0])),
        ]))
        .expect("plot3 with LineStyleOrder");
        let fig = clone_figure(current_figure_handle()).unwrap();
        let mut plots = fig.plots();
        let PlotElement::Line3(first) = plots.next().unwrap() else {
            panic!("expected first line3")
        };
        let PlotElement::Line3(second) = plots.next().unwrap() else {
            panic!("expected second line3")
        };
        assert_eq!(first.line_style, LineStyle::Dashed);
        assert_eq!(second.line_style, LineStyle::Dotted);
    }

    #[test]
    fn plot3_descriptor_signatures_cover_supported_forms() {
        let labels: Vec<&str> = PLOT3_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = plot3(X, Y, Z)"));
        assert!(labels.contains(&"h = plot3(X, Y, Z, Name, Value, ...)"));
        assert!(labels.contains(&"h = plot3(ax, X, Y, Z)"));
        assert!(labels.contains(&"h = plot3(X1, Y1, Z1, ..., Xn, Yn, Zn)"));
    }

    #[test]
    fn plot3_missing_post_axes_input_uses_stable_identifier() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        configure_subplot(1, 2, 1).unwrap();
        let fig_handle = current_figure_handle();
        let ax = current_axes_handle_for_figure(fig_handle).unwrap();
        let err = futures::executor::block_on(plot3_builtin(vec![Value::Num(ax)]))
            .expect_err("missing post-axes data should fail");
        assert_eq!(err.identifier(), PLOT3_ERROR_INVALID_ARGUMENT.identifier);
    }
}
