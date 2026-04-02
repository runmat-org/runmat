use runmat_accelerate_api::ProviderPrecision;
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_plot::gpu::{line3::Line3GpuInputs, ScalarType};
use runmat_plot::plots::{Line3Plot, LineStyle};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

use super::common::numeric_triplet;
use super::gpu_helpers::gpu_xyz_bounds_async;
use super::op_common::line_inputs::NumericInput;
use super::op_common::{apply_axes_target, split_leading_axes_handle};
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_line_style_args, LineAppearance, LineStyleParseOptions};

const BUILTIN_NAME: &str = "plot3";

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
    category = "plotting",
    summary = "Create MATLAB-compatible 3-D line plots.",
    keywords = "plot3,line,3d,visualization",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::plot3"
)]
pub async fn plot3_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (axes_target, args) = split_leading_axes_handle(args, BUILTIN_NAME)?;
    apply_axes_target(axes_target, BUILTIN_NAME)?;
    let (mut plans, _line_style_order) = parse_plot3_series_specs(args)?;

    let opts = PlotRenderOptions {
        title: "3-D Plot",
        x_label: "X",
        y_label: "Y",
        axis_equal: true,
        ..Default::default()
    };

    let mut plots = Vec::with_capacity(plans.len());
    for (series_idx, mut plan) in plans.drain(..).enumerate() {
        if !plan.line_style_explicit {
            plan.appearance.line_style = match series_idx % 4 {
                0 => LineStyle::Solid,
                1 => LineStyle::Dashed,
                2 => LineStyle::Dotted,
                _ => LineStyle::DashDot,
            };
        }
        let label = plan
            .label
            .take()
            .unwrap_or_else(|| format!("Series {}", series_idx + 1));
        if let Some((xg, yg, zg)) = plan.data.gpu_handles() {
            if let Ok(plot) = build_line3_gpu_plot_async(xg, yg, zg, &label, &plan.appearance).await
            {
                plots.push(plot);
                continue;
            }
        }
        let (x, y, z) = plan.data.into_tensors_async(BUILTIN_NAME).await?;
        let (x, y, z) = numeric_triplet(x, y, z, BUILTIN_NAME)?;
        plots.push(build_line3_plot(x, y, z, &label, &plan.appearance)?);
    }
    let mut plots_opt = Some(plots);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
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
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(err);
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
        runmat_builtins::Tensor,
        runmat_builtins::Tensor,
        runmat_builtins::Tensor,
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
    args: Vec<Value>,
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
            if let Some(text) = super::style::value_as_string(&token) {
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

fn build_line3_plot(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    label: &str,
    appearance: &LineAppearance,
) -> crate::BuiltinResult<Line3Plot> {
    Ok(Line3Plot::new(x, y, z)
        .map_err(|e| plotting_error(BUILTIN_NAME, format!("plot3: {e}")))?
        .with_style(
            appearance.color,
            appearance.line_width,
            appearance.line_style,
        )
        .with_label(label))
}

async fn build_line3_gpu_plot_async(
    x: &runmat_accelerate_api::GpuTensorHandle,
    y: &runmat_accelerate_api::GpuTensorHandle,
    z: &runmat_accelerate_api::GpuTensorHandle,
    label: &str,
    appearance: &LineAppearance,
) -> crate::BuiltinResult<Line3Plot> {
    let context = crate::builtins::plotting::gpu_helpers::ensure_shared_wgpu_context(BUILTIN_NAME)?;
    let x_ref = runmat_accelerate_api::export_wgpu_buffer(x)
        .ok_or_else(|| plotting_error(BUILTIN_NAME, "plot3: unable to export GPU X data"))?;
    let y_ref = runmat_accelerate_api::export_wgpu_buffer(y)
        .ok_or_else(|| plotting_error(BUILTIN_NAME, "plot3: unable to export GPU Y data"))?;
    let z_ref = runmat_accelerate_api::export_wgpu_buffer(z)
        .ok_or_else(|| plotting_error(BUILTIN_NAME, "plot3: unable to export GPU Z data"))?;
    if x_ref.len < 2 || x_ref.len != y_ref.len || x_ref.len != z_ref.len {
        return Err(plotting_error(
            BUILTIN_NAME,
            "plot3: X, Y, and Z inputs must have identical lengths of at least 2",
        ));
    }
    if x_ref.precision != y_ref.precision || x_ref.precision != z_ref.precision {
        return Err(plotting_error(
            BUILTIN_NAME,
            "plot3: gpuArray precision must match across X, Y, and Z",
        ));
    }
    let inputs = Line3GpuInputs {
        x_buffer: x_ref.buffer.clone(),
        y_buffer: y_ref.buffer.clone(),
        z_buffer: z_ref.buffer.clone(),
        len: x_ref.len as u32,
        scalar: ScalarType::from_is_f64(x_ref.precision == ProviderPrecision::F64),
    };
    let buffer = runmat_plot::gpu::line3::pack_vertices_from_xyz(
        &context.device,
        &context.queue,
        &inputs,
        &runmat_plot::gpu::line3::Line3GpuParams {
            color: appearance.color,
            half_width_data: appearance.line_width.max(1.0) * 0.01,
            thick: appearance.line_width > 1.0,
            line_style: appearance.line_style,
        },
    )
    .map_err(|e| plotting_error(BUILTIN_NAME, format!("plot3: {e}")))?;
    let bounds = gpu_xyz_bounds_async(x, y, z, BUILTIN_NAME).await?;
    Ok(Line3Plot::from_gpu_buffer(
        buffer,
        ((x_ref.len - 1) * if appearance.line_width > 1.0 { 6 } else { 2 }) as usize,
        appearance.color,
        appearance.line_width,
        appearance.line_style,
        bounds,
    )
    .with_label(label))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::state::current_axes_handle_for_figure;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, configure_subplot, current_figure_handle,
        reset_hold_state_for_run,
    };
    use runmat_builtins::{NumericDType, Tensor};
    use runmat_plot::plots::PlotElement;

    fn vec_tensor(data: &[f64]) -> Tensor {
        Tensor {
            data: data.to_vec(),
            shape: vec![data.len()],
            rows: data.len(),
            cols: 1,
            dtype: NumericDType::F64,
        }
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
            Value::String("o".into()),
        ]));
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Line3(line) = fig.plots().next().unwrap() else {
            panic!("expected line3")
        };
        assert_eq!(line.x_data, vec![1.0]);
        assert_eq!(line.y_data, vec![2.0]);
        assert_eq!(line.z_data, vec![3.0]);
    }
}
