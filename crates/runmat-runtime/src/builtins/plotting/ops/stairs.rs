//! MATLAB-compatible `stairs` builtin.

use log::warn;
use runmat_accelerate_api::{self, GpuTensorHandle, ProviderPrecision};
use runmat_builtins::Tensor;
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_plot::gpu::line::{
    self, LineGpuInputs as MarkerGpuInputs, LineGpuParams as MarkerGpuParams,
};
use runmat_plot::gpu::stairs::{StairsGpuInputs, StairsGpuParams};
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::{LineMarkerAppearance, LineStyle, StairsPlot};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

use super::common::numeric_pair;
use super::gpu_helpers::gpu_xy_bounds;
use super::op_common::line_inputs::NumericInput as StairsInput;
use super::op_common::{apply_axes_target, split_leading_axes_handle};
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{
    marker_metadata_from_appearance, parse_line_style_args, LineAppearance, LineStyleParseOptions,
    DEFAULT_LINE_MARKER_SIZE,
};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use std::convert::TryFrom;

use crate::BuiltinResult;
const BUILTIN_NAME: &str = "stairs";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::stairs")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "stairs",
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
    notes: "Stairs plots terminate fusion graphs; gpuArray inputs may remain on device when shared plotting context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::stairs")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "stairs",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "stairs performs I/O and therefore terminates fusion graphs.",
};

#[runtime_builtin(
    name = "stairs",
    category = "plotting",
    summary = "Render MATLAB-compatible stairs plots.",
    keywords = "stairs,plotting,step",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::stairs"
)]
pub fn stairs_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (axes_target, mut args) = split_leading_axes_handle(args, BUILTIN_NAME)?;
    apply_axes_target(axes_target, BUILTIN_NAME)?;

    let args_len = args.len();
    let (x, y, rest) = match args_len {
        0 => {
            return Err(plotting_error(
                BUILTIN_NAME,
                "stairs: expected Y data or X/Y data after optional axes handle",
            ));
        }
        1 => {
            let y = args.pop().expect("one arg");
            let x = infer_stairs_x_from_y(&y)?;
            (x, y, Vec::new())
        }
        _ => {
            let mut iter = args.into_iter();
            let x = iter.next().expect("x");
            let y = iter.next().expect("y");
            let rest = iter.collect();
            (x, y, rest)
        }
    };

    let parsed_style = parse_line_style_args(&rest, &LineStyleParseOptions::stairs())?;
    let mut x_input = Some(StairsInput::from_value(x, BUILTIN_NAME)?);
    let mut y_input = Some(StairsInput::from_value(y, BUILTIN_NAME)?);
    let opts = PlotRenderOptions {
        title: "Stairs",
        x_label: "X",
        y_label: "Y",
        ..Default::default()
    };
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let appearance = parsed_style.appearance.clone();
        let marker_meta = marker_metadata_from_appearance(&appearance);
        let label = parsed_style
            .label
            .clone()
            .unwrap_or_else(|| "Data".to_string());
        let x_arg = x_input.take().expect("stairs x consumed once");
        let y_arg = y_input.take().expect("stairs y consumed once");

        if let (Some(x_gpu), Some(y_gpu)) = (x_arg.gpu_handle(), y_arg.gpu_handle()) {
            match build_stairs_gpu_plot(
                BUILTIN_NAME,
                x_gpu,
                y_gpu,
                &appearance,
                marker_meta.clone(),
                &label,
            ) {
                Ok(plot) => {
                    let plot_index = figure.add_stairs_plot_on_axes(plot, axes);
                    *plot_index_slot.borrow_mut() = Some((axes, plot_index));
                    return Ok(());
                }
                Err(err) => {
                    warn!("stairs GPU path unavailable: {err}");
                }
            }
        }

        let (x_tensor, y_tensor) = (x_arg.into_tensor("stairs")?, y_arg.into_tensor("stairs")?);
        let (x_vals, y_vals) = numeric_pair(x_tensor, y_tensor, "stairs")?;
        let plot = build_stairs_plot(x_vals, y_vals, &appearance, marker_meta, &label)?;
        let plot_index = figure.add_stairs_plot_on_axes(plot, axes);
        *plot_index_slot.borrow_mut() = Some((axes, plot_index));
        Ok(())
    });
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle =
        crate::builtins::plotting::state::register_stairs_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(err);
    }
    Ok(handle)
}

fn build_stairs_plot(
    x: Vec<f64>,
    y: Vec<f64>,
    appearance: &LineAppearance,
    marker_meta: Option<LineMarkerAppearance>,
    label: &str,
) -> BuiltinResult<StairsPlot> {
    if x.len() != y.len() {
        return Err(plotting_error(
            BUILTIN_NAME,
            "stairs: X and Y inputs must share the same length",
        ));
    }
    if x.len() < 2 {
        return Err(plotting_error(
            BUILTIN_NAME,
            "stairs: inputs must contain at least two elements",
        ));
    }
    let mut plot = StairsPlot::new(x, y)
        .map_err(|e| plotting_error(BUILTIN_NAME, format!("stairs: {e}")))?
        .with_style(appearance.color, appearance.line_width)
        .with_label(label);
    apply_stairs_marker_metadata(&mut plot, marker_meta);
    Ok(plot)
}

fn infer_stairs_x_from_y(y: &Value) -> BuiltinResult<Value> {
    let len = match y {
        Value::GpuTensor(handle) => handle.shape.iter().copied().product::<usize>().max(1),
        other => {
            let tensor = Tensor::try_from(other)
                .map_err(|e| plotting_error(BUILTIN_NAME, format!("stairs: {e}")))?;
            tensor.data.len().max(1)
        }
    };
    let data = (1..=len).map(|i| i as f64).collect::<Vec<_>>();
    Ok(Value::Tensor(Tensor {
        data,
        shape: vec![len],
        rows: len,
        cols: 1,
        dtype: runmat_builtins::NumericDType::F64,
    }))
}

fn build_stairs_gpu_plot(
    name: &'static str,
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
    appearance: &LineAppearance,
    marker_meta: Option<LineMarkerAppearance>,
    label: &str,
) -> BuiltinResult<StairsPlot> {
    let context = super::gpu_helpers::ensure_shared_wgpu_context(name)?;

    let x_ref = runmat_accelerate_api::export_wgpu_buffer(x)
        .ok_or_else(|| plotting_error(name, format!("{name}: unable to export GPU X data")))?;
    let y_ref = runmat_accelerate_api::export_wgpu_buffer(y)
        .ok_or_else(|| plotting_error(name, format!("{name}: unable to export GPU Y data")))?;

    if x_ref.len < 2 {
        return Err(plotting_error(
            name,
            format!("{name}: inputs must contain at least two elements"),
        ));
    }
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

    let len_u32 = u32::try_from(x_ref.len).map_err(|_| {
        plotting_error(name, format!("{name}: point count exceeds supported range"))
    })?;
    let scalar = ScalarType::from_is_f64(x_ref.precision == ProviderPrecision::F64);

    let inputs = StairsGpuInputs {
        x_buffer: x_ref.buffer.clone(),
        y_buffer: y_ref.buffer.clone(),
        len: len_u32,
        scalar,
    };
    let params = StairsGpuParams {
        color: appearance.color,
    };

    let gpu_vertices = runmat_plot::gpu::stairs::pack_vertices_from_xy(
        &context.device,
        &context.queue,
        &inputs,
        &params,
    )
    .map_err(|e| plotting_error(name, format!("{name}: failed to build GPU vertices: {e}")))?;

    let marker_gpu = if let Some(marker) = marker_meta.clone() {
        let marker_inputs = MarkerGpuInputs {
            x_buffer: x_ref.buffer.clone(),
            y_buffer: y_ref.buffer.clone(),
            len: len_u32,
            scalar,
        };
        let marker_params = MarkerGpuParams {
            color: marker.face_color,
            half_width_data: 0.0,
            thick: false,
            line_style: LineStyle::Solid,
            marker_size: marker.size.max(DEFAULT_LINE_MARKER_SIZE),
        };
        Some(
            line::pack_marker_vertices_from_xy(
                &context.device,
                &context.queue,
                &marker_inputs,
                &marker_params,
            )
            .map_err(|e| {
                plotting_error(
                    name,
                    format!("{name}: failed to build marker vertices: {e}"),
                )
            })?,
        )
    } else {
        None
    };

    let bounds = gpu_xy_bounds(x, y, "stairs")?;
    let vertex_count = (x_ref.len - 1) * 4;
    let mut plot =
        StairsPlot::from_gpu_buffer(appearance.color, gpu_vertices, vertex_count, bounds)
            .with_style(appearance.color, appearance.line_width)
            .with_label(label);
    apply_stairs_marker_metadata(&mut plot, marker_meta);
    plot.set_marker_gpu_vertices(marker_gpu);
    Ok(plot)
}

fn apply_stairs_marker_metadata(plot: &mut StairsPlot, marker_meta: Option<LineMarkerAppearance>) {
    if let Some(marker) = marker_meta {
        plot.set_marker(Some(marker));
    } else {
        plot.set_marker(None);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::plotting::state::current_axes_handle_for_figure;
    use crate::builtins::plotting::tests::ensure_plot_test_env;
    use crate::builtins::plotting::{
        clear_figure, clone_figure, configure_subplot, current_figure_handle,
        reset_hold_state_for_run,
    };
    use runmat_builtins::{ResolveContext, Type};

    fn setup_plot_tests() {
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn stairs_requires_matching_lengths() {
        setup_plot_tests();
        let res = stairs_builtin(vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0])),
        ]);
        assert!(res.is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn stairs_requires_minimum_length() {
        setup_plot_tests();
        let res = stairs_builtin(vec![
            Value::Tensor(tensor_from(&[0.0])),
            Value::Tensor(tensor_from(&[1.0])),
        ]);
        assert!(res.is_err());
    }

    #[test]
    fn stairs_type_is_numeric_handle() {
        assert_eq!(
            handle_scalar_type(
                &[Type::tensor(), Type::tensor()],
                &ResolveContext::new(Vec::new())
            ),
            Type::Num
        );
    }

    #[test]
    fn stairs_accepts_leading_axes_handle() {
        setup_plot_tests();
        configure_subplot(1, 2, 1).unwrap();
        let fig_handle = current_figure_handle();
        let ax = current_axes_handle_for_figure(fig_handle).unwrap();
        let _ = stairs_builtin(vec![
            Value::Num(ax),
            Value::Tensor(tensor_from(&[0.0, 1.0, 2.0])),
            Value::Tensor(tensor_from(&[1.0, 2.0, 1.5])),
        ]);
        let fig = clone_figure(fig_handle).unwrap();
        assert_eq!(fig.plot_axes_indices(), &[1]);
    }

    #[test]
    fn stairs_ax_y_shorthand_infers_one_based_x_on_target_axes() {
        setup_plot_tests();
        configure_subplot(1, 2, 1).unwrap();
        let fig_handle = current_figure_handle();
        let ax = current_axes_handle_for_figure(fig_handle).unwrap();
        let _ = stairs_builtin(vec![
            Value::Num(ax),
            Value::Tensor(tensor_from(&[2.0, 4.0, 3.0])),
        ])
        .unwrap();
        let fig = clone_figure(fig_handle).unwrap();
        assert_eq!(fig.plot_axes_indices(), &[1]);
        let runmat_plot::plots::PlotElement::Stairs(plot) = fig.plots().next().unwrap() else {
            panic!("expected stairs plot")
        };
        assert_eq!(plot.x, vec![1.0, 2.0, 3.0]);
        assert_eq!(plot.y, vec![2.0, 4.0, 3.0]);
    }

    #[test]
    fn stairs_y_shorthand_infers_one_based_x() {
        setup_plot_tests();
        let _ = stairs_builtin(vec![Value::Tensor(tensor_from(&[2.0, 4.0, 3.0]))]).unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let plot = fig.plots().next().unwrap();
        let runmat_plot::plots::PlotElement::Stairs(plot) = plot else {
            panic!("expected stairs plot")
        };
        assert_eq!(plot.x, vec![1.0, 2.0, 3.0]);
        assert_eq!(plot.y, vec![2.0, 4.0, 3.0]);
    }
}
