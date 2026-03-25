use std::sync::Arc;

use log::warn;
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode};

use super::common::{tensor_to_surface_grid, SurfaceDataInput};
use super::op_common::surface_inputs::{
    axis_sources_from_xy_values, axis_sources_to_host, parse_surface_call_args,
};
use super::state::{color_limits_snapshot, render_active_plot, PlotRenderOptions};
use super::style::{parse_surface_style_args, SurfaceStyleDefaults};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

const BUILTIN_NAME: &str = "imagesc";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::imagesc")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "imagesc",
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
    notes: "imagesc is a plotting sink; GPU inputs may remain on device when a shared WGPU context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::imagesc")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "imagesc",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "imagesc terminates fusion graphs and performs rendering.",
};

#[runtime_builtin(
    name = "imagesc",
    category = "plotting",
    summary = "Render a MATLAB-compatible scaled image plot.",
    keywords = "imagesc,plotting,image,colormap",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::imagesc"
)]
pub async fn imagesc_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (x, y, c, rest) = parse_surface_call_args(args, BUILTIN_NAME)?;
    let c_input = SurfaceDataInput::from_value(c, BUILTIN_NAME)?;
    let (rows, cols) = c_input.grid_shape(BUILTIN_NAME)?;
    let (x_axis, y_axis) = axis_sources_from_xy_values(x, y, rows, cols, BUILTIN_NAME).await?;

    let defaults =
        SurfaceStyleDefaults::new(ColorMap::Parula, ShadingMode::None, false, 1.0, true, false);
    let style = Arc::new(parse_surface_style_args(BUILTIN_NAME, &rest, defaults)?);
    let color_limits = color_limits_snapshot();

    let mut surface = if let Some(c_gpu) = c_input.gpu_handle().cloned() {
        match super::gpu_helpers::axis_bounds_async(&c_gpu, BUILTIN_NAME).await {
            Ok((min_z, max_z)) => match super::surf::build_surface_gpu_plot_with_bounds_async(
                BUILTIN_NAME,
                &x_axis,
                &y_axis,
                &c_gpu,
                min_z,
                max_z,
                style.colormap,
                style.alpha,
                true,
            )
            .await
            {
                Ok(surface) => surface,
                Err(err) => {
                    warn!("imagesc GPU path unavailable: {err}");
                    let (x_host, y_host) =
                        axis_sources_to_host(&x_axis, &y_axis, BUILTIN_NAME).await?;
                    build_imagesc_cpu(&c_input, x_host, y_host, style.colormap, color_limits)
                        .await?
                }
            },
            Err(err) => {
                warn!("imagesc GPU bounds unavailable: {err}");
                let (x_host, y_host) = axis_sources_to_host(&x_axis, &y_axis, BUILTIN_NAME).await?;
                build_imagesc_cpu(&c_input, x_host, y_host, style.colormap, color_limits).await?
            }
        }
    } else {
        let (x_host, y_host) = axis_sources_to_host(&x_axis, &y_axis, BUILTIN_NAME).await?;
        build_imagesc_cpu(&c_input, x_host, y_host, style.colormap, color_limits).await?
    };

    surface = surface.with_flatten_z(true).with_image_mode(true);
    if color_limits.is_some() {
        surface = surface.with_color_limits(color_limits);
    }
    surface.colormap = style.colormap;
    let mut surface = Some(surface);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let opts = PlotRenderOptions {
        title: "Image",
        x_label: "X",
        y_label: "Y",
        axis_equal: true,
        ..Default::default()
    };
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let surface = surface.take().expect("imagesc plot consumed once");
        let plot_index = figure.add_surface_plot_on_axes(surface, axes);
        *plot_index_slot.borrow_mut() = Some((axes, plot_index));
        Ok(())
    });
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = crate::builtins::plotting::state::register_image_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(err);
    }
    Ok(handle)
}

async fn build_imagesc_cpu(
    c_input: &SurfaceDataInput,
    x_axis: Vec<f64>,
    y_axis: Vec<f64>,
    colormap: ColorMap,
    color_limits: Option<(f64, f64)>,
) -> crate::BuiltinResult<runmat_plot::plots::SurfacePlot> {
    let tensor = match c_input.clone() {
        SurfaceDataInput::Host(tensor) => tensor,
        SurfaceDataInput::Gpu(handle) => {
            super::common::gather_tensor_from_gpu_async(handle, BUILTIN_NAME).await?
        }
    };
    let grid = tensor_to_surface_grid(tensor, x_axis.len(), y_axis.len(), BUILTIN_NAME)?;
    let surface = super::surf::build_surface(x_axis, y_axis, grid)?
        .with_flatten_z(true)
        .with_image_mode(true)
        .with_colormap(colormap)
        .with_shading(ShadingMode::None)
        .with_color_limits(color_limits);
    Ok(surface)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_builtins::{NumericDType, Tensor};
    use runmat_plot::plots::PlotElement;

    fn grid_tensor(data: Vec<f64>, rows: usize, cols: usize) -> Tensor {
        Tensor {
            data,
            shape: vec![rows, cols],
            rows,
            cols,
            dtype: NumericDType::F64,
        }
    }

    #[test]
    fn imagesc_z_only_shorthand_builds_flattened_surface() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = futures::executor::block_on(imagesc_builtin(vec![Value::Tensor(grid_tensor(
            vec![1.0, 2.0, 3.0, 4.0],
            2,
            2,
        ))]));
        let fig = clone_figure(current_figure_handle()).unwrap();
        let plot = fig.plots().next().unwrap();
        let PlotElement::Surface(surface) = plot else {
            panic!("expected surface");
        };
        assert!(surface.flatten_z);
        assert!(surface.image_mode);
        assert_eq!(surface.x_data, vec![1.0, 2.0]);
        assert_eq!(surface.y_data, vec![1.0, 2.0]);
    }

    #[test]
    fn imagesc_applies_explicit_axes_and_color_limits() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        crate::builtins::plotting::state::set_color_limits_runtime(Some((0.0, 10.0)));

        let _ = futures::executor::block_on(imagesc_builtin(vec![
            Value::Tensor(Tensor {
                data: vec![10.0, 20.0],
                shape: vec![2],
                rows: 2,
                cols: 1,
                dtype: NumericDType::F64,
            }),
            Value::Tensor(Tensor {
                data: vec![1.0, 2.0],
                shape: vec![2],
                rows: 2,
                cols: 1,
                dtype: NumericDType::F64,
            }),
            Value::Tensor(grid_tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2)),
        ]));
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface");
        };
        assert_eq!(surface.x_data, vec![10.0, 20.0]);
        assert_eq!(surface.y_data, vec![1.0, 2.0]);
        assert_eq!(surface.color_limits, Some((0.0, 10.0)));
    }
}
