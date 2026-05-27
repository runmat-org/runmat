//! MATLAB-compatible `contour3` builtin (3-D contour lines).

#[cfg(not(target_arch = "wasm32"))]
use log::warn;
use runmat_builtins::Value;
#[cfg(test)]
use runmat_builtins::{NumericDType, Tensor};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::ColorMap;
#[cfg(test)]
use runmat_plot::plots::PlotElement;

use super::common::tensor_to_surface_grid;
#[cfg(not(target_arch = "wasm32"))]
use super::contour::build_contour_gpu_plot_with_z_mode_async;
use super::contour::{
    build_contour_plot_with_z_mode, parse_contour_args, ContourArgs, ContourLineColor, ContourZMode,
};
use super::state::{render_active_plot, PlotRenderOptions};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

const BUILTIN_NAME: &str = "contour3";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::contour3")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "contour3",
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
    notes: "contour3 renders contour lines at their level height; gpuArray Z inputs may stay device-resident through the shared WGPU plotting path.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::contour3")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "contour3",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "contour3 performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "contour3",
    category = "plotting",
    summary = "Render MATLAB-compatible 3-D contour line plots.",
    keywords = "contour3,plotting,contour,3d,isolines",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::contour3"
)]
pub async fn contour3_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let mut raw_args = args.into_iter();
    let first = raw_args.next().ok_or_else(|| {
        crate::builtins::plotting::plotting_error(
            BUILTIN_NAME,
            format!("{BUILTIN_NAME}: expected at least one input"),
        )
    })?;
    let ContourArgs {
        name,
        x_axis,
        y_axis,
        z_input,
        level_spec,
        line_color,
        line_width,
    } = parse_contour_args(BUILTIN_NAME, first, raw_args.collect())?;
    let opts = PlotRenderOptions {
        title: "3-D Contour Plot",
        x_label: "X",
        y_label: "Y",
        axis_equal: true,
        ..Default::default()
    };
    let color_map = ColorMap::Parula;
    let base_z = 0.0;
    let contour = if matches!(line_color, ContourLineColor::None) {
        None
    } else {
        let mut contour = None;
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(handle) = z_input.gpu_handle().cloned() {
            match build_contour_gpu_plot_with_z_mode_async(
                name,
                &x_axis,
                &y_axis,
                &handle,
                color_map,
                base_z,
                &level_spec,
                &line_color,
                ContourZMode::Level,
            )
            .await
            {
                Ok(gpu_contour) => {
                    contour = Some(gpu_contour);
                }
                Err(err) => {
                    warn!("contour3 GPU path unavailable: {err}");
                }
            }
        }
        if contour.is_none() {
            let z_tensor = match z_input {
                super::common::SurfaceDataInput::Host(tensor) => tensor,
                super::common::SurfaceDataInput::Gpu(handle) => {
                    super::common::gather_tensor_from_gpu_async(handle, name).await?
                }
            };
            let grid = tensor_to_surface_grid(z_tensor, x_axis.len(), y_axis.len(), name)?;
            contour = Some(build_contour_plot_with_z_mode(
                name,
                &x_axis,
                &y_axis,
                &grid,
                color_map,
                base_z,
                &level_spec,
                &line_color,
                ContourZMode::Level,
            )?);
        }
        contour.map(|contour| contour.with_line_width(line_width))
    };
    let mut contour = contour;
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let Some(contour) = contour.take() else {
            return Ok(());
        };
        let plot_index = figure.add_contour_plot_on_axes(contour, axes);
        *plot_index_slot.borrow_mut() = Some((axes, plot_index));
        Ok(())
    });
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle =
        crate::builtins::plotting::state::register_contour_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(err);
    }
    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, current_figure_handle};
    use runmat_builtins::{ResolveContext, Type};

    fn tensor_from(data: &[f64], rows: usize, cols: usize) -> Tensor {
        Tensor {
            data: data.to_vec(),
            shape: vec![rows, cols],
            rows,
            cols,
            dtype: NumericDType::F64,
        }
    }

    #[test]
    fn contour3_returns_contour_handle_and_level_height_vertices() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let handle = futures::executor::block_on(contour3_builtin(vec![
            Value::Tensor(tensor_from(&[0.0, 1.0, 1.0, 0.0], 2, 2)),
            Value::Tensor(tensor_from(&[0.5, 0.5], 1, 2)),
            Value::String("k".into()),
        ]))
        .expect("contour3 should return handle");
        assert!(handle.is_finite());

        let fig = clone_figure(current_figure_handle()).expect("figure");
        let PlotElement::Contour(contour) = fig.plots().next().expect("contour") else {
            panic!("expected contour plot");
        };
        assert!(contour.force_3d);
        let vertices = contour.cpu_vertices().expect("cpu vertices");
        assert!(!vertices.is_empty());
        assert!(vertices
            .iter()
            .all(|vertex| (vertex.position[2] - 0.5).abs() < f32::EPSILON));
        assert!(vertices
            .iter()
            .all(|vertex| vertex.color == [0.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn contour3_accepts_explicit_axes_and_linewidth() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let handle = futures::executor::block_on(contour3_builtin(vec![
            Value::Tensor(tensor_from(&[0.0, 1.0], 2, 1)),
            Value::Tensor(tensor_from(&[0.0, 1.0], 2, 1)),
            Value::Tensor(tensor_from(&[0.0, 1.0, 1.0, 0.0], 2, 2)),
            Value::Tensor(tensor_from(&[1.0], 1, 1)),
            Value::String("k".into()),
            Value::String("LineWidth".into()),
            Value::Num(1.0),
        ]))
        .expect("contour3 should accept explicit axes and style");
        assert!(handle.is_finite());
    }

    #[test]
    fn contour3_type_is_numeric_handle() {
        assert_eq!(
            handle_scalar_type(&[Type::tensor()], &ResolveContext::new(Vec::new())),
            Type::Num
        );
    }
}
