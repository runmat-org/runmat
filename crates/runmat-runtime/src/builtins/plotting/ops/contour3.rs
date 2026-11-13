//! MATLAB-compatible `contour3` builtin (3-D contour lines).

use runmat_builtins::{
    BuiltinErrorDescriptor, CONTOUR3_ERROR_INTERNAL, CONTOUR3_ERROR_INVALID_ARGUMENT,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::ColorMap;
#[cfg(test)]
use runmat_plot::plots::PlotElement;

use super::common::tensor_to_surface_grid_matlab_xy;
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
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::contour3"
)]
pub async fn contour3_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 1) {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:plotting:TooManyOutputs",
            "contour3: expected at most one output",
        ));
    }
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "contour3")?;
    let _arguments = args.fork();
    work.finish(
        contour3_impl(args, &work)
            .await
            .map_err(|cause| contour3_error(cause, &CONTOUR3_ERROR_INVALID_ARGUMENT)),
    )
}

fn contour3_error(
    cause: crate::RuntimeError,
    descriptor: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    if cause.identifier().is_some() {
        return cause;
    }
    crate::build_runtime_error(format!("{}: {}", descriptor.message, cause.message))
        .with_builtin(BUILTIN_NAME)
        .with_identifier(
            descriptor
                .identifier
                .expect("contour3 diagnostic identifier"),
        )
        .with_source(cause)
        .build()
}

async fn contour3_impl(
    mut args: crate::call::BuiltinArguments,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<f64> {
    let (axes_target, retained) = if args
        .first()
        .is_some_and(super::contour::is_typed_integer_value)
    {
        (None, args)
    } else {
        super::op_common::split_leading_axes_arguments(args, BUILTIN_NAME)?
    };
    args = retained;
    let first = args.pop_front().ok_or_else(|| {
        crate::builtins::plotting::plotting_error(BUILTIN_NAME, "contour3: expected Z input")
    })?;
    let ContourArgs {
        name,
        x_axis,
        y_axis,
        z_input,
        level_spec,
        line_color,
        line_width,
    } = parse_contour_args(BUILTIN_NAME, first, args.view())?;
    let opts = PlotRenderOptions {
        // MATLAB leaves the data aspect free; `axis equal` sets it.
        axis_equal: false,
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
            contour = build_contour_gpu_plot_with_z_mode_async(
                name,
                &x_axis,
                &y_axis,
                &handle,
                color_map.clone(),
                base_z,
                &level_spec,
                &line_color,
                ContourZMode::Level,
            )
            .await?;
        }
        if contour.is_none() {
            let z_tensor = match z_input {
                super::common::SurfaceDataInput::Host(tensor) => tensor,
                super::common::SurfaceDataInput::Gpu(handle) => {
                    super::common::gather_tensor_from_gpu_async(handle, name).await?
                }
            };
            let grid =
                tensor_to_surface_grid_matlab_xy(z_tensor, y_axis.len(), x_axis.len(), name)?;
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
    work.check()?;
    super::op_common::apply_axes_target(axes_target, BUILTIN_NAME)?;
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    work.check()?;
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
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(handle);
        }
        return Err(contour3_error(err, &CONTOUR3_ERROR_INTERNAL));
    }
    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    async fn contour3_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
        super::contour3_builtin(
            crate::context::legacy::active(),
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .await
    }

    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, current_figure_handle};
    use runmat_value::{Tensor, Value};

    fn tensor_from(data: &[f64], rows: usize, cols: usize) -> Tensor {
        Tensor::new(data.to_vec(), vec![rows, cols]).expect("contour3 test tensor")
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
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(1.0),
        )
        .unwrap();
        let handle = futures::executor::block_on(contour3_builtin(vec![
            Value::Num(ax),
            Value::Tensor(tensor_from(&[0.0, 1.0, 1.0, 0.0], 2, 2)),
            Value::Tensor(tensor_from(&[0.5, 0.5], 1, 2)),
        ]))
        .expect("leading axes uses shared contour grammar");
        assert_eq!(
            crate::builtins::plotting::get::get_builtin(vec![
                Value::Num(handle),
                Value::from("Parent")
            ])
            .unwrap(),
            Value::Num(ax)
        );
    }

    #[test]
    fn contour3_catalog_infers_host_double_handle() {
        use runmat_types::*;
        let numeric = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
        let input = ValueFact::proven(
            numeric.clone(),
            ShapeFact::from(vec![Some(3), Some(2)]),
            StorageFact::Dense,
        );
        let request = CallRequest {
            arguments: vec![input; 1],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let result =
            runmat_builtins::infer_catalog_call(&runmat_builtins::CONTOUR3_CATALOG_ENTRY, &request);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs.len(), 1);
        let output = &result.outputs[0];
        assert_eq!(output.kind, numeric);
        assert_eq!(output.shape, ShapeFact::Scalar);
        assert_eq!(output.storage, StorageFact::Dense);
        assert_eq!(output.residency, ResidencyFact::Host);
    }
}
