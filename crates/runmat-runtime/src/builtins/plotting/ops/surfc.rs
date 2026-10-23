//! MATLAB-compatible `surfc` builtin (surface with contour).
use runmat_builtins::BuiltinErrorDescriptor;
pub use runmat_builtins::{SURFC_ERROR_INTERNAL, SURFC_ERROR_INVALID_ARGUMENT};

use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode};
#[cfg(test)]
use runmat_value::{IntegerStorage, Tensor};

use super::common::{tensor_to_surface_grid_matlab_xy, SurfaceDataInput};
use super::contour::{build_contour_plot, default_level_count, ContourLevelSpec, ContourLineColor};
use super::mesh::{
    numeric_plot_data_from_tensor, numeric_plot_data_from_value, retain_surface_source_data,
};
use super::op_common::surface_composite::contour_for_surface_axes_input;
use super::op_common::surface_inputs::{
    axis_sources_to_host, parse_surface_call_args_matlab_xy, surface_axis_sources_from_xy_values,
};
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_surface_style_args, SurfaceStyleDefaults};
use super::surf::{build_surface, build_surface_gpu_plot_with_bounds_async};
use crate::build_runtime_error;
use crate::RuntimeError;
use std::sync::Arc;

const BUILTIN_NAME: &str = "surfc";

fn surfc_error_with_detail(
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

fn map_surfc_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    let mapped = surfc_error_with_detail(&SURFC_ERROR_INVALID_ARGUMENT, err.message.clone());
    let mut builder = crate::build_runtime_error(mapped.message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = mapped.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_surfc_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    let mapped = surfc_error_with_detail(&SURFC_ERROR_INTERNAL, err.message.clone());
    let mut builder = crate::build_runtime_error(mapped.message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = mapped.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "surfc",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::surfc"
)]
pub async fn surfc_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 1) {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:plotting:TooManyOutputs",
            "surfc: expected at most one output",
        ));
    }
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "surfc")?;
    let _arguments = args.fork();
    work.finish(surfc_impl(args, &work).await)
}

async fn surfc_impl(
    args: crate::call::BuiltinArguments,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<f64> {
    let (x, y, z, rest) = parse_surface_call_args_matlab_xy(args, BUILTIN_NAME)
        .map_err(map_surfc_invalid_argument)?;
    let source_x = numeric_plot_data_from_value(&x, BUILTIN_NAME)
        .await
        .map_err(map_surfc_invalid_argument)?;
    let source_y = numeric_plot_data_from_value(&y, BUILTIN_NAME)
        .await
        .map_err(map_surfc_invalid_argument)?;
    let z_input = SurfaceDataInput::from_value(z, "surfc").map_err(map_surfc_invalid_argument)?;
    let source_z = match &z_input {
        SurfaceDataInput::Host(tensor) => Some(numeric_plot_data_from_tensor(tensor)),
        SurfaceDataInput::Gpu(handle) => Some(numeric_plot_data_from_tensor(
            &super::common::gather_tensor_from_gpu_async(handle.clone(), BUILTIN_NAME)
                .await
                .map_err(map_surfc_invalid_argument)?,
        )),
    };
    let (rows, cols) = z_input
        .grid_shape(BUILTIN_NAME)
        .map_err(map_surfc_invalid_argument)?;
    let (x_axis, y_axis) = surface_axis_sources_from_xy_values(x, y, rows, cols, BUILTIN_NAME)
        .await
        .map_err(map_surfc_invalid_argument)?;
    let style = Arc::new(
        parse_surface_style_args(
            "surfc",
            rest.view(),
            SurfaceStyleDefaults::new(
                ColorMap::Parula,
                ShadingMode::Smooth,
                false,
                1.0,
                false,
                true,
            ),
        )
        .map_err(map_surfc_invalid_argument)?,
    );
    let opts = PlotRenderOptions {
        axis_equal: false,
        ..Default::default()
    };
    let level_spec = ContourLevelSpec::Count(default_level_count());
    // Build plots up-front so we can await GPU work without blocking the render loop.
    let contour_map = style.colormap.clone();
    let (mut surface, contour) = if let Some(z_gpu) = z_input.gpu_handle().cloned() {
        match super::op_common::surface_inputs::PreparedSurfaceGpu::prepare(
            BUILTIN_NAME,
            &x_axis,
            &y_axis,
            &z_gpu,
        )
        .await?
        {
            Some(mut prepared) => {
                let (min_z, max_z) = prepared.z_bounds;
                let mut surface = build_surface_gpu_plot_with_bounds_async(
                    BUILTIN_NAME,
                    &x_axis,
                    &y_axis,
                    &z_gpu,
                    &mut prepared,
                    min_z,
                    max_z,
                    style.colormap.clone(),
                    style.alpha,
                    style.flatten_z,
                )
                .await?;
                style.apply_to_plot(&mut surface);
                let base_z = surface.bounds().min.z;
                let contour = contour_for_surface_axes_input(
                    BUILTIN_NAME,
                    &x_axis,
                    &y_axis,
                    &z_gpu,
                    &mut prepared,
                    contour_map,
                    base_z,
                    &level_spec,
                )?;
                (surface, contour)
            }
            None => {
                let (x_host, y_host) = axis_sources_to_host(&x_axis, &y_axis, BUILTIN_NAME)
                    .await
                    .map_err(map_surfc_invalid_argument)?;
                let z_tensor = super::common::gather_tensor_from_gpu_async(z_gpu, BUILTIN_NAME)
                    .await
                    .map_err(map_surfc_invalid_argument)?;
                let grid = tensor_to_surface_grid_matlab_xy(z_tensor, rows, cols, BUILTIN_NAME)
                    .map_err(map_surfc_invalid_argument)?;
                let mut surface = build_surface(x_host.clone(), y_host.clone(), grid.clone())
                    .map_err(map_surfc_invalid_argument)?;
                style.apply_to_plot(&mut surface);
                let base_z = surface.bounds().min.z;
                let contour = build_contour_plot(
                    BUILTIN_NAME,
                    &x_host,
                    &y_host,
                    &grid,
                    contour_map,
                    base_z,
                    &level_spec,
                    &ContourLineColor::Auto,
                )
                .map_err(map_surfc_invalid_argument)?;
                (surface, contour)
            }
        }
    } else {
        let (x_host, y_host) = axis_sources_to_host(&x_axis, &y_axis, BUILTIN_NAME)
            .await
            .map_err(map_surfc_invalid_argument)?;
        let grid = tensor_to_surface_grid_matlab_xy(
            z_input
                .into_tensor(BUILTIN_NAME)
                .map_err(map_surfc_invalid_argument)?,
            rows,
            cols,
            BUILTIN_NAME,
        )
        .map_err(map_surfc_invalid_argument)?;
        let mut surface = build_surface(x_host.clone(), y_host.clone(), grid.clone())
            .map_err(map_surfc_invalid_argument)?;
        style.apply_to_plot(&mut surface);
        let base_z = surface.bounds().min.z;
        let contour = build_contour_plot(
            BUILTIN_NAME,
            &x_host,
            &y_host,
            &grid,
            contour_map,
            base_z,
            &level_spec,
            &ContourLineColor::Auto,
        )
        .map_err(map_surfc_invalid_argument)?;
        (surface, contour)
    };

    surface = surface
        .with_colormap(ColorMap::Parula)
        .with_shading(ShadingMode::Smooth);
    retain_surface_source_data(&mut surface, source_x, source_y, source_z);

    let mut surface_opt = Some(surface);
    let mut contour_opt = Some(contour);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    work.check()?;
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let surface = surface_opt.take().expect("surfc surface consumed once");
        let contour = contour_opt.take().expect("surfc contour consumed once");
        let plot_index = figure.add_surface_plot_on_axes(surface, axes);
        *plot_index_slot.borrow_mut() = Some((axes, plot_index));
        figure.add_contour_plot_on_axes(contour, axes);
        Ok(())
    });
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle =
        crate::builtins::plotting::state::register_surface_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(handle);
        }
        return Err(map_surfc_internal(err));
    }
    Ok(handle)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{SURFC_DESCRIPTOR, SURFC_INTEGER_CAPABILITIES};
    use runmat_value::Value;
    async fn surfc_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
        super::surfc_builtin(
            crate::context::legacy::active(),
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .await
    }

    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    fn setup_plot_tests() {
        ensure_plot_test_env();
    }

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![data.len()]).expect("surfc test vector")
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn surfc_requires_matching_grid() {
        setup_plot_tests();
        let res = futures::executor::block_on(surfc_builtin(vec![
            Value::Tensor(tensor_from(&[0.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(Tensor::new(vec![0.0], vec![1]).expect("scalar grid")),
        ]));
        assert!(res.is_err());
    }

    #[test]
    fn surfc_catalog_infers_host_double_handle() {
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
            arguments: vec![input; 3],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let result =
            runmat_builtins::infer_catalog_call(&runmat_builtins::SURFC_CATALOG_ENTRY, &request);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs.len(), 1);
        let output = &result.outputs[0];
        assert_eq!(output.kind, numeric);
        assert_eq!(output.shape, ShapeFact::Scalar);
        assert_eq!(output.storage, StorageFact::Dense);
        assert_eq!(output.residency, ResidencyFact::Host);
    }

    #[test]
    fn surfc_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SURFC_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = surfc(Z)"));
        assert!(labels.contains(&"h = surfc(X, Y, Z)"));
        assert!(labels.contains(&"h = surfc(X, Y, Z, Name, Value, ...)"));
    }

    #[test]
    fn surfc_missing_input_uses_stable_identifier() {
        setup_plot_tests();
        let err = futures::executor::block_on(surfc_builtin(vec![])).expect_err("missing input");
        assert_eq!(err.identifier(), SURFC_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn surfc_returns_surface_handle() {
        setup_plot_tests();
        let handle = futures::executor::block_on(surfc_builtin(vec![Value::Tensor(
            Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]).expect("surfc surface"),
        )]))
        .expect("surfc should return handle");
        assert!(handle.is_finite());
    }

    #[test]
    fn surfc_zdata_property_retains_wide_integer_storage() {
        let _guard = lock_plot_registry();
        setup_plot_tests();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let expected = IntegerStorage::U64(vec![
            9_007_199_254_740_993,
            9_007_199_254_740_994,
            9_007_199_254_740_995,
            9_007_199_254_740_996,
        ]);
        let z = Tensor::new_integer(expected.clone(), vec![2, 2]).expect("integer surface Z");
        let handle = futures::executor::block_on(surfc_builtin(vec![Value::Tensor(z)]))
            .expect("integer surface with contour");
        let value = get_builtin(vec![Value::Num(handle), Value::String("ZData".into())])
            .expect("surfc ZData");

        assert!(
            matches!(value, Value::Tensor(tensor) if tensor.shape() == [2, 2] && tensor.integer_storage() == Some(&expected))
        );
    }

    #[test]
    fn surfc_integer_capabilities_cover_all_coordinate_roles() {
        assert_eq!(SURFC_INTEGER_CAPABILITIES.len(), 3);
        assert!(SURFC_INTEGER_CAPABILITIES
            .iter()
            .all(|capability| capability.inputs[0].classes.len() == 8));
    }
}
