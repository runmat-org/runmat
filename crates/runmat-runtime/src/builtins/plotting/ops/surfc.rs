//! MATLAB-compatible `surfc` builtin (surface with contour).

use log::warn;
#[cfg(test)]
use runmat_builtins::Tensor;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode};

use super::common::{tensor_to_surface_grid_matlab_xy, SurfaceDataInput};
use super::contour::{build_contour_plot, default_level_count, ContourLevelSpec, ContourLineColor};
use super::op_common::surface_composite::contour_for_surface_axes_input;
use super::op_common::surface_inputs::{
    axis_sources_to_host, parse_surface_call_args_matlab_xy, surface_axis_sources_from_xy_values,
};
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_surface_style_args, SurfaceStyleDefaults};
use super::surf::{build_surface, build_surface_gpu_plot_with_bounds_async};
use crate::build_runtime_error;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::RuntimeError;
use std::sync::Arc;

const BUILTIN_NAME: &str = "surfc";

const SURFC_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the rendered surface plot (with contour overlay).",
}];

const SURFC_INPUTS_Z: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Surface height grid.",
}];

const SURFC_INPUTS_X_Y_Z: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X axis vector/meshgrid matrix matching Z columns.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y axis vector/meshgrid matrix matching Z rows.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Surface height grid.",
    },
];

const SURFC_INPUTS_Z_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Surface height grid.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value surface style options.",
    },
];

const SURFC_INPUTS_X_Y_Z_PROPS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X axis vector/meshgrid matrix matching Z columns.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y axis vector/meshgrid matrix matching Z rows.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Surface height grid.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value surface style options.",
    },
];

const SURFC_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = surfc(Z)",
        inputs: &SURFC_INPUTS_Z,
        outputs: &SURFC_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = surfc(X, Y, Z)",
        inputs: &SURFC_INPUTS_X_Y_Z,
        outputs: &SURFC_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = surfc(Z, Name, Value, ...)",
        inputs: &SURFC_INPUTS_Z_PROPS,
        outputs: &SURFC_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = surfc(X, Y, Z, Name, Value, ...)",
        inputs: &SURFC_INPUTS_X_Y_Z_PROPS,
        outputs: &SURFC_OUTPUT_HANDLE,
    },
];

pub const SURFC_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SURFC.INVALID_ARGUMENT",
    identifier: Some("RunMat:surfc:InvalidArgument"),
    when: "Surface/grid/axis inputs or style name/value arguments are invalid.",
    message: "surfc: invalid argument",
};

pub const SURFC_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SURFC.INTERNAL",
    identifier: Some("RunMat:surfc:Internal"),
    when: "Internal surface/contour construction or render preparation fails unexpectedly.",
    message: "surfc: internal operation failed",
};

const SURFC_ERRORS: [BuiltinErrorDescriptor; 2] =
    [SURFC_ERROR_INVALID_ARGUMENT, SURFC_ERROR_INTERNAL];

pub const SURFC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SURFC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SURFC_ERRORS,
};

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
    surfc_error_with_detail(&SURFC_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_surfc_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    surfc_error_with_detail(&SURFC_ERROR_INTERNAL, err.message)
}

#[runtime_builtin(
    name = "surfc",
    category = "plotting",
    summary = "Create composite surface plots with projected contour overlays.",
    keywords = "surfc,plotting,surface,contour",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::surfc::SURFC_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::surfc"
)]
pub async fn surfc_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (x, y, z, rest) = parse_surface_call_args_matlab_xy(args, BUILTIN_NAME)
        .map_err(map_surfc_invalid_argument)?;
    let z_input = SurfaceDataInput::from_value(z, "surfc").map_err(map_surfc_invalid_argument)?;
    let (rows, cols) = z_input
        .grid_shape(BUILTIN_NAME)
        .map_err(map_surfc_invalid_argument)?;
    let (x_axis, y_axis) = surface_axis_sources_from_xy_values(x, y, rows, cols, BUILTIN_NAME)
        .await
        .map_err(map_surfc_invalid_argument)?;
    let style = Arc::new(
        parse_surface_style_args(
            "surfc",
            &rest,
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
        title: "Surface with Contours",
        x_label: "X",
        y_label: "Y",
        axis_equal: false,
        ..Default::default()
    };
    let level_spec = ContourLevelSpec::Count(default_level_count());
    // Build plots up-front so we can await GPU work without blocking the render loop.
    let contour_map = style.colormap;
    let (mut surface, contour) = if let Some(z_gpu) = z_input.gpu_handle().cloned() {
        match super::gpu_helpers::axis_bounds_async(&z_gpu, BUILTIN_NAME).await {
            Ok((min_z, max_z)) => match build_surface_gpu_plot_with_bounds_async(
                BUILTIN_NAME,
                &x_axis,
                &y_axis,
                &z_gpu,
                min_z,
                max_z,
                style.colormap,
                style.alpha,
                style.flatten_z,
            )
            .await
            {
                Ok(mut surface) => {
                    let base_z = surface.bounds().min.z;
                    style.apply_to_plot(&mut surface);
                    let contour = contour_for_surface_axes_input(
                        BUILTIN_NAME,
                        &x_axis,
                        &y_axis,
                        &z_input,
                        Some(z_gpu),
                        contour_map,
                        base_z,
                        &level_spec,
                    )
                    .await
                    .map_err(map_surfc_invalid_argument)?;
                    (surface, contour)
                }
                Err(err) => {
                    warn!("surfc surface GPU path unavailable: {err}");
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
            },
            Err(err) => {
                warn!("surfc GPU bounds unavailable: {err}");
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

    let mut surface_opt = Some(surface);
    let mut contour_opt = Some(contour);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
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
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(map_surfc_internal(err));
    }
    Ok(handle)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::plotting::tests::ensure_plot_test_env;
    use runmat_builtins::{ResolveContext, Type};

    fn setup_plot_tests() {
        ensure_plot_test_env();
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
    fn surfc_requires_matching_grid() {
        setup_plot_tests();
        let res = futures::executor::block_on(surfc_builtin(vec![
            Value::Tensor(tensor_from(&[0.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(Tensor {
                data: vec![0.0],
                shape: vec![1],
                rows: 1,
                cols: 1,
                dtype: runmat_builtins::NumericDType::F64,
            }),
        ]));
        assert!(res.is_err());
    }

    #[test]
    fn surfc_type_is_numeric_handle() {
        assert_eq!(
            handle_scalar_type(
                &[Type::tensor(), Type::tensor(), Type::tensor()],
                &ResolveContext::new(Vec::new())
            ),
            Type::Num
        );
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
        let handle = futures::executor::block_on(surfc_builtin(vec![Value::Tensor(Tensor {
            data: vec![0.0, 1.0, 1.0, 0.0],
            shape: vec![2, 2],
            rows: 2,
            cols: 2,
            dtype: runmat_builtins::NumericDType::F64,
        })]))
        .expect("surfc should return handle");
        assert!(handle.is_finite());
    }
}
