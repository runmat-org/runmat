//! MATLAB-compatible `meshc` builtin (mesh with contour).

use log::warn;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode};
use runmat_value::Value;
#[cfg(test)]
use runmat_value::{IntegerStorage, Tensor};

use super::common::{tensor_to_surface_grid_matlab_xy, SurfaceDataInput};
use super::contour::{build_contour_plot, default_level_count, ContourLevelSpec, ContourLineColor};
use super::mesh::{
    build_mesh_surface, numeric_plot_data_from_tensor, numeric_plot_data_from_value,
    retain_surface_source_data,
};
use super::op_common::surface_composite::contour_for_surface_axes_input;
use super::op_common::surface_inputs::{
    axis_sources_to_host, parse_surface_call_args_matlab_xy, surface_axis_sources_from_xy_values,
};
use super::state::{render_active_plot, PlotRenderOptions};

use super::style::{parse_surface_style_args, SurfaceStyleDefaults};
use super::surf::build_surface_gpu_plot_with_bounds_async;
use crate::build_runtime_error;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::RuntimeError;
use std::sync::Arc;

const BUILTIN_NAME: &str = "meshc";

const INTEGER_X: [BuiltinIntegerInputCapability; 1] = [meshc_integer_input("X")];
const INTEGER_Y: [BuiltinIntegerInputCapability; 1] = [meshc_integer_input("Y")];
const INTEGER_Z: [BuiltinIntegerInputCapability; 1] = [meshc_integer_input("Z")];

const fn meshc_integer_input(name: &'static str) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "The compatibility target explicitly lists every built-in integer class for this surface coordinate role.",
    }
}

const fn meshc_integer_capability(
    form: &'static str,
    inputs: &'static [BuiltinIntegerInputCapability],
) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Native class, exact values, and shape remain authoritative on the primary Surface object and in RunMat scene persistence; surface rendering, contour generation, and client-side GPU handling are explicit floating boundaries. The documented independent C color-array form is a general unimplemented graphics gap.",
    }
}

pub const MESHC_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    meshc_integer_capability("s = meshc(integer_X, Y, Z, ...)", &INTEGER_X),
    meshc_integer_capability("s = meshc(X, integer_Y, Z, ...)", &INTEGER_Y),
    meshc_integer_capability("s = meshc(X, Y, integer_Z, ...)", &INTEGER_Z),
];

const MESHC_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the rendered mesh surface (with contour overlay).",
}];

const MESHC_INPUTS_Z: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Surface height grid.",
}];

const MESHC_INPUTS_X_Y_Z: [BuiltinParamDescriptor; 3] = [
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

const MESHC_INPUTS_Z_PROPS: [BuiltinParamDescriptor; 2] = [
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

const MESHC_INPUTS_X_Y_Z_PROPS: [BuiltinParamDescriptor; 4] = [
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

const MESHC_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = meshc(Z)",
        inputs: &MESHC_INPUTS_Z,
        outputs: &MESHC_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = meshc(X, Y, Z)",
        inputs: &MESHC_INPUTS_X_Y_Z,
        outputs: &MESHC_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = meshc(Z, Name, Value, ...)",
        inputs: &MESHC_INPUTS_Z_PROPS,
        outputs: &MESHC_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = meshc(X, Y, Z, Name, Value, ...)",
        inputs: &MESHC_INPUTS_X_Y_Z_PROPS,
        outputs: &MESHC_OUTPUT_HANDLE,
    },
];

pub const MESHC_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MESHC.INVALID_ARGUMENT",
    identifier: Some("RunMat:meshc:InvalidArgument"),
    when: "Surface/grid/axis inputs or style name/value arguments are invalid.",
    message: "meshc: invalid argument",
};

pub const MESHC_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MESHC.INTERNAL",
    identifier: Some("RunMat:meshc:Internal"),
    when: "Internal surface/contour construction or render preparation fails unexpectedly.",
    message: "meshc: internal operation failed",
};

const MESHC_ERRORS: [BuiltinErrorDescriptor; 2] =
    [MESHC_ERROR_INVALID_ARGUMENT, MESHC_ERROR_INTERNAL];

pub const MESHC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MESHC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MESHC_ERRORS,
};

fn meshc_error_with_detail(
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

fn map_meshc_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    meshc_error_with_detail(&MESHC_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_meshc_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    meshc_error_with_detail(&MESHC_ERROR_INTERNAL, err.message)
}

#[runtime_builtin(
    name = "meshc",
    category = "plotting",
    summary = "Render a MATLAB-compatible mesh with contour overlay.",
    keywords = "meshc,plotting,mesh,contour",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::meshc::MESHC_DESCRIPTOR),
    integer_capabilities(crate::builtins::plotting::meshc::MESHC_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::meshc"
)]
pub async fn meshc_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (x, y, z, rest) = parse_surface_call_args_matlab_xy(args, BUILTIN_NAME)
        .map_err(map_meshc_invalid_argument)?;
    let source_x = numeric_plot_data_from_value(&x, BUILTIN_NAME)
        .await
        .map_err(map_meshc_invalid_argument)?;
    let source_y = numeric_plot_data_from_value(&y, BUILTIN_NAME)
        .await
        .map_err(map_meshc_invalid_argument)?;
    let z_input = SurfaceDataInput::from_value(z, "meshc").map_err(map_meshc_invalid_argument)?;
    let source_z = match &z_input {
        SurfaceDataInput::Host(tensor) => Some(numeric_plot_data_from_tensor(tensor)),
        SurfaceDataInput::Gpu(handle) => Some(numeric_plot_data_from_tensor(
            &super::common::gather_tensor_from_gpu_async(handle.clone(), BUILTIN_NAME)
                .await
                .map_err(map_meshc_invalid_argument)?,
        )),
    };
    let (rows, cols) = z_input
        .grid_shape(BUILTIN_NAME)
        .map_err(map_meshc_invalid_argument)?;
    let (x_axis, y_axis) = surface_axis_sources_from_xy_values(x, y, rows, cols, BUILTIN_NAME)
        .await
        .map_err(map_meshc_invalid_argument)?;
    let style = Arc::new(
        parse_surface_style_args(
            "meshc",
            &rest,
            SurfaceStyleDefaults::new(
                ColorMap::Turbo,
                ShadingMode::Faceted,
                true,
                1.0,
                false,
                true,
            ),
        )
        .map_err(map_meshc_invalid_argument)?,
    );
    let opts = PlotRenderOptions {
        title: "Mesh with Contours",
        x_label: "X",
        y_label: "Y",
        axis_equal: false,
        ..Default::default()
    };
    let level_spec = ContourLevelSpec::Count(default_level_count());
    let contour_map = style.colormap.clone();
    let (mut surface, contour) = if let Some(z_gpu) = z_input.gpu_handle().cloned() {
        match super::gpu_helpers::axis_bounds_async(&z_gpu, BUILTIN_NAME).await {
            Ok((min_z, max_z)) => match build_surface_gpu_plot_with_bounds_async(
                BUILTIN_NAME,
                &x_axis,
                &y_axis,
                &z_gpu,
                min_z,
                max_z,
                style.colormap.clone(),
                style.alpha,
                style.flatten_z,
            )
            .await
            {
                Ok(surface_gpu) => {
                    let mut surface = surface_gpu.with_wireframe(true);
                    style.apply_to_plot(&mut surface);
                    let base_z = surface.bounds().min.z;
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
                    .map_err(map_meshc_invalid_argument)?;
                    (surface, contour)
                }
                Err(err) => {
                    warn!("meshc surface GPU path unavailable: {err}");
                    let (x_host, y_host) =
                        axis_sources_to_host(&x_axis, &y_axis, BUILTIN_NAME).await?;
                    let z_tensor = super::common::gather_tensor_from_gpu_async(z_gpu, BUILTIN_NAME)
                        .await
                        .map_err(map_meshc_invalid_argument)?;
                    let grid = tensor_to_surface_grid_matlab_xy(z_tensor, rows, cols, BUILTIN_NAME)
                        .map_err(map_meshc_invalid_argument)?;
                    let mut surface =
                        build_mesh_surface(x_host.clone(), y_host.clone(), grid.clone())
                            .map_err(map_meshc_invalid_argument)?;
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
                    .map_err(map_meshc_invalid_argument)?;
                    (surface, contour)
                }
            },
            Err(err) => {
                warn!("meshc GPU bounds unavailable: {err}");
                let (x_host, y_host) = axis_sources_to_host(&x_axis, &y_axis, BUILTIN_NAME)
                    .await
                    .map_err(map_meshc_invalid_argument)?;
                let z_tensor = super::common::gather_tensor_from_gpu_async(z_gpu, BUILTIN_NAME)
                    .await
                    .map_err(map_meshc_invalid_argument)?;
                let grid = tensor_to_surface_grid_matlab_xy(z_tensor, rows, cols, BUILTIN_NAME)
                    .map_err(map_meshc_invalid_argument)?;
                let mut surface = build_mesh_surface(x_host.clone(), y_host.clone(), grid.clone())
                    .map_err(map_meshc_invalid_argument)?;
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
                .map_err(map_meshc_invalid_argument)?;
                (surface, contour)
            }
        }
    } else {
        let (x_host, y_host) = axis_sources_to_host(&x_axis, &y_axis, BUILTIN_NAME)
            .await
            .map_err(map_meshc_invalid_argument)?;
        let grid = tensor_to_surface_grid_matlab_xy(
            z_input
                .into_tensor(BUILTIN_NAME)
                .map_err(map_meshc_invalid_argument)?,
            rows,
            cols,
            BUILTIN_NAME,
        )
        .map_err(map_meshc_invalid_argument)?;
        let mut surface = build_mesh_surface(x_host.clone(), y_host.clone(), grid.clone())
            .map_err(map_meshc_invalid_argument)?;
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
        .map_err(map_meshc_invalid_argument)?;
        (surface, contour)
    };

    surface = surface
        .with_colormap(ColorMap::Turbo)
        .with_wireframe(true)
        .with_shading(ShadingMode::Faceted);
    retain_surface_source_data(&mut surface, source_x, source_y, source_z);

    let mut surface_opt = Some(surface);
    let mut contour_opt = Some(contour);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let surface = surface_opt.take().expect("meshc surface consumed once");
        let contour = contour_opt.take().expect("meshc contour consumed once");
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
        return Err(map_meshc_internal(err));
    }
    Ok(handle)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use runmat_builtins::{ResolveContext, Type};

    fn setup_plot_tests() {
        ensure_plot_test_env();
    }

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![data.len()]).expect("meshc test vector")
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn meshc_requires_matching_grid() {
        setup_plot_tests();
        let res = futures::executor::block_on(meshc_builtin(vec![
            Value::Tensor(tensor_from(&[0.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(Tensor::new(vec![0.0], vec![1]).expect("scalar grid")),
        ]));
        assert!(res.is_err());
    }

    #[test]
    fn meshc_type_is_numeric_handle() {
        assert_eq!(
            handle_scalar_type(
                &[Type::tensor(), Type::tensor(), Type::tensor()],
                &ResolveContext::new(Vec::new())
            ),
            Type::Num
        );
    }

    #[test]
    fn meshc_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = MESHC_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = meshc(Z)"));
        assert!(labels.contains(&"h = meshc(X, Y, Z)"));
        assert!(labels.contains(&"h = meshc(X, Y, Z, Name, Value, ...)"));
    }

    #[test]
    fn meshc_missing_input_uses_stable_identifier() {
        setup_plot_tests();
        let err = futures::executor::block_on(meshc_builtin(vec![])).expect_err("missing input");
        assert_eq!(err.identifier(), MESHC_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn meshc_returns_surface_handle() {
        setup_plot_tests();
        let handle = futures::executor::block_on(meshc_builtin(vec![Value::Tensor(
            Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]).expect("meshc surface"),
        )]))
        .expect("meshc should return handle");
        assert!(handle.is_finite());
    }

    #[test]
    fn meshc_integer_capabilities_cover_all_coordinate_roles() {
        assert_eq!(MESHC_INTEGER_CAPABILITIES.len(), 3);
        assert!(MESHC_INTEGER_CAPABILITIES
            .iter()
            .all(|capability| capability.inputs[0].classes.len() == 8));
    }

    #[test]
    fn meshc_resident_integer_zdata_retains_exact_host_property_authority() {
        let _guard = lock_plot_registry();
        setup_plot_tests();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let expected = IntegerStorage::U64(vec![1, 2, 3, 4]);
            let source = Tensor::new_integer(expected.clone(), vec![2, 2]).expect("integer Z");
            let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &source)
                .expect("resident integer Z");
            let handle =
                futures::executor::block_on(meshc_builtin(vec![Value::GpuTensor(resident)]))
                    .expect("resident integer meshc");
            let value = get_builtin(vec![Value::Num(handle), Value::String("ZData".into())])
                .expect("meshc ZData");

            assert!(
                matches!(value, Value::Tensor(tensor) if tensor.shape == vec![2, 2] && tensor.integer_storage() == Some(&expected))
            );
        });
    }
}
