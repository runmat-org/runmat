//! MATLAB-compatible `mesh` builtin.

use log::warn;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, Tensor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, NumericPlotData, ShadingMode, SurfacePlot};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

use super::common::{tensor_to_surface_grid_matlab_xy, SurfaceDataInput};
use super::op_common::surface_inputs::{
    axis_sources_to_host, parse_surface_call_args_matlab_xy, surface_axis_sources_from_xy_values,
    AxisSource,
};
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_surface_style_args, SurfaceStyleDefaults};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use std::sync::Arc;

const BUILTIN_NAME: &str = "mesh";

const INTEGER_X: [BuiltinIntegerInputCapability; 1] = [mesh_integer_input("X")];
const INTEGER_Y: [BuiltinIntegerInputCapability; 1] = [mesh_integer_input("Y")];
const INTEGER_Z: [BuiltinIntegerInputCapability; 1] = [mesh_integer_input("Z")];

const fn mesh_integer_input(name: &'static str) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "R2026a explicitly lists every built-in integer class for this surface coordinate role.",
    }
}

const fn mesh_integer_capability(
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
        notes: "Native class, exact values, and shape remain authoritative on XData/YData/ZData and in RunMat scene persistence; rendering and client-side GPU handling are explicit floating boundaries. The documented independent C color-array form is a general unimplemented graphics gap, not an integer coercion rule.",
    }
}

pub const MESH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    mesh_integer_capability("s = mesh(integer_X, Y, Z, ...)", &INTEGER_X),
    mesh_integer_capability("s = mesh(X, integer_Y, Z, ...)", &INTEGER_Y),
    mesh_integer_capability("s = mesh(X, Y, integer_Z, ...)", &INTEGER_Z),
];

const MESH_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the rendered mesh surface.",
}];

const MESH_INPUTS_Z: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Surface height grid.",
}];

const MESH_INPUTS_X_Y_Z: [BuiltinParamDescriptor; 3] = [
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

const MESH_INPUTS_Z_PROPS: [BuiltinParamDescriptor; 2] = [
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

const MESH_INPUTS_X_Y_Z_PROPS: [BuiltinParamDescriptor; 4] = [
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

const MESH_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = mesh(Z)",
        inputs: &MESH_INPUTS_Z,
        outputs: &MESH_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = mesh(X, Y, Z)",
        inputs: &MESH_INPUTS_X_Y_Z,
        outputs: &MESH_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = mesh(Z, Name, Value, ...)",
        inputs: &MESH_INPUTS_Z_PROPS,
        outputs: &MESH_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = mesh(X, Y, Z, Name, Value, ...)",
        inputs: &MESH_INPUTS_X_Y_Z_PROPS,
        outputs: &MESH_OUTPUT_HANDLE,
    },
];

pub const MESH_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MESH.INVALID_ARGUMENT",
    identifier: Some("RunMat:mesh:InvalidArgument"),
    when: "Surface input arrays or style name/value arguments are invalid.",
    message: "mesh: invalid argument",
};

pub const MESH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MESH.INTERNAL",
    identifier: Some("RunMat:mesh:Internal"),
    when: "Internal surface generation/render preparation fails unexpectedly.",
    message: "mesh: internal operation failed",
};

const MESH_ERRORS: [BuiltinErrorDescriptor; 2] = [MESH_ERROR_INVALID_ARGUMENT, MESH_ERROR_INTERNAL];

pub const MESH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MESH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MESH_ERRORS,
};

fn mesh_error_with_detail(
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

pub(crate) fn map_mesh_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    mesh_error_with_detail(&MESH_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_mesh_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    mesh_error_with_detail(&MESH_ERROR_INTERNAL, err.message)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::mesh")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "mesh",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    // Plotting is a sink, but can consume gpuArray inputs zero-copy when a shared WGPU context exists.
    // Avoid forcing implicit gathers.
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Wireframe rendering terminates fusion graphs; gpuArray inputs may remain on device when shared plotting context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::mesh")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "mesh",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "mesh terminates fusion graphs.",
};

#[runtime_builtin(
    name = "mesh",
    category = "plotting",
    summary = "Render a MATLAB-compatible wireframe surface.",
    keywords = "mesh,wireframe,surface,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::mesh::MESH_DESCRIPTOR),
    integer_capabilities(crate::builtins::plotting::mesh::MESH_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::mesh"
)]
pub async fn mesh_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (x, y, z, rest) =
        parse_surface_call_args_matlab_xy(args, BUILTIN_NAME).map_err(map_mesh_invalid_argument)?;
    let source_x = numeric_plot_data_from_value(&x, BUILTIN_NAME)
        .await
        .map_err(map_mesh_invalid_argument)?;
    let source_y = numeric_plot_data_from_value(&y, BUILTIN_NAME)
        .await
        .map_err(map_mesh_invalid_argument)?;
    let z_input = SurfaceDataInput::from_value(z, "mesh").map_err(map_mesh_invalid_argument)?;
    let source_z = match &z_input {
        SurfaceDataInput::Host(tensor) => Some(numeric_plot_data_from_tensor(tensor)),
        SurfaceDataInput::Gpu(handle) => Some(numeric_plot_data_from_tensor(
            &super::common::gather_tensor_from_gpu_async(handle.clone(), BUILTIN_NAME)
                .await
                .map_err(map_mesh_invalid_argument)?,
        )),
    };
    let (rows, cols) = z_input
        .grid_shape(BUILTIN_NAME)
        .map_err(map_mesh_invalid_argument)?;

    // Match surf semantics: keep vector-like gpuArray axes on-device when possible; otherwise
    // gather to validate meshgrid matrix inputs and extract axis vectors.
    let (x_axis, y_axis) = surface_axis_sources_from_xy_values(x, y, rows, cols, BUILTIN_NAME)
        .await
        .map_err(map_mesh_invalid_argument)?;

    let style = Arc::new(
        parse_surface_style_args(
            "mesh",
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
        .map_err(map_mesh_invalid_argument)?,
    );
    let opts = PlotRenderOptions {
        title: "Mesh Plot",
        x_label: "X",
        y_label: "Y",
        axis_equal: false,
        ..Default::default()
    };

    let mut surface = if let Some(z_gpu) = z_input.gpu_handle().cloned() {
        match super::gpu_helpers::axis_bounds_async(&z_gpu, BUILTIN_NAME).await {
            Ok((min_z, max_z)) => match super::surf::build_surface_gpu_plot_with_bounds_async(
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
                Ok(surface_gpu) => surface_gpu,
                Err(err) => {
                    warn!("mesh GPU path unavailable: {err}");
                    build_mesh_cpu(&z_input, &x_axis, &y_axis, rows, cols)
                        .await
                        .map_err(map_mesh_invalid_argument)?
                }
            },
            Err(err) => {
                warn!("mesh GPU bounds unavailable: {err}");
                build_mesh_cpu(&z_input, &x_axis, &y_axis, rows, cols)
                    .await
                    .map_err(map_mesh_invalid_argument)?
            }
        }
    } else {
        build_mesh_cpu(&z_input, &x_axis, &y_axis, rows, cols)
            .await
            .map_err(map_mesh_invalid_argument)?
    };

    surface = surface
        .with_colormap(ColorMap::Turbo)
        .with_wireframe(true)
        .with_shading(ShadingMode::Faceted);
    style.apply_to_plot(&mut surface);
    retain_surface_source_data(&mut surface, source_x, source_y, source_z);

    let mut surface_opt = Some(surface);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let surface = surface_opt.take().expect("mesh plot consumed exactly once");
        let plot_index = figure.add_surface_plot_on_axes(surface, axes);
        *plot_index_slot.borrow_mut() = Some((axes, plot_index));
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
        return Err(map_mesh_internal(err));
    }
    Ok(handle)
}

pub(crate) fn host_numeric_plot_data(value: &Value) -> Option<NumericPlotData> {
    let tensor = Tensor::try_from(value).ok()?;
    Some(numeric_plot_data_from_tensor(&tensor))
}

pub(crate) async fn numeric_plot_data_from_value(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<Option<NumericPlotData>> {
    match value {
        Value::GpuTensor(handle) => Ok(Some(numeric_plot_data_from_tensor(
            &super::common::gather_tensor_from_gpu_async(handle.clone(), builtin).await?,
        ))),
        _ => Ok(host_numeric_plot_data(value)),
    }
}

pub(crate) fn numeric_plot_data_from_tensor(tensor: &Tensor) -> NumericPlotData {
    NumericPlotData::new(
        tensor
            .clone()
            .into_numeric_storage()
            .expect("surface source is numeric"),
        tensor.shape.clone(),
    )
    .expect("surface source shape is validated")
}

pub(crate) fn retain_surface_source_data(
    surface: &mut SurfacePlot,
    source_x: Option<NumericPlotData>,
    source_y: Option<NumericPlotData>,
    source_z: Option<NumericPlotData>,
) {
    let (fallback_x, fallback_y, fallback_z) = surface.source_data();
    let fallback_x = fallback_x.cloned();
    let fallback_y = fallback_y.cloned();
    let fallback_z = fallback_z.cloned();
    surface.set_source_data(
        source_x.or(fallback_x),
        source_y.or(fallback_y),
        source_z.or(fallback_z),
    );
}

async fn build_mesh_cpu(
    z_input: &SurfaceDataInput,
    x_axis: &AxisSource,
    y_axis: &AxisSource,
    rows: usize,
    cols: usize,
) -> BuiltinResult<SurfacePlot> {
    let (x_host, y_host) = axis_sources_to_host(x_axis, y_axis, BUILTIN_NAME).await?;
    let z_tensor = match z_input {
        SurfaceDataInput::Host(t) => t.clone(),
        SurfaceDataInput::Gpu(h) => {
            super::common::gather_tensor_from_gpu_async(h.clone(), BUILTIN_NAME).await?
        }
    };
    let grid = tensor_to_surface_grid_matlab_xy(z_tensor, rows, cols, BUILTIN_NAME)?;
    build_mesh_surface(x_host, y_host, grid)
}

pub(crate) fn build_mesh_surface(
    x_axis: Vec<f64>,
    y_axis: Vec<f64>,
    z_grid: Vec<Vec<f64>>,
) -> BuiltinResult<SurfacePlot> {
    if x_axis.is_empty() || y_axis.is_empty() {
        return Err(plotting_error(
            "mesh",
            "mesh: axis vectors must be non-empty",
        ));
    }

    let surface = SurfacePlot::new(x_axis, y_axis, z_grid)
        .map_err(|err| plotting_error("mesh", format!("mesh: {err}")))?
        .with_colormap(ColorMap::Turbo)
        .with_wireframe(true)
        .with_shading(ShadingMode::Faceted);
    Ok(surface)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use runmat_builtins::{IntegerStorage, ResolveContext, Type};

    fn setup_plot_tests() {
        ensure_plot_test_env();
    }

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![data.len()]).expect("mesh test vector")
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mesh_requires_matching_grid() {
        setup_plot_tests();
        let res = futures::executor::block_on(mesh_builtin(vec![
            Value::Tensor(tensor_from(&[0.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(Tensor::new(vec![0.0], vec![1]).expect("scalar grid")),
        ]));
        assert!(res.is_err());
    }

    #[test]
    fn mesh_type_is_numeric_handle() {
        assert_eq!(
            handle_scalar_type(
                &[Type::tensor(), Type::tensor(), Type::tensor()],
                &ResolveContext::new(Vec::new())
            ),
            Type::Num
        );
    }

    #[test]
    fn mesh_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = MESH_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = mesh(Z)"));
        assert!(labels.contains(&"h = mesh(X, Y, Z)"));
        assert!(labels.contains(&"h = mesh(X, Y, Z, Name, Value, ...)"));
    }

    #[test]
    fn mesh_missing_input_uses_stable_identifier() {
        setup_plot_tests();
        let err = futures::executor::block_on(mesh_builtin(vec![])).expect_err("missing input");
        assert_eq!(err.identifier(), MESH_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn mesh_zdata_property_retains_wide_integer_storage() {
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
        let z = Tensor::new_integer(expected.clone(), vec![2, 2]).expect("integer mesh Z");
        let handle = futures::executor::block_on(mesh_builtin(vec![Value::Tensor(z)]))
            .expect("integer mesh");
        let value = get_builtin(vec![Value::Num(handle), Value::String("ZData".into())])
            .expect("mesh ZData");

        assert!(
            matches!(value, Value::Tensor(tensor) if tensor.shape == vec![2, 2] && tensor.integer_storage() == Some(&expected))
        );
    }

    #[test]
    fn mesh_integer_capabilities_cover_all_coordinate_roles() {
        assert_eq!(MESH_INTEGER_CAPABILITIES.len(), 3);
        assert!(MESH_INTEGER_CAPABILITIES
            .iter()
            .all(|capability| capability.inputs[0].classes.len() == 8));
    }

    #[test]
    fn mesh_resident_integer_zdata_retains_exact_host_property_authority() {
        let _guard = lock_plot_registry();
        setup_plot_tests();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let expected = IntegerStorage::U64(vec![
                9_007_199_254_740_993,
                9_007_199_254_740_994,
                9_007_199_254_740_995,
                9_007_199_254_740_996,
            ]);
            let source = Tensor::new_integer(expected.clone(), vec![2, 2]).expect("integer Z");
            let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &source)
                .expect("resident integer Z");
            let handle =
                futures::executor::block_on(mesh_builtin(vec![Value::GpuTensor(resident)]))
                    .expect("resident integer mesh");
            let value = get_builtin(vec![Value::Num(handle), Value::String("ZData".into())])
                .expect("mesh ZData");

            assert!(
                matches!(value, Value::Tensor(tensor) if tensor.shape == vec![2, 2] && tensor.integer_storage() == Some(&expected))
            );
        });
    }
}
