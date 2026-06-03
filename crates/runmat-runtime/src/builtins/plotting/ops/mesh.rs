//! MATLAB-compatible `mesh` builtin.

use log::warn;
#[cfg(test)]
use runmat_builtins::Tensor;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode, SurfacePlot};

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
    builtin_path = "crate::builtins::plotting::mesh"
)]
pub async fn mesh_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (x, y, z, rest) =
        parse_surface_call_args_matlab_xy(args, BUILTIN_NAME).map_err(map_mesh_invalid_argument)?;
    let z_input = SurfaceDataInput::from_value(z, "mesh").map_err(map_mesh_invalid_argument)?;
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
                style.colormap,
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
    fn mesh_requires_matching_grid() {
        setup_plot_tests();
        let res = futures::executor::block_on(mesh_builtin(vec![
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
}
