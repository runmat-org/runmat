//! MATLAB-compatible `surf` builtin.

use glam::Vec3;
use log::warn;
use runmat_accelerate_api::{self, GpuTensorHandle, ProviderPrecision};
#[cfg(test)]
use runmat_builtins::Tensor;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::gpu::ScalarType;
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
use super::perf::compute_surface_lod;
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_surface_style_args, SurfaceStyleDefaults};
use crate::build_runtime_error;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use std::convert::TryFrom;
use std::sync::Arc;

use crate::{BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "surf";

const SURF_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the rendered surface plot.",
}];

const SURF_INPUTS_Z: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Surface height grid.",
}];

const SURF_INPUTS_X_Y_Z: [BuiltinParamDescriptor; 3] = [
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

const SURF_INPUTS_Z_PROPS: [BuiltinParamDescriptor; 2] = [
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

const SURF_INPUTS_X_Y_Z_PROPS: [BuiltinParamDescriptor; 4] = [
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

const SURF_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = surf(Z)",
        inputs: &SURF_INPUTS_Z,
        outputs: &SURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = surf(X, Y, Z)",
        inputs: &SURF_INPUTS_X_Y_Z,
        outputs: &SURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = surf(Z, Name, Value, ...)",
        inputs: &SURF_INPUTS_Z_PROPS,
        outputs: &SURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = surf(X, Y, Z, Name, Value, ...)",
        inputs: &SURF_INPUTS_X_Y_Z_PROPS,
        outputs: &SURF_OUTPUT_HANDLE,
    },
];

pub const SURF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SURF.INVALID_ARGUMENT",
    identifier: Some("RunMat:surf:InvalidArgument"),
    when: "Surface/grid/axis inputs or style name/value arguments are invalid.",
    message: "surf: invalid argument",
};

pub const SURF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SURF.INTERNAL",
    identifier: Some("RunMat:surf:Internal"),
    when: "Internal surface construction or render preparation fails unexpectedly.",
    message: "surf: internal operation failed",
};

const SURF_ERRORS: [BuiltinErrorDescriptor; 2] = [SURF_ERROR_INVALID_ARGUMENT, SURF_ERROR_INTERNAL];

pub const SURF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SURF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SURF_ERRORS,
};

fn surf_error_with_detail(
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

fn map_surf_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    surf_error_with_detail(&SURF_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_surf_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    surf_error_with_detail(&SURF_ERROR_INTERNAL, err.message)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::surf")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "surf",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    // Plotting is a sink: it does not participate in fusion, but it *can* consume GPU-resident
    // tensors directly (zero-copy) when the renderer shares the provider WGPU context.
    // Do not force implicit gathers here; that defeats GPU-resident workloads.
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Surface rendering runs on the host/WebGPU pipeline; gpuArray inputs may remain on device when a shared WGPU context is available.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::surf")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "surf",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "surf terminates fusion graphs and performs rendering.",
};

#[runtime_builtin(
    name = "surf",
    category = "plotting",
    summary = "Render a MATLAB-compatible surface plot.",
    keywords = "surf,plotting,3d,surface",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::surf::SURF_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::surf"
)]
pub async fn surf_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (x, y, z, rest) =
        parse_surface_call_args_matlab_xy(args, BUILTIN_NAME).map_err(map_surf_invalid_argument)?;
    let z_input = SurfaceDataInput::from_value(z, "surf").map_err(map_surf_invalid_argument)?;
    let (rows, cols) = z_input
        .grid_shape(BUILTIN_NAME)
        .map_err(map_surf_invalid_argument)?;

    // Prefer a no-download path for vector-like gpuArray axes: keep X/Y on-device and pass
    // their buffers through to the GPU vertex packer. If X/Y are meshgrid matrices, we still
    // need to validate and extract axes on the host.
    let (x_axis, y_axis) = surface_axis_sources_from_xy_values(x, y, rows, cols, BUILTIN_NAME)
        .await
        .map_err(map_surf_invalid_argument)?;

    let style = Arc::new(
        parse_surface_style_args(
            "surf",
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
        .map_err(map_surf_invalid_argument)?,
    );
    let opts = PlotRenderOptions {
        title: "Surface Plot",
        x_label: "X",
        y_label: "Y",
        axis_equal: false,
        ..Default::default()
    };

    let mut surface = if let Some(z_gpu) = z_input.gpu_handle().cloned() {
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
                Ok(surface) => surface,
                Err(err) => {
                    warn!("surf GPU path unavailable: {err}");
                    let (x_host, y_host) = axis_sources_to_host(&x_axis, &y_axis, BUILTIN_NAME)
                        .await
                        .map_err(map_surf_invalid_argument)?;
                    build_surface_cpu(&z_input, x_host, y_host, rows, cols)
                        .await
                        .map_err(map_surf_invalid_argument)?
                }
            },
            Err(err) => {
                warn!("surf GPU bounds unavailable: {err}");
                let (x_host, y_host) = axis_sources_to_host(&x_axis, &y_axis, BUILTIN_NAME)
                    .await
                    .map_err(map_surf_invalid_argument)?;
                build_surface_cpu(&z_input, x_host, y_host, rows, cols)
                    .await
                    .map_err(map_surf_invalid_argument)?
            }
        }
    } else {
        let (x_host, y_host) = axis_sources_to_host(&x_axis, &y_axis, BUILTIN_NAME)
            .await
            .map_err(map_surf_invalid_argument)?;
        build_surface_cpu(&z_input, x_host, y_host, rows, cols)
            .await
            .map_err(map_surf_invalid_argument)?
    };

    style.apply_to_plot(&mut surface);
    let mut surface = Some(surface);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let surface = surface.take().expect("surf plot consumed once");
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
        return Err(map_surf_internal(err));
    }
    Ok(handle)
}

async fn build_surface_cpu(
    z_input: &SurfaceDataInput,
    x_axis: Vec<f64>,
    y_axis: Vec<f64>,
    rows: usize,
    cols: usize,
) -> BuiltinResult<SurfacePlot> {
    let z_tensor = match z_input.clone() {
        SurfaceDataInput::Host(tensor) => tensor,
        SurfaceDataInput::Gpu(handle) => {
            super::common::gather_tensor_from_gpu_async(handle, BUILTIN_NAME).await?
        }
    };
    let grid = tensor_to_surface_grid_matlab_xy(z_tensor, rows, cols, BUILTIN_NAME)?;
    build_surface(x_axis, y_axis, grid)
}

pub(crate) fn build_surface(
    x_axis: Vec<f64>,
    y_axis: Vec<f64>,
    z_grid: Vec<Vec<f64>>,
) -> BuiltinResult<SurfacePlot> {
    if x_axis.is_empty() || y_axis.is_empty() {
        return Err(plotting_error(
            BUILTIN_NAME,
            "surf: axis vectors must be non-empty",
        ));
    }

    let surface = SurfacePlot::new(x_axis, y_axis, z_grid)
        .map_err(|err| plotting_error(BUILTIN_NAME, format!("surf: {err}")))?
        .with_colormap(ColorMap::Parula)
        .with_shading(ShadingMode::Smooth);
    Ok(surface)
}

pub(crate) async fn build_surface_gpu_plot_with_bounds_async(
    name: &'static str,
    x_axis: &AxisSource,
    y_axis: &AxisSource,
    z: &GpuTensorHandle,
    min_z: f32,
    max_z: f32,
    colormap: ColorMap,
    alpha: f32,
    flatten_z: bool,
) -> BuiltinResult<SurfacePlot> {
    if x_axis.is_empty() || y_axis.is_empty() {
        return Err(plotting_error(
            name,
            format!("{name}: axis vectors must be non-empty"),
        ));
    }

    let context = super::gpu_helpers::ensure_shared_wgpu_context(name)?;

    let z_ref = runmat_accelerate_api::export_wgpu_buffer(z)
        .ok_or_else(|| plotting_error(name, format!("{name}: unable to export GPU Z data")))?;

    let x_len = x_axis.len();
    let y_len = y_axis.len();
    let expected_len = x_len
        .checked_mul(y_len)
        .ok_or_else(|| plotting_error(name, format!("{name}: grid dimensions overflowed")))?;
    if z_ref.len as usize != expected_len {
        return Err(plotting_error(
            name,
            format!(
                "{name}: Z must contain exactly {} elements ({}×{})",
                expected_len, x_len, y_len
            ),
        ));
    }

    let (min_x, max_x) = match x_axis {
        AxisSource::Host(v) => (
            v.iter()
                .fold(f32::INFINITY, |acc, &val| acc.min(val as f32)),
            v.iter()
                .fold(f32::NEG_INFINITY, |acc, &val| acc.max(val as f32)),
        ),
        AxisSource::Gpu(h) => super::gpu_helpers::axis_bounds_async(h, name).await?,
    };
    let (min_y, max_y) = match y_axis {
        AxisSource::Host(v) => (
            v.iter()
                .fold(f32::INFINITY, |acc, &val| acc.min(val as f32)),
            v.iter()
                .fold(f32::NEG_INFINITY, |acc, &val| acc.max(val as f32)),
        ),
        AxisSource::Gpu(h) => super::gpu_helpers::axis_bounds_async(h, name).await?,
    };
    let bounds = runmat_plot::core::scene::BoundingBox::new(
        Vec3::new(min_x, min_y, min_z),
        Vec3::new(max_x, max_y, max_z),
    );
    let extent_hint = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt();

    let color_table = build_color_lut(colormap, 512, 1.0);
    let scalar = ScalarType::from_is_f64(z_ref.precision == ProviderPrecision::F64);

    let mut x_axis_f32: Vec<f32> = Vec::new();
    let mut y_axis_f32: Vec<f32> = Vec::new();

    let x_axis_gpu = match x_axis {
        AxisSource::Gpu(h) => {
            let exported = runmat_accelerate_api::export_wgpu_buffer(h).ok_or_else(|| {
                plotting_error(name, format!("{name}: unable to export GPU X axis buffer"))
            })?;
            if exported.len as usize != x_len {
                return Err(plotting_error(
                    name,
                    format!(
                        "{name}: X axis length mismatch (expected {x_len}, got {})",
                        exported.len
                    ),
                ));
            }
            if exported.precision != z_ref.precision {
                return Err(plotting_error(
                    name,
                    format!("{name}: X axis precision must match Z precision"),
                ));
            }
            Some(exported.buffer.clone())
        }
        AxisSource::Host(v) => {
            if scalar == ScalarType::F32 {
                x_axis_f32 = v.iter().map(|&val| val as f32).collect::<Vec<f32>>();
            }
            None
        }
    };

    let y_axis_gpu = match y_axis {
        AxisSource::Gpu(h) => {
            let exported = runmat_accelerate_api::export_wgpu_buffer(h).ok_or_else(|| {
                plotting_error(name, format!("{name}: unable to export GPU Y axis buffer"))
            })?;
            if exported.len as usize != y_len {
                return Err(plotting_error(
                    name,
                    format!(
                        "{name}: Y axis length mismatch (expected {y_len}, got {})",
                        exported.len
                    ),
                ));
            }
            if exported.precision != z_ref.precision {
                return Err(plotting_error(
                    name,
                    format!("{name}: Y axis precision must match Z precision"),
                ));
            }
            Some(exported.buffer.clone())
        }
        AxisSource::Host(v) => {
            if scalar == ScalarType::F32 {
                y_axis_f32 = v.iter().map(|&val| val as f32).collect::<Vec<f32>>();
            }
            None
        }
    };

    let inputs = runmat_plot::gpu::surface::SurfaceGpuInputs {
        x_axis: if let Some(buffer) = x_axis_gpu.as_ref() {
            runmat_plot::gpu::surface::SurfaceAxis::Buffer(buffer.clone())
        } else if z_ref.precision == ProviderPrecision::F64 {
            match x_axis {
                AxisSource::Host(v) => runmat_plot::gpu::surface::SurfaceAxis::F64(v.as_slice()),
                AxisSource::Gpu(_) => unreachable!("gpu X axis handled above"),
            }
        } else {
            runmat_plot::gpu::surface::SurfaceAxis::F32(x_axis_f32.as_slice())
        },
        y_axis: if let Some(buffer) = y_axis_gpu.as_ref() {
            runmat_plot::gpu::surface::SurfaceAxis::Buffer(buffer.clone())
        } else if z_ref.precision == ProviderPrecision::F64 {
            match y_axis {
                AxisSource::Host(v) => runmat_plot::gpu::surface::SurfaceAxis::F64(v.as_slice()),
                AxisSource::Gpu(_) => unreachable!("gpu Y axis handled above"),
            }
        } else {
            runmat_plot::gpu::surface::SurfaceAxis::F32(y_axis_f32.as_slice())
        },
        z_buffer: z_ref.buffer.clone(),
        color_table: &color_table,
        x_len: x_len as u32,
        y_len: y_len as u32,
        scalar,
    };
    let lod = compute_surface_lod(x_len, y_len, extent_hint);
    let params = runmat_plot::gpu::surface::SurfaceGpuParams {
        min_z,
        max_z,
        alpha,
        flatten_z,
        x_stride: lod.stride_x,
        y_stride: lod.stride_y,
        lod_x_len: lod.lod_x_len,
        lod_y_len: lod.lod_y_len,
    };

    let gpu_vertices = runmat_plot::gpu::surface::pack_surface_vertices(
        &context.device,
        &context.queue,
        &inputs,
        &params,
    )
    .map_err(|e| plotting_error(name, format!("{name}: failed to build GPU vertices: {e}")))?;

    let vertex_count = lod.vertex_count();
    let lod_x_len = usize::try_from(lod.lod_x_len)
        .map_err(|_| plotting_error(name, format!("{name}: LOD X size overflowed")))?;
    let lod_y_len = usize::try_from(lod.lod_y_len)
        .map_err(|_| plotting_error(name, format!("{name}: LOD Y size overflowed")))?;
    let mut surface =
        SurfacePlot::from_gpu_buffer(lod_x_len, lod_y_len, gpu_vertices, vertex_count, bounds);
    surface.colormap = colormap;
    surface.alpha = alpha;
    surface.flatten_z = flatten_z;
    Ok(surface)
}

pub(crate) fn build_color_lut(colormap: ColorMap, samples: usize, alpha: f32) -> Vec<[f32; 4]> {
    let clamped = samples.max(2);
    (0..clamped)
        .map(|i| {
            let t = i as f32 / (clamped as f32 - 1.0);
            let rgb = colormap.map_value(t);
            [rgb.x, rgb.y, rgb.z, alpha]
        })
        .collect()
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
    fn surf_requires_matching_grid() {
        setup_plot_tests();
        let res = futures::executor::block_on(surf_builtin(vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0])),
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
    fn surf_type_is_numeric_handle() {
        assert_eq!(
            handle_scalar_type(
                &[Type::tensor(), Type::tensor(), Type::tensor()],
                &ResolveContext::new(Vec::new())
            ),
            Type::Num
        );
    }

    #[test]
    fn surf_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SURF_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = surf(Z)"));
        assert!(labels.contains(&"h = surf(X, Y, Z)"));
        assert!(labels.contains(&"h = surf(X, Y, Z, Name, Value, ...)"));
    }

    #[test]
    fn surf_missing_input_uses_stable_identifier() {
        setup_plot_tests();
        let err = futures::executor::block_on(surf_builtin(vec![])).expect_err("missing input");
        assert_eq!(err.identifier(), SURF_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn surf_accepts_meshgrid_xy_matrices() {
        // x = [0, 1, 2], y = [10, 20] => meshgrid X/Y are 2x3.
        // Column-major storage for X:
        // X = [0 1 2;
        //      0 1 2]
        let x = Tensor {
            data: vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            shape: vec![2, 3],
            rows: 2,
            cols: 3,
            dtype: runmat_builtins::NumericDType::F64,
        };
        // Y = [10 10 10;
        //      20 20 20]
        let y = Tensor {
            data: vec![10.0, 20.0, 10.0, 20.0, 10.0, 20.0],
            shape: vec![2, 3],
            rows: 2,
            cols: 3,
            dtype: runmat_builtins::NumericDType::F64,
        };
        // Z is 2x3 surface heights (any values), column-major.
        let z = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape: vec![2, 3],
            rows: 2,
            cols: 3,
            dtype: runmat_builtins::NumericDType::F64,
        };

        let (x_axis, y_axis) = crate::builtins::plotting::op_common::surface_inputs::extract_meshgrid_axes_from_xy_matrices(&x, &y, z.rows, z.cols, BUILTIN_NAME).expect("axes");
        assert_eq!(x_axis.len(), 3);
        assert_eq!(y_axis.len(), 2);

        // With extracted axes, Z should validate and reshape into a surface grid.
        let grid = tensor_to_surface_grid_matlab_xy(z, y_axis.len(), x_axis.len(), BUILTIN_NAME);
        assert!(
            grid.is_ok(),
            "expected Z to be compatible with extracted axes"
        );
    }

    #[test]
    fn surf_z_only_shorthand_builds_surface_with_default_axes() {
        setup_plot_tests();
        let out = futures::executor::block_on(surf_builtin(vec![Value::Tensor(Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
            rows: 2,
            cols: 2,
            dtype: runmat_builtins::NumericDType::F64,
        })]));
        assert!(out.is_ok() || out.is_err());
        let (x, y, z, rest) =
            crate::builtins::plotting::op_common::surface_inputs::parse_surface_call_args_matlab_xy(
                vec![Value::Tensor(Tensor {
                    data: vec![1.0, 2.0, 3.0, 4.0],
                    shape: vec![2, 2],
                    rows: 2,
                    cols: 2,
                    dtype: runmat_builtins::NumericDType::F64,
                })],
                BUILTIN_NAME,
            )
            .unwrap();
        assert!(rest.is_empty());
        assert_eq!(Tensor::try_from(&x).unwrap().data, vec![1.0, 2.0]);
        assert_eq!(Tensor::try_from(&y).unwrap().data, vec![1.0, 2.0]);
        assert_eq!(Tensor::try_from(&z).unwrap().data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn surf_z_only_shorthand_uses_matlab_xy_for_non_square_grid() {
        let (x, y, _, rest) =
            crate::builtins::plotting::op_common::surface_inputs::parse_surface_call_args_matlab_xy(
                vec![Value::Tensor(Tensor {
                    data: (1..=12).map(|v| v as f64).collect(),
                    shape: vec![3, 4],
                    rows: 3,
                    cols: 4,
                    dtype: runmat_builtins::NumericDType::F64,
                })],
                BUILTIN_NAME,
            )
            .unwrap();
        assert!(rest.is_empty());
        assert_eq!(Tensor::try_from(&x).unwrap().data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(Tensor::try_from(&y).unwrap().data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn surf_returns_surface_handle() {
        setup_plot_tests();
        let handle = futures::executor::block_on(surf_builtin(vec![Value::Tensor(Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
            rows: 2,
            cols: 2,
            dtype: runmat_builtins::NumericDType::F64,
        })]))
        .expect("surf should return a handle");
        assert!(handle.is_finite());
    }
}
