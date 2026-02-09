//! MATLAB-compatible `surf` builtin.

use glam::Vec3;
use log::warn;
use runmat_accelerate_api::{self, GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::{ColorMap, ShadingMode, SurfacePlot};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

use super::common::{numeric_vector, tensor_to_surface_grid, SurfaceDataInput};
use super::perf::compute_surface_lod;
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_surface_style_args, SurfaceStyleDefaults};
use crate::builtins::plotting::type_resolvers::string_type;
use std::convert::TryFrom;
use std::sync::Arc;

use crate::BuiltinResult;

const BUILTIN_NAME: &str = "surf";

fn is_vector_like(tensor: &Tensor) -> bool {
    // MATLAB vectors are 1xN or Nx1 (scalars included). Treat empty shape as scalar.
    if tensor.shape.is_empty() {
        return true;
    }
    tensor.rows == 1 || tensor.cols == 1
}

pub(crate) fn is_vector_like_shape(shape: &[usize]) -> bool {
    if shape.is_empty() {
        return true;
    }
    let non_singleton = shape.iter().copied().filter(|d| *d > 1).count();
    non_singleton <= 1
}

pub(crate) fn vector_len_from_shape(shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 1;
    }
    shape.iter().copied().max().unwrap_or(1)
}

#[derive(Clone)]
pub(crate) enum AxisSource {
    Host(Vec<f64>),
    Gpu(GpuTensorHandle),
}

impl AxisSource {
    fn len(&self) -> usize {
        match self {
            AxisSource::Host(v) => v.len(),
            AxisSource::Gpu(h) => vector_len_from_shape(&h.shape),
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

async fn gather_axes_for_cpu_fallback(
    x: &AxisSource,
    y: &AxisSource,
    name: &'static str,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let x_vec = match x {
        AxisSource::Host(v) => v.clone(),
        AxisSource::Gpu(h) => {
            let t = super::common::gather_tensor_from_gpu_async(h.clone(), name).await?;
            numeric_vector(t)
        }
    };
    let y_vec = match y {
        AxisSource::Host(v) => v.clone(),
        AxisSource::Gpu(h) => {
            let t = super::common::gather_tensor_from_gpu_async(h.clone(), name).await?;
            numeric_vector(t)
        }
    };
    Ok((x_vec, y_vec))
}

fn matrix_rows_are_identical(tensor: &Tensor) -> bool {
    let rows = tensor.rows;
    let cols = tensor.cols;
    if rows == 0 || cols == 0 {
        return false;
    }
    for row in 1..rows {
        for col in 0..cols {
            let idx0 = 0 + rows * col;
            let idx = row + rows * col;
            if tensor.data[idx] != tensor.data[idx0] {
                return false;
            }
        }
    }
    true
}

fn matrix_cols_are_identical(tensor: &Tensor) -> bool {
    let rows = tensor.rows;
    let cols = tensor.cols;
    if rows == 0 || cols == 0 {
        return false;
    }
    for col in 1..cols {
        for row in 0..rows {
            let idx0 = row + rows * 0;
            let idx = row + rows * col;
            if tensor.data[idx] != tensor.data[idx0] {
                return false;
            }
        }
    }
    true
}

pub(crate) fn extract_meshgrid_axes_from_xy_matrices(
    x: &Tensor,
    y: &Tensor,
    rows: usize,
    cols: usize,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    if x.rows != rows || x.cols != cols || y.rows != rows || y.cols != cols {
        return Err(plotting_error(
            BUILTIN_NAME,
            "surf: when X and Y are matrices, they must match the shape of Z",
        ));
    }

    // MATLAB meshgrid: X has identical rows, Y has identical columns.
    if !matrix_rows_are_identical(x) || !matrix_cols_are_identical(y) {
        return Err(plotting_error(
            BUILTIN_NAME,
            "surf: matrix X/Y inputs must be meshgrid-style coordinate matrices",
        ));
    }

    // Extract x vector (length = cols) from the first row of X.
    let mut x_vec = Vec::with_capacity(cols);
    for col in 0..cols {
        let idx = 0 + rows * col;
        x_vec.push(x.data[idx]);
    }
    // Extract y vector (length = rows) from the first column of Y.
    let mut y_vec = Vec::with_capacity(rows);
    for row in 0..rows {
        y_vec.push(y.data[row]);
    }

    // Internally, `tensor_to_surface_grid` uses (x_len, y_len) as (rows, cols) and stores the
    // grid as [x_index][y_index]. For a MATLAB meshgrid, that means we pass the *row* axis as
    // the "x" axis vector and the *column* axis as the "y" axis vector.
    Ok((y_vec, x_vec))
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::surf")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "surf",
    op_kind: GpuOpKind::Custom("plot-render"),
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
    type_resolver(string_type),
    builtin_path = "crate::builtins::plotting::surf"
)]
pub async fn surf_builtin(
    x: Value,
    y: Value,
    z: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<String> {
    let z_input = SurfaceDataInput::from_value(z, "surf")?;
    let (rows, cols) = z_input.grid_shape(BUILTIN_NAME)?;

    // Prefer a no-download path for vector-like gpuArray axes: keep X/Y on-device and pass
    // their buffers through to the GPU vertex packer. If X/Y are meshgrid matrices, we still
    // need to validate and extract axes on the host.
    let (x_axis, y_axis) = match (x, y) {
        (Value::GpuTensor(xh), Value::GpuTensor(yh))
            if is_vector_like_shape(&xh.shape) && is_vector_like_shape(&yh.shape) =>
        {
            (AxisSource::Gpu(xh), AxisSource::Gpu(yh))
        }
        (x_val, y_val) => {
            // Gather only when needed (host axis inputs, or matrix meshgrid validation).
            let x_tensor = match x_val {
                Value::GpuTensor(handle) => {
                    super::common::gather_tensor_from_gpu_async(handle, BUILTIN_NAME).await?
                }
                other => Tensor::try_from(&other)
                    .map_err(|e| plotting_error(BUILTIN_NAME, format!("surf: {e}")))?,
            };
            let y_tensor = match y_val {
                Value::GpuTensor(handle) => {
                    super::common::gather_tensor_from_gpu_async(handle, BUILTIN_NAME).await?
                }
                other => Tensor::try_from(&other)
                    .map_err(|e| plotting_error(BUILTIN_NAME, format!("surf: {e}")))?,
            };
            if is_vector_like(&x_tensor) && is_vector_like(&y_tensor) {
                (
                    AxisSource::Host(numeric_vector(x_tensor)),
                    AxisSource::Host(numeric_vector(y_tensor)),
                )
            } else {
                let (x_vec, y_vec) =
                    extract_meshgrid_axes_from_xy_matrices(&x_tensor, &y_tensor, rows, cols)?;
                (AxisSource::Host(x_vec), AxisSource::Host(y_vec))
            }
        }
    };

    let style = Arc::new(parse_surface_style_args(
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
    )?);
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
                BUILTIN_NAME, &x_axis, &y_axis, &z_gpu, min_z, max_z,
            )
            .await
            {
                Ok(surface) => surface,
                Err(err) => {
                    warn!("surf GPU path unavailable: {err}");
                    let (x_host, y_host) =
                        gather_axes_for_cpu_fallback(&x_axis, &y_axis, BUILTIN_NAME).await?;
                    build_surface_cpu(&z_input, x_host, y_host).await?
                }
            },
            Err(err) => {
                warn!("surf GPU bounds unavailable: {err}");
                let (x_host, y_host) =
                    gather_axes_for_cpu_fallback(&x_axis, &y_axis, BUILTIN_NAME).await?;
                build_surface_cpu(&z_input, x_host, y_host).await?
            }
        }
    } else {
        let (x_host, y_host) =
            gather_axes_for_cpu_fallback(&x_axis, &y_axis, BUILTIN_NAME).await?;
        build_surface_cpu(&z_input, x_host, y_host).await?
    };

    style.apply_to_plot(&mut surface);
    let mut surface = Some(surface);
    let rendered = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let surface = surface.take().expect("surf plot consumed once");
        figure.add_surface_plot_on_axes(surface, axes);
        Ok(())
    })?;
    Ok(rendered)
}

async fn build_surface_cpu(
    z_input: &SurfaceDataInput,
    x_axis: Vec<f64>,
    y_axis: Vec<f64>,
) -> BuiltinResult<SurfacePlot> {
    let z_tensor = match z_input.clone() {
        SurfaceDataInput::Host(tensor) => tensor,
        SurfaceDataInput::Gpu(handle) => {
            super::common::gather_tensor_from_gpu_async(handle, BUILTIN_NAME).await?
        }
    };
    let grid = tensor_to_surface_grid(
        z_tensor,
        x_axis.len(),
        y_axis.len(),
        BUILTIN_NAME,
    )?;
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

pub(crate) async fn build_surface_gpu_plot(
    name: &'static str,
    x_axis: &[f64],
    y_axis: &[f64],
    z: &GpuTensorHandle,
) -> BuiltinResult<SurfacePlot> {
    let (min_z, max_z) = super::gpu_helpers::axis_bounds_async(z, name).await?;
    build_surface_gpu_plot_with_bounds_async(
        name,
        &AxisSource::Host(x_axis.to_vec()),
        &AxisSource::Host(y_axis.to_vec()),
        z,
        min_z,
        max_z,
    )
    .await
}

pub(crate) async fn build_surface_gpu_plot_with_bounds_async(
    name: &'static str,
    x_axis: &AxisSource,
    y_axis: &AxisSource,
    z: &GpuTensorHandle,
    min_z: f32,
    max_z: f32,
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
                expected_len,
                x_len,
                y_len
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

    let color_table = build_color_lut(ColorMap::Parula, 512, 1.0);
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
                    format!("{name}: X axis length mismatch (expected {x_len}, got {})", exported.len),
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
                    format!("{name}: Y axis length mismatch (expected {y_len}, got {})", exported.len),
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
    let lod = compute_surface_lod(x_axis.len(), y_axis.len(), extent_hint);
    let params = runmat_plot::gpu::surface::SurfaceGpuParams {
        min_z,
        max_z,
        alpha: 1.0,
        flatten_z: false,
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
    let mut surface = SurfacePlot::from_gpu_buffer(
        x_len,
        y_len,
        gpu_vertices,
        vertex_count,
        bounds,
    );
    surface.colormap = ColorMap::Parula;
    surface.shading_mode = ShadingMode::Smooth;
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
        let res = futures::executor::block_on(surf_builtin(
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0])),
            Value::Tensor(Tensor {
                data: vec![0.0],
                shape: vec![1],
                rows: 1,
                cols: 1,
                dtype: runmat_builtins::NumericDType::F64,
            }),
            Vec::new(),
        ));
        assert!(res.is_err());
    }

    #[test]
    fn surf_type_is_string() {
        assert_eq!(
            string_type(
                &[Type::tensor(), Type::tensor(), Type::tensor()],
                &ResolveContext::new(Vec::new())
            ),
            Type::String
        );
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

        let (x_axis, y_axis) =
            extract_meshgrid_axes_from_xy_matrices(&x, &y, z.rows, z.cols).expect("axes");
        assert_eq!(x_axis.len(), 2);
        assert_eq!(y_axis.len(), 3);

        // With extracted axes, Z should validate and reshape into a surface grid.
        let grid = tensor_to_surface_grid(z, x_axis.len(), y_axis.len(), BUILTIN_NAME);
        assert!(
            grid.is_ok(),
            "expected Z to be compatible with extracted axes"
        );
    }
}
