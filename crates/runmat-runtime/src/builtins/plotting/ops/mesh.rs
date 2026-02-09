//! MATLAB-compatible `mesh` builtin.

use log::warn;
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode, SurfacePlot};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

use super::common::{numeric_vector, tensor_to_surface_grid, SurfaceDataInput};
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_surface_style_args, SurfaceStyleDefaults};
use super::surf::{extract_meshgrid_axes_from_xy_matrices, is_vector_like_shape, AxisSource};
use crate::builtins::plotting::type_resolvers::string_type;
use crate::BuiltinResult;
use std::sync::Arc;

const BUILTIN_NAME: &str = "mesh";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::mesh")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "mesh",
    op_kind: GpuOpKind::Custom("plot-render"),
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
    type_resolver(string_type),
    builtin_path = "crate::builtins::plotting::mesh"
)]
pub async fn mesh_builtin(
    x: Value,
    y: Value,
    z: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<String> {
    let z_input = SurfaceDataInput::from_value(z, "mesh")?;
    let (rows, cols) = z_input.grid_shape(BUILTIN_NAME)?;

    // Match surf semantics: keep vector-like gpuArray axes on-device when possible; otherwise
    // gather to validate meshgrid matrix inputs and extract axis vectors.
    let (x_axis, y_axis) = match (x, y) {
        (Value::GpuTensor(xh), Value::GpuTensor(yh))
            if is_vector_like_shape(&xh.shape) && is_vector_like_shape(&yh.shape) =>
        {
            (AxisSource::Gpu(xh), AxisSource::Gpu(yh))
        }
        (x_val, y_val) => {
            let x_tensor = match x_val {
                Value::GpuTensor(handle) => {
                    super::common::gather_tensor_from_gpu_async(handle, BUILTIN_NAME).await?
                }
                other => Tensor::try_from(&other)
                    .map_err(|e| plotting_error(BUILTIN_NAME, format!("mesh: {e}")))?,
            };
            let y_tensor = match y_val {
                Value::GpuTensor(handle) => {
                    super::common::gather_tensor_from_gpu_async(handle, BUILTIN_NAME).await?
                }
                other => Tensor::try_from(&other)
                    .map_err(|e| plotting_error(BUILTIN_NAME, format!("mesh: {e}")))?,
            };

            if x_tensor.data.is_empty() || y_tensor.data.is_empty() {
                return Err(plotting_error(
                    BUILTIN_NAME,
                    "mesh: axis vectors must be non-empty",
                ));
            }

            if x_tensor.rows == 1 || x_tensor.cols == 1 {
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
    )?);
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
            )
            .await
            {
                Ok(surface_gpu) => surface_gpu,
                Err(err) => {
                    warn!("mesh GPU path unavailable: {err}");
                    build_mesh_cpu(&z_input, &x_axis, &y_axis).await?
                }
            },
            Err(err) => {
                warn!("mesh GPU bounds unavailable: {err}");
                build_mesh_cpu(&z_input, &x_axis, &y_axis).await?
            }
        }
    } else {
        build_mesh_cpu(&z_input, &x_axis, &y_axis).await?
    };

    surface = surface
        .with_colormap(ColorMap::Turbo)
        .with_wireframe(true)
        .with_shading(ShadingMode::Faceted);
    style.apply_to_plot(&mut surface);

    let mut surface_opt = Some(surface);
    let rendered = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let surface = surface_opt.take().expect("mesh plot consumed exactly once");
        figure.add_surface_plot_on_axes(surface, axes);
        Ok(())
    })?;
    Ok(rendered)
}

async fn build_mesh_cpu(
    z_input: &SurfaceDataInput,
    x_axis: &AxisSource,
    y_axis: &AxisSource,
) -> BuiltinResult<SurfacePlot> {
    let x_host = match x_axis {
        AxisSource::Host(v) => v.clone(),
        AxisSource::Gpu(h) => {
            let t = super::common::gather_tensor_from_gpu_async(h.clone(), BUILTIN_NAME).await?;
            numeric_vector(t)
        }
    };
    let y_host = match y_axis {
        AxisSource::Host(v) => v.clone(),
        AxisSource::Gpu(h) => {
            let t = super::common::gather_tensor_from_gpu_async(h.clone(), BUILTIN_NAME).await?;
            numeric_vector(t)
        }
    };
    let z_tensor = match z_input {
        SurfaceDataInput::Host(t) => t.clone(),
        SurfaceDataInput::Gpu(h) => super::common::gather_tensor_from_gpu_async(h.clone(), BUILTIN_NAME).await?,
    };
    let grid = tensor_to_surface_grid(z_tensor, x_host.len(), y_host.len(), BUILTIN_NAME)?;
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
        let res = futures::executor::block_on(mesh_builtin(
            Value::Tensor(tensor_from(&[0.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
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
    fn mesh_type_is_string() {
        assert_eq!(
            string_type(
                &[Type::tensor(), Type::tensor(), Type::tensor()],
                &ResolveContext::new(Vec::new())
            ),
            Type::String
        );
    }
}
