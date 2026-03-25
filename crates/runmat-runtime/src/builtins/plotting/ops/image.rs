use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use log::warn;
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode, SurfacePlot};

use super::common::{tensor_to_surface_grid, SurfaceDataInput};
use super::op_common::surface_inputs::{
    axis_sources_from_xy_values, axis_sources_to_host, parse_surface_call_args, AxisSource,
};
use super::state::{color_limits_snapshot, render_active_plot, PlotRenderOptions};
use super::style::{parse_surface_style_args, SurfaceStyleDefaults};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

const BUILTIN_NAME: &str = "image";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::image")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "image",
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
    notes: "image is a plotting sink; indexed and truecolor gpuArray inputs may remain on device when a shared WGPU context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::image")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "image",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "image terminates fusion graphs and performs rendering.",
};

#[runtime_builtin(
    name = "image",
    category = "plotting",
    summary = "Render MATLAB-compatible image plots on the modern surface path.",
    keywords = "image,plotting,imshow,colormap",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::image"
)]
pub async fn image_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (x, y, c, rest) = parse_surface_call_args(args, BUILTIN_NAME)?;
    let (rows, cols, kind) = classify_image_input(&c, BUILTIN_NAME).await?;
    let (x_axis, y_axis) = axis_sources_from_xy_values(x, y, rows, cols, BUILTIN_NAME).await?;
    let defaults =
        SurfaceStyleDefaults::new(ColorMap::Parula, ShadingMode::None, false, 1.0, true, false);
    let style = Arc::new(parse_surface_style_args(BUILTIN_NAME, &rest, defaults)?);
    let color_limits = color_limits_snapshot();

    let mut surface = match kind {
        ImageInputKind::TrueColorHost(tensor) => {
            let (x_host, y_host) = axis_sources_to_host(&x_axis, &y_axis, BUILTIN_NAME).await?;
            build_truecolor_image_surface(tensor, x_host, y_host)?
        }
        ImageInputKind::TrueColorGpu(handle, channels) => {
            build_truecolor_image_surface_gpu(&handle, &x_axis, &y_axis, rows, cols, channels)?
        }
        ImageInputKind::Indexed(input) => {
            build_indexed_image_surface(&input, &x_axis, &y_axis, style.colormap, color_limits)
                .await?
        }
    };

    surface = surface.with_flatten_z(true).with_image_mode(true);
    let mut surface = Some(surface);
    let plot_index_out = Rc::new(RefCell::new(None));
    let plot_index_slot = Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            title: "Image",
            x_label: "X",
            y_label: "Y",
            axis_equal: true,
            ..Default::default()
        },
        move |figure, axes| {
            let plot_index = figure.add_surface_plot_on_axes(
                surface.take().expect("image plot consumed once"),
                axes,
            );
            *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            Ok(())
        },
    );
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = crate::builtins::plotting::state::register_image_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(err);
    }
    Ok(handle)
}

enum ImageInputKind {
    Indexed(SurfaceDataInput),
    TrueColorHost(Tensor),
    TrueColorGpu(runmat_accelerate_api::GpuTensorHandle, u32),
}

async fn classify_image_input(
    value: &Value,
    builtin: &'static str,
) -> crate::BuiltinResult<(usize, usize, ImageInputKind)> {
    match value {
        Value::GpuTensor(handle) if handle.shape.len() >= 3 => {
            let channels = handle.shape.get(2).copied().unwrap_or(1);
            if channels != 3 && channels != 4 {
                return Err(crate::builtins::plotting::plotting_error(
                    builtin,
                    format!("{builtin}: truecolor image data must have 3 or 4 channels"),
                ));
            }
            let rows = handle.shape.first().copied().unwrap_or(0);
            let cols = handle.shape.get(1).copied().unwrap_or(0);
            Ok((rows, cols, ImageInputKind::TrueColorGpu(handle.clone(), channels as u32)))
        }
        _ => {
            let tensor = Tensor::try_from(value)
                .map_err(|e| crate::builtins::plotting::plotting_error(builtin, format!("{builtin}: {e}")))?;
            if tensor.shape.len() >= 3 {
                let (rows, cols) = truecolor_shape(&tensor, builtin)?;
                Ok((rows, cols, ImageInputKind::TrueColorHost(tensor)))
            } else {
                let input = SurfaceDataInput::from_value(value.clone(), builtin)?;
                let (rows, cols) = input.grid_shape(builtin)?;
                Ok((rows, cols, ImageInputKind::Indexed(input)))
            }
        }
    }
}

fn build_truecolor_image_surface_gpu(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    x_axis: &AxisSource,
    y_axis: &AxisSource,
    rows: usize,
    cols: usize,
    channels: u32,
) -> crate::BuiltinResult<SurfacePlot> {
    let context = super::gpu_helpers::ensure_shared_wgpu_context(BUILTIN_NAME)?;
    let image_ref = runmat_accelerate_api::export_wgpu_buffer(handle)
        .ok_or_else(|| crate::builtins::plotting::plotting_error(BUILTIN_NAME, "image: unable to export truecolor GPU image"))?;
    let scalar = runmat_plot::gpu::ScalarType::from_is_f64(
        image_ref.precision == runmat_accelerate_api::ProviderPrecision::F64,
    );
    let mut host_x_f32 = None;
    let mut host_y_f32 = None;
    let mut host_x_f64 = None;
    let mut host_y_f64 = None;
    let x_data = axis_source_to_gpu_axis(x_axis, scalar, &mut host_x_f32, &mut host_x_f64, BUILTIN_NAME)?;
    let y_data = axis_source_to_gpu_axis(y_axis, scalar, &mut host_y_f32, &mut host_y_f64, BUILTIN_NAME)?;
    let gpu_vertices = runmat_plot::gpu::image::pack_truecolor_vertices(
        &context.device,
        &context.queue,
        &runmat_plot::gpu::image::TrueColorImageGpuInputs {
            x_axis: x_data,
            y_axis: y_data,
            image_buffer: image_ref.buffer.clone(),
            rows: rows as u32,
            cols: cols as u32,
            channels,
            scalar,
        },
    )
    .map_err(|e| crate::builtins::plotting::plotting_error(BUILTIN_NAME, format!("image: failed to build GPU truecolor vertices: {e}")))?;
    let (x_host, y_host) = futures::executor::block_on(axis_sources_to_host(x_axis, y_axis, BUILTIN_NAME))?;
    let bounds = runmat_plot::core::BoundingBox::new(
        glam::Vec3::new(
            x_host.first().copied().unwrap_or(0.0) as f32,
            y_host.first().copied().unwrap_or(0.0) as f32,
            0.0,
        ),
        glam::Vec3::new(
            x_host.last().copied().unwrap_or(0.0) as f32,
            y_host.last().copied().unwrap_or(0.0) as f32,
            0.0,
        ),
    );
    let mut surface = SurfacePlot::from_gpu_buffer(rows, cols, gpu_vertices, rows * cols, bounds)
        .with_flatten_z(true)
        .with_image_mode(true)
        .with_shading(ShadingMode::None);
    surface.x_data = x_host;
    surface.y_data = y_host;
    Ok(surface)
}

fn axis_source_to_gpu_axis<'a>(
    source: &'a AxisSource,
    scalar: runmat_plot::gpu::ScalarType,
    host_f32: &'a mut Option<Vec<f32>>,
    host_f64: &'a mut Option<Vec<f64>>,
    builtin: &'static str,
) -> crate::BuiltinResult<runmat_plot::gpu::axis::AxisData<'a>> {
    match source {
        AxisSource::Gpu(handle) => {
            let exported = runmat_accelerate_api::export_wgpu_buffer(handle).ok_or_else(|| {
                crate::builtins::plotting::plotting_error(builtin, format!("{builtin}: unable to export GPU axis data"))
            })?;
            Ok(runmat_plot::gpu::axis::AxisData::Buffer(exported.buffer.clone()))
        }
        AxisSource::Host(values) => match scalar {
            runmat_plot::gpu::ScalarType::F32 => {
                *host_f32 = Some(values.iter().map(|v| *v as f32).collect());
                Ok(runmat_plot::gpu::axis::AxisData::F32(host_f32.as_ref().unwrap()))
            }
            runmat_plot::gpu::ScalarType::F64 => {
                *host_f64 = Some(values.clone());
                Ok(runmat_plot::gpu::axis::AxisData::F64(host_f64.as_ref().unwrap()))
            }
        },
    }
}

fn truecolor_shape(tensor: &Tensor, builtin: &'static str) -> crate::BuiltinResult<(usize, usize)> {
    let rows = tensor.shape.first().copied().unwrap_or(tensor.rows);
    let cols = tensor.shape.get(1).copied().unwrap_or(tensor.cols);
    let channels = tensor.shape.get(2).copied().unwrap_or(1);
    if rows == 0 || cols == 0 || (channels != 3 && channels != 4) {
        return Err(crate::builtins::plotting::plotting_error(
            builtin,
            format!("{builtin}: truecolor image data must be MxNx3 or MxNx4"),
        ));
    }
    let expected_len = rows * cols * channels;
    if tensor.data.len() != expected_len {
        return Err(crate::builtins::plotting::plotting_error(
            builtin,
            format!("{builtin}: truecolor image data length mismatch"),
        ));
    }
    Ok((rows, cols))
}

pub(crate) async fn build_indexed_image_surface(
    c_input: &SurfaceDataInput,
    x_axis: &super::op_common::surface_inputs::AxisSource,
    y_axis: &super::op_common::surface_inputs::AxisSource,
    colormap: ColorMap,
    color_limits: Option<(f64, f64)>,
) -> crate::BuiltinResult<SurfacePlot> {
    if let Some(c_gpu) = c_input.gpu_handle().cloned() {
        match super::gpu_helpers::axis_bounds_async(&c_gpu, BUILTIN_NAME).await {
            Ok((min_z, max_z)) => match super::surf::build_surface_gpu_plot_with_bounds_async(
                BUILTIN_NAME,
                x_axis,
                y_axis,
                &c_gpu,
                min_z,
                max_z,
                colormap,
                1.0,
                true,
            )
            .await
            {
                Ok(surface) => {
                    return Ok(surface
                        .with_flatten_z(true)
                        .with_image_mode(true)
                        .with_color_limits(color_limits));
                }
                Err(err) => warn!("image GPU path unavailable: {err}"),
            },
            Err(err) => warn!("image GPU bounds unavailable: {err}"),
        }
    }

    let (x_host, y_host) = axis_sources_to_host(x_axis, y_axis, BUILTIN_NAME).await?;
    let tensor = match c_input.clone() {
        SurfaceDataInput::Host(tensor) => tensor,
        SurfaceDataInput::Gpu(handle) => super::common::gather_tensor_from_gpu_async(handle, BUILTIN_NAME).await?,
    };
    let grid = tensor_to_surface_grid(tensor, x_host.len(), y_host.len(), BUILTIN_NAME)?;
    Ok(super::surf::build_surface(x_host, y_host, grid)?
        .with_flatten_z(true)
        .with_image_mode(true)
        .with_colormap(colormap)
        .with_shading(ShadingMode::None)
        .with_color_limits(color_limits))
}

fn build_truecolor_image_surface(
    tensor: Tensor,
    x_axis: Vec<f64>,
    y_axis: Vec<f64>,
) -> crate::BuiltinResult<SurfacePlot> {
    let rows = x_axis.len();
    let cols = y_axis.len();
    let channels = tensor.shape.get(2).copied().unwrap_or(3);
    let mut grid = vec![vec![glam::Vec4::ZERO; cols]; rows];
    for row in 0..rows {
        for col in 0..cols {
            let base = row + rows * col;
            let r = tensor.data[base] as f32;
            let g = tensor.data[base + rows * cols] as f32;
            let b = tensor.data[base + 2 * rows * cols] as f32;
            let a = if channels == 4 {
                tensor.data[base + 3 * rows * cols] as f32
            } else {
                1.0
            };
            grid[row][col] = glam::Vec4::new(r, g, b, a);
        }
    }
    let z = vec![vec![0.0; cols]; rows];
    Ok(SurfacePlot::new(x_axis, y_axis, z)
        .map_err(|e| crate::builtins::plotting::plotting_error(BUILTIN_NAME, format!("image: {e}")))?
        .with_flatten_z(true)
        .with_image_mode(true)
        .with_color_grid(grid)
        .with_shading(ShadingMode::None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run};
    use runmat_builtins::NumericDType;
    use runmat_plot::plots::PlotElement;

    fn truecolor_tensor() -> Tensor {
        Tensor {
            data: vec![
                1.0, 0.0, 0.0, 1.0,
                0.0, 1.0, 0.0, 1.0,
                0.0, 0.0, 1.0, 1.0,
            ],
            shape: vec![2, 2, 3],
            rows: 2,
            cols: 2,
            dtype: NumericDType::F64,
        }
    }

    #[test]
    fn image_truecolor_builds_image_surface_and_handle() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle = futures::executor::block_on(image_builtin(vec![Value::Tensor(truecolor_tensor())])).unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface");
        };
        assert!(surface.image_mode);
        assert!(surface.color_grid.is_some());
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap(),
            Value::String("image".into())
        );
    }
}
