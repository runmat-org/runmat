use runmat_plot::plots::ContourPlot;

use crate::builtins::plotting::common::{
    gather_tensor_from_gpu_async, tensor_to_surface_grid, SurfaceDataInput,
};
use crate::builtins::plotting::contour::{
    build_contour_gpu_plot, build_contour_plot, ContourLevelSpec, ContourLineColor,
};
use crate::builtins::plotting::op_common::surface_inputs::{axis_sources_to_host, AxisSource};
use crate::BuiltinResult;

pub async fn contour_for_surface_input(
    builtin: &'static str,
    x_axis: &[f64],
    y_axis: &[f64],
    z_input: &SurfaceDataInput,
    z_gpu: Option<runmat_accelerate_api::GpuTensorHandle>,
    contour_map: runmat_plot::plots::ColorMap,
    base_z: f32,
    level_spec: &ContourLevelSpec,
) -> BuiltinResult<ContourPlot> {
    if let Some(z_gpu) = z_gpu {
        if let Ok(contour) = build_contour_gpu_plot(
            builtin,
            x_axis,
            y_axis,
            &z_gpu,
            contour_map,
            base_z,
            level_spec,
            &ContourLineColor::Auto,
        ) {
            return Ok(contour);
        }
    }
    let z_tensor = match z_input {
        SurfaceDataInput::Host(t) => t.clone(),
        SurfaceDataInput::Gpu(h) => gather_tensor_from_gpu_async(h.clone(), builtin).await?,
    };
    let grid = tensor_to_surface_grid(z_tensor, x_axis.len(), y_axis.len(), builtin)?;
    build_contour_plot(
        builtin,
        x_axis,
        y_axis,
        &grid,
        contour_map,
        base_z,
        level_spec,
        &ContourLineColor::Auto,
    )
}

pub async fn contour_for_surface_axes_input(
    builtin: &'static str,
    x_axis: &AxisSource,
    y_axis: &AxisSource,
    z_input: &SurfaceDataInput,
    z_gpu: Option<runmat_accelerate_api::GpuTensorHandle>,
    contour_map: runmat_plot::plots::ColorMap,
    base_z: f32,
    level_spec: &ContourLevelSpec,
) -> BuiltinResult<ContourPlot> {
    let (x_host, y_host) = axis_sources_to_host(x_axis, y_axis, builtin).await?;
    contour_for_surface_input(
        builtin,
        &x_host,
        &y_host,
        z_input,
        z_gpu,
        contour_map,
        base_z,
        level_spec,
    )
    .await
}
