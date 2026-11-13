//! Required contour overlay of an already prepared surface; no independent retry.
use crate::builtins::plotting::contour::{
    build_contour_gpu_plot_prepared, ContourLevelSpec, ContourLineColor, ContourZMode,
};
use crate::builtins::plotting::op_common::surface_inputs::{AxisSource, PreparedSurfaceGpu};
use crate::BuiltinResult;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_plot::plots::ContourPlot;

pub(crate) fn contour_for_surface_axes_input(
    builtin: &'static str,
    x_axis: &AxisSource,
    y_axis: &AxisSource,
    z: &GpuTensorHandle,
    prepared: &mut PreparedSurfaceGpu<'_>,
    contour_map: runmat_plot::plots::ColorMap,
    base_z: f32,
    level_spec: &ContourLevelSpec,
) -> BuiltinResult<ContourPlot> {
    build_contour_gpu_plot_prepared(
        builtin,
        x_axis,
        y_axis,
        z,
        prepared,
        contour_map,
        base_z,
        level_spec,
        &ContourLineColor::Auto,
        ContourZMode::Base,
    )
}
