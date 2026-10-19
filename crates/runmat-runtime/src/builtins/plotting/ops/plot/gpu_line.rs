//! Shared plot/line zero-copy preparation. Absence is admitted before work;
//! operational errors never authorize host replay.
use super::{
    marker_metadata_from_appearance, plotting_error, BuiltinResult, GpuTensorHandle,
    LineAppearance, LinePlot, LineStyle,
};
use crate::builtins::plotting::gpu_helpers::GpuBufferExports;
use runmat_accelerate_api::ProviderPrecision;
use runmat_plot::{gpu::ScalarType, plots::LineGpuStyle};

pub(super) async fn prepare(
    builtin: &'static str,
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
    label: &str,
    appearance: &LineAppearance,
) -> BuiltinResult<Option<LinePlot>> {
    // Validate live inputs even when their physical class cannot render directly.
    for input in [x, y] {
        input.allocation_lease().validate_live().map_err(|error| {
            crate::runtime_error::with_provider_execution_source(
                plotting_error(builtin, format!("{builtin}: retired plotting input")),
                runmat_accelerate_api::ProviderExecutionError::from_source(error),
            )
        })?;
    }
    let (Some(x_precision), Some(y_precision)) = (
        runmat_accelerate_api::handle_precision(x),
        runmat_accelerate_api::handle_precision(y),
    ) else {
        // Native integers retain exact properties through the supported host path.
        return Ok(None);
    };
    let count = |input: &GpuTensorHandle| {
        if input.shape().contains(&0) {
            Some(0)
        } else {
            input
                .shape()
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))
        }
    };
    let x_len =
        count(x).ok_or_else(|| plotting_error(builtin, format!("{builtin}: shape overflow")))?;
    let y_len =
        count(y).ok_or_else(|| plotting_error(builtin, format!("{builtin}: shape overflow")))?;
    if x_precision != y_precision
        || x.device_id() != y.device_id()
        || x_len != y_len
        || x_len < 2
        || [x, y].iter().any(|input| {
            input.shape().iter().filter(|d| **d > 1).count() > 1
                || input.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real
                || runmat_accelerate_api::handle_is_logical(input)
        })
    {
        return Ok(None);
    }
    let len_u32 = match u32::try_from(x_len) {
        Ok(len) => len,
        Err(_) => return Ok(None),
    };
    let owner = runmat_accelerate_api::provider_for_handle(x).ok_or_else(|| {
        plotting_error(
            builtin,
            format!("{builtin}: no provider owns plotting input"),
        )
    })?;
    let Some(runmat_accelerate_api::AccelContextHandle::Wgpu(context)) =
        owner.export_context(runmat_accelerate_api::AccelContextKind::Plotting)
    else {
        return Ok(None);
    };
    crate::builtins::plotting::context::install_wgpu_context(&context);
    let context = runmat_plot::shared_wgpu_context().ok_or_else(|| {
        plotting_error(
            builtin,
            format!("{builtin}: shared plotting context unavailable"),
        )
    })?;
    let mut exports = GpuBufferExports::default();
    let Some(x_ref) = exports.export(x, builtin)? else {
        return Ok(None);
    };
    let Some(y_ref) = exports.export(y, builtin)? else {
        return Ok(None);
    };
    if !context.access.same_device(x_ref.access()) || !context.access.same_device(y_ref.access()) {
        return Err(crate::runtime_error::with_provider_execution_source(
            plotting_error(
                builtin,
                format!("{builtin}: GPU export belongs to another rendering device"),
            ),
            runmat_accelerate_api::ProviderExecutionError::from(anyhow::anyhow!(
                "rendering device does not own exported coordinates"
            ))
            .after_effect(),
        ));
    }
    let scalar = ScalarType::from_is_f64(x_precision == ProviderPrecision::F64);
    let inputs = runmat_plot::gpu::line::LineGpuInputs {
        x_buffer: x_ref.into(),
        y_buffer: y_ref.into(),
        len: len_u32,
        scalar,
    };
    let bounds = exports.complete(
        super::super::gpu_helpers::gpu_xy_bounds_async(x, y, builtin).await,
        builtin,
    )?;
    let marker_meta = marker_metadata_from_appearance(appearance);

    let marker_gpu_vertices = if let Some(marker) = marker_meta.as_ref() {
        let marker_params = runmat_plot::gpu::line::LineGpuParams {
            color: marker.face_color,
            half_width_px: 0.0,
            viewport_width_px: 1.0,
            viewport_height_px: 1.0,
            x_min: 0.0,
            x_span: 1.0,
            y_min: 0.0,
            y_span: 1.0,
            x_log: false,
            y_log: false,
            line_style: LineStyle::Solid,
            marker_size: marker.size.max(1.0),
        };
        Some(
            context
                .access
                .with_shared(|emission| {
                    runmat_plot::gpu::line::pack_marker_vertices_from_xy(
                        emission,
                        &inputs,
                        &marker_params,
                    )
                })
                .map_err(|error| {
                    crate::runtime_error::with_provider_execution_source(
                        plotting_error(
                            builtin,
                            format!("{builtin}: marker emission unavailable: {error}"),
                        ),
                        runmat_accelerate_api::ProviderExecutionError::from_source(error)
                            .after_effect(),
                    )
                })?
                .map_err(|e| {
                    crate::runtime_error::with_provider_execution_source(
                        plotting_error(
                            builtin,
                            format!("{builtin}: failed to build marker vertices: {e}"),
                        ),
                        runmat_accelerate_api::ProviderExecutionError::from(anyhow::anyhow!(e))
                            .after_effect(),
                    )
                })?,
        )
    } else {
        None
    };

    let gpu_style = LineGpuStyle {
        color: appearance.color,
        line_width: appearance.line_width,
        line_style: appearance.line_style,
        marker: marker_meta.clone(),
        handle_visibility: appearance.handle_visibility.clone(),
    };
    let mut plot = LinePlot::from_gpu_xy(inputs, gpu_style, bounds, marker_gpu_vertices);
    plot = plot
        .with_label(label)
        .with_handle_visibility(appearance.handle_visibility.clone());
    if let Some(visible) = appearance.visible {
        plot.set_visible(visible);
    }
    Ok(Some(plot))
}
