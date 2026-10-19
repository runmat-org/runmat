//! One retained physical preparation for a surface and its requested overlays.
use super::AxisSource;
use crate::builtins::plotting::{gpu_helpers::GpuBufferExports, plotting_error};
use crate::BuiltinResult;
use runmat_accelerate_api::GpuTensorHandle;

pub(crate) struct PreparedSurfaceGpu<'a> {
    x: &'a AxisSource,
    y: &'a AxisSource,
    z: &'a GpuTensorHandle,
    pub context: runmat_plot::SharedWgpuContext,
    pub exports: GpuBufferExports,
    pub x_bounds: (f32, f32),
    pub y_bounds: (f32, f32),
    pub z_bounds: (f32, f32),
}

impl<'a> PreparedSurfaceGpu<'a> {
    pub fn validate_host_sources(
        &self,
        name: &'static str,
        x: &[f64],
        y: &[f64],
        z: &GpuTensorHandle,
    ) -> BuiltinResult<()> {
        let same = |axis: &AxisSource, values: &[f64]| matches!(axis, AxisSource::Host(original) if original.len() == values.len() && original.as_ptr() == values.as_ptr());
        if !same(self.x, x)
            || !same(self.y, y)
            || !self.z.shares_allocation_with(z)
            || self.z.view_identity() != z.view_identity()
        {
            return Err(crate::builtins::plotting::gpu_helpers::preparation_failure(
                plotting_error(
                    name,
                    format!("{name}: prepared contour-fill sources changed"),
                ),
                name,
                true,
            ));
        }
        Ok(())
    }
    pub fn validate_sources(
        &self,
        name: &'static str,
        x: &AxisSource,
        y: &AxisSource,
        z: &GpuTensorHandle,
    ) -> BuiltinResult<()> {
        if !std::ptr::eq(self.x, x)
            || !std::ptr::eq(self.y, y)
            || !self.z.shares_allocation_with(z)
            || self.z.view_identity() != z.view_identity()
        {
            return Err(crate::builtins::plotting::gpu_helpers::preparation_failure(
                plotting_error(name, format!("{name}: prepared surface sources changed")),
                name,
                true,
            ));
        }
        Ok(())
    }
    /// All pure geometry/precision admission precedes the first export or
    /// reduction. None means the complete preparation was not attempted.
    pub async fn prepare(
        name: &'static str,
        x: &'a AxisSource,
        y: &'a AxisSource,
        z: &'a GpuTensorHandle,
    ) -> BuiltinResult<Option<Self>> {
        let x_len = x.len();
        let y_len = y.len();
        if x_len == 0 || y_len == 0 {
            return Err(plotting_error(
                name,
                format!("{name}: axis vectors must be non-empty"),
            ));
        }
        let expected = x_len
            .checked_mul(y_len)
            .ok_or_else(|| plotting_error(name, format!("{name}: grid dimensions overflowed")))?;
        if GpuBufferExports::element_count(z, name)? != expected {
            return Err(plotting_error(
                name,
                format!("{name}: Z must contain exactly {expected} elements ({x_len}×{y_len})"),
            ));
        }
        u32::try_from(x_len)
            .and_then(|_| u32::try_from(y_len))
            .map_err(|_| {
                plotting_error(
                    name,
                    format!("{name}: grid dimensions exceed the supported GPU range"),
                )
            })?;
        let mut inputs = vec![z];
        for axis in [x, y] {
            if let AxisSource::Gpu(handle) = axis {
                if GpuBufferExports::element_count(handle, name)? != axis.len() {
                    return Err(plotting_error(
                        name,
                        format!("{name}: axis shape does not match its admitted length"),
                    ));
                }
                inputs.push(handle);
            }
        }
        // The existing GPU cell packers need a 2-D cell; host image/surface
        // paths still support singleton axes. Decline before any export.
        if x_len < 2 || y_len < 2 {
            return Ok(None);
        }
        if inputs.iter().any(|handle| {
            runmat_accelerate_api::handle_precision(handle)
                != runmat_accelerate_api::handle_precision(z)
        }) {
            return Ok(None);
        }
        let Some((context, mut exports)) = GpuBufferExports::prepare(&inputs, name)? else {
            return Ok(None);
        };
        let result = async {
            let z_bounds = {
                let protected = exports.protected_handles();
                crate::builtins::plotting::gpu_helpers::axis_bounds_protecting(z, &protected, name)
                    .await?
            };
            exports.record_completed_effect();
            let mut axis_bounds = Vec::with_capacity(2);
            for axis in [x, y] {
                axis_bounds.push(match axis {
                    AxisSource::Host(values) => (
                        values
                            .iter()
                            .fold(f32::INFINITY, |acc, &v| acc.min(v as f32)),
                        values
                            .iter()
                            .fold(f32::NEG_INFINITY, |acc, &v| acc.max(v as f32)),
                    ),
                    AxisSource::Gpu(handle) => {
                        let protected = exports.protected_handles();
                        crate::builtins::plotting::gpu_helpers::axis_bounds_protecting(
                            handle, &protected, name,
                        )
                        .await?
                    }
                });
            }
            Ok((z_bounds, axis_bounds[0], axis_bounds[1]))
        }
        .await;
        let (z_bounds, x_bounds, y_bounds) = exports.complete(result, name)?;
        Ok(Some(Self {
            x,
            y,
            z,
            context,
            exports,
            x_bounds,
            y_bounds,
            z_bounds,
        }))
    }
}
