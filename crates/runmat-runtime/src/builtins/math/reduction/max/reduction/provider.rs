use super::geometry::default_dimension_from_shape;
use super::*;
use runmat_accelerate_api::GpuTensorHandle;

pub(super) async fn reduction_max_gpu(
    runtime: Option<&crate::context::RuntimeContext>,
    handle: GpuTensorHandle,
    args: &ReductionArgs,
    demand: super::extrema_provider::ExtremaDemand,
) -> BuiltinResult<Option<super::extrema_provider::ExtremaResult>> {
    if args.nan_mode == ReductionNaN::Omit {
        log::trace!("max: gpu path disabled (nan_mode=omit)");
        return Ok(None);
    }
    if args.comparison != ComparisonMethod::Auto {
        log::trace!("max: gpu path disabled (comparison != auto)");
        return Ok(None);
    }
    if args.linear_index {
        log::trace!("max: gpu path disabled (linear_index=true)");
        return Ok(None);
    }
    let target_dim = match args.selection {
        DimSelection::Auto => default_dimension_from_shape(handle.shape()),
        DimSelection::Dim(dim) => dim,
        DimSelection::Vec(ref dims) if dims.len() == 1 => dims[0],
        DimSelection::All => {
            if handle.shape().len() <= 1 {
                1
            } else {
                return Ok(None);
            }
        }
        _ => return Ok(None),
    };
    if target_dim == 0 {
        return Ok(None);
    }
    // MATLAB dimensions are 1-based; `reduce_max_dim` expects zero-based.
    let zero_based = target_dim.saturating_sub(1);
    if zero_based >= handle.shape().len() {
        return Ok(None);
    }
    log::trace!(
        "max: attempting reduce_max_dim dim={} (zero_based={}) shape={:?} device_id={}",
        target_dim,
        zero_based,
        handle.shape(),
        handle.device_id()
    );
    super::extrema_provider::ExtremaKind::Max
        .reduce_dim(runtime, &handle, zero_based, demand)
        .await
}
