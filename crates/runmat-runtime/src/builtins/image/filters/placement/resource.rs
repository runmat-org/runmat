//! Filter-owned checked host plan/evaluation/conversion workspace.
//! Provider readback payload and staging are accounted by source-owner hooks.

use super::super::geometry::{filter_output_shape, ImfilterKernelPoint};
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::ImfilterOptions;
use runmat_execution::{ResourceBoundUnavailable, ResourceByteBound};

pub(in crate::builtins::image::filters) fn conditional_host_bounds(
    builtin: &str,
    image_shape: &[usize],
    kernel_shape: &[usize],
    options: &ImfilterOptions,
    image_single: bool,
    kernel_single: bool,
    prepared_kernel: bool,
) -> (ResourceByteBound, ResourceByteBound) {
    let counts = (|| {
        let output_shape =
            filter_output_shape(image_shape, kernel_shape, options.shape, builtin).ok()?;
        let count = |shape: &[usize]| {
            shape.iter().try_fold(1_u64, |total, &extent| {
                total.checked_mul(u64::try_from(extent).ok()?)
            })
        };
        Some((
            count(image_shape)?,
            count(kernel_shape)?,
            count(&output_shape)?,
            u64::try_from(image_shape.len().max(kernel_shape.len()).max(2)).ok()?,
        ))
    })();
    let Some((image_len, kernel_len, output_len, rank)) = counts else {
        return (
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound),
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound),
        );
    };
    let output_single = image_single || kernel_single;
    let result = (|| {
        let point_width = u64::try_from(std::mem::size_of::<ImfilterKernelPoint>())
            .ok()?
            .checked_add(rank.checked_mul(8)?)?;
        let rank_vectors = rank.checked_mul(16)?.checked_mul(8)?;
        let bytes = [
            kernel_len.checked_mul(point_width)?,
            rank_vectors,
            if image_single {
                image_len.checked_mul(8)?
            } else {
                0
            },
            if kernel_single {
                kernel_len.checked_mul(8)?
            } else {
                0
            },
            output_len.checked_mul(8)?,
            if output_single {
                output_len.checked_mul(12)?
            } else {
                0
            },
            if prepared_kernel {
                kernel_len
                    .checked_mul(if kernel_single { 4 } else { 8 })?
                    .checked_mul(2)?
            } else {
                0
            },
        ]
        .into_iter()
        .try_fold(0_u64, |total, bytes| total.checked_add(bytes))?;
        let output = output_len.checked_mul(if output_single { 4 } else { 8 })?;
        Some((bytes, output))
    })();
    match result {
        Some((scratch, output)) => (
            ResourceByteBound::Known(scratch),
            ResourceByteBound::Known(output),
        ),
        None => (
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound),
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound),
        ),
    }
}

/// The bound above includes the final host payload. A conditional provider
/// upload accounts that payload separately, so remove exactly its bytes.
pub(in crate::builtins::image::filters) fn host_scratch_excluding_result(
    scratch: ResourceByteBound,
    result: ResourceByteBound,
) -> BuiltinResult<ResourceByteBound> {
    match (scratch, result) {
        (ResourceByteBound::Known(scratch), ResourceByteBound::Known(result)) => scratch
            .checked_sub(result)
            .map(ResourceByteBound::Known)
            .ok_or_else(|| {
                build_runtime_error("conditional host result exceeds its workspace bound")
                    .with_identifier("RunMat:Placement:InvalidConditionalPresentation")
                    .build()
            }),
        (ResourceByteBound::Unavailable(reason), _)
        | (_, ResourceByteBound::Unavailable(reason)) => Ok(ResourceByteBound::Unavailable(reason)),
    }
}
