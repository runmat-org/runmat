//! Exact physical admission for the WGPU `fspecial` constructor hook.

use super::*;
use runmat_accelerate_api::ProviderFspecialFilter;

#[cfg(test)]
mod tests;

#[cfg(feature = "wgpu")]
const FSPECIAL_F32_UNIFORM_BYTES: u64 =
    std::mem::size_of::<crate::backend::wgpu::params::FspecialParamsF32>() as u64;
#[cfg(feature = "wgpu")]
const FSPECIAL_F64_UNIFORM_BYTES: u64 =
    std::mem::size_of::<crate::backend::wgpu::params::FspecialParamsF64>() as u64;

pub(super) fn append(snapshot: &mut ProviderCapabilitySnapshot, wgpu: bool) {
    if wgpu {
        snapshot.operations.push(ProviderCapabilityOperation {
            identity: ProviderOperationIdentity::new("fspecial"),
            family: ProviderOperationFamily::Library,
        });
    }
}

pub(super) fn validate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    if query.operation.0 != "fspecial" {
        return None;
    }
    let reject = |code, detail| ProviderFeasibility::rejected(code, detail);
    if !wgpu {
        // The in-process hook is host-backed and has no provider operation
        // completion receipt. Do not present its execution as a WGPU route.
        return Some(reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "fspecial.in_process_unobserved",
        ));
    }
    if query.family != ProviderOperationFamily::Library
        || !query.inputs.is_empty()
        || query.outputs.len() != 1
        || query.workload.requested_outputs != Some(1)
    {
        return Some(reject(
            ProviderRejectionCode::InvalidShape,
            "fspecial.roles",
        ));
    }
    let Some(filter) = query.workload.fspecial.as_ref() else {
        return Some(reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "fspecial.filter_unknown",
        ));
    };
    let supported = match filter {
        ProviderFspecialFilter::Average { rows, cols } => *rows != 0 && *cols != 0,
        ProviderFspecialFilter::Gaussian {
            rows,
            cols,
            sigma_bits,
        } => {
            let sigma = f64::from_bits(*sigma_bits);
            *rows != 0 && *cols != 0 && sigma.is_finite() && sigma > 0.0
        }
        ProviderFspecialFilter::Laplacian { alpha_bits }
        | ProviderFspecialFilter::Unsharp { alpha_bits } => f64::from_bits(*alpha_bits).is_finite(),
        ProviderFspecialFilter::Prewitt | ProviderFspecialFilter::Sobel => true,
        _ => false,
    };
    if !supported {
        return Some(reject(
            ProviderRejectionCode::UnsupportedOperation,
            "fspecial.filter_route",
        ));
    }
    let Some(shape) = filter.output_shape() else {
        return Some(reject(
            ProviderRejectionCode::InvalidShape,
            "fspecial.shape_unknown",
        ));
    };
    let output = &query.outputs[0];
    let element = ProviderElementType::from(provider.precision());
    if output.shape.as_slice() != shape.as_slice()
        || output.element_type != element
        || output.storage != ProviderStorage::DenseReal
        || output.layout != ProviderLayout::ColumnMajorContiguous
        || output.residency != ProviderResidency::Device
    {
        return Some(reject(
            ProviderRejectionCode::InvalidShape,
            "fspecial.output_contract",
        ));
    }
    let Some(elements) = shape[0].checked_mul(shape[1]) else {
        return Some(reject(
            ProviderRejectionCode::ResourceLimit,
            "fspecial.output_overflow",
        ));
    };
    if elements == 0 || elements > u64::from(u32::MAX) || usize::try_from(elements).is_err() {
        return Some(reject(
            ProviderRejectionCode::ResourceLimit,
            "fspecial.dispatch_extent",
        ));
    }
    let Some(output_bytes) = elements.checked_mul(element.byte_width()) else {
        return Some(reject(
            ProviderRejectionCode::ResourceLimit,
            "fspecial.output_bytes",
        ));
    };
    let Some(geometry) = provider.convolution_geometry() else {
        return Some(reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "fspecial.adapter_geometry",
        ));
    };
    if geometry.workgroup_size == 0
        || geometry.max_workgroups_per_dimension == 0
        || elements.div_ceil(u64::from(geometry.workgroup_size))
            > u64::from(geometry.max_workgroups_per_dimension)
    {
        return Some(reject(
            ProviderRejectionCode::ResourceLimit,
            "fspecial.dispatch_limit",
        ));
    }
    if output_bytes > geometry.max_buffer_bytes || output_bytes > geometry.max_binding_bytes {
        return Some(reject(
            ProviderRejectionCode::ResourceLimit,
            "fspecial.output_binding",
        ));
    }
    let uniform_bytes = match provider.precision() {
        ProviderPrecision::F32 => FSPECIAL_F32_UNIFORM_BYTES,
        ProviderPrecision::F64 => FSPECIAL_F64_UNIFORM_BYTES,
    };
    if uniform_bytes > geometry.max_buffer_bytes {
        return Some(reject(
            ProviderRejectionCode::ResourceLimit,
            "fspecial.uniform_allocation",
        ));
    }
    let Some(peak_bytes) = output_bytes.checked_add(uniform_bytes) else {
        return Some(reject(
            ProviderRejectionCode::ResourceLimit,
            "fspecial.peak_overflow",
        ));
    };
    Some(ProviderFeasibility::supported(ProviderResourceEstimate {
        transient_bytes: Some(peak_bytes),
        output_bytes: Some(output_bytes),
        dispatches: Some(1),
    }))
}

#[cfg(feature = "wgpu")]
pub(crate) fn observation_capability(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
) -> runmat_accelerate_api::observation::ProviderObservationCapability {
    use runmat_accelerate_api::observation::{
        ProviderObservationCapability, ProviderObservationOverhead,
    };
    let Some(ProviderFeasibility::Supported { estimate }) = validate(provider, true, query) else {
        return ProviderObservationCapability::Unsupported;
    };
    let Some(submissions) = estimate.dispatches else {
        return ProviderObservationCapability::Unsupported;
    };
    let Some(requested_peak_buffer_bytes) = u64::from(submissions)
        .checked_mul(crate::backend::wgpu::dispatch::observation::FENCE_REQUESTED_BYTES)
    else {
        return ProviderObservationCapability::Unsupported;
    };
    ProviderObservationCapability::Supported(ProviderObservationOverhead {
        submissions,
        requested_peak_buffer_bytes,
        clear_commands: submissions,
        map_callbacks: submissions,
        wgpu_compute_query: crate::backend::wgpu::dispatch::observation::prospective_encoder_query(
            submissions,
            submissions,
        ),
    })
}
