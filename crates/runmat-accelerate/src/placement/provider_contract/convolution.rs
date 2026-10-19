//! Physical convolution hook admission. This does not choose host versus device.

use super::*;
use runmat_accelerate_api::{
    ProviderConvOrientation, ProviderConvolutionForm, ProviderConvolutionWorkload,
};
use runmat_types::ConvolutionMode;

#[cfg(test)]
mod tests;
mod resources;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConvolutionIdentity {
    Vector,
    Matrix,
}

impl ConvolutionIdentity {
    const fn name(self) -> &'static str {
        match self {
            Self::Vector => "conv",
            Self::Matrix => "conv2",
        }
    }

    fn from_query(query: &ProviderFeasibilityQuery) -> Option<Self> {
        [Self::Vector, Self::Matrix]
            .into_iter()
            .find(|identity| query.operation.0 == identity.name())
    }
}

pub(super) fn append(snapshot: &mut ProviderCapabilitySnapshot, wgpu: bool) {
    let identities = if wgpu {
        &[ConvolutionIdentity::Vector][..]
    } else {
        &[ConvolutionIdentity::Vector, ConvolutionIdentity::Matrix][..]
    };
    for identity in identities {
        snapshot.operations.push(ProviderCapabilityOperation {
            identity: ProviderOperationIdentity::new(identity.name()),
            family: ProviderOperationFamily::Library,
        });
    }
}

pub(super) fn validate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    let identity = ConvolutionIdentity::from_query(query)?;
    Some(work(provider, wgpu, identity, query))
}

#[cfg(feature = "wgpu")]
pub(crate) fn observation_capability(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
) -> runmat_accelerate_api::observation::ProviderObservationCapability {
    use runmat_accelerate_api::observation::{
        ProviderObservationCapability, ProviderObservationOverhead,
    };
    if query.operation.0 != "conv" {
        return ProviderObservationCapability::Unsupported;
    }
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

fn reject(code: ProviderRejectionCode, detail: &'static str) -> ProviderFeasibility {
    ProviderFeasibility::rejected(code, detail)
}

fn work(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    identity: ConvolutionIdentity,
    query: &ProviderFeasibilityQuery,
) -> ProviderFeasibility {
    if wgpu && identity == ConvolutionIdentity::Matrix {
        // WGPU's conv2d hook is an Unsupported stub, not a latent GPU kernel.
        return reject(
            ProviderRejectionCode::UnsupportedOperation,
            "conv2.wgpu_hook_absent",
        );
    }
    if query.family != ProviderOperationFamily::Library
        || query.inputs.len() != 2
        || query.outputs.len() > 1
        || query
            .workload
            .requested_outputs
            .is_some_and(|count| count != 1)
    {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "convolution.operand_or_family",
        );
    }
    let Some(route) = query.workload.convolution else {
        return reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "convolution.route_unknown",
        );
    };
    if !matches!(
        (identity, route.form),
        (
            ConvolutionIdentity::Vector,
            ProviderConvolutionForm::Vector { .. }
        ) | (ConvolutionIdentity::Matrix, ProviderConvolutionForm::Matrix)
    ) {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "convolution.form_identity_mismatch",
        );
    }
    let expected = ProviderElementType::from(provider.precision());
    for input in &query.inputs {
        if input.storage != ProviderStorage::DenseReal {
            return reject(
                ProviderRejectionCode::UnsupportedStorage,
                "convolution.storage",
            );
        }
        if input.layout != ProviderLayout::ColumnMajorContiguous
            || input.residency != ProviderResidency::Device
        {
            return reject(
                ProviderRejectionCode::UnsupportedLayout,
                "convolution.layout_or_residency",
            );
        }
        if input.element_type != expected {
            return reject(
                ProviderRejectionCode::UnsupportedElementType,
                "convolution.precision",
            );
        }
    }
    let Some((shape, visits, start)) = output_geometry(identity, route, &query.inputs, wgpu) else {
        return reject(ProviderRejectionCode::InvalidShape, "convolution.geometry");
    };
    if query
        .workload
        .elements
        .is_some_and(|declared| Some(declared) != query.inputs[0].checked_element_count())
    {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "convolution.input_elements_mismatch",
        );
    }
    if shape.iter().any(|extent| usize::try_from(*extent).is_err()) {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "convolution.platform_extent",
        );
    }
    let Some(output_elements) = shape.iter().try_fold(1_u64, |n, d| n.checked_mul(*d)) else {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "convolution.output_overflow",
        );
    };
    if usize::try_from(output_elements).is_err() {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "convolution.platform_allocation",
        );
    }
    let Some(output_bytes) = output_elements.checked_mul(expected.byte_width()) else {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "convolution.output_bytes_overflow",
        );
    };
    if query
        .workload
        .scalar_work_units
        .is_some_and(|declared| declared != visits)
    {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "convolution.work_mismatch",
        );
    }
    if let Some(output) = query.outputs.first() {
        if output.shape != shape
            || output.storage != ProviderStorage::DenseReal
            || output.element_type != expected
            || output.layout != ProviderLayout::ColumnMajorContiguous
            || output.residency != ProviderResidency::Device
        {
            return reject(
                ProviderRejectionCode::InvalidShape,
                "convolution.output_contract",
            );
        }
    }
    if wgpu {
        let Some(geometry) = provider.convolution_geometry() else {
            return reject(
                ProviderRejectionCode::UnknownResourceEstimate,
                "convolution.shader_geometry_unknown",
            );
        };
        let Some(input_bytes) = query
            .inputs
            .iter()
            .map(ProviderRepresentation::checked_byte_len)
            .collect::<Option<Vec<_>>>()
        else {
            return reject(
                ProviderRejectionCode::InvalidShape,
                "convolution.input_bytes_overflow",
            );
        };
        let input_elems = query
            .inputs
            .iter()
            .map(ProviderRepresentation::checked_element_count)
            .collect::<Option<Vec<_>>>();
        let Some(input_elems) = input_elems else {
            return reject(
                ProviderRejectionCode::InvalidShape,
                "convolution.input_overflow",
            );
        };
        if input_elems
            .into_iter()
            .chain(std::iter::once(output_elements))
            .any(|n| n > u64::from(u32::MAX))
            || start > u64::from(u32::MAX)
        {
            return reject(
                ProviderRejectionCode::ResourceLimit,
                "convolution.shader_index_limit",
            );
        }
        if input_bytes
            .into_iter()
            .chain(std::iter::once(output_bytes))
            .any(|bytes| {
                let physical = bytes.max(expected.byte_width());
                physical > geometry.max_buffer_bytes || physical > geometry.max_binding_bytes
            })
        {
            return reject(
                ProviderRejectionCode::ResourceLimit,
                "convolution.shader_binding_limit",
            );
        }
        if geometry.workgroup_size == 0 || geometry.max_workgroups_per_dimension == 0 {
            return reject(
                ProviderRejectionCode::UnknownResourceEstimate,
                "convolution.shader_launch_unknown",
            );
        }
        let workgroups = output_elements.div_ceil(u64::from(geometry.workgroup_size));
        if workgroups > u64::from(geometry.max_workgroups_per_dimension) {
            return reject(
                ProviderRejectionCode::ResourceLimit,
                "convolution.shader_dispatch_limit",
            );
        }
    }
    let backings = route
        .input_backings
        .and_then(|receipt| receipt.live_requested_bytes());
    if backings.is_some_and(|bytes| {
        query.inputs.iter().enumerate().any(|(index, input)| {
            input
                .checked_byte_len()
                .is_none_or(|logical| bytes[index] < logical)
        })
    }) {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "convolution.backing_smaller_than_view",
        );
    }
    let Some((estimate, largest_route_allocation)) = resources::estimate(
        identity,
        route,
        &query.inputs,
        output_elements,
        output_bytes,
        wgpu,
        backings,
    ) else {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "convolution.peak_bytes_overflow",
        );
    };
    // This is a single-allocation limit, not total free device memory. The
    // shared placement snapshot admits the aggregate live peak separately.
    if provider
        .capability_snapshot()
        .max_allocation_bytes
        .is_some_and(|limit| largest_route_allocation > limit)
    {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "convolution.provider_memory_limit",
        );
    }
    ProviderFeasibility::supported(estimate)
}

fn output_geometry(
    identity: ConvolutionIdentity,
    route: ProviderConvolutionWorkload,
    inputs: &[ProviderRepresentation],
    wgpu: bool,
) -> Option<(Vec<u64>, u64, u64)> {
    match (identity, route.form) {
        (ConvolutionIdentity::Vector, ProviderConvolutionForm::Vector { orientation }) => {
            let (a, b) = (vector_len(&inputs[0].shape)?, vector_len(&inputs[1].shape)?);
            let (len, start) = if a == 0 || b == 0 {
                // InProcess Valid with an empty kernel produces signal-length
                // zeros, while WGPU publishes empty. Do not model this as a
                // common physical recipe until the hook contract is aligned.
                if b == 0 && a != 0 && route.mode == ConvolutionMode::Valid {
                    return None;
                }
                (0, 0)
            } else {
                let full = a.checked_add(b)?.checked_sub(1)?;
                match route.mode {
                    ConvolutionMode::Full => (full, 0),
                    ConvolutionMode::Same => (a, b / 2),
                    ConvolutionMode::Valid if a < b => (0, 0),
                    ConvolutionMode::Valid => (a - b + 1, b - 1),
                }
            };
            let shape = match orientation {
                ProviderConvOrientation::Row => vec![1, len],
                ProviderConvOrientation::Column => vec![len, 1],
            };
            let visits =
                if a == 0 || b == 0 || (!wgpu && route.mode == ConvolutionMode::Valid && a < b) {
                    0
                } else if wgpu {
                    len.checked_mul(b)?
                } else {
                    a.checked_mul(b)?
                };
            Some((shape, visits, start))
        }
        (ConvolutionIdentity::Matrix, ProviderConvolutionForm::Matrix) => {
            let (ar, ac) = matrix_dims(&inputs[0].shape)?;
            let (br, bc) = matrix_dims(&inputs[1].shape)?;
            let (rows, cols) = if [ar, ac, br, bc].contains(&0) {
                match route.mode {
                    ConvolutionMode::Same => (ar, ac),
                    _ => (0, 0),
                }
            } else {
                match route.mode {
                    ConvolutionMode::Full => (
                        ar.checked_add(br)?.checked_sub(1)?,
                        ac.checked_add(bc)?.checked_sub(1)?,
                    ),
                    ConvolutionMode::Same => (ar, ac),
                    ConvolutionMode::Valid if ar < br || ac < bc => (0, 0),
                    ConvolutionMode::Valid => (ar - br + 1, ac - bc + 1),
                }
            };
            let visits = ar.checked_mul(ac)?.checked_mul(br)?.checked_mul(bc)?;
            Some((vec![rows, cols], visits, 0))
        }
        _ => None,
    }
}

fn vector_len(shape: &[u64]) -> Option<u64> {
    let (rows, cols) = matrix_dims(shape)?;
    if rows != 1 && cols != 1 {
        return None;
    }
    rows.checked_mul(cols)
}

fn matrix_dims(shape: &[u64]) -> Option<(u64, u64)> {
    if shape.len() > 2 && shape[2..].iter().any(|&d| d != 1) {
        return None;
    }
    let dimensions = match shape {
        [] => (1, 1),
        [n] => (*n, 1),
        [rows, cols, ..] => (*rows, *cols),
    };
    usize::try_from(dimensions.0).ok()?;
    usize::try_from(dimensions.1).ok()?;
    Some(dimensions)
}
