//! Exact feasibility of the three-input resident selection primitive.
use super::*;
use runmat_accelerate_api::RESIDENT_SELECT_OPERATION_IDENTITY;

mod resource;

#[cfg(test)]
mod tests;

pub(crate) struct ResidentSelectWork {
    pub estimate: ProviderResourceEstimate,
    #[cfg(feature = "wgpu")]
    pub uniform_bytes: u64,
    #[cfg(feature = "wgpu")]
    pub queue_submissions: u32,
    #[cfg(feature = "wgpu")]
    pub scalar_visits: u64,
}

pub(super) fn append(snapshot: &mut ProviderCapabilitySnapshot) {
    snapshot.operations.push(ProviderCapabilityOperation {
        identity: ProviderOperationIdentity::new(RESIDENT_SELECT_OPERATION_IDENTITY),
        family: ProviderOperationFamily::Elementwise,
    });
}

pub(super) fn validate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    analyze(provider, wgpu, query).map(|work| match work {
        Ok(work) => ProviderFeasibility::supported(work.estimate),
        Err(rejection) => rejection,
    })
}

#[cfg(feature = "wgpu")]
pub(crate) fn physical_work(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
) -> Option<ResidentSelectWork> {
    analyze(provider, true, query)?.ok()
}

fn analyze(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<Result<ResidentSelectWork, ProviderFeasibility>> {
    if query.operation.0 != RESIDENT_SELECT_OPERATION_IDENTITY {
        return None;
    }
    Some(evaluate(provider, wgpu, query))
}

fn evaluate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Result<ResidentSelectWork, ProviderFeasibility> {
    let reject = ProviderFeasibility::rejected;
    if query.family != ProviderOperationFamily::Elementwise
        || query.inputs.len() != 3
        || query.outputs.len() != 1
    {
        return Err(reject(
            ProviderRejectionCode::InvalidShape,
            "resident_select.arity_or_family",
        ));
    }
    let mask = &query.inputs[0];
    let then_value = &query.inputs[1];
    let else_value = &query.inputs[2];
    let output = &query.outputs[0];
    if query.inputs.iter().chain(&query.outputs).any(|value| {
        value.storage != ProviderStorage::DenseReal || value.residency != ProviderResidency::Device
    }) {
        return Err(reject(
            ProviderRejectionCode::UnsupportedStorage,
            "resident_select.resident_real_required",
        ));
    }
    let native = ProviderElementType::from(provider.precision());
    let branch = then_value.element_type;
    if mask.element_type != ProviderElementType::Logical
        || !matches!(branch, ProviderElementType::F32 | ProviderElementType::F64)
        || branch != else_value.element_type
        || (wgpu && branch != native)
        || output.element_type != branch
    {
        return Err(reject(
            ProviderRejectionCode::UnsupportedElementType,
            "resident_select.logical_mask_matching_real_branches",
        ));
    }
    for input in &query.inputs {
        let contiguous = input.layout == ProviderLayout::ColumnMajorContiguous;
        let supported_strided =
            wgpu && input.layout == ProviderLayout::Strided && input.shape.len() == 2;
        if !contiguous && !supported_strided {
            return Err(reject(
                ProviderRejectionCode::UnsupportedLayout,
                "resident_select.input_layout",
            ));
        }
    }
    if output.layout != ProviderLayout::ColumnMajorContiguous {
        return Err(reject(
            ProviderRejectionCode::UnsupportedLayout,
            "resident_select.output_layout",
        ));
    }
    let mut shapes = Vec::new();
    for representation in &query.inputs {
        let Ok(shape) = representation
            .shape
            .iter()
            .copied()
            .map(usize::try_from)
            .collect::<Result<Vec<_>, _>>()
        else {
            return Err(reject(
                ProviderRejectionCode::InvalidShape,
                "resident_select.dimension_overflow",
            ));
        };
        shapes.push(shape);
    }
    let Ok(shape) = runmat_types::geometry::broadcast_concrete_shape(&shapes[0], &shapes[1])
        .and_then(|shape| runmat_types::geometry::broadcast_concrete_shape(&shape, &shapes[2]))
    else {
        return Err(reject(
            ProviderRejectionCode::InvalidShape,
            "resident_select.broadcast_shape",
        ));
    };
    let Ok(shape_u64) = shape
        .iter()
        .copied()
        .map(u64::try_from)
        .collect::<Result<Vec<_>, _>>()
    else {
        return Err(reject(
            ProviderRejectionCode::InvalidShape,
            "resident_select.dimension_overflow",
        ));
    };
    if output.shape != shape_u64 {
        return Err(reject(
            ProviderRejectionCode::InvalidShape,
            "resident_select.output_shape",
        ));
    }
    let Some(elements) = shape_u64
        .iter()
        .try_fold(1_u64, |count, &extent| count.checked_mul(extent))
    else {
        return Err(reject(
            ProviderRejectionCode::InvalidShape,
            "resident_select.element_count_overflow",
        ));
    };
    if query
        .workload
        .elements
        .is_some_and(|declared| declared != elements)
    {
        return Err(reject(
            ProviderRejectionCode::InvalidShape,
            "resident_select.workload_geometry",
        ));
    }
    resource::estimate(provider, wgpu, branch, elements, &shape)
}
