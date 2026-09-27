#[derive(Debug, PartialEq, Eq)]
pub(crate) struct RemainderWork {
    pub dispatches: u32,
    pub queue_submissions: u32,
    pub allocations: u32,
    pub scratch_bytes: u64,
    pub extra_upload_bytes: u64,
    pub execution_units: u64,
}

#[cfg(feature = "wgpu")]
pub(crate) fn wgpu_work(
    precision: runmat_accelerate_api::ProviderPrecision,
    query: &runmat_accelerate_api::ProviderFeasibilityQuery,
) -> Option<RemainderWork> {
    use runmat_accelerate_api::{NumericalDomainEvidence, ProviderOperationFamily};
    let NumericalDomainEvidence::FloatingRemainder(kind) = query.workload.numerical_domain else {
        return None;
    };
    if query.operation.0 != super::identity(kind)
        || query.family != ProviderOperationFamily::Elementwise
    {
        return None;
    }
    let [left, right] = query.inputs.as_slice() else {
        return None;
    };
    if query.outputs.len() > 1
        || [left, right].iter().any(|input| {
            input.element_type != runmat_accelerate_api::ProviderElementType::from(precision)
                || input.storage != runmat_accelerate_api::ProviderStorage::DenseReal
                || !(input.layout == runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous
                    || (input.layout == runmat_accelerate_api::ProviderLayout::Strided
                        && input.residency == runmat_accelerate_api::ProviderResidency::Device
                        && input.shape.len() == 2))
        })
    {
        return None;
    }
    let shapes: Option<Vec<Vec<usize>>> = [left, right]
        .iter()
        .map(|input| {
            input
                .shape
                .iter()
                .map(|extent| usize::try_from(*extent).ok())
                .collect()
        })
        .collect();
    let shapes = shapes?;
    let shape = runmat_runtime::builtins::common::broadcast::broadcast_shapes(
        "remainder",
        &shapes[0],
        &shapes[1],
    )
    .ok()?;
    let elements = if shape.contains(&0) {
        0
    } else {
        shape
            .iter()
            .try_fold(1_u64, |n, d| n.checked_mul(*d as u64))?
    };
    if elements > u64::from(u32::MAX) {
        return None;
    }
    let capacity = crate::backend::wgpu::config::elementwise_chunk_capacity(
        crate::backend::wgpu::config::effective_workgroup_size(),
    )?;
    let dispatches = u32::try_from(elements.div_ceil(capacity)).ok()?;
    let params = std::mem::size_of::<crate::backend::wgpu::params::BinaryBroadcastParams>() as u64;
    Some(RemainderWork {
        dispatches,
        queue_submissions: if elements == 0 {
            0
        } else {
            dispatches.checked_add(1)?
        },
        allocations: if elements == 0 { 1 } else { 2 },
        scratch_bytes: if elements == 0 { 0 } else { params },
        extra_upload_bytes: if elements == 0 {
            0
        } else {
            params.checked_mul(u64::from(dispatches) + 1)?
        },
        execution_units: elements.checked_mul(
            crate::kernel_math::floating_remainder::scalar_work_units(precision),
        )?,
    })
}
