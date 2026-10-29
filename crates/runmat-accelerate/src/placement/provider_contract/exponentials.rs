//! Direct native real exponential hooks; complex/integer domains remain Runtime-owned.
use super::*;
use runmat_accelerate_api::{
    ProviderRealElementwiseOperation, ProviderRealElementwiseOutputRoute,
    ProviderRealElementwiseRequest,
};
use runmat_builtins::{BuiltinCatalogEntry, EXPM1_CATALOG_ENTRY, EXP_CATALOG_ENTRY};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
enum Operation {
    Exp,
    Expm1,
}

const OPERATIONS: &[(&BuiltinCatalogEntry, Operation)] = &[
    (&EXP_CATALOG_ENTRY, Operation::Exp),
    (&EXPM1_CATALOG_ENTRY, Operation::Expm1),
];

impl Operation {
    fn physical(self) -> ProviderRealElementwiseOperation {
        match self {
            Self::Exp => {
                ProviderRealElementwiseOperation::UnaryMath(runmat_types::ScalarUnaryMath::Exp)
            }
            Self::Expm1 => ProviderRealElementwiseOperation::Expm1,
        }
    }
}

pub(super) fn append(snapshot: &mut ProviderCapabilitySnapshot) {
    snapshot.operations.extend(
        OPERATIONS
            .iter()
            .map(|(entry, _)| ProviderCapabilityOperation {
                identity: ProviderOperationIdentity::new(entry.identity.name),
                family: ProviderOperationFamily::Elementwise,
            }),
    );
}

pub(super) fn validate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    snapshot: &ProviderCapabilitySnapshot,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    let (_, operation) = OPERATIONS
        .iter()
        .find(|(entry, _)| entry.identity.name == query.operation.0)?;
    Some(evaluate(provider, wgpu, snapshot, query, *operation))
}

fn evaluate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    snapshot: &ProviderCapabilitySnapshot,
    query: &ProviderFeasibilityQuery,
    operation: Operation,
) -> ProviderFeasibility {
    let reject = ProviderFeasibility::rejected;
    if query.family != ProviderOperationFamily::Elementwise {
        return reject(
            ProviderRejectionCode::UnsupportedOperation,
            "exponential.family",
        );
    }
    if query.inputs.len() != 1
        || query.outputs.len() > 1
        || query
            .workload
            .requested_outputs
            .is_some_and(|count| count > 1)
    {
        return reject(ProviderRejectionCode::InvalidShape, "exponential.arity");
    }
    let input = &query.inputs[0];
    if input.storage != ProviderStorage::DenseReal {
        return reject(
            ProviderRejectionCode::UnsupportedStorage,
            "exponential.real_dense",
        );
    }
    if !matches!(
        input.element_type,
        ProviderElementType::F32 | ProviderElementType::F64
    ) || (wgpu
        && provider.precision() == ProviderPrecision::F32
        && input.element_type != ProviderElementType::F32)
    {
        return reject(
            ProviderRejectionCode::UnsupportedElementType,
            "exponential.native_class",
        );
    }
    if input.layout != ProviderLayout::ColumnMajorContiguous {
        return reject(
            ProviderRejectionCode::UnsupportedLayout,
            "exponential.contiguous",
        );
    }
    if !matches!(
        input.residency,
        ProviderResidency::Host | ProviderResidency::Device
    ) {
        return reject(
            ProviderRejectionCode::UnsupportedOperation,
            "exponential.source_residency",
        );
    }
    let Some(elements) = input.checked_element_count().filter(|count| {
        usize::try_from(*count).is_ok()
            && input
                .shape
                .iter()
                .all(|extent| usize::try_from(*extent).is_ok())
    }) else {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "exponential.shape_overflow",
        );
    };
    if query
        .workload
        .elements
        .is_some_and(|count| count != elements)
    {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "exponential.workload_shape",
        );
    }
    let mut output = input.clone();
    output.residency = ProviderResidency::Device;
    if query.outputs.iter().any(|declared| declared != &output) {
        return reject(ProviderRejectionCode::InvalidShape, "exponential.output");
    }
    let scalar_bytes = if input.element_type == ProviderElementType::F32 {
        4
    } else {
        8
    };
    let Some(payload_bytes) = input.checked_byte_len() else {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "exponential.bytes_overflow",
        );
    };
    let physical = if wgpu {
        // WGPU empty tensors are allocation-owner sentinel views, not live
        // BufferEntries accepted by the direct unary hook. Empty evaluation
        // remains with the Runtime domain; do not fabricate a zero-work hook.
        if elements == 0 {
            return reject(
                ProviderRejectionCode::UnsupportedOperation,
                "exponential.empty_runtime_route",
            );
        }
        if elements > u64::from(u32::MAX) {
            return reject(
                ProviderRejectionCode::ResourceLimit,
                "exponential.dispatch_length",
            );
        }
        let mut source = input.clone();
        source.residency = ProviderResidency::Device;
        let request = ProviderRealElementwiseRequest {
            operation: operation.physical(),
            source,
            source_buffer_bytes: payload_bytes.max(scalar_bytes),
        };
        // The existing immutable physical owner supplies the actual enabled
        // launch, output allocation, simple uniform, and warmup submission.
        // A declined proposal is not replaced by process-global geometry.
        let Ok(Some(work)) = provider.propose_real_elementwise(&request) else {
            return reject(
                ProviderRejectionCode::UnknownResourceEstimate,
                "exponential.physical_owner",
            );
        };
        if work.output_route != ProviderRealElementwiseOutputRoute::FreshAllocation
            || work.output_buffer_bytes != payload_bytes.max(scalar_bytes)
            || work.workgroup_size == 0
            || work.chunk_capacity == 0
            || work.uniform_allocations != 1
            || work.uniform_buffer_bytes == 0
            || u64::from(work.dispatches) != elements.div_ceil(work.chunk_capacity)
            || work.dispatches.checked_add(1) != Some(work.queue_submissions)
            || snapshot
                .max_allocation_bytes
                .is_some_and(|limit| work.uniform_buffer_bytes > limit)
        {
            return reject(
                ProviderRejectionCode::UnknownResourceEstimate,
                "exponential.physical_work",
            );
        }
        Some(work)
    } else {
        None
    };
    let mut complete = query.clone();
    complete.outputs = vec![output];
    // Both native classes are implemented by these exact hooks, including
    // native Single on a Double provider. Do not widen unrelated inventories.
    let result = dense_feasibility_with_extra_elements(
        snapshot,
        &complete,
        |representation| {
            representation.checked_byte_len().map(|bytes| {
                if wgpu {
                    bytes.max(scalar_bytes)
                } else {
                    bytes
                }
            })
        },
        &[input.element_type],
    );
    match result {
        ProviderFeasibility::Supported { mut estimate } => {
            if let Some(work) = physical {
                let Some(bytes) = estimate
                    .transient_bytes
                    .and_then(|bytes| bytes.checked_add(work.uniform_buffer_bytes))
                else {
                    return reject(
                        ProviderRejectionCode::ResourceLimit,
                        "exponential.total_bytes",
                    );
                };
                estimate.transient_bytes = Some(bytes);
                estimate.output_bytes = Some(work.output_buffer_bytes);
                estimate.dispatches = Some(work.dispatches);
            } else {
                // These reference hooks execute native CPU maps, not device
                // queue dispatches or fabricated provider completion evidence.
                estimate.dispatches = Some(0);
            }
            ProviderFeasibility::supported(estimate)
        }
        rejected => rejected,
    }
}
