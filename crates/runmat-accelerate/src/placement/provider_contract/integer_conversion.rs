//! Typed resident casts. These are provider operations, not floating fusion.
use super::*;
use runmat_accelerate_api::IntegerElementType;
use runmat_builtins::*;

#[cfg(test)]
mod tests;

const OPERATIONS: &[(&BuiltinCatalogEntry, IntegerElementType)] = &[
    (&INT8_CATALOG_ENTRY, IntegerElementType::I8),
    (&INT16_CATALOG_ENTRY, IntegerElementType::I16),
    (&INT32_CATALOG_ENTRY, IntegerElementType::I32),
    (&INT64_CATALOG_ENTRY, IntegerElementType::I64),
    (&UINT8_CATALOG_ENTRY, IntegerElementType::U8),
    (&UINT16_CATALOG_ENTRY, IntegerElementType::U16),
    (&UINT32_CATALOG_ENTRY, IntegerElementType::U32),
    (&UINT64_CATALOG_ENTRY, IntegerElementType::U64),
];

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
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    let (_, target) = OPERATIONS
        .iter()
        .find(|(entry, _)| entry.identity.name == query.operation.0)?;
    let reject = |code, reason| Some(ProviderFeasibility::rejected(code, reason));
    if query.family != ProviderOperationFamily::Elementwise {
        return reject(
            ProviderRejectionCode::UnsupportedOperation,
            "integer_cast.family.mismatch",
        );
    }
    if query.inputs.len() != 1 || query.outputs.len() > 1 {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "integer_cast.operand_count",
        );
    }
    let input = &query.inputs[0];
    if input.storage != ProviderStorage::DenseReal {
        return reject(
            ProviderRejectionCode::UnsupportedStorage,
            "integer_cast.resident_real.required",
        );
    }
    let supported = match input.element_type {
        ProviderElementType::I8
        | ProviderElementType::I16
        | ProviderElementType::I32
        | ProviderElementType::I64
        | ProviderElementType::U8
        | ProviderElementType::U16
        | ProviderElementType::U32
        | ProviderElementType::U64 => true,
        ProviderElementType::F64 => provider.precision() == ProviderPrecision::F64,
        ProviderElementType::F32 => wgpu,
        // Logical descriptors do not identify the physical floating buffer.
        _ => false,
    };
    if !supported {
        return reject(
            ProviderRejectionCode::UnsupportedElementType,
            "integer_cast.input_type.unsupported",
        );
    }
    let output = ProviderRepresentation {
        element_type: ProviderElementType::from(*target),
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: input.shape.clone(),
        residency: ProviderResidency::Device,
    };
    if query.outputs.iter().any(|declared| declared != &output) {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "integer_cast.output_representation.mismatch",
        );
    }
    let elements = input
        .shape
        .iter()
        .try_fold(1_u64, |total, extent| total.checked_mul(*extent));
    if elements.is_none() || (wgpu && elements.is_some_and(|n| n > u64::from(u32::MAX))) {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "integer_cast.dispatch.length",
        );
    }
    // The current automatic adapter omits output descriptors. This domain knows
    // the exact typed result, so resource admission must not report zero bytes.
    let mut complete_query = query.clone();
    complete_query.outputs = vec![output];
    let mut snapshot = dense_capability_snapshot(provider, 3, &[]);
    append(&mut snapshot);
    Some(if wgpu {
        dense_feasibility_with_bytes(
            &snapshot,
            &complete_query,
            crate::backend::wgpu_storage::checked_allocation_bytes,
        )
    } else {
        dense_feasibility(&snapshot, &complete_query)
    })
}
