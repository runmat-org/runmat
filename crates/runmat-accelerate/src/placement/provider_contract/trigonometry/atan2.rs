//! Exact capabilities of the equal-shape real coordinate kernel.
use super::*;

pub(super) fn operation() -> ProviderCapabilityOperation {
    ProviderCapabilityOperation {
        identity: ProviderOperationIdentity::new(ATAN2_CATALOG_ENTRY.identity.name),
        family: ProviderOperationFamily::Elementwise,
    }
}

pub(super) fn validate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    if query.operation != operation().identity {
        return None;
    }
    let reject = |code, detail| Some(ProviderFeasibility::rejected(code, detail));
    if query.family != ProviderOperationFamily::Elementwise {
        return reject(
            ProviderRejectionCode::UnsupportedOperation,
            "atan2.family.mismatch",
        );
    }
    if query.inputs.len() != 2 || query.outputs.len() > 1 {
        return reject(ProviderRejectionCode::InvalidShape, "atan2.operand_count");
    }
    let y = &query.inputs[0];
    let x = &query.inputs[1];
    if query
        .inputs
        .iter()
        .any(|input| input.storage != ProviderStorage::DenseReal)
    {
        return reject(
            ProviderRejectionCode::UnsupportedStorage,
            "atan2.real_dense.required",
        );
    }
    if query
        .inputs
        .iter()
        .any(|input| input.element_type != ProviderElementType::from(provider.precision()))
    {
        return reject(
            ProviderRejectionCode::UnsupportedElementType,
            "atan2.provider_precision.required",
        );
    }
    if y.shape != x.shape {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "atan2.equal_shape.required",
        );
    }
    let count = y.checked_element_count();
    if count.is_none() || (wgpu && count.is_some_and(|n| n > u64::from(u32::MAX))) {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "atan2.dispatch.length",
        );
    }
    let mut output = y.clone();
    output.residency = ProviderResidency::Device;
    if query.outputs.iter().any(|declared| declared != &output) {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "atan2.output_representation.mismatch",
        );
    }
    let mut complete = query.clone();
    complete.outputs = vec![output];
    let mut snapshot = dense_capability_snapshot(provider, 1, &[]);
    snapshot.operations.push(operation());
    Some(dense_feasibility(&snapshot, &complete))
}
