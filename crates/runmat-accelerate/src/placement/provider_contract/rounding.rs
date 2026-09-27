//! Direct real floating rounding hooks. Digit controls have a separate workload.
use super::*;
use runmat_builtins::{
    CEIL_CATALOG_ENTRY, FIX_CATALOG_ENTRY, FLOOR_CATALOG_ENTRY, ROUND_CATALOG_ENTRY,
};

#[cfg(test)]
mod tests;

#[cfg(feature = "wgpu")]
mod work;
#[cfg(feature = "wgpu")]
pub(crate) use work::wgpu_work;

const ENTRIES: &[&runmat_builtins::BuiltinCatalogEntry] = &[
    &CEIL_CATALOG_ENTRY,
    &FIX_CATALOG_ENTRY,
    &FLOOR_CATALOG_ENTRY,
    &ROUND_CATALOG_ENTRY,
];

pub(super) enum HookProfile {
    InProcess,
    #[cfg(feature = "wgpu")]
    Wgpu,
}

pub(super) fn append(snapshot: &mut ProviderCapabilitySnapshot) {
    snapshot
        .operations
        .extend(ENTRIES.iter().map(|entry| ProviderCapabilityOperation {
            identity: ProviderOperationIdentity::new(entry.identity.name),
            family: ProviderOperationFamily::Elementwise,
        }));
}

pub(super) fn validate(
    provider: &(impl AccelProvider + ?Sized),
    profile: HookProfile,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    ENTRIES
        .iter()
        .find(|entry| entry.identity.name == query.operation.0)?;
    Some(evaluate(provider, profile, query))
}

fn evaluate(
    provider: &(impl AccelProvider + ?Sized),
    profile: HookProfile,
    query: &ProviderFeasibilityQuery,
) -> ProviderFeasibility {
    let reject = ProviderFeasibility::rejected;
    match query.workload.numerical_domain {
        runmat_accelerate_api::NumericalDomainEvidence::Unknown => {}
        runmat_accelerate_api::NumericalDomainEvidence::RoundDigits { digits, mode }
            if query.operation.0 == ROUND_CATALOG_ENTRY.identity.name
                && (mode == runmat_accelerate_api::RoundDigitsMode::Decimals || digits > 0) => {}
        _ => {
            return reject(
                ProviderRejectionCode::UnsupportedOperation,
                "rounding.control_evidence.mismatch",
            );
        }
    }
    if query.family != ProviderOperationFamily::Elementwise {
        return reject(
            ProviderRejectionCode::UnsupportedOperation,
            "rounding.family.mismatch",
        );
    }
    if query.inputs.len() != 1 || query.outputs.len() > 1 {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "rounding.operand_count",
        );
    }
    let input = &query.inputs[0];
    let element = match profile {
        HookProfile::InProcess => ProviderElementType::F64,
        #[cfg(feature = "wgpu")]
        HookProfile::Wgpu => ProviderElementType::from(provider.precision()),
    };
    if input.element_type != element || input.storage != ProviderStorage::DenseReal {
        return reject(
            ProviderRejectionCode::UnsupportedElementType,
            "rounding.real_precision.required",
        );
    }
    let Some(count) = input.checked_element_count() else {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "rounding.shape.overflow",
        );
    };
    #[cfg(feature = "wgpu")]
    if matches!(profile, HookProfile::Wgpu) && count > u64::from(u32::MAX) {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "rounding.dispatch.length",
        );
    }
    #[cfg(not(feature = "wgpu"))]
    let _ = count;
    let mut output = input.clone();
    output.residency = ProviderResidency::Device;
    if query.outputs.iter().any(|declared| declared != &output) {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "rounding.output_representation.mismatch",
        );
    }
    let mut complete = query.clone();
    complete.outputs = vec![output];
    let mut snapshot = dense_capability_snapshot(provider, 1, &[]);
    append(&mut snapshot);
    dense_feasibility(&snapshot, &complete)
}
