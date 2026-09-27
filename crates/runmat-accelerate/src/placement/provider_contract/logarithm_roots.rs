//! Exact logarithm/root forms whose runtime route includes a lower-bound probe.
use super::*;
use runmat_builtins::{
    BuiltinCatalogEntry, LOG10_CATALOG_ENTRY, LOG1P_CATALOG_ENTRY, LOG2_CATALOG_ENTRY,
    LOG_CATALOG_ENTRY, REALSQRT_CATALOG_ENTRY, SQRT_CATALOG_ENTRY,
};
use runmat_types::RealUnaryPredicate;

#[cfg(test)]
mod tests;

const OPERATIONS: &[(&BuiltinCatalogEntry, RealUnaryPredicate)] = &[
    (&LOG_CATALOG_ENTRY, RealUnaryPredicate::Nonnegative),
    (&LOG10_CATALOG_ENTRY, RealUnaryPredicate::Nonnegative),
    (&LOG1P_CATALOG_ENTRY, RealUnaryPredicate::AtLeastMinusOne),
    (&LOG2_CATALOG_ENTRY, RealUnaryPredicate::Nonnegative),
    (&SQRT_CATALOG_ENTRY, RealUnaryPredicate::Nonnegative),
    (&REALSQRT_CATALOG_ENTRY, RealUnaryPredicate::Nonnegative),
];

fn predicate(query: &ProviderFeasibilityQuery) -> Option<RealUnaryPredicate> {
    OPERATIONS.iter().find_map(|(entry, predicate)| {
        (entry.identity.name == query.operation.0).then_some(*predicate)
    })
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

pub(super) fn feasibility(
    profile: super::real_unary::HookProfile,
    precision: ProviderPrecision,
    snapshot: &ProviderCapabilitySnapshot,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    Some(super::real_unary::feasibility(
        profile,
        precision,
        snapshot,
        query,
        predicate(query)?,
    ))
}

#[cfg(feature = "wgpu")]
pub(crate) fn wgpu_work(
    precision: ProviderPrecision,
    query: &ProviderFeasibilityQuery,
) -> Option<super::real_unary::UnaryWork> {
    super::real_unary::work(
        super::real_unary::HookProfile::Wgpu,
        precision,
        query,
        predicate(query)?,
    )
    .ok()
}
