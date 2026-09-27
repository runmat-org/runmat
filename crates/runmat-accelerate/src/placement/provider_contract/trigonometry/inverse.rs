//! Exact inverse identities admitted by the shared real-domain compound route.
use super::super::{real_unary, *};
use runmat_types::RealUnaryPredicate;

const OPERATIONS: &[(&str, RealUnaryPredicate)] = &[
    (
        runmat_builtins::ASIN_CATALOG_ENTRY.identity.name,
        RealUnaryPredicate::ClosedUnitInterval,
    ),
    (
        runmat_builtins::ACOS_CATALOG_ENTRY.identity.name,
        RealUnaryPredicate::ClosedUnitInterval,
    ),
    (
        runmat_builtins::ACOSH_CATALOG_ENTRY.identity.name,
        RealUnaryPredicate::AtLeastOne,
    ),
    (
        runmat_builtins::ATANH_CATALOG_ENTRY.identity.name,
        RealUnaryPredicate::ClosedUnitInterval,
    ),
];

pub(super) fn append(snapshot: &mut ProviderCapabilitySnapshot) {
    snapshot.operations.extend(
        OPERATIONS
            .iter()
            .map(|(name, _)| ProviderCapabilityOperation {
                identity: ProviderOperationIdentity::new(*name),
                family: ProviderOperationFamily::Elementwise,
            }),
    );
}

fn predicate(query: &ProviderFeasibilityQuery) -> Option<RealUnaryPredicate> {
    OPERATIONS
        .iter()
        .find_map(|(name, predicate)| (*name == query.operation.0).then_some(*predicate))
}

pub(super) fn validate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    let predicate = predicate(query)?;
    let profile = if wgpu {
        #[cfg(feature = "wgpu")]
        {
            real_unary::HookProfile::Wgpu
        }
        #[cfg(not(feature = "wgpu"))]
        {
            return Some(ProviderFeasibility::rejected(
                ProviderRejectionCode::ProviderUnavailable,
                "real_unary.wgpu.unavailable",
            ));
        }
    } else {
        real_unary::HookProfile::InProcess
    };
    let mut snapshot = dense_capability_snapshot(provider, 1, &[]);
    append(&mut snapshot);
    Some(real_unary::feasibility(
        profile,
        provider.precision(),
        &snapshot,
        query,
        predicate,
    ))
}

#[cfg(feature = "wgpu")]
pub(crate) fn wgpu_work(
    precision: ProviderPrecision,
    query: &ProviderFeasibilityQuery,
) -> Option<real_unary::UnaryWork> {
    real_unary::work(
        real_unary::HookProfile::Wgpu,
        precision,
        query,
        predicate(query)?,
    )
    .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{NumericalDomainEvidence, ProviderWorkload};
    use runmat_types::RealUnaryClassification;

    #[test]
    fn inverse_capability_queries_require_exact_real_inside_evidence() {
        let provider = crate::simple_provider::InProcessProvider::new();
        for (name, predicate) in OPERATIONS {
            let mut query = ProviderFeasibilityQuery {
                operation: ProviderOperationIdentity::new(*name),
                family: ProviderOperationFamily::Elementwise,
                inputs: vec![ProviderRepresentation {
                    element_type: ProviderElementType::F64,
                    storage: ProviderStorage::DenseReal,
                    layout: ProviderLayout::ColumnMajorContiguous,
                    shape: vec![1, 2, 3],
                    residency: ProviderResidency::Host,
                }],
                outputs: vec![],
                workload: ProviderWorkload {
                    numerical_domain: NumericalDomainEvidence::RealUnary {
                        predicate: *predicate,
                        classification: RealUnaryClassification::Inside,
                    },
                    ..Default::default()
                },
            };
            assert!(
                matches!(
                    provider.query_feasibility(&query),
                    ProviderFeasibility::Supported { .. }
                ),
                "{name}"
            );
            for classification in [
                RealUnaryClassification::Outside,
                RealUnaryClassification::Indeterminate,
            ] {
                query.workload.numerical_domain = NumericalDomainEvidence::RealUnary {
                    predicate: *predicate,
                    classification,
                };
                assert!(matches!(
                    provider.query_feasibility(&query),
                    ProviderFeasibility::Rejected { .. }
                ));
            }
            query.workload.numerical_domain = NumericalDomainEvidence::Unknown;
            assert!(matches!(
                provider.query_feasibility(&query),
                ProviderFeasibility::Rejected { .. }
            ));
        }
    }
}
