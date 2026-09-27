use super::*;
use crate::simple_provider::InProcessProvider;
use futures::executor::block_on;
use runmat_accelerate_api::{HostTensorView, NumericalDomainEvidence, ProviderWorkload};
use runmat_types::RealUnaryClassification;

fn query(entry: &BuiltinCatalogEntry, predicate: RealUnaryPredicate) -> ProviderFeasibilityQuery {
    ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(entry.identity.name),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![ProviderRepresentation {
            element_type: ProviderElementType::F64,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![1, 2],
            residency: ProviderResidency::Host,
        }],
        outputs: vec![],
        workload: ProviderWorkload {
            numerical_domain: NumericalDomainEvidence::RealUnary {
                predicate,
                classification: RealUnaryClassification::Inside,
            },
            ..ProviderWorkload::default()
        },
    }
}

#[test]
fn exact_declarations_require_matching_domain_and_actual_hooks() {
    let provider = InProcessProvider::new();
    let snapshot = provider.capability_snapshot();
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0, 4.0],
            shape: &[1, 2],
        })
        .unwrap();
    for ((entry, predicate), operation) in OPERATIONS.iter().zip(0..) {
        let mut request = query(entry, *predicate);
        assert!(
            provider.query_feasibility(&request).is_supported(),
            "{}",
            entry.identity.name
        );
        assert!(
            snapshot.supports_operation(&request.operation, ProviderOperationFamily::Elementwise)
        );
        let output = block_on(match operation {
            0 => provider.unary_log(&input),
            1 => provider.unary_log10(&input),
            2 => provider.unary_log1p(&input),
            3 => provider.unary_log2(&input),
            _ => provider.unary_sqrt(&input),
        })
        .unwrap();
        let expected = match operation {
            0 => vec![1.0_f64.ln(), 4.0_f64.ln()],
            1 => vec![0.0, 4.0_f64.log10()],
            2 => vec![1.0_f64.ln_1p(), 4.0_f64.ln_1p()],
            3 => vec![0.0, 2.0],
            _ => vec![1.0, 2.0],
        };
        assert_eq!(block_on(provider.download(&output)).unwrap().data, expected);
        output.release().unwrap();
        for evidence in [
            NumericalDomainEvidence::Unknown,
            NumericalDomainEvidence::RealUnary {
                predicate: *predicate,
                classification: RealUnaryClassification::Outside,
            },
            NumericalDomainEvidence::RealUnary {
                predicate: RealUnaryPredicate::ClosedUnitInterval,
                classification: RealUnaryClassification::Inside,
            },
        ] {
            request.workload.numerical_domain = evidence;
            assert!(!provider.query_feasibility(&request).is_supported());
        }
    }
    input.release().unwrap();
}
