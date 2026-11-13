use super::*;
use runmat_accelerate_api::ProviderWorkload;

#[test]
fn mixed_inprocess_profiles_require_both_domains_and_exact_public_classes() {
    let provider = crate::simple_provider::InProcessProvider::new();
    let snapshot = provider.capability_snapshot();
    let single = ProviderRepresentation {
        element_type: ProviderElementType::F32,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: vec![2],
        residency: ProviderResidency::Device,
    };
    let double_scalar = ProviderRepresentation {
        element_type: ProviderElementType::F64,
        shape: vec![1, 1],
        residency: ProviderResidency::Host,
        ..single.clone()
    };
    let mut query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(
            runmat_builtins::POWER_CATALOG_ENTRY.identity.name,
        ),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![single.clone(), double_scalar.clone()],
        outputs: vec![],
        workload: ProviderWorkload {
            real_power_mode: Some(RealPowerArithmeticMode::MixedDoubleThenSingle {
                single_operand: RealPowerOperand::Base,
            }),
            ..ProviderWorkload::default()
        },
    };
    for domain in [PowerResultDomain::Real, PowerResultDomain::Complex] {
        query.workload.numerical_domain = NumericalDomainEvidence::RealPower(domain);
        let ProviderFeasibility::Supported { estimate } = feasibility(
            HookProfile::InProcess,
            ProviderPrecision::F64,
            &provider,
            &snapshot,
            &query,
        )
        .unwrap() else {
            panic!("both domains must have an exact portable recipe")
        };
        assert_eq!(estimate.dispatches, Some(0));
        assert!(estimate.transient_bytes.is_some());
        let work = work(
            HookProfile::InProcess,
            ProviderPrecision::F64,
            &provider,
            &query,
        )
        .unwrap();
        assert_eq!(
            work.output.element_type,
            if domain == PowerResultDomain::Real {
                ProviderElementType::F32
            } else {
                ProviderElementType::ComplexF32
            }
        );
        assert_eq!(
            work.output.storage,
            if domain == PowerResultDomain::Real {
                ProviderStorage::DenseReal
            } else {
                ProviderStorage::DenseComplexInterleaved
            }
        );
        assert_eq!(work.output.shape, vec![1, 2]);
    }
    // Host Single scalar, resident Double: no resident widening, still one
    // native Double compute and one Single publication.
    query.inputs = vec![
        ProviderRepresentation {
            element_type: ProviderElementType::F64,
            ..single.clone()
        },
        ProviderRepresentation {
            element_type: ProviderElementType::F32,
            ..double_scalar.clone()
        },
    ];
    query.workload.real_power_mode = Some(RealPowerArithmeticMode::MixedDoubleThenSingle {
        single_operand: RealPowerOperand::Exponent,
    });
    for domain in [PowerResultDomain::Real, PowerResultDomain::Complex] {
        query.workload.numerical_domain = NumericalDomainEvidence::RealPower(domain);
        assert!(feasibility(
            HookProfile::InProcess,
            ProviderPrecision::F64,
            &provider,
            &snapshot,
            &query
        )
        .unwrap()
        .is_supported());
    }
    let mut bounded = snapshot.clone();
    bounded.max_allocation_bytes = Some(8);
    assert!(!feasibility(
        HookProfile::InProcess,
        ProviderPrecision::F64,
        &provider,
        &bounded,
        &query
    )
    .unwrap()
    .is_supported());
    query.inputs[0].shape = vec![0, 2];
    bounded.max_allocation_bytes = Some(1024);
    assert!(!feasibility(
        HookProfile::InProcess,
        ProviderPrecision::F64,
        &provider,
        &bounded,
        &query
    )
    .unwrap()
    .is_supported());
}
