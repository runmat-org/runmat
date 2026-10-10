use super::super::fusion_elementwise::fixture::GeometryProvider;
use super::*;
use runmat_accelerate_api::ProviderWorkload;

#[test]
fn mixed_mode_requires_exact_roles_and_counts_double_stages_and_single_publication() {
    let provider = GeometryProvider(256);
    let mut snapshot = crate::simple_provider::InProcessProvider::new().capability_snapshot();
    append(&mut snapshot);
    let resident = ProviderRepresentation {
        element_type: ProviderElementType::F32,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: vec![1, 3],
        residency: ProviderResidency::Device,
    };
    let scalar = ProviderRepresentation {
        element_type: ProviderElementType::F64,
        shape: vec![1, 1],
        residency: ProviderResidency::Host,
        ..resident.clone()
    };
    let mut query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(
            runmat_builtins::POWER_CATALOG_ENTRY.identity.name,
        ),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![resident.clone(), scalar.clone()],
        outputs: Vec::new(),
        workload: ProviderWorkload::default(),
    };
    query.workload.numerical_domain =
        NumericalDomainEvidence::RealPower(PowerResultDomain::Complex);
    query.workload.real_power_mode = Some(RealPowerArithmeticMode::MixedDoubleThenSingle {
        single_operand: RealPowerOperand::Base,
    });
    let work = wgpu_work(ProviderPrecision::F64, &provider, &query).expect("mixed physical route");
    assert_eq!(work.output.element_type, ProviderElementType::ComplexF32);
    assert_eq!(work.output.shape, vec![1, 3]);
    assert_eq!(work.output.residency, ProviderResidency::Device);
    assert_eq!(work.dispatches, 7);
    assert_eq!(work.readback_bytes, 8);
    assert!(feasibility(
        HookProfile::Wgpu,
        ProviderPrecision::F64,
        &provider,
        &snapshot,
        &query
    )
    .unwrap()
    .is_supported());
    assert!(!feasibility(
        HookProfile::Wgpu,
        ProviderPrecision::F32,
        &provider,
        &snapshot,
        &query
    )
    .unwrap()
    .is_supported());
    query.workload.real_power_mode = Some(RealPowerArithmeticMode::MixedDoubleThenSingle {
        single_operand: RealPowerOperand::Exponent,
    });
    assert!(wgpu_work(ProviderPrecision::F64, &provider, &query).is_none());
    query.workload.real_power_mode = Some(RealPowerArithmeticMode::MixedDoubleThenSingle {
        single_operand: RealPowerOperand::Base,
    });
    query.inputs[1].shape = vec![1, 2];
    assert!(wgpu_work(ProviderPrecision::F64, &provider, &query).is_none());

    // The Single operand can instead be the host scalar. In that case the
    // resident is already Double: there is no widening dispatch or buffer.
    query.inputs = vec![
        ProviderRepresentation {
            element_type: ProviderElementType::F64,
            ..resident.clone()
        },
        ProviderRepresentation {
            element_type: ProviderElementType::F32,
            shape: vec![1, 1],
            residency: ProviderResidency::Host,
            ..resident.clone()
        },
    ];
    query.workload.real_power_mode = Some(RealPowerArithmeticMode::MixedDoubleThenSingle {
        single_operand: RealPowerOperand::Exponent,
    });
    let work = wgpu_work(ProviderPrecision::F64, &provider, &query)
        .expect("host Single scalar uses native resident Double");
    assert_eq!(work.dispatches, 6);
    assert!(!work.unknown_peak);

    // A zero-cardinality view does not reveal its live backing allocation.
    // A finite peak limit cannot silently treat that resident as zero bytes.
    query.inputs[0].shape = vec![0, 3];
    snapshot.max_allocation_bytes = Some(1024);
    assert!(
        wgpu_work(ProviderPrecision::F64, &provider, &query).is_some_and(|work| work.unknown_peak)
    );
    assert!(!feasibility(
        HookProfile::Wgpu,
        ProviderPrecision::F64,
        &provider,
        &snapshot,
        &query
    )
    .unwrap()
    .is_supported());
}
