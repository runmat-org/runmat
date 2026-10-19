use super::super::fusion_elementwise::fixture::GeometryProvider;
use super::*;
use runmat_accelerate_api::ProviderWorkload;

#[test]
fn native_single_on_double_owner_has_exact_class_domain_and_probe_recipe() {
    let provider = GeometryProvider(256);
    let mut snapshot = crate::simple_provider::InProcessProvider::new().capability_snapshot();
    append(&mut snapshot);
    let base = ProviderRepresentation {
        element_type: ProviderElementType::F32,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: vec![2, 4, 5],
        residency: ProviderResidency::Device,
    };
    let control = ProviderRepresentation {
        shape: vec![1, 1],
        ..base.clone()
    };
    let mut query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(
            runmat_builtins::POWER_CATALOG_ENTRY.identity.name,
        ),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![base, control],
        outputs: vec![],
        workload: ProviderWorkload {
            real_power_mode: Some(RealPowerArithmeticMode::NativeSingle),
            numerical_domain: NumericalDomainEvidence::RealPower(PowerResultDomain::Real),
            ..ProviderWorkload::default()
        },
    };
    let work = wgpu_work(ProviderPrecision::F64, &provider, &query).expect("typed Single work");
    assert_eq!(work.output.element_type, ProviderElementType::F32);
    assert_eq!(work.output.shape, vec![2, 4, 5]);
    assert_eq!(work.output.residency, ProviderResidency::Device);
    assert_eq!(work.dispatches, 4);
    assert_eq!(work.readback_bytes, 8);
    query.outputs = vec![work.output.clone()];
    let ProviderFeasibility::Supported { estimate } = feasibility(
        HookProfile::Wgpu,
        ProviderPrecision::F64,
        &provider,
        &snapshot,
        &query,
    )
    .unwrap() else {
        panic!("selected Device output must be admitted")
    };
    assert_eq!(estimate.output_bytes, Some(40 * 4));
    assert!(estimate
        .transient_bytes
        .is_some_and(|bytes| bytes >= 40 * 4));
    query.outputs[0].residency = ProviderResidency::Host;
    assert!(matches!(
        feasibility(HookProfile::Wgpu, ProviderPrecision::F64, &provider, &snapshot, &query),
        Some(ProviderFeasibility::Rejected { rejection })
            if rejection.code == ProviderRejectionCode::InvalidShape
    ));
    query.outputs[0].residency = ProviderResidency::Device;
    query.inputs[1].residency = ProviderResidency::Host;
    assert!(feasibility(
        HookProfile::Wgpu,
        ProviderPrecision::F64,
        &provider,
        &snapshot,
        &query
    )
    .unwrap()
    .is_supported());
    query.inputs[1].residency = ProviderResidency::Device;
    query.inputs[0].residency = ProviderResidency::Host;
    query.inputs[1].residency = ProviderResidency::Host;
    assert!(wgpu_work(ProviderPrecision::F64, &provider, &query).is_none());
    query.inputs[0].residency = ProviderResidency::Device;
    query.inputs[1].residency = ProviderResidency::Device;
    assert!(!feasibility(
        HookProfile::InProcess,
        ProviderPrecision::F64,
        &provider,
        &snapshot,
        &query
    )
    .unwrap()
    .is_supported());

    query.outputs.clear();
    query.workload.numerical_domain =
        NumericalDomainEvidence::RealPower(PowerResultDomain::Complex);
    assert_eq!(
        wgpu_work(ProviderPrecision::F64, &provider, &query)
            .expect("typed complex Single work")
            .output
            .element_type,
        ProviderElementType::ComplexF32
    );
    query.inputs[1].element_type = ProviderElementType::F64;
    assert!(wgpu_work(ProviderPrecision::F64, &provider, &query).is_none());
    query.inputs[1].element_type = ProviderElementType::F32;
    query.workload.real_power_mode = Some(RealPowerArithmeticMode::NativeDouble);
    assert!(wgpu_work(ProviderPrecision::F64, &provider, &query).is_none());
    query.workload.real_power_mode = Some(RealPowerArithmeticMode::NativeSingle);
    query.inputs[1].shape = vec![3, 1];
    assert!(wgpu_work(ProviderPrecision::F64, &provider, &query).is_none());
    query.inputs[1].shape = vec![1, 1];
    query.workload.numerical_domain = NumericalDomainEvidence::Unknown;
    assert!(wgpu_work(ProviderPrecision::F64, &provider, &query).is_none());

    query.workload.numerical_domain = NumericalDomainEvidence::RealPower(PowerResultDomain::Real);
    snapshot.max_allocation_bytes = Some(32);
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
