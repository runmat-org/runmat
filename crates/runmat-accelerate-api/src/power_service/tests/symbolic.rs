use super::*;
use crate::{
    ProviderElementType, ProviderLayout, ProviderRepresentation, ProviderResidency, ProviderStorage,
};

fn resident(shape: &[u64], element_type: ProviderElementType) -> ProviderRepresentation {
    ProviderRepresentation {
        element_type,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: shape.to_vec(),
        residency: ProviderResidency::Device,
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn symbolic_mixed_recipe_matches_live_preparation_and_preserves_source_controls() {
    let mode = RealPowerArithmeticMode::MixedDoubleThenSingle {
        single_operand: RealPowerOperand::Base,
    };
    let provider = provider_with_mode(false, true, mode);
    let control = PowerScalarControl::DoubleBits(2.0f64.to_bits());
    let operation = ProviderOperationIdentity::new("test.power");
    let plan = plan_resident_scalar_power(
        provider,
        resident(&[2, 4, 5], ProviderElementType::F32),
        control,
        RealPowerOperand::Base,
        &[1, 1],
        operation.clone(),
    )
    .unwrap()
    .unwrap();
    assert_eq!(plan.control(), control);
    assert_eq!(plan.mode(), mode);
    let work = plan.physical_work(PowerResultDomain::Real).unwrap();
    assert_eq!(
        work.widened_resident.as_ref().unwrap().element_type,
        ProviderElementType::F64
    );
    assert_eq!(work.scalar.shape, [2, 4, 5]);
    assert_eq!(work.scalar.element_type, ProviderElementType::F64);
    assert_eq!(work.native_result.element_type, ProviderElementType::F64);
    assert_eq!(
        work.single_publication.as_ref().unwrap().element_type,
        ProviderElementType::F32
    );
    assert_eq!(work.output.shape, [2, 4, 5]);
    assert!(work.paired_domain_probe);
    let complex = plan.physical_work(PowerResultDomain::Complex).unwrap();
    assert_eq!(
        complex.native_result.element_type,
        ProviderElementType::ComplexF64
    );
    assert_eq!(complex.output.element_type, ProviderElementType::ComplexF32);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 0);

    let source = provider
        .owner
        .allocate_zeroed(
            vec![2, 4, 5],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let actual = prepare_resident_scalar_power_against_plan(
        provider,
        &source.handle,
        &plan,
        control,
        RealPowerOperand::Base,
        &[1, 1],
        operation.clone(),
    )
    .unwrap();
    assert_eq!(actual.symbolic_plan(), &plan);
    assert_eq!(actual.physical_recipe(), plan.physical_recipe());
    assert_eq!(
        actual.feasibility_query(PowerResultDomain::Real),
        plan.feasibility_query(PowerResultDomain::Real)
    );

    for altered in [
        PowerScalarControl::SingleBits(2.0f32.to_bits()),
        PowerScalarControl::DoubleBits(2.0f64.to_bits() + 1),
    ] {
        assert!(prepare_resident_scalar_power_against_plan(
            provider,
            &source.handle,
            &plan,
            altered,
            RealPowerOperand::Base,
            &[1, 1],
            operation.clone(),
        )
        .is_err());
    }
    assert!(prepare_resident_scalar_power_against_plan(
        provider,
        &source.handle,
        &plan,
        control,
        RealPowerOperand::Exponent,
        &[1, 1],
        operation.clone(),
    )
    .is_err());
    assert!(prepare_resident_scalar_power_against_plan(
        provider,
        &source.handle,
        &plan,
        control,
        RealPowerOperand::Base,
        &[1, 1, 1],
        operation.clone(),
    )
    .is_err());
    assert!(prepare_resident_scalar_power_against_plan(
        provider,
        &source.handle,
        &plan,
        control,
        RealPowerOperand::Base,
        &[1, 1],
        ProviderOperationIdentity::new("other.power"),
    )
    .is_err());
    for (shape, element_type) in [
        (vec![2, 4, 5], NumericElementType::F64),
        (vec![2, 5, 4], NumericElementType::F32),
    ] {
        let changed = provider
            .owner
            .allocate_zeroed(shape, element_type, GpuTensorStorage::Real)
            .unwrap();
        assert!(prepare_resident_scalar_power_against_plan(
            provider,
            &changed.handle,
            &plan,
            control,
            RealPowerOperand::Base,
            &[1, 1],
            operation.clone(),
        )
        .is_err());
    }
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 0);
    assert_eq!(source.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn symbolic_native_single_uses_exact_upload_without_widening() {
    let provider = provider(false);
    let control = PowerScalarControl::SingleBits(1.8f32.to_bits());
    let operation = ProviderOperationIdentity::new("test.power");
    let plan = plan_resident_scalar_power(
        provider,
        resident(&[2, 4, 5], ProviderElementType::F32),
        control,
        RealPowerOperand::Base,
        &[1, 1],
        operation.clone(),
    )
    .unwrap()
    .unwrap();
    assert_eq!(plan.mode(), RealPowerArithmeticMode::NativeSingle);
    assert_eq!(
        plan.physical_recipe().scalar,
        PowerScalarConstruction::ExactTypedUpload
    );
    let work = plan.physical_work(PowerResultDomain::Real).unwrap();
    assert!(work.widened_resident.is_none());
    assert!(work.single_publication.is_none());
    assert_eq!(work.scalar.shape, [1, 1]);
    assert_eq!(work.scalar.element_type, ProviderElementType::F32);
    assert_eq!(work.native_result.element_type, ProviderElementType::F32);
    let source = provider
        .owner
        .allocate_zeroed(
            vec![2, 4, 5],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let prepared = prepare_resident_scalar_power_against_plan(
        provider,
        &source.handle,
        &plan,
        control,
        RealPowerOperand::Base,
        &[1, 1],
        operation,
    )
    .unwrap();
    assert_eq!(prepared.symbolic_plan(), &plan);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn legacy_proven_real_plan_has_no_complex_physical_branch() {
    let provider = provider(false);
    provider.advertise_power.store(false, Ordering::SeqCst);
    let plan = plan_resident_scalar_power(
        provider,
        resident(&[2, 4, 5], ProviderElementType::F64),
        PowerScalarControl::DoubleBits(2.0f64.to_bits()),
        RealPowerOperand::Base,
        &[1, 1],
        ProviderOperationIdentity::new("test.power"),
    )
    .unwrap()
    .expect("exact integral exponent permits the legacy real-only route");
    assert_eq!(
        plan.physical_recipe().route,
        PowerPhysicalRoute::LegacyProvenReal
    );
    assert!(plan.physical_work(PowerResultDomain::Real).is_some());
    assert!(plan.physical_work(PowerResultDomain::Complex).is_none());
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 0);
}
