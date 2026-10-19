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
fn future_allocator_extent_is_revalidated_then_bound_to_authentic_live_input() {
    let provider = provider_with_mode(
        false,
        true,
        RealPowerArithmeticMode::MixedDoubleThenSingle {
            single_operand: RealPowerOperand::Base,
        },
    );
    provider.exact_scalar.store(true, Ordering::SeqCst);
    let control = PowerScalarControl::DoubleBits(2.0_f64.to_bits());
    let plan = || {
        crate::power_service::plan_scalar_power_for_dense_output(
            provider,
            resident(&[2, 2], ProviderElementType::F32),
            control,
            RealPowerOperand::Base,
            &[1, 1],
            ProviderOperationIdentity::new("test.power"),
        )
        .unwrap()
        .unwrap()
    };
    provider.dense_output_padding.store(16, Ordering::SeqCst);
    let padded = plan();
    assert!(matches!(
        padded
            .feasibility_query(PowerResultDomain::Real)
            .unwrap()
            .workload
            .real_power_scalar
            .unwrap()
            .input,
        Some(crate::PowerInputDescription::PlannedDense {
            requested_bytes: 32
        })
    ));
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let bind = |planned: &PowerSymbolicPlan| {
        crate::power_service::prepare_resident_scalar_power_against_plan(
            provider,
            &input.handle,
            planned,
            control,
            RealPowerOperand::Base,
            &[1, 1],
            ProviderOperationIdentity::new("test.power"),
        )
    };
    assert!(
        bind(&padded).is_err(),
        "future rounded extent is not authority to accept a different live backing"
    );
    provider.dense_output_padding.store(0, Ordering::SeqCst);
    assert!(
        padded.replan(provider).unwrap().is_none(),
        "changed allocator projection invalidates the future receipt"
    );
    let exact = plan();
    let prepared = bind(&exact).unwrap();
    assert!(matches!(
        prepared
            .symbolic_plan()
            .feasibility_query(PowerResultDomain::Real)
            .unwrap()
            .workload
            .real_power_scalar
            .unwrap()
            .input,
        Some(crate::PowerInputDescription::Live { .. })
    ));
    assert_eq!(prepared.control(), control);
    assert_eq!(input.probe.retirements(), 0);
    assert_eq!(
        provider.probes.load(Ordering::SeqCst),
        0,
        "planning/binding never probes device payloads"
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn typed_future_allocator_extent_survives_live_binding() {
    let provider = provider(false);
    let control = PowerScalarControl::SingleBits(1.8_f32.to_bits());
    let operation = ProviderOperationIdentity::new("test.power");
    let plan = crate::power_service::plan_scalar_power_for_dense_output(
        provider,
        resident(&[2, 2], ProviderElementType::F32),
        control,
        RealPowerOperand::Base,
        &[1, 1],
        operation.clone(),
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        plan.physical_recipe().route,
        PowerPhysicalRoute::TypedDomain
    );
    assert!(matches!(
        plan.resident_input_description(),
        Some(crate::PowerInputDescription::PlannedDense { .. })
    ));
    let source = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F32, GpuTensorStorage::Real)
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
    assert!(matches!(
        prepared.symbolic_plan().resident_input_description(),
        Some(crate::PowerInputDescription::Live { .. })
    ));
    assert!(plan.matches_live_typed_binding(prepared.symbolic_plan(), &source.handle));
    let changed_view = provider
        .owner
        .allocate_zeroed(vec![2, 3], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    assert!(!plan.matches_live_typed_binding(prepared.symbolic_plan(), &changed_view.handle));
    let mut altered_control = prepared.symbolic_plan().clone();
    altered_control.control = PowerScalarControl::SingleBits(2.0_f32.to_bits());
    assert!(!plan.matches_live_typed_binding(&altered_control, &source.handle));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 0);
    assert_eq!(source.probe.retirements(), 0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn unavailable_exact_byte_geometry_does_not_fabricate_a_proposal_or_hide_typed_route() {
    let mode = RealPowerArithmeticMode::MixedDoubleThenSingle {
        single_operand: RealPowerOperand::Base,
    };
    let provider = provider_with_mode(false, true, mode);
    provider.exact_scalar.store(true, Ordering::SeqCst);
    let source = resident(&[u64::MAX / 2], ProviderElementType::F32);
    assert!(source.checked_element_count().is_some());
    assert!(source.checked_byte_len().is_none());
    let plan = plan_resident_scalar_power(
        provider,
        source,
        PowerScalarControl::DoubleBits(2.0_f64.to_bits()),
        RealPowerOperand::Base,
        &[1, 1],
        ProviderOperationIdentity::new("test.power"),
    )
    .unwrap()
    .unwrap();
    // This recorder's existing typed admission is intentionally independent
    // of byte geometry. The exact specialization must not override it.
    assert_eq!(
        plan.physical_recipe().route,
        PowerPhysicalRoute::TypedDomain
    );
    assert_eq!(provider.exact_proposals.load(Ordering::SeqCst), 0);
}

#[test]
fn certified_family_is_capability_bound_and_does_not_admit_other_mixed_forms() {
    let mode = RealPowerArithmeticMode::MixedDoubleThenSingle {
        single_operand: RealPowerOperand::Base,
    };
    let provider = provider_with_mode(false, true, mode);
    provider.exact_scalar.store(true, Ordering::SeqCst);
    for shape in [&[2, 4, 5][..], &[1, 40], &[40, 1], &[1, 1]] {
        for exponent in [0.0_f64, -0.0, 1.0, 2.0] {
            let control = PowerScalarControl::DoubleBits(exponent.to_bits());
            let plan = plan_resident_scalar_power(
                provider,
                resident(shape, ProviderElementType::F32),
                control,
                RealPowerOperand::Base,
                &[1, 1],
                ProviderOperationIdentity::new("test.power"),
            )
            .unwrap()
            .unwrap();
            assert_eq!(plan.control(), control);
            assert_eq!(plan.mode(), mode);
            assert_eq!(
                plan.physical_recipe().route,
                PowerPhysicalRoute::ExactSingleBase
            );
            assert_eq!(
                plan.physical_recipe().scalar,
                PowerScalarConstruction::EmbeddedExactControl
            );
            assert!(!plan.physical_recipe().widen_resident);
            assert!(!plan.physical_recipe().single_publication);
            assert!(plan.feasibility_query(PowerResultDomain::Complex).is_none());
            let work = plan.physical_work(PowerResultDomain::Real).unwrap();
            assert!(!work.paired_domain_probe);
            assert!(work.widened_resident.is_none());
            assert!(work.single_publication.is_none());
            assert_eq!(work.scalar.residency, ProviderResidency::Host);
            assert_eq!(work.output.element_type, ProviderElementType::F32);
        }
    }
    let source = resident(&[2, 4, 5], ProviderElementType::F32);
    for control in [
        PowerScalarControl::SingleBits(2.0_f32.to_bits()),
        PowerScalarControl::DoubleBits(3.0_f64.to_bits()),
        PowerScalarControl::DoubleBits(2.0_f64.to_bits() + 1),
    ] {
        assert!(exact_single_base_request(&source, control, RealPowerOperand::Base, 160).is_none());
    }
    let control = PowerScalarControl::DoubleBits(2.0_f64.to_bits());
    assert!(exact_single_base_request(&source, control, RealPowerOperand::Exponent, 160).is_none());
    assert!(exact_single_base_request(
        &resident(&[2, 4, 5], ProviderElementType::F64),
        control,
        RealPowerOperand::Base,
        320
    )
    .is_none());
    let mut strided = source.clone();
    strided.layout = ProviderLayout::Strided;
    assert!(exact_single_base_request(&strided, control, RealPowerOperand::Base, 160).is_none());
    strided.shape = vec![5, 8];
    assert!(exact_single_base_request(&strided, control, RealPowerOperand::Base, 160).is_some());
    let empty = resident(&[0, 4, 5], ProviderElementType::F32);
    assert!(exact_single_base_request(&empty, control, RealPowerOperand::Base, 4096).is_some());
    assert!(plan_resident_scalar_power(
        provider,
        source,
        control,
        RealPowerOperand::Base,
        &[1, 2],
        ProviderOperationIdentity::new("test.power")
    )
    .is_err());
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 0);
    assert_eq!(provider.widenings.load(Ordering::SeqCst), 0);
    assert_eq!(provider.power_calls.load(Ordering::SeqCst), 0);
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
    assert_eq!(work.scalar.shape, [1, 1]);
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
    assert!(plan.matches_live_typed_binding(actual.symbolic_plan(), &source.handle));
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
    assert!(plan.matches_live_typed_binding(prepared.symbolic_plan(), &source.handle));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn exact_scalar_upload_projection_keeps_the_actual_host_scalar_view() {
    let provider = provider(false);
    for shape in [&[1_usize][..], &[1, 1, 1][..]] {
        let plan = plan_resident_scalar_power(
            provider,
            resident(&[2, 4, 5], ProviderElementType::F32),
            PowerScalarControl::SingleBits(1.8_f32.to_bits()),
            RealPowerOperand::Base,
            shape,
            ProviderOperationIdentity::new("test.power"),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            plan.physical_recipe().scalar,
            PowerScalarConstruction::ExactTypedUpload
        );
        assert_eq!(
            plan.physical_work(PowerResultDomain::Real)
                .unwrap()
                .scalar
                .shape,
            shape
                .iter()
                .map(|&extent| extent as u64)
                .collect::<Vec<_>>()
        );
    }
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
    assert_eq!(
        plan.physical_recipe().scalar,
        PowerScalarConstruction::ProviderFill
    );
    assert!(plan.physical_work(PowerResultDomain::Real).is_some());
    assert!(plan.physical_work(PowerResultDomain::Complex).is_none());
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 0);
}
