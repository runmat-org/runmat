use super::*;
use crate::fusion::image_reduction::{
    ImageSelectedMomentBindings, ImageSelectedMomentConversion, ImageSelectedSigmaRaw,
};
use runmat_accelerate_api::{
    PowerScalarControl, ProviderImageMeanClass, ProviderImageMeanHook,
    ProviderImageMomentConversion, ProviderImageMomentProgram,
};
use runmat_types::{NumericClass, NumericDomain, NumericFact, ValueKindFact};
use std::future::Future;

fn selected_sigma_fixture(
    provider: &'static StagedProvider,
    input: &GpuTensorHandle,
) -> (
    ImageNormalizeDescriptor,
    ProviderImageStageWorkload,
    super::super::power::ImagePowerPlans,
    super::super::ordered_mean::ImageOrderedMeanPlans,
    ImageSelectedMomentBindings,
) {
    let (desc, mut program) =
        power_service::selected_program(PowerScalarControl::SingleBits(1.8_f32.to_bits()));
    let mean = ProviderImageMeanCall {
        hook: ProviderImageMeanHook::ReduceNd([1, 2]),
        output_class: ProviderImageMeanClass::Native,
        shape: [2, 1, 1],
    };
    program.mu = vec![mean];
    program.variance = vec![mean];
    let mut sigma = ordered_mean::publication(28, 29);
    sigma.conversions.push(ProviderImageMomentConversion {
        source_input: 29,
        source_output: 30,
        output_fact: ordered_mean::moment_fact(),
    });
    sigma.public_output = 30;
    program.moment_program = Some(ProviderImageMomentProgram {
        mu: vec![ordered_mean::publication(17, 18)],
        variance: vec![ordered_mean::publication(22, 24)],
        sigma,
    });
    let identity = runmat_execution::Digest::sha256(b"selected-sigma-cast-fixture");
    let binding = runmat_builtins::SINGLE_CATALOG_ENTRY
        .binding_identity(&runmat_builtins::SINGLE_CATALOG_ENTRY.bindings[0]);
    let sigma_binding = runmat_builtins::SQRT_CATALOG_ENTRY
        .binding_identity(&runmat_builtins::SQRT_CATALOG_ENTRY.bindings[0]);
    let bindings = ImageSelectedMomentBindings::fixture(
        identity,
        vec![vec![]],
        vec![vec![]],
        vec![ImageSelectedMomentConversion::fixture(
            29,
            30,
            binding,
            ProviderPrecision::F32,
        )],
        ImageSelectedSigmaRaw::fixture(28, 29, sigma_binding, identity),
    );
    let powers = super::super::power::plan_selected(provider, &program)
        .unwrap()
        .unwrap();
    let means = super::super::ordered_mean::plan_selected(
        provider,
        input,
        &program,
        &powers,
        Some(&bindings),
    )
    .unwrap()
    .unwrap();
    (desc, program, powers, means, bindings)
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn sigma_cast_is_planned_before_effects_and_revalidated_against_live_source() {
    runmat_gc::gc_test_context(|| {
        let provider = provider_with_precision(None, None, ProviderPrecision::F64);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .allocations
            .allocate_zeroed(
                vec![2, 3, 4],
                NumericElementType::F32,
                GpuTensorStorage::Real,
            )
            .unwrap();
        let (_, program, powers, plans, bindings) = selected_sigma_fixture(provider, &input.handle);
        let sigma = plans.sigma.as_ref().unwrap();
        assert_eq!(sigma.casts.len(), 1);
        assert_eq!(sigma.raw_source_identity().0, 28);
        assert_eq!(sigma.raw_source_identity().1, 29);
        assert_eq!(sigma.casts[0].source_input, 29);
        assert_eq!(sigma.casts[0].source_output, 30);
        assert!(provider.trace().is_empty());
        let mut changed_source = program.clone();
        changed_source
            .moment_program
            .as_mut()
            .unwrap()
            .sigma
            .source_input = 99;
        assert!(super::super::ordered_mean::plan_selected(
            provider,
            &input.handle,
            &changed_source,
            &powers,
            Some(&bindings),
        )
        .unwrap()
        .is_none());
        let mut changed_class = program.clone();
        changed_class
            .moment_program
            .as_mut()
            .unwrap()
            .sigma
            .raw_fact
            .kind = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
        assert!(super::super::ordered_mean::plan_selected(
            provider,
            &input.handle,
            &changed_class,
            &powers,
            Some(&bindings),
        )
        .unwrap()
        .is_none());
        let wrong_sqrt_binding = ImageSelectedMomentBindings::fixture(
            bindings.semantic_identity(),
            bindings.mu().to_vec(),
            bindings.variance().to_vec(),
            bindings.sigma().to_vec(),
            ImageSelectedSigmaRaw::fixture(
                28,
                29,
                runmat_builtins::SINGLE_CATALOG_ENTRY
                    .binding_identity(&runmat_builtins::SINGLE_CATALOG_ENTRY.bindings[0]),
                bindings.semantic_identity(),
            ),
        );
        assert!(super::super::ordered_mean::plan_selected(
            provider,
            &input.handle,
            &program,
            &powers,
            Some(&wrong_sqrt_binding),
        )
        .unwrap()
        .is_none());
        *provider.reject_cast_query.lock().unwrap() = true;
        assert!(super::super::ordered_mean::plan_selected(
            provider,
            &input.handle,
            &program,
            &powers,
            Some(&bindings),
        )
        .unwrap()
        .is_none());
        *provider.reject_cast_query.lock().unwrap() = false;
        *provider.reject_numeric_output_projection.lock().unwrap() = true;
        assert!(super::super::ordered_mean::plan_selected(
            provider,
            &input.handle,
            &program,
            &powers,
            Some(&bindings),
        )
        .unwrap()
        .is_none());
        *provider.reject_numeric_output_projection.lock().unwrap() = false;
        let shifted = provider
            .allocations
            .allocate_zeroed(
                vec![2, 1, 1],
                NumericElementType::F32,
                GpuTensorStorage::Real,
            )
            .unwrap();
        assert!(sigma.revalidate_source(provider, &shifted.handle).unwrap());
        assert!(sigma.revalidate_raw(provider, &shifted.handle).unwrap());
        *provider.numeric_output_bytes_override.lock().unwrap() = Some(999);
        assert!(!sigma.revalidate_source(provider, &shifted.handle).unwrap());
        assert!(!sigma.revalidate_raw(provider, &shifted.handle).unwrap());
        *provider.numeric_output_bytes_override.lock().unwrap() = None;
        let wrong_class = provider
            .allocations
            .allocate_zeroed(
                vec![2, 1, 1],
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )
            .unwrap();
        assert!(!sigma.revalidate_raw(provider, &wrong_class.handle).unwrap());
        let foreign = provider
            .foreign_allocations
            .allocate_zeroed(
                vec![2, 1, 1],
                NumericElementType::F32,
                GpuTensorStorage::Real,
            )
            .unwrap();
        assert!(!sigma.revalidate_raw(provider, &foreign.handle).unwrap());
        assert_eq!(
            program.moment_program.as_ref().unwrap().sigma.raw_output,
            29
        );
        assert!(provider.trace().is_empty());
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn sigma_cast_pending_drop_retains_borrowed_raw() {
    runmat_gc::gc_test_context(|| {
        let provider = provider_with_precision(None, Some(0), ProviderPrecision::F64);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .allocations
            .allocate_zeroed(
                vec![2, 3, 4],
                NumericElementType::F32,
                GpuTensorStorage::Real,
            )
            .unwrap();
        let (desc, program, powers, means, _) = selected_sigma_fixture(provider, &input.handle);
        let raw = provider
            .allocations
            .allocate_zeroed(
                vec![2, 1, 1],
                NumericElementType::F32,
                GpuTensorStorage::Real,
            )
            .unwrap();
        let published = provider
            .allocations
            .allocate_zeroed(
                vec![2, 1, 1],
                NumericElementType::F32,
                GpuTensorStorage::Real,
            )
            .unwrap();
        runmat_accelerate_api::mark_residency(&raw.handle);
        assert!(crate::fusion_residency::is_resident(&raw.handle));
        let session = ImageNormalizeStageSession {
            provider,
            input: input.handle.clone(),
            desc,
            program: program.clone(),
            selected_power_plans: Some(powers),
            selected_mean_plans: Some(means),
            selected_binary_plans: None,
            selected_real_plans: None,
            reserved_mu: None,
            cold_mu: None,
            retained_view_permit: None,
            provider_metadata_permit: None,
            state: State::InFlight,
        };
        let mut scratch = Vec::new();
        let selected = &session
            .selected_mean_plans
            .as_ref()
            .unwrap()
            .sigma
            .as_ref()
            .unwrap()
            .casts;
        let protected = [&published.handle];
        let mut future = Box::pin(session.publish_moment(
            raw.handle.clone(),
            &program.moment_program.as_ref().unwrap().sigma,
            &mut scratch,
            &protected,
            Some(selected),
        ));
        assert!(matches!(
            future.as_mut().poll(&mut std::task::Context::from_waker(
                futures::task::noop_waker_ref()
            )),
            std::task::Poll::Pending
        ));
        drop(future);
        assert!(crate::fusion_residency::is_resident(&raw.handle));
        assert!(raw.handle.allocation_lease().validate_live().is_ok());
        assert!(published.handle.allocation_lease().validate_live().is_ok());
        assert_eq!(raw.probe.retirements(), 0);
        assert_eq!(published.probe.retirements(), 0);
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn sigma_cast_rejects_published_alias_without_retiring_it() {
    runmat_gc::gc_test_context(|| {
        let provider = provider_with_precision(None, None, ProviderPrecision::F64);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .allocations
            .allocate_zeroed(
                vec![2, 3, 4],
                NumericElementType::F32,
                GpuTensorStorage::Real,
            )
            .unwrap();
        let (desc, program, powers, means, _) = selected_sigma_fixture(provider, &input.handle);
        let raw = provider
            .allocations
            .allocate_zeroed(
                vec![2, 1, 1],
                NumericElementType::F32,
                GpuTensorStorage::Real,
            )
            .unwrap();
        let published = provider
            .allocations
            .allocate_zeroed(
                vec![2, 1, 1],
                NumericElementType::F32,
                GpuTensorStorage::Real,
            )
            .unwrap();
        let admission = ImageNormalizeStageSession {
            provider,
            input: input.handle.clone(),
            desc: desc.clone(),
            program: program.clone(),
            selected_power_plans: None,
            selected_mean_plans: None,
            selected_binary_plans: None,
            selected_real_plans: None,
            reserved_mu: None,
            cold_mu: None,
            retained_view_permit: None,
            provider_metadata_permit: None,
            state: State::InFlight,
        };
        assert!(admission
            .admit_reduction(
                published.handle.clone(),
                &raw.handle,
                &[2, 1, 1],
                Some(ProviderPrecision::F32),
                &[],
                &[&published.handle],
            )
            .unwrap_err()
            .is_after_effect());
        assert!(published.handle.allocation_lease().validate_live().is_ok());
        assert_eq!(published.probe.retirements(), 0);
        *provider.fault.lock().unwrap() = Some((0, StageFault::Alias(published.handle.clone())));
        let session = ImageNormalizeStageSession {
            provider,
            input: input.handle.clone(),
            desc,
            program: program.clone(),
            selected_power_plans: Some(powers),
            selected_mean_plans: Some(means),
            selected_binary_plans: None,
            selected_real_plans: None,
            reserved_mu: None,
            cold_mu: None,
            retained_view_permit: None,
            provider_metadata_permit: None,
            state: State::InFlight,
        };
        let selected = &session
            .selected_mean_plans
            .as_ref()
            .unwrap()
            .sigma
            .as_ref()
            .unwrap()
            .casts;
        let error = futures::executor::block_on(session.publish_moment(
            raw.handle.clone(),
            &program.moment_program.as_ref().unwrap().sigma,
            &mut Vec::new(),
            &[&published.handle],
            Some(selected),
        ))
        .unwrap_err();
        assert!(error.is_after_effect());
        assert_eq!(provider.trace(), [StageHook::Single]);
        assert!(published.handle.allocation_lease().validate_live().is_ok());
        assert_eq!(published.probe.retirements(), 0);
        assert_eq!(input.probe.retirements(), 0);
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn sigma_same_class_cast_publishes_the_authenticated_input_alias() {
    runmat_gc::gc_test_context(|| {
        let provider = provider_with_precision(None, None, ProviderPrecision::F64);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .allocations
            .allocate_zeroed(
                vec![2, 3, 4],
                NumericElementType::F32,
                GpuTensorStorage::Real,
            )
            .unwrap();
        let (desc, program, powers, means, _) = selected_sigma_fixture(provider, &input.handle);
        let raw = provider
            .allocations
            .allocate_zeroed(
                vec![2, 1, 1],
                NumericElementType::F32,
                GpuTensorStorage::Real,
            )
            .unwrap();
        *provider.alias_same_class_cast.lock().unwrap() = true;
        let session = ImageNormalizeStageSession {
            provider,
            input: input.handle.clone(),
            desc,
            program: program.clone(),
            selected_power_plans: Some(powers),
            selected_mean_plans: Some(means),
            selected_binary_plans: None,
            selected_real_plans: None,
            reserved_mu: None,
            cold_mu: None,
            retained_view_permit: None,
            provider_metadata_permit: None,
            state: State::InFlight,
        };
        let selected = &session
            .selected_mean_plans
            .as_ref()
            .unwrap()
            .sigma
            .as_ref()
            .unwrap()
            .casts;
        let public = futures::executor::block_on(session.publish_moment(
            raw.handle.clone(),
            &program.moment_program.as_ref().unwrap().sigma,
            &mut Vec::new(),
            &[],
            Some(selected),
        ))
        .unwrap();
        assert!(public.shares_allocation_with(&raw.handle));
        assert_eq!(provider.trace(), [StageHook::Single]);
        assert!(public.allocation_lease().validate_live().is_ok());
        assert_eq!(raw.probe.retirements(), 0);
    });
}
