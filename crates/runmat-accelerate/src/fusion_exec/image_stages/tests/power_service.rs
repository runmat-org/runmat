use super::*;
use runmat_accelerate_api::{
    power_service::PowerScalarConstruction, PowerScalarControl, ProviderImagePowerCall,
    ProviderImagePowerProgram, ProviderOperationIdentity, RealPowerArithmeticMode,
    RealPowerOperand,
};
use runmat_types::NumericClass;
use std::future::Future;

fn powers(gamma: PowerScalarControl) -> ProviderImagePowerProgram {
    let square = ProviderImagePowerCall {
        output: 22,
        base: 20,
        exponent: 21,
        base_class: NumericClass::Single,
        exponent_class: NumericClass::Double,
        control: PowerScalarControl::DoubleBits(2.0_f64.to_bits()),
        exponent_shape: vec![1, 1],
        resident_operand: RealPowerOperand::Base,
        mode: RealPowerArithmeticMode::MixedDoubleThenSingle {
            single_operand: RealPowerOperand::Base,
        },
        output_class: NumericClass::Single,
        operation: ProviderOperationIdentity::new(
            runmat_builtins::POWER_CATALOG_ENTRY.identity.name,
        ),
        binding_variant: "default".into(),
        semantic_revision: 6,
    };
    let final_power = ProviderImagePowerCall {
        output: 39,
        base: 37,
        exponent: 38,
        base_class: NumericClass::Single,
        exponent_class: match gamma {
            PowerScalarControl::SingleBits(_) => NumericClass::Single,
            PowerScalarControl::DoubleBits(_) => NumericClass::Double,
        },
        control: gamma,
        exponent_shape: vec![1, 1],
        resident_operand: RealPowerOperand::Base,
        mode: RealPowerArithmeticMode::for_classes(
            NumericClass::Single,
            match gamma {
                PowerScalarControl::SingleBits(_) => NumericClass::Single,
                PowerScalarControl::DoubleBits(_) => NumericClass::Double,
            },
        )
        .unwrap(),
        output_class: NumericClass::Single,
        operation: ProviderOperationIdentity::new(
            runmat_builtins::POWER_CATALOG_ENTRY.identity.name,
        ),
        binding_variant: "default".into(),
        semantic_revision: 6,
    };
    ProviderImagePowerProgram {
        revision: ProviderImagePowerProgram::REVISION,
        square,
        final_power,
    }
}

pub(super) fn selected_program(
    gamma: PowerScalarControl,
) -> (ImageNormalizeDescriptor, ProviderImageStageWorkload) {
    let mut source = source_recipe();
    let mut desc = descriptor();
    desc.gamma = Some(gamma.value());
    source.final_power.represented_exponent = gamma.value();
    let program = source
        .staged_workload_selected(
            &desc,
            powers(gamma),
            runmat_accelerate_api::scalar_program::ScalarLiteral::F32Bits(
                (desc.epsilon as f32).to_bits(),
            ),
        )
        .unwrap();
    (desc, program)
}

fn power_session(
    provider: &'static StagedProvider,
    input: GpuTensorHandle,
    desc: ImageNormalizeDescriptor,
    program: ProviderImageStageWorkload,
    plans: super::super::power::ImagePowerPlans,
) -> ImageNormalizeStageSession {
    ImageNormalizeStageSession {
        provider,
        input,
        desc,
        program,
        selected_power_plans: Some(plans),
        selected_mean_plans: None,
        selected_binary_plans: None,
        selected_real_plans: None,
        reserved_mu: None,
        cold_mu: None,
        retained_view_permit: None,
        provider_metadata_permit: None,
        state: State::BeforeMean,
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn shared_service_executes_mixed_square_native_single_and_distinct_double_gamma() {
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
        let (desc, program) = selected_program(PowerScalarControl::SingleBits(1.8_f32.to_bits()));
        let plans = super::super::power::plan_selected(provider, &program)
            .unwrap()
            .unwrap();
        assert_eq!(
            plans.square.physical_recipe().mode,
            RealPowerArithmeticMode::MixedDoubleThenSingle {
                single_operand: RealPowerOperand::Base
            }
        );
        assert!(plans.square.physical_recipe().widen_resident);
        assert_eq!(
            plans.final_power.physical_recipe().scalar,
            PowerScalarConstruction::ExactTypedUpload
        );
        assert!(provider.trace().is_empty(), "symbolic plan has no effects");
        let session = power_session(provider, input.handle.clone(), desc, program, plans);
        let selected = session.program.selected_powers.as_ref().unwrap();
        let planned = session.selected_power_plans.as_ref().unwrap();
        let square = futures::executor::block_on(session.execute_stage_power(
            &input.handle,
            Some((&selected.square, &planned.square)),
            0,
            &[&input.handle],
        ))
        .unwrap();
        let final_power = futures::executor::block_on(session.execute_stage_power(
            &input.handle,
            Some((&selected.final_power, &planned.final_power)),
            0,
            &[&input.handle],
        ))
        .unwrap();
        assert_eq!(
            runmat_accelerate_api::handle_precision(&square),
            Some(ProviderPrecision::F32)
        );
        assert_eq!(
            runmat_accelerate_api::handle_precision(&final_power),
            Some(ProviderPrecision::F32)
        );
        assert_eq!(
            provider.controls.lock().unwrap().as_slice(),
            &[
                PowerScalarControl::DoubleBits(2.0_f64.to_bits()),
                PowerScalarControl::SingleBits(1.8_f32.to_bits()),
            ]
        );
        assert_eq!(input.probe.retirements(), 0);
        let (desc, program) =
            selected_program(PowerScalarControl::DoubleBits(f64::from(1.8_f32).to_bits()));
        let plans = super::super::power::plan_selected(provider, &program)
            .unwrap()
            .unwrap();
        assert_eq!(
            plans.final_power.mode(),
            RealPowerArithmeticMode::MixedDoubleThenSingle {
                single_operand: RealPowerOperand::Base
            }
        );
        let session = power_session(provider, input.handle.clone(), desc, program, plans);
        let selected = session.program.selected_powers.as_ref().unwrap();
        let planned = session.selected_power_plans.as_ref().unwrap();
        let _double_gamma = futures::executor::block_on(session.execute_stage_power(
            &input.handle,
            Some((&selected.final_power, &planned.final_power)),
            0,
            &[&input.handle],
        ))
        .unwrap();
        assert_eq!(
            provider.controls.lock().unwrap().last(),
            Some(&PowerScalarControl::DoubleBits(
                f64::from(1.8_f32).to_bits()
            ))
        );
        assert_eq!(input.probe.retirements(), 0);
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn changed_control_or_future_view_rejects_before_provider_effects() {
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
        let (_, mut program) = selected_program(PowerScalarControl::SingleBits(1.8_f32.to_bits()));
        program.selected_powers.as_mut().unwrap().square.control =
            PowerScalarControl::SingleBits(2.0_f32.to_bits());
        assert!(super::super::power::plan_selected(provider, &program)
            .unwrap()
            .is_none());
        assert!(provider.trace().is_empty());
        let (_, program) = selected_program(PowerScalarControl::SingleBits(1.8_f32.to_bits()));
        let plans = super::super::power::plan_selected(provider, &program)
            .unwrap()
            .unwrap();
        let foreign = provider
            .allocations
            .allocate_zeroed(
                vec![2, 3, 4],
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )
            .unwrap();
        let call = &program.selected_powers.as_ref().unwrap().square;
        let rejected =
            runmat_accelerate_api::power_service::prepare_resident_scalar_power_against_plan(
                provider,
                &foreign.handle,
                &plans.square,
                call.control,
                call.resident_operand,
                &[1, 1],
                call.operation.clone(),
            );
        assert!(rejected.is_err());
        assert!(provider.trace().is_empty());
        assert_eq!(input.probe.retirements(), 0);
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn f32_only_provider_rejects_original_double_square_before_effects() {
    runmat_gc::gc_test_context(|| {
        let provider = provider_with_precision(None, None, ProviderPrecision::F32);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let (_, program) = selected_program(PowerScalarControl::SingleBits(1.8_f32.to_bits()));
        assert_eq!(
            program.selected_powers.as_ref().unwrap().square.control,
            PowerScalarControl::DoubleBits(2.0_f64.to_bits())
        );
        assert!(super::super::power::plan_selected(provider, &program)
            .unwrap()
            .is_none());
        assert!(provider.trace().is_empty());
        assert!(provider.controls.lock().unwrap().is_empty());
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn exact_singleton_source_rank_survives_plan_and_live_prepare() {
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
        let (desc, mut program) =
            selected_program(PowerScalarControl::SingleBits(1.8_f32.to_bits()));
        program
            .selected_powers
            .as_mut()
            .unwrap()
            .final_power
            .exponent_shape = vec![1, 1, 1];
        assert!(program.validate());
        let plans = super::super::power::plan_selected(provider, &program)
            .unwrap()
            .unwrap();
        assert_eq!(plans.final_power.scalar_shape(), [1, 1, 1]);
        let session = power_session(provider, input.handle.clone(), desc, program, plans);
        let selected = session.program.selected_powers.as_ref().unwrap();
        let planned = session.selected_power_plans.as_ref().unwrap();
        assert!(futures::executor::block_on(session.execute_stage_power(
            &input.handle,
            Some((&selected.final_power, &planned.final_power)),
            0,
            &[&input.handle],
        ))
        .is_ok());
        assert_eq!(
            provider.controls.lock().unwrap().as_slice(),
            [PowerScalarControl::SingleBits(1.8_f32.to_bits())]
        );
        assert_eq!(
            provider.uploaded_shapes.lock().unwrap().as_slice(),
            [vec![1, 1, 1]]
        );
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn dropped_pending_shared_service_keeps_borrowed_input_live() {
    runmat_gc::gc_test_context(|| {
        let provider = provider_with_precision(None, Some(2), ProviderPrecision::F64);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .allocations
            .allocate_zeroed(
                vec![2, 3, 4],
                NumericElementType::F32,
                GpuTensorStorage::Real,
            )
            .unwrap();
        let (desc, program) = selected_program(PowerScalarControl::SingleBits(1.8_f32.to_bits()));
        let plans = super::super::power::plan_selected(provider, &program)
            .unwrap()
            .unwrap();
        let session = power_session(provider, input.handle.clone(), desc, program, plans);
        let selected = session.program.selected_powers.as_ref().unwrap();
        let planned = session.selected_power_plans.as_ref().unwrap();
        let protected = [&input.handle];
        let mut future = Box::pin(session.execute_stage_power(
            &input.handle,
            Some((&selected.square, &planned.square)),
            0,
            &protected,
        ));
        assert!(matches!(
            future.as_mut().poll(&mut std::task::Context::from_waker(
                futures::task::noop_waker_ref()
            )),
            std::task::Poll::Pending
        ));
        drop(future);
        assert_eq!(input.probe.retirements(), 0);
        assert!(input.handle.allocation_lease().validate_live().is_ok());
        assert_eq!(provider.trace(), [StageHook::Double, StageHook::Upload]);
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_plan_absence_never_replays_legacy_control() {
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
        let (desc, program) = selected_program(PowerScalarControl::SingleBits(1.8_f32.to_bits()));
        let plans = super::super::power::plan_selected(provider, &program)
            .unwrap()
            .unwrap();
        let mut session = power_session(provider, input.handle.clone(), desc, program, plans);
        session.selected_power_plans = None;
        assert!(futures::executor::block_on(session.execute_stage_power(
            &input.handle,
            None,
            0,
            &[&input.handle],
        ))
        .is_err());
        assert!(provider.trace().is_empty());
        assert!(provider.controls.lock().unwrap().is_empty());
        assert_eq!(input.probe.retirements(), 0);
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn shared_scalar_rejects_alias_of_input_or_published_moment() {
    runmat_gc::gc_test_context(|| {
        for alias_case in 0..3 {
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
            let mean = provider
                .allocations
                .allocate_zeroed(
                    vec![2, 1, 1],
                    NumericElementType::F32,
                    GpuTensorStorage::Real,
                )
                .unwrap();
            let same_shape_scratch = provider
                .allocations
                .allocate_zeroed(
                    vec![2, 3, 4],
                    NumericElementType::F64,
                    GpuTensorStorage::Real,
                )
                .unwrap();
            let alias = match alias_case {
                0 => input.handle.clone(),
                1 => mean.handle.clone(),
                _ => same_shape_scratch.handle.clone(),
            };
            *provider.fault.lock().unwrap() = Some((1, StageFault::Alias(alias)));
            let (desc, program) =
                selected_program(PowerScalarControl::SingleBits(1.8_f32.to_bits()));
            let plans = super::super::power::plan_selected(provider, &program)
                .unwrap()
                .unwrap();
            let session = power_session(provider, input.handle.clone(), desc, program, plans);
            let selected = session.program.selected_powers.as_ref().unwrap();
            let planned = session.selected_power_plans.as_ref().unwrap();
            assert!(futures::executor::block_on(session.execute_stage_power(
                &input.handle,
                Some((&selected.square, &planned.square)),
                0,
                &[&input.handle, &mean.handle, &same_shape_scratch.handle],
            ))
            .is_err());
            assert_eq!(provider.trace(), [StageHook::Double, StageHook::Upload]);
            assert_eq!(input.probe.retirements(), 0);
            assert_eq!(mean.probe.retirements(), 0);
            assert_eq!(same_shape_scratch.probe.retirements(), 0);
        }
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn invalidated_borrowed_lease_stops_before_next_power_effect() {
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
        let mean = provider
            .allocations
            .allocate_zeroed(
                vec![2, 1, 1],
                NumericElementType::F32,
                GpuTensorStorage::Real,
            )
            .unwrap();
        *provider.invalidate_on_call.lock().unwrap() = Some((0, mean.probe.clone()));
        let (desc, program) = selected_program(PowerScalarControl::SingleBits(1.8_f32.to_bits()));
        let plans = super::super::power::plan_selected(provider, &program)
            .unwrap()
            .unwrap();
        let session = power_session(provider, input.handle.clone(), desc, program, plans);
        let selected = session.program.selected_powers.as_ref().unwrap();
        let planned = session.selected_power_plans.as_ref().unwrap();
        let error = futures::executor::block_on(session.execute_stage_power(
            &input.handle,
            Some((&selected.square, &planned.square)),
            0,
            &[&input.handle, &mean.handle],
        ))
        .unwrap_err();
        assert!(
            !error.is_unsupported(),
            "prior widening forbids host replay"
        );
        assert!(error.is_after_effect());
        assert!(std::error::Error::source(&error)
            .and_then(|source| {
                source
                    .downcast_ref::<runmat_accelerate_api::allocation::AllocationAcquisitionError>()
            })
            .is_some());
        assert_eq!(provider.trace(), [StageHook::Double]);
        assert!(provider.controls.lock().unwrap().is_empty());
        assert_eq!(provider.probes()[0].retirements(), 1);
        assert_eq!(input.probe.retirements(), 0);
        assert_eq!(mean.probe.retirements(), 0);
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pending_power_restores_published_borrowed_residency_on_drop() {
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
        let mean = provider
            .allocations
            .allocate_zeroed(
                vec![2, 1, 1],
                NumericElementType::F32,
                GpuTensorStorage::Real,
            )
            .unwrap();
        runmat_accelerate_api::mark_residency(&mean.handle);
        assert!(crate::fusion_residency::is_resident(&mean.handle));
        *provider.mutate_protected_residency.lock().unwrap() = Some(mean.handle.clone());
        let (desc, program) = selected_program(PowerScalarControl::SingleBits(1.8_f32.to_bits()));
        let plans = super::super::power::plan_selected(provider, &program)
            .unwrap()
            .unwrap();
        let session = power_session(provider, input.handle.clone(), desc, program, plans);
        let selected = session.program.selected_powers.as_ref().unwrap();
        let planned = session.selected_power_plans.as_ref().unwrap();
        let protected = [&input.handle, &mean.handle];
        let mut future = Box::pin(session.execute_stage_power(
            &input.handle,
            Some((&selected.square, &planned.square)),
            0,
            &protected,
        ));
        assert!(matches!(
            future.as_mut().poll(&mut std::task::Context::from_waker(
                futures::task::noop_waker_ref()
            )),
            std::task::Poll::Pending
        ));
        assert!(!crate::fusion_residency::is_resident(&mean.handle));
        drop(future);
        assert!(crate::fusion_residency::is_resident(&mean.handle));
        assert!(input.handle.allocation_lease().validate_live().is_ok());
        assert!(mean.handle.allocation_lease().validate_live().is_ok());
        assert_eq!(input.probe.retirements(), 0);
        assert_eq!(mean.probe.retirements(), 0);
    });
}
