use super::*;
use crate::fusion::image_reduction::{ImageSelectedMomentBindings, ImageSelectedMomentConversion};
use runmat_accelerate_api::{
    PowerScalarControl, ProviderImageMomentConversion, ProviderImageMomentProgram,
    ProviderImageMomentPublication, ProviderRealMeanNaNMode,
};
use runmat_types::{
    DimensionFact, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};
use std::future::Future;

pub(super) fn moment_fact() -> ValueFact {
    moment_fact_shape([2, 1, 1])
}

fn moment_fact_shape(shape: [usize; 3]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::Shaped {
            dims: vec![
                DimensionFact::Known(shape[0]),
                DimensionFact::Known(shape[1]),
                DimensionFact::Known(shape[2]),
            ],
        },
        StorageFact::Dense,
    )
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn nested_dim_cast_has_bound_future_extent_and_no_post_effect_replay() {
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
            power_service::selected_program(PowerScalarControl::SingleBits(1.8_f32.to_bits()));
        program.mu = vec![
            ProviderImageMeanCall {
                hook: ProviderImageMeanHook::ReduceDim(1),
                output_class: ProviderImageMeanClass::Native,
                shape: [2, 1, 4],
            },
            ProviderImageMeanCall {
                hook: ProviderImageMeanHook::ReduceDim(2),
                output_class: ProviderImageMeanClass::Native,
                shape: [2, 1, 1],
            },
        ];
        program.variance = vec![ProviderImageMeanCall {
            hook: ProviderImageMeanHook::ReduceNd([1, 2]),
            output_class: ProviderImageMeanClass::Native,
            shape: [2, 1, 1],
        }];
        let mut inner = ProviderImageMomentPublication {
            source_input: 17,
            raw_output: 18,
            raw_fact: moment_fact_shape([2, 1, 4]),
            conversions: vec![ProviderImageMomentConversion {
                source_input: 18,
                source_output: 19,
                output_fact: moment_fact_shape([2, 1, 4]),
            }],
            public_output: 19,
        };
        program.moment_program = Some(ProviderImageMomentProgram {
            mu: vec![inner.clone(), publication(19, 20)],
            variance: vec![publication(22, 24)],
            sigma: publication(28, 29),
        });
        let binding = runmat_builtins::SINGLE_CATALOG_ENTRY
            .binding_identity(&runmat_builtins::SINGLE_CATALOG_ENTRY.bindings[0]);
        let bindings = ImageSelectedMomentBindings::fixture(
            runmat_execution::Digest::sha256(b"test-only-selected-nested-source"),
            vec![
                vec![ImageSelectedMomentConversion::fixture(
                    18,
                    19,
                    binding,
                    ProviderPrecision::F32,
                )],
                vec![],
            ],
            vec![vec![]],
            vec![],
            crate::fusion::image_reduction::ImageSelectedSigmaRaw::fixture(
                28,
                29,
                runmat_builtins::SQRT_CATALOG_ENTRY
                    .binding_identity(&runmat_builtins::SQRT_CATALOG_ENTRY.bindings[0]),
                runmat_execution::Digest::sha256(b"test-only-selected-nested-source"),
            ),
        );
        assert!(program.moment_program.as_ref().unwrap().validate());
        let powers = super::super::power::plan_selected(provider, &program)
            .unwrap()
            .unwrap();
        let planned = super::super::ordered_mean::plan_selected(
            provider,
            &input.handle,
            &program,
            &powers,
            Some(&bindings),
        )
        .unwrap()
        .unwrap();
        assert_eq!(planned.mu.len(), 2);
        assert_eq!(planned.mu[0].recipe.axes(), [1]);
        assert_eq!(planned.mu[0].casts[0].output_bytes, 32);
        assert_eq!(planned.mu[1].recipe.source().shape, vec![2, 1, 4]);
        assert_eq!(planned.mu[1].recipe.source_buffer_bytes(), 32);
        assert!(provider.trace().is_empty());
        let mut forged_binding = bindings.clone();
        forged_binding.replace_mu_for_test(
            0,
            vec![bindings.mu()[0][0].clone().with_binding_for_test(
                runmat_builtins::BuiltinBindingIdentity {
                    variant: "forged",
                    ..binding
                },
            )],
        );
        assert!(super::super::ordered_mean::plan_selected(
            provider,
            &input.handle,
            &program,
            &powers,
            Some(&forged_binding),
        )
        .unwrap()
        .is_none());
        inner.conversions[0].source_input = 99;
        let mut forged_source = program.clone();
        forged_source.moment_program.as_mut().unwrap().mu[0] = inner;
        assert!(super::super::ordered_mean::plan_selected(
            provider,
            &input.handle,
            &forged_source,
            &powers,
            Some(&bindings),
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
        *provider.alias_same_class_cast.lock().unwrap() = true;
        let mut session = ImageNormalizeStageSession {
            provider,
            input: input.handle.clone(),
            desc: desc.clone(),
            program: program.clone(),
            selected_power_plans: Some(powers),
            selected_mean_plans: Some(planned),
            selected_binary_plans: None,
            selected_real_plans: None,
            reserved_mu: None,
            cold_mu: None,
            retained_view_permit: None,
            provider_metadata_permit: None,
            state: State::BeforeMean,
        };
        let mean = futures::executor::block_on(session.mean()).unwrap();
        assert_eq!(mean.0.shape(), [2, 1, 1]);
        assert_eq!(
            provider.trace(),
            [
                StageHook::Nd(vec![1]),
                StageHook::Single,
                StageHook::Nd(vec![2])
            ]
        );
        let powers = super::super::power::plan_selected(provider, &program)
            .unwrap()
            .unwrap();
        let planned = super::super::ordered_mean::plan_selected(
            provider,
            &input.handle,
            &program,
            &powers,
            Some(&bindings),
        )
        .unwrap()
        .unwrap();
        let continued_desc = desc.clone();
        let continued_program = program.clone();
        let mut rejected = ImageNormalizeStageSession {
            provider,
            input: input.handle.clone(),
            desc,
            program,
            selected_power_plans: Some(powers),
            selected_mean_plans: Some(planned),
            selected_binary_plans: None,
            selected_real_plans: None,
            reserved_mu: None,
            cold_mu: None,
            retained_view_permit: None,
            provider_metadata_permit: None,
            state: State::BeforeMean,
        };
        *provider.reject_cast_query.lock().unwrap() = true;
        let error = futures::executor::block_on(rejected.mean()).unwrap_err();
        assert!(error.is_after_effect());
        assert!(matches!(rejected.state, State::Failed));
        assert_eq!(provider.trace().last(), Some(&StageHook::Nd(vec![1])));
        assert_eq!(provider.trace().len(), 4);
        assert!(input.handle.allocation_lease().validate_live().is_ok());
        assert!(mean.0.allocation_lease().validate_live().is_ok());
        *provider.reject_cast_query.lock().unwrap() = false;
        let powers = super::super::power::plan_selected(provider, &continued_program)
            .unwrap()
            .unwrap();
        let planned = super::super::ordered_mean::plan_selected(
            provider,
            &input.handle,
            &continued_program,
            &powers,
            Some(&bindings),
        )
        .unwrap()
        .unwrap();
        let alias_desc = continued_desc.clone();
        let alias_program = continued_program.clone();
        let mut changed_during_cast = ImageNormalizeStageSession {
            provider,
            input: input.handle.clone(),
            desc: continued_desc,
            program: continued_program,
            selected_power_plans: Some(powers),
            selected_mean_plans: Some(planned),
            selected_binary_plans: None,
            selected_real_plans: None,
            reserved_mu: None,
            cold_mu: None,
            retained_view_permit: None,
            provider_metadata_permit: None,
            state: State::BeforeMean,
        };
        *provider.reject_cast_query_after_cast.lock().unwrap() = true;
        let error = futures::executor::block_on(changed_during_cast.mean()).unwrap_err();
        assert!(error.is_after_effect());
        assert!(matches!(changed_during_cast.state, State::Failed));
        assert_eq!(
            provider
                .trace()
                .iter()
                .rev()
                .take(2)
                .cloned()
                .collect::<Vec<_>>(),
            [StageHook::Single, StageHook::Nd(vec![1])]
        );
        assert_eq!(provider.trace().len(), 6);
        assert!(input.handle.allocation_lease().validate_live().is_ok());
        assert!(mean.0.allocation_lease().validate_live().is_ok());
        *provider.reject_cast_query.lock().unwrap() = false;
        *provider.reject_cast_query_after_cast.lock().unwrap() = false;
        let powers = super::super::power::plan_selected(provider, &alias_program)
            .unwrap()
            .unwrap();
        let planned = super::super::ordered_mean::plan_selected(
            provider,
            &input.handle,
            &alias_program,
            &powers,
            Some(&bindings),
        )
        .unwrap()
        .unwrap();
        let mut wrong_alias = ImageNormalizeStageSession {
            provider,
            input: input.handle.clone(),
            desc: alias_desc,
            program: alias_program,
            selected_power_plans: Some(powers),
            selected_mean_plans: Some(planned),
            selected_binary_plans: None,
            selected_real_plans: None,
            reserved_mu: None,
            cold_mu: None,
            retained_view_permit: None,
            provider_metadata_permit: None,
            state: State::BeforeMean,
        };
        *provider.alias_same_class_cast.lock().unwrap() = false;
        let error = futures::executor::block_on(wrong_alias.mean()).unwrap_err();
        assert!(error.is_after_effect());
        assert!(matches!(wrong_alias.state, State::Failed));
        assert_eq!(provider.trace().len(), 8);
        assert!(input.handle.allocation_lease().validate_live().is_ok());
        assert!(mean.0.allocation_lease().validate_live().is_ok());
    });
}

pub(super) fn publication(source_input: u64, raw_output: u64) -> ProviderImageMomentPublication {
    ProviderImageMomentPublication {
        source_input,
        raw_output,
        raw_fact: moment_fact(),
        conversions: Vec::new(),
        public_output: raw_output,
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn ordered_vec_mean_is_source_bound_and_future_variance_is_symbolic() {
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
        let (_, mut program) =
            power_service::selected_program(PowerScalarControl::SingleBits(1.8_f32.to_bits()));
        let one_vec_call = ProviderImageMeanCall {
            hook: ProviderImageMeanHook::ReduceNd([1, 2]),
            output_class: ProviderImageMeanClass::Native,
            shape: [2, 1, 1],
        };
        program.mu = vec![one_vec_call];
        program.variance = vec![one_vec_call];
        program.moment_program = Some(ProviderImageMomentProgram {
            mu: vec![publication(17, 18)],
            variance: vec![publication(22, 24)],
            sigma: publication(28, 29),
        });
        let powers = super::super::power::plan_selected(provider, &program)
            .unwrap()
            .unwrap();
        let planned = super::super::ordered_mean::plan_selected(
            provider,
            &input.handle,
            &program,
            &powers,
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(planned.mu[0].recipe.axes(), [1, 2]);
        assert_eq!(
            planned.mu[0].recipe.output_precision(),
            ProviderPrecision::F32
        );
        assert_eq!(
            planned.mu[0].recipe.nan_mode(),
            ProviderRealMeanNaNMode::Include
        );
        assert_eq!(planned.variance[0].source_input, 22);
        assert_eq!(planned.variance[0].recipe.source().shape, vec![2, 3, 4]);
        assert!(
            provider.trace().is_empty(),
            "symbolic/live prepare is effect-free"
        );
        let mut wrong_source = program.clone();
        wrong_source.moment_program.as_mut().unwrap().variance[0].source_input = 99;
        assert!(super::super::ordered_mean::plan_selected(
            provider,
            &input.handle,
            &wrong_source,
            &powers,
            None,
        )
        .unwrap()
        .is_none());
        let mut wrong_class = program.clone();
        wrong_class.moment_program.as_mut().unwrap().mu[0]
            .raw_fact
            .kind = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
        assert!(super::super::ordered_mean::plan_selected(
            provider,
            &input.handle,
            &wrong_class,
            &powers,
            None,
        )
        .unwrap()
        .is_none());
        let mut nested = program.clone();
        nested.mu = vec![
            ProviderImageMeanCall {
                hook: ProviderImageMeanHook::ReduceDim(1),
                output_class: ProviderImageMeanClass::Native,
                shape: [2, 1, 4],
            },
            ProviderImageMeanCall {
                hook: ProviderImageMeanHook::ReduceDim(2),
                output_class: ProviderImageMeanClass::Native,
                shape: [2, 1, 1],
            },
        ];
        nested.moment_program.as_mut().unwrap().mu = vec![publication(17, 18), publication(18, 19)];
        assert!(super::super::ordered_mean::plan_selected(
            provider,
            &input.handle,
            &nested,
            &powers,
            None,
        )
        .unwrap()
        .is_none());
        assert!(provider.trace().is_empty());
        let mu = super::super::ordered_mean::execute_call(
            provider,
            super::super::ordered_mean::OrderedMeanExecutionContext {
                current: &input.handle,
                image_input: &input.handle,
                scratch: &[],
                published: &[],
                prior_effect: false,
            },
            &program.mu[0],
            &program.moment_program.as_ref().unwrap().mu[0],
            &planned.mu[0],
        )
        .unwrap();
        assert_eq!(mu.shape(), [2, 1, 1]);
        assert_eq!(provider.trace(), [StageHook::Nd(vec![1, 2])]);
        let changed_axes = ProviderImageMeanCall {
            hook: ProviderImageMeanHook::ReduceDim(1),
            ..program.mu[0]
        };
        assert!(super::super::ordered_mean::execute_call(
            provider,
            super::super::ordered_mean::OrderedMeanExecutionContext {
                current: &input.handle,
                image_input: &input.handle,
                scratch: &[],
                published: &[],
                prior_effect: true,
            },
            &changed_axes,
            &program.moment_program.as_ref().unwrap().mu[0],
            &planned.mu[0],
        )
        .is_err());
        let changed = publication(99, 18);
        assert!(super::super::ordered_mean::execute_call(
            provider,
            super::super::ordered_mean::OrderedMeanExecutionContext {
                current: &input.handle,
                image_input: &input.handle,
                scratch: &[],
                published: &[],
                prior_effect: true,
            },
            &program.mu[0],
            &changed,
            &planned.mu[0],
        )
        .is_err());
        assert_eq!(provider.trace(), [StageHook::Nd(vec![1, 2])]);
        assert_eq!(input.probe.retirements(), 0);
        *provider.fault.lock().unwrap() = Some((1, StageFault::Alias(mu.clone())));
        let alias_error = super::super::ordered_mean::execute_call(
            provider,
            super::super::ordered_mean::OrderedMeanExecutionContext {
                current: &input.handle,
                image_input: &input.handle,
                scratch: &[],
                published: &[&mu],
                prior_effect: true,
            },
            &program.mu[0],
            &program.moment_program.as_ref().unwrap().mu[0],
            &planned.mu[0],
        )
        .unwrap_err();
        assert!(alias_error.is_after_effect());
        assert!(mu.allocation_lease().validate_live().is_ok());
        assert_eq!(provider.probes()[0].retirements(), 0);
        input.probe.mark_unavailable();
        let lease_error = super::super::ordered_mean::execute_call(
            provider,
            super::super::ordered_mean::OrderedMeanExecutionContext {
                current: &input.handle,
                image_input: &input.handle,
                scratch: &[],
                published: &[&mu],
                prior_effect: true,
            },
            &program.mu[0],
            &program.moment_program.as_ref().unwrap().mu[0],
            &planned.mu[0],
        )
        .unwrap_err();
        assert!(lease_error.is_after_effect());
        assert_eq!(
            std::error::Error::source(&lease_error)
                .and_then(|source| source
                    .downcast_ref::<runmat_accelerate_api::allocation::AllocationAcquisitionError>(
                )),
            Some(&runmat_accelerate_api::allocation::AllocationAcquisitionError::NotLive),
        );
        assert_eq!(
            provider.trace().len(),
            2,
            "no effect after the live lease fails"
        );
        assert!(mu.allocation_lease().validate_live().is_ok());
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn ordered_completion_rejects_forged_resources_and_output_views() {
    runmat_gc::gc_test_context(|| {
        let receipt_faults = [
            OrderedCompletionFault::RequestedPeak,
            OrderedCompletionFault::ObservedPeak,
            OrderedCompletionFault::OutputBytes,
            OrderedCompletionFault::Submissions,
        ];
        for fault in receipt_faults {
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
            let (_, mut program) =
                power_service::selected_program(PowerScalarControl::SingleBits(1.8_f32.to_bits()));
            let call = ProviderImageMeanCall {
                hook: ProviderImageMeanHook::ReduceNd([1, 2]),
                output_class: ProviderImageMeanClass::Native,
                shape: [2, 1, 1],
            };
            program.mu = vec![call];
            program.variance = vec![call];
            program.moment_program = Some(ProviderImageMomentProgram {
                mu: vec![publication(17, 18)],
                variance: vec![publication(22, 24)],
                sigma: publication(28, 29),
            });
            let powers = super::super::power::plan_selected(provider, &program)
                .unwrap()
                .unwrap();
            let plan = super::super::ordered_mean::plan_selected(
                provider,
                &input.handle,
                &program,
                &powers,
                None,
            )
            .unwrap()
            .unwrap();
            *provider.ordered_completion_fault.lock().unwrap() = Some(fault);
            let error = super::super::ordered_mean::execute_call(
                provider,
                super::super::ordered_mean::OrderedMeanExecutionContext {
                    current: &input.handle,
                    image_input: &input.handle,
                    scratch: &[],
                    published: &[],
                    prior_effect: false,
                },
                &call,
                &program.moment_program.as_ref().unwrap().mu[0],
                &plan.mu[0],
            )
            .unwrap_err();
            assert!(error.is_after_effect());
            assert!(!error.is_unsupported());
            assert_eq!(provider.trace(), [StageHook::Nd(vec![1, 2])]);
            assert!(input.handle.allocation_lease().validate_live().is_ok());
            assert_eq!(input.probe.retirements(), 0);
        }
        for stage_fault in [
            StageFault::WrongShape,
            StageFault::WrongClass,
            StageFault::WrongOwner,
        ] {
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
            let (_, mut program) =
                power_service::selected_program(PowerScalarControl::SingleBits(1.8_f32.to_bits()));
            let call = ProviderImageMeanCall {
                hook: ProviderImageMeanHook::ReduceNd([1, 2]),
                output_class: ProviderImageMeanClass::Native,
                shape: [2, 1, 1],
            };
            program.mu = vec![call];
            program.variance = vec![call];
            program.moment_program = Some(ProviderImageMomentProgram {
                mu: vec![publication(17, 18)],
                variance: vec![publication(22, 24)],
                sigma: publication(28, 29),
            });
            let powers = super::super::power::plan_selected(provider, &program)
                .unwrap()
                .unwrap();
            let plan = super::super::ordered_mean::plan_selected(
                provider,
                &input.handle,
                &program,
                &powers,
                None,
            )
            .unwrap()
            .unwrap();
            *provider.fault.lock().unwrap() = Some((0, stage_fault));
            let error = super::super::ordered_mean::execute_call(
                provider,
                super::super::ordered_mean::OrderedMeanExecutionContext {
                    current: &input.handle,
                    image_input: &input.handle,
                    scratch: &[],
                    published: &[],
                    prior_effect: false,
                },
                &call,
                &program.moment_program.as_ref().unwrap().mu[0],
                &plan.mu[0],
            )
            .unwrap_err();
            assert!(error.is_after_effect());
            assert_eq!(provider.trace(), [StageHook::Nd(vec![1, 2])]);
            assert!(input.handle.allocation_lease().validate_live().is_ok());
            assert_eq!(input.probe.retirements(), 0);
        }
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
        let (_, mut program) =
            power_service::selected_program(PowerScalarControl::SingleBits(1.8_f32.to_bits()));
        let call = ProviderImageMeanCall {
            hook: ProviderImageMeanHook::ReduceNd([1, 2]),
            output_class: ProviderImageMeanClass::Native,
            shape: [2, 1, 1],
        };
        program.mu = vec![call];
        program.variance = vec![call];
        program.moment_program = Some(ProviderImageMomentProgram {
            mu: vec![publication(17, 18)],
            variance: vec![publication(22, 24)],
            sigma: publication(28, 29),
        });
        let powers = super::super::power::plan_selected(provider, &program)
            .unwrap()
            .unwrap();
        let plan = super::super::ordered_mean::plan_selected(
            provider,
            &input.handle,
            &program,
            &powers,
            None,
        )
        .unwrap()
        .unwrap();
        *provider.invalidate_on_call.lock().unwrap() = Some((0, input.probe.clone()));
        let error = super::super::ordered_mean::execute_call(
            provider,
            super::super::ordered_mean::OrderedMeanExecutionContext {
                current: &input.handle,
                image_input: &input.handle,
                scratch: &[],
                published: &[],
                prior_effect: false,
            },
            &call,
            &program.moment_program.as_ref().unwrap().mu[0],
            &plan.mu[0],
        )
        .unwrap_err();
        assert!(error.is_after_effect());
        assert_eq!(
            std::error::Error::source(&error)
                .and_then(|source| source
                    .downcast_ref::<runmat_accelerate_api::allocation::AllocationAcquisitionError>(
                )),
            Some(&runmat_accelerate_api::allocation::AllocationAcquisitionError::NotLive),
        );
        assert_eq!(provider.trace(), [StageHook::Nd(vec![1, 2])]);
        assert_eq!(input.probe.retirements(), 0);
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pending_selected_mean_publication_restores_raw_residency_and_prevents_replay() {
    runmat_gc::gc_test_context(|| {
        let provider = provider_with_precision(None, Some(1), ProviderPrecision::F64);
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
            power_service::selected_program(PowerScalarControl::SingleBits(1.8_f32.to_bits()));
        let call = ProviderImageMeanCall {
            hook: ProviderImageMeanHook::ReduceNd([1, 2]),
            output_class: ProviderImageMeanClass::Native,
            shape: [2, 1, 1],
        };
        program.mu = vec![call];
        program.variance = vec![call];
        let mut mu = publication(17, 18);
        mu.conversions.push(ProviderImageMomentConversion {
            source_input: 18,
            source_output: 19,
            output_fact: moment_fact(),
        });
        mu.public_output = 19;
        program.moment_program = Some(ProviderImageMomentProgram {
            mu: vec![mu],
            variance: vec![publication(22, 24)],
            sigma: publication(28, 29),
        });
        let identity = runmat_execution::Digest::sha256(b"pending-selected-mean-source");
        let binding = runmat_builtins::SINGLE_CATALOG_ENTRY
            .binding_identity(&runmat_builtins::SINGLE_CATALOG_ENTRY.bindings[0]);
        let bindings = ImageSelectedMomentBindings::fixture(
            identity,
            vec![vec![ImageSelectedMomentConversion::fixture(
                18,
                19,
                binding,
                ProviderPrecision::F32,
            )]],
            vec![vec![]],
            vec![],
            crate::fusion::image_reduction::ImageSelectedSigmaRaw::fixture(
                28,
                29,
                runmat_builtins::SQRT_CATALOG_ENTRY
                    .binding_identity(&runmat_builtins::SQRT_CATALOG_ENTRY.bindings[0]),
                identity,
            ),
        );
        let powers = super::super::power::plan_selected(provider, &program)
            .unwrap()
            .unwrap();
        let ordered = super::super::ordered_mean::plan_selected(
            provider,
            &input.handle,
            &program,
            &powers,
            Some(&bindings),
        )
        .unwrap()
        .unwrap();
        *provider.mutate_cast_residency.lock().unwrap() = true;
        let mut session = ImageNormalizeStageSession {
            provider,
            input: input.handle.clone(),
            desc,
            program,
            selected_power_plans: Some(powers),
            selected_mean_plans: Some(ordered),
            selected_binary_plans: None,
            selected_real_plans: None,
            reserved_mu: None,
            cold_mu: None,
            retained_view_permit: None,
            provider_metadata_permit: None,
            state: State::BeforeMean,
        };
        let mut future = Box::pin(session.mean());
        assert!(matches!(
            future.as_mut().poll(&mut std::task::Context::from_waker(
                futures::task::noop_waker_ref()
            )),
            std::task::Poll::Pending
        ));
        let raw = provider.ordered_output.lock().unwrap().clone().unwrap();
        assert!(!crate::fusion_residency::is_resident(&raw));
        assert_eq!(provider.trace(), [StageHook::Nd(vec![1, 2])]);
        drop(future);
        assert!(crate::fusion_residency::is_resident(&raw));
        assert!(matches!(session.state, State::InFlight));
        assert!(input.handle.allocation_lease().validate_live().is_ok());
        assert!(raw.allocation_lease().validate_live().is_ok());
        assert_eq!(input.probe.retirements(), 0);
        assert_eq!(provider.probes()[0].retirements(), 0);
    });
}
