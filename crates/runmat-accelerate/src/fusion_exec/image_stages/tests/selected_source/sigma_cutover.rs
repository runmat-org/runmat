//! Actual compiled source cutover oracle; provider receipts are deterministic
//! API/lifetime fixtures, not hardware numerical or timing measurements.
use super::super::real_elementwise_provider::RealFault;
use super::*;
use runmat_accelerate_api::ProviderRealElementwiseOperation as Operation;
use std::future::Future;

async fn bound(
    provider: &'static StagedProvider,
    shape: [usize; 3],
) -> (
    ImageNormalizeStageSession,
    GpuTensorHandle,
    RetirementProbe,
    ImageSourceInvocation,
) {
    // The canonical physical query declares zero-dispatch same-class casts.
    // Use the fixture's actual same-allocation owner route, not a fresh copy.
    *provider.alias_same_class_cast.lock().unwrap() = true;
    let graph = compiled_image_graph(shape);
    let groups = detect_fusion_groups(&graph);
    let plans = FusionPlan::from_graph(&graph, &groups);
    let plan = plans
        .groups
        .iter()
        .find(|plan| plan.group.kind == FusionKind::ImageNormalize)
        .unwrap();
    let FusionPattern::ImageNormalize(pattern) = plan.group.pattern.as_ref().unwrap() else {
        panic!("actual selected image");
    };
    let input = provider
        .allocations
        .allocate_zeroed(
            shape.to_vec(),
            NumericElementType::F32,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let values = plan
        .inputs
        .iter()
        .map(|id| {
            if *id == pattern.input {
                Value::GpuTensor(input.handle.clone())
            } else {
                exact_source_input(&graph, *id, plan.group.span.start)
            }
        })
        .collect::<Vec<_>>();
    let invocation =
        ImageSourceInvocation::bind(&graph, plan, &values, ProviderPrecision::F64, None)
            .await
            .unwrap();
    let session = ImageNormalizeStageSession::begin_selected(
        provider,
        &input.handle,
        &invocation,
        ImageArithmeticProof::fixture(),
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        provider.calls.load(Ordering::SeqCst),
        0,
        "all recipes planned before first selected effect"
    );
    assert!(provider.real_controls.lock().unwrap().is_empty());
    (session, input.handle, input.probe, invocation)
}

fn operations(
    session: &ImageNormalizeStageSession,
) -> &super::super::super::sigma_operations::SelectedSigmaOperationsPlan {
    session
        .selected_mean_plans
        .as_ref()
        .unwrap()
        .sigma
        .as_ref()
        .unwrap()
        .operations
        .as_ref()
        .unwrap()
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn compiled_sigma_cutover_uses_exact_controls_ids_and_observed_receipts() {
    for shape in [[2, 3, 4], [2, 4, 5]] {
        let provider = provider_with_precision(None, None, ProviderPrecision::F64);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let (mut session, input, input_probe, invocation) = bound(provider, shape).await;
        let source = invocation.selected_sigma_operations().unwrap();
        assert_eq!(operations(&session).source_ids(), source.source_ids());
        assert_eq!(operations(&session).identity(), source.identity().unwrap());
        let planned = operations(&session).proposals();
        let (add, sqrt) = (planned.0.clone(), planned.1.clone());
        let recipe_ids = operations(&session).recipe_identities();
        assert!(add.requested_incremental_peak_bytes().unwrap() > 0);
        assert!(sqrt.requested_incremental_peak_bytes().unwrap() > 0);
        let mean = session.mean().await.unwrap().0;
        let raw_mean = provider.ordered_output.lock().unwrap().clone().unwrap();
        assert!(mean.shares_allocation_with(&raw_mean));
        let mean_cast = session
            .selected_mean_plans
            .as_ref()
            .unwrap()
            .mu
            .last()
            .unwrap()
            .casts
            .last()
            .expect("actual compiled single(mean) publication");
        assert_eq!(mean_cast.source.element_type, mean_cast.output.element_type);
        assert_eq!(mean_cast.resources.dispatches, Some(0));
        assert_eq!(mean.physical().requested_bytes(), mean_cast.output_bytes);
        let before_sigma = provider.trace();
        drop(Box::pin(session.sigma()));
        assert!(matches!(session.state, State::MeanPublished(_)));
        assert_eq!(
            provider.trace(),
            before_sigma,
            "unpolled Sigma has no effects"
        );
        *provider.mutate_cast_residency.lock().unwrap() = true;
        *provider.mutate_protected_residency.lock().unwrap() = Some(mean.clone());
        let sigma = session.sigma().await.unwrap().0;
        assert_eq!(
            provider.real_controls.lock().unwrap().as_slice(),
            [
                Operation::ScalarAdd {
                    control: source.control(),
                    mode: source.mode()
                },
                Operation::Sqrt
            ]
        );
        let receipts = operations(&session).receipts();
        let ids = source.source_ids();
        assert_eq!(receipts.len(), 2);
        assert_eq!(
            (receipts[0].recipe_identity, receipts[1].recipe_identity),
            recipe_ids
        );
        assert_eq!(
            (receipts[0].source_input, receipts[0].source_output),
            (ids.0, ids.2)
        );
        assert_eq!(
            (receipts[1].source_input, receipts[1].source_output),
            (ids.2, ids.3)
        );
        for (receipt, physical) in receipts.iter().zip([add, sqrt]) {
            assert_eq!(receipt.selected_identity, source.identity().unwrap());
            assert_eq!(
                receipt.observed.output_buffer_bytes,
                Some(physical.output_buffer_bytes)
            );
            assert_eq!(
                receipt.observed.uniform_buffer_bytes,
                physical.uniform_buffer_bytes
            );
            assert_eq!(
                receipt.observed.uniform_allocations,
                physical.uniform_allocations
            );
            assert_eq!(receipt.observed.dispatches, physical.dispatches);
            assert_eq!(
                receipt.observed.queue_submissions,
                physical.queue_submissions
            );
        }
        assert!(
            session.sigma().await.is_err(),
            "published Sigma is not replayable"
        );
        drop(session);
        for handle in [&input, &mean, &sigma] {
            handle.allocation_lease().validate_live().unwrap();
        }
        assert_eq!(input_probe.retirements(), 0);
        assert!(crate::fusion_residency::is_resident(&mean));
        assert!(crate::fusion_residency::is_resident(&sigma));
    }
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn observed_rejections_and_original_hook_errors_are_terminal_without_replay() {
    for sqrt in [false, true] {
        for fault in [
            RealFault::ObservedOutput,
            RealFault::ObservedUniformBytes,
            RealFault::ObservedUniformCount,
            RealFault::ObservedDispatches,
            RealFault::ObservedSubmissions,
            RealFault::HookError,
            RealFault::WrongShape,
            RealFault::WrongClass,
            RealFault::WrongOwner,
        ] {
            let original = matches!(&fault, RealFault::HookError);
            let provider = provider_with_precision(None, None, ProviderPrecision::F64);
            let _selected = ThreadProviderGuard::set(Some(provider));
            let (mut session, input, probe, _) = bound(provider, [2, 3, 4]).await;
            let mean = session.mean().await.unwrap().0;
            *provider.real_fault.lock().unwrap() = Some((sqrt, fault));
            *provider.mutate_cast_residency.lock().unwrap() = true;
            *provider.mutate_protected_residency.lock().unwrap() = Some(mean.clone());
            let error = session.sigma().await.err().unwrap();
            assert!(error.is_after_effect());
            if original {
                assert!(error
                    .to_string()
                    .contains("original selected Sigma hook cause"));
                assert!(std::error::Error::source(&error)
                    .unwrap()
                    .downcast_ref::<std::io::Error>()
                    .is_some());
            }
            assert!(matches!(session.state, State::Failed));
            let count = provider.real_controls.lock().unwrap().len();
            assert_eq!(count, if sqrt { 2 } else { 1 });
            assert_eq!(operations(&session).receipts().len(), usize::from(sqrt));
            assert!(session.sigma().await.is_err());
            assert_eq!(provider.real_controls.lock().unwrap().len(), count);
            assert_eq!(probe.retirements(), 0);
            for handle in [&input, &mean] {
                handle.allocation_lease().validate_live().unwrap();
                assert!(crate::fusion_residency::is_resident(handle));
            }
            assert_eq!(
                provider.probes().last().unwrap().retirements(),
                1,
                "owned malformed or errored output is retired"
            );
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn same_allocation_published_alias_is_never_retired_by_shared_recipe_rejection() {
    for sqrt in [false, true] {
        let provider = provider_with_precision(None, None, ProviderPrecision::F64);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let (mut session, input, probe, _) = bound(provider, [2, 3, 4]).await;
        let mean = session.mean().await.unwrap().0;
        *provider.real_fault.lock().unwrap() = Some((sqrt, RealFault::Alias(mean.clone())));
        *provider.mutate_protected_residency.lock().unwrap() = Some(mean.clone());
        let error = session.sigma().await.err().unwrap();
        assert!(error.is_after_effect());
        mean.allocation_lease().validate_live().unwrap();
        input.allocation_lease().validate_live().unwrap();
        assert_eq!(probe.retirements(), 0);
        assert!(crate::fusion_residency::is_resident(&mean));
        assert!(session.sigma().await.is_err());
    }
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn pending_selected_operation_cancellation_restores_borrowed_metadata_and_closes_cutover() {
    for sqrt in [false, true] {
        let provider = provider_with_precision(None, None, ProviderPrecision::F64);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let (mut session, input, probe, _) = bound(provider, [2, 3, 4]).await;
        let mean = session.mean().await.unwrap().0;
        *provider.real_fault.lock().unwrap() = Some((sqrt, RealFault::Pending));
        *provider.mutate_cast_residency.lock().unwrap() = true;
        *provider.mutate_protected_residency.lock().unwrap() = Some(mean.clone());
        let mut future = Box::pin(session.sigma());
        assert!(matches!(
            future.as_mut().poll(&mut std::task::Context::from_waker(
                futures::task::noop_waker_ref()
            )),
            std::task::Poll::Pending
        ));
        let pending = provider.probes().last().unwrap().clone();
        assert_eq!(pending.retirements(), 0);
        assert!(!crate::fusion_residency::is_resident(&mean));
        drop(future);
        assert_eq!(pending.retirements(), 1);
        assert!(matches!(session.state, State::InFlight));
        assert!(session.sigma().await.is_err());
        assert_eq!(operations(&session).receipts().len(), usize::from(sqrt));
        assert!(crate::fusion_residency::is_resident(&mean));
        mean.allocation_lease().validate_live().unwrap();
        input.allocation_lease().validate_live().unwrap();
        assert_eq!(probe.retirements(), 0);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn sealed_source_mutation_and_plan_rejection_have_no_legacy_fallback() {
    let provider = provider_with_precision(None, None, ProviderPrecision::F64);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input, probe, invocation) = bound(provider, [2, 4, 5]).await;
    let source = invocation.selected_sigma_operations().unwrap();
    let bindings = invocation.selected_moment_bindings().unwrap();
    let publication = session
        .selected_mean_plans
        .as_ref()
        .unwrap()
        .sigma
        .as_ref()
        .unwrap();
    let moments = session.program.moment_program.as_ref().unwrap();
    for change in 0..4 {
        let mut wrong = moments.clone();
        match change {
            0 => wrong.sigma.source_input = wrong.sigma.raw_output,
            1 => wrong.variance.last_mut().unwrap().public_output = wrong.sigma.raw_output,
            2 => wrong.sigma.raw_fact = super::super::ordered_mean::moment_fact(),
            _ => wrong.sigma.raw_fact.shape = runmat_types::ShapeFact::Unknown,
        }
        if change == 2 {
            wrong.sigma.raw_fact.kind =
                runmat_types::ValueKindFact::Numeric(runmat_types::NumericFact {
                    class: runmat_types::NumericClass::Double,
                    domain: runmat_types::NumericDomain::Real,
                });
        }
        assert!(super::super::super::sigma_operations::plan(
            provider,
            &source,
            publication,
            &wrong,
            &bindings
        )
        .unwrap()
        .is_none());
    }
    for sqrt in [false, true] {
        *provider.real_fault.lock().unwrap() = Some((sqrt, RealFault::RejectPlan));
        assert!(ImageNormalizeStageSession::begin_selected(
            provider,
            &input,
            &invocation,
            ImageArithmeticProof::fixture()
        )
        .unwrap()
        .is_none());
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        assert!(provider.real_controls.lock().unwrap().is_empty());
    }
    *provider.real_fault.lock().unwrap() = None;
    let mean = session.mean().await.unwrap().0;
    let before = provider.trace();
    session.program.epsilon =
        runmat_accelerate_api::scalar_program::ScalarLiteral::F32Bits(0.5_f32.to_bits());
    let error = session.sigma().await.err().unwrap();
    assert!(error.is_after_effect());
    assert_eq!(
        provider.trace(),
        before,
        "changed control rejected before Sigma's first hook"
    );
    assert!(provider.real_controls.lock().unwrap().is_empty());
    assert!(session.sigma().await.is_err());
    mean.allocation_lease().validate_live().unwrap();
    assert_eq!(probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn actual_compiled_mixed_double_control_is_not_erased_or_sent_to_legacy_hooks() {
    let provider = provider_with_precision(None, None, ProviderPrecision::F64);
    let _selected = ThreadProviderGuard::set(Some(provider));
    for (source, exact) in [("0.125", 0.125_f64), ("0.1", 0.1_f64)] {
        let graph = compiled_image_graph_with_epsilon([2, 3, 4], source);
        let groups = detect_fusion_groups(&graph);
        let plans = FusionPlan::from_graph(&graph, &groups);
        let plan = plans
            .groups
            .iter()
            .find(|plan| plan.group.kind == FusionKind::ImageNormalize)
            .unwrap();
        let FusionPattern::ImageNormalize(pattern) = plan.group.pattern.as_ref().unwrap() else {
            panic!("selected mixed image");
        };
        let input = provider
            .allocations
            .allocate_zeroed(
                vec![2, 3, 4],
                NumericElementType::F32,
                GpuTensorStorage::Real,
            )
            .unwrap();
        let values = plan
            .inputs
            .iter()
            .map(|id| {
                if *id == pattern.input {
                    Value::GpuTensor(input.handle.clone())
                } else {
                    exact_source_input(&graph, *id, plan.group.span.start)
                }
            })
            .collect::<Vec<_>>();
        let invocation =
            ImageSourceInvocation::bind(&graph, plan, &values, ProviderPrecision::F64, None)
                .await
                .unwrap();
        assert_eq!(
            invocation.requested_staged_workload().unwrap().epsilon,
            runmat_accelerate_api::scalar_program::ScalarLiteral::F64Bits(exact.to_bits()),
            "the selected request retains the original Double control before provider admission"
        );
        assert!(
            invocation.selected_sigma_operations().is_none(),
            "authentic mixed-double Add has no native equivalence authority"
        );
        assert!(ImageNormalizeStageSession::begin_selected(
            provider,
            &input.handle,
            &invocation,
            ImageArithmeticProof::fixture()
        )
        .unwrap()
        .is_none());
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        assert!(provider.real_controls.lock().unwrap().is_empty());
        assert_eq!(input.probe.retirements(), 0);
    }
}
