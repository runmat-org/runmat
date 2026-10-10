//! Deterministic publication order and lease/effect tests, not numerical proof.
use super::*;
use runmat_accelerate_api::{
    ProviderImageMomentConversion, ProviderImageMomentProgram, ProviderImageMomentPublication,
};
use runmat_types::{
    DimensionFact, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn metadata_case(
    provider: &'static StagedProvider,
    class: NumericClass,
) -> (
    GpuTensorHandle,
    RetirementProbe,
    ProviderImageMomentPublication,
) {
    let element = match class {
        NumericClass::Single => NumericElementType::F32,
        NumericClass::Double => NumericElementType::F64,
        _ => unreachable!("floating cast test"),
    };
    let allocation = provider
        .allocations
        .allocate_zeroed(vec![2, 1, 1], element, GpuTensorStorage::Real)
        .unwrap();
    let raw = allocation
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .unwrap();
    let mut request = publication(29, 28, [2, 1, 1], &[30]);
    let kind = ValueKindFact::Numeric(NumericFact {
        class,
        domain: NumericDomain::Real,
    });
    request.raw_fact.kind = kind.clone();
    request.conversions[0].output_fact.kind = kind;
    runmat_accelerate_api::mark_residency(&raw);
    assert!(crate::fusion_residency::is_resident(&raw));
    *provider.mutate_cast_residency.lock().unwrap() = true;
    (raw, allocation.probe, request)
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn floating_publication_restores_mutated_residency_and_captured_intent_on_success() {
    for class in [NumericClass::Single, NumericClass::Double] {
        let provider = provider(None);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let (session, input, input_probe) = begin(provider);
        let (raw, raw_probe, request) = metadata_case(provider, class);
        let mut scratch = Vec::new();
        let public = futures::executor::block_on(session.publish_moment(
            raw.clone(),
            &request,
            &mut scratch,
            &[],
            None,
        ))
        .unwrap();
        assert!(crate::fusion_residency::is_resident(&raw));
        assert_eq!(
            runmat_accelerate_api::handle_provenance(&raw),
            Some(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        );
        assert_eq!(
            runmat_accelerate_api::handle_provenance(&public),
            Some(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        );
        assert!(!public.shares_allocation_with(&raw));
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        let public = ImageNormalizeStageSession::finish(Ok(public), scratch).unwrap();
        assert_eq!(raw_probe.retirements(), 0);
        assert_eq!(input_probe.retirements(), 0);
        drop((public, raw, session, input));
        assert_eq!(raw_probe.retirements(), 1);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn floating_publication_restores_mutated_residency_before_terminal_error() {
    for class in [NumericClass::Single, NumericClass::Double] {
        let provider = provider(Some(0));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let (session, input, input_probe) = begin(provider);
        let (raw, raw_probe, request) = metadata_case(provider, class);
        let mut scratch = Vec::new();
        let failure = futures::executor::block_on(session.publish_moment(
            raw.clone(),
            &request,
            &mut scratch,
            &[],
            None,
        ))
        .unwrap_err();
        assert!(!failure.is_unsupported());
        assert!(crate::fusion_residency::is_resident(&raw));
        assert_eq!(
            runmat_accelerate_api::handle_provenance(&raw),
            Some(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        );
        assert!(raw.allocation_lease().validate_live().is_ok());
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(raw_probe.retirements(), 0);
        assert_eq!(input_probe.retirements(), 0);
        drop((scratch, raw, session, input));
        assert_eq!(raw_probe.retirements(), 1);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn floating_publication_pending_cancellation_restores_mutated_residency() {
    use std::future::Future;
    use std::task::{Context, Poll};
    for class in [NumericClass::Single, NumericClass::Double] {
        let provider = provider_with(None, Some(0));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let (session, input, input_probe) = begin(provider);
        let (raw, raw_probe, request) = metadata_case(provider, class);
        let mut scratch = Vec::new();
        let mut pending =
            Box::pin(session.publish_moment(raw.clone(), &request, &mut scratch, &[], None));
        assert!(matches!(
            pending
                .as_mut()
                .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
            Poll::Pending
        ));
        assert!(!crate::fusion_residency::is_resident(&raw));
        drop(pending);
        assert!(crate::fusion_residency::is_resident(&raw));
        assert_eq!(
            runmat_accelerate_api::handle_provenance(&raw),
            Some(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        );
        assert!(raw.allocation_lease().validate_live().is_ok());
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        assert_eq!(raw_probe.retirements(), 0);
        assert_eq!(input_probe.retirements(), 0);
        drop((scratch, raw, session, input));
        assert_eq!(raw_probe.retirements(), 1);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn floating_publication_same_allocation_alias_restores_metadata_and_original_intent() {
    for class in [NumericClass::Single, NumericClass::Double] {
        let provider = provider(None);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let (session, input, input_probe) = begin(provider);
        let (raw, raw_probe, request) = metadata_case(provider, class);
        // An immutable provider alias can carry different view intent; only
        // the pre-hook source intent may determine the public result.
        let provider_alias = raw
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .unwrap();
        *provider.fault.lock().unwrap() = Some((0, StageFault::Alias(provider_alias)));
        let mut scratch = Vec::new();
        let public = futures::executor::block_on(session.publish_moment(
            raw.clone(),
            &request,
            &mut scratch,
            &[],
            None,
        ))
        .unwrap();
        assert!(public.shares_allocation_with(&raw));
        assert!(crate::fusion_residency::is_resident(&raw));
        assert_eq!(
            runmat_accelerate_api::handle_provenance(&public),
            Some(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        );
        provider.fault.lock().unwrap().take();
        let public = ImageNormalizeStageSession::finish(Ok(public), scratch).unwrap();
        drop(raw);
        assert_eq!(raw_probe.retirements(), 0);
        assert!(public.allocation_lease().validate_live().is_ok());
        assert_eq!(input_probe.retirements(), 0);
        drop((public, session, input));
        assert_eq!(raw_probe.retirements(), 1);
    }
}

fn publication(
    raw: u64,
    input: u64,
    shape: [usize; 3],
    outputs: &[u64],
) -> ProviderImageMomentPublication {
    let fact = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::Shaped {
            dims: shape.into_iter().map(DimensionFact::Known).collect(),
        },
        StorageFact::Dense,
    );
    let mut current = raw;
    let conversions = outputs
        .iter()
        .map(|&output| {
            let conversion = ProviderImageMomentConversion {
                source_input: current,
                source_output: output,
                output_fact: fact.clone(),
            };
            current = output;
            conversion
        })
        .collect();
    ProviderImageMomentPublication {
        source_input: input,
        raw_output: raw,
        raw_fact: fact,
        conversions,
        public_output: current,
    }
}

fn install(session: &mut ImageNormalizeStageSession) {
    session.program.moment_program = Some(ProviderImageMomentProgram {
        mu: vec![
            publication(10, 1, [2, 1, 4], &[11]),
            publication(18, 11, [2, 1, 1], &[19, 20]),
        ],
        variance: vec![
            publication(25, 24, [2, 1, 4], &[]),
            publication(27, 25, [2, 1, 1], &[28]),
        ],
        sigma: publication(29, 28, [2, 1, 1], &[30]),
    });
    assert!(session.program.validate());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn all_selected_conversions_execute_at_their_exact_stage_boundaries() {
    let provider = provider(None);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input, probe) = begin(provider);
    install(&mut session);
    let mean = futures::executor::block_on(session.mean()).unwrap().0;
    assert_eq!(
        provider.trace(),
        [
            StageHook::Dim(1),
            StageHook::Single,
            StageHook::Dim(2),
            StageHook::Single,
            StageHook::Single
        ]
    );
    let sigma = futures::executor::block_on(session.sigma()).unwrap().0;
    let trace = provider.trace();
    assert_eq!(
        &trace[5..],
        &[
            StageHook::Sub,
            StageHook::Upload,
            StageHook::Pow,
            StageHook::Dim(1),
            StageHook::Dim(2),
            StageHook::Single,
            StageHook::Add,
            StageHook::Sqrt,
            StageHook::Single
        ]
    );
    drop(session);
    assert!(mean.allocation_lease().validate_live().is_ok());
    assert!(sigma.allocation_lease().validate_live().is_ok());
    assert_eq!(probe.retirements(), 0);
    drop((mean, sigma, input));
    assert_eq!(probe.retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn conversion_unsupported_after_result_is_terminal_and_never_replays() {
    let provider = provider(Some(1));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input, probe) = begin(provider);
    install(&mut session);
    let failure = futures::executor::block_on(session.mean()).unwrap_err();
    assert!(!failure.is_unsupported());
    assert_eq!(provider.trace(), [StageHook::Dim(1), StageHook::Single]);
    assert!(futures::executor::block_on(session.mean()).is_err());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    assert_eq!(provider.probes()[0].retirements(), 1);
    assert_eq!(probe.retirements(), 0);
    drop((session, input));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn malformed_conversion_class_shape_and_owner_are_terminal() {
    for fault in [
        StageFault::WrongShape,
        StageFault::WrongClass,
        StageFault::WrongOwner,
    ] {
        let provider = provider(None);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let (mut session, input, probe) = begin(provider);
        install(&mut session);
        *provider.fault.lock().unwrap() = Some((1, fault));
        let failure = futures::executor::block_on(session.mean()).unwrap_err();
        assert!(!failure.is_unsupported());
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        assert!(provider
            .probes()
            .iter()
            .all(|probe| probe.retirements() == 1));
        assert_eq!(probe.retirements(), 0);
        assert!(input.allocation_lease().validate_live().is_ok());
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn cancellation_inside_conversion_retires_raw_and_latches_session() {
    use std::future::Future;
    use std::task::{Context, Poll};
    let provider = provider_with(None, Some(1));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input, probe) = begin(provider);
    install(&mut session);
    let mut pending = Box::pin(session.mean());
    assert!(matches!(
        pending
            .as_mut()
            .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
        Poll::Pending
    ));
    drop(pending);
    assert_eq!(provider.probes()[0].retirements(), 1);
    assert!(futures::executor::block_on(session.mean()).is_err());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert_eq!(probe.retirements(), 0);
    drop((session, input));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pinned_warm_mean_still_runs_selected_publication_conversion() {
    let provider = provider(None);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .allocations
        .allocate_zeroed(
            vec![2, 3, 4],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let raw = provider
        .produce(vec![2, 1, 1], StageHook::Nd(vec![1, 2]))
        .unwrap();
    let ex2 = provider
        .produce(vec![2, 1, 1], StageHook::Nd(vec![1, 2]))
        .unwrap();
    let raw_probe = provider.probes()[0].clone();
    *provider.reserved_mu.lock().unwrap() = Some((raw, ex2));
    let mut source = source_recipe();
    source.mu.calls = vec![ImageMeanProviderCall {
        hook: ImageMeanProviderHook::ReduceNd([1, 2]),
        output_class: crate::fusion::image_reduction::ImageMeanOutput::Native,
        shape: [2, 1, 1],
    }];
    let mut session = ImageNormalizeStageSession::begin(
        provider,
        &input.handle,
        descriptor(),
        source,
        ImageArithmeticProof::fixture(),
    )
    .unwrap()
    .unwrap();
    session.program.moment_program = Some(ProviderImageMomentProgram {
        mu: vec![publication(18, 1, [2, 1, 1], &[19])],
        variance: vec![
            publication(25, 24, [2, 1, 4], &[]),
            publication(27, 25, [2, 1, 1], &[]),
        ],
        sigma: publication(29, 28, [2, 1, 1], &[30]),
    });
    let before = provider.calls.load(Ordering::SeqCst);
    let public = futures::executor::block_on(session.mean()).unwrap().0;
    assert_eq!(provider.calls.load(Ordering::SeqCst), before + 1);
    assert_eq!(provider.trace().last(), Some(&StageHook::Single));
    assert_eq!(raw_probe.retirements(), 1);
    drop(session);
    assert!(public.allocation_lease().validate_live().is_ok());
    assert_eq!(input.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn certified_input_alias_survives_scratch_retirement_unrelated_borrowed_alias_is_rejected() {
    let provider = provider(None);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (session, input, input_probe) = begin(provider);
    let raw = provider.produce(vec![2, 1, 1], StageHook::Sqrt).unwrap();
    let raw_probe = provider.probes()[0].clone();
    *provider.fault.lock().unwrap() = Some((1, StageFault::Alias(raw.clone())));
    let mut scratch = Vec::new();
    let public = futures::executor::block_on(session.publish_moment(
        raw.clone(),
        &publication(29, 28, [2, 1, 1], &[30]),
        &mut scratch,
        &[],
        None,
    ))
    .unwrap();
    assert!(public.shares_allocation_with(&raw));
    drop(raw);
    provider.fault.lock().unwrap().take();
    let public = ImageNormalizeStageSession::finish(Ok(public), scratch).unwrap();
    assert_eq!(raw_probe.retirements(), 0);
    assert!(public.allocation_lease().validate_live().is_ok());
    let other = provider.produce(vec![2, 1, 1], StageHook::Sqrt).unwrap();
    *provider.fault.lock().unwrap() = Some((
        provider.calls.load(Ordering::SeqCst),
        StageFault::Alias(public.clone()),
    ));
    let mut scratch = Vec::new();
    assert!(futures::executor::block_on(session.publish_moment(
        other,
        &publication(29, 28, [2, 1, 1], &[30]),
        &mut scratch,
        &[&public],
        None,
    ))
    .is_err());
    drop(scratch);
    provider.fault.lock().unwrap().take();
    assert!(public.allocation_lease().validate_live().is_ok());
    assert_eq!(input_probe.retirements(), 0);
    drop((public, session, input));
    assert_eq!(raw_probe.retirements(), 1);
}
