use super::*;

fn selected_output_program() -> ProviderImageOutputProgram {
    use runmat_accelerate_api::{
        ProviderImageOutputCall as Call, ProviderImageOutputHook as Hook,
        ProviderImageOutputOperand as Operand,
    };
    let tensor = |id| Operand::Produced(id);
    let call = |source_output, hook, operands, checkpoint| Call {
        source_output,
        hook,
        operands,
        output_shape: [2, 3, 4],
        output_precision: ProviderPrecision::F32,
        checkpoint,
    };
    ProviderImageOutputProgram {
        calls: vec![
            call(
                11,
                Hook::ElemSub,
                vec![Operand::Input, Operand::Mean],
                false,
            ),
            call(12, Hook::ElemDiv, vec![tensor(11), Operand::Sigma], false),
            call(
                13,
                Hook::ScalarMul,
                vec![tensor(12), Operand::ScalarF32Bits(1.0123f32.to_bits())],
                false,
            ),
            call(
                14,
                Hook::ScalarAdd,
                vec![tensor(13), Operand::ScalarF32Bits((-0.02f32).to_bits())],
                false,
            ),
            call(15, Hook::CastSingle, vec![tensor(14)], true),
            call(
                16,
                Hook::ScalarMax,
                vec![tensor(15), Operand::ScalarF32Bits(0.0f32.to_bits())],
                true,
            ),
            call(
                17,
                Hook::ElemPow,
                vec![tensor(16), Operand::ScalarF32Bits(1.8f32.to_bits())],
                false,
            ),
            call(18, Hook::CastSingle, vec![tensor(17)], true),
        ],
    }
}

fn begin_selected(
    provider: &'static StagedProvider,
) -> (ImageNormalizeStageSession, GpuTensorHandle) {
    let input = provider
        .allocations
        .allocate_zeroed(
            vec![2, 3, 4],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        )
        .expect("owned input");
    let mut desc = descriptor();
    desc.gain = Some(f64::from(1.0123f32));
    desc.bias = Some(f64::from(-0.02f32));
    desc.clamp_zero = true;
    desc.gamma = Some(f64::from(1.8f32));
    let mut source = source_recipe();
    source.final_power.represented_exponent = f64::from(1.8f32);
    let session = ImageNormalizeStageSession::begin_selected_fixture(
        provider,
        &input.handle,
        desc,
        source,
        selected_output_program(),
        ImageArithmeticProof::fixture(),
    )
    .expect("selected output owner")
    .expect("held hook form");
    (session, input.handle)
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_f32_stage_rejects_f64_power_upload_before_provider_effects() {
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
    let mut program = selected_output_program();
    program.calls[6].operands[1] =
        runmat_accelerate_api::ProviderImageOutputOperand::ScalarF64Bits(1.8f64.to_bits());
    assert!(
        program.validate(),
        "F64 control remains a valid hook request"
    );
    let mut source = source_recipe();
    source.final_power.represented_exponent = f64::from(1.8f32);
    let mut desc = descriptor();
    desc.gamma = Some(f64::from(1.8f32));
    assert!(ImageNormalizeStageSession::begin_selected_fixture(
        provider,
        &input.handle,
        desc,
        source,
        program,
        ImageArithmeticProof::fixture(),
    )
    .unwrap()
    .is_none());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_five_checkpoints_require_source_ordered_publication_ack() {
    let provider = provider(None);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input) = begin_selected(provider);
    let (mean, sigma, outputs) = futures::executor::block_on(async {
        let mean = session.mean().await.unwrap().0;
        let sigma = session.sigma().await.unwrap().0;
        let first = session.next_output_checkpoint().await.unwrap();
        assert_eq!(first.source_output, 15);
        let before = provider.calls.load(Ordering::SeqCst);
        assert!(session.next_output_checkpoint().await.is_err());
        assert!(session.acknowledge_output_checkpoint(16).is_err());
        assert_eq!(provider.calls.load(Ordering::SeqCst), before);
        session.acknowledge_output_checkpoint(15).unwrap();
        let second = session.next_output_checkpoint().await.unwrap();
        assert_eq!(second.source_output, 16);
        session.acknowledge_output_checkpoint(16).unwrap();
        let last = session.next_output_checkpoint().await.unwrap();
        assert_eq!(last.source_output, 18);
        session.acknowledge_output_checkpoint(18).unwrap();
        (mean, sigma, vec![first.handle, second.handle, last.handle])
    });
    assert_eq!(outputs.len(), 3);
    assert!(outputs
        .iter()
        .all(|handle| handle.allocation_lease().validate_live().is_ok()));
    assert_eq!(
        provider.trace()[9..],
        [
            StageHook::Sub,
            StageHook::Div,
            StageHook::Mul,
            StageHook::Add,
            StageHook::Single,
            StageHook::Max,
            StageHook::Upload,
            StageHook::Pow,
            StageHook::Single,
        ]
    );
    let probes = provider.probes();
    assert_eq!(probes.len(), 18);
    drop(session);
    for (handle, probe) in outputs.iter().zip([&probes[13], &probes[14], &probes[17]]) {
        assert!(handle.allocation_lease().validate_live().is_ok());
        assert_eq!(probe.retirements(), 0, "VM-retained checkpoint stays live");
    }
    drop(outputs);
    for probe in [&probes[13], &probes[14], &probes[17]] {
        assert_eq!(probe.retirements(), 1, "checkpoint retires exactly once");
    }
    drop(mean);
    drop(sigma);
    drop(input);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_output_failure_after_first_publication_cannot_replay() {
    let provider = provider(Some(14));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input) = begin_selected(provider);
    futures::executor::block_on(async {
        let _mean = session.mean().await.unwrap();
        let _sigma = session.sigma().await.unwrap();
        let first = session.next_output_checkpoint().await.unwrap();
        assert_eq!(first.source_output, 15);
        session.acknowledge_output_checkpoint(15).unwrap();
        let error = session.next_output_checkpoint().await.unwrap_err();
        assert!(
            !error.is_unsupported(),
            "post-effect unsupported cannot replay"
        );
        let before = provider.calls.load(Ordering::SeqCst);
        assert!(session.next_output_checkpoint().await.is_err());
        assert_eq!(provider.calls.load(Ordering::SeqCst), before);
        assert!(first.handle.allocation_lease().validate_live().is_ok());
    });
    drop(input);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn dropping_polled_output_hook_cannot_replay_or_retire_published_moments() {
    use std::future::Future;
    use std::task::{Context, Poll};

    let provider = provider_with(None, Some(9));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input) = begin_selected(provider);
    let mean = futures::executor::block_on(session.mean()).unwrap().0;
    let sigma = futures::executor::block_on(session.sigma()).unwrap().0;
    assert_eq!(provider.calls.load(Ordering::SeqCst), 9);
    let probes = provider.probes();
    let mut future = Box::pin(session.next_output_checkpoint());
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
    drop(future);
    assert!(futures::executor::block_on(session.next_output_checkpoint()).is_err());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 9);
    drop(session);
    assert_eq!(probes[1].retirements(), 0);
    assert_eq!(probes[8].retirements(), 0);
    assert!(mean.allocation_lease().validate_live().is_ok());
    assert!(sigma.allocation_lease().validate_live().is_ok());
    drop(mean);
    drop(sigma);
    assert_eq!(probes[1].retirements(), 1);
    assert_eq!(probes[8].retirements(), 1);
    drop(input);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn output_hook_rejects_wrong_shape_class_owner_and_borrowed_aliases() {
    for variant in [
        StageFault::WrongShape,
        StageFault::WrongClass,
        StageFault::WrongOwner,
    ] {
        let provider = provider(None);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let (mut session, input) = begin_selected(provider);
        let mean = futures::executor::block_on(session.mean()).unwrap().0;
        let sigma = futures::executor::block_on(session.sigma()).unwrap().0;
        *provider.fault.lock().unwrap() = Some((9, variant));
        let failure = futures::executor::block_on(session.next_output_checkpoint()).unwrap_err();
        assert!(!failure.is_unsupported());
        let rejected = provider.probes();
        assert_eq!(rejected.len(), 10);
        assert_eq!(rejected[9].retirements(), 1, "invalid result is retired");
        let before = provider.calls.load(Ordering::SeqCst);
        assert!(futures::executor::block_on(session.next_output_checkpoint()).is_err());
        assert_eq!(provider.calls.load(Ordering::SeqCst), before);
        for handle in [&input, &mean, &sigma] {
            assert!(handle.allocation_lease().validate_live().is_ok());
        }
    }
    for alias in [0usize, 1, 2] {
        let provider = provider(None);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let (mut session, input) = begin_selected(provider);
        let mean = futures::executor::block_on(session.mean()).unwrap().0;
        let sigma = futures::executor::block_on(session.sigma()).unwrap().0;
        let handles = [&input, &mean, &sigma];
        *provider.fault.lock().unwrap() = Some((9, StageFault::Alias((*handles[alias]).clone())));
        let failure = futures::executor::block_on(session.next_output_checkpoint()).unwrap_err();
        assert!(!failure.is_unsupported());
        let before = provider.calls.load(Ordering::SeqCst);
        assert!(futures::executor::block_on(session.next_output_checkpoint()).is_err());
        assert_eq!(provider.calls.load(Ordering::SeqCst), before);
        assert!(handles[alias].allocation_lease().validate_live().is_ok());
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn output_hook_rejects_alias_to_previous_published_checkpoint() {
    let provider = provider(None);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input) = begin_selected(provider);
    let mean = futures::executor::block_on(session.mean()).unwrap().0;
    let sigma = futures::executor::block_on(session.sigma()).unwrap().0;
    let first = futures::executor::block_on(session.next_output_checkpoint()).unwrap();
    session
        .acknowledge_output_checkpoint(first.source_output)
        .unwrap();
    let prior_probe = provider.probes()[13].clone();
    *provider.fault.lock().unwrap() = Some((14, StageFault::Alias(first.handle.clone())));
    let failure = futures::executor::block_on(session.next_output_checkpoint()).unwrap_err();
    assert!(!failure.is_unsupported());
    let before = provider.calls.load(Ordering::SeqCst);
    assert!(futures::executor::block_on(session.next_output_checkpoint()).is_err());
    assert_eq!(provider.calls.load(Ordering::SeqCst), before);
    drop(session);
    assert_eq!(prior_probe.retirements(), 0);
    assert!(first.handle.allocation_lease().validate_live().is_ok());
    drop(first);
    *provider.fault.lock().unwrap() = None;
    assert_eq!(prior_probe.retirements(), 1);
    drop(mean);
    drop(sigma);
    drop(input);
}
