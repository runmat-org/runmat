use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn staged_three_checkpoints_keep_published_moments_and_borrowed_input_alive() {
    let provider = provider(None);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input, input_probe) = begin(provider);
    let (mean, sigma, image) = futures::executor::block_on(async {
        let mean = session.mean().await.expect("mean stage").0;
        assert_eq!(mean.shape(), [2, 1, 1]);
        let sigma = session.sigma().await.expect("sigma stage").0;
        assert_eq!(sigma.shape(), [2, 1, 1]);
        let image = session.output().await.expect("final stage").0;
        (mean, sigma, image)
    });
    assert_eq!(image.shape(), [2, 3, 4]);
    for handle in [&mean, &sigma, &image] {
        assert_eq!(handle.physical().element_type(), NumericElementType::F32);
        assert_eq!(handle.physical().storage(), GpuTensorStorage::Real);
        assert_eq!(handle.device_id(), provider.device_id());
        assert!(!handle.shares_allocation_with(&input));
        assert!(handle.allocation_lease().validate_live().is_ok());
    }
    assert!(!mean.shares_allocation_with(&sigma));
    assert!(!mean.shares_allocation_with(&image));
    assert!(!sigma.shares_allocation_with(&image));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 13);
    assert_eq!(
        provider.trace(),
        [
            StageHook::Dim(1),
            StageHook::Dim(2),
            StageHook::Sub,
            StageHook::Upload,
            StageHook::Pow,
            StageHook::Dim(1),
            StageHook::Dim(2),
            StageHook::Add,
            StageHook::Sqrt,
            StageHook::Sub,
            StageHook::Div,
            StageHook::Upload,
            StageHook::Pow,
        ]
    );
    let probes = provider.probes();
    assert_eq!(probes.len(), 13);
    drop(session);
    assert_eq!(input_probe.retirements(), 0);
    assert_eq!(probes[1].retirements(), 0, "published mean remains live");
    assert_eq!(probes[8].retirements(), 0, "published sigma remains live");
    assert_eq!(probes[12].retirements(), 0, "published image remains live");
    for scratch in [0, 2, 3, 4, 5, 6, 7, 9, 10, 11] {
        assert_eq!(probes[scratch].retirements(), 1);
    }
    drop(mean);
    drop(sigma);
    drop(image);
    assert_eq!(probes[1].retirements(), 1);
    assert_eq!(probes[8].retirements(), 1);
    assert_eq!(probes[12].retirements(), 1);
    drop(input);
    assert_eq!(input_probe.retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_warm_mean_consumes_the_pinned_handle_without_requery_or_dispatch() {
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
    let cached_mean = provider
        .produce(vec![2, 1, 1], StageHook::Nd(vec![1, 2]))
        .unwrap();
    let cached_ex2 = provider
        .produce(vec![2, 1, 1], StageHook::Nd(vec![1, 2]))
        .unwrap();
    let cached_mean_probe = provider.probes()[0].clone();
    let cached_ex2_probe = provider.probes()[1].clone();
    *provider.reserved_mu.lock().unwrap() = Some((cached_mean, cached_ex2));
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
    assert!(provider.reserved_mu.lock().unwrap().is_none());
    let before = provider.calls.load(Ordering::SeqCst);
    let selected_mean = futures::executor::block_on(session.mean()).unwrap().0;
    assert_eq!(provider.calls.load(Ordering::SeqCst), before);
    assert_eq!(cached_ex2_probe.retirements(), 1);
    assert_eq!(cached_mean_probe.retirements(), 0);
    drop(session);
    drop(selected_mean);
    assert_eq!(cached_mean_probe.retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_warm_mean_rejects_invalid_hidden_pair_before_any_new_hook() {
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
    let cached_mean = provider
        .produce(vec![2, 1, 1], StageHook::Nd(vec![1, 2]))
        .unwrap();
    let wrong_ex2 = provider
        .produce(vec![2, 1, 2], StageHook::Nd(vec![1, 2]))
        .unwrap();
    *provider.reserved_mu.lock().unwrap() = Some((cached_mean, wrong_ex2));
    let mut source = source_recipe();
    source.mu.calls = vec![ImageMeanProviderCall {
        hook: ImageMeanProviderHook::ReduceNd([1, 2]),
        output_class: crate::fusion::image_reduction::ImageMeanOutput::Native,
        shape: [2, 1, 1],
    }];
    let before = provider.calls.load(Ordering::SeqCst);
    assert!(ImageNormalizeStageSession::begin(
        provider,
        &input.handle,
        descriptor(),
        source,
        ImageArithmeticProof::fixture(),
    )
    .is_err());
    assert_eq!(provider.calls.load(Ordering::SeqCst), before);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn abandonment_after_mean_retires_only_session_references() {
    let provider = provider(None);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input, input_probe) = begin(provider);
    let published_mean = futures::executor::block_on(session.mean()).expect("mean").0;
    let probes = provider.probes();
    drop(session);
    assert_eq!(
        probes[0].retirements(),
        1,
        "private width reduction retired"
    );
    assert_eq!(probes[1].retirements(), 0, "published mean not retired");
    assert_eq!(input_probe.retirements(), 0, "borrowed image not retired");
    drop(published_mean);
    assert_eq!(probes[1].retirements(), 1);
    drop(input);
    assert_eq!(input_probe.retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn first_unsupported_hook_is_preeffect_but_later_unsupported_cannot_replay() {
    let first = provider(Some(0));
    let _selected = ThreadProviderGuard::set(Some(first));
    let (mut session, input, input_probe) = begin(first);
    let error = futures::executor::block_on(session.mean()).expect_err("hook unavailable");
    assert!(error.is_unsupported());
    assert!(first.probes().is_empty());
    drop(session);
    assert_eq!(input_probe.retirements(), 0);
    drop(input);

    let later = provider(Some(1));
    let _selected = ThreadProviderGuard::set(Some(later));
    let (mut session, input, input_probe) = begin(later);
    let error = futures::executor::block_on(session.mean()).expect_err("later hook unavailable");
    assert!(!error.is_unsupported(), "allocation already occurred");
    assert_eq!(later.probes()[0].retirements(), 1);
    drop(session);
    assert_eq!(input_probe.retirements(), 0);
    drop(input);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn sigma_failure_retains_published_mean_and_forbids_stage_replay() {
    let provider = provider(Some(2));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input, input_probe) = begin(provider);
    let mean = futures::executor::block_on(session.mean())
        .expect("mean published before sigma")
        .0;
    let error = futures::executor::block_on(session.sigma()).expect_err("sigma hook unavailable");
    assert!(!error.is_unsupported(), "mean publication forbids fallback");
    assert!(futures::executor::block_on(session.sigma()).is_err());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 3, "no replay");
    let probes = provider.probes();
    assert_eq!(probes.len(), 2);
    drop(session);
    assert_eq!(probes[1].retirements(), 0, "published mean remains live");
    assert_eq!(input_probe.retirements(), 0, "borrowed image remains live");
    drop(mean);
    assert_eq!(probes[1].retirements(), 1);
    drop(input);
    assert_eq!(input_probe.retirements(), 1);
}

mod failure;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn unsupported_controls_are_rejected_before_any_provider_hook_or_allocation() {
    let provider = provider(None);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .allocations
        .allocate_zeroed(
            vec![2, 3, 4],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        )
        .expect("owned input");
    for mut desc in [
        ImageNormalizeDescriptor {
            gamma: Some(0.5),
            ..descriptor()
        },
        ImageNormalizeDescriptor {
            clamp_zero: true,
            ..descriptor()
        },
        ImageNormalizeDescriptor {
            epsilon: f64::NAN,
            ..descriptor()
        },
        ImageNormalizeDescriptor {
            epsilon: f64::from(f32::EPSILON) / 3.0,
            ..descriptor()
        },
    ] {
        assert!(ImageNormalizeStageSession::begin(
            provider,
            &input.handle,
            desc.clone(),
            source_recipe(),
            ImageArithmeticProof::fixture(),
        )
        .expect("valid owner")
        .is_none());
        desc.batch = 17;
        assert!(ImageNormalizeStageSession::begin(
            provider,
            &input.handle,
            desc,
            source_recipe(),
            ImageArithmeticProof::fixture(),
        )
        .expect("valid owner")
        .is_none());
    }
    for (square, final_power, gamma) in [
        (1.0, 1.0, None),
        (2.0 + f64::from(f32::EPSILON), 1.0, None),
        (2.0, 2.0, None),
        (2.0, 0.0, None),
        (2.0, 1.0, Some(2.0)),
    ] {
        let mut source = source_recipe();
        source.square.represented_exponent = square;
        source.final_power.represented_exponent = final_power;
        assert!(ImageNormalizeStageSession::begin(
            provider,
            &input.handle,
            ImageNormalizeDescriptor {
                gamma,
                ..descriptor()
            },
            source,
            ImageArithmeticProof::fixture(),
        )
        .expect("pre-effect source mismatch")
        .is_none());
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert!(provider.probes().is_empty());
    assert_eq!(input.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn dropping_pending_mean_retires_scratch_and_cannot_replay_the_session() {
    use std::future::Future;
    use std::task::{Context, Poll};

    let provider = provider_with(None, Some(1));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input, input_probe) = begin(provider);
    let mut future = Box::pin(session.mean());
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    let scratch = provider.probes();
    assert_eq!(scratch.len(), 1);
    assert_eq!(scratch[0].retirements(), 0);
    drop(future);
    assert_eq!(scratch[0].retirements(), 1, "pending scratch is retired");
    assert!(futures::executor::block_on(session.mean()).is_err());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1, "no stage replay");
    drop(session);
    assert_eq!(input_probe.retirements(), 0);
    drop(input);
    assert_eq!(input_probe.retirements(), 1);
}
