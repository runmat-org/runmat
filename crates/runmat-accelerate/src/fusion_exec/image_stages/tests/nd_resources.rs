use super::*;
use runmat_accelerate_api::ProviderExecutedColdNdMean;

fn alloc(
    provider: &StagedProvider,
    shape: &[usize],
) -> runmat_accelerate_api::test_support::TestAllocation {
    provider
        .allocations
        .allocate_zeroed(
            shape.to_vec(),
            NumericElementType::F32,
            GpuTensorStorage::Real,
        )
        .unwrap()
}

#[test]
fn selected_warm_receipt_binds_exact_source_view_and_retained_pair() {
    let provider = provider(None);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let source = alloc(provider, &[2, 3, 4]);
    let same_shape_other_view = alloc(provider, &[2, 3, 4]);
    let mean = alloc(provider, &[2, 1, 1]);
    let ex2 = alloc(provider, &[2, 1, 1]);
    let pair_bytes =
        mean.handle.physical().requested_bytes() + ex2.handle.physical().requested_bytes();
    let pinned = ProviderPinnedNdMean::new(
        source.handle.clone(),
        vec![1, 2],
        mean.handle.clone(),
        ex2.handle.clone(),
        pair_bytes,
    );
    let estimate = super::super::nd_resources::warm_mean(
        provider,
        &source.handle,
        &pinned,
        &[1, 2],
        &[2, 1, 1],
    )
    .unwrap();
    assert_eq!(
        estimate.output_bytes,
        Some(mean.handle.physical().requested_bytes())
    );
    assert_eq!(estimate.dispatches, Some(0));
    assert!(super::super::nd_resources::warm_mean(
        provider,
        &source.handle,
        &pinned,
        &[2, 1],
        &[2, 1, 1],
    )
    .is_err());
    assert_eq!(
        estimate.transient_bytes,
        source
            .handle
            .physical()
            .requested_bytes()
            .checked_add(pair_bytes)
    );
    assert!(super::super::nd_resources::warm_mean(
        provider,
        &same_shape_other_view.handle,
        &pinned,
        &[1, 2],
        &[2, 1, 1],
    )
    .is_err());
    source.probe.mark_unavailable();
    assert!(super::super::nd_resources::warm_mean(
        provider,
        &source.handle,
        &pinned,
        &[1, 2],
        &[2, 1, 1],
    )
    .is_err());
    let pinned_other = ProviderPinnedNdMean::new(
        same_shape_other_view.handle.clone(),
        vec![1, 2],
        mean.handle,
        ex2.handle,
        pair_bytes,
    );
    ex2.probe.mark_unavailable();
    assert!(super::super::nd_resources::warm_mean(
        provider,
        &same_shape_other_view.handle,
        &pinned_other,
        &[1, 2],
        &[2, 1, 1],
    )
    .is_err());
}

#[test]
fn selected_cold_completion_checks_pair_uniform_and_observed_bytes() {
    let provider = provider(None);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let source = alloc(provider, &[2, 3, 4]);
    let other_source = alloc(provider, &[2, 3, 4]);
    let mean = alloc(provider, &[2, 1, 1]);
    let ex2 = alloc(provider, &[2, 1, 1]);
    let pair_bytes =
        mean.handle.physical().requested_bytes() + ex2.handle.physical().requested_bytes();
    let selected = ProviderColdNdMean::new(
        source.handle.clone(),
        vec![1, 2],
        vec![2, 1, 1],
        mean.handle.physical().requested_bytes(),
        64,
        0,
    );
    let resources = super::super::nd_resources::cold_mean(
        provider,
        &source.handle,
        &selected,
        &[1, 2],
        &[2, 1, 1],
    )
    .unwrap();
    assert_eq!(resources.dispatches, Some(1));
    assert!(super::super::nd_resources::cold_mean(
        provider,
        &source.handle,
        &selected,
        &[2, 1],
        &[2, 1, 1],
    )
    .is_err());
    assert_eq!(resources.output_bytes, Some(pair_bytes / 2));
    assert!(super::super::nd_resources::cold_mean(
        provider,
        &other_source.handle,
        &selected,
        &[1, 2],
        &[2, 1, 1],
    )
    .is_err());
    let executed = ProviderExecutedColdNdMean {
        mean: mean.handle,
        ex2: ex2.handle,
        requested_pair_bytes: pair_bytes,
        observed_pair_bytes: pair_bytes,
        requested_uniform_bytes: 64,
        selected_submissions: 1,
    };
    super::super::nd_resources::executed_cold_mean(
        provider,
        &resources,
        pair_bytes,
        64,
        &executed,
        &[2, 1, 1],
    )
    .unwrap();
    let wrong_observed = ProviderExecutedColdNdMean {
        observed_pair_bytes: pair_bytes - 1,
        ..executed
    };
    assert!(super::super::nd_resources::executed_cold_mean(
        provider,
        &resources,
        pair_bytes,
        64,
        &wrong_observed,
        &[2, 1, 1],
    )
    .is_err());
}

#[test]
fn selected_cold_completion_rejects_actual_wrong_owner_class_and_shape_without_retirement() {
    #[derive(Clone, Copy, Debug)]
    enum Fault {
        Owner,
        Class,
        Shape,
    }

    for fault in [Fault::Owner, Fault::Class, Fault::Shape] {
        let provider = provider(None);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let source = alloc(provider, &[2, 3, 4]);
        let source_probe = source.probe.clone();
        let bad_mean = match fault {
            Fault::Owner => provider
                .foreign_allocations
                .allocate_zeroed(
                    vec![2, 1, 1],
                    NumericElementType::F32,
                    GpuTensorStorage::Real,
                )
                .unwrap(),
            Fault::Class => provider
                .allocations
                .allocate_zeroed(
                    vec![2, 1, 1],
                    NumericElementType::I32,
                    GpuTensorStorage::Real,
                )
                .unwrap(),
            Fault::Shape => provider
                .allocations
                .allocate_zeroed(
                    vec![1, 2, 1],
                    NumericElementType::F32,
                    GpuTensorStorage::Real,
                )
                .unwrap(),
        };
        let bad_probe = bad_mean.probe.clone();
        let ex2 = alloc(provider, &[2, 1, 1]);
        let ex2_probe = ex2.probe.clone();
        let output_bytes = bad_mean.handle.physical().requested_bytes();
        let pair_bytes = output_bytes + ex2.handle.physical().requested_bytes();
        let selected = ProviderColdNdMean::new(
            source.handle.clone(),
            vec![1, 2],
            vec![2, 1, 1],
            output_bytes,
            64,
            0,
        );
        let resources = super::super::nd_resources::cold_mean(
            provider,
            &source.handle,
            &selected,
            &[1, 2],
            &[2, 1, 1],
        )
        .unwrap();
        let executed = ProviderExecutedColdNdMean {
            mean: bad_mean.handle,
            ex2: ex2.handle,
            requested_pair_bytes: pair_bytes,
            observed_pair_bytes: pair_bytes,
            requested_uniform_bytes: 64,
            selected_submissions: 1,
        };
        let error = super::super::nd_resources::executed_cold_mean(
            provider,
            &resources,
            pair_bytes,
            64,
            &executed,
            &[2, 1, 1],
        )
        .expect_err("actual invalid completion must be terminal");
        assert!(!error.is_unsupported(), "{fault:?}: no numerical replay");
        assert!(
            error
                .to_string()
                .contains("completion disagrees with live resources"),
            "{fault:?}: original admission cause survives"
        );
        assert_eq!(source_probe.retirements(), 0, "{fault:?}: borrowed source");
        assert_eq!(bad_probe.retirements(), 0, "{fault:?}: borrowed mean");
        assert_eq!(ex2_probe.retirements(), 0, "{fault:?}: borrowed E[x²]");
        drop(executed);
        assert_eq!(bad_probe.retirements(), 1, "{fault:?}: owned mean retires");
        assert_eq!(ex2_probe.retirements(), 1, "{fault:?}: owned E[x²] retires");
        assert_eq!(source_probe.retirements(), 0, "{fault:?}: source survives");
    }
}
