//! Source-control, owner, resource and cancellation tests; no fake GPU math.
use super::*;
#[path = "tests/fixture.rs"]
mod fixture;
use fixture::*;
use std::sync::atomic::Ordering;
#[path = "tests/double_source.rs"]
mod double_source;

fn plan(fixture: &'static Fixture) -> ProviderMixedRealAddRecipe {
    ProviderMixedRealAddRecipe::plan(
        fixture,
        request(ScalarLiteral::F64Bits(
            (f64::from(f32::EPSILON) / 4.0 - f64::EPSILON).to_bits(),
        )),
    )
    .unwrap()
    .unwrap()
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn all_three_plans_precede_effects_and_original_control_bits_remain_bound() {
    for decline in 1..=3 {
        let f = fixture();
        f.decline_at.store(decline, Ordering::SeqCst);
        assert!(ProviderMixedRealAddRecipe::plan(
            f,
            request(ScalarLiteral::F64Bits(1.0_f64.to_bits()))
        )
        .unwrap()
        .is_none());
        assert_eq!(f.calls.load(Ordering::SeqCst), 0);
        assert!(f.probes.lock().unwrap().is_empty());
    }
    let f = fixture();
    let recipe = plan(f);
    let source = allocate_source(f);
    f.decline_at.store(3, Ordering::SeqCst);
    assert!(ready(recipe.execute(&source.handle))
        .err()
        .unwrap()
        .is_unsupported());
    assert_eq!(f.calls.load(Ordering::SeqCst), 0);
    let f = fixture();
    for bits in [
        (-0.0_f64).to_bits(),
        0x7ff8_0000_0000_1234,
        1.0_f64.to_bits(),
    ] {
        let recipe = ProviderMixedRealAddRecipe::plan(f, request(ScalarLiteral::F64Bits(bits)))
            .unwrap()
            .unwrap();
        assert_eq!(
            recipe.request().operation,
            request(ScalarLiteral::F64Bits(bits)).operation
        );
        let different =
            ProviderMixedRealAddRecipe::plan(f, request(ScalarLiteral::F64Bits(bits ^ 1)))
                .unwrap()
                .unwrap();
        assert_ne!(recipe.identity().unwrap(), different.identity().unwrap());
    }
    assert!(ProviderMixedRealAddRecipe::plan(
        f,
        request(ScalarLiteral::F32Bits(1.0_f32.to_bits()))
    )
    .unwrap()
    .is_none());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn successful_sequence_retains_both_double_results_until_final_validation() {
    let f = fixture();
    let recipe = plan(f);
    let source = allocate_source(f);
    let result = ready(recipe.execute(&source.handle)).unwrap();
    assert_eq!(
        result.output.physical().element_type(),
        NumericElementType::F32
    );
    assert_eq!(result.observed.resources, *recipe.resources());
    assert_eq!(recipe.resources().retained_output_buffer_bytes, 40);
    assert_eq!(recipe.resources().uniform_buffer_bytes, 288);
    assert_eq!(
        recipe.resources().requested_incremental_peak_bytes(),
        Some(328)
    );
    assert_eq!(recipe.resources().dispatches, 3);
    assert_eq!(recipe.resources().queue_submissions, 3);
    assert_eq!(
        *f.retained.lock().unwrap(),
        vec![vec![], vec![0], vec![0, 0]]
    );
    assert_eq!(
        f.probes
            .lock()
            .unwrap()
            .iter()
            .map(|probe| probe.retirements())
            .collect::<Vec<_>>(),
        vec![1, 1, 0]
    );
    let controls = f.controls.lock().unwrap();
    assert!(matches!(
        controls[0],
        ProviderRealElementwiseOperation::FloatingCast {
            target: ProviderPrecision::F64
        }
    ));
    let ProviderRealElementwiseOperation::ScalarAdd { control, .. } = recipe.request().operation
    else {
        unreachable!()
    };
    assert_eq!(
        controls[1],
        ProviderRealElementwiseOperation::ScalarAdd {
            control,
            mode: runmat_types::RealFloatingAddMode::Double
        }
    );
    assert!(matches!(
        controls[2],
        ProviderRealElementwiseOperation::FloatingCast {
            target: ProviderPrecision::F32
        }
    ));
    drop(controls);
    drop(result);
    assert_eq!(f.probes.lock().unwrap()[2].retirements(), 1);
    assert_eq!(source.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn prefix_effects_are_terminal_and_protected_aliases_never_retired() {
    for at in 1..=3 {
        for fault in [
            Fault::BeforeUnsupported,
            Fault::AfterCause,
            Fault::Alias,
            Fault::WrongClass,
            Fault::WrongReceipt,
        ] {
            let f = fixture();
            let source = allocate_source(f);
            let protected = allocate_source(f);
            *f.fault.lock().unwrap() = Some((at, fault));
            *f.alias.lock().unwrap() = Some(protected.handle.clone());
            let error = ready(plan(f).execute_protected(&source.handle, &[&protected.handle]))
                .err()
                .unwrap();
            if matches!(fault, Fault::BeforeUnsupported) && at == 1 {
                assert!(error.is_unsupported());
            } else {
                assert!(error.is_after_effect());
                assert!(!error.is_unsupported());
            }
            assert_eq!(f.calls.load(Ordering::SeqCst), at);
            assert_eq!(source.probe.retirements(), 0);
            assert_eq!(protected.probe.retirements(), 0);
            for probe in f.probes.lock().unwrap().iter() {
                assert_eq!(probe.retirements(), 1);
            }
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn each_suspended_step_cancellation_retires_only_owned_prefix_and_pending_result() {
    use std::{
        future::Future,
        task::{Context, Poll},
    };
    for at in 1..=3 {
        let f = fixture();
        let source = allocate_source(f);
        let protected = allocate_source(f);
        let recipe = plan(f);
        *f.fault.lock().unwrap() = Some((at, Fault::Pending));
        drop(Box::pin(recipe.execute(&source.handle)));
        assert_eq!(f.calls.load(Ordering::SeqCst), 0);
        let borrowed = [&protected.handle];
        let mut future = Box::pin(recipe.execute_protected(&source.handle, &borrowed));
        assert!(matches!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(std::task::Waker::noop())),
            Poll::Pending
        ));
        assert_eq!(f.probes.lock().unwrap().len(), at);
        assert!(f
            .probes
            .lock()
            .unwrap()
            .iter()
            .all(|probe| probe.retirements() == 0));
        drop(future);
        assert!(f
            .probes
            .lock()
            .unwrap()
            .iter()
            .all(|probe| probe.retirements() == 1));
        assert_eq!(source.probe.retirements(), 0);
        assert_eq!(protected.probe.retirements(), 0);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn original_hook_and_cleanup_causes_are_retained_without_replay() {
    let f = fixture();
    let source = allocate_source(f);
    f.cleanup_failure_at.store(1, Ordering::SeqCst);
    *f.fault.lock().unwrap() = Some((2, Fault::AfterCause));
    let error = ready(plan(f).execute(&source.handle)).err().unwrap();
    assert!(error.is_after_effect());
    assert!(!error.is_unsupported());
    let completion = std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<ProviderExecutionCompletionError>()
        .unwrap();
    assert_eq!(completion.cleanup_errors().len(), 1);
    let primary = completion.primary().unwrap();
    assert!(std::error::Error::source(primary)
        .unwrap()
        .downcast_ref::<std::io::Error>()
        .is_some());
    assert!(primary.to_string().contains("original compound hook cause"));
    assert_eq!(source.probe.retirements(), 0);
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn foreign_dead_and_wrong_shape_sources_fail_before_first_hook() {
    let f = fixture();
    let recipe = plan(f);
    let foreign = allocate_source(fixture());
    assert!(ready(recipe.execute(&foreign.handle)).is_err());
    let wrong_shape = f
        .owner
        .allocate_zeroed(vec![1, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    assert!(ready(recipe.execute(&wrong_shape.handle)).is_err());
    let dead = allocate_source(f);
    dead.probe.mark_unavailable();
    assert!(ready(recipe.execute(&dead.handle)).is_err());
    assert_eq!(f.calls.load(Ordering::SeqCst), 0);
    assert!(f.probes.lock().unwrap().is_empty());
    assert_eq!(foreign.probe.retirements(), 0);
    assert_eq!(wrong_shape.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn failed_intermediate_retirement_prevents_publication_and_retires_final_owned_output() {
    let f = fixture();
    let source = allocate_source(f);
    f.cleanup_failure_at.store(1, Ordering::SeqCst);
    let error = ready(plan(f).execute(&source.handle)).err().unwrap();
    assert!(error.is_after_effect());
    assert!(!error.is_unsupported());
    let completion = std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<ProviderExecutionCompletionError>()
        .unwrap();
    assert!(completion.primary().is_none());
    assert_eq!(completion.cleanup_errors().len(), 1);
    assert!(f
        .probes
        .lock()
        .unwrap()
        .iter()
        .all(|probe| probe.retirements() == 1));
    assert_eq!(source.probe.retirements(), 0);
    assert_eq!(f.calls.load(Ordering::SeqCst), 3);
}
