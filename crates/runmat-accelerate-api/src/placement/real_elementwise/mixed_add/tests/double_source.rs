//! Deterministic real leases and controlled hooks, never a numerical GPU receipt.
use super::*;

fn plan_double(f: &'static Fixture, bits: u32) -> ProviderMixedRealAddRecipe {
    ProviderMixedRealAddRecipe::plan(f, double_source_request(bits))
        .unwrap()
        .unwrap()
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn both_double_source_plans_precede_effect_and_original_single_bits_bind_identity() {
    for decline in [2, 3] {
        let f = fixture();
        f.decline_at.store(decline, Ordering::SeqCst);
        assert!(
            ProviderMixedRealAddRecipe::plan(f, double_source_request(1.0_f32.to_bits()))
                .unwrap()
                .is_none()
        );
        assert_eq!(f.calls.load(Ordering::SeqCst), 0);
        assert!(f.probes.lock().unwrap().is_empty());
        let f = fixture();
        let source = allocate_double_source(f);
        let recipe = plan_double(f, 1.0_f32.to_bits());
        f.decline_at.store(decline, Ordering::SeqCst);
        assert!(ready(recipe.execute(&source.handle))
            .err()
            .unwrap()
            .is_unsupported());
        assert_eq!(f.calls.load(Ordering::SeqCst), 0);
        assert_eq!(source.probe.retirements(), 0);
    }
    for bits in [0_u32, 0x8000_0000, 0x7fc0_1234, 1, 1.0_f32.to_bits()] {
        let f = fixture();
        let recipe = plan_double(f, bits);
        assert_eq!(*recipe.request(), double_source_request(bits));
        assert_ne!(
            recipe.identity().unwrap(),
            plan_double(f, bits ^ 1).identity().unwrap()
        );
        let source = allocate_double_source(f);
        let result = ready(recipe.execute(&source.handle)).unwrap();
        let controls = f.controls.lock().unwrap();
        assert_eq!(controls.len(), 2);
        assert_eq!(
            controls[0],
            ProviderRealElementwiseOperation::ScalarAdd {
                control: ScalarLiteral::F64Bits(f64::from(f32::from_bits(bits)).to_bits()),
                mode: runmat_types::RealFloatingAddMode::Double,
            }
        );
        assert!(matches!(
            controls[1],
            ProviderRealElementwiseOperation::FloatingCast {
                target: ProviderPrecision::F32
            }
        ));
        assert!(matches!(
            result.observed.steps,
            ProviderMixedRealAddStepObservations::AddNarrow(_)
        ));
    }
    let f = fixture();
    let mut wrong = double_source_request(1.0_f32.to_bits());
    wrong.operation = ProviderRealElementwiseOperation::ScalarAdd {
        control: ScalarLiteral::F64Bits(1.0_f64.to_bits()),
        mode: runmat_types::RealFloatingAddMode::MixedDoubleThenSingle,
    };
    assert!(ProviderMixedRealAddRecipe::plan(f, wrong)
        .unwrap()
        .is_none());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn double_source_retains_only_real_owned_intermediate_and_exact_two_step_resources() {
    let f = fixture();
    let source = allocate_double_source(f);
    let recipe = plan_double(f, (-0.0_f32).to_bits());
    let result = ready(recipe.execute(&source.handle)).unwrap();
    assert_eq!(
        result.output.physical().element_type(),
        NumericElementType::F32
    );
    assert_eq!(result.output.shape(), source.handle.shape());
    assert!(!result.output.shares_allocation_with(&source.handle));
    assert_eq!(result.observed.resources, *recipe.resources());
    assert_eq!(result.observed.steps.len(), 2);
    assert_eq!(recipe.resources().retained_output_buffer_bytes, 24);
    assert_eq!(recipe.resources().uniform_buffer_bytes, 272);
    assert_eq!(
        recipe.resources().requested_incremental_peak_bytes(),
        Some(296)
    );
    assert_eq!(recipe.resources().dispatches, 2);
    assert_eq!(recipe.resources().queue_submissions, 2);
    assert_eq!(*f.retained.lock().unwrap(), vec![vec![], vec![0]]);
    assert_eq!(
        f.probes
            .lock()
            .unwrap()
            .iter()
            .map(|probe| probe.retirements())
            .collect::<Vec<_>>(),
        vec![1, 0]
    );
    drop(result);
    assert!(f
        .probes
        .lock()
        .unwrap()
        .iter()
        .all(|probe| probe.retirements() == 1));
    assert_eq!(source.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn double_source_failures_are_terminal_after_prefix_and_never_retire_protected_allocations() {
    for at in 1..=2 {
        for fault in [
            Fault::BeforeUnsupported,
            Fault::AfterCause,
            Fault::Alias,
            Fault::WrongClass,
            Fault::WrongReceipt,
        ] {
            let f = fixture();
            let source = allocate_double_source(f);
            let borrowed = allocate_double_source(f);
            *f.fault.lock().unwrap() = Some((at, fault));
            *f.alias.lock().unwrap() = Some(borrowed.handle.clone());
            let error = ready(
                plan_double(f, 1.0_f32.to_bits())
                    .execute_protected(&source.handle, &[&borrowed.handle]),
            )
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
            assert_eq!(borrowed.probe.retirements(), 0);
            assert!(f
                .probes
                .lock()
                .unwrap()
                .iter()
                .all(|probe| probe.retirements() == 1));
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn double_source_pending_cancel_retires_owned_prefix_and_pending_result_only() {
    use std::{
        future::Future,
        task::{Context, Poll, Waker},
    };
    for at in 1..=2 {
        let f = fixture();
        let source = allocate_double_source(f);
        let borrowed = allocate_double_source(f);
        let recipe = plan_double(f, 1.0_f32.to_bits());
        *f.fault.lock().unwrap() = Some((at, Fault::Pending));
        let protected = [&borrowed.handle];
        let mut future = Box::pin(recipe.execute_protected(&source.handle, &protected));
        assert!(matches!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop())),
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
        assert_eq!(borrowed.probe.retirements(), 0);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn double_source_wrong_live_input_fails_before_hooks_and_cleanup_retains_original_causes() {
    let f = fixture();
    let recipe = plan_double(f, 1.0_f32.to_bits());
    let single = allocate_source(f);
    assert!(ready(recipe.execute(&single.handle)).is_err());
    let foreign = allocate_double_source(fixture());
    assert!(ready(recipe.execute(&foreign.handle)).is_err());
    let wrong_shape = f
        .owner
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    assert!(ready(recipe.execute(&wrong_shape.handle)).is_err());
    let dead = allocate_double_source(f);
    dead.probe.mark_unavailable();
    assert!(ready(recipe.execute(&dead.handle)).is_err());
    assert_eq!(f.calls.load(Ordering::SeqCst), 0);
    let source = allocate_double_source(f);
    f.cleanup_failure_at.store(1, Ordering::SeqCst);
    *f.fault.lock().unwrap() = Some((2, Fault::AfterCause));
    let error = ready(recipe.execute(&source.handle)).err().unwrap();
    assert!(error.is_after_effect());
    let completion = std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<ProviderExecutionCompletionError>()
        .unwrap();
    assert!(completion
        .primary()
        .unwrap()
        .to_string()
        .contains("original compound hook cause"));
    assert_eq!(completion.cleanup_errors().len(), 1);
    assert!(std::error::Error::source(completion.primary().unwrap())
        .unwrap()
        .downcast_ref::<std::io::Error>()
        .is_some());
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
    assert_eq!(source.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn double_source_failed_intermediate_retirement_retires_final_output_without_publication() {
    let f = fixture();
    let source = allocate_double_source(f);
    f.cleanup_failure_at.store(1, Ordering::SeqCst);
    let error = ready(plan_double(f, 1.0_f32.to_bits()).execute(&source.handle))
        .err()
        .unwrap();
    assert!(error.is_after_effect());
    assert!(!error.is_unsupported());
    let completion = std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<ProviderExecutionCompletionError>()
        .unwrap();
    assert!(completion.primary().is_none());
    assert_eq!(completion.cleanup_errors().len(), 1);
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
    assert!(f
        .probes
        .lock()
        .unwrap()
        .iter()
        .all(|probe| probe.retirements() == 1));
    assert_eq!(source.probe.retirements(), 0);
}
