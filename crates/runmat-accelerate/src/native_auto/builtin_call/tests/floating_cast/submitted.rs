//! Actual cross-precision dispatch and private-fence completion, not enqueue timing.
use super::*;
use runmat_accelerate_api::{
    AccelContextHandle, AccelContextKind, HostNumericDataOwned, NumericElementType,
};
use runmat_runtime::context::RuntimePlacementFeedbackSource;
use std::sync::atomic::Ordering;

mod gate;
mod provider;

#[derive(Default)]
struct DeferredProbe {
    decisions: Probe,
    source: RefCell<Option<Box<dyn RuntimePlacementFeedbackSource>>>,
    deferrals: Cell<usize>,
}
impl RuntimePlacementService for DeferredProbe {
    fn whole_work_runtime_binding_session_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        binding: runmat_builtins::BuiltinBindingIdentity,
        registration: Digest,
    ) -> Option<PlacementRevision> {
        self.decisions
            .whole_work_runtime_binding_session_revision(program, binding, registration)
    }

    fn plan_with_observation_permit(
        &self,
        request: PlacementPlanRequest,
    ) -> Result<runmat_runtime::context::RuntimeGraphPlacementSelection, RuntimeError> {
        self.decisions.plan_with_observation_permit(request)
    }

    fn plan(
        &self,
        request: PlacementPlanRequest,
    ) -> Result<runmat_execution::PlacementDecision, RuntimeError> {
        self.decisions.plan(request)
    }
    fn observe(&self, feedback: PlacementFeedback) -> Result<(), RuntimeError> {
        self.decisions.observe(feedback)
    }
    fn defer_feedback(
        &self,
        source: Box<dyn RuntimePlacementFeedbackSource>,
    ) -> Result<(), Box<dyn RuntimePlacementFeedbackSource>> {
        self.deferrals.set(self.deferrals.get() + 1);
        assert!(
            self.source.borrow_mut().replace(source).is_none(),
            "only one outstanding receipt"
        );
        Ok(())
    }
    fn invalidate(&self, _: runmat_execution::PlacementInvalidation) {}
}

#[test]
#[ignore = "requires actual shader-f64 WGPU adapter and RUNMAT_WGPU_FORCE_PRECISION=f64; unavailable F64 fails"]
fn actual_single_f64_dispatch_defers_exactly_one_success_until_private_fence_completion() {
    let provider = Box::leak(Box::new(provider::SubmittedCast::new()));
    // Fresh, unregistered inner device: the wrapper is its sole route authority,
    // with no rewritten IDs and no unrelated poller owning this device.
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let probe = Rc::new(DeferredProbe::default());
    probe.decisions.provider.set(true);
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_placement(probe.clone())
            .with_acceleration(Rc::new(Adapter(provider, 1))),
    );
    let values = [
        1.25_f64,
        -0.0,
        16_777_217.0,
        1.0 + f64::from(f32::EPSILON) / 2.0,
    ];
    let input = Value::Tensor(Tensor::new(values.to_vec(), vec![1, 2, 2]).unwrap());
    let output = futures::executor::block_on(context.scope(async {
        runmat_runtime::dispatcher::call_builtin_async_in_context(&context, "single", &[input])
            .await
    }))
    .unwrap();
    let Value::GpuTensor(resident_output) = output.value() else {
        panic!(
            "selected cross-precision device route: plans={}, deferrals={}, casts={}, uploads={}, downloads={}",
            probe.decisions.requests.borrow().len(),
            probe.deferrals.get(),
            provider.casts.load(Ordering::Relaxed),
            provider.uploads.load(Ordering::Relaxed),
            provider.downloads.load(Ordering::Relaxed),
        )
    };
    assert_eq!(resident_output.device_id(), provider.device_id());
    assert_eq!(resident_output.shape(), &[1, 2, 2]);
    assert_eq!(
        resident_output.descriptor().element_type,
        Some(NumericElementType::F32)
    );
    assert_eq!(provider.casts.load(Ordering::Relaxed), 1);
    assert_eq!(
        provider.uploads.load(Ordering::Relaxed),
        1,
        "only input staging; no fallback restoration"
    );
    assert_eq!(
        provider.downloads.load(Ordering::Relaxed),
        0,
        "no host fallback"
    );
    assert_eq!(probe.decisions.requests.borrow().len(), 1);
    assert_eq!(probe.deferrals.get(), 1);
    assert!(probe.decisions.feedback.borrow().is_empty());
    let gate = provider.gate.lock().unwrap().clone().unwrap();
    assert!(
        gate.has_submission(),
        "one real submitted command buffer, not an alias route"
    );
    assert!(
        probe
            .source
            .borrow_mut()
            .as_mut()
            .unwrap()
            .take_ready()
            .is_none(),
        "publication cannot complete the still-held private fence"
    );

    let (submission, retirement) = gate.release();
    let AccelContextHandle::Wgpu(context) = provider
        .inner
        .export_context(AccelContextKind::Plotting)
        .unwrap();
    // Only the test waits. Production submission and publication stay nonblocking.
    // Wait for this exact ticket, never a newest/global queue prefix or a sleep.
    let access = context.access.clone();
    access.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
    // Device completion and delivery by the separate progress owner are
    // distinct events. Await this exact callback's retirement, not a sleep
    // or another global queue poll, before expecting its published receipt.
    futures::executor::block_on(retirement).expect("original progress callback retired");
    let mut source = probe.source.borrow_mut().take().unwrap();
    let feedback = source
        .take_ready()
        .expect("completed private map fence delivers the deferred source");
    let snapshot = feedback.snapshot().expect("completed sample");
    assert!(snapshot.succeeded);
    assert!(snapshot.total_elapsed_ns > 0);
    assert_eq!(
        snapshot.signature,
        probe.decisions.requests.borrow()[0].signature
    );
    probe.observe_deferred_feedback(feedback).unwrap();
    assert!(
        source.take_ready().is_none(),
        "receipt is consumed exactly once"
    );
    assert_eq!(probe.decisions.feedback.borrow().len(), 1);
    assert!(!gate.has_submission());

    let downloaded =
        futures::executor::block_on(provider.inner.download_numeric(resident_output)).unwrap();
    assert_eq!(downloaded.shape, vec![1, 2, 2]);
    let HostNumericDataOwned::F32(actual) = downloaded.data else {
        panic!("physical F32 output")
    };
    assert_eq!(actual.len(), values.len());
    assert_eq!(
        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        values.map(|v| (v as f32).to_bits())
    );
    drop(output);
}
