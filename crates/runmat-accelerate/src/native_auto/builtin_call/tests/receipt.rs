//! Injected completion evidence verifies lifecycle, not actual provider kernels.
use super::*;
use runmat_accelerate_api::observation::{
    current_provider_observation, ProviderExecutionDisposition, ProviderObservationClock,
    ProviderObservationSupport, ProviderOperationReceipt,
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

struct Clock(AtomicU64);
impl ProviderObservationClock for Clock {
    fn now(&self) -> Option<Duration> {
        Some(Duration::from_nanos(self.0.load(Ordering::SeqCst)))
    }
}
struct ObservingProvider {
    clock: Arc<Clock>,
    queries: Mutex<Vec<runmat_accelerate_api::ProviderFeasibilityQuery>>,
}
impl AccelProvider for ObservingProvider {
    fn upload(&self, _: &HostTensorView<'_>) -> Result<runmat_accelerate_api::GpuTensorHandle> {
        unreachable!("no product kernel claimed")
    }
    fn download<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async { unreachable!("no product kernel claimed") })
    }
    fn free(&self, _: &runmat_accelerate_api::GpuTensorHandle) -> Result<()> {
        unreachable!("no allocation in lifecycle fixture")
    }
    fn device_info(&self) -> String {
        "receipt-fixture".into()
    }
    fn begin_operation_observation(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> ProviderObservationSupport {
        self.queries.lock().unwrap().push(query.clone());
        ProviderObservationSupport::Observed(ProviderOperationReceipt::with_clock(
            self.clock.clone(),
        ))
    }
}
fn provider() -> ObservingProvider {
    ObservingProvider {
        clock: Arc::new(Clock(AtomicU64::new(100))),
        queries: Mutex::default(),
    }
}
fn query() -> runmat_accelerate_api::ProviderFeasibilityQuery {
    runmat_accelerate_api::ProviderFeasibilityQuery {
        operation: runmat_accelerate_api::ProviderOperationIdentity::new(
            "fixture.exact-selected-form",
        ),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![],
        outputs: vec![],
        workload: ProviderWorkload {
            elements: Some(7),
            ..Default::default()
        },
    }
}

#[test]
fn direct_receipt_preserves_query_and_includes_staging_before_body() {
    let probe = Rc::new(Probe::default());
    let provider = provider();
    let query = query();
    let mut prepared = observed(&probe, true, vec![]);
    let observation = prepared.observation.as_mut().unwrap();
    observation.prepare_provider(&provider, Some(&query));
    observation
        .recorder()
        .unwrap()
        .submission()
        .unwrap()
        .complete();
    provider.clock.0.store(125, Ordering::SeqCst);
    observation.execution_started(); // must not restart the receipt epoch
    futures::executor::block_on(prepared.execute(|_| async {
        let recorder = current_provider_observation().unwrap();
        recorder.submission().unwrap().complete();
        recorder.execution_disposition(ProviderExecutionDisposition::SelectedProvider);
        provider.clock.0.store(180, Ordering::SeqCst);
        Ok(Value::Num(1.0)) // residency is deliberately not evidence
    }))
    .unwrap();
    assert_eq!(*provider.queries.lock().unwrap(), [query]);
    let feedback = probe.feedback.borrow();
    assert_eq!(feedback.len(), 1);
    assert!(feedback[0].succeeded);
    assert_eq!(feedback[0].total_elapsed_ns, 80);
}

#[test]
fn direct_unproven_or_fallback_execution_never_becomes_positive() {
    for fallback in [false, true] {
        let probe = Rc::new(Probe::default());
        let provider = provider();
        let mut prepared = observed(&probe, true, vec![]);
        prepared
            .observation
            .as_mut()
            .unwrap()
            .prepare_provider(&provider, Some(&query()));
        futures::executor::block_on(prepared.execute(|_| async {
            let recorder = current_provider_observation().unwrap();
            recorder.submission().unwrap().complete();
            if fallback {
                recorder.execution_disposition(ProviderExecutionDisposition::SharedRuntimeFallback);
                recorder.execution_disposition(ProviderExecutionDisposition::SelectedProvider);
            }
            // Host fallback may restore a resident result. That cannot repair
            // the failed route evidence. No provider allocation is fabricated.
            Ok(Value::GpuTensor(
                runmat_accelerate_api::GpuTensorHandle::new(vec![1, 1], 9181, 9182),
            ))
        }))
        .unwrap();
        let feedback = probe.feedback.borrow();
        assert_eq!(feedback.len(), 1);
        assert!(!feedback[0].succeeded);
        assert_eq!(feedback[0].total_elapsed_ns, 0);
    }
}

#[test]
fn absent_admitted_intent_does_not_start_provider_receipt() {
    let probe = Rc::new(Probe::default());
    let provider = provider();
    let mut prepared = observed(&probe, true, vec![]);
    prepared
        .observation
        .as_mut()
        .unwrap()
        .prepare_provider(&provider, None);
    assert!(prepared.observation.as_ref().unwrap().recorder().is_none());
    assert!(provider.queries.lock().unwrap().is_empty());
}

#[test]
fn direct_receipt_abandonment_and_body_error_never_publish_success() {
    for poll in [false, true] {
        let probe = Rc::new(Probe::default());
        let provider = provider();
        let mut prepared = observed(&probe, true, vec![]);
        prepared
            .observation
            .as_mut()
            .unwrap()
            .prepare_provider(&provider, Some(&query()));
        let recorder = prepared.observation.as_ref().unwrap().recorder().unwrap();
        let mut future = Box::pin(prepared.execute(|_| std::future::pending()));
        if poll {
            use std::future::Future;
            assert!(future
                .as_mut()
                .poll(&mut std::task::Context::from_waker(
                    &futures::task::noop_waker()
                ))
                .is_pending());
        }
        drop(future);
        assert_eq!(probe.feedback.borrow().len(), 1);
        assert!(!probe.feedback.borrow()[0].succeeded);
        assert!(matches!(
            recorder.submission(),
            Err(runmat_accelerate_api::observation::ProviderObservationFailure::Abandoned)
        ));
    }
    let probe = Rc::new(Probe::default());
    let provider = provider();
    let mut prepared = observed(&probe, true, vec![]);
    prepared
        .observation
        .as_mut()
        .unwrap()
        .prepare_provider(&provider, Some(&query()));
    assert!(futures::executor::block_on(
        prepared.execute(|_| async { Err(RuntimeError::new("body failed")) })
    )
    .is_err());
    assert_eq!(probe.feedback.borrow().len(), 1);
    assert!(!probe.feedback.borrow()[0].succeeded);
}
