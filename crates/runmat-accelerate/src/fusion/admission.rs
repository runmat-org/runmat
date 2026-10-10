//! One observed owner for domain admission, including rejected or abandoned
//! probes. Elapsed host time is not a measurement of the later fused kernel.
use crate::placement::{FusionPlacementObserver, PlacementCorrelationId};
use runmat_runtime::builtins::common::spec::FusionAdmissionWork;
use runmat_time::Instant;

pub(super) struct AdmissionObservation {
    pub work: FusionAdmissionWork,
    observer: FusionPlacementObserver,
    started: Instant,
    completed: bool,
}

impl AdmissionObservation {
    pub fn new(correlation: Option<PlacementCorrelationId>) -> Self {
        Self {
            work: FusionAdmissionWork::None,
            observer: FusionPlacementObserver::new(correlation),
            started: Instant::now(),
            completed: false,
        }
    }

    pub fn completed(mut self) -> FusionAdmissionWork {
        self.completed = true;
        self.work.clone()
    }
}

impl Drop for AdmissionObservation {
    fn drop(&mut self) {
        self.observer
            .admission(self.started, &self.work, self.completed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::{begin_trace, report, PlacementEventKind};
    use std::{
        future::Future,
        task::{Context, Poll},
    };

    #[test]
    fn cancelled_probe_records_one_incomplete_preparation_not_kernel_success() {
        let trace = begin_trace("fusion.cancelled_admission", None);
        let correlation = trace.correlation();
        let mut observation = AdmissionObservation::new(Some(correlation));
        observation.work = FusionAdmissionWork::Host { elements: 8 };
        let mut future = Box::pin(async move {
            let observation = observation;
            futures::future::pending::<()>().await;
            observation.completed()
        });
        assert!(matches!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
            Poll::Pending
        ));
        drop(future);
        trace.complete(None, None);
        let report = report();
        let recorded = report
            .traces
            .iter()
            .find(|trace| trace.correlation == correlation)
            .unwrap();
        let prepares: Vec<_> = recorded
            .events
            .iter()
            .filter(|event| event.kind == PlacementEventKind::Prepare)
            .collect();
        assert_eq!(prepares.len(), 1);
        assert!(prepares[0].duration_ns.is_some());
        assert!(prepares[0]
            .attributes
            .iter()
            .any(|attribute| attribute.key == "completed" && attribute.value == 0));
        assert!(!recorded
            .events
            .iter()
            .any(|event| event.kind == PlacementEventKind::Kernel));
    }

    #[test]
    fn cancelled_multistage_domain_proof_reports_all_started_work_once() {
        use runmat_runtime::builtins::common::spec::FusionProofStage;
        let trace = begin_trace("fusion.cancelled_real_domain", None);
        let correlation = trace.correlation();
        let mut observation = AdmissionObservation::new(Some(correlation));
        observation.work = FusionAdmissionWork::DomainProof {
            binding: runmat_builtins::BuiltinBindingIdentity {
                builtin: runmat_builtins::ACOSH_CATALOG_ENTRY.identity,
                variant: "default",
            },
            provider_precision: runmat_accelerate_api::ProviderPrecision::F64,
            stages: vec![
                FusionProofStage::HostScan { elements: 3 },
                FusionProofStage::ScalarReadback { elements: 1 },
            ],
        };
        let mut future = Box::pin(async move {
            let observation = observation;
            futures::future::pending::<()>().await;
            observation.completed()
        });
        assert!(matches!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
            Poll::Pending
        ));
        drop(future);
        trace.complete(None, None);
        let report = report();
        let recorded = report
            .traces
            .iter()
            .find(|trace| trace.correlation == correlation)
            .expect("domain trace");
        let prepares: Vec<_> = recorded
            .events
            .iter()
            .filter(|event| event.kind == PlacementEventKind::Prepare)
            .collect();
        assert_eq!(prepares.len(), 1);
        assert!(prepares[0]
            .attributes
            .iter()
            .any(|attribute| { attribute.key == "proof_stages" && attribute.value == 2 }));
        assert!(prepares[0]
            .attributes
            .iter()
            .any(|attribute| { attribute.key == "completed" && attribute.value == 0 }));
        assert!(!recorded
            .events
            .iter()
            .any(|event| event.kind == PlacementEventKind::Kernel));
    }
}
