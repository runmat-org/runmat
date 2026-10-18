use super::*;
use crate::context::physical_test_support::OwnedPhysicalProvider;
use crate::context::{RuntimePlacementService, RuntimeServicePorts, RuntimeWholeWorkProbeLease};
use crate::RuntimeError;
use runmat_accelerate_api::observation::{ProviderOperationReceipt, WgpuSubmissionMeasurement};
use runmat_accelerate_api::{
    ProviderElementType, ProviderLayout, ProviderRepresentation, ProviderResidency,
    ProviderStorage, ThreadProviderGuard,
};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::profiling::{
    CommandProfilingReceipt, CommandProfilingRoot, ProviderObservationCounts,
};
use runmat_execution::{
    CancellationReason, Digest, ExecutionCandidateKind, PlacementDecision, PlacementFeedback,
    PlacementInvalidation, PlacementPlanRequest, PlacementRevision, SelectedExecutionCandidate,
    WholeWorkDecision, WholeWorkDecisionKind, WholeWorkPlacementRequest,
};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Default)]
pub(crate) struct Witness {
    selections: Cell<usize>,
    feedback: RefCell<Vec<PlacementFeedback>>,
    pending: RefCell<Vec<Box<dyn RuntimePlacementFeedbackSource>>>,
}

impl Witness {
    pub(crate) fn context(self: &Rc<Self>) -> RuntimeContext {
        RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
            .with_service_ports(RuntimeServicePorts::default().with_placement(self.clone()))
    }

    pub(crate) fn selections(&self) -> usize {
        self.selections.get()
    }

    pub(crate) fn feedback(&self) -> Vec<PlacementFeedback> {
        self.feedback.borrow().clone()
    }

    pub(crate) fn drain(&self) {
        let mut pending = std::mem::take(&mut *self.pending.borrow_mut());
        for mut source in pending.drain(..) {
            if let Some(feedback) = source.take_ready() {
                RuntimePlacementService::observe_deferred_feedback(self, feedback).unwrap();
            } else if !source.terminal_without_feedback() {
                self.pending.borrow_mut().push(source);
            }
        }
    }
}

impl RuntimePlacementService for Witness {
    fn whole_work_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        provider: Digest,
    ) -> Option<PlacementRevision> {
        let digest = Digest::sha256(b"host-presentation-witness");
        Some(PlacementRevision {
            program,
            catalog: digest,
            compiler: digest,
            authority: runmat_execution::PlacementExecutionAuthorityRevision::Provider(provider),
            policy: digest,
        })
    }

    fn select_whole_work(
        &self,
        request: WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, RuntimeError> {
        request.validate().expect("admitted host presentation");
        let candidate = request.candidates.first().expect("host upload candidate");
        self.selections.set(self.selections.get() + 1);
        Ok(RuntimeWholeWorkSelection {
            decision: WholeWorkDecision {
                signature: request.signature.clone(),
                selected: candidate.selected(&request.signature.operation),
                kind: WholeWorkDecisionKind::Unmeasured,
                predicted_total_ns: None,
            },
            probe: None,
            observation_permit: None,
        })
    }
    fn plan(&self, _: PlacementPlanRequest) -> Result<PlacementDecision, RuntimeError> {
        unreachable!("receipt owner never schedules")
    }
    fn observe(&self, feedback: PlacementFeedback) -> Result<(), RuntimeError> {
        self.feedback.borrow_mut().push(feedback);
        Ok(())
    }
    fn defer_feedback(
        &self,
        source: Box<dyn RuntimePlacementFeedbackSource>,
    ) -> Result<(), Box<dyn RuntimePlacementFeedbackSource>> {
        self.pending.borrow_mut().push(source);
        Ok(())
    }
    fn invalidate(&self, _: PlacementInvalidation) {}
}

struct Probe(Arc<AtomicUsize>);
impl RuntimeWholeWorkProbeLease for Probe {}
impl Drop for Probe {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

struct TestObservationPermit(std::sync::atomic::AtomicBool);
impl crate::context::RuntimeWholeWorkObservationPermit for TestObservationPermit {
    fn is_current(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

fn receipt_owner(
    witness: Option<&Rc<Witness>>,
    cancellation: &CancellationSource,
    probe_count: Arc<AtomicUsize>,
    receipt: ProviderOperationReceipt,
) -> HostPresentationReceipt {
    receipt_owner_with_command(witness, cancellation, probe_count, receipt, None)
}

fn receipt_owner_with_command(
    witness: Option<&Rc<Witness>>,
    cancellation: &CancellationSource,
    probe_count: Arc<AtomicUsize>,
    receipt: ProviderOperationReceipt,
    command: Option<CommandProfilingReceipt>,
) -> HostPresentationReceipt {
    let mut runtime =
        RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
            .with_cancellation_token(cancellation.token());
    if let Some(command) = command {
        runtime = runtime.with_command_profiling(command);
    }
    if let Some(witness) = witness {
        runtime = runtime
            .with_service_ports(RuntimeServicePorts::default().with_placement(witness.clone()));
    }
    probe_count.fetch_add(1, Ordering::SeqCst);
    let digest = Digest::sha256(b"host-presentation-witness");
    let selected = RuntimeWholeWorkSelection {
        decision: WholeWorkDecision {
            signature: PlacementSignature {
                region: None,
                operation: "imhist".into(),
                runtime_facts: digest,
                provider_provenance: None,
                graph_output_authority: None,
                revision: PlacementRevision {
                    program: None,
                    catalog: digest,
                    compiler: digest,
                    authority: runmat_execution::PlacementExecutionAuthorityRevision::Provider(
                        digest,
                    ),
                    policy: digest,
                },
            },
            selected: SelectedExecutionCandidate {
                node: "imhist".into(),
                candidate: "host-upload".into(),
                kind: ExecutionCandidateKind::SharedRuntime,
            },
            kind: WholeWorkDecisionKind::Exploratory,
            predicted_total_ns: None,
        },
        probe: Some(Box::new(Probe(probe_count))),
        observation_permit: None,
    };
    HostPresentationReceipt::new(runtime, selected, receipt)
}

#[test]
fn command_receipt_settles_only_after_the_unique_provider_completion() {
    let witness = Rc::new(Witness::default());
    let cancellation = CancellationSource::new();
    let command = CommandProfilingRoot::new();
    let borrowed = command.receipt();
    let owner = receipt_owner_with_command(
        Some(&witness),
        &cancellation,
        Arc::new(AtomicUsize::new(0)),
        ProviderOperationReceipt::new(),
        Some(borrowed.clone()),
    );
    let submission = owner.recorder().submission().expect("physical submission");
    owner.finish(Ok(true));
    witness.drain();
    assert_eq!(
        command.seal().observations,
        Some(ProviderObservationCounts {
            pending: 1,
            ..ProviderObservationCounts::default()
        })
    );
    submission.complete();
    witness.drain();
    assert!(borrowed.begin_provider_operation().is_none());
}

#[test]
fn command_receipt_observes_immediate_completion_without_device_claim() {
    let witness = Rc::new(Witness::default());
    let cancellation = CancellationSource::new();
    let command = CommandProfilingRoot::new();
    let owner = receipt_owner_with_command(
        Some(&witness),
        &cancellation,
        Arc::new(AtomicUsize::new(0)),
        ProviderOperationReceipt::new(),
        Some(command.receipt()),
    );
    owner.finish(Ok(true));
    assert_eq!(witness.feedback().len(), 1);
    assert_eq!(
        command.seal().observations,
        Some(ProviderObservationCounts {
            incomplete: 1,
            ..ProviderObservationCounts::default()
        })
    );
}

#[test]
fn deferred_verified_dispatch_count_does_not_invent_device_time_or_command_coverage() {
    let witness = Rc::new(Witness::default());
    let cancellation = CancellationSource::new();
    let command = CommandProfilingRoot::new();
    let owner = receipt_owner_with_command(
        Some(&witness),
        &cancellation,
        Arc::new(AtomicUsize::new(0)),
        ProviderOperationReceipt::new(),
        Some(command.receipt()),
    );
    let submission = owner.recorder().submission().expect("physical submission");
    owner.finish(Ok(true));
    assert!(witness.feedback().is_empty());
    submission
        .capture_completion()
        .complete_with_wgpu(WgpuSubmissionMeasurement {
            compute_elapsed: None,
            device_elapsed: None,
            dispatches: Some(2),
            query_allocation: None,
        });
    witness.drain();
    assert_eq!(witness.feedback().len(), 1);
    let snapshot = command.seal();
    assert_eq!(snapshot.observations.expect("valid").incomplete, 1);
    assert_eq!(
        snapshot.observed_metric_coverage,
        Some(runmat_execution::profiling::ObservedMetricCoverage {
            gpu_execution: false,
            kernel_count: true,
        })
    );
    assert_eq!(snapshot.kernel_count, None);
    assert_eq!(snapshot.gpu_execution, None);
}

#[test]
fn publication_waits_for_actual_submission_completion_and_holds_probe() {
    let witness = Rc::new(Witness::default());
    let cancellation = CancellationSource::new();
    let probes = Arc::new(AtomicUsize::new(0));
    let receipt = ProviderOperationReceipt::new();
    let owner = receipt_owner(Some(&witness), &cancellation, probes.clone(), receipt);
    let submission = owner.recorder().submission().unwrap();
    owner.finish(Ok(true));
    assert!(witness.feedback.borrow().is_empty());
    assert_eq!(witness.pending.borrow().len(), 1);
    assert_eq!(probes.load(Ordering::SeqCst), 1);
    witness.drain();
    assert!(witness.feedback.borrow().is_empty());
    submission.complete();
    witness.drain();
    assert_eq!(witness.feedback.borrow().len(), 1);
    assert_eq!(probes.load(Ordering::SeqCst), 0);
}

#[test]
fn revoked_selection_discards_late_provider_completion_without_feedback() {
    let witness = Rc::new(Witness::default());
    let cancellation = CancellationSource::new();
    let probes = Arc::new(AtomicUsize::new(0));
    let command = CommandProfilingRoot::new();
    let receipt = ProviderOperationReceipt::new();
    let mut owner = receipt_owner_with_command(
        Some(&witness),
        &cancellation,
        probes.clone(),
        receipt,
        Some(command.receipt()),
    );
    let permit = Arc::new(TestObservationPermit(std::sync::atomic::AtomicBool::new(
        true,
    )));
    let PresentationSelection::Portable(selected) = &mut owner.selected else {
        panic!("expected portable selection");
    };
    selected.observation_permit = Some(
        super::super::RuntimeWholeWorkObservationAuthority::Dynamic(permit.clone()),
    );
    let submission = owner.recorder().submission().unwrap();
    owner.finish(Ok(true));
    assert_eq!(witness.pending.borrow().len(), 1);
    permit.0.store(false, Ordering::Release);
    submission.complete();
    witness.drain();
    assert!(witness.feedback.borrow().is_empty());
    assert!(witness.pending.borrow().is_empty());
    assert_eq!(probes.load(Ordering::SeqCst), 0);
    assert_eq!(command.seal().observations.expect("valid").missing, 1);
}

#[test]
fn revoked_selection_cannot_register_a_late_deferred_source() {
    let witness = Rc::new(Witness::default());
    let cancellation = CancellationSource::new();
    let probes = Arc::new(AtomicUsize::new(0));
    let receipt = ProviderOperationReceipt::new();
    let mut owner = receipt_owner(Some(&witness), &cancellation, probes.clone(), receipt);
    let permit = Arc::new(TestObservationPermit(std::sync::atomic::AtomicBool::new(
        true,
    )));
    let PresentationSelection::Portable(selected) = &mut owner.selected else {
        panic!("expected portable selection");
    };
    selected.observation_permit = Some(
        super::super::RuntimeWholeWorkObservationAuthority::Dynamic(permit.clone()),
    );
    let submission = owner.recorder().submission().unwrap();
    permit.0.store(false, Ordering::Release);
    owner.finish(Ok(true));
    assert!(witness.pending.borrow().is_empty());
    assert!(witness.feedback.borrow().is_empty());
    assert_eq!(probes.load(Ordering::SeqCst), 0);
    submission.complete();
    witness.drain();
    assert!(witness.feedback.borrow().is_empty());
}

#[test]
fn failures_cancellation_and_abandonment_do_not_fabricate_samples() {
    for (failure, expected_samples) in [
        (Some(ProviderObservationFailure::ProviderFailed), 1),
        (Some(ProviderObservationFailure::SharedRuntimeFallback), 0),
        (Some(ProviderObservationFailure::Cancelled), 0),
    ] {
        let witness = Rc::new(Witness::default());
        let cancellation = CancellationSource::new();
        let probes = Arc::new(AtomicUsize::new(0));
        let owner = receipt_owner(
            Some(&witness),
            &cancellation,
            probes.clone(),
            ProviderOperationReceipt::new(),
        );
        owner.finish(Err(failure.unwrap()));
        witness.drain();
        assert_eq!(witness.feedback.borrow().len(), expected_samples);
        assert_eq!(probes.load(Ordering::SeqCst), 0);
    }

    let witness = Rc::new(Witness::default());
    let cancellation = CancellationSource::new();
    let probes = Arc::new(AtomicUsize::new(0));
    let owner = receipt_owner(
        Some(&witness),
        &cancellation,
        probes.clone(),
        ProviderOperationReceipt::new(),
    );
    drop(owner);
    assert_eq!(probes.load(Ordering::SeqCst), 0);
    assert!(witness.feedback.borrow().is_empty());

    let owner = receipt_owner(
        Some(&witness),
        &cancellation,
        probes.clone(),
        ProviderOperationReceipt::new(),
    );
    cancellation.cancel(CancellationReason::User);
    owner.finish(Ok(true));
    witness.drain();
    assert!(witness.feedback.borrow().is_empty());
    assert_eq!(probes.load(Ordering::SeqCst), 0);
}

#[test]
fn absent_placement_service_retires_receipt_without_feedback() {
    let cancellation = CancellationSource::new();
    let probes = Arc::new(AtomicUsize::new(0));
    let owner = receipt_owner(
        None,
        &cancellation,
        probes.clone(),
        ProviderOperationReceipt::new(),
    );
    owner.finish(Ok(true));
    assert_eq!(probes.load(Ordering::SeqCst), 0);
}

#[test]
fn admitted_upload_cannot_publish_success_without_its_realized_output_witness() {
    let provider: &'static OwnedPhysicalProvider =
        Box::leak(Box::new(OwnedPhysicalProvider::new()));
    let _owner = ThreadProviderGuard::set(Some(provider));
    let exact = provider.allocate_f64(&[1, 1]);
    let (larger, larger_view) = provider.allocate_padded_f64_view(&[2, 2], &[1, 1]);
    assert_eq!(larger_view.shape(), exact.handle.shape());
    assert_eq!(
        larger_view.physical().element_type(),
        exact.handle.physical().element_type()
    );
    assert!(larger_view.physical().requested_bytes() > exact.handle.physical().requested_bytes());
    let admitted = || AdmittedHostUploadOutputs {
        provider,
        outputs: vec![ProviderRepresentation {
            element_type: ProviderElementType::F64,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![1, 1],
            residency: ProviderResidency::Device,
        }],
        one_output_requested_bytes: Some(exact.handle.physical().requested_bytes()),
    };

    for mode in 0..3 {
        let witness = Rc::new(Witness::default());
        let cancellation = CancellationSource::new();
        let probes = Arc::new(AtomicUsize::new(0));
        let mut owner = receipt_owner(
            Some(&witness),
            &cancellation,
            probes.clone(),
            ProviderOperationReceipt::new(),
        )
        .with_admitted_upload_outputs(admitted());
        let submission = owner.recorder().submission().expect("upload submission");
        match mode {
            0 => { /* Attempt to bypass the publication validator. */ }
            1 => {
                owner
                    .validate_published_outputs(&[&exact.handle])
                    .expect("first exact output");
                let error = owner
                    .validate_published_outputs(&[&larger_view])
                    .expect_err("larger backing allocation must be rejected");
                assert_eq!(
                    error.identifier(),
                    Some("RunMat:Placement:InvalidPublishedOutput")
                );
                assert!(owner.validate_published_outputs(&[&exact.handle]).is_err());
            }
            _ => owner
                .validate_published_outputs(&[&exact.handle])
                .expect("exact realized output"),
        }
        submission.complete();
        owner.finish(Ok(true));
        witness.drain();
        assert_eq!(witness.feedback().len(), usize::from(mode == 2));
        assert_eq!(probes.load(Ordering::SeqCst), 0);
    }
    assert!(exact.handle.allocation_lease().validate_live().is_ok());
    assert!(larger.handle.allocation_lease().validate_live().is_ok());
    assert_eq!(exact.probe.retirements(), 0);
    assert_eq!(larger.probe.retirements(), 0);
}
