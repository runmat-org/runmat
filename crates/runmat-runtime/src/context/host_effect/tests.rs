use super::*;
use crate::context::{
    ParallelCapability, RuntimeHostEffectResourceService, RuntimeParallelResources,
    RuntimeParallelService, RuntimePlacementService, RuntimeServicePorts,
    RuntimeWholeWorkProbeLease,
};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::{
    CancellationReason, PlacementDecision, PlacementFeedback, PlacementInvalidation,
    PlacementPlanRequest, PlacementRevision, ResourceBoundUnavailable, WholeWorkDecision,
    WholeWorkDecisionKind, WholeWorkObservation,
};
use std::cell::{Cell, RefCell};
use std::path::Path;
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, OnceLock,
};

struct Resources(Option<HostEffectResourceEvidence>);
impl RuntimeHostEffectResourceService for Resources {
    fn admission(
        &self,
        request: &HostEffectRequest<'_>,
        _runtime: &RuntimeContext,
    ) -> Option<HostEffectResourceAdmission> {
        (request.operation == "unzip"
            && matches!(
                request.workload,
                HostEffectWorkload::ArchiveExtract {
                    source: HostEffectSource::LocalArchive(source),
                    destination: Some(destination),
                    ..
                } if source == Path::new("/fixture.zip") && destination == Path::new("/out")
            )
            && Arc::ptr_eq(request.filesystem, fixture_fs())
            && request.domain_facts == Digest::sha256(b"source-and-destination"))
        .then_some(self.0)
        .flatten()
        .map(test_admission)
    }
}

struct Parallel(Option<u64>);
impl RuntimeParallelService for Parallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        RuntimeParallelResources {
            memory_available_bytes: self.0,
            ..RuntimeParallelResources::default()
        }
    }
}

#[derive(Default)]
struct Placement {
    selections: Cell<usize>,
    observations: RefCell<Vec<WholeWorkObservation>>,
    probes: Arc<AtomicUsize>,
    revision_available: Cell<bool>,
    on_select: RefCell<Option<Box<dyn Fn()>>>,
    on_observe: RefCell<Option<Box<dyn Fn()>>>,
    managed_ledger: RefCell<Option<ManagedHostEffectLedger>>,
}
struct Probe(Arc<AtomicUsize>);
impl RuntimeWholeWorkProbeLease for Probe {}
impl Drop for Probe {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}
impl RuntimePlacementService for Placement {
    fn whole_work_host_service_revision_ref<'a>(
        &self,
        program: Option<&'a runmat_execution::ProgramRevision>,
        host_service: Digest,
    ) -> Option<runmat_execution::PlacementRevisionRef<'a>> {
        self.revision_available.get().then(|| {
            let digest = Digest::sha256(b"host-effect-test-placement");
            runmat_execution::PlacementRevisionRef {
                program,
                catalog: digest,
                compiler: digest,
                authority: runmat_execution::PlacementExecutionAuthorityRevision::HostService(
                    host_service,
                ),
                policy: digest,
            }
        })
    }

    fn whole_work_host_service_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        host_service: Digest,
    ) -> Option<PlacementRevision> {
        if !self.revision_available.get() {
            return None;
        }
        let digest = Digest::sha256(b"host-effect-test-placement");
        Some(PlacementRevision {
            program,
            catalog: digest,
            compiler: digest,
            authority: runmat_execution::PlacementExecutionAuthorityRevision::HostService(
                host_service,
            ),
            policy: digest,
        })
    }
    fn select_whole_work(
        &self,
        request: WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, crate::RuntimeError> {
        request.validate().unwrap();
        let candidate = &request.candidates[0];
        if let Some(available) = request.resources.memory_available_bytes {
            let demand = match (
                candidate.resources.retained_bytes,
                candidate.resources.scratch_bytes,
            ) {
                (ResourceByteBound::Known(retained), ResourceByteBound::Known(scratch)) => {
                    retained.checked_add(scratch)
                }
                _ => None,
            };
            if demand.is_none_or(|demand| demand > available) {
                return Err(build_runtime_error("test placement resource limit")
                    .with_identifier("RunMat:Placement:ResourceLimit")
                    .build());
            }
        }
        if let Some(callback) = self.on_select.borrow().as_ref() {
            callback();
        }
        self.selections.set(self.selections.get() + 1);
        self.probes.fetch_add(1, Ordering::SeqCst);
        Ok(RuntimeWholeWorkSelection {
            decision: WholeWorkDecision {
                signature: request.signature.to_owned(),
                selected: candidate.selected(&request.signature.operation),
                kind: WholeWorkDecisionKind::Unmeasured,
                predicted_total_ns: None,
            },
            probe: Some(Box::new(Probe(self.probes.clone()))),
            observation_permit: None,
        })
    }
    fn select_whole_work_borrowed(
        &self,
        request: runmat_execution::WholeWorkPlacementRequestRef<'_>,
    ) -> Result<RuntimeWholeWorkSelection, crate::RuntimeError> {
        // Production consumes the borrowed view directly; this test-only
        // adapter reuses its existing staged placement oracle.
        self.select_whole_work(WholeWorkPlacementRequest {
            signature: request.signature.to_owned(),
            constraint: request.constraint,
            candidates: request.candidates.to_vec(),
            resources: request.resources.clone(),
            deterministic: request.deterministic,
            require_transactional_results: request.require_transactional_results,
        })
    }
    fn select_whole_work_borrowed_managed(
        &self,
        request: runmat_execution::WholeWorkPlacementRequestRef<'_>,
    ) -> Result<crate::context::RuntimeManagedWholeWorkSelection, crate::RuntimeError> {
        request.validate().unwrap();
        let candidate = &request.candidates[0];
        if let Some(available) = request.resources.memory_available_bytes {
            let demand = match (
                candidate.resources.retained_bytes,
                candidate.resources.scratch_bytes,
            ) {
                (ResourceByteBound::Known(retained), ResourceByteBound::Known(scratch)) => {
                    retained.checked_add(scratch)
                }
                _ => None,
            };
            if demand.is_none_or(|demand| demand > available) {
                return Err(build_runtime_error("test placement resource limit")
                    .with_identifier("RunMat:Placement:ResourceLimit")
                    .build());
            }
        }
        if let Some(callback) = self.on_select.borrow().as_ref() {
            callback();
        }
        self.selections.set(self.selections.get() + 1);
        self.probes.fetch_add(1, Ordering::SeqCst);
        crate::context::test_managed_selection(
            request,
            Some(Box::new(Probe(self.probes.clone()))),
            self.managed_ledger
                .borrow()
                .as_ref()
                .expect("staged fixture ledger"),
        )
    }
    fn observe_managed_whole_work(
        &self,
        selected: &crate::context::RuntimeManagedWholeWorkSelection,
        elapsed_ns: u64,
        outcome: runmat_execution::WholeWorkObservationOutcome,
    ) -> Result<(), crate::RuntimeError> {
        if let Some(callback) = self.on_observe.borrow().as_ref() {
            callback();
        }
        self.observations
            .borrow_mut()
            .push(selected.test_observation(elapsed_ns, outcome));
        Ok(())
    }
    fn observe_whole_work(
        &self,
        observation: WholeWorkObservation,
    ) -> Result<(), crate::RuntimeError> {
        if let Some(callback) = self.on_observe.borrow().as_ref() {
            callback();
        }
        self.observations.borrow_mut().push(observation);
        Ok(())
    }
    fn plan(&self, _: PlacementPlanRequest) -> Result<PlacementDecision, crate::RuntimeError> {
        unreachable!("host effect uses complete-work selection")
    }
    fn observe(&self, _: PlacementFeedback) -> Result<(), crate::RuntimeError> {
        unreachable!("host effect uses complete-work observation")
    }
    fn invalidate(&self, _: PlacementInvalidation) {}
}

fn fixture_fs() -> &'static Arc<dyn runmat_filesystem::FsProvider> {
    static FS: OnceLock<Arc<dyn runmat_filesystem::FsProvider>> = OnceLock::new();
    FS.get_or_init(|| Arc::new(runmat_filesystem::MemoryFsProvider::new()))
}

fn request(output: HostEffectOutput) -> HostEffectRequest<'static> {
    HostEffectRequest {
        operation: "unzip",
        workload: HostEffectWorkload::ArchiveExtract {
            source: HostEffectSource::LocalArchive(Path::new("/fixture.zip")),
            destination: Some(Path::new("/out")),
            output,
        },
        domain_facts: Digest::sha256(b"source-and-destination"),
        filesystem: fixture_fs(),
    }
}

#[test]
fn numeric_text_final_facts_require_matching_typed_workload_and_digest() {
    let facts = runmat_builtins::ReadMatrixTextResourceFacts {
        opened_extent: 8,
        source_resource_revision: [7; 32],
        rows: 2,
        columns: 2,
        result_class: runmat_builtins::ReadMatrixResultClass::Numeric(
            runmat_types::NumericClass::Double,
        ),
    };
    let domain_facts = Digest::sha256(serde_json::to_vec(&facts).expect("typed facts"));
    let matching = HostEffectRequest {
        operation: "readmatrix",
        workload: HostEffectWorkload::ReadMatrixText {
            source: Path::new("/numbers.csv"),
            controls: Digest::sha256(b"validated options"),
            output: HostEffectOutput::ReturnedValue,
        },
        domain_facts,
        filesystem: fixture_fs(),
    };
    HostEffectFinalFacts::ReadMatrixText(&facts)
        .validate_for(&matching)
        .expect("same opened-source facts");
    let wrong_digest = HostEffectRequest {
        domain_facts: Digest::sha256(b"different source"),
        ..matching
    };
    assert!(HostEffectFinalFacts::ReadMatrixText(&facts)
        .validate_for(&wrong_digest)
        .is_err());
    assert!(HostEffectFinalFacts::ReadMatrixText(&facts)
        .validate_for(&request(HostEffectOutput::ReturnedValue))
        .is_err());
}

fn runtime_context(
    placement: Rc<Placement>,
    evidence: Option<HostEffectResourceEvidence>,
    available: Option<u64>,
) -> RuntimeContext {
    placement.revision_available.set(true);
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_placement(placement)
                .with_host_effect_resources(Rc::new(Resources(evidence)))
                .with_parallel(Rc::new(Parallel(available))),
        )
}

fn bounds(retained: ResourceByteBound, scratch: ResourceByteBound) -> HostEffectResourceEvidence {
    HostEffectResourceEvidence {
        authority_revision: Digest::sha256(b"typed-service-revision"),
        scratch_bytes: scratch,
        returned_value_bytes: retained,
    }
}

#[test]
fn archive_identity_retains_the_original_tuple_encoding() {
    #[derive(serde::Serialize)]
    enum PreviousSourceFacts<'a> {
        LocalArchive(&'a [u8]),
        HttpArchive(&'a str),
    }
    let local = request(HostEffectOutput::ReturnedValue).workload;
    let expected = serde_json::to_vec(&(
        PreviousSourceFacts::LocalArchive(b"/fixture.zip"),
        Some(b"/out".as_slice()),
        HostEffectOutput::ReturnedValue,
    ))
    .unwrap();
    assert_eq!(local.identity_bytes(), expected);

    let http = HostEffectWorkload::ArchiveExtract {
        source: HostEffectSource::HttpArchive("https://example.test/archive.zip"),
        destination: None,
        output: HostEffectOutput::EffectOnly,
    };
    let expected = serde_json::to_vec(&(
        PreviousSourceFacts::HttpArchive("https://example.test/archive.zip"),
        None::<&[u8]>,
        HostEffectOutput::EffectOnly,
    ))
    .unwrap();
    assert_eq!(http.identity_bytes(), expected);
}

#[path = "tests/audio.rs"]
mod audio;
#[path = "tests/mat.rs"]
mod mat;
#[path = "tests/stage.rs"]
mod stage;
#[path = "tests/dataset.rs"]
mod dataset;

#[test]
fn finite_missing_or_unknown_bound_rejects_before_selection() {
    let placement = Rc::new(Placement::default());
    for evidence in [
        None,
        Some(bounds(
            ResourceByteBound::Known(4),
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoPeakMemoryBound),
        )),
        Some(bounds(
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound),
            ResourceByteBound::Known(4),
        )),
    ] {
        let runtime = runtime_context(placement.clone(), evidence, Some(32));
        let error = select_host_effect(Some(&runtime), request(HostEffectOutput::ReturnedValue))
            .err()
            .expect("finite unknown must reject");
        assert_eq!(
            error.identifier(),
            Some("RunMat:Placement:InsufficientResourceEvidence")
        );
    }
    assert_eq!(placement.selections.get(), 0);

    let runtime = runtime_context(
        placement.clone(),
        Some(bounds(
            ResourceByteBound::Known(4),
            ResourceByteBound::Known(4),
        )),
        Some(32),
    );
    let mut foreign_source = request(HostEffectOutput::ReturnedValue);
    foreign_source.workload = HostEffectWorkload::ArchiveExtract {
        source: HostEffectSource::LocalArchive(Path::new("/other.zip")),
        destination: Some(Path::new("/out")),
        output: HostEffectOutput::ReturnedValue,
    };
    assert!(select_host_effect(Some(&runtime), foreign_source).is_err());
    assert_eq!(placement.selections.get(), 0);
    let other_fs: Arc<dyn runmat_filesystem::FsProvider> =
        Arc::new(runmat_filesystem::MemoryFsProvider::new());
    let mut foreign_owner = request(HostEffectOutput::ReturnedValue);
    foreign_owner.filesystem = &other_fs;
    assert!(select_host_effect(Some(&runtime), foreign_owner).is_err());
    assert_eq!(placement.selections.get(), 0);

    let runtime = runtime_context(
        placement.clone(),
        Some(bounds(
            ResourceByteBound::Known(4),
            ResourceByteBound::Known(4),
        )),
        Some(32),
    );
    placement.revision_available.set(false);
    let error = select_host_effect(Some(&runtime), request(HostEffectOutput::ReturnedValue))
        .err()
        .expect("finite missing revision must reject");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(placement.selections.get(), 0);
}

#[test]
fn finite_effect_only_uses_zero_retained_but_still_budgets_scratch() {
    let placement = Rc::new(Placement::default());
    let evidence = Some(bounds(
        ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound),
        ResourceByteBound::Known(24),
    ));
    let enough = runtime_context(placement.clone(), evidence, Some(24));
    let receipt = select_host_effect(Some(&enough), request(HostEffectOutput::EffectOnly))
        .unwrap()
        .expect("effect-only call admitted");
    assert_eq!(placement.probes.load(Ordering::SeqCst), 1);
    receipt.finish(WholeWorkObservationOutcome::Success);
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    assert_eq!(placement.observations.borrow().len(), 1);
    assert!(placement.observations.borrow()[0].elapsed_ns > 0);
    let failed = select_host_effect(Some(&enough), request(HostEffectOutput::EffectOnly))
        .unwrap()
        .unwrap();
    failed.finish(WholeWorkObservationOutcome::OperationalFailure);
    assert_eq!(placement.observations.borrow().len(), 2);
    assert_eq!(
        placement.observations.borrow()[1].outcome,
        WholeWorkObservationOutcome::OperationalFailure
    );
    let insufficient = runtime_context(placement.clone(), evidence, Some(23));
    assert!(
        select_host_effect(Some(&insufficient), request(HostEffectOutput::EffectOnly)).is_err()
    );
    assert_eq!(placement.selections.get(), 2);
}

#[test]
fn finite_returned_value_charges_retained_and_scratch_together() {
    let placement = Rc::new(Placement::default());
    let evidence = Some(bounds(
        ResourceByteBound::Known(12),
        ResourceByteBound::Known(24),
    ));
    let enough = runtime_context(placement.clone(), evidence, Some(36));
    select_host_effect(Some(&enough), request(HostEffectOutput::ReturnedValue))
        .unwrap()
        .unwrap()
        .finish(WholeWorkObservationOutcome::Success);
    let short = runtime_context(placement.clone(), evidence, Some(35));
    assert!(select_host_effect(Some(&short), request(HostEffectOutput::ReturnedValue)).is_err());
    assert_eq!(placement.selections.get(), 1);
}

#[test]
fn dropped_and_cancelled_receipts_release_probe_without_failure_feedback() {
    let placement = Rc::new(Placement::default());
    let runtime = runtime_context(
        placement.clone(),
        Some(bounds(
            ResourceByteBound::Known(8),
            ResourceByteBound::Known(8),
        )),
        Some(16),
    );
    drop(select_host_effect(Some(&runtime), request(HostEffectOutput::ReturnedValue)).unwrap());
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    assert!(placement.observations.borrow().is_empty());

    let cancellation = CancellationSource::new();
    let cancelled = runtime.with_cancellation_token(cancellation.token());
    let receipt = select_host_effect(Some(&cancelled), request(HostEffectOutput::ReturnedValue))
        .unwrap()
        .unwrap();
    cancellation.cancel(CancellationReason::User);
    receipt.finish(WholeWorkObservationOutcome::OperationalFailure);
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    assert!(placement.observations.borrow().is_empty());
}

#[test]
fn revoked_resource_authority_rejects_completion_and_releases_probe() {
    use std::sync::atomic::AtomicBool;

    struct Revocable(Arc<AtomicBool>);
    impl HostEffectResourceLivenessLease for Revocable {
        fn is_current(&self) -> bool {
            self.0.load(Ordering::SeqCst)
        }
    }
    struct RevocableResources(Arc<AtomicBool>);
    impl RuntimeHostEffectResourceService for RevocableResources {
        fn admission(
            &self,
            request: &HostEffectRequest<'_>,
            _: &RuntimeContext,
        ) -> Option<HostEffectResourceAdmission> {
            Arc::ptr_eq(request.filesystem, fixture_fs()).then(|| HostEffectResourceAdmission {
                evidence: bounds(ResourceByteBound::Known(8), ResourceByteBound::Known(8)),
                liveness_lease: Arc::new(Revocable(self.0.clone())),
                output_allocation_permit: None,
            })
        }
    }
    let valid = Arc::new(AtomicBool::new(true));
    let placement = Rc::new(Placement::default());
    placement.revision_available.set(true);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_placement(placement.clone())
                .with_host_effect_resources(Rc::new(RevocableResources(valid.clone())))
                .with_parallel(Rc::new(Parallel(Some(16)))),
        );
    let receipt = select_host_effect(Some(&runtime), request(HostEffectOutput::ReturnedValue))
        .unwrap()
        .unwrap();
    assert!(receipt.checkpoint().is_ok());
    valid.store(false, Ordering::SeqCst);
    assert_eq!(
        receipt.checkpoint().unwrap_err().identifier(),
        Some("RunMat:Placement:StaleResourceAuthority")
    );
    receipt.finish(WholeWorkObservationOutcome::Success);
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    assert!(placement.observations.borrow().is_empty());
    assert_eq!(
        select_host_effect(Some(&runtime), request(HostEffectOutput::ReturnedValue))
            .err()
            .unwrap()
            .identifier(),
        Some("RunMat:Placement:StaleResourceAuthority")
    );
}

#[cfg(unix)]
#[test]
fn non_utf8_source_and_destination_reach_service_losslessly_and_have_distinct_identity() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    use std::path::PathBuf;

    #[derive(Debug, PartialEq, Eq)]
    struct ObservedPaths {
        source: Vec<u8>,
        destination: Vec<u8>,
    }
    struct RawPathResources(Rc<RefCell<Vec<ObservedPaths>>>);
    impl RuntimeHostEffectResourceService for RawPathResources {
        fn admission(
            &self,
            request: &HostEffectRequest<'_>,
            _runtime: &RuntimeContext,
        ) -> Option<HostEffectResourceAdmission> {
            let HostEffectWorkload::ArchiveExtract {
                source: HostEffectSource::LocalArchive(source),
                destination: Some(destination),
                ..
            } = request.workload
            else {
                return None;
            };
            self.0.borrow_mut().push(ObservedPaths {
                source: source.as_os_str().as_encoded_bytes().to_vec(),
                destination: destination.as_os_str().as_encoded_bytes().to_vec(),
            });
            Some(test_admission(bounds(
                ResourceByteBound::Known(1),
                ResourceByteBound::Known(1),
            )))
        }
    }

    fn invalid_path(stem: &[u8], byte: u8) -> PathBuf {
        let mut bytes = stem.to_vec();
        bytes.push(byte);
        PathBuf::from(OsString::from_vec(bytes))
    }
    fn raw_request<'a>(source: &'a Path, destination: &'a Path) -> HostEffectRequest<'a> {
        HostEffectRequest {
            operation: "unzip",
            workload: HostEffectWorkload::ArchiveExtract {
                source: HostEffectSource::LocalArchive(source),
                destination: Some(destination),
                output: HostEffectOutput::ReturnedValue,
            },
            domain_facts: Digest::sha256(b"same-domain-controls"),
            filesystem: fixture_fs(),
        }
    }

    let source_a = invalid_path(b"/archive-", 0xff);
    let source_b = invalid_path(b"/archive-", 0xfe);
    let destination_a = invalid_path(b"/output-", 0xff);
    let destination_b = invalid_path(b"/output-", 0xfe);
    assert_eq!(source_a.to_string_lossy(), source_b.to_string_lossy());
    assert_eq!(
        destination_a.to_string_lossy(),
        destination_b.to_string_lossy()
    );

    let seen = Rc::new(RefCell::new(Vec::new()));
    let placement = Rc::new(Placement::default());
    placement.revision_available.set(true);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_placement(placement)
                .with_host_effect_resources(Rc::new(RawPathResources(seen.clone())))
                .with_parallel(Rc::new(Parallel(Some(2)))),
        );
    let identity = |source: &Path, destination: &Path| {
        let receipt = select_host_effect(Some(&runtime), raw_request(source, destination))
            .expect("exact raw paths are admitted")
            .expect("whole-work selection");
        match receipt.selected {
            SelectedHostEffect::Portable(ref selected) => selected.decision.signature.runtime_facts,
            SelectedHostEffect::Managed(_) => {
                panic!("direct admission must use portable selection")
            }
        }
    };
    let first = identity(&source_a, &destination_a);
    let changed_source = identity(&source_b, &destination_a);
    let changed_destination = identity(&source_a, &destination_b);
    assert_ne!(first, changed_source);
    assert_ne!(first, changed_destination);
    assert_eq!(
        seen.borrow().as_slice(),
        &[
            ObservedPaths {
                source: source_a.as_os_str().as_encoded_bytes().to_vec(),
                destination: destination_a.as_os_str().as_encoded_bytes().to_vec(),
            },
            ObservedPaths {
                source: source_b.as_os_str().as_encoded_bytes().to_vec(),
                destination: destination_a.as_os_str().as_encoded_bytes().to_vec(),
            },
            ObservedPaths {
                source: source_a.as_os_str().as_encoded_bytes().to_vec(),
                destination: destination_b.as_os_str().as_encoded_bytes().to_vec(),
            },
        ]
    );
}
