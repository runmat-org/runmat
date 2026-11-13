use super::*;
use crate::context::host_effect::{
    HostEffectRequest, HostEffectResourceAdmission, HostEffectResourceEvidence, HostEffectWorkload,
};
use crate::context::{
    RuntimeContext, RuntimeHostEffectResourceService, RuntimePlacementService, RuntimeServicePorts,
    RuntimeWholeWorkSelection,
};
use runmat_execution::{
    Digest, PlacementDecision, PlacementFeedback, PlacementInvalidation, PlacementPlanRequest,
    PlacementRevision, ResourceByteBound, WholeWorkDecision, WholeWorkDecisionKind,
    WholeWorkObservation, WholeWorkObservationOutcome, WholeWorkPlacementRequest,
};
use std::cell::{Cell, RefCell};
use std::path::Path;
use std::rc::Rc;

fn manifest(id: &str) -> Vec<u8> {
    let value = DataManifest {
        schema_version: 1,
        format: "runmat-data".into(),
        dataset_id: id.into(),
        name: Some("same.data".into()),
        created_at: "2026-01-01T00:00:00Z".into(),
        updated_at: "2026-01-01T00:00:00Z".into(),
        arrays: BTreeMap::new(),
        attrs: BTreeMap::new(),
        txn_sequence: 0,
    };
    serde_json::to_vec(&value).expect("typed manifest fixture")
}

struct Evidence {
    original: Arc<dyn FsProvider>,
    seen: Cell<usize>,
}
impl RuntimeHostEffectResourceService for Evidence {
    fn admission(
        &self,
        request: &HostEffectRequest<'_>,
        _: &RuntimeContext,
    ) -> Option<HostEffectResourceAdmission> {
        assert!(Arc::ptr_eq(request.filesystem, &self.original));
        assert!(matches!(
            request.workload,
            HostEffectWorkload::DatasetPath {
                operation: DatasetPathOperation::Open,
                source,
                destination: None,
                ..
            } if source == Path::new("/original/same.data")
        ));
        assert_eq!(request.domain_facts, admission::facts(&"same.data"));
        self.seen.set(self.seen.get() + 1);
        // Synthetic test evidence exercises ownership/receipt, not a
        // production finite resource certificate.
        Some(crate::context::host_effect::test_admission(
            HostEffectResourceEvidence {
                authority_revision: Digest::sha256(b"dataset-test-owner"),
                scratch_bytes: ResourceByteBound::Known(1_024),
                returned_value_bytes: ResourceByteBound::Known(1_024),
            },
        ))
    }
}

struct SwappingPlacement {
    replacement: Arc<dyn FsProvider>,
    guard: RefCell<Option<fs::ProviderGuard>>,
    selected: Cell<usize>,
    observations: RefCell<Vec<WholeWorkObservation>>,
}
impl RuntimePlacementService for SwappingPlacement {
    fn whole_work_host_service_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        authority: Digest,
    ) -> Option<PlacementRevision> {
        let revision = Digest::sha256(b"dataset-placement-test");
        Some(PlacementRevision {
            program,
            catalog: revision,
            compiler: revision,
            authority: runmat_execution::PlacementExecutionAuthorityRevision::HostService(
                authority,
            ),
            policy: revision,
        })
    }

    fn select_whole_work(
        &self,
        request: WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, crate::RuntimeError> {
        request.validate().expect("typed selected host effect");
        self.selected.set(self.selected.get() + 1);
        *self.guard.borrow_mut() = Some(fs::replace_provider(self.replacement.clone()));
        // Model the host changing its CWD while the invocation's executor is
        // active. The blocking host API belongs on that external host thread.
        std::thread::spawn(|| fs::set_current_dir("/replacement"))
            .join()
            .expect("host CWD update thread")
            .expect("replacement provider CWD");
        assert_eq!(fs::current_dir().unwrap(), Path::new("/replacement"));
        let candidate = &request.candidates[0];
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

    fn observe_whole_work(
        &self,
        observation: WholeWorkObservation,
    ) -> Result<(), crate::RuntimeError> {
        self.observations.borrow_mut().push(observation);
        Ok(())
    }

    fn plan(&self, _: PlacementPlanRequest) -> Result<PlacementDecision, crate::RuntimeError> {
        unreachable!("dataset path route uses whole-work selection")
    }
    fn observe(&self, _: PlacementFeedback) -> Result<(), crate::RuntimeError> {
        unreachable!("dataset path route uses whole-work observation")
    }
    fn invalidate(&self, _: PlacementInvalidation) {}
}

#[test]
fn selected_open_keeps_original_provider_and_cwd_through_await_and_publication() {
    let _serial = fs::provider_override_lock();
    let original = Arc::new(fs::MemoryFsProvider::with_current_dir("/original"));
    let replacement = Arc::new(fs::MemoryFsProvider::with_current_dir("/replacement"));
    original
        .write_project_path("/original/same.data/manifest.json", &manifest("original"))
        .unwrap();
    replacement
        .write_project_path(
            "/original/same.data/manifest.json",
            &manifest("wrong-owner"),
        )
        .unwrap();
    replacement
        .write_project_path(
            "/replacement/same.data/manifest.json",
            &manifest("wrong-cwd"),
        )
        .unwrap();
    let _original_guard = fs::replace_provider(original.clone() as Arc<dyn FsProvider>);
    let evidence = Rc::new(Evidence {
        original: original.clone(),
        seen: Cell::new(0),
    });
    let placement = Rc::new(SwappingPlacement {
        replacement: replacement.clone(),
        guard: RefCell::new(None),
        selected: Cell::new(0),
        observations: RefCell::new(Vec::new()),
    });
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_host_effect_resources(evidence.clone())
                .with_placement(placement.clone()),
        );
    let value = futures::executor::block_on(data_open_builtin(
        Some(runtime),
        Value::String("same.data".into()),
    ))
    .expect("captured original provider is still authoritative");
    let Value::Object(object) = value.value() else {
        panic!("data.open must return a Dataset object");
    };
    assert_eq!(
        object.properties().get("__data_id"),
        Some(&Value::String("original".into()))
    );
    assert_eq!(evidence.seen.get(), 1);
    assert_eq!(placement.selected.get(), 1);
    let observations = placement.observations.borrow();
    assert_eq!(observations.len(), 1);
    assert_eq!(
        observations[0].outcome,
        WholeWorkObservationOutcome::Success
    );
}

#[test]
fn typed_cooperation_cancellation_aborts_without_operational_failure_feedback() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
    use runmat_execution::cancellation::CancellationSource;
    use std::num::NonZeroU64;

    let _serial = fs::provider_override_lock();
    let original = Arc::new(fs::MemoryFsProvider::with_current_dir("/original"));
    let replacement = Arc::new(fs::MemoryFsProvider::with_current_dir("/replacement"));
    let _original_guard = fs::replace_provider(original.clone() as Arc<dyn FsProvider>);
    let evidence = Rc::new(Evidence {
        original: original.clone(),
        seen: Cell::new(0),
    });
    let placement = Rc::new(SwappingPlacement {
        replacement,
        guard: RefCell::new(None),
        selected: Cell::new(0),
        observations: RefCell::new(Vec::new()),
    });
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_host_effect_resources(evidence)
                .with_placement(placement.clone()),
        );
    let authority = DataPathAuthority::capture(&["same.data"]).unwrap();
    let source = authority.resolve("same.data").unwrap();
    let receipt = admission::select(
        Some(&runtime),
        &authority,
        DatasetPathOperation::Open,
        &source,
        None,
        admission::facts(&"same.data"),
    )
    .unwrap();
    let mut budget = Some(CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        CancellationSource::new().token(),
        || Box::pin(async { Err(ExecutionServiceError::Cancelled) }),
    ));
    let error = futures::executor::block_on(admission::execute(Some(&runtime), receipt, async {
        crate::data::cooperate_data_work(&mut budget, 1).await?;
        Ok(Value::Bool(true))
    }))
    .expect_err("typed cancellation must abort publication");
    assert_eq!(error.identifier.as_deref(), Some("RunMat:Cancelled"));
    assert!(matches!(
        error
            .source
            .as_deref()
            .and_then(|source| source.downcast_ref::<ExecutionServiceError>()),
        Some(ExecutionServiceError::Cancelled)
    ));
    assert!(
        !runtime.is_cancelled(),
        "the test isolates typed error provenance"
    );
    let observed = placement.observations.borrow();
    assert!(
        observed.is_empty(),
        "cancelled work must not publish a cost/failure observation"
    );
}
