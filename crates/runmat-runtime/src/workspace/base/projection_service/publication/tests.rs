use super::*;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
    RuntimeServicePorts, RuntimeWorkspaceService,
};
use crate::workspace::{
    BaseWorkspaceOwner, BaseWorkspaceProjectionService, GlobalBindingTarget, WorkspaceSlotReadLease,
};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_value::Value;
use std::cell::RefCell;
use std::rc::Rc;

struct ManagedObserverMemory(ManagedHostEffectLedger);

impl RuntimeParallelService for ManagedObserverMemory {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..RuntimeParallelResources::default()
        }
    }

    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum OwnedOrigin {
    ExecutorSlot {
        name: String,
        slot: usize,
        global: Option<GlobalBindingTarget>,
    },
    DynamicEffect {
        name: String,
        global: Option<GlobalBindingTarget>,
    },
    BindGlobal {
        name: String,
        target: GlobalBindingTarget,
    },
    UnbindGlobal {
        name: String,
    },
    Retired {
        name: String,
    },
    Cleared,
}

fn own(origin: WorkspacePublicationOrigin<'_>) -> OwnedOrigin {
    match origin {
        WorkspacePublicationOrigin::ExecutorSlot { name, slot, global } => {
            OwnedOrigin::ExecutorSlot {
                name: name.to_owned(),
                slot,
                global: global.cloned(),
            }
        }
        WorkspacePublicationOrigin::DynamicEffect { name, global } => OwnedOrigin::DynamicEffect {
            name: name.to_owned(),
            global: global.cloned(),
        },
        WorkspacePublicationOrigin::BindGlobal { name, target } => OwnedOrigin::BindGlobal {
            name: name.to_owned(),
            target: target.clone(),
        },
        WorkspacePublicationOrigin::UnbindGlobal { name } => OwnedOrigin::UnbindGlobal {
            name: name.to_owned(),
        },
        WorkspacePublicationOrigin::Retired { name } => OwnedOrigin::Retired {
            name: name.to_owned(),
        },
        WorkspacePublicationOrigin::Cleared => OwnedOrigin::Cleared,
    }
}

#[derive(Debug)]
struct RecordingProtocol {
    prepared: RefCell<Option<Vec<OwnedOrigin>>>,
    observed: Rc<RefCell<Vec<OwnedOrigin>>>,
}

impl WorkspacePublicationProtocol for RecordingProtocol {
    fn prepare(
        &self,
        _context: Option<&RuntimeContext>,
        changes: WorkspacePublicationBatch<'_>,
    ) -> Result<WorkspacePublicationToken, RuntimeError> {
        let mut prepared = self.prepared.try_borrow_mut().map_err(|_| {
            crate::build_runtime_error("publication preparation is reentrant").build()
        })?;
        if prepared.is_some() {
            return Err(crate::build_runtime_error("publication is already prepared").build());
        }
        *prepared = Some(changes.iter().map(own).collect());
        Ok(WorkspacePublicationToken::from_generation(1))
    }

    fn checkpoint(
        &self,
        token: WorkspacePublicationToken,
        _context: Option<&RuntimeContext>,
    ) -> Result<(), RuntimeError> {
        assert_eq!(token.generation(), 1);
        Ok(())
    }

    fn commit(&self, token: WorkspacePublicationToken) {
        assert_eq!(token.generation(), 1);
        let prepared = self
            .prepared
            .borrow_mut()
            .take()
            .expect("publication was prepared before commit");
        self.observed.borrow_mut().extend(prepared);
    }

    fn abort(&self, token: WorkspacePublicationToken) {
        assert_eq!(token.generation(), 1);
        self.prepared.borrow_mut().take();
    }
}

fn recording_observer(observed: Rc<RefCell<Vec<OwnedOrigin>>>) -> WorkspacePublicationObserver {
    WorkspacePublicationObserver::new(RecordingProtocol {
        prepared: RefCell::new(None),
        observed,
    })
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn managed_observer_retains_exact_control_backing_until_final_owner_drops() {
    let ledger = ManagedHostEffectLedger::new(65_536);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_parallel(Rc::new(ManagedObserverMemory(ledger.clone()))),
        );
    let before = ledger.snapshot().free_bytes();
    let observed = Rc::new(RefCell::new(Vec::new()));
    let observer = WorkspacePublicationObserver::new_with_context(
        &runtime,
        RecordingProtocol {
            prepared: RefCell::new(None),
            observed,
        },
    )
    .expect("managed observer admission");
    assert!(ledger.snapshot().free_bytes() < before);
    let alias = observer.clone();
    drop(observer);
    assert!(ledger.snapshot().free_bytes() < before);
    drop(alias);
    assert_eq!(ledger.snapshot().free_bytes(), before);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn committed_last_writer_origin_survives_service_drop() {
    let owner = BaseWorkspaceOwner::new();
    let observed = Rc::new(RefCell::new(Vec::new()));
    let observer = recording_observer(Rc::clone(&observed));
    let service = BaseWorkspaceProjectionService::new(owner.begin_projection().unwrap())
        .with_publication_observer(observer);
    service
        .assign_executor_slot(2, "source", Value::Num(1.0))
        .unwrap();
    service.assign("source", Value::Num(3.0)).unwrap();
    service.assign("written", Value::Num(-1.0)).unwrap();
    service
        .assign_executor_slot(4, "written", Value::Num(4.0))
        .unwrap();
    service.bind_global("global").unwrap();
    service.bind_global("unbound").unwrap();
    service.unbind_global("unbound").unwrap();
    service.publish().unwrap();
    drop(service);
    assert_eq!(
        *observed.borrow(),
        vec![
            OwnedOrigin::BindGlobal {
                name: "global".into(),
                target: GlobalBindingTarget::Named(crate::workspace::WorkspaceName::unbounded(
                    "global"
                )),
            },
            OwnedOrigin::DynamicEffect {
                name: "source".into(),
                global: None,
            },
            OwnedOrigin::UnbindGlobal {
                name: "unbound".into()
            },
            OwnedOrigin::ExecutorSlot {
                name: "written".into(),
                slot: 4,
                global: None,
            },
        ]
    );
    assert_eq!(owner.lookup("source").unwrap(), Some(Value::Num(3.0)));
    assert_eq!(owner.lookup("written").unwrap(), Some(Value::Num(4.0)));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn failed_publication_never_reports_staged_source_origin() {
    let owner = BaseWorkspaceOwner::new();
    let observed = Rc::new(RefCell::new(Vec::new()));
    let observer = recording_observer(Rc::clone(&observed));
    let service = BaseWorkspaceProjectionService::new(owner.begin_projection().unwrap())
        .with_publication_observer(observer);
    service
        .assign_executor_slot(1, "unpublished", Value::Num(7.0))
        .unwrap();
    owner.close().unwrap();
    assert!(service.publish().is_err());
    assert!(observed.borrow().is_empty());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn clear_then_source_store_reports_the_post_clear_origin() {
    let owner = BaseWorkspaceOwner::new();
    let observed = Rc::new(RefCell::new(Vec::new()));
    let observer = recording_observer(Rc::clone(&observed));
    let service = BaseWorkspaceProjectionService::new(owner.begin_projection().unwrap())
        .with_publication_observer(observer);
    service
        .assign_executor_slot(1, "retired", Value::Num(1.0))
        .unwrap();
    service.clear().unwrap();
    service
        .assign_executor_slot(2, "restored", Value::Num(2.0))
        .unwrap();
    service.publish().unwrap();
    assert_eq!(
        *observed.borrow(),
        vec![
            OwnedOrigin::Cleared,
            OwnedOrigin::ExecutorSlot {
                name: "restored".into(),
                slot: 2,
                global: None,
            },
        ]
    );
    assert_eq!(owner.lookup("retired").unwrap(), None);
    assert_eq!(owner.lookup("restored").unwrap(), Some(Value::Num(2.0)));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn context_bound_publication_requires_the_matching_live_context() {
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    let owner = BaseWorkspaceOwner::new();
    let service = BaseWorkspaceProjectionService::new_for_context(
        owner.begin_projection().unwrap(),
        &runtime,
    );
    service
        .assign_executor_slot_ref(&runtime, 3, "bound", &Value::Num(9.0))
        .unwrap();

    let error = service
        .publish()
        .expect_err("contextless publish must fail");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Workspace:PublicationContextRequired")
    );
    let error = service
        .take_published_changes()
        .expect_err("contextless drain must not consume a bound invocation");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Workspace:PublicationContextRequired")
    );
    service.publish_with_context(&runtime).unwrap();
    assert_eq!(owner.lookup("bound").unwrap(), Some(Value::Num(9.0)));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn completion_capture_stabilizes_prefix_but_rejects_intervening_writes() {
    use crate::workspace::{WorkspaceCompletionAction, WorkspaceCompletionSource};

    struct Assignment<'a> {
        name: &'a str,
        value: Value,
    }
    impl WorkspaceCompletionSource for Assignment<'_> {
        fn len(&self) -> usize {
            1
        }

        fn action(&self, index: usize) -> Option<WorkspaceCompletionAction<'_>> {
            (index == 0).then_some(WorkspaceCompletionAction::AssignLocal {
                name: self.name,
                value: (&self.value).into(),
            })
        }
    }

    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    let owner = BaseWorkspaceOwner::new();
    let service = BaseWorkspaceProjectionService::new_for_context(
        owner.begin_projection().unwrap(),
        &runtime,
    );
    service
        .assign_executor_slot_ref(&runtime, 0, "prefix", &Value::Num(7.0))
        .unwrap();
    assert!(service.has_pending_changes().unwrap());
    let identity = service
        .capture_workspace_completion_identity_with_context(&runtime)
        .unwrap()
        .unwrap();
    assert!(!service.has_pending_changes().unwrap());
    assert_eq!(owner.lookup("prefix").unwrap(), Some(Value::Num(7.0)));
    // The child can read the target before completing without changing the
    // identity captured for this exact mutation.
    let snapshot = service.try_snapshot_with_context(&runtime).unwrap();
    assert_eq!(snapshot, vec![("prefix".into(), Value::Num(7.0))]);
    service
        .apply_workspace_completion_with_context(
            &runtime,
            identity,
            &Assignment {
                name: "completed",
                value: Value::Num(11.0),
            },
        )
        .unwrap();
    assert_eq!(
        service
            .try_lookup_with_context(&runtime, "completed")
            .unwrap(),
        Some(Value::Num(11.0))
    );
    assert_eq!(
        service.try_lookup_with_context(&runtime, "prefix").unwrap(),
        Some(Value::Num(7.0))
    );

    let stale = service
        .capture_workspace_completion_identity_with_context(&runtime)
        .unwrap()
        .unwrap();
    service
        .assign_executor_slot_ref(&runtime, 1, "intervening", &Value::Num(13.0))
        .unwrap();
    let error = service
        .apply_workspace_completion_with_context(
            &runtime,
            stale,
            &Assignment {
                name: "rejected",
                value: Value::Num(17.0),
            },
        )
        .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:Workspace:CompletionConflict")
    );
    assert!(service.has_pending_changes().unwrap());
    assert_eq!(
        service
            .try_lookup_with_context(&runtime, "intervening")
            .unwrap(),
        Some(Value::Num(13.0))
    );
    assert_eq!(
        service
            .try_lookup_with_context(&runtime, "rejected")
            .unwrap(),
        None
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn cancelled_context_drains_the_already_published_prefix_without_new_admission() {
    for finite in [false, true] {
        let cancellation = CancellationSource::new();
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(crate::execution::RuntimeExecutionService::new()),
            cancellation.token(),
        );
        let runtime = if finite {
            runtime.with_service_ports(RuntimeServicePorts::default().with_parallel(Rc::new(
                ManagedObserverMemory(ManagedHostEffectLedger::new(65_536)),
            )))
        } else {
            runtime
        };
        let owner = BaseWorkspaceOwner::new();
        let service = BaseWorkspaceProjectionService::new_for_context(
            owner.begin_projection().unwrap(),
            &runtime,
        );
        service
            .assign_executor_slot_ref(&runtime, 3, "prefix", &Value::Num(91.0))
            .unwrap();
        service.publish_with_context(&runtime).unwrap();
        cancellation.cancel(CancellationReason::User);
        let changes = service
            .take_published_changes_with_context(&runtime)
            .expect("committed effects drain after cancellation");
        assert_eq!(changes.changes().len(), 1);
        assert_eq!(changes.changes()[0].name(), Some("prefix"));
        assert_eq!(changes.changes()[0].value(), Some(&Value::Num(91.0)));
        assert_eq!(owner.lookup("prefix").unwrap(), Some(Value::Num(91.0)));
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn finite_borrowed_aggregate_write_refuses_before_publication() {
    let ledger = ManagedHostEffectLedger::new(65_536);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_parallel(Rc::new(ManagedObserverMemory(ledger.clone()))),
        );
    let owner = BaseWorkspaceOwner::new();
    let service = BaseWorkspaceProjectionService::new_for_context(
        owner.begin_projection().unwrap(),
        &runtime,
    );
    let value = Value::Cell(
        runmat_value::CellArray::new(vec![Value::String("nested".into())], 1, 1).unwrap(),
    );
    let error = service
        .assign_executor_slot_ref(&runtime, 3, "unsupported", &value)
        .expect_err("finite borrowed aggregate needs its admitted value owner");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Workspace:ExactWriteAdmissionUnavailable")
    );
    assert_eq!(owner.lookup("unsupported").unwrap(), None);
    assert!(!service.has_pending_changes().unwrap());

    // The same all-class value can cross the direct executor boundary when
    // its immutable generation and independent roots are admitted first.
    let retained = WorkspaceSlotReadLease::prepare_owned(value, &runtime).unwrap();
    service
        .assign_executor_slot_lease_with_context(&runtime, 3, "supported", retained.clone())
        .unwrap();
    let held_headroom = ledger
        .reserve_stage(ledger.snapshot().free_bytes())
        .unwrap();
    assert_eq!(ledger.snapshot().free_bytes(), 0);
    service
        .publish_admitted_mutations_with_context(&runtime)
        .unwrap();
    assert!(!service.has_pending_changes().unwrap());
    drop(held_headroom);
    assert_eq!(
        owner.lookup("supported").unwrap(),
        Some(retained.value().clone())
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn finite_private_owned_write_moves_the_original_aggregate_generation() {
    let ledger = ManagedHostEffectLedger::new(65_536);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_parallel(Rc::new(ManagedObserverMemory(ledger.clone()))),
        );
    let (owner, service) = BaseWorkspaceProjectionService::private_empty(Some(&runtime)).unwrap();
    let baseline = ledger.snapshot().live_claimed_bytes;
    let cell =
        runmat_value::CellArray::new(vec![Value::String("owned input".into())], 1, 1).unwrap();
    let payload = cell.data().as_ptr();
    let shape = cell.shape.as_ptr();
    let value = Value::Cell(cell);
    assert!(service
        .assign_ref_with_context(&runtime, "input", &value)
        .is_err());
    assert!(service
        .try_lookup_with_context(&runtime, "input")
        .unwrap()
        .is_none());

    service
        .assign_with_context(&runtime, "input", value)
        .unwrap();
    let generation = service
        .capture_workspace_generation_with_context(&runtime)
        .unwrap()
        .unwrap();
    let retained = generation.read_lease("input").unwrap();
    let Value::Cell(published) = retained.value() else {
        panic!("owned Cell must retain its class");
    };
    assert_eq!(published.data().as_ptr(), payload);
    assert_eq!(published.shape.as_ptr(), shape);
    assert!(ledger.snapshot().live_claimed_bytes > baseline);
    drop(generation);
    drop(service);
    drop(owner);
    assert_eq!(published.data(), &[Value::String("owned input".into())]);
    drop(retained);
    assert_eq!(ledger.snapshot().live_claimed_bytes, baseline);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn finite_repeated_executor_store_retires_prior_gc_generation_before_publication() {
    runmat_gc::gc_test_context(|| {
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(RuntimeServicePorts::default().with_parallel(Rc::new(
                    ManagedObserverMemory(ManagedHostEffectLedger::new(65_536)),
                )));
        let owner = BaseWorkspaceOwner::new();
        let service = BaseWorkspaceProjectionService::new_for_context(
            owner.begin_projection().unwrap(),
            &runtime,
        );
        let rooted = |label: &str| {
            let root = runmat_gc::gc_allocate_rooted(Value::String(label.into())).unwrap();
            let handle = root.handle();
            let value = Value::HandleObject(runmat_value::HandleRef {
                class_name: "WorkspaceHandle".into(),
                target: handle,
                valid: true,
            });
            (root, handle, value)
        };
        let (first_root, first_handle, first) = rooted("first");
        service
            .assign_executor_slot_lease_with_context(
                &runtime,
                0,
                "held",
                WorkspaceSlotReadLease::prepare_owned(first, &runtime).unwrap(),
            )
            .unwrap();
        first_root.unroot().unwrap();

        let (second_root, second_handle, second) = rooted("second");
        service
            .assign_executor_slot_lease_with_context(
                &runtime,
                0,
                "held",
                WorkspaceSlotReadLease::prepare_owned(second, &runtime).unwrap(),
            )
            .unwrap();
        second_root.unroot().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&first_handle).is_err());
        assert!(runmat_gc::gc_clone_value(&second_handle).is_ok());
        assert!(service.has_pending_changes().unwrap());
        assert_eq!(owner.lookup("held").unwrap(), None);
        service.clear_with_context(&runtime).unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(
            runmat_gc::gc_clone_value(&second_handle).is_err(),
            "an unpublished clear must not retain its superseded pending generation"
        );
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn finite_publication_retries_after_canonical_borrow_conflict() {
    let ledger = ManagedHostEffectLedger::new(65_536);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_parallel(Rc::new(ManagedObserverMemory(ledger.clone()))),
        );
    let owner = BaseWorkspaceOwner::new();
    let service = BaseWorkspaceProjectionService::new_for_context(
        owner.begin_projection().unwrap(),
        &runtime,
    );
    service
        .assign_executor_slot_ref(&runtime, 3, "pending", &Value::Num(7.0))
        .unwrap();
    let admitted_peak = ledger.snapshot().peak_reserved_bytes;

    let error = owner
        .with_binding_names(|_| {
            service
                .publish_with_context(&runtime)
                .expect_err("canonical read borrow must reject publication")
        })
        .unwrap();
    // The service maps the canonical BaseWorkspaceError::BorrowConflict to Busy.
    assert_eq!(error.identifier(), Some("RunMat:Workspace:Busy"));
    assert!(service.has_pending_changes().unwrap());
    assert_eq!(owner.lookup("pending").unwrap(), None);
    assert_eq!(ledger.snapshot().peak_reserved_bytes, admitted_peak);

    service
        .publish_with_context(&runtime)
        .expect("the same staged mutation remains publishable after the borrow ends");
    assert_eq!(owner.lookup("pending").unwrap(), Some(Value::Num(7.0)));
    let published = service
        .take_published_changes_with_context(&runtime)
        .unwrap();
    assert_eq!(published.changes().len(), 1);
    assert_eq!(published.changes()[0].name(), Some("pending"));
    assert_eq!(published.changes()[0].value(), Some(&Value::Num(7.0)));
    assert_eq!(ledger.snapshot().peak_reserved_bytes, admitted_peak);
}

#[derive(Debug, Default)]
struct RejectSecondProspective {
    first: RefCell<Option<WorkspacePublicationStamp>>,
}

impl WorkspacePublicationProtocol for RejectSecondProspective {
    fn prepare_mutation(
        &self,
        _context: Option<&RuntimeContext>,
        mutation: WorkspacePublicationMutation<'_>,
    ) -> Result<Option<WorkspacePublicationToken>, RuntimeError> {
        if self.first.borrow().is_some() {
            return Err(crate::build_runtime_error("second metadata preparation refused").build());
        }
        assert!(matches!(
            mutation.origin,
            WorkspacePublicationOrigin::ExecutorSlot { .. }
        ));
        *self.first.borrow_mut() = Some(mutation.stamp);
        Ok(Some(WorkspacePublicationToken::from_generation(2)))
    }

    fn prepare(
        &self,
        _context: Option<&RuntimeContext>,
        batch: WorkspacePublicationBatch<'_>,
    ) -> Result<WorkspacePublicationToken, RuntimeError> {
        assert_eq!(batch.publication_stamp(), *self.first.borrow());
        Ok(WorkspacePublicationToken::from_generation(3))
    }

    fn checkpoint(
        &self,
        _token: WorkspacePublicationToken,
        _context: Option<&RuntimeContext>,
    ) -> Result<(), RuntimeError> {
        Ok(())
    }

    fn commit(&self, _token: WorkspacePublicationToken) {}

    fn abort(&self, _token: WorkspacePublicationToken) {}
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn failed_second_metadata_preparation_preserves_first_source_prefix() {
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    let owner = BaseWorkspaceOwner::new();
    let service = BaseWorkspaceProjectionService::new_for_context(
        owner.begin_projection().unwrap(),
        &runtime,
    )
    .with_publication_observer(WorkspacePublicationObserver::new(
        RejectSecondProspective::default(),
    ));
    service
        .assign_executor_slot_ref(&runtime, 1, "first", &Value::Num(7.0))
        .unwrap();
    assert!(service
        .assign_executor_slot_ref(&runtime, 2, "second", &Value::Num(9.0))
        .is_err());
    service.publish_with_context(&runtime).unwrap();
    assert_eq!(owner.lookup("first").unwrap(), Some(Value::Num(7.0)));
    assert_eq!(owner.lookup("second").unwrap(), None);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn contextual_detachment_stages_one_final_retirement_batch() {
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    let owner = BaseWorkspaceOwner::new();
    let observed = Rc::new(RefCell::new(Vec::new()));
    let service = BaseWorkspaceProjectionService::new_for_context(
        owner.begin_projection().unwrap(),
        &runtime,
    )
    .with_publication_observer(recording_observer(Rc::clone(&observed)));
    service
        .assign_executor_slot_ref(&runtime, 1, "kept", &Value::Num(1.0))
        .unwrap();
    service
        .assign_executor_slot_ref(&runtime, 2, "removed", &Value::Num(2.0))
        .unwrap();
    service.publish_with_context(&runtime).unwrap();
    observed.borrow_mut().clear();
    let detached = RuntimeWorkspaceService::detach_selected_with_context(
        &service,
        &runtime,
        &crate::workspace::WorkspaceDetachSelection::Names(vec!["removed".into()]),
    )
    .unwrap();
    assert_eq!(detached.entries().len(), 1);
    assert_eq!(
        *observed.borrow(),
        vec![OwnedOrigin::Retired {
            name: "removed".into()
        }]
    );
    assert_eq!(owner.lookup("removed").unwrap(), None);
    assert_eq!(owner.lookup("kept").unwrap(), Some(Value::Num(1.0)));
}
