use super::*;
use crate::builtin::{RuntimeBuiltinFuture, RuntimeBuiltinInvocation};
use crate::context::{RuntimeServicePorts, RuntimeWholeWorkProbeLease};
use runmat_execution::{
    PlacementDecision, PlacementFeedback, PlacementInvalidation, PlacementPlanRequest,
    PlacementRevision, WholeWorkDecision, WholeWorkDecisionKind,
};
use runmat_value::Value;
use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::task::{Context, Poll};

mod comparison_tests;
mod prepared_calls;

struct Scoped {
    binding: RuntimeBuiltinBinding,
    generation: Cell<u64>,
}

impl RuntimeBuiltinService for Scoped {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        if name == self.binding.identity.builtin.name {
            vec![self.binding]
        } else {
            Vec::new()
        }
    }

    fn binding_registration_revision(&self, binding: RuntimeBuiltinBinding) -> Option<Digest> {
        (binding.identity == self.binding.identity
            && std::ptr::fn_addr_eq(binding.implementation, self.binding.implementation))
        .then(|| {
            Digest::sha256_serialized(&self.generation.get())
                .expect("typed test generation serializes")
        })
    }
}

#[derive(Default)]
struct Placement {
    selections: Cell<usize>,
    observations: Cell<usize>,
    signatures: RefCell<Vec<Digest>>,
    probe_drops: Arc<AtomicUsize>,
    revision_available: Cell<bool>,
    on_select: RefCell<Option<Box<dyn Fn()>>>,
}

struct Probe(Arc<AtomicUsize>);
impl RuntimeWholeWorkProbeLease for Probe {}
impl Drop for Probe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl RuntimePlacementService for Placement {
    fn whole_work_runtime_binding_session_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        binding: runmat_builtins::BuiltinBindingIdentity,
        registration: Digest,
    ) -> Option<PlacementRevision> {
        self.revision_available.get().then(|| {
            let policy = Digest::sha256(b"direct-call-test-policy");
            let authority = Digest::sha256_serialized(&(binding, registration))
                .expect("typed test authority serializes");
            PlacementRevision {
                program,
                catalog: policy,
                compiler: policy,
                authority: PlacementExecutionAuthorityRevision::RuntimeBindingSession(authority),
                policy,
            }
        })
    }

    fn select_whole_work(
        &self,
        request: WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, crate::RuntimeError> {
        request.validate().expect("admitted test request");
        assert_eq!(request.candidates.len(), 1);
        assert_eq!(
            request.candidates[0].kind,
            ExecutionCandidateKind::SharedRuntime
        );
        assert_eq!(request.resources.memory_available_bytes, None);
        self.selections.set(self.selections.get() + 1);
        self.signatures
            .borrow_mut()
            .push(request.signature.runtime_facts);
        if let Some(callback) = self.on_select.borrow().as_ref() {
            callback();
        }
        Ok(RuntimeWholeWorkSelection {
            decision: WholeWorkDecision {
                signature: request.signature.clone(),
                selected: request.candidates[0].selected(&request.signature.operation),
                kind: WholeWorkDecisionKind::Exploratory,
                predicted_total_ns: None,
            },
            probe: Some(Box::new(Probe(self.probe_drops.clone()))),
            observation_permit: None,
        })
    }

    fn plan(&self, _: PlacementPlanRequest) -> Result<PlacementDecision, crate::RuntimeError> {
        unreachable!("direct-call test uses whole-work selection")
    }

    fn observe(&self, feedback: PlacementFeedback) -> Result<(), crate::RuntimeError> {
        assert_eq!(feedback.kind, ExecutionCandidateKind::SharedRuntime);
        self.observations.set(self.observations.get() + 1);
        Ok(())
    }

    fn invalidate(&self, _: PlacementInvalidation) {}
}

fn strcmp_binding() -> RuntimeBuiltinBinding {
    crate::builtins::wasm_registry::register_all();
    crate::builtin::runtime_builtin_bindings_by_name("strcmp")[0]
}

fn harness(binding: RuntimeBuiltinBinding) -> (RuntimeContext, Rc<Scoped>, Rc<Placement>) {
    let scoped = Rc::new(Scoped {
        binding,
        generation: Cell::new(1),
    });
    let placement = Rc::new(Placement {
        revision_available: Cell::new(true),
        ..Placement::default()
    });
    let ports = RuntimeServicePorts::default()
        .with_builtin(scoped.clone())
        .with_placement(placement.clone());
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(ports);
    (runtime, scoped, placement)
}

fn run(
    runtime: &RuntimeContext,
    left: Value,
    right: Value,
    outputs: usize,
) -> Result<crate::execution::RootedValueSequence, crate::RuntimeError> {
    futures::executor::block_on(crate::call_builtin_sequence_async_with_outputs_in_context(
        runtime,
        "strcmp",
        &[left, right],
        outputs,
    ))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn scoped_scalar_strcmp_reports_one_complete_host_call_only() {
    let (runtime, _scoped, placement) = harness(strcmp_binding());
    let result = run(
        &runtime,
        Value::String("same".into()),
        Value::String("same".into()),
        1,
    )
    .expect("scalar strcmp");
    assert_eq!(result.as_slice(), &[Value::Bool(true)]);
    assert_eq!(placement.selections.get(), 1);
    assert_eq!(placement.observations.get(), 1);
    assert_eq!(placement.probe_drops.load(Ordering::SeqCst), 1);

    let context_free =
        futures::executor::block_on(crate::call_builtin_sequence_async_with_outputs(
            "strcmp",
            &[Value::String("a".into()), Value::String("a".into())],
            1,
        ))
        .expect("context-free strcmp still executes");
    assert_eq!(context_free.as_slice(), &[Value::Bool(true)]);
    assert!(run(
        &runtime,
        Value::String("a".into()),
        Value::String("b".into()),
        2
    )
    .is_err());
    let numeric = run(&runtime, Value::Num(1.0), Value::String("a".into()), 1)
        .expect("numeric strcmp returns false");
    assert_eq!(numeric.as_slice(), &[Value::Bool(false)]);
    assert_eq!(placement.selections.get(), 1);
    assert_eq!(placement.observations.get(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn process_global_selected_binding_uses_its_registry_witness() {
    let _binding = strcmp_binding();
    let placement = Rc::new(Placement {
        revision_available: Cell::new(true),
        ..Placement::default()
    });
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_placement(placement.clone()));
    let result = run(
        &runtime,
        Value::String("global".into()),
        Value::String("global".into()),
        1,
    )
    .expect("global compiled strcmp");
    assert_eq!(result.as_slice(), &[Value::Bool(true)]);
    assert_eq!(placement.selections.get(), 1);
    assert_eq!(placement.observations.get(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn admitted_workload_and_output_mode_separate_feedback_keys() {
    let (runtime, _scoped, placement) = harness(strcmp_binding());
    let first = [Value::String("a".into()), Value::String("a".into())];
    futures::executor::block_on(crate::call_builtin_sequence_async_in_context(
        &runtime, "strcmp", &first,
    ))
    .expect("default output mode");
    run(
        &runtime,
        Value::String("a".into()),
        Value::String("a".into()),
        1,
    )
    .expect("explicit output mode");
    run(
        &runtime,
        Value::String("ab".into()),
        Value::String("ab".into()),
        1,
    )
    .expect("explicit output mode and changed input lengths");
    let signatures = placement.signatures.borrow();
    assert_eq!(signatures.len(), 3);
    assert_ne!(signatures[0], signatures[1]);
    assert_ne!(signatures[1], signatures[2]);
    assert_eq!(placement.observations.get(), 3);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn absent_or_changed_registration_revision_declines_only_feedback() {
    let (runtime, scoped, placement) = harness(strcmp_binding());
    placement.revision_available.set(false);
    assert!(run(
        &runtime,
        Value::String("a".into()),
        Value::String("a".into()),
        1
    )
    .is_ok());
    assert_eq!(placement.selections.get(), 0);
    placement.revision_available.set(true);
    let changed = scoped.clone();
    *placement.on_select.borrow_mut() = Some(Box::new(move || {
        changed.generation.set(changed.generation.get() + 1);
    }));
    assert!(run(
        &runtime,
        Value::String("a".into()),
        Value::String("a".into()),
        1
    )
    .is_ok());
    assert_eq!(placement.selections.get(), 1);
    assert_eq!(placement.observations.get(), 0);
    assert_eq!(placement.probe_drops.load(Ordering::SeqCst), 1);
}

/// A contract fixture, not another builtin route: it isolates the generic
/// admission check from strcmp's own fixed-output descriptor.
#[derive(serde::Serialize)]
struct ReadyDirectTwoOutputFacts {
    requested_outputs: Option<usize>,
}

impl HostCallDomainFacts for ReadyDirectTwoOutputFacts {
    const ROUTE_ID: &'static [u8] = b"runmat.test.host-ready-direct-two-output.v1";

    fn requested_outputs(&self) -> Option<usize> {
        self.requested_outputs
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn generic_admission_preserves_domain_owned_multi_output_mode() {
    let binding = strcmp_binding();
    let (runtime, scoped, placement) = harness(binding);
    let observation = CallObservation::capture(Some(&runtime)).expect("placement context");
    let authority = RegistrationAuthority::Scoped {
        generation: scoped
            .binding_registration_revision(binding)
            .expect("exact scoped registration"),
        service: scoped,
    };
    observation
        .sink(binding, authority, Some(2))
        .expect("selected binding has exact authority")
        .admit(ReadyDirectTwoOutputFacts {
            requested_outputs: Some(2),
        });
    observation.finish_completed(PostCallObservationOutcome::CompletedSuccess);
    assert_eq!(placement.selections.get(), 1);
    assert_eq!(placement.observations.get(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn strcmp_two_outputs_rejects_before_admission_without_feedback() {
    let (runtime, _scoped, placement) = harness(strcmp_binding());
    assert!(run(
        &runtime,
        Value::String("a".into()),
        Value::String("a".into()),
        2,
    )
    .is_err());
    assert_eq!(placement.selections.get(), 0);
    assert_eq!(placement.observations.get(), 0);
}

fn pending_after_admission(invocation: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    let sink = invocation.call_observation;
    crate::builtin::admitted_builtin_future(None, "fixture", async move {
        if let Some(sink) = sink {
            sink.admit(
                crate::builtins::strings::core::strcmp::ScalarStringCompareWorkload {
                    left_utf8_bytes: 1,
                    right_utf8_bytes: 1,
                    requested_outputs: Some(1),
                },
            );
        }
        futures::future::pending::<Result<crate::execution::RootedValueSequence, crate::RuntimeError>>().await
    })
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pending_drop_releases_selected_probe_without_failure_sample() {
    let actual = strcmp_binding();
    let pending = RuntimeBuiltinBinding::new(actual.identity, pending_after_admission);
    let (runtime, _scoped, placement) = harness(pending);
    let arguments = [Value::String("a".into()), Value::String("a".into())];
    let unpolled = crate::call_builtin_sequence_async_with_outputs_in_context(
        &runtime, "strcmp", &arguments, 1,
    );
    drop(unpolled);
    assert_eq!(placement.selections.get(), 0);
    let mut future = Box::pin(crate::call_builtin_sequence_async_with_outputs_in_context(
        &runtime, "strcmp", &arguments, 1,
    ));
    let waker = futures::task::noop_waker();
    let mut task = Context::from_waker(&waker);
    assert!(matches!(
        Pin::as_mut(&mut future).poll(&mut task),
        Poll::Pending
    ));
    assert_eq!(placement.selections.get(), 1);
    drop(future);
    assert_eq!(placement.probe_drops.load(Ordering::SeqCst), 1);
    assert_eq!(placement.observations.get(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn optional_and_preselected_outer_boundaries_each_finish_once() {
    let binding = strcmp_binding();
    let (runtime, _scoped, placement) = harness(binding);
    let arguments = [Value::String("a".into()), Value::String("a".into())];
    let optional = futures::executor::block_on(
        crate::dispatcher::try_call_builtin_sequence_async_with_outputs(
            "strcmp",
            &arguments,
            1,
            Some(&runtime),
        ),
    )
    .expect("optional boundary")
    .expect("exact binding exists");
    assert_eq!(optional.as_slice(), &[Value::Bool(true)]);
    let selected =
        futures::executor::block_on(crate::dispatcher::call_selected_builtin_async_with_outputs(
            binding,
            &arguments,
            1,
            Some(runtime),
        ))
        .expect("preselected boundary");
    assert_eq!(selected.as_slice(), &[Value::Bool(true)]);
    assert_eq!(placement.selections.get(), 2);
    assert_eq!(placement.observations.get(), 2);
    assert_eq!(placement.probe_drops.load(Ordering::SeqCst), 2);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cancellation_after_selection_drops_feedback_without_failure() {
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    let binding = strcmp_binding();
    let scoped = Rc::new(Scoped {
        binding,
        generation: Cell::new(1),
    });
    let placement = Rc::new(Placement {
        revision_available: Cell::new(true),
        ..Placement::default()
    });
    let source = CancellationSource::new();
    let cancelling = source.clone();
    *placement.on_select.borrow_mut() = Some(Box::new(move || {
        cancelling.cancel(CancellationReason::User);
    }));
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    )
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_builtin(scoped)
            .with_placement(placement.clone()),
    );
    let _ = run(
        &runtime,
        Value::String("a".into()),
        Value::String("a".into()),
        1,
    );
    assert_eq!(placement.selections.get(), 1);
    assert_eq!(placement.observations.get(), 0);
    assert_eq!(placement.probe_drops.load(Ordering::SeqCst), 1);
}

struct FiniteMemory(crate::context::host_effect::ManagedHostEffectLedger);
impl crate::context::RuntimeParallelService for FiniteMemory {
    fn supports(&self, _: crate::context::ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> crate::context::RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        crate::context::RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..Default::default()
        }
    }

    fn managed_host_effect_ledger(
        &self,
    ) -> Option<crate::context::host_effect::ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn unknown_peak_under_finite_memory_declines_sample_not_execution() {
    let binding = strcmp_binding();
    let (runtime, scoped, placement) = harness(binding);
    let ledger = crate::context::host_effect::ManagedHostEffectLedger::new(64 * 1_024);
    let runtime = runtime.with_service_ports(
        RuntimeServicePorts::default()
            .with_builtin(scoped)
            .with_placement(placement.clone())
            .with_parallel(Rc::new(FiniteMemory(ledger))),
    );
    let result = run(
        &runtime,
        Value::String("a".into()),
        Value::String("a".into()),
        1,
    )
    .expect("host scalar route still executes");
    assert_eq!(result.as_slice(), &[Value::Bool(true)]);
    assert_eq!(placement.selections.get(), 0);
    assert_eq!(placement.observations.get(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn post_call_workspace_failure_is_not_misattributed_to_host_execution() {
    let (runtime, scoped, placement) = harness(strcmp_binding());
    let owner = crate::workspace::BaseWorkspaceOwner::new();
    let service = Rc::new(crate::workspace::BaseWorkspaceService::new(owner.clone()));
    *placement.on_select.borrow_mut() = Some(Box::new(move || {
        owner
            .close()
            .expect("revoke workspace during selected call");
    }));
    let runtime = runtime.with_service_ports(
        RuntimeServicePorts::default()
            .with_builtin(scoped)
            .with_placement(placement.clone())
            .with_workspace(service),
    );
    let error = run(
        &runtime,
        Value::String("a".into()),
        Value::String("a".into()),
        1,
    )
    .expect_err("outer fence must report revoked workspace");
    assert_eq!(error.identifier(), Some("RunMat:Workspace:Closed"));
    assert_eq!(placement.selections.get(), 1);
    assert_eq!(placement.observations.get(), 0);
    assert_eq!(placement.probe_drops.load(Ordering::SeqCst), 1);
}
