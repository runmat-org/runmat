//! Actual VM publication-boundary faults. These stay in the library test
//! target so their request-local interpreter-state hook cannot enter release
//! or integration-test linkage.

use super::*;
use crate::interpreter::state::{
    InterpreterState, VmFusionPublicationFault, VmFusionPublicationFaultRoute,
};
use futures::executor::block_on;
use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
use runmat_accelerate::placement::PlacementSession;
use runmat_execution::cancellation::CancellationSource;
use runmat_runtime::context::{RuntimeContext, RuntimeInvocationArity, RuntimeServicePorts};
use std::cell::{Cell, RefCell};
use std::error::Error;
use std::future::Future;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::task::Poll;

#[path = "../../../../tests/support/typed_reduction_placement.rs"]
mod typed_reduction_placement;
use typed_reduction_placement::CapturingPlacement;

#[path = "publication_lifecycle_tests/cpu_placement.rs"]
mod cpu_placement;
#[path = "publication_lifecycle_tests/cpu_publication_tests.rs"]
mod cpu_publication_tests;
#[path = "publication_lifecycle_tests/elementwise_publication_tests.rs"]
mod elementwise_publication_tests;
#[path = "publication_lifecycle_tests/graph.rs"]
mod graph;
#[path = "publication_lifecycle_tests/managed_intrinsic_tests.rs"]
mod managed_intrinsic_tests;

const FOLDED_MIN: &str =
    "x = gpuArray(reshape(single(1:16384),128,128)); y = min(x+x); z = gather(y);";
const HOST_FOLDED_MIN: &str = "x = reshape(single(1:64),8,8); y = min(x+x); z = gather(y);";
const CAUGHT_FOLDED_MIN: &str = "x = gpuArray(reshape(single(1:16384),128,128)); try; y = min(x+x); catch e; caught = e.identifier; end;";

fn actual_provider_scope() -> runmat_accelerate_api::ThreadProviderGuard {
    let scope = runmat_accelerate_api::ThreadProviderGuard::set(None);
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("actual WGPU adapter required");
    // The provider is shared across witnesses. A cancelled invocation may
    // leave physical completion work after its logical scope has ended.
    // Establish a clean queue boundary before observing this witness's receipts.
    block_on(
        runmat_accelerate_api::AccelProvider::observe_region_queue_tail(
            provider,
            runmat_accelerate_api::AccelProvider::device_id(provider),
            None,
        ),
    )
    .expect("previous provider queue work must retire before the lifecycle witness");
    scope
}

fn compile_source(source: &str) -> crate::bytecode::Bytecode {
    let bytecode = compile_source_raw(source);
    graph::assert_compiled_add_min(&bytecode);
    bytecode
}

fn compile_source_raw(source: &str) -> crate::bytecode::Bytecode {
    compile_source_with_bindings(source, &std::collections::HashMap::<String, usize>::new())
}

fn compile_source_with_bindings(
    source: &str,
    variables: &dyn runmat_hir::WorkspaceBindingView,
) -> crate::bytecode::Bytecode {
    let ast = runmat_parser::parse(source).expect("parse actual folded source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::new(variables)).expect("HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("MIR");
    crate::bytecode::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id)
        .expect("compile actual folded source")
}

fn placement() -> (Rc<PlacementSession>, Rc<CapturingPlacement>) {
    let session = Rc::new(PlacementSession::default());
    let capture = Rc::new(CapturingPlacement {
        session: session.clone(),
        reduction_selection: RefCell::new(Vec::new()),
        delivered: Arc::new(Mutex::new(Vec::new())),
        invalidate_selected_provider: Cell::new(false),
        revoked_before_publication: Cell::new(0),
    });
    (session, capture)
}

fn runtime(
    placement: Rc<CapturingPlacement>,
    cancellation: Option<&CancellationSource>,
) -> RuntimeContext {
    let service = Rc::new(runmat_runtime::execution::RuntimeExecutionService::new());
    let runtime = match cancellation {
        Some(source) => RuntimeContext::with_cancellation(service, source.token()),
        None => RuntimeContext::new(service),
    };
    runtime.with_service_ports(RuntimeServicePorts::default().with_placement(placement))
}

fn state(
    bytecode: crate::bytecode::Bytecode,
    vars: crate::InterpreterVars,
    runtime: RuntimeContext,
    fault: Option<VmFusionPublicationFault>,
) -> Result<Box<InterpreterState>, runmat_runtime::RuntimeError> {
    let supplied = vars.len();
    let vars = vars.prepare_extent(&runtime, bytecode.var_count)?;
    let mut state = InterpreterState::from_vars_in_context(
        bytecode,
        vars,
        supplied,
        Some("<main>"),
        RuntimeInvocationArity::Script,
        runtime,
    )?;
    assert!(
        state.fusion_installation.is_some(),
        "authentic VM installation"
    );
    state.fusion_publication_fault = fault;
    Ok(Box::new(state))
}

/// Instrument the actual owned runner while retaining the same pending-frame
/// completion owner and issuer scope used by the public interpret_workspace.
async fn scoped_faultable_run(
    bytecode: crate::bytecode::Bytecode,
    vars: crate::InterpreterVars,
    runtime: RuntimeContext,
    fault: Option<VmFusionPublicationFault>,
) -> Result<crate::interpreter::state::InterpreterOutcome, runmat_runtime::RuntimeError> {
    let runtime = runtime.with_invocation_arity(RuntimeInvocationArity::Script);
    let (runtime, workspace_activation) =
        crate::runtime::workspace_service::VmWorkspaceService::install(runtime, None);
    let runtime = super::super::with_default_acceleration_service(runtime);
    let runtime = super::super::runtime_with_bytecode_debug_frame(runtime, &bytecode, "<main>");
    let selected_runtime = runtime.clone();
    let frame = workspace_activation.frame();
    let execution =
        workspace_activation.scope(runmat_runtime::data::with_tx_registry_scope(async move {
            crate::bytecode::validate_sequence_register_flow(&bytecode.instructions)
                .map_err(runmat_runtime::RuntimeError::new)?;
            let _inputs = crate::runtime::gc::ManagedRoots::from_traced(&vars)
                .map_err(runmat_runtime::RuntimeError::new)?;
            let state = state(bytecode, vars, selected_runtime, fault)?;
            super::super::run_interpreter_owned(state).await
        }));
    runtime
        .scope(super::super::workspace_execution::PendingWorkspaceExecution::new(execution, frame))
        .await
        .outcome
}

fn assert_no_reduction_receipt(capture: &CapturingPlacement) {
    let selections = capture.reduction_selection.borrow();
    let [selected] = selections.as_slice() else {
        panic!("one selected physical provider reduction required");
    };
    assert_eq!(selected.request.signature, selected.signature);
    assert_eq!(
        selected.kind,
        runmat_execution::ExecutionCandidateKind::ProviderFusion
    );
    let offered = selected
        .request
        .graph
        .nodes
        .iter()
        .flat_map(|node| &node.candidates)
        .find(|candidate| candidate.descriptor.identity == selected.candidate)
        .expect("selected provider candidate belongs to the exact request");
    assert_eq!(offered.descriptor.kind, selected.kind);
    assert_eq!(offered.descriptor.preparation, selected.preparation);
    assert_eq!(
        selected.signature.provider_provenance,
        Some(
            typed_reduction_placement::feedback::selected_provider_capability_digest(
                selected.device_id
            )
        )
    );
    assert!(matches!(
        selected.signature.revision.authority,
        runmat_execution::PlacementExecutionAuthorityRevision::RuntimeBindingSession(_)
    ));
    assert!(
        capture
            .delivered
            .lock()
            .unwrap()
            .iter()
            .all(|feedback| { feedback.signature.operation != "fusion.reduction" }),
        "no failed/cancelled/dropped VM publication may deposit a reduction receipt"
    );
}

#[test]
#[ignore = "requires an actual WGPU adapter; unavailable hardware fails when explicitly run"]
fn real_store_rejection_and_caught_rejection_never_publish_reduction_feedback() {
    runmat_gc::gc_test_context(|| {
        let _provider_scope = actual_provider_scope();
        for (source, caught) in [(FOLDED_MIN, false), (CAUGHT_FOLDED_MIN, true)] {
            let (_session, capture) = placement();
            let runtime = runtime(capture.clone(), None);
            let bytecode = compile_source(source);
            let caught_slot = caught.then(|| {
                bytecode
                    .var_names
                    .iter()
                    .find_map(|(slot, name)| (name == "caught").then_some(*slot))
                    .expect("compiled catch identifier variable")
            });
            let vars = vec![Value::Num(0.0); bytecode.var_count];
            let reached = Rc::new(Cell::new(false));
            let result = block_on(scoped_faultable_run(
                bytecode,
                crate::InterpreterVars::from_owned(vars),
                runtime.clone(),
                Some(VmFusionPublicationFault::Reject(
                    VmFusionPublicationFaultRoute::AnySelected,
                    reached.clone(),
                )),
            ));
            assert!(
                reached.get(),
                "fault reached the checked ordinary StoreVar; result={result:?}; selected_reductions={}",
                capture.reduction_selection.borrow().len()
            );
            if caught {
                let crate::InterpreterOutcome::Completed(completion) =
                    result.expect("ordinary catch handles the real store failure");
                let vars = &completion.values;
                assert_eq!(
                    *vars
                        .get(caught_slot.expect("caught branch has slot"))
                        .unwrap(),
                    Value::CharArray(runmat_value::CharArray::new_row(
                        "RunMat:TransientSequenceNotStorable"
                    )),
                    "catch must observe the checked StoreVar rejection"
                );
            } else {
                let error = match result {
                    Ok(_) => panic!("ordinary StoreVar must reject a transient sequence"),
                    Err(error) => error,
                };
                assert_eq!(
                    error.identifier(),
                    Some("RunMat:TransientSequenceNotStorable")
                );
            }
            block_on(runtime.scope(async {}));
            assert_no_reduction_receipt(capture.as_ref());
        }
    });
}

#[test]
#[ignore = "requires an actual WGPU adapter; unavailable hardware fails when explicitly run"]
fn cancellation_and_drop_at_checked_store_do_not_publish_reduction_feedback() {
    runmat_gc::gc_test_context(|| {
        let _provider_scope = actual_provider_scope();
        for cancel in [true, false] {
            let (_session, capture) = placement();
            let cancellation = CancellationSource::new();
            let runtime = runtime(capture.clone(), cancel.then_some(&cancellation));
            let bytecode = compile_source(FOLDED_MIN);
            let vars = vec![Value::Num(0.0); bytecode.var_count];
            let reached = Rc::new(Cell::new(false));
            let fault = if cancel {
                VmFusionPublicationFault::Cancel(
                    VmFusionPublicationFaultRoute::AnySelected,
                    cancellation.clone(),
                    reached.clone(),
                )
            } else {
                VmFusionPublicationFault::Suspend(
                    VmFusionPublicationFaultRoute::AnySelected,
                    reached.clone(),
                )
            };
            if cancel {
                let error = match block_on(scoped_faultable_run(
                    bytecode,
                    crate::InterpreterVars::from_owned(vars),
                    runtime.clone(),
                    Some(fault),
                )) {
                    Ok(_) => panic!("cancellation terminates the selected call"),
                    Err(error) => error,
                };
                assert_eq!(error.identifier(), Some("RunMat:ExecutionCancelled"));
                assert!(matches!(
                    error
                        .source()
                        .and_then(|source| source
                            .downcast_ref::<runmat_runtime::execution::ExecutionServiceError>()),
                    Some(runmat_runtime::execution::ExecutionServiceError::Cancelled)
                ));
                assert!(cancellation.token().is_cancelled());
            } else {
                let mut pending = Box::pin(scoped_faultable_run(
                    bytecode,
                    crate::InterpreterVars::from_owned(vars),
                    runtime.clone(),
                    Some(fault),
                ));
                block_on(std::future::poll_fn(|context| {
                    assert!(matches!(pending.as_mut().poll(context), Poll::Pending));
                    if reached.get() {
                        Poll::Ready(())
                    } else {
                        Poll::Pending
                    }
                }));
                drop(pending);
            }
            assert!(
                reached.get(),
                "the actual checked publication PC was reached"
            );
            block_on(runtime.scope(async {}));
            assert_no_reduction_receipt(capture.as_ref());
        }
    });
}

#[test]
#[ignore = "requires an actual WGPU adapter; unavailable hardware fails when explicitly run"]
fn unpolled_vm_future_never_selects_or_publishes_reduction() {
    runmat_gc::gc_test_context(|| {
        let _provider_scope = actual_provider_scope();
        let (_session, capture) = placement();
        let runtime = runtime(capture.clone(), None);
        let bytecode = compile_source(FOLDED_MIN);
        let vars = vec![Value::Num(0.0); bytecode.var_count];
        let unpolled = scoped_faultable_run(
            bytecode,
            crate::InterpreterVars::from_owned(vars),
            runtime.clone(),
            None,
        );
        drop(unpolled);
        block_on(runtime.scope(async {}));
        assert!(capture.reduction_selection.borrow().is_empty());
        assert!(capture
            .delivered
            .lock()
            .unwrap()
            .iter()
            .all(|feedback| { feedback.signature.operation != "fusion.reduction" }));
    });
}
