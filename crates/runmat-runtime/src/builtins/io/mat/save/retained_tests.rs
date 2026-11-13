use super::*;
use crate::context::{RuntimeContext, RuntimeServicePorts};
use crate::execution::RuntimeExecutionService;
use crate::workspace::{BaseWorkspaceOwner, BaseWorkspaceService, WorkspaceDelta};
use futures::task::noop_waker_ref;
use runmat_value::{CellArray, HandleRef};
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

fn runtime_with_workspace() -> (RuntimeContext, BaseWorkspaceOwner) {
    let owner = BaseWorkspaceOwner::new();
    let service = Rc::new(BaseWorkspaceService::new(owner.clone()));
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        RuntimeServicePorts::default()
            .with_workspace(service.clone())
            .with_base_workspace(service),
    );
    (runtime, owner)
}

fn stored_handle(
    owner: &BaseWorkspaceOwner,
    name: &str,
) -> (runmat_gc::ExplicitRoot, runmat_gc::GcHandle) {
    let initial = runmat_gc::gc_allocate_rooted(Value::String(name.into())).unwrap();
    let handle = initial.handle();
    let nested = Value::Cell(
        CellArray::new(
            vec![Value::HandleObject(HandleRef {
                class_name: "SaveRetentionHandle".into(),
                target: handle,
                valid: true,
            })],
            1,
            1,
        )
        .unwrap(),
    );
    let mut delta = WorkspaceDelta::new();
    delta.assign(name, nested).unwrap();
    owner.apply(delta).unwrap();
    (initial, handle)
}

fn pending_selection(
    runtime: RuntimeContext,
    name: &'static str,
    snapshot: bool,
) -> Pin<Box<dyn Future<Output = ()>>> {
    Box::pin(async move {
        let mut reads = WorkspaceReadSet::capture(Some(&runtime)).unwrap();
        if snapshot {
            let mut cooperation =
                MatCodecCooperation::new(Some(&runtime), None, BUILTIN_NAME, &SAVE_ERROR_IO);
            let values = collect_workspace_entries(&mut reads, &mut cooperation)
                .await
                .unwrap();
            assert!(find_in_entries(&values, name).is_some());
        } else {
            lookup_workspace(&mut reads, name).await.unwrap();
        }
        futures::future::pending::<()>().await;
    })
}

fn poll_to_pending(future: &mut Pin<Box<dyn Future<Output = ()>>>) {
    let mut context = Context::from_waker(noop_waker_ref());
    assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
}

#[test]
fn named_selection_roots_nested_handles_across_pending_remove_until_drop() {
    runmat_gc::gc_test_context(|| {
        let (runtime, owner) = runtime_with_workspace();
        let (initial, handle) = stored_handle(&owner, "held");
        initial.unroot().unwrap();
        let mut future = pending_selection(runtime, "held", false);
        poll_to_pending(&mut future);

        let mut delta = WorkspaceDelta::new();
        delta.remove("held");
        owner.apply(delta).unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());

        drop(future);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}

#[test]
fn snapshot_selection_roots_nested_handles_across_pending_close_until_drop() {
    runmat_gc::gc_test_context(|| {
        let (runtime, owner) = runtime_with_workspace();
        let (initial, handle) = stored_handle(&owner, "held");
        initial.unroot().unwrap();
        let mut future = pending_selection(runtime, "held", true);
        poll_to_pending(&mut future);

        owner.close().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());

        drop(future);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}

#[test]
fn codec_checkpoint_rejects_a_closed_captured_workspace_after_pending_without_rebinding() {
    let (runtime, owner) = runtime_with_workspace();
    let reads = WorkspaceReadSet::capture(Some(&runtime)).unwrap();
    let mut cooperation =
        MatCodecCooperation::new(Some(&runtime), None, BUILTIN_NAME, &SAVE_ERROR_IO)
            .with_workspace_liveness(reads.liveness_service());
    let (sender, receiver) = futures::channel::oneshot::channel::<()>();
    let mut effect = Box::pin(cooperation.await_io(async move {
        receiver.await.unwrap();
        Ok::<(), ()>(())
    }));
    let mut context = Context::from_waker(noop_waker_ref());
    assert!(matches!(effect.as_mut().poll(&mut context), Poll::Pending));

    owner.close().unwrap();
    sender.send(()).unwrap();

    let Poll::Ready(Err(error)) = effect.as_mut().poll(&mut context) else {
        panic!("closed workspace must reject the post-pending effect checkpoint")
    };
    assert_eq!(error.identifier(), Some("RunMat:Workspace:Closed"));
    assert_eq!(error.context.builtin.as_deref(), Some(BUILTIN_NAME));
}
