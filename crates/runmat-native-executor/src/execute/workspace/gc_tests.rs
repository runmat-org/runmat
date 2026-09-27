use super::*;
use futures::FutureExt;
use runmat_native_codegen::{NativeBlockId, NativeFunction, NativeFunctionAbi};
use runmat_types::{ProgramFunctionId, ProgramSourceId};

fn workspace(values: BTreeMap<String, Value>) -> NativeWorkspaceState {
    let function = NativeFunction {
        id: ProgramFunctionId(0),
        source: ProgramSourceId(0),
        name: "gc_workspace".into(),
        class_method_owner: None,
        capabilities: Default::default(),
        abi: NativeFunctionAbi {
            fixed_inputs: vec![],
            varargin: None,
            fixed_outputs: vec![],
            varargout: None,
            implicit_nargin: None,
            implicit_nargout: None,
        },
        argument_validations: vec![],
        locals: vec![],
        entry: NativeBlockId(0),
        blocks: vec![],
        expected_sites: vec![],
    };
    NativeWorkspaceState::new(
        &function,
        NativeWorkspaceInput {
            values,
            ..Default::default()
        },
    )
    .unwrap()
}

fn handle_value(handle: runmat_gc::GcHandle) -> Value {
    Value::HandleObject(runmat_value::HandleRef {
        class_name: "WorkspaceGcFixture".into(),
        target: handle,
        valid: true,
    })
}

#[test]
fn assigned_unmapped_value_survives_suspension_and_service_outlives_state() {
    runmat_gc::gc_test_context(|| {
        let state = workspace(BTreeMap::new());
        let service = state.service();
        let allocation =
            runmat_gc::gc_allocate_rooted(Value::String("pending workspace".into())).unwrap();
        let handle = allocation.handle();
        let assigned = handle_value(handle);
        let expected = assigned.clone();
        let pending_service = Rc::clone(&service);
        let mut operation = Box::pin(async move {
            pending_service.assign("unmapped", assigned).unwrap();
            allocation.unroot().unwrap();
            let mut yielded = false;
            futures::future::poll_fn(|cx| {
                if yielded {
                    std::task::Poll::Ready(())
                } else {
                    yielded = true;
                    cx.waker().wake_by_ref();
                    std::task::Poll::Pending
                }
            })
            .await;
            pending_service.lookup("unmapped").unwrap()
        });
        assert!(operation.as_mut().now_or_never().is_none());
        runmat_gc::gc_collect_major().unwrap();
        assert_eq!(
            runmat_gc::gc_clone_value(&handle).unwrap(),
            Value::String("pending workspace".into())
        );
        assert_eq!(futures::executor::block_on(operation), expected);
        drop(state);
        runmat_gc::gc_collect_major().unwrap();
        assert!(
            runmat_gc::gc_clone_value(&handle).is_ok(),
            "service retains storage and roots"
        );
        drop(service);
        runmat_gc::gc_collect_major().unwrap();
        assert!(
            runmat_gc::gc_clone_value(&handle).is_err(),
            "last service drop releases roots"
        );
    });
}

#[test]
fn input_snapshot_aliases_remove_clear_and_replacement_have_exact_root_lifetimes() {
    runmat_gc::gc_test_context(|| {
        let allocation = runmat_gc::gc_allocate_rooted(Value::String("aliased".into())).unwrap();
        let handle = allocation.handle();
        let nested =
            Value::Cell(runmat_value::CellArray::new(vec![handle_value(handle)], 1, 1).unwrap());
        let state = workspace(BTreeMap::from([
            ("a".into(), nested.clone()),
            ("b".into(), nested),
        ]));
        let service = state.service();
        allocation.unroot().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
        service.remove("a").unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(
            runmat_gc::gc_clone_value(&handle).is_ok(),
            "independent alias root survives removal"
        );
        service.assign("b", Value::Num(0.0)).unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(
            runmat_gc::gc_clone_value(&handle).is_err(),
            "replacement releases last root"
        );

        let allocation = runmat_gc::gc_allocate_rooted(Value::Num(17.0)).unwrap();
        let handle = allocation.handle();
        service.assign("clear_me", handle_value(handle)).unwrap();
        allocation.unroot().unwrap();
        service.clear().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}

#[test]
fn reentrant_mutation_is_structured_and_gc_never_borrows_workspace_storage() {
    runmat_gc::gc_test_context(|| {
        let allocation = runmat_gc::gc_allocate_rooted(Value::Num(19.0)).unwrap();
        let handle = allocation.handle();
        let state = workspace(BTreeMap::from([("retained".into(), handle_value(handle))]));
        let service = state.service();
        allocation.unroot().unwrap();
        let borrow = state.values.borrow_mut();
        runmat_gc::gc_collect_major().unwrap();
        assert!(
            runmat_gc::gc_clone_value(&handle).is_ok(),
            "explicit roots do not scan the RefCell"
        );
        for result in [
            service.assign("retained", Value::Num(0.0)),
            service.remove("retained"),
            service.clear(),
        ] {
            assert_eq!(
                result.unwrap_err().identifier(),
                Some("RunMat:NativeWorkspaceBorrow")
            );
        }
        assert!(
            !state.needs_reconciliation(),
            "failed mutation leaves both storage and dirty status intact"
        );
        drop(borrow);
        assert_eq!(service.lookup("retained"), Some(handle_value(handle)));
        service.clear().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}
