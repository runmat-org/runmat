use super::*;
use futures::task::noop_waker;
use runmat_value::{HandleRef, Value};
use std::cell::RefCell;

struct Retired(&'static str, Rc<RefCell<Vec<&'static str>>>);

impl Drop for Retired {
    fn drop(&mut self) {
        self.1.borrow_mut().push(self.0);
    }
}

#[test]
fn pending_child_retires_before_awaiting_root() {
    let owner = ProcessContinuationDriver::new();
    let handle = owner.handle();
    let retired = Rc::new(RefCell::new(Vec::new()));
    let root_retired = Rc::clone(&retired);
    let child_retired = Rc::clone(&retired);
    let mut driven = Box::pin(owner.drive(async move {
        let _root = Retired("root", root_retired);
        handle
            .enqueue(async move {
                let _child = Retired("child", child_retired);
                futures::future::pending::<()>().await;
                unreachable!("suspended child was cancelled")
            })
            .expect("live invocation")
            .await
    }));
    let waker = noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(driven.as_mut().poll(&mut context), Poll::Pending));
    drop(driven);
    assert_eq!(*retired.borrow(), ["child", "root"]);
}

#[test]
fn canceled_ancestor_retires_descendants_deepest_first() {
    let owner = ProcessContinuationDriver::new();
    let handle = owner.handle();
    let retired = Rc::new(RefCell::new(Vec::new()));
    let root_retired = Rc::clone(&retired);
    let child_retired = Rc::clone(&retired);
    let grandchild_retired = Rc::clone(&retired);
    let child_handle = handle.clone();
    let mut driven = Box::pin(owner.drive(async move {
        let _root = Retired("root", root_retired);
        let child = handle.enqueue(async move {
            let _child = Retired("child", child_retired);
            child_handle
                .enqueue(async move {
                    let _grandchild = Retired("grandchild", grandchild_retired);
                    futures::future::pending::<()>().await;
                    unreachable!("suspended grandchild was cancelled")
                })?
                .await
        })?;
        child.await
    }));
    let waker = noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(driven.as_mut().poll(&mut context), Poll::Pending));
    drop(driven);
    assert_eq!(*retired.borrow(), ["grandchild", "child", "root"]);
}

#[test]
fn ready_root_retires_orphaned_child_before_root() {
    let owner = ProcessContinuationDriver::new();
    let handle = owner.handle();
    let retired = Rc::new(RefCell::new(Vec::new()));
    let root = Retired("root", Rc::clone(&retired));
    let mut child_retired = Some(Rc::clone(&retired));
    let mut driven = Box::pin(owner.drive(futures::future::poll_fn(move |_| {
        let _keep_root_until_driver_retirement = &root;
        let child = Retired("child", child_retired.take().expect("first root poll only"));
        let ticket = handle
            .enqueue(async move {
                let _child = child;
                futures::future::pending::<()>().await;
                unreachable!("orphaned child must retire")
            })
            .expect("root may enqueue before completion");
        drop(ticket);
        Poll::Ready(())
    })));
    let waker = noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(
        driven.as_mut().poll(&mut context),
        Poll::Ready(())
    ));
    assert_eq!(*retired.borrow(), ["child", "root"]);
}

#[test]
fn ready_child_retires_orphaned_grandchild_before_reply() {
    for owned in [false, true] {
        let owner = ProcessContinuationDriver::new();
        let handle = owner.handle();
        let retired = Rc::new(RefCell::new(Vec::new()));
        let observed_retirement = Rc::clone(&retired);
        let child = Retired("child", Rc::clone(&retired));
        let mut grandchild_retired = Some(Rc::clone(&retired));
        let child_handle = handle.clone();
        let result = futures::executor::block_on(owner.drive(async move {
            let child = futures::future::poll_fn(move |_| {
                let grandchild = Retired(
                    "grandchild",
                    grandchild_retired.take().expect("first child poll only"),
                );
                let ticket = child_handle
                    .enqueue(async move {
                        let _grandchild = grandchild;
                        futures::future::pending::<()>().await;
                        unreachable!("orphaned grandchild must retire")
                    })
                    .expect("child may enqueue before completion");
                drop(ticket);
                let outputs = RootedValueSequence::new(runmat_value::ValueSequence::empty())
                    .map_err(|error| error.into_runtime_error());
                let captures =
                    RootedValues::new(Vec::new()).map_err(|error| error.into_runtime_error());
                let result =
                    outputs.and_then(|outputs| captures.map(|captures| (outputs, captures)));
                // Keep the child's owned state in its future until the driver retires it.
                let _ = &child;
                Poll::Ready(result)
            });
            let result = if owned {
                handle.enqueue_owned(child)?.await?
            } else {
                handle.enqueue(child)?.await?
            };
            assert_eq!(*observed_retirement.borrow(), ["grandchild", "child"]);
            Ok::<_, RuntimeError>(result)
        }))
        .expect("ready child returns one typed result");
        assert!(result.0.value().is_empty());
        assert_eq!(*retired.borrow(), ["grandchild", "child"]);
    }
}

#[test]
fn completed_child_transfers_one_typed_reply() {
    let owner = ProcessContinuationDriver::new();
    let handle = owner.handle();
    let result = futures::executor::block_on(owner.drive(async move {
        handle
            .enqueue(async {
                let outputs = RootedValueSequence::new(runmat_value::ValueSequence::empty())
                    .map_err(|error| error.into_runtime_error())?;
                let captures =
                    RootedValues::new(Vec::new()).map_err(|error| error.into_runtime_error())?;
                Ok((outputs, captures))
            })?
            .await
    }))
    .expect("child completion");
    assert!(result.0.value().is_empty());
    assert!(result.1.values().is_empty());
}

#[test]
fn ready_sibling_advances_past_independent_pending_child() {
    let owner = ProcessContinuationDriver::new();
    let handle = owner.handle();
    let retired = Rc::new(RefCell::new(Vec::new()));
    let pending_retired = Rc::clone(&retired);
    let completed = futures::executor::block_on(owner.drive(async move {
        let pending = handle.enqueue(async move {
            let _pending = Retired("pending", pending_retired);
            futures::future::pending::<()>().await;
            unreachable!("independent pending child was not released")
        })?;
        let mut yielded = false;
        futures::future::poll_fn(|context| {
            if std::mem::replace(&mut yielded, true) {
                Poll::Ready(())
            } else {
                context.waker().wake_by_ref();
                Poll::Pending
            }
        })
        .await;
        let ready = handle.enqueue(async {
            Ok((
                RootedValueSequence::new(runmat_value::ValueSequence::empty())
                    .map_err(|error| error.into_runtime_error())?,
                RootedValues::new(Vec::new()).map_err(|error| error.into_runtime_error())?,
            ))
        })?;
        let result = ready.await?;
        drop(pending);
        Ok::<_, RuntimeError>(result)
    }))
    .expect("ready sibling makes progress without the pending sibling");
    assert!(completed.0.value().is_empty());
    assert_eq!(*retired.borrow(), ["pending"]);
}

#[test]
fn queued_arguments_are_rooted_before_the_first_child_poll() {
    let initial = runmat_gc::gc_allocate_rooted(Value::String("queued input".into()))
        .expect("allocate managed input");
    let handle = initial.handle();
    let arguments = RootedValues::new(vec![Value::HandleObject(HandleRef {
        class_name: "QueuedInput".into(),
        target: handle,
        valid: true,
    })])
    .expect("register selected arguments before enqueue");
    initial
        .unroot()
        .expect("transfer input root to queued call");
    let owner = ProcessContinuationDriver::new();
    let ticket = owner
        .handle()
        .enqueue(async move {
            let _arguments = arguments;
            unreachable!("this witness does not poll the child")
        })
        .expect("queue owned call");
    runmat_gc::gc_collect_major().expect("collect before first child poll");
    assert_eq!(
        runmat_gc::gc_clone_value(&handle).expect("queued argument owns its member"),
        Value::String("queued input".into())
    );
    drop(ticket);
    drop(owner);
    runmat_gc::gc_collect_major().expect("collect after queued call retirement");
    assert!(runmat_gc::gc_clone_value(&handle).is_err());
}
