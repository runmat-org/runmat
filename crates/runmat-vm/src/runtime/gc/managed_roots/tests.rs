use super::*;
use std::future::Future;
use std::task::{Context, Poll};

#[test]
fn host_values_need_no_managed_root() {
    let values = [Value::Tensor(
        runmat_value::Tensor::new(vec![0.0; 1024], vec![32, 32]).unwrap(),
    )];
    assert!(ManagedRoots::new(&values).unwrap().0.is_none());
}

#[test]
fn suspended_managed_arguments_survive_gc_until_drop_cancel_or_error() {
    enum Terminal {
        Drop,
        Cancel,
        Error,
    }
    for terminal in [Terminal::Drop, Terminal::Cancel, Terminal::Error] {
        runmat_gc::gc_test_context(|| {
            let target = runmat_gc::gc_allocate_rooted(Value::Num(7.0)).unwrap();
            let identity = target.handle();
            let argument = Value::HandleObject(runmat_value::HandleRef {
                class_name: "ManagedArgument".into(),
                target: identity,
                valid: true,
            });
            let (sender, receiver) = futures::channel::oneshot::channel::<Result<(), ()>>();
            let body = Box::pin(async move {
                let _roots = ManagedRoots::new(std::slice::from_ref(&argument)).unwrap();
                drop(argument);
                receiver.await.unwrap()
            });
            let cancellation = runmat_execution::cancellation::CancellationSource::new();
            let mut pending = Box::pin(futures::future::select(
                body,
                Box::pin(cancellation.token().cancelled()),
            ));
            let mut context = Context::from_waker(futures::task::noop_waker_ref());
            assert!(pending.as_mut().poll(&mut context).is_pending());
            target.unroot().unwrap();
            runmat_gc::gc_collect_major().unwrap();
            assert!(runmat_gc::gc_with_value(&identity, |_| ()).is_ok());
            match terminal {
                Terminal::Cancel => {
                    cancellation.cancel(runmat_execution::CancellationReason::User);
                    let Poll::Ready(futures::future::Either::Right((_, body))) =
                        pending.as_mut().poll(&mut context)
                    else {
                        panic!("cancellation must complete")
                    };
                    drop(body);
                }
                Terminal::Error => {
                    sender.send(Err(())).unwrap();
                    let Poll::Ready(futures::future::Either::Left((result, listener))) =
                        pending.as_mut().poll(&mut context)
                    else {
                        panic!("error must complete")
                    };
                    assert!(result.is_err());
                    drop(listener);
                }
                Terminal::Drop => {}
            }
            drop(pending);
            runmat_gc::gc_collect_major().unwrap();
            assert!(runmat_gc::gc_with_value(&identity, |_| ()).is_err());
        });
    }
}
