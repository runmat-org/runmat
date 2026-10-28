use super::*;
use futures::executor::block_on;
use runmat_value::{Value, ValueSequence};
use std::{
    cell::RefCell,
    pin::Pin,
    rc::Rc,
    task::{Context, Poll},
};

fn arguments(runtime: Option<&RuntimeContext>) -> RootedFunctionArguments {
    let values = match runtime {
        Some(runtime) => {
            crate::execution::RootedValues::in_context(vec![Value::Num(3.)], runtime).unwrap()
        }
        None => crate::execution::RootedValues::new(vec![Value::Num(3.)]).unwrap(),
    };
    RootedFunctionArguments::ordinary(values)
}
fn returned() -> BuiltinResult<RootedValueSequence> {
    RootedValueSequence::new(ValueSequence::single(Value::Num(9.)).unwrap())
        .map_err(|error| error.into_runtime_error())
}

#[test]
fn one_auxiliary_capability_remains_live_for_all_later_user_event_kinds_then_retires() {
    block_on(async {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let token = runtime.cancellation();
        let mut scope = Scope::new(Some(&runtime), Some(&token)).unwrap();
        let retained = scope.handle();
        let retained_header = crate::execution::RetainedValueReadLease::prepare_callable_rooted(
            crate::execution::RootedValue::new(Value::ScopedFunctionHandle(retained.clone()))
                .unwrap(),
            Some(&runtime),
        )
        .expect("a parked live scope can prepare its child callable header");
        assert_eq!(
            validate_target(Some(&runtime), &retained)
                .unwrap_err()
                .identifier(),
            Some("RunMat:ScopedCallable:Unavailable"),
            "retention does not grant invocation or cached-result access"
        );
        drop(retained_header);
        let mut calls = 0;
        for _event in ["creation", "iteration", "output", "plot", "hybrid"] {
            assert!(retained.is_live());
            let error = invoke(
                Some(&runtime),
                retained.clone(),
                arguments(Some(&runtime)),
                1,
                None,
            )
            .err()
            .expect("parked handler cannot be invoked");
            assert_eq!(
                error.identifier(),
                Some("RunMat:ScopedCallable:Unavailable")
            );
            let target = retained.clone();
            let result = scope
                .run(
                    |current, original| {
                        assert!(current.same_capability(&target));
                        Ok(async move {
                            invoke(
                                original.as_ref(),
                                target,
                                arguments(original.as_ref()),
                                1,
                                None,
                            )?
                            .await
                        })
                    },
                    async |request| {
                        assert_eq!(request.arguments.view().get(0), Some(&Value::Num(3.)));
                        calls += 1;
                        Ok(ControlFlow::<(), _>::Continue(returned()?))
                    },
                )
                .await
                .unwrap();
            let ControlFlow::Continue(result) = result else {
                panic!("completed real event")
            };
            assert_eq!(result.as_slice(), &[Value::Num(9.)]);
        }
        assert_eq!(calls, 5);
        drop(scope);
        assert!(!retained.is_live());
        let error = invoke(Some(&runtime), retained, arguments(Some(&runtime)), 1, None)
            .err()
            .expect("auxiliary has retired");
        assert_eq!(error.identifier(), Some("RunMat:ScopedCallable:Expired"));
    });
}

struct ReadyChild {
    runtime: Option<RuntimeContext>,
    handle: ScopedFunctionHandle,
    observed: Rc<RefCell<Option<String>>>,
    pending: bool,
}
impl Future for ReadyChild {
    type Output = BuiltinResult<RootedValueSequence>;
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        if self.pending {
            Poll::Pending
        } else {
            Poll::Ready(returned())
        }
    }
}
impl Drop for ReadyChild {
    fn drop(&mut self) {
        let error = validate_target(self.runtime.as_ref(), &self.handle)
            .expect_err("child destruction must not enter the parked handler");
        *self.observed.borrow_mut() = error.identifier().map(str::to_owned);
    }
}
#[test]
fn auxiliary_event_parks_before_child_destructor_but_does_not_expire_whole_scope() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let token = runtime.cancellation();
    for pending in [false, true] {
        let mut scope = Scope::new(Some(&runtime), Some(&token)).unwrap();
        let retained = scope.handle();
        let observed = Rc::new(RefCell::new(None));
        let probe = Rc::clone(&observed);
        let mut work = Box::pin(scope.run(
            |handle, original| {
                Ok(ReadyChild {
                    runtime: original,
                    handle,
                    observed: probe,
                    pending,
                })
            },
            async |_| -> BuiltinResult<ControlFlow<(), RootedValueSequence>> {
                panic!("destructor control makes no numerical request")
            },
        ));
        if pending {
            let waker = futures::task::noop_waker();
            let mut context = Context::from_waker(&waker);
            assert!(work.as_mut().poll(&mut context).is_pending());
            drop(work);
        } else {
            let result = block_on(work).unwrap();
            assert!(matches!(result, ControlFlow::Continue(_)));
        }
        assert_eq!(
            observed.borrow().as_deref(),
            Some("RunMat:ScopedCallable:Unavailable")
        );
        assert!(retained.is_live());
        drop(scope);
        assert!(!retained.is_live());
    }
}
