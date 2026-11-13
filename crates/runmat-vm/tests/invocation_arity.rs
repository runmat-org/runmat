#[path = "support/mod.rs"]
mod support;

use futures::channel::oneshot;
use futures::future::{select, Either};
use runmat_runtime::context::RuntimeContext;
use runmat_value::Value;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::Context;

#[path = "invocation_arity/gated_sin.rs"]
mod gated_sin;
use gated_sin::runtime_with_gate;

fn expect_handler_reached<F: Future>(future: Pin<&mut F>, reached: oneshot::Receiver<()>) {
    match futures::executor::block_on(select(reached, future)) {
        Either::Left((Ok(()), _still_pending)) => {}
        Either::Left((Err(_), _)) => panic!("handler dropped before reporting reach"),
        Either::Right(_) => panic!("call completed before reaching gated handler"),
    }
}

fn output(
    bytecode: &runmat_vm::Bytecode,
    values: &runmat_vm::InterpreterVars,
    name: &str,
) -> Value {
    let slot = bytecode
        .var_names
        .iter()
        .find_map(|(slot, candidate)| (candidate == name).then_some(*slot))
        .expect("compiled output slot");
    values
        .get(slot)
        .expect("completed output remains in its compiled slot")
        .clone()
}

#[test]
fn overlapping_function_futures_use_their_own_arity_after_the_other_suspends() {
    runmat_gc::gc_test_context(|| {
        let first_bytecode = support::compile_source(
            "result = probe(7);\nfunction [a,b] = probe(varargin)\n  sin(1);\n  a = nargin;\n  b = nargout;\nend\n",
        )
        .expect("first function program");
        let second_bytecode = support::compile_source(
            "[first,second] = probe(1,2,3);\nfunction [a,b] = probe(varargin)\n  sin(1);\n  a = nargin;\n  b = nargout;\nend\n",
        )
        .expect("second function program");
        let base = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let (first_runtime, first_gate, first_reached) = runtime_with_gate(base.clone());
        let (second_runtime, second_gate, second_reached) = runtime_with_gate(base);
        let first_vars = Vec::new();
        let second_vars = Vec::new();
        let mut first = Box::pin(runmat_vm::interpret_workspace(
            &first_bytecode,
            runmat_vm::InterpreterVars::from_owned(first_vars),
            Some("<main>"),
            first_runtime,
            None,
            None,
        ));
        let mut second = Box::pin(runmat_vm::interpret_workspace(
            &second_bytecode,
            runmat_vm::InterpreterVars::from_owned(second_vars),
            Some("<main>"),
            second_runtime,
            None,
            None,
        ));
        expect_handler_reached(first.as_mut(), first_reached);
        expect_handler_reached(second.as_mut(), second_reached);

        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        assert!(first.as_mut().poll(&mut context).is_pending());
        assert!(second.as_mut().poll(&mut context).is_pending());

        first_gate
            .send(())
            .expect("first invocation remains pending");
        let Ok(runmat_vm::InterpreterOutcome::Completed(first_result)) =
            futures::executor::block_on(first.as_mut()).outcome
        else {
            panic!("first function did not complete after its own gate");
        };
        assert_eq!(
            output(&first_bytecode, &first_result.values, "result"),
            Value::Num(1.0)
        );

        second_gate
            .send(())
            .expect("second invocation remains pending");
        let Ok(runmat_vm::InterpreterOutcome::Completed(second_result)) =
            futures::executor::block_on(second.as_mut()).outcome
        else {
            panic!("second function did not complete after its own gate");
        };
        assert_eq!(
            output(&second_bytecode, &second_result.values, "first"),
            Value::Num(3.0)
        );
        assert_eq!(
            output(&second_bytecode, &second_result.values, "second"),
            Value::Num(2.0)
        );
    });
}

#[test]
fn dropping_a_pending_function_does_not_change_sibling_function_arity() {
    runmat_gc::gc_test_context(|| {
        let bytecode = support::compile_source(
            "[first,second] = probe(1,2,3);\nfunction [a,b] = probe(varargin)\n  sin(1);\n  a = nargin;\n  b = nargout;\nend\n",
        )
        .expect("function program");
        let base = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let (abandoned_runtime, abandoned_gate, abandoned_reached) =
            runtime_with_gate(base.clone());
        let (live_runtime, live_gate, live_reached) = runtime_with_gate(base);
        let abandoned_vars = Vec::new();
        let live_vars = Vec::new();
        let mut abandoned = Box::pin(runmat_vm::interpret_workspace(
            &bytecode,
            runmat_vm::InterpreterVars::from_owned(abandoned_vars),
            Some("<main>"),
            abandoned_runtime,
            None,
            None,
        ));
        let mut live = Box::pin(runmat_vm::interpret_workspace(
            &bytecode,
            runmat_vm::InterpreterVars::from_owned(live_vars),
            Some("<main>"),
            live_runtime,
            None,
            None,
        ));
        expect_handler_reached(abandoned.as_mut(), abandoned_reached);
        expect_handler_reached(live.as_mut(), live_reached);
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        assert!(abandoned.as_mut().poll(&mut context).is_pending());
        assert!(live.as_mut().poll(&mut context).is_pending());
        drop(abandoned);
        assert!(
            abandoned_gate.send(()).is_err(),
            "dropped call retires its gate"
        );
        live_gate.send(()).expect("sibling still owns its gate");
        let Ok(runmat_vm::InterpreterOutcome::Completed(result)) =
            futures::executor::block_on(live.as_mut()).outcome
        else {
            panic!("sibling function did not complete after peer drop");
        };
        assert_eq!(output(&bytecode, &result.values, "first"), Value::Num(3.0));
        assert_eq!(output(&bytecode, &result.values, "second"), Value::Num(2.0));
    });
}
