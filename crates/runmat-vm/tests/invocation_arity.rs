#[path = "support/mod.rs"]
mod support;

use futures::channel::oneshot;
use runmat_runtime::builtin::RuntimeBuiltinBinding;
use runmat_runtime::context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts};
use runmat_value::Value;
use std::cell::RefCell;
use std::future::Future;
use std::rc::Rc;
use std::task::{Context, Poll};

struct GatedSin {
    gate: Rc<RefCell<Option<oneshot::Receiver<()>>>>,
}

impl RuntimeBuiltinService for GatedSin {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        if name != "sin" {
            return Vec::new();
        }
        let entry = runmat_builtins::builtin_catalog_primary_entry_by_name(name)
            .expect("canonical sin catalog entry");
        let gate = Rc::clone(&self.gate);
        vec![RuntimeBuiltinBinding::new(
            runmat_builtins::BuiltinBindingIdentity {
                builtin: entry.identity,
                variant: "default",
            },
            move |_| {
                let pending = gate.borrow_mut().take().expect("no gate invocation replay");
                Box::pin(async move {
                    pending
                        .await
                        .map_err(|_| runmat_runtime::RuntimeError::new("sin gate abandoned"))?;
                    Ok(Value::Num(0.0))
                })
            },
        )]
    }
}

fn runtime_with_gate(base: RuntimeContext) -> (RuntimeContext, oneshot::Sender<()>) {
    let (sender, receiver) = oneshot::channel();
    let runtime = base.with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(
        GatedSin {
            gate: Rc::new(RefCell::new(Some(receiver))),
        },
    )));
    (runtime, sender)
}

fn output(bytecode: &runmat_vm::Bytecode, values: &[Value], name: &str) -> Value {
    let slot = bytecode
        .var_names
        .iter()
        .find_map(|(slot, candidate)| (candidate == name).then_some(*slot))
        .expect("compiled output slot");
    values[slot].clone()
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
        let (first_runtime, first_gate) = runtime_with_gate(base.clone());
        let (second_runtime, second_gate) = runtime_with_gate(base);
        let mut first_vars = Vec::new();
        let mut second_vars = Vec::new();
        let mut first = Box::pin(runmat_vm::interpret_with_vars_in_context(
            &first_bytecode,
            &mut first_vars,
            Some("<main>"),
            first_runtime,
        ));
        let mut second = Box::pin(runmat_vm::interpret_with_vars_in_context(
            &second_bytecode,
            &mut second_vars,
            Some("<main>"),
            second_runtime,
        ));
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        assert!(first.as_mut().poll(&mut context).is_pending());
        assert!(second.as_mut().poll(&mut context).is_pending());

        first_gate
            .send(())
            .expect("first invocation remains pending");
        let Poll::Ready(Ok(runmat_vm::InterpreterOutcome::Completed(first_result))) =
            first.as_mut().poll(&mut context)
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
        let Poll::Ready(Ok(runmat_vm::InterpreterOutcome::Completed(second_result))) =
            second.as_mut().poll(&mut context)
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
        let (abandoned_runtime, abandoned_gate) = runtime_with_gate(base.clone());
        let (live_runtime, live_gate) = runtime_with_gate(base);
        let mut abandoned_vars = Vec::new();
        let mut live_vars = Vec::new();
        let mut abandoned = Box::pin(runmat_vm::interpret_with_vars_in_context(
            &bytecode,
            &mut abandoned_vars,
            Some("<main>"),
            abandoned_runtime,
        ));
        let mut live = Box::pin(runmat_vm::interpret_with_vars_in_context(
            &bytecode,
            &mut live_vars,
            Some("<main>"),
            live_runtime,
        ));
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        assert!(abandoned.as_mut().poll(&mut context).is_pending());
        assert!(live.as_mut().poll(&mut context).is_pending());
        drop(abandoned);
        assert!(
            abandoned_gate.send(()).is_err(),
            "dropped call retires its gate"
        );
        live_gate.send(()).expect("sibling still owns its gate");
        let Poll::Ready(Ok(runmat_vm::InterpreterOutcome::Completed(result))) =
            live.as_mut().poll(&mut context)
        else {
            panic!("sibling function did not complete after peer drop");
        };
        assert_eq!(output(&bytecode, &result.values, "first"), Value::Num(3.0));
        assert_eq!(output(&bytecode, &result.values, "second"), Value::Num(2.0));
    });
}
