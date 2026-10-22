#![cfg(feature = "native-accel")]

#[path = "support/mod.rs"]
mod test_helpers;

use futures::executor::block_on;
use runmat_runtime::context::{
    RuntimeAccelerationService, RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts,
};
use runmat_value::Value;
use std::{cell::Cell, collections::HashSet, rc::Rc};

struct InjectedAcceleration;
impl RuntimeAccelerationService for InjectedAcceleration {}

/// Dispatch runs inside the actual function frame, so this observes the context
/// installed by the entry API rather than the embedding context before entry.
struct ObserveFunctionContext {
    expected: Option<Rc<dyn RuntimeAccelerationService>>,
    calls: Cell<usize>,
}

impl RuntimeBuiltinService for ObserveFunctionContext {
    fn bindings_by_name(&self, name: &str) -> Vec<runmat_runtime::builtin::RuntimeBuiltinBinding> {
        assert_eq!(name, "logical");
        let runtime = runmat_runtime::context::legacy::active().expect("function runtime scope");
        let acceleration = runtime
            .service_ports()
            .acceleration()
            .expect("function entry must install a missing acceleration service");
        if let Some(expected) = &self.expected {
            assert!(
                Rc::ptr_eq(expected, acceleration),
                "injected service replaced"
            );
        }
        self.calls.set(self.calls.get() + 1);
        vec![runmat_runtime::builtin::RuntimeBuiltinBinding::new(
            runmat_builtins::BuiltinBindingIdentity {
                builtin: runmat_builtins::LOGICAL_CATALOG_ENTRY.identity,
                variant: "default",
            },
            |invocation| {
                assert_eq!(invocation.args.len(), 1);
                assert_eq!(invocation.args.get(0), Some(&Value::Bool(true)));
                runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async {
                    runmat_runtime::execution::RootedValueSequence::new(
                        runmat_value::ValueSequence::single(Value::Bool(true))
                            .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?,
                    )
                    .map_err(|error| error.into_runtime_error())
                })
            },
        )]
    }
}

fn witness(expected: Option<Rc<dyn RuntimeAccelerationService>>, with_counts: bool) {
    runmat_gc::gc_test_context(|| {
        runmat_vm::reset_thread_state_for_tests();
        let observer = Rc::new(ObserveFunctionContext {
            expected: expected.clone(),
            calls: Cell::new(0),
        });
        let mut ports = RuntimeServicePorts::default().with_builtin(observer.clone());
        if let Some(service) = expected {
            ports = ports.with_acceleration(service);
        }
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(ports);
        let _scope = runtime.enter();
        let bytecode = test_helpers::compile_source("x = true; y = logical(x);").unwrap();
        let slot = *bytecode
            .var_names
            .iter()
            .find(|(_, name)| name.as_str() == "y")
            .unwrap()
            .0;
        let vars = vec![Value::Num(0.0); bytecode.var_count];
        let result = if with_counts {
            block_on(runmat_vm::interpret_function_with_counts(
                &bytecode,
                vars,
                "installation_witness",
                1,
                0,
                HashSet::new(),
            ))
        } else {
            block_on(runmat_vm::interpret_function(&bytecode, vars))
        }
        .expect("standalone function API");
        assert_eq!(result[slot], Value::Bool(true));
        assert_eq!(observer.calls.get(), 1);
        runmat_vm::reset_thread_state_for_tests();
    });
}

#[test]
fn standalone_function_entry_installs_missing_acceleration_service() {
    witness(None, false);
}

#[test]
fn standalone_function_counted_entry_preserves_injected_acceleration_service() {
    witness(Some(Rc::new(InjectedAcceleration)), true);
}
