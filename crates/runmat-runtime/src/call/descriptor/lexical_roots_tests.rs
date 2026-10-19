use super::*;
use runmat_value::{CellArray, Closure, LexicalEnvironment};
use std::rc::Rc;

#[test]
fn an_unpolled_descriptor_future_roots_its_nested_activation() {
    runmat_gc::gc_test_context(|| {
        let root =
            runmat_gc::gc_allocate_rooted(Value::Cell(CellArray::new(Vec::new(), 1, 0).unwrap()))
                .unwrap();
        let handle = root.handle();
        let binding = runmat_value::BoundFunctionBinding::unavailable(7);
        let closure = Closure::from_data(runmat_value::ClosureData {
            function_name: "nested".into(),
            binding: Some(binding.clone()),
            source_role: runmat_types::FunctionSourceRole::Nested,
            source_file: None,
            captures: Vec::new().into(),
            lexical_environment: Some(LexicalEnvironment::new(handle, None)),
        });
        let descriptor = CallableDescriptor {
            target: CallableTarget::BoundLexical {
                binding,
                closure: RetainedClosureValue::new(closure),
            },
            args: Vec::new(),
            requested_outputs: 1,
            metadata: CallableMetadata::feval(Some("nested".into())),
        };
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let future = execute_callable_descriptor_in_context(&runtime, descriptor);
        drop(root);
        runmat_gc::gc_collect_major().unwrap();
        assert!(
            runmat_gc::gc_with_value(&handle, |_| ()).is_ok(),
            "the unpolled future must retain the exact activation"
        );
        drop(future);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&handle, |_| ()).is_err());
    });
}

#[test]
fn direct_bound_environment_api_roots_before_first_poll() {
    runmat_gc::gc_test_context(|| {
        let root =
            runmat_gc::gc_allocate_rooted(Value::Cell(CellArray::new(Vec::new(), 1, 0).unwrap()))
                .unwrap();
        let handle = root.handle();
        let environment = LexicalEnvironment::new(handle, None);
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let future = crate::user_functions::try_call_bound_function_with_environment_in_context(
            &runtime,
            runmat_value::BoundFunctionBinding::unavailable(7),
            &[],
            1,
            Some(environment),
        );
        drop(root);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&handle, |_| ()).is_ok());
        drop(future);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&handle, |_| ()).is_err());
    });
}
