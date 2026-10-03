use super::*;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use std::{cell::RefCell, rc::Rc};

    #[derive(Default)]
    struct RecordingForeign {
        calls: RefCell<Vec<crate::context::ForeignCall>>,
    }

    impl crate::context::RuntimeForeignService for RecordingForeign {
        fn invoke(
            &self,
            _context: crate::context::RuntimeContext,
            call: crate::context::ForeignCall,
        ) -> crate::context::RuntimeServiceFuture<
            Result<runmat_value::ValueSequence, crate::RuntimeError>,
        > {
            self.calls.borrow_mut().push(call);
            Box::pin(async {
                runmat_value::ValueSequence::single(Value::String("returned".into()))
                    .map_err(|error| invalid_call(error.to_string()))
            })
        }
    }

    fn context() -> (crate::context::RuntimeContext, Rc<RecordingForeign>) {
        let recorder = Rc::new(RecordingForeign::default());
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            crate::context::RuntimeServicePorts::default().with_foreign(recorder.clone()),
        );
        (runtime, recorder)
    }

    #[test]
    fn explicit_context_routes_constructor_and_static_method_without_reparsing() {
        let (runtime, recorder) = context();
        let constructed = futures::executor::block_on(java_object_builtin(
            Some(runtime.clone()),
            vec![Value::String("java.lang.StringBuilder".into())],
        ));
        assert_eq!(constructed.unwrap(), Value::String("returned".into()));
        let invoked = futures::executor::block_on(java_method_builtin(
            Some(runtime),
            vec![
                Value::String("max".into()),
                Value::String("java.lang.Math".into()),
                Value::Int(runmat_value::IntValue::I32(4)),
            ],
        ));
        assert_eq!(invoked.unwrap(), Value::String("returned".into()));
        let calls = recorder.calls.borrow();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].adapter, JAVA_ADAPTER);
        assert_eq!(calls[0].symbol, "construct");
        assert_eq!(calls[1].symbol, "call_static");
        assert_eq!(
            calls[1].arguments[0],
            Value::String("java.lang.Math".into())
        );
        assert_eq!(calls[1].arguments[1], Value::String("max".into()));
        assert_eq!(
            calls[1].arguments[2],
            Value::Int(runmat_value::IntValue::I32(4))
        );
    }

    #[test]
    fn invalid_method_fails_before_foreign_access_and_missing_context_is_typed() {
        let (runtime, recorder) = context();
        let invalid = futures::executor::block_on(java_method_builtin(
            Some(runtime),
            vec![Value::String("max".into())],
        ))
        .unwrap_err();
        assert_eq!(invalid.identifier(), Some("RunMat:Foreign:InvalidCall"));
        assert!(recorder.calls.borrow().is_empty());
        let missing = futures::executor::block_on(java_object_builtin(
            None,
            vec![Value::String("java.lang.StringBuilder".into())],
        ))
        .unwrap_err();
        assert_eq!(missing.identifier(), Some("RunMat:Foreign:HostUnavailable"));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn isjava_rejects_ordinary_runmat_values_without_host_access() {
    assert_eq!(
        is_java_builtin(Value::Num(7.0)).unwrap(),
        Value::Bool(false)
    );
}
