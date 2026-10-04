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
            let sink = matches!(
                call.symbol.as_str(),
                "add_classpath" | "remove_classpath" | "set_classpath"
            );
            self.calls.borrow_mut().push(call);
            Box::pin(async move {
                if sink {
                    Ok(runmat_value::ValueSequence::empty())
                } else {
                    runmat_value::ValueSequence::single(Value::String("returned".into()))
                        .map_err(|error| invalid_call(error.to_string()))
                }
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

    #[test]
    fn classpath_mutations_accept_the_adapters_zero_output_sequence() {
        let (runtime, recorder) = context();
        let path = Value::String("lib/example.jar".into());
        let added = futures::executor::block_on(java_add_path_builtin(
            Some(runtime.clone()),
            vec![path.clone()],
        ))
        .unwrap();
        assert_eq!(added, Value::OutputList(Vec::new()));
        let removed = futures::executor::block_on(java_remove_path_builtin(
            Some(runtime.clone()),
            vec![path.clone()],
        ))
        .unwrap();
        assert_eq!(removed, Value::OutputList(Vec::new()));
        let replaced =
            futures::executor::block_on(java_class_path_builtin(Some(runtime), vec![path]))
                .unwrap();
        assert_eq!(replaced, Value::OutputList(Vec::new()));
        let calls = recorder.calls.borrow();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].symbol, "add_classpath");
        assert_eq!(calls[1].symbol, "remove_classpath");
        assert_eq!(calls[2].symbol, "set_classpath");
        assert!(calls.iter().all(|call| call.requested_outputs == 0));
    }

    #[test]
    fn classpath_controls_reject_numeric_and_boolean_before_any_foreign_call() {
        let (runtime, recorder) = context();
        for value in [Value::Num(42.0), Value::Bool(true)] {
            let error = futures::executor::block_on(java_add_path_builtin(
                Some(runtime.clone()),
                vec![Value::String("valid.jar".into()), value.clone()],
            ))
            .unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:Foreign:InvalidCall"));
            assert!(recorder.calls.borrow().is_empty());
            let error = futures::executor::block_on(java_class_path_builtin(
                Some(runtime.clone()),
                vec![value],
            ))
            .unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:Foreign:InvalidCall"));
            assert!(recorder.calls.borrow().is_empty());
        }
    }

    #[test]
    fn classpath_controls_accept_only_actual_text_scalar_shapes() {
        let (runtime, recorder) = context();
        let text = "lib/example.jar";
        let trailing = runmat_value::CharArray::new_with_shape(
            text.chars().collect(),
            vec![1, text.chars().count(), 1],
        )
        .unwrap();
        let added = futures::executor::block_on(java_add_path_builtin(
            Some(runtime.clone()),
            vec![Value::CharArray(trailing)],
        ));
        assert!(added.is_ok());
        let pages = runmat_value::CharArray::new_with_shape(
            text.chars().collect(),
            vec![1, 1, text.chars().count()],
        )
        .unwrap();
        let error = futures::executor::block_on(java_add_path_builtin(
            Some(runtime),
            vec![Value::CharArray(pages)],
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:Foreign:InvalidCall"));
        assert_eq!(recorder.calls.borrow().len(), 1);
        assert_eq!(
            recorder.calls.borrow()[0].arguments[0],
            Value::String(text.into())
        );
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

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
fn browser_usejava_returns_false_only_for_valid_feature_forms() {
    assert_eq!(
        futures::executor::block_on(use_java_builtin(None, Value::String("AwT".into()))).unwrap(),
        Value::Bool(false)
    );
    for invalid in [
        Value::Num(1.0),
        Value::Bool(true),
        Value::String("unknown".into()),
    ] {
        let error = futures::executor::block_on(use_java_builtin(None, invalid)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:Foreign:InvalidCall"));
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
fn browser_rejects_each_valid_jvm_operation_without_a_native_host() {
    let class = Value::String("java.lang.String".into());
    let method = Value::String("valueOf".into());
    let path = Value::String("lib/example.jar".into());
    let calls = [
        futures::executor::block_on(java_object_builtin(None, vec![class.clone()])),
        futures::executor::block_on(java_object_edt_builtin(None, vec![class.clone()])),
        futures::executor::block_on(java_method_builtin(
            None,
            vec![method.clone(), class.clone()],
        )),
        futures::executor::block_on(java_method_edt_builtin(None, vec![method, class.clone()])),
        futures::executor::block_on(java_array_builtin(None, vec![class, Value::Num(1.0)])),
        futures::executor::block_on(java_add_path_builtin(None, vec![path.clone()])),
        futures::executor::block_on(java_remove_path_builtin(None, vec![path])),
        futures::executor::block_on(java_class_path_builtin(None, vec![])),
        futures::executor::block_on(jenv_builtin(None, vec![])),
    ];
    for call in calls {
        assert_eq!(
            call.unwrap_err().identifier(),
            Some("RunMat:Foreign:UnsupportedOnWasm")
        );
    }
}
