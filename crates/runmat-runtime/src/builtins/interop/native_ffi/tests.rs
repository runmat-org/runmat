use super::*;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use std::{cell::Cell, rc::Rc};

    #[derive(Default)]
    struct CountForeign(Rc<Cell<usize>>);

    impl crate::context::RuntimeForeignService for CountForeign {
        fn invoke(
            &self,
            _context: crate::context::RuntimeContext,
            _call: crate::context::ForeignCall,
        ) -> crate::context::RuntimeServiceFuture<crate::context::ForeignCallCompletion> {
            self.0.set(self.0.get() + 1);
            Box::pin(async {
                crate::context::ForeignCallCompletion::invoked(
                    crate::execution::RootedValueSequence::new(runmat_value::ValueSequence::empty())
                        .map_err(|error| error.into_runtime_error()),
                )
            })
        }
    }

    fn recording_context() -> (crate::context::RuntimeContext, Rc<Cell<usize>>) {
        let calls = Rc::new(Cell::new(0));
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            crate::context::RuntimeServicePorts::default()
                .with_foreign(Rc::new(CountForeign(Rc::clone(&calls)))),
        );
        (runtime, calls)
    }

    #[test]
    fn invalid_public_controls_do_not_invoke_foreign_service() {
        let (runtime, calls) = recording_context();
        let invalid = Value::output_list(Vec::new());
        let results = [
            futures::executor::block_on(clibgen_build_interface_builtin(
                Some(runtime.clone()),
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::String("fixture.h".into()),
                    Value::String("InterfaceName".into()),
                    Value::String("missing_library".into()),
                ]),
            )),
            futures::executor::block_on(loadlibrary_builtin(
                Some(runtime.clone()),
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            )),
            futures::executor::block_on(calllib_builtin(
                Some(runtime.clone()),
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            )),
            futures::executor::block_on(libfunctions_builtin(
                Some(runtime.clone()),
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::String("fixture".into()),
                    Value::String("bad".into()),
                ]),
            )),
            futures::executor::block_on(libpointer_builtin(
                Some(runtime.clone()),
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::String("int32Ptr".into()),
                    Value::Num(1.0),
                    Value::Num(2.0),
                ]),
            )),
            futures::executor::block_on(libstruct_builtin(
                Some(runtime.clone()),
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::String("fixture_record".into()),
                    Value::Num(1.0),
                ]),
            )),
            futures::executor::block_on(setdatatype_builtin(
                Some(runtime.clone()),
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(1.0),
                    Value::String("int32Ptr".into()),
                    Value::Num(1.0),
                ]),
            )),
            futures::executor::block_on(unloadlibrary_builtin(
                Some(runtime.clone()),
                None,
                invalid.clone(),
            )),
            futures::executor::block_on(libisloaded_builtin(Some(runtime), None, invalid)),
        ];
        for result in results {
            assert_eq!(
                result.expect_err("invalid control").identifier(),
                Some("RunMat:Foreign:InvalidCall")
            );
        }
        assert_eq!(
            calls.get(),
            0,
            "known-invalid calls must not invoke foreign service"
        );
    }

    #[test]
    fn unpolled_native_query_has_no_foreign_effect() {
        let (runtime, calls) = recording_context();
        let future = libisloaded_builtin(Some(runtime), None, Value::String("fixture".into()));
        drop(future);
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn impossible_public_output_counts_fail_before_foreign_service() {
        let (runtime, calls) = recording_context();
        let checks = [
            futures::executor::block_on(clibgen_build_interface_builtin(
                Some(runtime.clone()),
                Some(2),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::String("fixture.h".into()),
                    Value::String("Libraries".into()),
                    Value::String("fixture".into()),
                ]),
            )),
            futures::executor::block_on(loadlibrary_builtin(
                Some(runtime.clone()),
                Some(3),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::String("fixture".into()),
                    Value::String("fixture.h".into()),
                ]),
            )),
            futures::executor::block_on(unloadlibrary_builtin(
                Some(runtime.clone()),
                Some(1),
                Value::String("fixture".into()),
            )),
            futures::executor::block_on(libisloaded_builtin(
                Some(runtime.clone()),
                Some(2),
                Value::String("fixture".into()),
            )),
            futures::executor::block_on(libfunctions_builtin(
                Some(runtime.clone()),
                Some(2),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                    "fixture".into(),
                )]),
            )),
            futures::executor::block_on(libpointer_builtin(
                Some(runtime.clone()),
                Some(2),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            )),
            futures::executor::block_on(libstruct_builtin(
                Some(runtime.clone()),
                Some(2),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                    "fixture_record".into(),
                )]),
            )),
            futures::executor::block_on(setdatatype_builtin(
                Some(runtime),
                Some(1),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(1.0),
                    Value::String("int32Ptr".into()),
                    Value::Num(1.0),
                ]),
            )),
        ];
        for result in checks {
            let error = result.expect_err("impossible public output count");
            assert_eq!(error.identifier(), Some("RunMat:Foreign:InvalidCall"));
            assert!(error.message().contains("too many output arguments"));
        }
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn calllib_output_count_is_left_to_prepared_signature() {
        let (runtime, calls) = recording_context();
        let _ = futures::executor::block_on(calllib_builtin(
            Some(runtime),
            Some(3),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("fixture".into()),
                Value::String("fixture_add".into()),
            ]),
        ));
        assert_eq!(
            calls.get(),
            1,
            "wrapper must not impose a fixed calllib output limit"
        );
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
fn valid_native_call_reports_unsupported_browser_host() {
    let result = futures::executor::block_on(libisloaded_builtin(
        None,
        None,
        Value::String("fixture".into()),
    ));
    assert_eq!(
        result.expect_err("native FFI unavailable").identifier(),
        Some("RunMat:Foreign:UnsupportedOnWasm")
    );
}
