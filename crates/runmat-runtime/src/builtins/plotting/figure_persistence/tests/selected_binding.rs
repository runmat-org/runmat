use super::*;

#[test]
fn selected_sink_binding_projects_exact_requested_status_count() {
    let _guard = setup();
    let figure = figure_builtin(vec![]).expect("figure");
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _conflicting_ambient = crate::output_count::push_output_count(Some(2));

    for name in [SAVEAS, SAVEFIG, HGSAVE] {
        let zero_path = temp_path(&format!("{name}_selected_zero"));
        let one_path = temp_path(&format!("{name}_selected_one"));
        let rejected_path = temp_path(&format!("{name}_selected_two"));
        for path in [&zero_path, &one_path, &rejected_path] {
            let _ = std::fs::remove_file(path);
        }

        let zero = block_on(crate::call_builtin_async_with_outputs_in_context(
            &runtime,
            name,
            &[
                Value::Num(figure),
                Value::String(zero_path.to_string_lossy().into_owned()),
            ],
            0,
        ))
        .unwrap_or_else(|error| panic!("{name} zero-output save failed: {error}"));
        assert_eq!(zero, Value::OutputList(Vec::new()), "{name}");

        let one = block_on(crate::call_builtin_async_with_outputs_in_context(
            &runtime,
            name,
            &[
                Value::Num(figure),
                Value::String(one_path.to_string_lossy().into_owned()),
            ],
            1,
        ))
        .unwrap_or_else(|error| panic!("{name} status-output save failed: {error}"));
        assert_eq!(one, Value::Bool(true), "{name}");

        let error = block_on(crate::call_builtin_async_with_outputs_in_context(
            &runtime,
            name,
            &[
                Value::Num(figure),
                Value::String(rejected_path.to_string_lossy().into_owned()),
            ],
            2,
        ));
        let error = match error {
            Ok(value) => panic!("{name} unexpectedly returned {value:?}"),
            Err(error) => error,
        };
        assert_eq!(
            error.identifier(),
            ERROR_INVALID_ARGUMENT.identifier,
            "{name}"
        );
        assert!(!rejected_path.exists(), "{name}");

        let _ = std::fs::remove_file(zero_path);
        let _ = std::fs::remove_file(one_path);
    }
}

#[test]
fn selected_load_bindings_reject_output_count_before_filesystem_effects() {
    let _guard = setup();
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    for (name, outputs) in [(OPENFIG, 2), (HGLOAD, 3)] {
        let error = block_on(crate::call_builtin_async_with_outputs_in_context(
            &runtime,
            name,
            &[Value::String("does-not-exist.fig".into())],
            outputs,
        ))
        .expect_err("output count rejects before file access");
        assert_eq!(
            error.identifier(),
            ERROR_INVALID_ARGUMENT.identifier,
            "{name}"
        );
        assert!(error.message().contains("output"), "{name}: {error}");
    }
}
