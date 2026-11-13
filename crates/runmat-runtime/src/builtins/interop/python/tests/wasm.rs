use super::*;

fn assert_unsupported(error: crate::RuntimeError) {
    assert_eq!(error.identifier(), Some("RunMat:Foreign:UnsupportedOnWasm"));
}

#[wasm_bindgen_test::wasm_bindgen_test]
async fn valid_pyenv_status_and_configuration_reject_without_native_host() {
    assert_unsupported(pyenv_builtin(None, Vec::new()).await.unwrap_err());
    assert_unsupported(
        pyenv_builtin(
            None,
            vec![
                Value::String("ExecutionMode".into()),
                Value::String("OutOfProcess".into()),
            ],
        )
        .await
        .unwrap_err(),
    );
}

#[wasm_bindgen_test::wasm_bindgen_test]
async fn valid_pyrun_and_pyrunfile_reject_without_native_host() {
    let outputs = Value::String("answer".into());
    let input_name = Value::String("input_value".into());
    let input = Value::Int(runmat_value::IntValue::I64(40));
    assert_unsupported(
        pyrun_builtin(
            None,
            vec![
                Value::String("answer = input_value + 2".into()),
                outputs.clone(),
                input_name.clone(),
                input.clone(),
            ],
        )
        .await
        .unwrap_err(),
    );
    assert_unsupported(
        pyrunfile_builtin(
            None,
            vec![
                Value::String("answer.py".into()),
                outputs,
                input_name,
                input,
            ],
        )
        .await
        .unwrap_err(),
    );
}

#[wasm_bindgen_test::wasm_bindgen_test]
async fn valid_terminate_form_rejects_without_native_host() {
    let environment = Value::Object(ObjectInstance::new(PYTHON_ENVIRONMENT_CLASS));
    assert_unsupported(
        terminate_builtin(None, vec![environment])
            .await
            .unwrap_err(),
    );
}

#[wasm_bindgen_test::wasm_bindgen_test]
async fn invalid_forms_keep_validation_error_before_unsupported_host_error() {
    assert_eq!(
        pyrun_builtin(None, Vec::new())
            .await
            .unwrap_err()
            .identifier(),
        Some("RunMat:Foreign:InvalidCall")
    );
    assert_eq!(
        terminate_builtin(None, vec![Value::Num(1.0)])
            .await
            .unwrap_err()
            .identifier(),
        Some("RunMat:Foreign:InvalidCall")
    );
}
