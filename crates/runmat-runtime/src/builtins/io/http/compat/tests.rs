use super::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
fn selected_local_mail_command_never_reports_browser_success() {
    let previous = crate::builtins::common::env::var("RUNMAT_SENDMAIL_COMMAND").ok();
    crate::builtins::common::env::set_var("RUNMAT_SENDMAIL_COMMAND", "/unavailable/sendmail");
    let error = run(sendmail_builtin(
        None,
        vec![Value::from("alice@example.test"), Value::from("Subject")],
    ))
    .expect_err("browser local command route must reject");
    match previous {
        Some(value) => crate::builtins::common::env::set_var("RUNMAT_SENDMAIL_COMMAND", &value),
        None => crate::builtins::common::env::remove_var("RUNMAT_SENDMAIL_COMMAND"),
    }
    assert_eq!(
        error.identifier(),
        Some(SendmailFailure::Command.identifier())
    );
}

#[test]
fn websave_rejects_excess_outputs_before_input_or_http_service() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = run(websave_builtin(None, vec![Value::Num(1.0)])).expect_err("excess outputs");
    assert_eq!(error.identifier(), Some("RunMat:websave:TooManyOutputs"));
}

#[test]
fn websave_rejects_known_invalid_timeout_before_resident_query_readback() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident query value");
    let error = run(websave_builtin(
        None,
        vec![
            Value::from("result.bin"),
            Value::from("https://example.test"),
            Value::from("Timeout"),
            Value::Num(-1.0),
            Value::from("q"),
            Value::GpuTensor(allocation.handle.clone()),
        ],
    ))
    .expect_err("invalid host timeout");
    assert!(error.message().contains("Timeout"));
    assert!(!error.message().contains("provider"));
    assert_eq!(allocation.probe.retirements(), 0);
}

fn run(value: impl std::future::Future<Output = BuiltinResult<Value>>) -> BuiltinResult<Value> {
    futures::executor::block_on(value)
}

#[test]
fn url_round_trip_uses_percent_encoding() {
    let encoded = run(urlencode_builtin(vec![Value::String("a b/c".to_string())])).unwrap();
    assert_eq!(encoded, char_value("a%20b%2Fc"));
    let decoded = run(urldecode_builtin(vec![char_value("a%20b%2Fc")])).unwrap();
    assert_eq!(decoded, char_value("a b/c"));
}

#[test]
fn url_text_helpers_reject_integer_and_resident_values_before_provider_access() {
    assert_eq!(
        URLENCODE_INTEGER_AUDIT.kind,
        BuiltinIntegerAuditKind::NotApplicable
    );
    assert_eq!(
        URLDECODE_INTEGER_AUDIT.kind,
        BuiltinIntegerAuditKind::NotApplicable
    );
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let original = allocation.handle.wire_descriptor();
    assert!(
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
            .is_none()
    );
    for error in [
        run(urlencode_builtin(vec![Value::Int(
            runmat_value::IntValue::U64(u64::MAX),
        )]))
        .expect_err("integer text must reject"),
        run(urldecode_builtin(vec![Value::GpuTensor(
            allocation.handle.clone(),
        )]))
        .expect_err("resident text must reject"),
    ] {
        assert!(error
            .message()
            .contains("host string scalar or character vector"));
        assert!(!error.message().to_ascii_lowercase().contains("provider"));
    }
    assert_eq!(allocation.handle.wire_descriptor(), original);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn websave_rest_applies_options_and_query_pairs() {
    let mut url = Url::parse("https://example.test/data").unwrap();
    let mut headers = Vec::new();
    let mut timeout = DEFAULT_TIMEOUT;
    let mut user_agent = USER_AGENT.to_string();
    let mut options = runmat_value::StructValue::new();
    options.insert("Timeout", Value::Num(7.0));
    options.insert("UserAgent", Value::String("agent".to_string()));
    let mut header_struct = runmat_value::StructValue::new();
    header_struct.insert("XRunMat", Value::String("yes".to_string()));
    options.insert("HeaderFields", Value::Struct(header_struct));

    parse_websave_rest(
        &mut url,
        &mut headers,
        &mut timeout,
        &mut user_agent,
        &[
            Value::Struct(options),
            Value::String("q".to_string()),
            Value::String("hello world".to_string()),
        ],
        None,
    )
    .unwrap();

    assert_eq!(url.query(), Some("q=hello+world"));
    assert_eq!(headers, vec![("XRunMat".to_string(), "yes".to_string())]);
    assert_eq!(timeout, Duration::from_secs(7));
    assert_eq!(user_agent, "agent");
}

#[test]
fn websave_options_api_key_is_header_not_query_and_known_invalid_name_preflights() {
    let mut source = runmat_value::StructValue::new();
    source.insert("KeyName", Value::from("X-API-Key"));
    source.insert("KeyValue", Value::from("secret"));
    let prepared = super::super::api_key::PreparedApiKey::parse(&source);
    let mut url = Url::parse("https://example.test/data").unwrap();
    let mut headers = Vec::new();
    let mut timeout = DEFAULT_TIMEOUT;
    let mut user_agent = USER_AGENT.to_string();
    let args = [Value::Struct(source)];
    preflight_websave_rest(&args, Some(&prepared)).unwrap();
    parse_websave_rest(
        &mut url,
        &mut headers,
        &mut timeout,
        &mut user_agent,
        &args,
        Some(&prepared),
    )
    .unwrap();
    assert_eq!(url.query(), None);
    assert_eq!(headers, vec![("X-API-Key".into(), "secret".into())]);

    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let mut invalid = runmat_value::StructValue::new();
    invalid.insert("KeyName", Value::from("bad\nname"));
    invalid.insert("KeyValue", Value::GpuTensor(allocation.handle.clone()));
    let prepared = super::super::api_key::PreparedApiKey::parse(&invalid);
    let error = preflight_websave_rest(&[Value::Struct(invalid)], Some(&prepared))
        .expect_err("known malformed header must reject before a resident value is gathered");
    assert!(error.message().contains("KeyName"));
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn websave_query_values_preserve_exact_integer_vectors() {
    let value = Value::Tensor(
        runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
            vec![1, 2],
        )
        .expect("integer query vector"),
    );
    assert_eq!(
        query_value_to_string(&value, "id").expect("query encoding"),
        "9007199254740993,18446744073709551615"
    );
}

#[test]
fn sendmail_rejects_integer_payloads_before_transport() {
    let error = run(sendmail_builtin(
        None,
        vec![
            Value::String("recipient@example.test".to_string()),
            Value::String("subject".to_string()),
            Value::Int(runmat_value::IntValue::U64(9_007_199_254_740_993)),
        ],
    ))
    .expect_err("integer message must reject before transport selection");
    assert!(error.message().contains("must be text"));

    let nested =
        runmat_value::CellArray::new(vec![Value::Int(runmat_value::IntValue::I8(1))], 1, 1)
            .unwrap();
    let error = run(sendmail_builtin(
        None,
        vec![
            Value::Cell(nested),
            Value::String("subject".to_string()),
            Value::String("message".to_string()),
        ],
    ))
    .expect_err("nested integer recipient must reject before transport selection");
    assert!(error.message().contains("must be text"));
}

#[test]
fn sendmail_two_argument_form_uses_an_empty_message() {
    let (recipients, subject, message) = parse_sendmail_args(&[
        Value::String("recipient@example.test".to_string()),
        Value::String("subject".to_string()),
    ])
    .expect("two-argument sendmail form");
    assert_eq!(recipients, ["recipient@example.test"]);
    assert_eq!(subject, "subject");
    assert!(message.is_empty());
}

#[path = "tests/sendmail_preferences.rs"]
mod sendmail_preferences;
