use super::*;
use crate::context::{
    PreferenceGroups, PreferenceServiceError, PreferenceValues, RuntimePreferenceService,
    RuntimeServiceFuture, RuntimeServicePorts,
};
use std::{cell::Cell, rc::Rc};

fn run<T>(future: impl std::future::Future<Output = T>) -> T {
    futures::executor::block_on(future)
}

#[derive(Clone)]
enum Reply {
    Values(Option<PreferenceValues>),
    Io,
}

struct MailPreferences {
    calls: Rc<Cell<usize>>,
    reply: Reply,
}

impl RuntimePreferenceService for MailPreferences {
    fn all_groups(
        &self,
        _runtime: crate::context::RuntimeContext,
    ) -> RuntimeServiceFuture<Result<PreferenceGroups, PreferenceServiceError>> {
        Box::pin(async { unreachable!("sendmail reads only the Internet group") })
    }

    fn group_values(
        &self,
        _runtime: crate::context::RuntimeContext,
        group: String,
    ) -> RuntimeServiceFuture<Result<Option<PreferenceValues>, PreferenceServiceError>> {
        assert_eq!(group, "Internet");
        let calls = self.calls.clone();
        let reply = self.reply.clone();
        Box::pin(async move {
            calls.set(calls.get() + 1);
            match reply {
                Reply::Values(values) => Ok(values),
                Reply::Io => Err(PreferenceServiceError::Io(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "mail preferences unavailable",
                ))),
            }
        })
    }

    fn contains_names(
        &self,
        _runtime: crate::context::RuntimeContext,
        _group: String,
        _names: Vec<String>,
    ) -> RuntimeServiceFuture<Result<Vec<bool>, PreferenceServiceError>> {
        Box::pin(async { unreachable!("sendmail does not query names separately") })
    }

    fn read_or_create_many(
        &self,
        _runtime: crate::context::RuntimeContext,
        _group: String,
        _names: Vec<String>,
        _defaults: Option<Vec<Value>>,
    ) -> RuntimeServiceFuture<Result<Vec<Value>, PreferenceServiceError>> {
        Box::pin(async { unreachable!("sendmail must not mutate preferences") })
    }

    fn set_many(
        &self,
        _runtime: crate::context::RuntimeContext,
        _group: String,
        _entries: Vec<(String, Value)>,
    ) -> RuntimeServiceFuture<Result<(), PreferenceServiceError>> {
        Box::pin(async { unreachable!("sendmail must not mutate preferences") })
    }
}

fn context(reply: Reply) -> (crate::context::RuntimeContext, Rc<Cell<usize>>) {
    let calls = Rc::new(Cell::new(0));
    let service = Rc::new(MailPreferences {
        calls: calls.clone(),
        reply,
    });
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_preferences(service));
    (runtime, calls)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn aliases_use_one_installed_group_snapshot_and_canonical_text_geometry() {
    let mut values = PreferenceValues::new();
    values.insert(
        "SMTP_Server".into(),
        Value::CharArray(
            CharArray::from_unicode_scalars_with_shape(vec!['x', 'y'], vec![1, 1, 2]).unwrap(),
        ),
    );
    values.insert(
        "SMTPServer".into(),
        Value::CharArray(CharArray::new_row("relay  ")),
    );
    let (runtime, calls) = context(Reply::Values(Some(values)));
    assert_eq!(
        run(read_smtp_server(Some(runtime))).unwrap(),
        Some("relay".into())
    );
    assert_eq!(calls.get(), 1);

    let mut values = PreferenceValues::new();
    values.insert("SMTP_Server".into(), Value::String("preferred".into()));
    values.insert("SMTPServer".into(), Value::String("secondary".into()));
    let (runtime, calls) = context(Reply::Values(Some(values)));
    assert_eq!(
        run(read_smtp_server(Some(runtime))).unwrap(),
        Some("preferred".into())
    );
    assert_eq!(calls.get(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn later_invocation_observes_the_same_installed_preference_authority() {
    let service = Rc::new(crate::context::IsolatedMemoryPreferenceService::new());
    let first = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_preferences(service.clone()));
    run(service.set_many(
        first,
        "Internet".into(),
        vec![("SMTP_Server".into(), Value::String("durable.test".into()))],
    ))
    .expect("set through installed authority");
    let later = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_preferences(service));
    assert_eq!(
        run(read_smtp_server(Some(later))).expect("read same authority"),
        Some("durable.test".into())
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn missing_port_and_storage_error_do_not_fall_back_to_ambient_preferences() {
    let missing = run(read_smtp_server(None)).expect_err("missing invocation context");
    assert!(missing.message().contains("preferences"));
    let no_port = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let missing = run(read_smtp_server(Some(no_port))).expect_err("missing installed port");
    assert!(missing.message().contains("preferences"));
    let (runtime, calls) = context(Reply::Io);
    let error = run(read_smtp_server(Some(runtime))).expect_err("typed storage failure");
    assert!(error.message().contains("mail preferences unavailable"));
    assert_eq!(
        error.identifier(),
        Some("RunMat:sendmail:PreferenceFailure")
    );
    assert!(std::error::Error::source(&error).is_some());
    assert_eq!(calls.get(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn preference_read_is_unpolled_effect_free_and_cancellation_suppresses_publication() {
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;

    let (runtime, calls) = context(Reply::Values(None));
    drop(read_smtp_server(Some(runtime)));
    assert_eq!(calls.get(), 0);

    let source = CancellationSource::new();
    let (runtime, calls) = context(Reply::Values(None));
    let runtime = runtime.with_cancellation_token(source.token());
    source.cancel(CancellationReason::User);
    let error = run(read_smtp_server(Some(runtime))).expect_err("cancelled read");
    assert_eq!(
        error.identifier(),
        crate::execution::ExecutionServiceError::Cancelled
            .into_runtime_error()
            .identifier()
    );
    assert_eq!(calls.get(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn invalid_arguments_reject_before_installed_preference_service() {
    let (runtime, calls) = context(Reply::Values(None));
    let paged = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2])
            .expect("paged subject"),
    );
    let error = run(sendmail_builtin(
        Some(runtime),
        vec![Value::String("recipient@example.test".into()), paged],
    ))
    .expect_err("non-scalar subject");
    assert!(error.message().contains("subject"));
    assert_eq!(error.identifier(), Some("RunMat:sendmail:InvalidText"));
    assert_eq!(calls.get(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn all_known_invalid_mail_forms_have_stable_identifiers_before_preferences() {
    let (runtime, calls) = context(Reply::Values(None));
    let inputs = [
        (
            vec![Value::String("a".into())],
            "RunMat:sendmail:ArgumentCount",
        ),
        (
            vec![Value::String("".into()), Value::String("s".into())],
            "RunMat:sendmail:EmptyRecipient",
        ),
        (
            vec![
                Value::Int(runmat_value::IntValue::U64(7)),
                Value::String("s".into()),
            ],
            "RunMat:sendmail:InvalidMailInput",
        ),
        (
            vec![
                Value::String("a".into()),
                Value::String("s".into()),
                Value::String("b".into()),
                Value::String("file".into()),
            ],
            "RunMat:sendmail:UnsupportedAttachments",
        ),
    ];
    for (args, expected) in inputs {
        let error =
            run(sendmail_builtin(Some(runtime.clone()), args)).expect_err("invalid mail form");
        assert_eq!(error.identifier(), Some(expected));
    }
    assert_eq!(calls.get(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn excessive_output_request_rejects_before_mail_parsing_or_service_effects() {
    let (runtime, calls) = context(Reply::Values(None));
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = run(sendmail_builtin(
        Some(runtime),
        vec![
            Value::String("a@example.test".into()),
            Value::String("status".into()),
        ],
    ))
    .expect_err("fixed status output limit");
    assert_eq!(
        error.identifier(),
        Some(SendmailFailure::Outputs.identifier())
    );
    assert_eq!(calls.get(), 0);
}
