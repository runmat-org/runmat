use super::*;

#[test]
fn portable_source_ordinals_ignore_unrelated_session_source_allocations() {
    let source = crate::ExecutableSource::new(
        "fixture",
        "main.m",
        "function y = main(); y = helper(); end\nfunction y = helper(); y = 4; end\n",
    );
    let mut first = crate::RunMatSession::with_options(false, false).expect("first session");
    let first_unit =
        futures::executor::block_on(first.compile_executable_unit(source.clone(), None))
            .expect("first product");
    let first_bytes = first_unit
        .portable_envelope()
        .expect("first envelope")
        .canonical_bytes()
        .expect("first bytes");

    let mut second = crate::RunMatSession::with_options(false, false).expect("second session");
    second
        .compile_input_for_source_name("unrelated.m", "unrelated = 17;")
        .expect("allocate an unrelated runtime source identity");
    let second_unit = futures::executor::block_on(second.compile_executable_unit(source, None))
        .expect("second product");
    let second_bytes = second_unit
        .portable_envelope()
        .expect("second envelope")
        .canonical_bytes()
        .expect("second bytes");
    assert_eq!(first_bytes, second_bytes);
    assert_eq!(first_unit.mir(), second_unit.mir());
    assert_eq!(first_unit.analysis(), second_unit.analysis());
    assert_eq!(first_unit.source_map(), second_unit.source_map());
}

#[test]
fn changed_reachable_source_text_changes_portable_identity() {
    let mut session = crate::RunMatSession::with_options(false, false).expect("session");
    let compile = |session: &mut crate::RunMatSession, value| {
        let source = crate::ExecutableSource::new(
            "fixture",
            "main.m",
            format!(
                "function y = main(); y = helper(); end\nfunction y = helper(); y = {value}; end\n"
            ),
        );
        futures::executor::block_on(session.compile_executable_unit(source, None))
            .expect("portable product")
            .portable_envelope()
            .expect("portable envelope")
            .manifest
            .identity
    };
    assert_ne!(compile(&mut session, 1), compile(&mut session, 2));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn pending_source_lease_coexists_with_interpreted_and_native_unit_invocations() {
    let mut session = crate::RunMatSession::with_options(false, false).expect("session");
    let held = runmat_runtime::source_context::lease_source_context_in_context(
        &session.runtime_context,
        "<eval>",
        None,
        "pending = 99;",
    )
    .expect("session source lease");
    let unit = futures::executor::block_on(session.compile_executable_unit(
        crate::ExecutableSource::new("fixture", "answer.m", "function y = answer(); y = 7; end\n"),
        None,
    ))
    .expect("unit");
    let invocation = crate::ProcedureInvocation {
        target: crate::ProcedureTarget::Function("answer".into()),
        arguments: Vec::new(),
        requested_outputs: 1,
    };
    let expected =
        runmat_value::ValueSequence::single(runmat_value::Value::Num(7.0)).expect("one output");
    let established = futures::executor::block_on(session.invoke_executable(
        &unit,
        invocation.clone(),
        &crate::InvocationControl::default(),
    ))
    .expect("interpreted or tiered execution");
    assert_eq!(established, expected);
    let native = futures::executor::block_on(session.invoke_executable(
        &unit,
        invocation,
        &crate::InvocationControl::default().force_generic_native(),
    ))
    .expect("native execution");
    assert_eq!(native, expected);
    assert_eq!(
        runmat_runtime::source_context::source_info_in_context(
            &session.runtime_context,
            held.source_id(),
        )
        .expect("pending eval source remains live")
        .text
        .as_ref(),
        "pending = 99;"
    );
    let held_id = held.source_id();
    drop(held);
    assert!(runmat_runtime::source_context::source_info_in_context(
        &session.runtime_context,
        held_id,
    )
    .is_none());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn dropped_pending_public_eval_retires_only_its_source() {
    use futures::channel::oneshot;
    use std::future::Future;
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll};

    let mut session = crate::RunMatSession::with_options(false, false).expect("session");
    let runtime = session.runtime_context.clone();
    let sibling = runmat_runtime::source_context::lease_source_context_in_context(
        &runtime,
        "<sibling>",
        None,
        "sibling = 7;",
    )
    .expect("sibling source");
    let observed = Arc::new(Mutex::new(None));
    let observed_by_handler = Arc::clone(&observed);
    let (release, gate) = oneshot::channel::<()>();
    let gate = Arc::new(Mutex::new(Some(gate)));
    session.install_async_input_handler(move |_| {
        *observed_by_handler.lock().expect("observed source") =
            runmat_runtime::source_context::current_source_info();
        let gate = gate
            .lock()
            .expect("input gate")
            .take()
            .expect("no input replay");
        async move {
            gate.await.map_err(|_| "input abandoned".to_string())?;
            Ok(crate::execution::InputResponse::Line("1".into()))
        }
    });
    let request = crate::abi::ExecutionRequest::for_source(
        crate::abi::SourceInput::Text {
            name: "pending_eval.m".into(),
            text: "result = eval(\"input('wait: '); 41 + 1\");".into(),
        },
        session.compat_mode(),
        crate::abi::HostExecutionPolicy::default(),
        session.workspace_handle(),
    );
    let mut future = Box::pin(session.execute_request(request));
    for _ in 0..16 {
        let mut task = Context::from_waker(futures::task::noop_waker_ref());
        assert!(matches!(
            Future::poll(future.as_mut(), &mut task),
            Poll::Pending
        ));
        if observed.lock().expect("observed source").is_some() {
            break;
        }
    }
    let eval_source = observed
        .lock()
        .expect("observed source")
        .clone()
        .expect("public eval reached pending input");
    assert_eq!(eval_source.name.as_ref(), "<eval>");
    let eval_id = eval_source.source_id.expect("eval source identity");
    assert_eq!(eval_source.text.as_ref(), "input('wait: '); 41 + 1");
    assert_eq!(
        runmat_runtime::source_context::source_info_in_context(&runtime, eval_id)
            .expect("pending eval source still live")
            .text,
        eval_source.text
    );
    drop(future);
    assert!(runmat_runtime::source_context::source_info_in_context(&runtime, eval_id).is_none());
    assert_eq!(
        runmat_runtime::source_context::source_info_in_context(&runtime, sibling.source_id())
            .expect("unrelated source remains live")
            .text
            .as_ref(),
        "sibling = 7;"
    );
    assert!(
        release.send(()).is_err(),
        "dropped input future released gate"
    );
}

#[test]
fn retained_function_handle_error_keeps_original_source_after_later_public_calls() {
    let mut session = crate::RunMatSession::with_options(false, false).expect("session");
    let mut execute = |name: &str, text: &str| {
        let request = crate::abi::ExecutionRequest::for_source(
            crate::abi::SourceInput::Text {
                name: name.into(),
                text: text.into(),
            },
            session.compat_mode(),
            crate::abi::HostExecutionPolicy::default(),
            session.workspace_handle(),
        );
        futures::executor::block_on(session.execute_request(request))
            .result
            .expect("public source execution")
    };
    let definition = execute(
        "original_retained.m",
        "function y = retainedBoom()\nerror(\"RunMat:retained:boom\", \"retained body failed\");\ny = 1;\nend",
    );
    assert!(definition.diagnostics.is_empty());
    let capture = execute("capture_handle.m", "saved = @retainedBoom;");
    assert!(capture.diagnostics.is_empty());
    let outcome = execute("later_call.m", "saved();");
    let error = outcome
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "RunMat:retained:boom")
        .expect("retained body error remains visible");
    assert!(
        error
            .callstack
            .iter()
            .any(|frame| frame.contains("original_retained.m")),
        "retained call must cite the definition source, not the later call: {error:?}"
    );
    assert!(
        error
            .callstack
            .iter()
            .all(|frame| !frame.contains("capture_handle.m")),
        "retained body must not be attributed to handle construction: {error:?}"
    );
}
