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
