//! Timer class dispatch and local-task completion through a real session.

use super::*;

#[test]
fn timer_constructs_finds_starts_and_deletes_through_vm() {
    let _timer_lock = TIMER_CORE_TEST_LOCK.lock().unwrap();
    let mut session = RunMatSession::with_options(false, false).expect("session init");
    let source = r#"
        t = timer('Name', 'clientTimer', 'Tag', 'clientBatch', 'TimerFcn', @(src, evt) 0);
        f = timerfind('Tag', 'clientBatch');
        foundName = f.Name;
        start(t);
        wait(t);
        tasks = t.TasksExecuted;
        running = t.Running;
        delete(t);
        validAfterDelete = isvalid(t);
    "#;
    let outcome = execute_text_request(&mut session, source).expect("timer workflow succeeds");
    assert!(
        outcome.diagnostics.is_empty(),
        "timer workflow diagnostics: {:?}; upserts: {:?}",
        outcome.diagnostics,
        outcome.workspace_delta.upserts
    );

    assert_eq!(
        outcome_named_upsert_value(&outcome, "foundName"),
        Some(&runmat_value::Value::String("clientTimer".into()))
    );
    assert!(outcome_has_named_upsert(
        &outcome,
        "tasks",
        &runmat_value::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "running",
        &runmat_value::Value::String("off".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "validAfterDelete",
        &runmat_value::Value::Bool(false)
    ));
}

#[test]
fn timer_text_callback_executes_in_the_session_base_workspace() {
    let _timer_lock = TIMER_CORE_TEST_LOCK.lock().unwrap();
    let mut session = RunMatSession::with_options(false, false).expect("session init");
    let source = r#"
        text_counter = 0;
        t = timer('TimerFcn', 'text_counter = text_counter + 1;');
        start(t);
        wait(t);
        observed = text_counter;
        delete(t);
    "#;
    let outcome = execute_text_request(&mut session, source).expect("text callback workflow");
    assert!(outcome.diagnostics.is_empty(), "{:?}", outcome.diagnostics);
    assert!(outcome_has_named_upsert(
        &outcome,
        "observed",
        &runmat_value::Value::Num(1.0)
    ));
    let persisted = execute_text_request(&mut session, "after_callback = text_counter;")
        .expect("callback workspace persists");
    assert!(
        persisted.diagnostics.is_empty(),
        "{:?}",
        persisted.diagnostics
    );
    assert!(outcome_has_named_upsert(
        &persisted,
        "after_callback",
        &runmat_value::Value::Num(1.0)
    ));
}

#[test]
fn delayed_timer_text_callback_uses_the_committed_session_function_registry() {
    let _timer_lock = TIMER_CORE_TEST_LOCK.lock().unwrap();
    let mut session = RunMatSession::with_options(false, false).expect("session init");
    let admit = r#"
        delayed_counter = 10;
        delayed_origin = "";
        delayed_timer = timer( ...
            'StartDelay', 0.2, ...
            'TimerFcn', '[delayed_counter, delayed_origin] = callback_increment(delayed_counter);');
        start(delayed_timer);
        function [value, origin] = callback_increment(value)
            value = value + 1;
            origin = mfilename();
        end
    "#;
    let admitted =
        execute_text_request_named_source(&mut session, "foreground_callback_owner.m", admit)
            .expect("admit delayed callback");
    assert!(
        admitted.diagnostics.is_empty(),
        "{:?}",
        admitted.diagnostics
    );

    let collision = execute_text_request_named_source(
        &mut session,
        "transient_callback_collision.m",
        r#"
            pause(0.4);
            error("RunMat:test:transientCallbackCollision", "expected failure");
            function [value, origin] = callback_increment(value)
                value = value + 100;
                origin = mfilename();
            end
        "#,
    )
    .expect("transient collision returns a language outcome");
    assert!(
        !collision.diagnostics.is_empty(),
        "transient collision must retain its expected error"
    );

    let completed = execute_text_request(
        &mut session,
        r#"
            wait(delayed_timer);
            observed_delayed = delayed_counter;
            observed_origin = delayed_origin;
            delete(delayed_timer);
        "#,
    )
    .expect("drive delayed callback in later request");
    assert!(
        completed.diagnostics.is_empty(),
        "{:?}",
        completed.diagnostics
    );
    assert!(outcome_has_named_upsert(
        &completed,
        "observed_delayed",
        &runmat_value::Value::Num(11.0)
    ));
    assert!(outcome_has_named_upsert(
        &completed,
        "observed_origin",
        &runmat_value::Value::String("foreground_callback_owner".into())
    ));
}

#[test]
fn timer_text_callback_error_keeps_its_callback_source_frame() {
    let _timer_lock = TIMER_CORE_TEST_LOCK.lock().unwrap();
    let mut session = RunMatSession::with_options(false, false).expect("session init");
    let outcome = execute_text_request(
        &mut session,
        r#"
            source_timer = timer( ...
                'Name', 'sourceProbe', ...
                'TimerFcn', 'error("RunMat:test:timerSource", "expected timer failure");');
            start(source_timer);
            wait(source_timer);
        "#,
    )
    .expect("timer failure is reported as a language diagnostic");
    let diagnostic = outcome
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "RunMat:test:timerSource")
        .expect("timer error diagnostic");
    assert!(
        diagnostic
            .callstack
            .iter()
            .any(|frame| frame.contains("<timer:sourceProbe:TimerFcn>")),
        "callback source missing from callstack: {:?}",
        diagnostic.callstack
    );
}

#[test]
fn delayed_text_callback_from_failed_foreground_generation_is_invalidated() {
    let _timer_lock = TIMER_CORE_TEST_LOCK.lock().unwrap();
    let mut session = RunMatSession::with_options(false, false).expect("session init");
    let admitted = execute_text_request_named_source(
        &mut session,
        "failed_timer_owner.m",
        r#"
            invalidated_counter = 0;
            invalidated_timer = timer( ...
                'StartDelay', 0.2, ...
                'TimerFcn', 'invalidated_counter = failed_generation_helper();');
            start(invalidated_timer);
            error("RunMat:test:failedTimerOwner", "expected foreground failure");
            function value = failed_generation_helper()
                value = 1;
            end
        "#,
    )
    .expect("foreground failure is reported as a language outcome");
    assert!(admitted
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RunMat:test:failedTimerOwner"));

    let waited = execute_text_request(&mut session, "wait(invalidated_timer);")
        .expect("invalidated callback failure is reported as a language outcome");
    assert!(
        waited.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "RunMat:SourceEvaluation:GenerationInvalidated"
        }),
        "{:?}",
        waited.diagnostics
    );

    let observed = execute_text_request(
        &mut session,
        "observed_invalidated = invalidated_counter; delete(invalidated_timer);",
    )
    .expect("inspect workspace after invalidated callback");
    assert!(
        observed.diagnostics.is_empty(),
        "{:?}",
        observed.diagnostics
    );
    assert!(outcome_has_named_upsert(
        &observed,
        "observed_invalidated",
        &runmat_value::Value::Num(0.0)
    ));
}

#[test]
fn admitted_pending_callback_finishes_once_after_its_generation_is_invalidated() {
    let _timer_lock = TIMER_CORE_TEST_LOCK.lock().unwrap();
    let mut session = RunMatSession::with_options(false, false).expect("session init");
    let admitted = execute_text_request_named_source(
        &mut session,
        "pending_timer_owner.m",
        r#"
            pending_callback_counter = 0;
            pending_callback_timer = timer( ...
                'ExecutionMode', 'fixedRate', ...
                'BusyMode', 'queue', ...
                'Period', 0.05, ...
                'TasksToExecute', 2, ...
                'TimerFcn', [ ...
                    'pending_callback_counter = pending_callback_counter + 1; ', ...
                    'pause(0.2); ', ...
                    'pending_callback_counter = pending_callback_counter + 1;']);
            start(pending_callback_timer);
            pause(0.05);
            error("RunMat:test:pendingTimerOwner", "expected foreground failure");
        "#,
    )
    .expect("foreground failure is reported as a language outcome");
    assert!(admitted
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RunMat:test:pendingTimerOwner"));

    let waited = execute_text_request(&mut session, "wait(pending_callback_timer);")
        .expect("later invalidated firing is reported as a language outcome");
    assert!(
        waited.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "RunMat:SourceEvaluation:GenerationInvalidated"
        }),
        "{:?}",
        waited.diagnostics
    );

    let observed = execute_text_request(
        &mut session,
        "observed_pending = pending_callback_counter; delete(pending_callback_timer);",
    )
    .expect("inspect completed admitted callback");
    assert!(
        observed.diagnostics.is_empty(),
        "{:?}",
        observed.diagnostics
    );
    assert!(outcome_has_named_upsert(
        &observed,
        "observed_pending",
        &runmat_value::Value::Num(2.0)
    ));
}

#[test]
fn timer_absolute_startat_datevec_dispatches_in_matlab_mode() {
    let _timer_lock = TIMER_CORE_TEST_LOCK.lock().unwrap();
    let mut session = RunMatSession::with_options(false, false).expect("session init");
    let source = r#"
        t = timer('TimerFcn', @(src, evt) 0);
        startat(t, datevec(now + 2/86400));
        running = t.Running;
        stop(t);
        stopped = t.Running;
        delete(t);
    "#;
    let outcome = execute_text_request(&mut session, source).expect("absolute startat request");
    assert!(
        outcome.diagnostics.is_empty(),
        "absolute startat diagnostics: {:?}",
        outcome.diagnostics
    );
    assert!(outcome_has_named_upsert(
        &outcome,
        "running",
        &runmat_value::Value::String("on".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "stopped",
        &runmat_value::Value::String("off".into())
    ));
}

#[test]
fn timer_member_method_start_works_through_vm() {
    let _timer_lock = TIMER_CORE_TEST_LOCK.lock().unwrap();
    let mut session = RunMatSession::with_options(false, false).expect("session init");
    let source = r#"
        t = timer('TimerFcn', @(src, evt) 0);
        t.start();
        wait(t);
        tasks = t.TasksExecuted;
        running = t.Running;
    "#;
    let outcome = execute_text_request(&mut session, source).expect("timer member start succeeds");
    assert!(
        outcome.diagnostics.is_empty(),
        "timer member workflow diagnostics: {:?}; upserts: {:?}",
        outcome.diagnostics,
        outcome.workspace_delta.upserts
    );
    assert!(outcome_has_named_upsert(
        &outcome,
        "tasks",
        &runmat_value::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "running",
        &runmat_value::Value::String("off".into())
    ));
}

#[test]
fn timer_rejects_invalid_post_construction_assignments_through_vm() {
    let _timer_lock = TIMER_CORE_TEST_LOCK.lock().unwrap();
    for (source, expected) in [
        ("t = timer; t.Period = 0;", "Period must be a finite scalar"),
        (
            "t = timer; t.TimerFcn = 42;",
            "TimerFcn must be text, a function handle, or a callback cell array",
        ),
    ] {
        let mut session = RunMatSession::with_options(false, false).expect("session init");
        let outcome =
            execute_text_request(&mut session, source).expect("assignment failure is diagnostic");
        assert!(
            outcome
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains(expected)),
            "missing timer assignment diagnostic containing {expected:?}; diagnostics: {:?}",
            outcome.diagnostics
        );
    }
}

#[test]
fn timer_property_setters_publish_to_the_live_handle_through_vm() {
    use runmat_value::Value;

    let _timer_lock = TIMER_CORE_TEST_LOCK.lock().unwrap();
    let mut session = RunMatSession::with_options(false, false).expect("session init");
    let source = r#"
        t = timer;
        alias = t;
        t.Period = 0.25;
        t.StartDelay = 0.1;
        t.TasksToExecute = 2;
        t.BusyMode = 'queue';
        t.ExecutionMode = 'fixedSpacing';
        t.Name = 'updated';
        t.Tag = 'batch';
        t.ObjectVisibility = 'off';
        t.UserData = 42;
        t.TimerFcn = @(src, evt) 0;
        t.StartFcn = @(src, evt) 0;
        t.StopFcn = @(src, evt) 0;
        t.ErrorFcn = @(src, evt) 0;
        period = alias.Period;
        tag = alias.Tag;
        payload = alias.UserData;
        start(alias);
        wait(alias);
        tasks = t.TasksExecuted;
        delete(t);
    "#;
    let outcome = execute_text_request(&mut session, source).expect("timer property workflow");
    assert!(outcome.diagnostics.is_empty(), "{:?}", outcome.diagnostics);
    for (name, expected) in [
        ("period", Value::Num(0.25)),
        ("tag", Value::String("batch".into())),
        ("payload", Value::Num(42.0)),
        ("tasks", Value::Num(2.0)),
    ] {
        assert!(
            outcome_has_named_upsert(&outcome, name, &expected),
            "{name}"
        );
    }
}
