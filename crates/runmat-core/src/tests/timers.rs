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
