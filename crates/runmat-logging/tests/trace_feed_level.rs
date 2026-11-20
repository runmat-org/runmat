//! The trace feed keeps the run's INFO spans whatever the log level is, adds the
//! debug spans when the log level allows them, and the log level still selects the
//! log feed. Runs in its own process because the
//! subscriber is process-global.

use std::sync::{Arc, Mutex};

use runmat_logging::{
    init_logging, set_runtime_log_hook, set_trace_hook, update_log_filter, LoggingOptions,
    RuntimeLogRecord, TraceEvent,
};

#[test]
fn trace_feed_keeps_spans_at_a_quiet_log_level() {
    let logs: Arc<Mutex<Vec<RuntimeLogRecord>>> = Arc::default();
    let traces: Arc<Mutex<Vec<TraceEvent>>> = Arc::default();
    {
        let logs = logs.clone();
        set_runtime_log_hook(move |record| logs.lock().unwrap().push(record.clone()));
        let traces = traces.clone();
        set_trace_hook(move |events| traces.lock().unwrap().extend_from_slice(events));
    }
    // The browser session's default level: the runtime's config default is warn.
    let _guard = init_logging(LoggingOptions {
        enable_otlp: false,
        enable_traces: true,
        pid: 1,
        default_filter: Some("warn".into()),
    });

    let run = |label: &str| {
        let span = tracing::info_span!("runtime.execute", run = label);
        let _entered = span.enter();
        tracing::debug_span!("runtime.detail").in_scope(|| {});
        tracing::info!(target: "runmat.runtime", "Execution started");
        tracing::warn!(target: "runmat.runtime", "a warning");
    };
    let span_phases = |traces: &[TraceEvent]| {
        traces
            .iter()
            .filter(|event| event.name == "runtime.execute")
            .map(|event| event.ph.clone())
            .collect::<Vec<_>>()
    };
    let has_debug_span =
        |traces: &[TraceEvent]| traces.iter().any(|event| event.name == "runtime.detail");
    let messages = |logs: &[RuntimeLogRecord]| {
        logs.iter()
            .map(|record| record.message.clone())
            .collect::<Vec<_>>()
    };

    run("warn");
    assert_eq!(span_phases(&traces.lock().unwrap()), ["B", "E"]);
    assert!(!has_debug_span(&traces.lock().unwrap()));
    assert_eq!(messages(&logs.lock().unwrap()), ["a warning"]);

    // The host raises the log level; info lines now reach the log feed.
    update_log_filter("info").expect("reload the log level");
    logs.lock().unwrap().clear();
    traces.lock().unwrap().clear();
    run("info");
    assert_eq!(span_phases(&traces.lock().unwrap()), ["B", "E"]);
    assert_eq!(
        messages(&logs.lock().unwrap()),
        ["Execution started", "a warning"]
    );

    // Desktop's "all": the trace feed adds the debug spans.
    update_log_filter("trace").expect("reload the log level");
    traces.lock().unwrap().clear();
    run("all");
    assert_eq!(span_phases(&traces.lock().unwrap()), ["B", "E"]);
    assert!(has_debug_span(&traces.lock().unwrap()));

    // The quietest level keeps the INFO spans and drops the debug ones.
    update_log_filter("error").expect("reload the log level");
    logs.lock().unwrap().clear();
    traces.lock().unwrap().clear();
    run("error");
    assert_eq!(span_phases(&traces.lock().unwrap()), ["B", "E"]);
    assert!(!has_debug_span(&traces.lock().unwrap()));
    assert!(logs.lock().unwrap().is_empty());
}
