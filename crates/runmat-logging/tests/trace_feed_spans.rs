//! Each span reaches the trace feed once: one B when it is created and one E when
//! it closes, under an id no other span shares, with its fields as args. Runs in
//! its own process because the subscriber is process-global.

use std::sync::{Arc, Mutex};

use runmat_logging::{init_logging, set_trace_hook, LoggingOptions, TraceEvent};

fn arg<'a>(event: &'a TraceEvent, key: &str) -> Option<&'a str> {
    event.args.as_ref()?.get(key)?.as_str()
}

#[test]
fn spans_reach_the_trace_feed_once_with_their_fields() {
    let traces: Arc<Mutex<Vec<TraceEvent>>> = Arc::default();
    {
        let traces = traces.clone();
        set_trace_hook(move |events| traces.lock().unwrap().extend_from_slice(events));
    }
    let _guard = init_logging(LoggingOptions {
        enable_otlp: false,
        enable_traces: true,
        pid: 1,
        default_filter: Some("info".into()),
    });

    {
        let run = tracing::info_span!(
            "runtime.execute",
            input_len = 42u64,
            step = tracing::field::Empty
        );
        // An async span is entered at every poll; the feed still shows one span.
        for _ in 0..3 {
            let _entered = run.enter();
        }
        run.in_scope(|| {
            tracing::info_span!("runtime.compile").in_scope(|| {
                tracing::info!(target: "runmat.runtime", "Execution started");
            });
            // The registry reuses the closed compile span's id for this one.
            tracing::info_span!("interpreter.run", function = "loop_demo").in_scope(|| {});
        });
        run.record("step", "run");
    }

    let events = traces.lock().unwrap().clone();
    let spans: Vec<&TraceEvent> = events
        .iter()
        .filter(|e| e.ph == "B" || e.ph == "E")
        .collect();
    let phases = |name: &str| {
        spans
            .iter()
            .filter(|e| e.name == name)
            .map(|e| e.ph.as_str())
            .collect::<Vec<_>>()
    };
    assert_eq!(phases("runtime.execute"), ["B", "E"]);
    assert_eq!(phases("runtime.compile"), ["B", "E"]);
    assert_eq!(phases("interpreter.run"), ["B", "E"]);

    let begin = |name: &str| {
        *spans
            .iter()
            .find(|e| e.name == name && e.ph == "B")
            .unwrap()
    };
    let end = |name: &str| {
        *spans
            .iter()
            .find(|e| e.name == name && e.ph == "E")
            .unwrap()
    };
    let execute = begin("runtime.execute");
    let compile = begin("runtime.compile");
    let interpreter = begin("interpreter.run");

    // Distinct ids, even though the registry reused one.
    assert_ne!(compile.span_id, interpreter.span_id);
    assert_eq!(compile.span_id, end("runtime.compile").span_id);
    assert_eq!(compile.parent_span_id, execute.span_id);
    assert_eq!(interpreter.parent_span_id, execute.span_id);
    assert_eq!(execute.parent_span_id, None);

    // Fields: the span's own at B, later records at E.
    assert_eq!(arg(execute, "input_len"), Some("42"));
    assert_eq!(arg(interpreter, "function"), Some("loop_demo"));
    assert_eq!(arg(execute, "step"), None);
    assert_eq!(arg(end("runtime.execute"), "step"), Some("run"));

    // A log line inside a span names that span as its parent.
    let started = events
        .iter()
        .find(|e| e.ph == "i" && e.name == "Execution started")
        .expect("the info line reached the trace feed");
    assert_eq!(started.parent_span_id, compile.span_id);
}
