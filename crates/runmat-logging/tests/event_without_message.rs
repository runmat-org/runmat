//! An event with fields but no message (`tracing::info!(target: "runmat.fea", nodes = 3)`) has
//! the macro's default name, `event <source file>:<line>`, which carries the build machine's
//! path. The log and trace feeds name such an event by its target instead, so the Desktop never
//! shows a source path as run progress. Runs in its own process because the hooks and the
//! subscriber are process-global.

use std::sync::{Arc, Mutex};

use runmat_logging::{
    runtime_log_bridge_layer, runtime_trace_bridge_layer, set_runtime_log_hook, set_trace_hook,
    RuntimeLogRecord, SharedLogLevel, TraceEvent,
};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{EnvFilter, Layer};

#[test]
fn an_event_without_a_message_is_named_by_its_target_not_its_source_path() {
    let logs: Arc<Mutex<Vec<RuntimeLogRecord>>> = Arc::default();
    let traces: Arc<Mutex<Vec<TraceEvent>>> = Arc::default();
    {
        let logs = logs.clone();
        set_runtime_log_hook(move |record| logs.lock().unwrap().push(record.clone()));
        let traces = traces.clone();
        set_trace_hook(move |events| traces.lock().unwrap().extend_from_slice(events));
    }
    let level = SharedLogLevel::new(EnvFilter::new("info"));
    let subscriber = tracing_subscriber::registry()
        .with(runtime_log_bridge_layer().with_filter(level.log_filter()))
        .with(runtime_trace_bridge_layer(7, &level));
    tracing::subscriber::set_global_default(subscriber).expect("install the subscriber");

    let span = tracing::info_span!("runtime.execute");
    let entered = span.enter();
    tracing::info!(target: "runmat.fea.solve", nodes = 3_u64);
    tracing::info!(target: "runmat.fea.solve", "Solving");
    tracing::info!(name: "fea.solve.assembled", target: "runmat.fea.solve", nodes = 3_u64);
    drop(entered);

    let logs = logs.lock().unwrap();
    let messages: Vec<&str> = logs.iter().map(|record| record.message.as_str()).collect();
    assert_eq!(
        messages,
        ["runmat.fea.solve", "Solving", "fea.solve.assembled"]
    );
    assert_eq!(
        logs[0].fields.as_ref().and_then(|fields| fields.get("nodes")),
        Some(&serde_json::json!("3"))
    );

    let traces = traces.lock().unwrap();
    let instants: Vec<&str> = traces
        .iter()
        .filter(|event| event.ph == "i")
        .map(|event| event.name.as_str())
        .collect();
    assert_eq!(
        instants,
        ["runmat.fea.solve", "Solving", "fea.solve.assembled"]
    );
    for name in messages.iter().chain(instants.iter()) {
        assert!(
            !name.contains(".rs") && !name.starts_with("event "),
            "{name}"
        );
    }
}
