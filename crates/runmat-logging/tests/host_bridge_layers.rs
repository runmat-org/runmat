//! A host that builds its own subscriber (Desktop's native shell) puts the log
//! bridge under its reloadable log level and the trace bridge beside it. The
//! trace feed then keeps the run's spans at any log level. Runs in its own
//! process because the subscriber is process-global.

use std::sync::{Arc, Mutex};

use runmat_logging::{
    runtime_log_bridge_layer, runtime_trace_bridge_layer, set_runtime_log_hook, set_trace_hook,
    RuntimeLogRecord, TraceEvent,
};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{reload, EnvFilter, Layer};

#[test]
fn host_built_subscriber_keeps_spans_when_the_log_level_is_error() {
    let logs: Arc<Mutex<Vec<RuntimeLogRecord>>> = Arc::default();
    let traces: Arc<Mutex<Vec<TraceEvent>>> = Arc::default();
    {
        let logs = logs.clone();
        set_runtime_log_hook(move |record| logs.lock().unwrap().push(record.clone()));
        let traces = traces.clone();
        set_trace_hook(move |events| traces.lock().unwrap().extend_from_slice(events));
    }
    let (level, handle) = reload::Layer::new(EnvFilter::new("info"));
    let subscriber = tracing_subscriber::registry()
        .with(runtime_log_bridge_layer().with_filter(level))
        .with(runtime_trace_bridge_layer(7));
    tracing::subscriber::set_global_default(subscriber).expect("install the subscriber");

    let run = || {
        let span = tracing::info_span!("runtime.execute");
        let _entered = span.enter();
        tracing::info!(target: "runmat.runtime", "Execution started");
    };
    let span_phases = |traces: &[TraceEvent]| {
        traces
            .iter()
            .filter(|event| event.name == "runtime.execute")
            .map(|event| event.ph.clone())
            .collect::<Vec<_>>()
    };

    run();
    assert_eq!(span_phases(&traces.lock().unwrap()), ["B", "E"]);
    assert_eq!(logs.lock().unwrap().len(), 1);

    // The Log filter set to "error": the log feed goes quiet, the trace stays whole.
    handle
        .reload(EnvFilter::new("error"))
        .expect("reload the log level");
    logs.lock().unwrap().clear();
    traces.lock().unwrap().clear();
    run();
    assert_eq!(span_phases(&traces.lock().unwrap()), ["B", "E"]);
    assert!(logs.lock().unwrap().is_empty());
}
