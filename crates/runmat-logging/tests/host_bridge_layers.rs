//! A host that builds its own subscriber (Desktop's native shell) puts the log
//! bridge under its shared log level and the trace bridge beside it. The trace feed
//! keeps the run's INFO spans at any level and adds the debug spans when the level
//! allows them. Runs in its own process because the subscriber is process-global.

use std::sync::{Arc, Mutex};

use runmat_logging::{
    runtime_log_bridge_layer, runtime_trace_bridge_layer, set_runtime_log_hook, set_trace_hook,
    RuntimeLogRecord, SharedLogLevel, TraceEvent,
};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{EnvFilter, Layer};

#[test]
fn host_built_subscriber_follows_the_shared_level_with_an_info_floor() {
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

    let run = || {
        let span = tracing::info_span!("runtime.execute");
        let _entered = span.enter();
        tracing::debug_span!("runtime.detail").in_scope(|| {});
        tracing::info!(target: "runmat.runtime", "Execution started");
    };
    let names = |traces: &[TraceEvent], name: &str| {
        traces
            .iter()
            .filter(|event| event.name == name)
            .map(|event| event.ph.clone())
            .collect::<Vec<_>>()
    };
    let reset = || {
        logs.lock().unwrap().clear();
        traces.lock().unwrap().clear();
    };

    run();
    assert_eq!(
        names(&traces.lock().unwrap(), "runtime.execute"),
        ["B", "E"]
    );
    assert!(names(&traces.lock().unwrap(), "runtime.detail").is_empty());
    assert_eq!(logs.lock().unwrap().len(), 1);

    // The Log filter set to "all": the trace feed adds the debug span.
    level
        .set("trace,wgpu=trace,runmat=trace")
        .expect("set the log level");
    reset();
    run();
    assert_eq!(
        names(&traces.lock().unwrap(), "runtime.execute"),
        ["B", "E"]
    );
    assert_eq!(names(&traces.lock().unwrap(), "runtime.detail"), ["B", "E"]);

    // The Log filter set to "error": the log feed goes quiet, the INFO spans stay.
    level.set("error").expect("set the log level");
    reset();
    run();
    assert_eq!(
        names(&traces.lock().unwrap(), "runtime.execute"),
        ["B", "E"]
    );
    assert!(names(&traces.lock().unwrap(), "runtime.detail").is_empty());
    assert!(logs.lock().unwrap().is_empty());
}
