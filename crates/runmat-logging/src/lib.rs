use once_cell::sync::OnceCell;
use serde::Serialize;
use serde_json::Value as JsonValue;
use std::sync::Arc;
use tracing::subscriber::DefaultGuard;
use tracing::Subscriber;
use tracing_log::LogTracer;
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::Layer;

#[cfg(not(target_arch = "wasm32"))]
use std::time::SystemTime;

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeLogRecord {
    pub ts: String,
    pub level: String,
    pub target: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fields: Option<JsonValue>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TraceEvent {
    pub name: String,
    pub cat: String,
    pub ph: String,
    pub ts: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dur: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tid: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<JsonValue>,
}

type LogHook = Arc<dyn Fn(&RuntimeLogRecord) + Send + Sync>;
type TraceHook = Arc<dyn Fn(&[TraceEvent]) + Send + Sync>;

static LOG_HOOK: OnceCell<LogHook> = OnceCell::new();
static TRACE_HOOK: OnceCell<TraceHook> = OnceCell::new();
static FALLBACK_TRACE_ID: OnceCell<String> = OnceCell::new();

pub struct LoggingGuard {
    _guard: Option<DefaultGuard>,
}

#[derive(Clone, Default)]
pub struct LoggingOptions {
    pub enable_otlp: bool,
    pub enable_traces: bool,
    pub pid: i64,
}

pub fn set_runtime_log_hook<F>(hook: F)
where
    F: Fn(&RuntimeLogRecord) + Send + Sync + 'static,
{
    let _ = LOG_HOOK.set(Arc::new(hook));
}

pub fn set_trace_hook<F>(hook: F)
where
    F: Fn(&[TraceEvent]) + Send + Sync + 'static,
{
    let _ = TRACE_HOOK.set(Arc::new(hook));
}

pub fn init_logging(opts: LoggingOptions) -> LoggingGuard {
    // Install LogTracer so log:: macros flow into tracing
    let _ = LogTracer::init();

    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_from_env("RUNMAT_LOG"))
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let build_subscriber = || {
        let bridge_layer = LogBridgeLayer;
        let trace_layer = if opts.enable_traces {
            Some(TraceBridgeLayer { pid: opts.pid })
        } else {
            None
        };
        tracing_subscriber::registry()
            .with(env_filter.clone())
            .with(bridge_layer)
            .with(trace_layer.clone())
    };

    let subscriber = build_subscriber();

    #[cfg(feature = "otlp")]
    let subscriber = {
        let otel_layer = opts.enable_otlp.then(otel_layer);
        subscriber.with(otel_layer)
    };

    let guard = match tracing::subscriber::set_global_default(subscriber) {
        Ok(()) => None,
        Err(_) => Some(tracing::subscriber::set_default(build_subscriber())),
    };

    LoggingGuard { _guard: guard }
}

struct LogBridgeLayer;
#[derive(Clone)]
struct TraceBridgeLayer {
    pid: i64,
}

#[cfg(target_arch = "wasm32")]
fn now_rfc3339() -> String {
    js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_else(|| "1970-01-01T00:00:00.000Z".to_string())
}

#[cfg(not(target_arch = "wasm32"))]
fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

#[cfg(target_arch = "wasm32")]
fn now_timestamp_micros() -> i64 {
    // Date::now returns milliseconds since epoch as f64
    (js_sys::Date::now() * 1000.0).round() as i64
}

#[cfg(not(target_arch = "wasm32"))]
fn now_timestamp_micros() -> i64 {
    chrono::Utc::now().timestamp_micros()
}

impl<S> Layer<S> for LogBridgeLayer
where
    S: Subscriber,
{
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut visitor = JsonVisitor::default();
        event.record(&mut visitor);

        let record = RuntimeLogRecord {
            ts: now_rfc3339(),
            level: event.metadata().level().to_string(),
            target: event.metadata().target().to_string(),
            message: visitor
                .message
                .unwrap_or_else(|| event.metadata().name().to_string()),
            trace_id: current_trace_id(),
            span_id: current_span_id(),
            fields: visitor
                .fields
                .and_then(|v| v.as_object().cloned().map(JsonValue::Object))
                .filter(|obj| obj.as_object().map(|m| !m.is_empty()).unwrap_or(false)),
        };

        if let Some(hook) = LOG_HOOK.get() {
            hook(&record);
        }
    }
}

impl<S> Layer<S> for TraceBridgeLayer
where
    S: Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        // Only emit trace events if a hook is set
        let hook = match TRACE_HOOK.get() {
            Some(h) => h,
            None => return,
        };

        let meta = event.metadata();
        let ts = now_timestamp_micros();

        let trace_id = current_trace_id();
        let span_id = current_span_id();

        let mut visitor = JsonVisitor::default();
        event.record(&mut visitor);

        let args = visitor
            .fields
            .as_ref()
            .and_then(|v| v.as_object())
            .cloned()
            .map(JsonValue::Object);

        let ev = TraceEvent {
            name: visitor.message.unwrap_or_else(|| meta.name().to_string()),
            cat: meta.target().to_string(),
            ph: "i".to_string(), // instant event
            ts,
            dur: None,
            pid: Some(self.pid),
            tid: None,
            trace_id,
            span_id,
            args,
        };

        hook(&[ev]);
    }

    fn on_enter(&self, id: &tracing::span::Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
        if TRACE_HOOK.get().is_none() {
            return;
        }
        if let Some(span) = ctx.span(id) {
            emit_span_event(span, "B", self.pid);
        }
    }

    fn on_exit(&self, id: &tracing::span::Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
        if TRACE_HOOK.get().is_none() {
            return;
        }
        if let Some(span) = ctx.span(id) {
            emit_span_event(span, "E", self.pid);
        }
    }
}

fn emit_span_event<S>(span: tracing_subscriber::registry::SpanRef<'_, S>, phase: &str, pid: i64)
where
    S: Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    let hook = match TRACE_HOOK.get() {
        Some(h) => h,
        None => return,
    };
    let meta = span.metadata();
    let ts = now_timestamp_micros();

    let trace_id = current_trace_id();
    let span_id = current_span_id();

    let ev = TraceEvent {
        name: meta.name().to_string(),
        cat: meta.target().to_string(),
        ph: phase.to_string(),
        ts,
        dur: None,
        pid: Some(pid),
        tid: None,
        trace_id,
        span_id,
        args: None,
    };
    hook(&[ev]);
}

fn current_trace_id() -> Option<String> {
    current_trace_span_ids().0
}

fn current_span_id() -> Option<String> {
    current_trace_span_ids().1
}

fn current_trace_span_ids() -> (Option<String>, Option<String>) {
    #[cfg(feature = "otlp")]
    {
        use opentelemetry::trace::TraceContextExt;
        use tracing_opentelemetry::OpenTelemetrySpanExt;
        let span = tracing::Span::current();
        let ctx = span.context();
        let span = ctx.span();
        let sc = span.span_context();
        if sc.is_valid() {
            return (
                Some(sc.trace_id().to_string()),
                Some(sc.span_id().to_string()),
            );
        }
    }
    let span_id = tracing::Span::current()
        .id()
        .map(|id| id.into_u64().to_string());
    let trace_id = Some(fallback_trace_id());
    (trace_id, span_id)
}

fn fallback_trace_id() -> String {
    FALLBACK_TRACE_ID
        .get_or_init(|| {
            #[cfg(target_arch = "wasm32")]
            {
                let micros = now_timestamp_micros() as u128;
                let rand = (js_sys::Math::random() * 1_000_000.0) as u128;
                format!("{:x}-{:x}", micros, rand)
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let nanos = SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or_default();
                let tid = format!("{:?}", std::thread::current().id());
                format!("{:x}-{tid}", nanos)
            }
        })
        .clone()
}

#[derive(Default)]
struct JsonVisitor {
    message: Option<String>,
    fields: Option<JsonValue>,
}

impl tracing::field::Visit for JsonVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        let entry = serde_json::json!(format!("{value:?}"));
        if field.name() == "message" {
            self.message = Some(entry.as_str().unwrap_or_default().to_string());
        } else {
            let obj = self
                .fields
                .get_or_insert_with(|| JsonValue::Object(Default::default()));
            if let JsonValue::Object(map) = obj {
                map.insert(field.name().to_string(), entry);
            }
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        let entry = JsonValue::String(value.to_string());
        if field.name() == "message" {
            self.message = Some(value.to_string());
        } else {
            let obj = self
                .fields
                .get_or_insert_with(|| JsonValue::Object(Default::default()));
            if let JsonValue::Object(map) = obj {
                map.insert(field.name().to_string(), entry);
            }
        }
    }
}

#[cfg(feature = "otlp")]
fn otel_layer<S>() -> tracing_opentelemetry::OpenTelemetryLayer<S, opentelemetry_sdk::trace::Tracer>
where
    S: tracing::Subscriber,
    for<'span> S: tracing_subscriber::registry::LookupSpan<'span>,
{
    use opentelemetry::KeyValue;
    use opentelemetry_otlp::WithExportConfig;
    use opentelemetry_sdk::{runtime::Tokio, trace, Resource};

    let endpoint = std::env::var("RUNMAT_OTEL_ENDPOINT").unwrap_or_default();
    let otel_exporter = opentelemetry_otlp::new_exporter()
        .http()
        .with_endpoint(endpoint);
    let otel_tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(otel_exporter)
        .with_trace_config(
            trace::config()
                .with_resource(Resource::new(vec![KeyValue::new("service.name", "runmat")])),
        )
        .install_batch(Tokio)
        .expect("failed to install OTEL pipeline");

    tracing_opentelemetry::OpenTelemetryLayer::new(otel_tracer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tracing::info;

    #[test]
    fn log_hook_receives_record() {
        let captured: Arc<Mutex<Vec<RuntimeLogRecord>>> = Arc::new(Mutex::new(Vec::new()));
        let hook = {
            let c = captured.clone();
            move |rec: &RuntimeLogRecord| {
                c.lock().unwrap().push(rec.clone());
            }
        };
        set_runtime_log_hook(hook);
        let _guard = init_logging(LoggingOptions {
            enable_otlp: false,
            enable_traces: false,
            pid: 1,
        });

        info!("hello world");

        let items = captured.lock().unwrap();
        assert!(!items.is_empty());
        assert!(items.iter().any(|r| r.message.contains("hello world")));
    }

    #[test]
    fn trace_hook_receives_events() {
        let captured: Arc<Mutex<Vec<TraceEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let hook = {
            let c = captured.clone();
            move |events: &[TraceEvent]| {
                c.lock().unwrap().extend_from_slice(events);
            }
        };
        set_trace_hook(hook);
        let _guard = init_logging(LoggingOptions {
            enable_otlp: false,
            enable_traces: true,
            pid: 1,
        });

        let span = tracing::info_span!("test_span");
        let _enter = span.enter();
        info!("inside span");

        let items = captured.lock().unwrap();
        assert!(!items.is_empty());
        assert!(items.iter().any(|e| e.name == "test_span"
            || e.message()
                .unwrap_or_else(|| "".to_string())
                .contains("inside span")));
    }

    // helper to get message from TraceEvent args if present
    impl TraceEvent {
        fn message(&self) -> Option<String> {
            if let Some(args) = &self.args {
                if let Some(obj) = args.as_object() {
                    if let Some(val) = obj.get("message") {
                        return val.as_str().map(|s| s.to_string());
                    }
                }
            }
            None
        }
    }
}
