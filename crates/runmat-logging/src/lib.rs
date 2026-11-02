use once_cell::sync::OnceCell;
use serde::Serialize;
use serde_json::Value as JsonValue;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::subscriber::DefaultGuard;
use tracing::Subscriber;
use tracing_log::LogTracer;
#[cfg(any(feature = "otlp", feature = "browser-otlp"))]
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::Layer;

pub mod otel;
#[cfg(not(target_arch = "wasm32"))]
mod file_sink;
#[cfg(any(feature = "otlp", feature = "browser-otlp"))]
mod export_model;
#[cfg(all(feature = "browser-otlp", target_arch = "wasm32"))]
pub mod browser_otel;
#[cfg(all(feature = "otlp", not(target_arch = "wasm32")))]
mod native_otel;
#[cfg(all(feature = "otlp", not(target_arch = "wasm32")))]
pub use native_otel::{
    configure_otel, emit_test_otel, flush_otel, otel_layer, otel_stats, retained_otel_layer,
    shutdown_otel_result, ExportStats,
};

#[cfg(not(target_arch = "wasm32"))]
use tracing_subscriber::reload;

#[cfg(all(
    not(target_arch = "wasm32"),
    not(any(feature = "otlp", feature = "browser-otlp"))
))]
use std::time::SystemTime;

#[cfg(any(feature = "otlp", feature = "browser-otlp"))]
use opentelemetry::trace::TraceContextExt;
#[cfg(any(feature = "otlp", feature = "browser-otlp"))]
use opentelemetry::trace::{SpanContext, SpanId, TraceFlags, TraceId, TraceState};
#[cfg(any(feature = "otlp", feature = "browser-otlp"))]
use opentelemetry::Context as OtelContext;

const DEFAULT_LOG_FILTER: &str = "info";

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
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
#[serde(rename_all = "camelCase")]
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
    pub parent_span_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<JsonValue>,
}

type LogHook = Arc<dyn Fn(&RuntimeLogRecord) + Send + Sync>;
type TraceHook = Arc<dyn Fn(&[TraceEvent]) + Send + Sync>;

static LOG_HOOK: OnceCell<LogHook> = OnceCell::new();
static TRACE_HOOK: OnceCell<TraceHook> = OnceCell::new();
#[cfg(not(any(feature = "otlp", feature = "browser-otlp")))]
static FALLBACK_TRACE_ID: OnceCell<String> = OnceCell::new();
static EVENT_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);
#[cfg(not(target_arch = "wasm32"))]
type LogFilterReloader = Box<dyn Fn(&str) -> Result<(), String> + Send + Sync>;

/// Reloads the filter of the subscriber that is actually installed process-wide.
#[cfg(not(target_arch = "wasm32"))]
static LOG_FILTER_RELOADER: OnceCell<LogFilterReloader> = OnceCell::new();

pub struct LoggingGuard {
    _guard: Option<DefaultGuard>,
    #[cfg(not(target_arch = "wasm32"))]
    file_sink: Option<file_sink::FileSink>,
}

impl LoggingGuard {
    /// A guard for hosts that installed their own global subscriber.
    pub fn host_managed() -> Self {
        Self {
            _guard: None,
            #[cfg(not(target_arch = "wasm32"))]
            file_sink: None,
        }
    }

    /// Reports a latched configured-file error even if the final flush succeeds.
    pub fn flush_file_result(&self) -> Result<(), String> {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(sink) = &self.file_sink {
            return sink.flush_result();
        }
        Ok(())
    }
}

impl Drop for LoggingGuard {
    fn drop(&mut self) {
        // Owned file sinks warn once themselves; legacy/no-file guards are
        // unchanged. Explicit hosts use flush_file_result to report failure.
        let _ = self.flush_file_result();
    }
}

/// Lets a host that owns the global subscriber serve `update_log_filter`.
/// The first registration wins.
#[cfg(not(target_arch = "wasm32"))]
pub fn set_log_filter_reloader<F>(reloader: F)
where
    F: Fn(&str) -> Result<(), String> + Send + Sync + 'static,
{
    let _ = LOG_FILTER_RELOADER.set(Box::new(reloader));
}

/// The layer that forwards events to the runtime log and trace hooks, for hosts
/// that build their own subscriber instead of calling [`init_logging`].
pub fn runtime_bridge_layer<S>(enable_traces: bool, pid: i64) -> impl Layer<S>
where
    S: Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    LogBridgeLayer.and_then(enable_traces.then_some(TraceBridgeLayer { pid }))
}

#[derive(Clone, Default)]
pub struct LoggingOptions {
    pub enable_otlp: bool,
    pub enable_traces: bool,
    pub pid: i64,
    pub default_filter: Option<String>,
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
    init_logging_impl(
        opts,
        false,
        #[cfg(not(target_arch = "wasm32"))]
        None,
        false,
    )
    .expect("non-strict logging always retains its thread-local fallback")
}

/// A host that owns `log::Log` forwards records explicitly through [`forward_log_record`].
/// Do not compete with that owner by attempting to install another global logger.
pub fn init_logging_with_host_logger(opts: LoggingOptions) -> LoggingGuard {
    init_logging_impl(
        opts,
        true,
        #[cfg(not(target_arch = "wasm32"))]
        None,
        false,
    )
    .expect("non-strict logging always retains its thread-local fallback")
}

/// Add one owned native file sink to the existing subscriber. Unlike legacy
/// initializers, this rejects an occupied global subscriber without installing
/// a competing thread-local subscriber. The host still owns `log::Log`.
#[cfg(not(target_arch = "wasm32"))]
pub fn init_logging_with_host_logger_and_file(
    opts: LoggingOptions,
    file: std::fs::File,
) -> Result<LoggingGuard, String> {
    init_logging_impl(opts, true, Some(file_sink::FileSink::new(file)), true)
}

pub fn log_metadata_enabled(metadata: &tracing_log::log::Metadata<'_>) -> bool {
    use tracing_log::AsTrace;
    tracing::dispatcher::get_default(|dispatch| dispatch.enabled(&metadata.as_trace()))
}

pub fn forward_log_record(record: &tracing_log::log::Record<'_>) -> std::io::Result<()> {
    tracing_log::format_trace(record)
}

fn init_logging_impl(
    opts: LoggingOptions,
    host_logger: bool,
    #[cfg(not(target_arch = "wasm32"))] file_sink: Option<file_sink::FileSink>,
    global_only: bool,
) -> Result<LoggingGuard, String> {
    if !host_logger {
        // Hosts with an existing log owner must use init_logging_with_host_logger.
        if LogTracer::init().is_err() {
            eprintln!("RunMat: log bridge was not installed because the host already owns log::Log; forward records explicitly");
        }
    }

    let fallback_filter = opts.default_filter.as_deref().unwrap_or(DEFAULT_LOG_FILTER);

    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_from_env("RUNMAT_LOG"))
        .or_else(|_| EnvFilter::try_new(fallback_filter))
        .unwrap_or_else(|_| EnvFilter::new(fallback_filter));

    let build_subscriber = || {
        let bridge_layer = LogBridgeLayer;
        let trace_layer = if opts.enable_traces {
            Some(TraceBridgeLayer { pid: opts.pid })
        } else {
            None
        };

        #[cfg(not(target_arch = "wasm32"))]
        {
            let (filter_layer, filter_handle) = reload::Layer::new(env_filter.clone());
            let subscriber = tracing_subscriber::registry()
                .with(filter_layer)
                .with(file_sink.as_ref().map(|sink| {
                    tracing_subscriber::fmt::layer()
                        .with_ansi(false)
                        .with_writer(sink.clone())
                        .log_internal_errors(false)
                        .with_span_events(
                            tracing_subscriber::fmt::format::FmtSpan::NEW
                                | tracing_subscriber::fmt::format::FmtSpan::CLOSE,
                        )
                }))
                .with(bridge_layer)
                .with(trace_layer.clone());
            (subscriber, Some(filter_handle))
        }

        #[cfg(target_arch = "wasm32")]
        {
            let subscriber = tracing_subscriber::registry()
                .with(env_filter.clone())
                .with(bridge_layer)
                .with(trace_layer.clone());
            (subscriber, None::<()>)
        }
    };

    let (subscriber, filter_handle) = build_subscriber();

    #[cfg(all(feature = "otlp", not(target_arch = "wasm32")))]
    let subscriber = { subscriber.with(opts.enable_otlp.then(otel_layer).flatten()) };
    #[cfg(all(feature = "browser-otlp", target_arch = "wasm32"))]
    let subscriber = subscriber.with(
        opts.enable_otlp
            .then(browser_otel::browser_otel_layer)
            .flatten(),
    );

    let guard = match tracing::subscriber::set_global_default(subscriber) {
        Ok(()) => {
            // Only the installed subscriber's handle is live; a dropped one reports
            // "subscriber no longer exists" on every reload.
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(handle) = filter_handle {
                set_log_filter_reloader(move |spec| {
                    let filter = EnvFilter::try_new(spec).map_err(|err| err.to_string())?;
                    handle.reload(filter).map_err(|err| err.to_string())
                });
            }
            #[cfg(target_arch = "wasm32")]
            let _ = filter_handle;
            None
        }
        Err(_) if global_only => return Err("Another tracing subscriber is already installed; configured file logging was not activated".into()),
        Err(_) => {
            let subscriber = build_subscriber().0;
            #[cfg(all(feature = "otlp", not(target_arch = "wasm32")))]
            let subscriber = subscriber.with(opts.enable_otlp.then(otel_layer).flatten());
            #[cfg(all(feature = "browser-otlp", target_arch = "wasm32"))]
            let subscriber = subscriber.with(
                opts.enable_otlp
                    .then(browser_otel::browser_otel_layer)
                    .flatten(),
            );
            Some(tracing::subscriber::set_default(subscriber))
        }
    };

    Ok(LoggingGuard {
        _guard: guard,
        #[cfg(not(target_arch = "wasm32"))]
        file_sink,
    })
}

/// Flushes and shuts down the process-global OpenTelemetry trace provider.
///
/// Short-lived native hosts call this before exiting so their final spans are not left in the
/// batch exporter's queue.
#[cfg(all(feature = "otlp", not(target_arch = "wasm32")))]
pub fn shutdown_otel() {
    if otel_stats().is_some() && shutdown_otel_result().is_err() {
        native_otel::record_shutdown_failure();
    }
}

#[cfg(any(not(feature = "otlp"), target_arch = "wasm32"))]
pub fn shutdown_otel() {}

#[cfg(not(target_arch = "wasm32"))]
pub fn update_log_filter(spec: &str) -> Result<(), String> {
    let reload = LOG_FILTER_RELOADER
        .get()
        .ok_or_else(|| "log filter handle not initialised".to_string())?;
    reload(spec)
}

#[cfg(target_arch = "wasm32")]
pub fn update_log_filter(_spec: &str) -> Result<(), String> {
    Err("runtime log filtering is not yet supported in wasm builds".to_string())
}

pub fn with_signal_trace<T>(trace_id: Option<&str>, name: &str, f: impl FnOnce() -> T) -> T {
    let Some(trace_id) = trace_id else {
        return f();
    };
    if let Some(span) = build_signal_span(trace_id, name) {
        let _guard = span.enter();
        return f();
    }
    f()
}

fn build_signal_span(trace_id: &str, name: &str) -> Option<tracing::Span> {
    let span = tracing::span!(
        tracing::Level::INFO,
        "signal",
        signal = name,
        trace_id = trace_id
    );
    #[cfg(any(feature = "otlp", feature = "browser-otlp"))]
    {
        if let Some((trace_id, span_id)) = parse_trace_parent(trace_id) {
            let context = SpanContext::new(
                trace_id,
                span_id,
                TraceFlags::SAMPLED,
                true,
                TraceState::default(),
            );
            span.set_parent(OtelContext::new().with_remote_span_context(context));
        }
    }
    Some(span)
}

#[cfg(any(feature = "otlp", feature = "browser-otlp"))]
fn parse_trace_parent(trace_id: &str) -> Option<(TraceId, SpanId)> {
    if trace_id.len() != 32 {
        return None;
    }
    let trace_id = TraceId::from_hex(trace_id).ok()?;
    let span_id = SpanId::from_hex(&trace_id_hex_tail(trace_id)).ok()?;
    Some((trace_id, span_id))
}

#[cfg(any(feature = "otlp", feature = "browser-otlp"))]
fn trace_id_hex_tail(trace_id: TraceId) -> String {
    let hex = trace_id.to_string();
    hex.chars()
        .rev()
        .take(16)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
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
    S: Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    fn on_event(&self, event: &tracing::Event<'_>, ctx: tracing_subscriber::layer::Context<'_, S>) {
        let mut visitor = JsonVisitor::default();
        event.record(&mut visitor);
        let oversized = visitor.overflow;
        #[cfg(all(feature = "otlp", not(target_arch = "wasm32")))]
        let (otel_context, span_started) = event_otel_context(event, &ctx);
        #[cfg(all(
            any(feature = "otlp", feature = "browser-otlp"),
            any(target_arch = "wasm32", not(feature = "otlp"))
        ))]
        let (otel_context, _) = event_otel_context(event, &ctx);
        #[cfg(any(feature = "otlp", feature = "browser-otlp"))]
        let (trace_id, span_id) = valid_context_ids(&otel_context);
        #[cfg(not(any(feature = "otlp", feature = "browser-otlp")))]
        let (trace_id, span_id) = {
            let _ = &ctx;
            current_trace_span_ids()
        };

        let record = RuntimeLogRecord {
            ts: now_rfc3339(),
            level: event.metadata().level().to_string(),
            target: event.metadata().target().to_string(),
            message: visitor
                .message
                .unwrap_or_else(|| event.metadata().name().to_string()),
            trace_id,
            span_id,
            fields: visitor
                .fields
                .and_then(|v| v.as_object().cloned().map(JsonValue::Object))
                .filter(|obj| obj.as_object().map(|m| !m.is_empty()).unwrap_or(false)),
        };

        if let Some(hook) = LOG_HOOK.get() {
            hook(&record);
        }
        #[cfg(all(feature = "otlp", not(target_arch = "wasm32")))]
        if oversized {
            native_otel::drop_export_record();
        } else {
            native_otel::export_log(&record, &otel_context, span_started);
        }
        #[cfg(all(feature = "browser-otlp", target_arch = "wasm32"))]
        if oversized {
            browser_otel::drop_export_record();
        } else {
            browser_otel::export_log(&record, &otel_context);
        }
        #[cfg(not(any(feature = "otlp", feature = "browser-otlp")))]
        let _ = oversized;
    }
}

#[cfg(any(feature = "otlp", feature = "browser-otlp"))]
fn valid_context_ids(context: &OtelContext) -> (Option<String>, Option<String>) {
    let span = context.span();
    let identity = span.span_context();
    if identity.is_valid() {
        (
            Some(identity.trace_id().to_string()),
            Some(identity.span_id().to_string()),
        )
    } else {
        (None, None)
    }
}

#[cfg(any(feature = "otlp", feature = "browser-otlp"))]
fn event_otel_context<S>(
    event: &tracing::Event<'_>,
    ctx: &tracing_subscriber::layer::Context<'_, S>,
) -> (OtelContext, Option<std::time::SystemTime>)
where
    S: Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    let Some(span) = ctx.event_span(event) else {
        return (OtelContext::new(), None);
    };
    let mut extensions = span.extensions_mut();
    let Some(data) = extensions.get_mut::<tracing_opentelemetry::OtelData>() else {
        return (OtelContext::new(), None);
    };
    let started = data.builder.start_time;
    #[cfg(all(feature = "otlp", not(target_arch = "wasm32")))]
    return (native_otel::context_for(data), started);
    #[cfg(all(feature = "browser-otlp", target_arch = "wasm32"))]
    return (browser_otel::context_for(data), started);
    #[cfg(not(any(
        all(feature = "otlp", not(target_arch = "wasm32")),
        all(feature = "browser-otlp", target_arch = "wasm32")
    )))]
    (OtelContext::new(), started)
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
        let parent_span_id = current_span_id();
        let span_id = Some(next_event_span_id());

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
            parent_span_id,
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
    let span_id = Some(span.id().clone().into_u64().to_string());
    let parent_span_id = span
        .parent()
        .map(|parent| parent.id().clone().into_u64().to_string());

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
        parent_span_id,
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

#[cfg(any(feature = "otlp", feature = "browser-otlp"))]
fn current_trace_span_ids() -> (Option<String>, Option<String>) {
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
    (None, None)
}

#[cfg(not(any(feature = "otlp", feature = "browser-otlp")))]
fn current_trace_span_ids() -> (Option<String>, Option<String>) {
    let span_id = tracing::Span::current()
        .id()
        .map(|id| id.into_u64().to_string());
    let trace_id = Some(fallback_trace_id());
    (trace_id, span_id)
}

#[cfg(not(any(feature = "otlp", feature = "browser-otlp")))]
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

fn next_event_span_id() -> String {
    let id = EVENT_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("ev-{id}")
}

#[derive(Default)]
struct JsonVisitor {
    message: Option<String>,
    fields: Option<JsonValue>,
    overflow: bool,
    bytes: usize,
}

struct BoundedText {
    value: String,
    limit: usize,
    overflow: bool,
}
impl std::fmt::Write for BoundedText {
    fn write_str(&mut self, value: &str) -> std::fmt::Result {
        let remaining = self.limit.saturating_sub(self.value.len());
        if value.len() <= remaining {
            self.value.push_str(value);
            return Ok(());
        }
        let mut end = remaining;
        while !value.is_char_boundary(end) {
            end -= 1;
        }
        self.value.push_str(&value[..end]);
        self.overflow = true;
        Err(std::fmt::Error)
    }
}

impl tracing::field::Visit for JsonVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        use std::fmt::Write;
        let limit = if field.name() == "message" {
            8192
        } else {
            4096
        };
        let mut text = BoundedText {
            value: String::new(),
            limit,
            overflow: false,
        };
        if write!(&mut text, "{value:?}").is_err() {
            text.overflow = true;
        }
        self.overflow |= text.overflow;
        self.record_text(field, text.value);
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        use std::fmt::Write;
        let mut text = BoundedText {
            value: String::new(),
            limit: if field.name() == "message" {
                8192
            } else {
                4096
            },
            overflow: false,
        };
        if text.write_str(value).is_err() {
            text.overflow = true;
        }
        self.overflow |= text.overflow;
        self.record_text(field, text.value);
    }
}

impl JsonVisitor {
    fn record_text(&mut self, field: &tracing::field::Field, mut value: String) {
        if self.overflow {
            value.push_str(" [truncated]");
        }
        if self
            .bytes
            .saturating_add(value.len())
            .saturating_add(field.name().len())
            > 32 * 1024
            || field.name().len() > 256
        {
            self.overflow = true;
            return;
        }
        self.bytes += value.len() + field.name().len();
        if field.name() == "message" {
            self.message = Some(value);
        } else {
            let fields = self
                .fields
                .get_or_insert_with(|| JsonValue::Object(Default::default()));
            if let JsonValue::Object(map) = fields {
                if map.len() >= 32 {
                    self.overflow = true;
                    return;
                }
                map.insert(field.name().to_owned(), JsonValue::String(value));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tracing::warn;

    #[cfg(all(feature = "otlp", not(target_arch = "wasm32")))]
    #[test]
    fn permanent_layer_resolves_explicit_parent_without_phantom_root_ids() {
        type CapturedContextIds = Vec<(Option<String>, Option<String>)>;
        struct Capture(Arc<Mutex<CapturedContextIds>>);
        impl<S> Layer<S> for Capture
        where
            S: Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
        {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                ctx: tracing_subscriber::layer::Context<'_, S>,
            ) {
                self.0
                    .lock()
                    .unwrap()
                    .push(valid_context_ids(&event_otel_context(event, &ctx).0));
            }
        }
        let ids = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry()
            .with(retained_otel_layer())
            .with(Capture(ids.clone()));
        tracing::subscriber::with_default(subscriber, || {
            let parent = tracing::info_span!("explicit_parent");
            let expected = valid_context_ids(&parent.context());
            tracing::info!(parent:&parent,"explicit parent, not entered");
            tracing::info!(parent:None,"unparented event");
            let actual = ids.lock().unwrap();
            assert_eq!(actual[0], expected);
            assert!(actual[0].0.is_some());
            assert_eq!(actual[1], (None, None));
        });
    }

    #[test]
    fn host_log_owner_forwards_original_records_into_real_subscriber() {
        struct Capture(Arc<Mutex<Vec<String>>>);
        impl<S: Subscriber> Layer<S> for Capture {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                _: tracing_subscriber::layer::Context<'_, S>,
            ) {
                let mut visitor = JsonVisitor::default();
                event.record(&mut visitor);
                self.0
                    .lock()
                    .unwrap()
                    .push(visitor.message.unwrap_or_default());
            }
        }
        let captured = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry()
            .with(EnvFilter::new("debug"))
            .with(Capture(captured.clone()));
        tracing::subscriber::with_default(subscriber, || {
            let metadata = tracing_log::log::Metadata::builder()
                .level(tracing_log::log::Level::Debug)
                .target("bridge_fixture")
                .build();
            assert!(
                log_metadata_enabled(&metadata),
                "subscriber-only debug must be enabled independently of the terminal sink"
            );
            forward_log_record(
                &tracing_log::log::Record::builder()
                    .metadata(metadata)
                    .args(format_args!("original bridge message"))
                    .build(),
            )
            .unwrap();
        });
        assert_eq!(*captured.lock().unwrap(), vec!["original bridge message"]);
    }

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
            default_filter: None,
        });

        warn!("hello world");

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
            default_filter: None,
        });

        let span = tracing::warn_span!("test_span");
        let _enter = span.enter();
        warn!("inside span");

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
