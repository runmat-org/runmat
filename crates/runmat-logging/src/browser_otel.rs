//! Event-driven browser OTLP HTTP/protobuf transport. Credentials are runtime-only.
use crate::{
    export_model::{
        self as model, Counters, Record, Signal, FLUSH_TIMEOUT_MS, MAX_BATCH_BYTES,
        MAX_RESPONSE_BYTES,
    },
    otel::{OtelOptions, OtelProtocol, OtelSignal, SignalOptions, StartupHeaders},
    RuntimeLogRecord,
};
use futures_channel::{mpsc, oneshot};
use futures_util::{
    future::{select, Either},
    pin_mut, StreamExt,
};
pub use model::ExportStats as BrowserExportStats;
use once_cell::sync::OnceCell;
use opentelemetry::{trace::TracerProvider, Context};
use opentelemetry_proto::tonic::collector::{
    logs::v1::ExportLogsServiceResponse, trace::v1::ExportTraceServiceResponse,
};
use opentelemetry_sdk::{
    export::trace::SpanData,
    trace::{self, SpanProcessor},
    Resource,
};
use prost::Message;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

static EXPORTER: OnceCell<BrowserExporter> = OnceCell::new();
type Reply = oneshot::Sender<Result<BrowserExportStats, String>>;
enum Command {
    Record(Record),
    Flush(Reply),
    Probe(Reply),
    Shutdown(Reply),
}
struct BrowserExporter {
    startup_options: OtelOptions,
    startup_headers: StartupHeaders,
    sender: Arc<Mutex<mpsc::Sender<Command>>>,
    counters: Arc<Counters>,
    provider: trace::TracerProvider,
    resource: Resource,
    traces: bool,
    logs: bool,
    closed: AtomicBool,
}
#[derive(Clone)]
struct QueueProcessor {
    sender: Arc<Mutex<mpsc::Sender<Command>>>,
    counters: Arc<Counters>,
    resource: Resource,
    traces: bool,
}
impl std::fmt::Debug for QueueProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BrowserOtelQueueProcessor")
    }
}
impl SpanProcessor for QueueProcessor {
    fn on_start(&self, _: &mut trace::Span, _: &Context) {}
    fn on_end(&self, span: SpanData) {
        if !self.traces
            || EXPORTER
                .get()
                .is_none_or(|e| e.closed.load(Ordering::Acquire))
        {
            return;
        }
        if let Some(record) = model::span_record(span, &self.resource, &self.counters) {
            if send(&self.sender, Command::Record(record)).is_err() {
                self.counters.drop_record();
            }
        }
    }
    fn force_flush(&self) -> opentelemetry::trace::TraceResult<()> {
        Err(opentelemetry::trace::TraceError::Other(
            "await flush_browser_otel in browser hosts".into(),
        ))
    }
    fn shutdown(&mut self) -> opentelemetry::trace::TraceResult<()> {
        Err(opentelemetry::trace::TraceError::Other(
            "await shutdown_browser_otel in browser hosts".into(),
        ))
    }
}
fn send(sender: &Mutex<mpsc::Sender<Command>>, command: Command) -> Result<(), String> {
    sender
        .try_lock()
        .map_err(|_| "OpenTelemetry queue is contended")?
        .try_send(command)
        .map_err(|_| "OpenTelemetry queue is full or exporter has stopped".into())
}
fn warn_failure(counters: &Counters) {
    counters.failure(||web_sys::console::warn_1(&JsValue::from_str("RunMat: OpenTelemetry export failed; check collector CORS, network and authentication. Further failures are counted silently.")));
}
pub fn configure_browser_otel(options: OtelOptions, headers: StartupHeaders) -> Result<(), String> {
    if let Some(existing) = EXPORTER.get() {
        if !existing.closed.load(Ordering::Acquire)
            && existing.startup_options == options
            && existing.startup_headers == headers
        {
            return Ok(());
        }
        return Err(
            "OpenTelemetryRestartRequired: collector policy changed or exporter is shut down"
                .into(),
        );
    }
    model::validate_headers(&headers)?;
    let resource = model::config_resource(&options)?;
    if !options.enabled {
        return Ok(());
    }
    if options.protocol != OtelProtocol::HttpProtobuf {
        return Err("browsers support OTLP http/protobuf only, not gRPC".into());
    }
    if options.signals.contains(&OtelSignal::Metrics) {
        return Err("OpenTelemetry metrics are not supported".into());
    }
    if options.queue_capacity == 0 || options.queue_capacity > 4096 {
        return Err("OpenTelemetry queue capacity must be between 1 and 4096".into());
    }
    let traces = options.signals.contains(&OtelSignal::Traces);
    let logs = options.signals.contains(&OtelSignal::Logs);
    if !traces && !logs {
        return Err("select traces and/or logs for OpenTelemetry".into());
    }
    let trace_options = options.signal_options("TRACES", &headers)?;
    let log_options = options.signal_options("LOGS", &headers)?;
    for signal in [&trace_options, &log_options] {
        model::validate_signal(signal)?;
        let url = web_sys::Url::new(&signal.endpoint).map_err(|_| "invalid OTLP endpoint URL")?;
        if !matches!(url.protocol().as_str(), "http:" | "https:")
            || !url.username().is_empty()
            || !url.password().is_empty()
        {
            return Err("OTLP endpoint must be HTTP(S) without userinfo credentials".into());
        }
        let checked = web_sys::Headers::new().map_err(|_| "cannot create OTLP headers")?;
        for (key, value) in &signal.headers.0 {
            checked
                .set(key, value)
                .map_err(|_| "invalid OTLP startup header")?;
        }
    }
    // futures mpsc reserves one slot for its sole Sender; do not clone that Sender.
    let (sender, receiver) = mpsc::channel(options.queue_capacity - 1);
    let sender = Arc::new(Mutex::new(sender));
    let counters = Arc::new(Counters::default());
    let provider = trace::TracerProvider::builder()
        .with_config(model::span_config())
        .with_span_processor(QueueProcessor {
            sender: sender.clone(),
            counters: counters.clone(),
            resource: resource.clone(),
            traces,
        })
        .build();
    EXPORTER
        .set(BrowserExporter {
            startup_options: options,
            startup_headers: headers,
            sender,
            counters: counters.clone(),
            provider,
            resource: resource.clone(),
            traces,
            logs,
            closed: AtomicBool::new(false),
        })
        .map_err(|_| "OpenTelemetry is already configured")?;
    use tracing::instrument::WithSubscriber;
    wasm_bindgen_futures::spawn_local(
        worker(
            receiver,
            counters,
            resource,
            traces.then_some(trace_options),
            logs.then_some(log_options),
        )
        .with_subscriber(tracing::subscriber::NoSubscriber::default()),
    );
    Ok(())
}
pub fn browser_otel_layer<S>() -> Option<tracing_opentelemetry::OpenTelemetryLayer<S, trace::Tracer>>
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    EXPORTER.get().map(|exporter| {
        tracing_opentelemetry::layer().with_tracer(exporter.provider.tracer("runmat"))
    })
}
pub(crate) fn context_for(data: &mut tracing_opentelemetry::OtelData) -> Context {
    use tracing_opentelemetry::PreSampledTracer;
    EXPORTER
        .get()
        .map(|exporter| exporter.provider.tracer("runmat").sampled_context(data))
        .unwrap_or_default()
}
pub(crate) fn export_log(record: &RuntimeLogRecord, context: &Context) {
    let Some(exporter) = EXPORTER
        .get()
        .filter(|e| e.logs && !e.closed.load(Ordering::Acquire))
    else {
        return;
    };
    if let Some(record) = model::log_record(record, context, &exporter.resource, &exporter.counters)
    {
        if send(&exporter.sender, Command::Record(record)).is_err() {
            exporter.counters.drop_record();
        }
    }
}
pub fn browser_otel_stats() -> Option<BrowserExportStats> {
    EXPORTER.get().map(|e| e.counters.snapshot())
}
pub(crate) fn drop_export_record() {
    if let Some(exporter) = EXPORTER.get() {
        exporter.counters.drop_record();
    }
}
enum Operation {
    Flush,
    Probe,
    Shutdown,
}
async fn request(operation: Operation) -> Result<BrowserExportStats, String> {
    let exporter = EXPORTER
        .get()
        .ok_or("OpenTelemetry export is not enabled")?;
    let shutdown = matches!(operation, Operation::Shutdown);
    if shutdown {
        if exporter.closed.swap(true, Ordering::AcqRel) {
            return Err("OpenTelemetry exporter shutdown was already requested".into());
        }
    } else if exporter.closed.load(Ordering::Acquire) {
        return Err("OpenTelemetry exporter is shut down".into());
    }
    if matches!(operation, Operation::Probe) && (!exporter.traces || !exporter.logs) {
        return Err("the OpenTelemetry test requires traces and logs".into());
    }
    let (reply, receiver) = oneshot::channel();
    if let Err(error) = send(
        &exporter.sender,
        match operation {
            Operation::Flush => Command::Flush(reply),
            Operation::Probe => Command::Probe(reply),
            Operation::Shutdown => Command::Shutdown(reply),
        },
    ) {
        if shutdown {
            exporter.closed.store(false, Ordering::Release);
        }
        return Err(error);
    }
    let timer = gloo_timers::future::TimeoutFuture::new(FLUSH_TIMEOUT_MS);
    pin_mut!(receiver, timer);
    match select(receiver, timer).await {
        Either::Left((result, _)) => result
            .map_err(|_| "OpenTelemetry exporter stopped before acknowledgement".to_string())?,
        Either::Right(_) => Err("OpenTelemetry operation timed out".into()),
    }
}
pub async fn flush_browser_otel() -> Result<BrowserExportStats, String> {
    request(Operation::Flush).await
}
pub async fn shutdown_browser_otel() -> Result<BrowserExportStats, String> {
    request(Operation::Shutdown).await
}
pub async fn emit_test_browser_otel() -> Result<BrowserExportStats, String> {
    request(Operation::Probe).await
}

async fn fetch(options: &SignalOptions, body: Vec<u8>) -> Result<Vec<u8>, String> {
    let controller =
        web_sys::AbortController::new().map_err(|_| "cannot create OTLP request cancellation")?;
    let init = web_sys::RequestInit::new();
    init.set_method("POST");
    init.set_mode(web_sys::RequestMode::Cors);
    init.set_credentials(web_sys::RequestCredentials::Omit);
    init.set_redirect(web_sys::RequestRedirect::Error);
    init.set_signal(Some(&controller.signal()));
    let bytes = js_sys::Uint8Array::from(body.as_slice());
    init.set_body(&bytes);
    let request = web_sys::Request::new_with_str_and_init(&options.endpoint, &init)
        .map_err(|_| "cannot create OTLP browser request")?;
    request
        .headers()
        .set("content-type", "application/x-protobuf")
        .map_err(|_| "cannot set OTLP content type")?;
    for (key, value) in &options.headers.0 {
        request
            .headers()
            .set(key, value)
            .map_err(|_| "invalid OTLP startup header")?;
    }
    let perform = async {
        let global = js_sys::global();
        let function: js_sys::Function = js_sys::Reflect::get(&global, &JsValue::from_str("fetch"))
            .map_err(|_| "browser Fetch unavailable")?
            .dyn_into()
            .map_err(|_| "browser Fetch unavailable")?;
        let promise: js_sys::Promise = function
            .call1(&global, &request)
            .map_err(|_| "OTLP browser request failed; check collector CORS")?
            .dyn_into()
            .map_err(|_| "browser Fetch did not return a promise")?;
        let response: web_sys::Response = JsFuture::from(promise)
            .await
            .map_err(|_| "OTLP browser request failed; check collector CORS, network and TLS")?
            .dyn_into()
            .map_err(|_| "invalid OTLP browser response")?;
        if !response.ok() {
            return Err(format!("OTLP HTTP status {}", response.status()));
        }
        if response
            .headers()
            .get("content-length")
            .ok()
            .flatten()
            .and_then(|value| value.parse::<usize>().ok())
            .is_some_and(|size| size > MAX_RESPONSE_BYTES)
        {
            controller.abort();
            return Err("OTLP response exceeds size limit".into());
        }
        let Some(stream) = response.body() else {
            return Ok(Vec::new());
        };
        let reader: web_sys::ReadableStreamDefaultReader = stream
            .get_reader()
            .dyn_into()
            .map_err(|_| "cannot read OTLP response stream")?;
        let mut bytes = Vec::new();
        loop {
            let chunk = JsFuture::from(reader.read())
                .await
                .map_err(|_| "cannot read OTLP response stream")?;
            let done = js_sys::Reflect::get(&chunk, &JsValue::from_str("done"))
                .map_err(|_| "invalid OTLP stream response")?
                .as_bool()
                .ok_or("invalid OTLP stream completion")?;
            if done {
                break;
            }
            let value = js_sys::Reflect::get(&chunk, &JsValue::from_str("value"))
                .map_err(|_| "invalid OTLP response chunk")?;
            let chunk: js_sys::Uint8Array = value
                .dyn_into()
                .map_err(|_| "invalid OTLP response chunk")?;
            if bytes.len().saturating_add(chunk.length() as usize) > MAX_RESPONSE_BYTES {
                controller.abort();
                return Err("OTLP response exceeds size limit".into());
            }
            bytes.extend(chunk.to_vec());
        }
        Ok(bytes)
    };
    let timer = gloo_timers::future::TimeoutFuture::new(options.timeout.as_millis() as u32);
    pin_mut!(perform, timer);
    match select(perform, timer).await {
        Either::Left((result, _)) => result,
        Either::Right(_) => {
            controller.abort();
            Err("OTLP browser request timed out".into())
        }
    }
}
async fn export_records(
    records: &[Record],
    counters: &Counters,
    traces: Option<&SignalOptions>,
    logs: Option<&SignalOptions>,
) -> Result<(), String> {
    let mut error = None;
    for signal in [Signal::Traces, Signal::Logs] {
        let selected = records.iter().filter(|record| record.signal == signal);
        let length = selected.clone().map(|record| record.bytes.len()).sum();
        if length == 0 {
            continue;
        }
        let mut body = Vec::with_capacity(length);
        let mut count = 0;
        for record in selected {
            body.extend_from_slice(&record.bytes);
            count += 1;
        }
        let options = match signal {
            Signal::Traces => traces,
            Signal::Logs => logs,
        }
        .ok_or("OpenTelemetry signal is not enabled")?;
        let result = fetch(options, body).await.and_then(|body| match signal {
            Signal::Traces => {
                let response = ExportTraceServiceResponse::decode(body.as_slice())
                    .map_err(|_| "invalid OTLP traces response")?;
                response.partial_success.map_or(Ok(false), |p| {
                    model::partial_result(p.rejected_spans, !p.error_message.is_empty())
                })
            }
            Signal::Logs => {
                let response = ExportLogsServiceResponse::decode(body.as_slice())
                    .map_err(|_| "invalid OTLP logs response")?;
                response.partial_success.map_or(Ok(false), |p| {
                    model::partial_result(p.rejected_log_records, !p.error_message.is_empty())
                })
            }
        });
        match result {
            Ok(warning) => counters.accepted(signal, count, warning),
            Err(failure) => {
                warn_failure(counters);
                error = Some(failure);
            }
        }
    }
    error.map_or(Ok(()), Err)
}
async fn worker(
    mut receiver: mpsc::Receiver<Command>,
    counters: Arc<Counters>,
    resource: Resource,
    traces: Option<SignalOptions>,
    logs: Option<SignalOptions>,
) {
    let mut pending = None;
    loop {
        let command = match pending.take() {
            Some(command) => Some(command),
            None => receiver.next().await,
        };
        match command {
            Some(Command::Record(record)) => {
                let mut size = record.bytes.len();
                let mut records = vec![record];
                while records.len() < 256 {
                    match receiver.try_recv() {
                        Ok(Command::Record(record))
                            if size + record.bytes.len() <= MAX_BATCH_BYTES =>
                        {
                            size += record.bytes.len();
                            records.push(record);
                        }
                        Ok(command) => {
                            pending = Some(command);
                            break;
                        }
                        _ => break,
                    }
                }
                // Failures are durably counted; background export never fails a user's run.
                if export_records(&records, &counters, traces.as_ref(), logs.as_ref())
                    .await
                    .is_err()
                {}
            }
            Some(Command::Probe(reply)) => {
                if reply.is_canceled() {
                    continue;
                }
                let result = match model::probe_records(&resource, &counters) {
                    Ok((span, log)) => {
                        let records = [span, log];
                        let result =
                            export_records(&records, &counters, traces.as_ref(), logs.as_ref())
                                .await;
                        drop(records);
                        result.map(|_| counters.snapshot())
                    }
                    Err(error) => Err(error),
                };
                let _ = reply.send(result);
            }
            Some(Command::Flush(reply)) => {
                let _ = reply.send(counters.flush_result());
            }
            Some(Command::Shutdown(reply)) => {
                let _ = reply.send(counters.flush_result());
                break;
            }
            None => break,
        }
    }
}
