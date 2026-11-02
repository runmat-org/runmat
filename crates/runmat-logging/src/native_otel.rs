//! One retained SDK layer and one bounded process-owned export worker.
use crate::{
    export_model::{
        self as model, Counters, Record, Signal, FLUSH_TIMEOUT_MS, MAX_BATCH_BYTES,
        MAX_RESPONSE_BYTES,
    },
    otel::{OtelOptions, OtelProtocol, OtelSignal, SignalOptions, StartupHeaders},
    RuntimeLogRecord,
};
pub use model::ExportStats;
use once_cell::sync::OnceCell;
use opentelemetry::{trace::TracerProvider, Context};
use opentelemetry_proto::tonic::collector::{
    logs::v1::{
        logs_service_client::LogsServiceClient, ExportLogsServiceRequest, ExportLogsServiceResponse,
    },
    trace::v1::{
        trace_service_client::TraceServiceClient, ExportTraceServiceRequest,
        ExportTraceServiceResponse,
    },
};
use opentelemetry_sdk::{
    export::trace::SpanData,
    trace::{self, SpanProcessor},
    Resource,
};
use prost::Message;
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    time::{Duration, Instant, SystemTime},
};
use tonic::transport::{Channel, ClientTlsConfig, Endpoint};

static EXPORTER: OnceCell<Exporter> = OnceCell::new();
static PROVIDER: OnceCell<trace::TracerProvider> = OnceCell::new();
static SELECTION: Mutex<Option<Selection>> = Mutex::new(None);
#[derive(PartialEq, Eq)]
struct Selection {
    options: OtelOptions,
    headers: StartupHeaders,
    traces: Option<SignalOptions>,
    logs: Option<SignalOptions>,
}
type Reply = mpsc::SyncSender<Result<ExportStats, String>>;
enum Command {
    Record(Record),
    Flush(Reply),
    Probe { reply: Reply, deadline: Instant },
    Shutdown(Reply),
}
struct Exporter {
    sender: mpsc::SyncSender<Command>,
    counters: Arc<Counters>,
    resource: Resource,
    traces: bool,
    logs: bool,
    closed: AtomicBool,
    selected_at: SystemTime,
}
#[derive(Debug)]
struct DeferredProcessor;
impl SpanProcessor for DeferredProcessor {
    fn on_start(&self, _: &mut trace::Span, _: &Context) {}
    fn on_end(&self, span: SpanData) {
        let Some(exporter) = EXPORTER
            .get()
            .filter(|e| e.traces && !e.closed.load(Ordering::Acquire))
        else {
            return;
        };
        // tracing-opentelemetry builds the SDK span on close. Its original start_time,
        // unlike SDK on_start invocation time, fences out pre-selection tracing spans.
        if span.start_time < exporter.selected_at {
            return;
        }
        if let Some(record) = model::span_record(span, &exporter.resource, &exporter.counters) {
            enqueue(exporter, Command::Record(record));
        }
    }
    fn force_flush(&self) -> opentelemetry::trace::TraceResult<()> {
        flush_otel()
            .map(|_| ())
            .map_err(|e| opentelemetry::trace::TraceError::Other(e.into()))
    }
    fn shutdown(&mut self) -> opentelemetry::trace::TraceResult<()> {
        shutdown_otel_result()
            .map(|_| ())
            .map_err(|e| opentelemetry::trace::TraceError::Other(e.into()))
    }
}
fn provider() -> &'static trace::TracerProvider {
    PROVIDER.get_or_init(|| {
        trace::TracerProvider::builder()
            .with_config(model::span_config())
            .with_span_processor(DeferredProcessor)
            .build()
    })
}
/// Permanently attach this layer when a product host installs its subscriber, before a project
/// selects user export. Never wrap this layer in reload::Layer (WithContext must remain downcastable).
pub fn retained_otel_layer<S>() -> tracing_opentelemetry::OpenTelemetryLayer<S, trace::Tracer>
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    tracing_opentelemetry::layer().with_tracer(provider().tracer("runmat"))
}
pub fn otel_layer<S>() -> Option<tracing_opentelemetry::OpenTelemetryLayer<S, trace::Tracer>>
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    EXPORTER.get().map(|_| retained_otel_layer())
}
pub(crate) fn context_for(data: &mut tracing_opentelemetry::OtelData) -> Context {
    use tracing_opentelemetry::PreSampledTracer;
    provider().tracer("runmat").sampled_context(data)
}
pub fn configure_otel(options: OtelOptions, headers: StartupHeaders) -> Result<(), String> {
    let mut selection = SELECTION
        .lock()
        .map_err(|_| "OpenTelemetry startup lock poisoned")?;
    model::validate_headers(&headers)?;
    let resource = model::config_resource(&options)?;
    let traces = options.enabled && options.signals.contains(&OtelSignal::Traces);
    let logs = options.enabled && options.signals.contains(&OtelSignal::Logs);
    if options.enabled && (options.signals.contains(&OtelSignal::Metrics) || (!traces && !logs)) {
        return Err("OpenTelemetry supports traces and/or logs, not metrics".into());
    }
    if options.queue_capacity == 0 || options.queue_capacity > 4096 {
        return Err("OpenTelemetry queue capacity must be between 1 and 4096".into());
    }
    let trace_options = traces
        .then(|| options.signal_options("TRACES", &headers))
        .transpose()?;
    let log_options = logs
        .then(|| options.signal_options("LOGS", &headers))
        .transpose()?;
    let requested = Selection {
        options,
        headers,
        traces: trace_options.clone(),
        logs: log_options.clone(),
    };
    if let Some(existing) = selection.as_ref() {
        if existing == &requested
            && EXPORTER
                .get()
                .is_none_or(|e| !e.closed.load(Ordering::Acquire))
        {
            return Ok(());
        }
        return Err("OpenTelemetryRestartRequired: process collector policy changed or exporter is shut down".into());
    }
    if !requested.options.enabled {
        *selection = Some(requested);
        return Ok(());
    }
    let (sender, receiver) = mpsc::sync_channel(requested.options.queue_capacity);
    let counters = Arc::new(Counters::default());
    let worker_counters = counters.clone();
    let worker_resource = resource.clone();
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name("runmat-otel".into())
        .spawn(move || {
            let _diagnostics =
                tracing::subscriber::set_default(tracing::subscriber::NoSubscriber::default());
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(_) => {
                    let _ = ready_tx.send(Err("cannot start OpenTelemetry I/O runtime".into()));
                    return;
                }
            };
            let transports = {
                let _enter = runtime.enter();
                Transport::pair(trace_options, log_options)
            };
            match transports {
                Ok(transports) => {
                    let _ = ready_tx.send(Ok(()));
                    worker(
                        runtime,
                        receiver,
                        worker_counters,
                        worker_resource,
                        transports,
                    );
                }
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                }
            }
        })
        .map_err(|_| "cannot start OpenTelemetry export thread")?;
    ready_rx
        .recv_timeout(Duration::from_secs(10))
        .map_err(|_| "OpenTelemetry startup timed out")??;
    EXPORTER
        .set(Exporter {
            sender,
            counters,
            resource,
            traces,
            logs,
            closed: AtomicBool::new(false),
            selected_at: opentelemetry::time::now(),
        })
        .map_err(|_| "OpenTelemetry is already configured")?;
    *selection = Some(requested);
    Ok(())
}
fn warn_failure(counters: &Counters) {
    counters.failure(|| {
        eprintln!("RunMat: OpenTelemetry export failed; subsequent failures are counted silently")
    });
}
fn enqueue(exporter: &Exporter, command: Command) {
    if exporter.sender.try_send(command).is_err() {
        exporter.counters.drop_record();
    }
}
pub(crate) fn export_log(record: &RuntimeLogRecord, context: &Context, born: Option<SystemTime>) {
    let Some(exporter) = EXPORTER
        .get()
        .filter(|e| e.logs && !e.closed.load(Ordering::Acquire))
    else {
        return;
    };
    if born.is_some_and(|time| time < exporter.selected_at) {
        return;
    }
    if let Some(record) = model::log_record(record, context, &exporter.resource, &exporter.counters)
    {
        enqueue(exporter, Command::Record(record));
    }
}
pub fn otel_stats() -> Option<ExportStats> {
    EXPORTER.get().map(|e| e.counters.snapshot())
}
pub(crate) fn drop_export_record() {
    if let Some(exporter) = EXPORTER.get() {
        exporter.counters.drop_record();
    }
}
pub(crate) fn record_shutdown_failure() {
    if let Some(exporter) = EXPORTER.get() {
        warn_failure(&exporter.counters);
    }
}
enum Operation {
    Flush,
    Probe,
    Shutdown,
}
fn request(operation: Operation) -> Result<ExportStats, String> {
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
    let deadline = Instant::now() + Duration::from_millis(u64::from(FLUSH_TIMEOUT_MS));
    let (reply, receiver) = mpsc::sync_channel(1);
    let mut command = match operation {
        Operation::Flush => Command::Flush(reply),
        Operation::Probe => Command::Probe { reply, deadline },
        Operation::Shutdown => Command::Shutdown(reply),
    };
    loop {
        match exporter.sender.try_send(command) {
            Ok(()) => break,
            Err(mpsc::TrySendError::Full(returned)) if Instant::now() < deadline => {
                command = returned;
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(_) => {
                if shutdown {
                    exporter.closed.store(false, Ordering::Release);
                }
                return Err("OpenTelemetry queue is full or exporter has stopped".into());
            }
        }
    }
    receiver
        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .map_err(|_| "OpenTelemetry operation timed out".to_string())?
}
pub fn flush_otel() -> Result<ExportStats, String> {
    request(Operation::Flush)
}
pub fn shutdown_otel_result() -> Result<ExportStats, String> {
    request(Operation::Shutdown)
}
/// One FIFO probe owns its own span/log response; concurrent flushes cannot consume its failure.
pub fn emit_test_otel() -> Result<ExportStats, String> {
    request(Operation::Probe)
}

enum Transport {
    Http {
        client: reqwest::Client,
        options: SignalOptions,
    },
    Grpc {
        channel: Channel,
        options: SignalOptions,
    },
}
impl Transport {
    fn pair(
        traces: Option<SignalOptions>,
        logs: Option<SignalOptions>,
    ) -> Result<(Option<Self>, Option<Self>), String> {
        Ok((
            traces.map(Self::new).transpose()?,
            logs.map(Self::new).transpose()?,
        ))
    }
    fn new(options: SignalOptions) -> Result<Self, String> {
        let uri =
            reqwest::Url::parse(&options.endpoint).map_err(|_| "invalid OTLP endpoint URL")?;
        if !matches!(uri.scheme(), "http" | "https")
            || !uri.username().is_empty()
            || uri.password().is_some()
        {
            return Err("OTLP endpoint must be HTTP(S) without userinfo credentials".into());
        }
        model::validate_signal(&options)?;
        for (key, value) in &options.headers.0 {
            reqwest::header::HeaderName::from_bytes(key.as_bytes())
                .map_err(|_| "invalid OTLP header name")?;
            reqwest::header::HeaderValue::from_str(value)
                .map_err(|_| "invalid OTLP header value")?;
        }
        match options.protocol {
            OtelProtocol::HttpProtobuf => Ok(Self::Http {
                client: reqwest::Client::builder()
                    .timeout(options.timeout)
                    .redirect(reqwest::redirect::Policy::none())
                    .build()
                    .map_err(|_| "cannot create OTLP HTTP client")?,
                options,
            }),
            OtelProtocol::Grpc => {
                metadata(&options.headers)?;
                let mut endpoint = Endpoint::from_shared(options.endpoint.clone())
                    .map_err(|_| "invalid OTLP gRPC endpoint")?
                    .timeout(options.timeout)
                    .connect_timeout(options.timeout);
                if uri.scheme() == "https" {
                    endpoint = endpoint
                        .tls_config(ClientTlsConfig::new())
                        .map_err(|_| "cannot configure OTLP TLS")?;
                }
                Ok(Self::Grpc {
                    channel: endpoint.connect_lazy(),
                    options,
                })
            }
        }
    }
    async fn send(&mut self, signal: Signal, body: Vec<u8>) -> Result<bool, String> {
        match self {
            Self::Http { client, options } => {
                let response = http_export(client, options, body).await?;
                match signal {
                    Signal::Traces => {
                        let response = ExportTraceServiceResponse::decode(response.as_slice())
                            .map_err(|_| "invalid OTLP traces response")?;
                        response.partial_success.map_or(Ok(false), |p| {
                            model::partial_result(p.rejected_spans, !p.error_message.is_empty())
                        })
                    }
                    Signal::Logs => {
                        let response = ExportLogsServiceResponse::decode(response.as_slice())
                            .map_err(|_| "invalid OTLP logs response")?;
                        response.partial_success.map_or(Ok(false), |p| {
                            model::partial_result(
                                p.rejected_log_records,
                                !p.error_message.is_empty(),
                            )
                        })
                    }
                }
            }
            Self::Grpc { channel, options } => match signal {
                Signal::Traces => {
                    let payload = ExportTraceServiceRequest::decode(body.as_slice())
                        .map_err(|_| "invalid internal OTLP traces payload")?;
                    let mut request = tonic::Request::new(payload);
                    *request.metadata_mut() = metadata(&options.headers)?;
                    request.set_timeout(options.timeout);
                    let mut client = TraceServiceClient::new(channel.clone())
                        .max_decoding_message_size(MAX_RESPONSE_BYTES);
                    let response = tokio::time::timeout(options.timeout, client.export(request))
                        .await
                        .map_err(|_| "OTLP traces gRPC operation timed out")?
                        .map_err(|status| format!("OTLP traces gRPC status {}", status.code()))?
                        .into_inner();
                    response.partial_success.map_or(Ok(false), |p| {
                        model::partial_result(p.rejected_spans, !p.error_message.is_empty())
                    })
                }
                Signal::Logs => {
                    let payload = ExportLogsServiceRequest::decode(body.as_slice())
                        .map_err(|_| "invalid internal OTLP logs payload")?;
                    let mut request = tonic::Request::new(payload);
                    *request.metadata_mut() = metadata(&options.headers)?;
                    request.set_timeout(options.timeout);
                    let mut client = LogsServiceClient::new(channel.clone())
                        .max_decoding_message_size(MAX_RESPONSE_BYTES);
                    let response = tokio::time::timeout(options.timeout, client.export(request))
                        .await
                        .map_err(|_| "OTLP logs gRPC operation timed out")?
                        .map_err(|status| format!("OTLP logs gRPC status {}", status.code()))?
                        .into_inner();
                    response.partial_success.map_or(Ok(false), |p| {
                        model::partial_result(p.rejected_log_records, !p.error_message.is_empty())
                    })
                }
            },
        }
    }
}
fn metadata(headers: &StartupHeaders) -> Result<tonic::metadata::MetadataMap, String> {
    let mut map = tonic::metadata::MetadataMap::new();
    for (key, value) in &headers.0 {
        let key =
            tonic::metadata::MetadataKey::<tonic::metadata::Ascii>::from_bytes(key.as_bytes())
                .map_err(|_| "invalid OTLP gRPC header name")?;
        let value = tonic::metadata::MetadataValue::try_from(value.as_str())
            .map_err(|_| "invalid OTLP gRPC header value")?;
        map.insert(key, value);
    }
    Ok(map)
}
async fn http_export(
    client: &reqwest::Client,
    options: &SignalOptions,
    body: Vec<u8>,
) -> Result<Vec<u8>, String> {
    let mut request = client
        .post(&options.endpoint)
        .header("content-type", "application/x-protobuf");
    for (key, value) in &options.headers.0 {
        request = request.header(key, value);
    }
    let mut response = request
        .body(body)
        .send()
        .await
        .map_err(|_| "OTLP HTTP request failed (network, TLS, or timeout)")?;
    if !response.status().is_success() {
        return Err(format!("OTLP HTTP status {}", response.status().as_u16()));
    }
    if response
        .content_length()
        .is_some_and(|size| size > MAX_RESPONSE_BYTES as u64)
    {
        return Err("OTLP response exceeds size limit".into());
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| "cannot read OTLP response")?
    {
        if bytes.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
            return Err("OTLP response exceeds size limit".into());
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}
fn export_records(
    runtime: &tokio::runtime::Runtime,
    records: &[Record],
    counters: &Counters,
    transports: &mut (Option<Transport>, Option<Transport>),
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
        let transport = match signal {
            Signal::Traces => transports.0.as_mut(),
            Signal::Logs => transports.1.as_mut(),
        }
        .ok_or("OpenTelemetry signal is not enabled")?;
        match runtime.block_on(transport.send(signal, body)) {
            Ok(warning) => counters.accepted(signal, count, warning),
            Err(failure) => {
                warn_failure(counters);
                error = Some(failure);
            }
        }
    }
    error.map_or(Ok(()), Err)
}
fn worker(
    runtime: tokio::runtime::Runtime,
    receiver: mpsc::Receiver<Command>,
    counters: Arc<Counters>,
    resource: Resource,
    mut transports: (Option<Transport>, Option<Transport>),
) {
    let mut batch = Vec::new();
    let mut size = 0;
    let mut due: Option<Instant> = None;
    loop {
        let command = match due {
            Some(deadline) => {
                receiver.recv_timeout(deadline.saturating_duration_since(Instant::now()))
            }
            None => receiver
                .recv()
                .map_err(|_| mpsc::RecvTimeoutError::Disconnected),
        };
        let mut reply = None;
        let mut probe = None;
        let mut shutdown = false;
        match command {
            Ok(Command::Record(record)) => {
                if size + record.bytes.len() > MAX_BATCH_BYTES && !batch.is_empty() {
                    if export_records(&runtime, &batch, &counters, &mut transports).is_err() {
                        // Transport failure is already durably counted/warned; background export cannot fail a run.
                    }
                    batch.clear();
                    size = 0;
                    due = None;
                }
                size += record.bytes.len();
                batch.push(record);
                due.get_or_insert_with(|| Instant::now() + Duration::from_millis(200));
                if batch.len() < 256 && size < MAX_BATCH_BYTES {
                    continue;
                }
            }
            Ok(Command::Flush(sender)) => reply = Some(sender),
            Ok(Command::Shutdown(sender)) => {
                reply = Some(sender);
                shutdown = true;
            }
            Ok(Command::Probe {
                reply: sender,
                deadline,
            }) => probe = Some((sender, deadline)),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => shutdown = true,
        }
        if !batch.is_empty() {
            if export_records(&runtime, &batch, &counters, &mut transports).is_err() {
                // Explicit barriers below report cumulative failure without consuming another caller's result.
            }
            batch.clear();
            size = 0;
            due = None;
        }
        if let Some((sender, deadline)) = probe {
            let result = if Instant::now() >= deadline {
                Err("OpenTelemetry test timed out before export".into())
            } else {
                model::probe_records(&resource, &counters).and_then(|(span, log)| {
                    let records = [span, log];
                    let result = export_records(&runtime, &records, &counters, &mut transports);
                    drop(records);
                    result.map(|_| counters.snapshot())
                })
            };
            let _ = sender.send(result);
        }
        if let Some(sender) = reply {
            let _ = sender.send(counters.flush_result());
        }
        if shutdown {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn concurrent_probe_receipts_cannot_be_consumed_by_flush() {
        use std::io::{BufRead, Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let collector = std::thread::spawn(move || {
            for index in 0..4 {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut reader = std::io::BufReader::new(&mut stream);
                let mut line = String::new();
                let mut length = 0;
                reader.read_line(&mut line).unwrap();
                loop {
                    line.clear();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" {
                        break;
                    }
                    if let Some((key, value)) = line.split_once(':') {
                        if key.eq_ignore_ascii_case("content-length") {
                            length = value.trim().parse().unwrap();
                        }
                    }
                }
                let mut bytes = vec![0; length];
                reader.read_exact(&mut bytes).unwrap();
                drop(reader);
                let body = if index == 1 {
                    ExportLogsServiceResponse{partial_success:Some(opentelemetry_proto::tonic::collector::logs::v1::ExportLogsPartialSuccess{rejected_log_records:0,error_message:"secret collector warning".into()})}.encode_to_vec()
                } else {
                    vec![]
                };
                let status = if index == 0 { 401 } else { 200 };
                write!(
                    stream,
                    "HTTP/1.1 {status} fixture\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                stream.write_all(&body).unwrap();
            }
        });
        let (sender, receiver) = mpsc::sync_channel(8);
        let counters = Arc::new(Counters::default());
        let worker_counters = counters.clone();
        let worker_thread = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let make = |signal| SignalOptions {
                endpoint: format!("{origin}/v1/{signal}"),
                protocol: OtelProtocol::HttpProtobuf,
                headers: StartupHeaders::default(),
                timeout: Duration::from_secs(2),
            };
            let transports = {
                let _entered = runtime.enter();
                Transport::pair(Some(make("traces")), Some(make("logs"))).unwrap()
            };
            worker(
                runtime,
                receiver,
                worker_counters,
                Resource::empty(),
                transports,
            );
        });
        let (first, one) = mpsc::sync_channel(1);
        let (second, two) = mpsc::sync_channel(1);
        let (flush, flushed) = mpsc::sync_channel(1);
        let deadline = Instant::now() + Duration::from_secs(10);
        // Both probes and an intervening flush are in flight before either caller awaits its own receipt.
        assert!(sender
            .send(Command::Probe {
                reply: first,
                deadline
            })
            .is_ok());
        assert!(sender.send(Command::Flush(flush)).is_ok());
        assert!(sender
            .send(Command::Probe {
                reply: second,
                deadline
            })
            .is_ok());
        assert_eq!(
            one.recv_timeout(Duration::from_secs(10))
                .unwrap()
                .unwrap_err(),
            "OTLP HTTP status 401"
        );
        assert!(flushed
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            .is_err());
        let recovered = two.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
        assert_eq!(recovered.exported_spans, 1);
        assert_eq!(recovered.exported_logs, 2);
        assert_eq!(recovered.collector_warnings, 1);
        let (reply, done) = mpsc::sync_channel(1);
        assert!(sender.send(Command::Shutdown(reply)).is_ok());
        assert!(done.recv_timeout(Duration::from_secs(5)).unwrap().is_err());
        worker_thread.join().unwrap();
        collector.join().unwrap();
        assert_eq!(counters.snapshot().queued_bytes, 0);
    }
}
