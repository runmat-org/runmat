//! Shared bounded SDK/protobuf boundary. No transport or alternative trace model lives here.
use crate::otel::OtelOptions;
use crate::RuntimeLogRecord;
use opentelemetry::{
    trace::{Span, TraceContextExt, Tracer, TracerProvider},
    Array, Context, KeyValue, Value,
};
use opentelemetry_proto::tonic::{
    collector::{logs::v1::ExportLogsServiceRequest, trace::v1::ExportTraceServiceRequest},
    common::v1::{any_value, AnyValue, ArrayValue, InstrumentationScope, KeyValueList},
    logs::v1::{LogRecord, ResourceLogs, ScopeLogs},
};
use opentelemetry_sdk::{
    export::trace::SpanData,
    trace::{self, SpanProcessor},
    Resource,
};
use prost::Message;
use std::{
    borrow::Cow,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

pub(crate) const MAX_RECORD_BYTES: usize = 64 * 1024;
pub(crate) const MAX_QUEUE_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const MAX_BATCH_BYTES: usize = 512 * 1024;
pub(crate) const MAX_RESPONSE_BYTES: usize = 64 * 1024;
pub(crate) const MAX_FIELD_BYTES: usize = 4096;
pub(crate) const MAX_MESSAGE_BYTES: usize = 8192;
pub(crate) const MAX_FIELDS: usize = 32;
pub(crate) const FLUSH_TIMEOUT_MS: u32 = 15_000;

pub(crate) fn validate_headers(headers: &crate::otel::StartupHeaders) -> Result<(), String> {
    if headers.0.len() > 16 {
        return Err("OpenTelemetry headers exceed their count bound".into());
    }
    let mut size = 0usize;
    for (key, value) in &headers.0 {
        size = size.saturating_add(key.len()).saturating_add(value.len());
        if key.len() > 128 || value.len() > 4096 || size > 8192 {
            return Err("OpenTelemetry headers exceed their byte bound".into());
        }
    }
    Ok(())
}
pub(crate) fn validate_signal(options: &crate::otel::SignalOptions) -> Result<(), String> {
    validate_headers(&options.headers)?;
    if options.endpoint.len() > 2048
        || options.timeout.is_zero()
        || options.timeout > std::time::Duration::from_secs(30)
    {
        return Err(
            "OpenTelemetry endpoint or timeout exceeds its bound (timeout must be 1–30000ms)"
                .into(),
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ExportStats {
    pub dropped_records: u64,
    pub failed_batches: u64,
    pub exported_spans: u64,
    pub exported_logs: u64,
    pub collector_warnings: u64,
    pub queued_bytes: u64,
}
#[derive(Debug, Default)]
pub(crate) struct Counters {
    pub dropped: AtomicU64,
    pub failed: AtomicU64,
    pub spans: AtomicU64,
    pub logs: AtomicU64,
    pub warnings: AtomicU64,
    warned: AtomicBool,
    pub bytes: AtomicUsize,
}
impl Counters {
    pub fn snapshot(&self) -> ExportStats {
        ExportStats {
            dropped_records: self.dropped.load(Ordering::Relaxed),
            failed_batches: self.failed.load(Ordering::Relaxed),
            exported_spans: self.spans.load(Ordering::Relaxed),
            exported_logs: self.logs.load(Ordering::Relaxed),
            collector_warnings: self.warnings.load(Ordering::Relaxed),
            queued_bytes: self.bytes.load(Ordering::Relaxed) as u64,
        }
    }
    pub fn drop_record(&self) {
        self.dropped.fetch_add(1, Ordering::Relaxed);
    }
    pub fn failure(&self, warn: impl FnOnce()) {
        self.failed.fetch_add(1, Ordering::Relaxed);
        if !self.warned.swap(true, Ordering::Relaxed) {
            warn();
        }
    }
    pub fn accepted(&self, signal: Signal, count: u64, warning: bool) {
        match signal {
            Signal::Traces => &self.spans,
            Signal::Logs => &self.logs,
        }
        .fetch_add(count, Ordering::Relaxed);
        if warning {
            self.warnings.fetch_add(1, Ordering::Relaxed);
        }
    }
    pub fn flush_result(&self) -> Result<ExportStats, String> {
        if self.failed.load(Ordering::Relaxed) > 0 {
            Err("OpenTelemetry has unacknowledged failed batches; use Send test span to test current collector health".into())
        } else {
            Ok(self.snapshot())
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Signal {
    Traces,
    Logs,
}

/// Capacity is reserved before encoding; held through queued/in-flight ownership and returned on drop.
pub(crate) struct Record {
    pub signal: Signal,
    pub bytes: Vec<u8>,
    charged: usize,
    counters: Arc<Counters>,
}
impl Drop for Record {
    fn drop(&mut self) {
        self.counters
            .bytes
            .fetch_sub(self.charged, Ordering::AcqRel);
    }
}
fn encode<M: Message>(signal: Signal, message: M, counters: &Arc<Counters>) -> Option<Record> {
    let length = message.encoded_len();
    if length > MAX_RECORD_BYTES {
        counters.drop_record();
        return None;
    }
    let charged = length + std::mem::size_of::<Record>();
    if counters
        .bytes
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| {
            old.checked_add(charged)
                .filter(|new| *new <= MAX_QUEUE_BYTES)
        })
        .is_err()
    {
        counters.drop_record();
        return None;
    }
    let mut bytes = Vec::with_capacity(length);
    // encoded_len is the exact prost allocation requirement; no secret-bearing encode errors are exposed.
    if message.encode(&mut bytes).is_err() {
        counters.bytes.fetch_sub(charged, Ordering::AcqRel);
        counters.drop_record();
        return None;
    }
    Some(Record {
        signal,
        bytes,
        charged,
        counters: counters.clone(),
    })
}
pub(crate) fn config_resource(options: &OtelOptions) -> Result<Resource, String> {
    if options.endpoint.len() > 2048
        || options.service_name.len() > 256
        || options.resource_attributes.len() >= MAX_FIELDS
    {
        return Err("OpenTelemetry endpoint/service/resource metadata exceeds its bound".into());
    }
    let mut size = options.service_name.len();
    for (key, value) in &options.resource_attributes {
        size = size.saturating_add(key.len()).saturating_add(value.len());
        if key.len() > 256 || value.len() > MAX_FIELD_BYTES || size > 16 * 1024 {
            return Err("OpenTelemetry resource metadata exceeds its bound".into());
        }
    }
    Ok(Resource::new(
        options
            .resource_attributes
            .iter()
            .map(|(k, v)| KeyValue::new(k.clone(), v.clone()))
            .chain([KeyValue::new("service.name", options.service_name.clone())]),
    ))
}
fn value_size(value: &Value) -> Option<usize> {
    let size = match value {
        Value::String(v) => v.as_str().len(),
        Value::Bool(_) | Value::I64(_) | Value::F64(_) => 8,
        Value::Array(array) => match array {
            Array::Bool(v) => v.len(),
            Array::I64(v) => v.len().checked_mul(8)?,
            Array::F64(v) => v.len().checked_mul(8)?,
            Array::String(values) => {
                if values.len() > MAX_FIELDS {
                    return None;
                }
                values
                    .iter()
                    .try_fold(0usize, |size, v| size.checked_add(v.as_str().len()))?
            }
        },
    };
    (size <= MAX_FIELD_BYTES).then_some(size)
}
fn attributes_size<'a>(
    values: impl IntoIterator<Item = (&'a opentelemetry::Key, &'a Value)>,
) -> Option<usize> {
    let mut total = 0usize;
    for (index, (key, value)) in values.into_iter().enumerate() {
        if index >= MAX_FIELDS || key.as_str().len() > 256 {
            return None;
        }
        total = total
            .checked_add(key.as_str().len())?
            .checked_add(value_size(value)?)?;
        if total > MAX_RECORD_BYTES {
            return None;
        }
    }
    Some(total)
}
fn span_bounded(span: &SpanData) -> bool {
    let checked = || -> Option<usize> {
        if span.name.len() > 1024
            || span.events.len() > MAX_FIELDS
            || span.links.len() > 16
            || span.instrumentation_lib.name.len() > 256
            || span
                .instrumentation_lib
                .version
                .as_ref()
                .is_some_and(|v| v.len() > 256)
            || span
                .instrumentation_lib
                .schema_url
                .as_ref()
                .is_some_and(|v| v.len() > 512)
        {
            return None;
        }
        let mut size = span.name.len()
            + attributes_size(span.attributes.iter().map(|kv| (&kv.key, &kv.value)))?;
        size = size.checked_add(attributes_size(span.resource.iter())?)?;
        size = size.checked_add(attributes_size(
            span.instrumentation_lib
                .attributes
                .iter()
                .map(|kv| (&kv.key, &kv.value)),
        )?)?;
        for event in span.events.iter() {
            if event.name.len() > MAX_MESSAGE_BYTES {
                return None;
            }
            size = size
                .checked_add(event.name.len())?
                .checked_add(attributes_size(
                    event.attributes.iter().map(|kv| (&kv.key, &kv.value)),
                )?)?;
            if size > MAX_RECORD_BYTES {
                return None;
            }
        }
        for link in span.links.iter() {
            size = size.checked_add(attributes_size(
                link.attributes.iter().map(|kv| (&kv.key, &kv.value)),
            )?)?;
        }
        if let opentelemetry::trace::Status::Error { description } = &span.status {
            if description.len() > 1024 {
                return None;
            }
            size = size.checked_add(description.len())?;
        }
        Some(size)
    };
    // Reserve room for bounded resource attributes and protobuf field/container overhead.
    checked().is_some_and(|size| size <= MAX_RECORD_BYTES / 2)
}
pub(crate) fn span_record(
    mut span: SpanData,
    resource: &Resource,
    counters: &Arc<Counters>,
) -> Option<Record> {
    if !span_bounded(&span) {
        counters.drop_record();
        return None;
    }
    span.resource = Cow::Owned(resource.clone());
    encode(
        Signal::Traces,
        ExportTraceServiceRequest {
            resource_spans: vec![span.into()],
        },
        counters,
    )
}
fn json_size(value: &serde_json::Value, depth: usize) -> Option<usize> {
    if depth > 4 {
        return None;
    }
    let result = match value {
        serde_json::Value::String(v) => v.len(),
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => 32,
        serde_json::Value::Array(v) => {
            if v.len() > MAX_FIELDS {
                return None;
            }
            v.iter().try_fold(0usize, |total, v| {
                total.checked_add(json_size(v, depth + 1)?)
            })?
        }
        serde_json::Value::Object(v) => {
            if v.len() > MAX_FIELDS {
                return None;
            }
            v.iter().try_fold(0usize, |total, (k, v)| {
                total
                    .checked_add(k.len())?
                    .checked_add(json_size(v, depth + 1)?)
            })?
        }
    };
    (result <= MAX_FIELD_BYTES).then_some(result)
}
fn json_any(value: &serde_json::Value) -> AnyValue {
    AnyValue {
        value: match value {
            serde_json::Value::Null => None,
            serde_json::Value::Bool(v) => Some(any_value::Value::BoolValue(*v)),
            serde_json::Value::Number(v) if v.is_i64() => {
                Some(any_value::Value::IntValue(v.as_i64().unwrap_or_default()))
            }
            serde_json::Value::Number(v) => Some(any_value::Value::DoubleValue(
                v.as_f64().unwrap_or_default(),
            )),
            serde_json::Value::String(v) => Some(any_value::Value::StringValue(v.clone())),
            serde_json::Value::Array(values) => Some(any_value::Value::ArrayValue(ArrayValue {
                values: values.iter().map(json_any).collect(),
            })),
            serde_json::Value::Object(values) => {
                Some(any_value::Value::KvlistValue(KeyValueList {
                    values: values
                        .iter()
                        .map(
                            |(key, value)| opentelemetry_proto::tonic::common::v1::KeyValue {
                                key: key.clone(),
                                value: Some(json_any(value)),
                            },
                        )
                        .collect(),
                }))
            }
        },
    }
}
pub(crate) fn log_record(
    record: &RuntimeLogRecord,
    context: &Context,
    resource: &Resource,
    counters: &Arc<Counters>,
) -> Option<Record> {
    let fields = record.fields.as_ref().and_then(|value| value.as_object());
    let total = fields
        .map(|fields| {
            fields.iter().try_fold(
                record.message.len() + record.target.len() + 16 * 1024,
                |total, (key, value)| {
                    total
                        .checked_add(key.len())?
                        .checked_add(json_size(value, 0)?)
                },
            )
        })
        .unwrap_or(Some(record.message.len() + record.target.len() + 16 * 1024));
    if record.message.len() > MAX_MESSAGE_BYTES
        || record.target.len() > 256
        || record.level.len() > 16
        || total.is_none_or(|size| size > MAX_RECORD_BYTES)
        || fields.is_some_and(|fields| {
            fields.len() > MAX_FIELDS
                || fields
                    .iter()
                    .any(|(key, value)| key.len() > 256 || json_size(value, 0).is_none())
        })
    {
        counters.drop_record();
        return None;
    }
    let context_span = context.span();
    let span = context_span.span_context();
    let mut attributes = vec![opentelemetry_proto::tonic::common::v1::KeyValue {
        key: "log.target".into(),
        value: Some(AnyValue {
            value: Some(any_value::Value::StringValue(record.target.clone())),
        }),
    }];
    for (key, value) in fields.into_iter().flat_map(|fields| fields.iter()) {
        attributes.push(opentelemetry_proto::tonic::common::v1::KeyValue {
            key: key.clone(),
            value: Some(json_any(value)),
        });
    }
    let timestamp = opentelemetry::time::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64;
    let log = LogRecord {
        time_unix_nano: timestamp,
        observed_time_unix_nano: timestamp,
        severity_number: match record.level.as_str() {
            "ERROR" => 17,
            "WARN" => 13,
            "DEBUG" => 5,
            "TRACE" => 1,
            _ => 9,
        },
        severity_text: record.level.clone(),
        body: Some(AnyValue {
            value: Some(any_value::Value::StringValue(record.message.clone())),
        }),
        trace_id: if span.is_valid() {
            span.trace_id().to_bytes().to_vec()
        } else {
            vec![]
        },
        span_id: if span.is_valid() {
            span.span_id().to_bytes().to_vec()
        } else {
            vec![]
        },
        flags: if span.is_sampled() { 1 } else { 0 },
        attributes,
        ..Default::default()
    };
    let resource = opentelemetry_proto::tonic::resource::v1::Resource {
        attributes: resource
            .iter()
            .map(
                |(key, value)| opentelemetry_proto::tonic::common::v1::KeyValue {
                    key: key.as_str().to_owned(),
                    value: Some(value.clone().into()),
                },
            )
            .collect(),
        dropped_attributes_count: 0,
    };
    encode(
        Signal::Logs,
        ExportLogsServiceRequest {
            resource_logs: vec![ResourceLogs {
                resource: Some(resource),
                scope_logs: vec![ScopeLogs {
                    scope: Some(InstrumentationScope {
                        name: "runmat".into(),
                        ..Default::default()
                    }),
                    log_records: vec![log],
                    schema_url: String::new(),
                }],
                schema_url: String::new(),
            }],
        },
        counters,
    )
}
pub(crate) fn span_config() -> trace::Config {
    trace::config()
        .with_sampler(trace::Sampler::AlwaysOn)
        .with_resource(Resource::empty())
        .with_max_attributes_per_span(MAX_FIELDS as u32)
        .with_max_events_per_span(MAX_FIELDS as u32)
        .with_max_links_per_span(16)
        .with_max_attributes_per_event(MAX_FIELDS as u32)
        .with_max_attributes_per_link(MAX_FIELDS as u32)
}
/// Capture a real SDK span without dispatching it through the ordinary asynchronous queue.
pub(crate) fn probe_records(
    resource: &Resource,
    counters: &Arc<Counters>,
) -> Result<(Record, Record), String> {
    #[derive(Debug)]
    struct Capture(Arc<Mutex<Option<SpanData>>>);
    impl SpanProcessor for Capture {
        fn on_start(&self, _: &mut trace::Span, _: &Context) {}
        fn on_end(&self, span: SpanData) {
            *self.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(span);
        }
        fn force_flush(&self) -> opentelemetry::trace::TraceResult<()> {
            Ok(())
        }
        fn shutdown(&mut self) -> opentelemetry::trace::TraceResult<()> {
            Ok(())
        }
    }
    let capture = Arc::new(Mutex::new(None));
    let provider = trace::TracerProvider::builder()
        .with_config(span_config())
        .with_span_processor(Capture(capture.clone()))
        .build();
    let mut span = provider.tracer("runmat.settings").build(
        opentelemetry::trace::SpanBuilder::from_name("runmat.otel.test")
            .with_start_time(opentelemetry::time::now()),
    );
    let context = Context::new().with_remote_span_context(span.span_context().clone());
    let log = log_record(
        &RuntimeLogRecord {
            ts: String::new(),
            level: "INFO".into(),
            target: "runmat.settings".into(),
            message: "RunMat OpenTelemetry test".into(),
            trace_id: None,
            span_id: None,
            fields: None,
        },
        &context,
        resource,
        counters,
    );
    span.end_with_timestamp(opentelemetry::time::now());
    let log = log.ok_or("OpenTelemetry test queue byte budget is exhausted")?;
    let data = capture
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .ok_or("OpenTelemetry SDK did not complete the test span")?;
    let trace = span_record(data, resource, counters)
        .ok_or("OpenTelemetry test queue byte budget is exhausted")?;
    Ok((trace, log))
}
pub(crate) fn partial_result(rejected: i64, message_present: bool) -> Result<bool, String> {
    if rejected < 0 {
        Err("collector returned an invalid rejected-record count".into())
    } else if rejected > 0 {
        Err("collector rejected part of the OpenTelemetry batch".into())
    } else {
        Ok(message_present)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_model_mechanism_matrix() {
        let counters = Arc::new(Counters::default());
        let resource = Resource::empty();
        let mut record = RuntimeLogRecord {
            ts: String::new(),
            level: "INFO".into(),
            target: "test".into(),
            message: "x".repeat(MAX_MESSAGE_BYTES + 1),
            trace_id: None,
            span_id: None,
            fields: None,
        };
        assert!(log_record(&record, &Context::new(), &resource, &counters).is_none());
        assert_eq!(counters.snapshot().queued_bytes, 0);
        record.message = "x".repeat(MAX_MESSAGE_BYTES);
        let mut held = Vec::new();
        while let Some(value) = log_record(&record, &Context::new(), &resource, &counters) {
            held.push(value);
        }
        assert!(counters.snapshot().queued_bytes <= MAX_QUEUE_BYTES as u64);
        assert!(counters.snapshot().dropped_records >= 2);
        drop(held);
        assert_eq!(counters.snapshot().queued_bytes, 0);
        let (span, log) = probe_records(&resource, &counters).unwrap();
        let trace = ExportTraceServiceRequest::decode(span.bytes.as_slice()).unwrap();
        let logs = ExportLogsServiceRequest::decode(log.bytes.as_slice()).unwrap();
        let span = &trace.resource_spans[0].scope_spans[0].spans[0];
        let log = &logs.resource_logs[0].scope_logs[0].log_records[0];
        assert_eq!(span.trace_id, log.trace_id);
        assert_eq!(span.span_id, log.span_id);
        assert_eq!(log.trace_id.len(), 16);
        assert!(log.trace_id.iter().any(|byte| *byte != 0));
        assert!(partial_result(0, true).unwrap());
        assert!(partial_result(1, true).is_err());
        assert!(partial_result(-1, false).is_err());
        let mut headers = crate::otel::StartupHeaders::default();
        headers.0.insert("authorization".into(), "s".repeat(8193));
        assert!(validate_headers(&headers).is_err());
        let options = OtelOptions {
            service_name: "s".repeat(257),
            ..Default::default()
        };
        assert!(config_resource(&options).is_err());
    }
}
