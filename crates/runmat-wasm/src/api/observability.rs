//! Runtime-only collector credentials and acknowledged browser export operations.
use runmat_logging::otel::{OtelOptions, OtelProtocol, OtelSignal, StartupHeaders};
use std::collections::BTreeMap;
use wasm_bindgen::prelude::*;

pub(crate) fn configure_for_session(
    config: &runmat_config::runtime::OtelConfig,
    options: &JsValue,
) -> Result<(), JsValue> {
    use runmat_config::runtime::{OtelProtocol as Protocol, OtelSignal as Signal};
    let runtime = OtelOptions {
        enabled: config.enabled,
        endpoint: config.endpoint.clone(),
        protocol: match config.protocol {
            Protocol::HttpProtobuf => OtelProtocol::HttpProtobuf,
            Protocol::Grpc => OtelProtocol::Grpc,
        },
        signals: config
            .signals
            .iter()
            .map(|value| match value {
                Signal::Traces => OtelSignal::Traces,
                Signal::Logs => OtelSignal::Logs,
                Signal::Metrics => OtelSignal::Metrics,
            })
            .collect(),
        // Browsers do not have a process environment; all secrets come from runtime-only host input.
        headers_from_env: None,
        service_name: config.service_name.clone(),
        resource_attributes: config.resource_attributes.clone(),
        ..OtelOptions::default()
    };
    crate::runtime::logging::configure_logging(runtime, startup_headers(options)?)
        .map_err(|message| JsValue::from_str(&message))
}

fn startup_headers(options: &JsValue) -> Result<StartupHeaders, JsValue> {
    if options.is_null() || options.is_undefined() {
        return Ok(StartupHeaders::default());
    }
    let headers = js_sys::Reflect::get(options, &JsValue::from_str("openTelemetryHeaders"))
        .map_err(|_| JsValue::from_str("OpenTelemetry headers could not be read"))?;
    if headers.is_null() || headers.is_undefined() {
        return Ok(StartupHeaders::default());
    }
    if !headers.is_object() || js_sys::Array::is_array(&headers) {
        return Err(JsValue::from_str(
            "openTelemetryHeaders must be an object of string header values",
        ));
    }
    let mut values = BTreeMap::new();
    let keys = js_sys::Object::keys(&js_sys::Object::from(headers.clone()));
    if keys.length() > 16 {
        return Err(JsValue::from_str(
            "OpenTelemetry headers exceed their count bound",
        ));
    }
    let mut bytes = 0usize;
    for key in keys {
        let name = key
            .as_string()
            .ok_or_else(|| JsValue::from_str("invalid OpenTelemetry header name"))?;
        let raw = js_sys::Reflect::get(&headers, &key)
            .map_err(|_| JsValue::from_str("OpenTelemetry startup header could not be read"))?;
        if !raw.is_string() {
            return Err(JsValue::from_str(
                "OpenTelemetry startup header values must be strings",
            ));
        }
        let length = js_sys::JsString::from(raw.clone()).length();
        if length > 4096 || name.len() > 128 {
            return Err(JsValue::from_str(
                "OpenTelemetry headers exceed their byte bound",
            ));
        }
        let value = raw
            .as_string()
            .ok_or_else(|| JsValue::from_str("OpenTelemetry startup header must be a string"))?;
        bytes = bytes.saturating_add(name.len()).saturating_add(value.len());
        if value.len() > 4096 || bytes > 8192 {
            return Err(JsValue::from_str(
                "OpenTelemetry headers exceed their byte bound",
            ));
        }
        values.insert(name, value);
    }
    Ok(StartupHeaders(values))
}

#[wasm_bindgen(js_name = sendOpenTelemetryTest)]
pub async fn send_open_telemetry_test() -> Result<JsValue, JsValue> {
    export_result(runmat_logging::browser_otel::emit_test_browser_otel().await)
}

#[wasm_bindgen(js_name = flushOpenTelemetry)]
pub async fn flush_open_telemetry() -> Result<JsValue, JsValue> {
    export_result(runmat_logging::browser_otel::flush_browser_otel().await)
}

/// Runtime-wide shutdown, not session disposal. Other sessions share this exporter.
#[wasm_bindgen(js_name = shutdownOpenTelemetry)]
pub async fn shutdown_open_telemetry() -> Result<JsValue, JsValue> {
    export_result(runmat_logging::browser_otel::shutdown_browser_otel().await)
}

#[wasm_bindgen(js_name = openTelemetryStats)]
pub fn open_telemetry_stats() -> Result<JsValue, JsValue> {
    let stats = runmat_logging::browser_otel::browser_otel_stats()
        .ok_or_else(|| JsValue::from_str("OpenTelemetry export is not enabled"))?;
    export_result(Ok(stats))
}

fn export_result(
    result: Result<runmat_logging::browser_otel::BrowserExportStats, String>,
) -> Result<JsValue, JsValue> {
    let stats = result.map_err(|message| JsValue::from_str(&message))?;
    // Keep host receipts stable and credential-free, independent of SDK internals.
    serde_wasm_bindgen::to_value(&serde_json::json!({
        "droppedRecords": stats.dropped_records, "failedBatches": stats.failed_batches,
        "exportedSpans": stats.exported_spans, "exportedLogs": stats.exported_logs,
        "collectorWarnings": stats.collector_warnings, "queuedBytes": stats.queued_bytes,
    }))
    .map_err(|_| JsValue::from_str("OpenTelemetry receipt could not be serialized"))
}
