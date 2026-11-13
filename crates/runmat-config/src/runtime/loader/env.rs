use anyhow::{bail, Context, Result};
use serde_json::Value;

/// Un-defaulted overrides retain exact per-field provenance. Secret headers and
/// per-signal transport variables belong to exporter startup, not persistence.
pub(super) fn environment_layers() -> Result<Vec<(String, Value)>> {
    let mut layers = Vec::new();
    for variable in [
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
    ] {
        if std::env::var_os(variable).is_some_and(|endpoint| !endpoint.is_empty()) {
            layers.push((
                variable.into(),
                at_path("logging.otel.enabled", Value::Bool(true)),
            ));
        }
    }
    for (variable, path) in [
        ("OTEL_EXPORTER_OTLP_ENDPOINT", "logging.otel.endpoint"),
        ("OTEL_EXPORTER_OTLP_PROTOCOL", "logging.otel.protocol"),
        ("OTEL_SERVICE_NAME", "logging.otel.service_name"),
        ("RUNMAT_CALLSTACK_LIMIT", "callstack_limit"),
        ("RUNMAT_VERBOSE", "verbose"),
        ("RUNMAT_JIT_ENABLED", "jit.enabled"),
        ("RUNMAT_JIT_THRESHOLD", "jit.threshold"),
        ("RUNMAT_JIT_OPTIMIZATION_LEVEL", "jit.optimization_level"),
        ("RUNMAT_GC_PRESET", "gc.preset"),
        ("RUNMAT_GC_YOUNG_SIZE_MB", "gc.young_size_mb"),
        ("RUNMAT_GC_THREADS", "gc.threads"),
        ("RUNMAT_ACCELERATE_ENABLED", "accelerate.enabled"),
        ("RUNMAT_ACCELERATE_PROVIDER", "accelerate.provider"),
        ("RUNMAT_LANGUAGE_COMPAT", "language.compat"),
        ("RUNMAT_LOG_LEVEL", "logging.level"),
        ("RUNMAT_LOG_DEBUG", "logging.debug"),
        ("RUNMAT_LOG_FILE", "logging.file"),
        ("RUNMAT_OTEL_ENDPOINT", "logging.otel.endpoint"),
        ("RUNMAT_OTEL_ENABLED", "logging.otel.enabled"),
    ] {
        if let Some(text) = std::env::var_os(variable) {
            let text = text
                .into_string()
                .map_err(|_| anyhow::anyhow!("{variable} must contain Unicode text"))?;
            let value = match path {
                "callstack_limit" | "jit.threshold" | "gc.young_size_mb" | "gc.threads" => {
                    serde_json::to_value(
                        text.parse::<u64>()
                            .with_context(|| format!("invalid integer in {variable}"))?,
                    )?
                }
                "verbose"
                | "jit.enabled"
                | "accelerate.enabled"
                | "logging.debug"
                | "logging.otel.enabled" => {
                    Value::Bool(text.parse::<bool>().with_context(|| {
                        format!("invalid boolean in {variable}; expected true or false")
                    })?)
                }
                _ => Value::String(text),
            };
            layers.push((variable.into(), at_path(path, value)));
            if variable == "RUNMAT_OTEL_ENDPOINT" || variable == "OTEL_EXPORTER_OTLP_ENDPOINT" {
                layers.push((
                    variable.into(),
                    at_path("logging.otel.enabled", Value::Bool(true)),
                ));
            }
        }
    }
    if let Ok(text) = std::env::var("OTEL_RESOURCE_ATTRIBUTES") {
        let mut attributes = serde_json::Map::new();
        for entry in text.split(',').filter(|s| !s.trim().is_empty()) {
            let (key, value) = entry
                .split_once('=')
                .context("OTEL_RESOURCE_ATTRIBUTES expects comma-separated key=value entries")?;
            if key.trim().is_empty() {
                bail!("OTEL_RESOURCE_ATTRIBUTES contains an empty key");
            }
            attributes.insert(key.trim().into(), Value::String(value.trim().into()));
        }
        layers.push((
            "OTEL_RESOURCE_ATTRIBUTES".into(),
            at_path(
                "logging.otel.resource_attributes",
                Value::Object(attributes),
            ),
        ));
    }
    // All schema fields are addressable. JSON literals encode arrays, numbers,
    // booleans and null; plain text is accepted for strings. Child paths win.
    let mut generic: Vec<_> = std::env::vars_os()
        .filter_map(|(key, value)| {
            let key = key.to_str()?;
            key.strip_prefix("RUNMAT_RUNTIME__").map(|path| {
                (
                    key.to_owned(),
                    path.to_lowercase().replace("__", "."),
                    value,
                )
            })
        })
        .collect();
    generic.sort_by(|a, b| a.1.cmp(&b.1));
    for (variable, path, text) in generic {
        if path.is_empty() || path.split('.').any(str::is_empty) {
            bail!("invalid runtime setting path in {variable}");
        }
        let text = text
            .into_string()
            .map_err(|_| anyhow::anyhow!("{variable} must contain Unicode text"))?;
        let value = serde_json::from_str(&text).unwrap_or(Value::String(text));
        layers.push((variable, at_path(&path, value)));
    }
    Ok(layers)
}

fn at_path(path: &str, mut value: Value) -> Value {
    for key in path.split('.').rev() {
        value = serde_json::json!({key: value});
    }
    value
}
