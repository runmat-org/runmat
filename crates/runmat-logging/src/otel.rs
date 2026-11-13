//! User-owned observability configuration. Secrets deliberately have no serialization API.
use std::collections::BTreeMap;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OtelProtocol {
    HttpProtobuf,
    Grpc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OtelSignal {
    Traces,
    Logs,
    Metrics,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OtelOptions {
    pub enabled: bool,
    pub endpoint: String,
    pub protocol: OtelProtocol,
    pub signals: Vec<OtelSignal>,
    pub headers_from_env: Option<String>,
    pub service_name: String,
    pub resource_attributes: BTreeMap<String, String>,
    pub queue_capacity: usize,
    pub timeout: Duration,
}

impl Default for OtelOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: "http://localhost:4318".into(),
            protocol: OtelProtocol::HttpProtobuf,
            signals: vec![OtelSignal::Traces, OtelSignal::Logs],
            headers_from_env: Some("RUNMAT_OTEL_HEADERS".into()),
            service_name: "runmat".into(),
            resource_attributes: BTreeMap::new(),
            queue_capacity: 2048,
            timeout: Duration::from_secs(5),
        }
    }
}

/// Runtime-only keychain/environment values. Debug output always redacts them.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct StartupHeaders(pub BTreeMap<String, String>);

impl std::fmt::Debug for StartupHeaders {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StartupHeaders([REDACTED])")
    }
}

#[cfg(any(feature = "otlp", feature = "browser-otlp"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SignalOptions {
    pub endpoint: String,
    pub protocol: OtelProtocol,
    pub headers: StartupHeaders,
    pub timeout: Duration,
}

impl OtelOptions {
    /// Environment is consulted once at startup, never written into settings.
    /// The host's keychain headers take precedence over environment headers.
    pub fn resolve_environment(mut self) -> Result<Self, String> {
        self.resolve_with(|key| std::env::var(key).ok())?;
        Ok(self)
    }

    fn resolve_with(&mut self, env: impl Fn(&str) -> Option<String>) -> Result<(), String> {
        if [
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
        ]
        .iter()
        .any(|key| env(key).is_some_and(|value| !value.trim().is_empty()))
        {
            self.enabled = true;
        }
        if let Some(endpoint) = env("OTEL_EXPORTER_OTLP_ENDPOINT").filter(|v| !v.is_empty()) {
            self.endpoint = endpoint;
            self.enabled = true;
        }
        if let Some(endpoint) = env("RUNMAT_OTEL_ENDPOINT").filter(|v| !v.trim().is_empty()) {
            self.endpoint = endpoint;
            self.enabled = true;
        }
        if let Some(protocol) = env("OTEL_EXPORTER_OTLP_PROTOCOL") {
            self.protocol = parse_protocol(&protocol)?;
        }
        if let Some(name) = env("OTEL_SERVICE_NAME").filter(|v| !v.is_empty()) {
            self.service_name = name;
        }
        if let Some(attrs) = env("OTEL_RESOURCE_ATTRIBUTES") {
            self.resource_attributes.extend(parse_pairs(&attrs)?);
        }
        // Environment-only child hosts have no ConfigLoader. Preserve explicit opt-out
        // even when a collector endpoint is inherited from the parent shell.
        for key in [
            "RUNMAT_OTEL_ENABLED",
            "RUNMAT_RUNTIME__LOGGING__OTEL__ENABLED",
        ] {
            if let Some(value) = env(key) {
                self.enabled = match value.as_str() {
                    "true" | "1" => true,
                    "false" | "0" => false,
                    _ => {
                        return Err(
                            "OpenTelemetry enabled environment value must be true or false".into(),
                        )
                    }
                };
            }
        }
        Ok(())
    }

    #[cfg(any(feature = "otlp", feature = "browser-otlp"))]
    pub(crate) fn signal_options(
        &self,
        signal: &str,
        host: &StartupHeaders,
    ) -> Result<SignalOptions, String> {
        if host.0.len() > 16
            || host
                .0
                .iter()
                .try_fold(0usize, |size, (key, value)| {
                    size.checked_add(key.len())?.checked_add(value.len())
                })
                .is_none_or(|size| size > 8192)
        {
            return Err("OpenTelemetry headers exceed their bound".into());
        }
        let lookup = |suffix: &str| {
            std::env::var(format!("OTEL_EXPORTER_OTLP_{signal}_{suffix}"))
                .ok()
                .or_else(|| std::env::var(format!("OTEL_EXPORTER_OTLP_{suffix}")).ok())
        };
        let protocol = lookup("PROTOCOL")
            .map(|v| parse_protocol(&v))
            .transpose()?
            .unwrap_or(self.protocol);
        let specific_endpoint = std::env::var(format!("OTEL_EXPORTER_OTLP_{signal}_ENDPOINT")).ok();
        let endpoint = specific_endpoint.unwrap_or_else(|| match protocol {
            OtelProtocol::HttpProtobuf => format!(
                "{}/v1/{}",
                self.endpoint.trim_end_matches('/'),
                signal.to_ascii_lowercase()
            ),
            OtelProtocol::Grpc => self.endpoint.clone(),
        });
        let timeout = lookup("TIMEOUT")
            .map(|v| {
                v.parse::<u64>()
                    .map(Duration::from_millis)
                    .map_err(|_| "OTLP timeout must be milliseconds".to_string())
            })
            .transpose()?
            .unwrap_or(self.timeout);
        let mut headers = lookup("HEADERS")
            .map(|value| {
                if value.len() <= 8192 {
                    Ok(value)
                } else {
                    Err("OpenTelemetry headers exceed their byte bound".to_string())
                }
            })
            .transpose()?
            .map(|v| parse_pairs(&v))
            .transpose()?
            .unwrap_or_default();
        if let Some(value) = self
            .headers_from_env
            .as_ref()
            .and_then(|key| std::env::var(key).ok())
        {
            if value.len() > 8192 {
                return Err("OpenTelemetry headers exceed their byte bound".into());
            }
            headers.extend(parse_pairs(&value)?);
        }
        if timeout.is_zero() || timeout > Duration::from_secs(30) {
            return Err("OpenTelemetry timeout must be between 1 and 30000 milliseconds".into());
        }
        let combined_size =
            headers
                .iter()
                .chain(host.0.iter())
                .try_fold(0usize, |size, (key, value)| {
                    if key.len() > 128 || value.len() > 4096 {
                        return None;
                    }
                    size.checked_add(key.len())?.checked_add(value.len())
                });
        if headers.len() + host.0.len() > 16 || combined_size.is_none_or(|size| size > 8192) {
            return Err("OpenTelemetry headers exceed their aggregate bound".into());
        }
        headers.extend(host.0.clone());
        Ok(SignalOptions {
            endpoint,
            protocol,
            headers: StartupHeaders(headers),
            timeout,
        })
    }
}

pub(crate) fn parse_protocol(value: &str) -> Result<OtelProtocol, String> {
    match value {
        "http/protobuf" => Ok(OtelProtocol::HttpProtobuf),
        "grpc" => Ok(OtelProtocol::Grpc),
        _ => Err("OTLP protocol must be http/protobuf or grpc".into()),
    }
}

fn parse_pairs(value: &str) -> Result<BTreeMap<String, String>, String> {
    value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            let (key, value) = part
                .trim()
                .split_once('=')
                .ok_or("invalid OTEL key/value list")?;
            if key.trim().is_empty() {
                return Err("empty OTEL key".into());
            }
            Ok((decode(key.trim())?, decode(value.trim())?))
        })
        .collect()
}

fn decode(value: &str) -> Result<String, String> {
    let mut bytes = Vec::with_capacity(value.len());
    let mut input = value.as_bytes().iter().copied();
    while let Some(byte) = input.next() {
        if byte == b'%' {
            let hi = input.next().and_then(|v| (v as char).to_digit(16));
            let lo = input.next().and_then(|v| (v as char).to_digit(16));
            bytes.push(match (hi, lo) {
                (Some(hi), Some(lo)) => (hi * 16 + lo) as u8,
                _ => return Err("invalid OTEL percent encoding".into()),
            });
        } else {
            bytes.push(byte);
        }
    }
    String::from_utf8(bytes).map_err(|_| "invalid OTEL UTF-8".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shortcut_wins_standard_environment() {
        let mut options = OtelOptions::default();
        options
            .resolve_with(|key| match key {
                "RUNMAT_OTEL_ENDPOINT" => Some("http://shortcut:4318".into()),
                "OTEL_EXPORTER_OTLP_ENDPOINT" => Some("http://standard:4318".into()),
                "OTEL_SERVICE_NAME" => Some("worker".into()),
                _ => None,
            })
            .unwrap();
        assert!(options.enabled);
        assert_eq!(options.endpoint, "http://shortcut:4318");
        assert_eq!(options.service_name, "worker");
    }
    #[test]
    fn headers_are_decoded_and_redacted() {
        let pairs = parse_pairs("authorization=Bearer%20secret,k=a=b").unwrap();
        assert_eq!(pairs["authorization"], "Bearer secret");
        assert_eq!(pairs["k"], "a=b");
        assert!(!format!("{:?}", StartupHeaders(pairs)).contains("secret"));
    }
}
