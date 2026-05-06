use crate::{
    AccelPowerPreference, AccelerateProviderPreference, AutoOffloadLogLevel, TelemetryDrainMode,
};

/// Parse a boolean value from string with various formats
pub(crate) fn parse_bool(s: &str) -> Option<bool> {
    match s.to_lowercase().as_str() {
        "1" | "true" | "yes" | "on" | "enable" | "enabled" => Some(true),
        "0" | "false" | "no" | "off" | "disable" | "disabled" => Some(false),
        "" => Some(false),
        _ => None,
    }
}

pub(crate) fn parse_auto_offload_log_level(value: &str) -> Option<AutoOffloadLogLevel> {
    match value.trim().to_ascii_lowercase().as_str() {
        "off" => Some(AutoOffloadLogLevel::Off),
        "info" => Some(AutoOffloadLogLevel::Info),
        "trace" => Some(AutoOffloadLogLevel::Trace),
        _ => None,
    }
}

pub(crate) fn parse_provider_preference(value: &str) -> Option<AccelerateProviderPreference> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => Some(AccelerateProviderPreference::Auto),
        "wgpu" => Some(AccelerateProviderPreference::Wgpu),
        "inprocess" | "cpu" | "host" => Some(AccelerateProviderPreference::InProcess),
        _ => None,
    }
}

pub(crate) fn parse_power_preference(value: &str) -> Option<AccelPowerPreference> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => Some(AccelPowerPreference::Auto),
        "high" | "highperformance" | "performance" => Some(AccelPowerPreference::HighPerformance),
        "low" | "lowpower" | "battery" => Some(AccelPowerPreference::LowPower),
        _ => None,
    }
}

pub(crate) fn parse_telemetry_drain_mode(value: &str) -> Option<TelemetryDrainMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "started" | "start" | "session" | "all" | "full" | "both" | "runtime" => {
            Some(TelemetryDrainMode::All)
        }
        "none" | "off" | "" => Some(TelemetryDrainMode::None),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_auto_offload_log_level_cases() {
        assert_eq!(
            parse_auto_offload_log_level("off"),
            Some(AutoOffloadLogLevel::Off)
        );
        assert_eq!(
            parse_auto_offload_log_level("INFO"),
            Some(AutoOffloadLogLevel::Info)
        );
        assert_eq!(
            parse_auto_offload_log_level("trace"),
            Some(AutoOffloadLogLevel::Trace)
        );
        assert_eq!(parse_auto_offload_log_level("unknown"), None);
    }

    #[test]
    fn bool_parsing() {
        assert_eq!(parse_bool("true"), Some(true));
        assert_eq!(parse_bool("1"), Some(true));
        assert_eq!(parse_bool("yes"), Some(true));
        assert_eq!(parse_bool("false"), Some(false));
        assert_eq!(parse_bool("0"), Some(false));
        assert_eq!(parse_bool("invalid"), None);
    }
}
