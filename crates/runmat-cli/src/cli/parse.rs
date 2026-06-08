use crate::cli::{FigureSize, LogLevel};

/// Custom parser for boolean environment variables that accepts both "1"/"0" and "true"/"false"
pub fn parse_bool_env(s: &str) -> Result<bool, String> {
    match s.to_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        "" => Ok(false),
        _ => Err(format!(
            "Invalid boolean value '{s}'. Expected: 1/0, true/false, yes/no, on/off"
        )),
    }
}

/// Custom parser for log level environment variables that handles empty strings
pub fn parse_log_level_env(s: &str) -> Result<LogLevel, String> {
    if s.is_empty() {
        return Ok(LogLevel::Info);
    }

    match s.to_lowercase().as_str() {
        "error" => Ok(LogLevel::Error),
        "warn" => Ok(LogLevel::Warn),
        "info" => Ok(LogLevel::Info),
        "debug" => Ok(LogLevel::Debug),
        "trace" => Ok(LogLevel::Trace),
        _ => Err(format!(
            "Invalid log level '{s}'. Expected: error, warn, info, debug, trace"
        )),
    }
}

pub fn parse_figure_size(s: &str) -> Result<FigureSize, String> {
    let trimmed = s.trim();
    let parts: Vec<&str> = trimmed.split('x').collect();
    if parts.len() != 2 {
        return Err(format!(
            "Invalid figure size '{trimmed}'. Expected WIDTHxHEIGHT (e.g. 1280x720)"
        ));
    }
    let width = parts[0]
        .parse::<u32>()
        .map_err(|_| format!("Invalid figure width '{}'", parts[0]))?;
    let height = parts[1]
        .parse::<u32>()
        .map_err(|_| format!("Invalid figure height '{}'", parts[1]))?;
    if width == 0 || height == 0 {
        return Err("Figure size must be non-zero".to_string());
    }
    Ok(FigureSize { width, height })
}
