//! Input/Output operations for RunMat runtime
//!
//! This module provides language-compatible I/O functions like fprintf, disp, etc.

use regex::Regex;

/// Format and display string with one numeric argument (fprintf with %d, %f, %.4f etc.)
// merged into single fprintf below if needed
pub fn fprintf_format_builtin(format_str: String, value: f64) -> Result<f64, String> {
    // Parse a single numeric placeholder and format accordingly
    let fmt = Regex::new(r"%(?P<prec>\.\d+)?(?P<spec>[df])").unwrap();
    let output = if let Some(caps) = fmt.captures(&format_str) {
        let spec = caps.name("spec").map(|m| m.as_str()).unwrap_or("f");
        let prec = caps
            .name("prec")
            .and_then(|m| m.as_str().strip_prefix('.'))
            .and_then(|n| n.parse::<usize>().ok())
            .unwrap_or(6);
        match spec {
            "d" => format_str.replacen(&caps[0], &format!("{value:.0}"), 1),
            _ => format_str.replacen(&caps[0], &format!("{value:.prec$}"), 1),
        }
    } else {
        format_str.replace("\\n", "\n")
    };

    print!("{output}");
    use std::io::{self, Write};
    io::stdout()
        .flush()
        .map_err(|e| format!("Failed to flush stdout: {e}"))?;
    Ok(output.len() as f64)
}

/// Format and display string with two numeric arguments
pub fn fprintf_format2_builtin(
    format_str: String,
    value1: f64,
    value2: f64,
) -> Result<f64, String> {
    // Replace two placeholders in order using a regex
    let fmt = Regex::new(r"%(?P<prec>\.\d+)?(?P<spec>[df])").unwrap();
    let mut output = format_str;
    for val in [value1, value2] {
        if let Some(caps) = fmt.captures(&output) {
            let spec = caps.name("spec").map(|m| m.as_str()).unwrap_or("f");
            let prec = caps
                .name("prec")
                .and_then(|m| m.as_str().strip_prefix('.'))
                .and_then(|n| n.parse::<usize>().ok())
                .unwrap_or(6);
            let rep = match spec {
                "d" => format!("{val:.0}"),
                _ => format!("{val:.prec$}"),
            };
            output = output.replacen(&caps[0], &rep, 1);
        }
    }
    output = output.replace("\\n", "\n");
    print!("{output}");
    use std::io::{self, Write};
    io::stdout()
        .flush()
        .map_err(|e| format!("Failed to flush stdout: {e}"))?;
    Ok(output.len() as f64)
}
