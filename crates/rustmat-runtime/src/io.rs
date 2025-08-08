//! Input/Output operations for RustMat runtime
//! 
//! This module provides MATLAB-compatible I/O functions like fprintf, disp, etc.

use rustmat_macros::runtime_builtin;
use regex::Regex;
use std::sync::{Mutex, OnceLock};

/// Display a string to the console (MATLAB fprintf with single string argument)
#[runtime_builtin(name = "fprintf")]
pub fn fprintf_string_builtin(format_str: String) -> Result<f64, String> {
    print!("{}", format_str);
    use std::io::{self, Write};
    io::stdout().flush().map_err(|e| format!("Failed to flush stdout: {}", e))?;
    Ok(format_str.len() as f64) // fprintf returns number of characters written
}

/// Format and display string with one numeric argument (fprintf with %d, %f, %.4f etc.)
#[runtime_builtin(name = "fprintf")]
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
            "d" => format_str.replacen(&caps[0], &format!("{:.0}", value), 1),
            _ => format_str.replacen(&caps[0], &format!("{:.*}", prec, value), 1),
        }
    } else {
        format_str.replace("\\n", "\n")
    };
    
    print!("{}", output);
    use std::io::{self, Write};
    io::stdout().flush().map_err(|e| format!("Failed to flush stdout: {}", e))?;
    Ok(output.len() as f64)
}

/// Format and display string with two numeric arguments
#[runtime_builtin(name = "fprintf")]
pub fn fprintf_format2_builtin(format_str: String, value1: f64, value2: f64) -> Result<f64, String> {
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
                "d" => format!("{:.0}", val),
                _ => format!("{:.*}", prec, val),
            };
            output = output.replacen(&caps[0], &rep, 1);
        }
    }
    output = output.replace("\\n", "\n");
    print!("{}", output);
    use std::io::{self, Write};
    io::stdout().flush().map_err(|e| format!("Failed to flush stdout: {}", e))?;
    Ok(output.len() as f64)
}

/// Display a string with automatic newline (MATLAB disp)
#[runtime_builtin(name = "disp")]
pub fn disp_string_builtin(s: String) -> Result<f64, String> {
    println!("{}", s);
    Ok(0.0)
}

/// Display a number with automatic newline
#[runtime_builtin(name = "disp")]
pub fn disp_number_builtin(n: f64) -> Result<f64, String> {
    println!("{}", n);
    Ok(0.0)
}

// Global timer state for tic/toc functionality
static TIMER_START: OnceLock<Mutex<Option<std::time::Instant>>> = OnceLock::new();

/// Start a stopwatch timer (MATLAB tic function)
#[runtime_builtin(name = "tic")]
pub fn tic_builtin() -> Result<f64, String> {
    let timer = TIMER_START.get_or_init(|| Mutex::new(None));
    let mut start_time = timer.lock().map_err(|_| "Failed to acquire timer lock")?;
    *start_time = Some(std::time::Instant::now());
    Ok(0.0) // tic returns 0 in MATLAB
}

/// Read elapsed time from stopwatch (MATLAB toc function)
#[runtime_builtin(name = "toc")]
pub fn toc_builtin() -> Result<f64, String> {
    let timer = TIMER_START.get_or_init(|| Mutex::new(None));
    let start_time = timer.lock().map_err(|_| "Failed to acquire timer lock")?;
    
    match *start_time {
        Some(start) => {
            let elapsed = start.elapsed().as_secs_f64();
            Ok(elapsed)
        }
        None => Err("tic must be called before toc".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fprintf() {
        let result = fprintf_string_builtin("Hello, world!".to_string());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 13.0);
    }
    
    #[test]
    fn test_disp_string() {
        let result = disp_string_builtin("Test message".to_string());
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_disp_number() {
        let result = disp_number_builtin(3.14159);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_tic_toc() {
        // Test tic
        let result = tic_builtin();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0.0);
        
        // Small delay
        std::thread::sleep(std::time::Duration::from_millis(10));
        
        // Test toc
        let result = toc_builtin();
        assert!(result.is_ok());
        let elapsed = result.unwrap();
        assert!(elapsed >= 0.01); // At least 10ms
        assert!(elapsed < 1.0);    // Less than 1 second
    }
}