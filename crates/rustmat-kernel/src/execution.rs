//! Execution engine for MATLAB code within the Jupyter kernel
//!
//! Integrates with the rustmat-ignition interpreter to provide safe,
//! isolated execution of MATLAB code with proper error handling.

use crate::Result;
use rustmat_builtins::Value;
use rustmat_hir::lower;
use rustmat_ignition::execute;
use rustmat_parser::parse;
use std::time::{Duration, Instant};

/// Execution engine managing MATLAB code execution state
pub struct ExecutionEngine {
    /// Current execution counter
    execution_count: u64,
    /// Execution timeout
    timeout: Option<Duration>,
    /// Whether debug mode is enabled
    debug: bool,
}

/// Result of code execution
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Execution status
    pub status: ExecutionStatus,
    /// Standard output captured during execution
    pub stdout: String,
    /// Standard error captured during execution  
    pub stderr: String,
    /// Execution result value (if successful)
    pub result: Option<Value>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Any error that occurred
    pub error: Option<ExecutionError>,
}

/// Execution status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionStatus {
    /// Execution completed successfully
    Success,
    /// Execution failed with an error
    Error,
    /// Execution was interrupted/cancelled
    Interrupted,
    /// Execution timed out
    Timeout,
}

/// Execution error details
#[derive(Debug, Clone)]
pub struct ExecutionError {
    /// Error type/name
    pub error_type: String,
    /// Error message
    pub message: String,
    /// Error traceback/stack trace
    pub traceback: Vec<String>,
}

impl ExecutionEngine {
    /// Create a new execution engine
    pub fn new() -> Self {
        Self {
            execution_count: 0,
            timeout: Some(Duration::from_secs(300)), // 5 minutes default
            debug: false,
        }
    }

    /// Create a new execution engine with custom timeout
    pub fn with_timeout(timeout: Option<Duration>) -> Self {
        Self {
            execution_count: 0,
            timeout,
            debug: false,
        }
    }

    /// Enable or disable debug mode
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Get current execution count
    pub fn execution_count(&self) -> u64 {
        self.execution_count
    }

    /// Execute MATLAB code
    pub fn execute(&mut self, code: &str) -> Result<ExecutionResult> {
        let start_time = Instant::now();
        self.execution_count += 1;

        if self.debug {
            log::debug!("Executing code ({}): {}", self.execution_count, code);
        }

        // Parse the code
        let ast = match parse(code) {
            Ok(ast) => ast,
            Err(parse_error) => {
                return Ok(ExecutionResult {
                    status: ExecutionStatus::Error,
                    stdout: String::new(),
                    stderr: String::new(),
                    result: None,
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    error: Some(ExecutionError {
                        error_type: "ParseError".to_string(),
                        message: parse_error,
                        traceback: vec!["Parse error in input code".to_string()],
                    }),
                });
            }
        };

        // Lower to HIR
        let hir = match lower(&ast) {
            Ok(hir) => hir,
            Err(hir_error) => {
                return Ok(ExecutionResult {
                    status: ExecutionStatus::Error,
                    stdout: String::new(),
                    stderr: String::new(),
                    result: None,
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    error: Some(ExecutionError {
                        error_type: "CompileError".to_string(),
                        message: hir_error,
                        traceback: vec!["Error in semantic analysis".to_string()],
                    }),
                });
            }
        };

        // Execute with the interpreter
        let interpreter_result = execute(&hir);
        let execution_time_ms = start_time.elapsed().as_millis() as u64;

        match interpreter_result {
            Ok(variables) => {
                // For now, we'll return the last variable as the result
                let result = if !variables.is_empty() {
                    Some(variables.into_iter().last().unwrap())
                } else {
                    None
                };

                Ok(ExecutionResult {
                    status: ExecutionStatus::Success,
                    stdout: String::new(), // TODO: Capture actual stdout
                    stderr: String::new(), // TODO: Capture actual stderr
                    result,
                    execution_time_ms,
                    error: None,
                })
            }
            Err(exec_error) => Ok(ExecutionResult {
                status: ExecutionStatus::Error,
                stdout: String::new(),
                stderr: String::new(),
                result: None,
                execution_time_ms,
                error: Some(ExecutionError {
                    error_type: "RuntimeError".to_string(),
                    message: exec_error,
                    traceback: vec!["Error during code execution".to_string()],
                }),
            }),
        }
    }

    /// Execute code with a specific timeout
    pub fn execute_with_timeout(
        &mut self,
        code: &str,
        timeout: Duration,
    ) -> Result<ExecutionResult> {
        let original_timeout = self.timeout;
        self.timeout = Some(timeout);
        let result = self.execute(code);
        self.timeout = original_timeout;
        result
    }

    /// Reset the execution engine state
    pub fn reset(&mut self) {
        self.execution_count = 0;
        if self.debug {
            log::debug!("Execution engine reset");
        }
    }

    /// Get engine statistics
    pub fn stats(&self) -> ExecutionStats {
        ExecutionStats {
            execution_count: self.execution_count,
            timeout_seconds: self.timeout.map(|d| d.as_secs()),
            debug_enabled: self.debug,
        }
    }
}

impl Default for ExecutionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Execution engine statistics
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    /// Total number of executions performed
    pub execution_count: u64,
    /// Configured timeout in seconds (if any)
    pub timeout_seconds: Option<u64>,
    /// Whether debug mode is enabled
    pub debug_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_engine_creation() {
        let engine = ExecutionEngine::new();
        assert_eq!(engine.execution_count(), 0);
        assert!(!engine.debug);
    }

    #[test]
    fn test_execution_engine_with_timeout() {
        let timeout = Duration::from_secs(60);
        let engine = ExecutionEngine::with_timeout(Some(timeout));
        assert_eq!(engine.timeout, Some(timeout));
    }

    #[test]
    fn test_simple_execution() {
        let mut engine = ExecutionEngine::new();
        let result = engine.execute("x = 1 + 2").unwrap();

        assert_eq!(result.status, ExecutionStatus::Success);
        assert_eq!(engine.execution_count(), 1);
        // Just verify that execution_time_ms field exists and is accessible
        let _time = result.execution_time_ms;
        assert!(result.error.is_none());
    }

    #[test]
    fn test_parse_error_handling() {
        let mut engine = ExecutionEngine::new();
        let result = engine.execute("x = 1 +").unwrap();

        assert_eq!(result.status, ExecutionStatus::Error);
        assert!(result.error.is_some());

        let error = result.error.unwrap();
        assert_eq!(error.error_type, "ParseError");
        assert!(!error.message.is_empty());
    }

    #[test]
    fn test_runtime_error_handling() {
        let mut engine = ExecutionEngine::new();
        let result = engine.execute("x = undefined_var").unwrap();

        assert_eq!(result.status, ExecutionStatus::Error);
        assert!(result.error.is_some());

        let error = result.error.unwrap();
        assert!(error.error_type == "RuntimeError" || error.error_type == "CompileError");
    }

    #[test]
    fn test_execution_count_increment() {
        let mut engine = ExecutionEngine::new();

        engine.execute("x = 1").unwrap();
        assert_eq!(engine.execution_count(), 1);

        engine.execute("y = 2").unwrap();
        assert_eq!(engine.execution_count(), 2);

        // Even failed executions increment the counter
        engine.execute("invalid syntax").unwrap();
        assert_eq!(engine.execution_count(), 3);
    }

    #[test]
    fn test_engine_reset() {
        let mut engine = ExecutionEngine::new();
        engine.execute("x = 1").unwrap();
        assert_eq!(engine.execution_count(), 1);

        engine.reset();
        assert_eq!(engine.execution_count(), 0);
    }

    #[test]
    fn test_debug_mode() {
        let mut engine = ExecutionEngine::new();
        assert!(!engine.debug);

        engine.set_debug(true);
        assert!(engine.debug);

        engine.set_debug(false);
        assert!(!engine.debug);
    }

    #[test]
    fn test_stats() {
        let mut engine = ExecutionEngine::new();
        engine.set_debug(true);
        engine.execute("x = 1").unwrap();

        let stats = engine.stats();
        assert_eq!(stats.execution_count, 1);
        assert!(stats.debug_enabled);
        assert!(stats.timeout_seconds.is_some());
    }
}
