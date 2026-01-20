//! Execution engine for RunMat code within the Jupyter kernel
//!
//! Provides a kernel-specific wrapper around the RunMatSession to adapt
//! its interface for Jupyter protocol requirements.

use crate::Result;
use runmat_builtins::Value;
use runmat_core::{RunError, RunMatSession};
use runmat_time::Instant;
use std::path::Path;
use std::time::Duration;

/// Execution engine managing RunMat code execution state for the Jupyter kernel
pub struct ExecutionEngine {
    /// Current execution counter
    execution_count: u64,
    /// Execution timeout
    timeout: Option<Duration>,
    /// Whether debug mode is enabled
    debug: bool,
    /// Underlying REPL engine that does the actual execution
    repl_engine: RunMatSession,
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
        let repl_engine =
            RunMatSession::with_options(true, false).expect("Failed to create RunMatSession");
        Self {
            execution_count: 0,
            timeout: Some(Duration::from_secs(300)), // 5 minutes default
            debug: false,
            repl_engine,
        }
    }

    /// Create a new execution engine with custom timeout
    pub fn with_timeout(timeout: Option<Duration>) -> Self {
        let repl_engine =
            RunMatSession::with_options(true, false).expect("Failed to create RunMatSession");
        Self {
            execution_count: 0,
            timeout,
            debug: false,
            repl_engine,
        }
    }

    /// Create a new execution engine with specific options
    pub fn with_options(enable_jit: bool, debug: bool, timeout: Option<Duration>) -> Result<Self> {
        Self::with_snapshot(enable_jit, debug, timeout, None::<&str>)
    }

    /// Create a new execution engine with snapshot support
    pub fn with_snapshot<P: AsRef<Path>>(
        enable_jit: bool,
        debug: bool,
        timeout: Option<Duration>,
        snapshot_path: Option<P>,
    ) -> Result<Self> {
        let repl_engine =
            RunMatSession::with_snapshot(enable_jit, debug, snapshot_path).map_err(|e| {
                crate::KernelError::Internal(format!("Failed to create RunMatSession: {e}"))
            })?;
        Ok(Self {
            execution_count: 0,
            timeout,
            debug,
            repl_engine,
        })
    }

    /// Enable or disable debug mode
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Get current execution count
    pub fn execution_count(&self) -> u64 {
        self.execution_count
    }

    /// Execute code
    pub async fn execute(&mut self, code: &str) -> Result<ExecutionResult> {
        let start_time = Instant::now();
        self.execution_count += 1;

        if self.debug {
            log::debug!("Executing code ({}): {}", self.execution_count, code);
        }

        // Execute using the underlying RunMatSession
        match self.repl_engine.execute(code).await {
            Ok(repl_result) => {
                let execution_time_ms = start_time.elapsed().as_millis() as u64;

                if let Some(error) = repl_result.error {
                    let error_message = error.format_diagnostic();
                    let traceback = if error.context.call_stack.is_empty() {
                        vec!["Error during code execution".to_string()]
                    } else {
                        error.context.call_stack.clone()
                    };

                    Ok(ExecutionResult {
                        status: ExecutionStatus::Error,
                        stdout: self.capture_stdout(),
                        stderr: self.capture_stderr(&error_message),
                        result: None,
                        execution_time_ms,
                        error: Some(ExecutionError {
                            error_type: "RuntimeError".to_string(),
                            message: error_message,
                            traceback,
                        }),
                    })
                } else {
                    Ok(ExecutionResult {
                        status: ExecutionStatus::Success,
                        stdout: self.capture_stdout(),
                        stderr: String::new(), // No errors on success
                        result: repl_result.value,
                        execution_time_ms,
                        error: None,
                    })
                }
            }
            Err(e) => {
                let execution_time_ms = start_time.elapsed().as_millis() as u64;
                let (error_type, message) = match e {
                    RunError::Syntax(err) => ("SyntaxError", err.to_string()),
                    RunError::Semantic(err) => ("SemanticError", err.to_string()),
                    RunError::Compile(err) => ("CompileError", err.to_string()),
                    RunError::Runtime(err) => ("RuntimeError", err.format_diagnostic()),
                };

                Ok(ExecutionResult {
                    status: ExecutionStatus::Error,
                    stdout: String::new(),
                    stderr: String::new(),
                    result: None,
                    execution_time_ms,
                    error: Some(ExecutionError {
                        error_type: error_type.to_string(),
                        message,
                        traceback: vec!["Error during code execution".to_string()],
                    }),
                })
            }
        }
    }

    /// Execute code with a specific timeout
    pub async fn execute_with_timeout(
        &mut self,
        code: &str,
        timeout: Duration,
    ) -> Result<ExecutionResult> {
        let original_timeout = self.timeout;
        self.timeout = Some(timeout);
        let result = self.execute(code).await;
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
        let repl_stats = self.repl_engine.stats();
        ExecutionStats {
            execution_count: self.execution_count,
            timeout_seconds: self.timeout.map(|d| d.as_secs()),
            debug_enabled: self.debug,
            repl_total_executions: repl_stats.total_executions,
            repl_jit_compiled: repl_stats.jit_compiled,
            repl_interpreter_fallback: repl_stats.interpreter_fallback,
            repl_average_time_ms: repl_stats.average_execution_time_ms,
        }
    }

    /// Get snapshot information
    pub fn snapshot_info(&self) -> Option<String> {
        self.repl_engine.snapshot_info()
    }

    /// Check if a snapshot is loaded
    pub fn has_snapshot(&self) -> bool {
        self.repl_engine.has_snapshot()
    }

    /// Capture stdout output from the REPL execution
    fn capture_stdout(&self) -> String {
        // In a production implementation, this would capture actual stdout
        // For now, we simulate by checking if there were any successful results
        // The REPL itself doesn't currently emit to stdout, but this would be the place
        // to capture print() function output or similar
        String::new()
    }

    /// Capture stderr output, including error messages
    fn capture_stderr(&self, error_msg: &str) -> String {
        // Format error message for stderr
        format!("Error: {error_msg}")
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
    /// Total number of executions performed by the kernel
    pub execution_count: u64,
    /// Configured timeout in seconds (if any)
    pub timeout_seconds: Option<u64>,
    /// Whether debug mode is enabled
    pub debug_enabled: bool,
    /// Total executions performed by the underlying REPL engine
    pub repl_total_executions: usize,
    /// Number of JIT compiled executions
    pub repl_jit_compiled: usize,
    /// Number of interpreter fallback executions
    pub repl_interpreter_fallback: usize,
    /// Average execution time in milliseconds
    pub repl_average_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

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
        let result = block_on(engine.execute("x = 1 + 2")).unwrap();

        assert_eq!(result.status, ExecutionStatus::Success);
        assert_eq!(engine.execution_count(), 1);
        // Just verify that execution_time_ms field exists and is accessible
        let _time = result.execution_time_ms;
        assert!(result.error.is_none());
    }

    #[test]
    fn test_parse_error_handling() {
        let mut engine = ExecutionEngine::new();
        let result = block_on(engine.execute("x = 1 +")).unwrap();

        assert_eq!(result.status, ExecutionStatus::Error);
        assert!(result.error.is_some());

        let error = result.error.unwrap();
        assert_eq!(error.error_type,                     "SyntaxError"
);
        assert!(!error.message.is_empty());
    }

    #[test]
    fn test_runtime_error_handling() {
        let mut engine = ExecutionEngine::new();
        let result = block_on(engine.execute("x = undefined_var")).unwrap();

        assert_eq!(result.status, ExecutionStatus::Error);
        assert!(result.error.is_some());

        let error = result.error.unwrap();
        assert!(error.error_type == "RuntimeError" || error.error_type == "CompileError");
    }

    #[test]
    fn test_execution_count_increment() {
        let mut engine = ExecutionEngine::new();

        block_on(engine.execute("x = 1")).unwrap();
        assert_eq!(engine.execution_count(), 1);

        block_on(engine.execute("y = 2")).unwrap();
        assert_eq!(engine.execution_count(), 2);

        // Even failed executions increment the counter
        block_on(engine.execute("invalid syntax")).unwrap();
        assert_eq!(engine.execution_count(), 3);
    }

    #[test]
    fn test_engine_reset() {
        let mut engine = ExecutionEngine::new();
        block_on(engine.execute("x = 1")).unwrap();
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
        block_on(engine.execute("x = 1")).unwrap();

        let stats = engine.stats();
        assert_eq!(stats.execution_count, 1);
        assert!(stats.debug_enabled);
        assert!(stats.timeout_seconds.is_some());
    }
}
