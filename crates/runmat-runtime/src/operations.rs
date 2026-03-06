use std::collections::BTreeMap;

use chrono::Utc;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationContext {
    pub trace_id: Option<String>,
    pub request_id: Option<String>,
}

impl OperationContext {
    pub fn new(trace_id: Option<String>, request_id: Option<String>) -> Self {
        Self {
            trace_id,
            request_id,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationEnvelope<T> {
    pub operation: String,
    pub op_version: String,
    pub trace_id: Option<String>,
    pub request_id: Option<String>,
    pub data: T,
}

impl<T> OperationEnvelope<T> {
    pub fn new(operation: &str, op_version: &str, context: &OperationContext, data: T) -> Self {
        Self {
            operation: operation.to_string(),
            op_version: op_version.to_string(),
            trace_id: context.trace_id.clone(),
            request_id: context.request_id.clone(),
            data,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationErrorType {
    Input,
    Validation,
    Capacity,
    Backend,
    Internal,
    Contract,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationErrorSeverity {
    Warning,
    Error,
    Fatal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationErrorEnvelope {
    pub error_code: String,
    pub error_type: OperationErrorType,
    pub message: String,
    pub operation: String,
    pub op_version: String,
    pub retryable: bool,
    pub severity: OperationErrorSeverity,
    pub context: BTreeMap<String, String>,
    pub trace_id: Option<String>,
    pub request_id: Option<String>,
    pub timestamp: String,
}

pub struct OperationErrorSpec<'a> {
    pub error_code: &'a str,
    pub error_type: OperationErrorType,
    pub retryable: bool,
    pub severity: OperationErrorSeverity,
}

pub fn operation_error(
    operation: &str,
    op_version: &str,
    context: &OperationContext,
    spec: OperationErrorSpec<'_>,
    message: impl Into<String>,
    error_context: BTreeMap<String, String>,
) -> OperationErrorEnvelope {
    OperationErrorEnvelope {
        error_code: spec.error_code.to_string(),
        error_type: spec.error_type,
        message: message.into(),
        operation: operation.to_string(),
        op_version: op_version.to_string(),
        retryable: spec.retryable,
        severity: spec.severity,
        context: error_context,
        trace_id: context.trace_id.clone(),
        request_id: context.request_id.clone(),
        timestamp: Utc::now().to_rfc3339(),
    }
}
