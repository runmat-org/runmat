use runmat_hir::SemanticError;
use runmat_parser::SyntaxError;
use runmat_runtime::RuntimeError;
use runmat_vm::CompileError;

use crate::telemetry::TelemetryFailureInfo;

#[derive(Debug)]
pub enum RunError {
    Syntax(SyntaxError),
    Semantic(SemanticError),
    Compile(CompileError),
    Runtime(RuntimeError),
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::Syntax(err) => write!(f, "{err}"),
            RunError::Semantic(err) => write!(f, "{err}"),
            RunError::Compile(err) => write!(f, "{err}"),
            RunError::Runtime(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for RunError {}

impl From<SyntaxError> for RunError {
    fn from(value: SyntaxError) -> Self {
        RunError::Syntax(value)
    }
}

impl From<SemanticError> for RunError {
    fn from(value: SemanticError) -> Self {
        RunError::Semantic(value)
    }
}

impl From<CompileError> for RunError {
    fn from(value: CompileError) -> Self {
        RunError::Compile(value)
    }
}

impl From<RuntimeError> for RunError {
    fn from(value: RuntimeError) -> Self {
        RunError::Runtime(value)
    }
}

impl RunError {
    pub fn telemetry_failure_info(&self) -> TelemetryFailureInfo {
        match self {
            RunError::Syntax(_err) => TelemetryFailureInfo {
                stage: "parser".to_string(),
                code: "RunMat:ParserError".to_string(),
                has_span: true,
                component: Some("unknown".to_string()),
            },
            RunError::Semantic(err) => TelemetryFailureInfo {
                stage: "hir".to_string(),
                code: err
                    .identifier
                    .clone()
                    .unwrap_or_else(|| "RunMat:SemanticError".to_string()),
                has_span: err.span.is_some(),
                component: telemetry_component_for_identifier(err.identifier.as_deref()),
            },
            RunError::Compile(err) => TelemetryFailureInfo {
                stage: "compile".to_string(),
                code: err
                    .identifier
                    .clone()
                    .unwrap_or_else(|| "RunMat:CompileError".to_string()),
                has_span: err.span.is_some(),
                component: telemetry_component_for_identifier(err.identifier.as_deref()),
            },
            RunError::Runtime(err) => runtime_error_telemetry_failure_info(err),
        }
    }
}

pub fn runtime_error_telemetry_failure_info(err: &RuntimeError) -> TelemetryFailureInfo {
    let identifier = err
        .identifier()
        .map(|value| value.to_string())
        .unwrap_or_else(|| "RunMat:RuntimeError".to_string());
    TelemetryFailureInfo {
        stage: "runtime".to_string(),
        code: identifier.clone(),
        has_span: err.span.is_some(),
        component: telemetry_component_for_identifier(Some(identifier.as_str())),
    }
}

fn telemetry_component_for_identifier(identifier: Option<&str>) -> Option<String> {
    let lower = identifier?.to_ascii_lowercase();
    if lower.contains("undefined") || lower.contains("name") || lower.contains("import") {
        return Some("name_resolution".to_string());
    }
    if lower.contains("type") || lower.contains("dimension") || lower.contains("bounds") {
        return Some("typecheck".to_string());
    }
    if lower.contains("cancel") || lower.contains("interrupt") {
        return Some("cancellation".to_string());
    }
    if lower.contains("io") || lower.contains("filesystem") {
        return Some("io".to_string());
    }
    if lower.contains("network") || lower.contains("timeout") {
        return Some("network".to_string());
    }
    if lower.contains("internal") || lower.contains("panic") {
        return Some("internal".to_string());
    }
    None
}
