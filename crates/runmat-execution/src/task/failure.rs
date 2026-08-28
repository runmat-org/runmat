use serde::{Deserialize, Serialize};

use crate::ContractError;

pub const MAX_PROGRAM_FAILURE_FRAMES: usize = 4096;
const MAX_PROGRAM_FAILURE_MESSAGE_BYTES: usize = 1024 * 1024;
const MAX_PROGRAM_FAILURE_FIELD_BYTES: usize = 64 * 1024;

fn valid_text(value: &str, max_bytes: usize) -> bool {
    !value.is_empty() && value.len() <= max_bytes && !value.contains('\0')
}

fn valid_optional_text(value: Option<&str>, max_bytes: usize) -> bool {
    value.is_none_or(|value| valid_text(value, max_bytes))
}

fn valid_span(span: &ProgramSourceSpan) -> bool {
    span.offset.checked_add(span.length).is_some()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramSourceSpan {
    pub offset: u64,
    pub length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramCallFrame {
    pub function: String,
    pub source_id: Option<u64>,
    pub span: Option<ProgramSourceSpan>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramRuntimeFailure {
    pub message: String,
    pub identifier: Option<String>,
    pub span: Option<ProgramSourceSpan>,
    pub builtin: Option<String>,
    pub task_id: Option<String>,
    pub call_frames: Vec<ProgramCallFrame>,
    pub call_frames_elided: u32,
    pub call_stack: Vec<String>,
    pub phase: Option<String>,
}

impl ProgramRuntimeFailure {
    pub fn validate(&self) -> Result<(), ContractError> {
        if !valid_text(&self.message, MAX_PROGRAM_FAILURE_MESSAGE_BYTES)
            || self.call_frames.len() > MAX_PROGRAM_FAILURE_FRAMES
            || self.call_stack.len() > MAX_PROGRAM_FAILURE_FRAMES
            || !valid_optional_text(self.identifier.as_deref(), 4096)
            || !valid_optional_text(self.builtin.as_deref(), MAX_PROGRAM_FAILURE_FIELD_BYTES)
            || !valid_optional_text(self.task_id.as_deref(), MAX_PROGRAM_FAILURE_FIELD_BYTES)
            || !valid_optional_text(self.phase.as_deref(), MAX_PROGRAM_FAILURE_FIELD_BYTES)
            || self.span.as_ref().is_some_and(|span| !valid_span(span))
            || self
                .call_stack
                .iter()
                .any(|frame| !valid_text(frame, MAX_PROGRAM_FAILURE_FIELD_BYTES))
            || self.call_frames.iter().any(|frame| {
                !valid_text(&frame.function, MAX_PROGRAM_FAILURE_FIELD_BYTES)
                    || frame.span.as_ref().is_some_and(|span| !valid_span(span))
            })
        {
            return Err(ContractError::invalid(
                "program runtime failure",
                "failure exceeds its structural bounds",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ProgramCallFrame, ProgramRuntimeFailure, ProgramSourceSpan};

    fn failure() -> ProgramRuntimeFailure {
        ProgramRuntimeFailure {
            message: "worker failed".into(),
            identifier: Some("RunMat:WorkerFailure".into()),
            span: Some(ProgramSourceSpan {
                offset: 4,
                length: 2,
            }),
            builtin: None,
            task_id: None,
            call_frames: vec![ProgramCallFrame {
                function: "work".into(),
                source_id: Some(1),
                span: None,
            }],
            call_frames_elided: 0,
            call_stack: vec!["work at source.m:1:1".into()],
            phase: Some("execute".into()),
        }
    }

    #[test]
    fn rejects_embedded_nul_text() {
        let mut failure = failure();
        failure.call_stack[0].push('\0');
        assert!(failure.validate().is_err());
    }

    #[test]
    fn rejects_overflowing_source_spans() {
        let mut failure = failure();
        failure.span = Some(ProgramSourceSpan {
            offset: u64::MAX,
            length: 1,
        });
        assert!(failure.validate().is_err());
    }

    #[test]
    fn rejects_excessive_callstacks() {
        let mut failure = failure();
        failure.call_stack = vec!["frame".into(); super::MAX_PROGRAM_FAILURE_FRAMES + 1];
        assert!(failure.validate().is_err());
    }
}
