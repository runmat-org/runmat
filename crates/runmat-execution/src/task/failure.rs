use serde::{Deserialize, Serialize};

use crate::ContractError;
use runmat_types::ProgramSourceId;

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
    /// ProgramSourceId ordinal in the admitted artifact source map, never a
    /// worker/session SourceId. A receiver without that map must not use this
    /// number as a local runtime source identity.
    #[serde(rename = "source_id")]
    pub program_source_id: Option<ProgramSourceId>,
    pub span: Option<ProgramSourceSpan>,
    /// Presentation only for a frame whose issuing source has retired. It
    /// must never be used to admit code or to resolve a callable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detached_diagnostic: Option<ProgramDetachedDiagnosticSource>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramDetachedDiagnosticSource {
    pub display_name: String,
    /// Both coordinates are present for a located frame; both are absent for
    /// a function-only diagnostic whose producer had no source span.
    pub line: Option<u64>,
    pub column: Option<u64>,
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
        let detached_bytes = self.call_frames.iter().try_fold(0usize, |total, frame| {
            frame
                .detached_diagnostic
                .as_ref()
                .map_or(Some(total), |diagnostic| {
                    total.checked_add(diagnostic.display_name.len())
                })
        });
        if !valid_text(&self.message, MAX_PROGRAM_FAILURE_MESSAGE_BYTES)
            || self.call_frames.len() > MAX_PROGRAM_FAILURE_FRAMES
            || self.call_stack.len() > MAX_PROGRAM_FAILURE_FRAMES
            || detached_bytes.is_none_or(|bytes| bytes > MAX_PROGRAM_FAILURE_MESSAGE_BYTES)
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
                    || frame.program_source_id.is_some() && frame.detached_diagnostic.is_some()
                    || frame
                        .detached_diagnostic
                        .as_ref()
                        .is_some_and(|diagnostic| {
                            !valid_text(&diagnostic.display_name, MAX_PROGRAM_FAILURE_FIELD_BYTES)
                                || diagnostic.line.is_some() != diagnostic.column.is_some()
                                || diagnostic.line == Some(0)
                                || diagnostic.column == Some(0)
                        })
            })
        {
            return Err(ContractError::invalid(
                "program runtime failure",
                "failure exceeds its structural bounds",
            ));
        }
        Ok(())
    }

    pub fn validate_for_program_sources(
        &self,
        source_map: Option<&crate::ExecutableSourceMapV2>,
    ) -> Result<(), ContractError> {
        self.validate()?;
        if self.call_frames.iter().any(|frame| {
            frame
                .program_source_id
                .is_some_and(|source| source_map.is_none_or(|map| !map.contains_source(source)))
        }) {
            return Err(ContractError::invalid(
                "portable runtime failure source",
                "source ordinal is absent from the selected artifact map",
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
                program_source_id: Some(runmat_types::ProgramSourceId(1)),
                span: None,
                detached_diagnostic: None,
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
    fn portable_frame_source_is_a_checked_program_ordinal() {
        let failure = failure();
        let json = serde_json::to_value(&failure).unwrap();
        assert_eq!(json["call_frames"][0]["source_id"], 1);
        assert!(json["call_frames"][0].get("detached_diagnostic").is_none());
        let decoded: ProgramRuntimeFailure = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(decoded, failure);
        let mut oversized = json;
        oversized["call_frames"][0]["source_id"] = serde_json::json!(u64::from(u32::MAX) + 1);
        assert!(serde_json::from_value::<ProgramRuntimeFailure>(oversized).is_err());
    }

    #[test]
    fn portable_frame_rejects_an_unadmitted_map_even_when_length_matches() {
        let map = crate::ExecutableSourceMapV2 {
            entries: vec![crate::ExecutableSourceRecordV2 {
                source_id: runmat_types::ProgramSourceId(1),
                owner_identity: "owner".into(),
                relative_path: "source.m".into(),
                content_revision: crate::Digest::sha256(b"source"),
                display_name: "source.m".into(),
                text: "source".into(),
            }],
        };
        let mut failure = failure();
        failure.call_frames[0].program_source_id = Some(runmat_types::ProgramSourceId(0));
        assert!(failure.validate_for_program_sources(Some(&map)).is_err());
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

    #[test]
    fn detached_frame_is_presentation_only_and_exclusive_with_an_admitted_source() {
        let mut failure = failure();
        failure.call_frames[0].program_source_id = None;
        failure.call_frames[0].detached_diagnostic = Some(super::ProgramDetachedDiagnosticSource {
            display_name: "old/retired.m".into(),
            line: Some(2),
            column: Some(3),
        });
        failure.validate().unwrap();
        let encoded = serde_json::to_value(&failure).unwrap();
        assert_eq!(encoded["call_frames"][0]["detached_diagnostic"]["line"], 2);
        assert_eq!(
            serde_json::from_value::<ProgramRuntimeFailure>(encoded).unwrap(),
            failure
        );

        failure.call_frames[0].program_source_id = Some(runmat_types::ProgramSourceId(1));
        assert!(failure.validate().is_err());
        failure.call_frames[0].program_source_id = None;
        failure.call_frames[0]
            .detached_diagnostic
            .as_mut()
            .unwrap()
            .column = None;
        assert!(failure.validate().is_err());
    }

    #[test]
    fn detached_unknown_frame_is_structured_and_budgeted() {
        let mut failure = failure();
        failure.call_frames[0].program_source_id = None;
        failure.call_frames[0].detached_diagnostic = Some(super::ProgramDetachedDiagnosticSource {
            display_name: "old.m".into(),
            line: None,
            column: None,
        });
        failure.validate().unwrap();
        failure.call_frames[0]
            .detached_diagnostic
            .as_mut()
            .unwrap()
            .display_name = "x".repeat(super::MAX_PROGRAM_FAILURE_MESSAGE_BYTES + 1);
        assert!(failure.validate().is_err());
    }
}
