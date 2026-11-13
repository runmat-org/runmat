use crate::BuiltinCliTranscriptStep;

use super::{push, BuiltinCatalogValidationError};

const MAX_TRANSCRIPT_STEPS: usize = 128;
const MAX_TRANSCRIPT_BYTES: usize = 1024 * 1024;

pub(super) fn validate_cli_transcript(
    builtin: &'static str,
    transcript: &[BuiltinCliTranscriptStep],
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    if transcript.is_empty() || transcript.len() > MAX_TRANSCRIPT_STEPS {
        push(
            errors,
            builtin,
            "CLI transcript must contain between 1 and 128 steps",
        );
    }
    let mut aggregate_bytes = 0usize;
    for step in transcript {
        match step {
            BuiltinCliTranscriptStep::ExpectOutput(value)
            | BuiltinCliTranscriptStep::SendLine(value)
                if value.is_empty() =>
            {
                push(errors, builtin, "CLI transcript text must not be empty");
            }
            BuiltinCliTranscriptStep::SendBytes([]) => {
                push(errors, builtin, "CLI transcript bytes must not be empty");
            }
            _ => {}
        }
        aggregate_bytes = aggregate_bytes.saturating_add(match step {
            BuiltinCliTranscriptStep::ExpectOutput(value)
            | BuiltinCliTranscriptStep::SendLine(value) => value.len(),
            BuiltinCliTranscriptStep::SendBytes(value) => value.len(),
            BuiltinCliTranscriptStep::SendEndOfInput | BuiltinCliTranscriptStep::SendInterrupt => 0,
        });
    }
    if aggregate_bytes > MAX_TRANSCRIPT_BYTES {
        push(
            errors,
            builtin,
            "CLI transcript aggregate payload exceeds the one-megabyte bound",
        );
    }
    let terminal = transcript.iter().position(|step| {
        matches!(
            step,
            BuiltinCliTranscriptStep::SendEndOfInput | BuiltinCliTranscriptStep::SendInterrupt
        )
    });
    if terminal.is_none()
        || transcript
            .iter()
            .skip(terminal.unwrap_or(0) + 1)
            .any(|step| !matches!(step, BuiltinCliTranscriptStep::ExpectOutput(_)))
        || transcript
            .iter()
            .filter(|step| {
                matches!(
                    step,
                    BuiltinCliTranscriptStep::SendEndOfInput
                        | BuiltinCliTranscriptStep::SendInterrupt
                )
            })
            .count()
            != 1
    {
        push(
            errors,
            builtin,
            "CLI transcript requires one terminal action followed only by output expectations",
        );
    }
}
