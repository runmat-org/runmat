//! Encoded sources share the normal size/repeat/skip and final-output owners.
//! No spelling or encoding-name dispatch occurs at this boundary.

use super::*;
use crate::builtins::io::filetext::encoding::EncodedInputStep;
use runmat_value::Utf16CodeUnit;

#[cfg(test)]
mod tests;
use super::{output, presentation, workspace::StreamValues};
type StreamUnits<'a> = StreamValues<'a, Utf16CodeUnit>;

pub(super) async fn read_encoded_source(
    file: &mut registry::FileOperation,
    size: &SizeSpec,
    precision: &PrecisionSpec,
    skip: usize,
    cooperation: &mut super::super::super::cooperation::FileCooperation,
    runtime: Option<&crate::context::RuntimeContext>,
    projection: presentation::Projection<'_>,
) -> BuiltinResult<(FreadEval, bool)> {
    let mut values = StreamUnits::new(runtime)?;
    let target = size.element_limit().unwrap_or(usize::MAX);
    if target != 0 {
        file.prepare_encoded_input(runtime)?;
    }
    let mut count = 0usize;
    let mut eof = false;
    while count < target {
        let step = match file.encoded_input_step() {
            Ok(step) => step,
            Err(source) if source.kind() == ErrorKind::Interrupted => {
                cooperation.completed(1).await?;
                continue;
            }
            Err(source) => {
                return Err(ReadFailure::Io {
                    operation: "failed to decode character source",
                    source,
                }
                .into())
            }
        };
        let (unit, bytes) = match step {
            EncodedInputStep::Unit { unit, bytes } => (Some(unit), bytes),
            EncodedInputStep::Progress { bytes } => (None, bytes),
            EncodedInputStep::End => {
                eof = true;
                break;
            }
        };
        cooperation.completed(bytes).await?;
        let Some(unit) = unit else {
            continue;
        };
        values.push(unit)?;
        count += 1;
        if count.is_multiple_of(precision.repeat) && skip != 0 {
            file.skip_encoded_bytes(i64::try_from(skip).map_err(|_| {
                ReadFailure::Semantic("fread: skip exceeds file cursor range".to_string())
            })?)
            .map_err(|source| ReadFailure::Io {
                operation: "failed to skip encoded source bytes",
                source,
            })?;
            cooperation.completed(skip).await?;
        }
    }
    let value = output::finish(&values, size, projection, cooperation).await?;
    Ok((FreadEval::new(value, count), eof))
}
