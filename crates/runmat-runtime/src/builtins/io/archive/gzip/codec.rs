//! Bounded gzip codec steps, independent of filesystem and HTTP ownership.

use compression_codecs::core::util::{PartialBuffer, WriteBuffer};
use compression_codecs::{DecodeV2, EncodeV2, GzipDecoder, GzipEncoder};
use std::io;

use crate::execution::{CpuCooperationBudget, ExecutionServiceError};

// A codec buffer bound, not a scheduling policy. The selected execution service
// decides when these completed input/output bytes require a cooperative turn.
const BUFFER_BYTES: usize = 8192;

#[derive(Clone, Copy)]
pub(super) enum Operation {
    Compress,
    Decompress,
}

impl Operation {
    pub(super) fn builtin_name(&self) -> &'static str {
        match self {
            Self::Compress => "gzip",
            Self::Decompress => "gunzip",
        }
    }
}

enum Codec {
    Encoder(GzipEncoder),
    Decoder(GzipDecoder),
}

impl Codec {
    fn new(operation: &Operation) -> Self {
        match operation {
            Operation::Compress => {
                Self::Encoder(GzipEncoder::new(flate2::Compression::default().into()))
            }
            Operation::Decompress => Self::Decoder(GzipDecoder::new()),
        }
    }

    fn step(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
        eof: bool,
    ) -> io::Result<bool> {
        match self {
            Self::Encoder(encoder) => {
                if eof {
                    encoder.finish(output)
                } else {
                    encoder.encode(input, output)?;
                    Ok(false)
                }
            }
            Self::Decoder(decoder) => {
                if eof {
                    decoder.finish(output)
                } else {
                    decoder.decode(input, output)
                }
            }
        }
    }
}

pub(super) async fn transform(
    bytes: &[u8],
    operation: Operation,
    mut cooperation: Option<&mut CpuCooperationBudget>,
) -> crate::BuiltinResult<Vec<u8>> {
    checkpoint(&mut cooperation, 0).await?;
    let mut codec = Codec::new(&operation);
    let mut scratch = vec![0; BUFFER_BYTES];
    let mut result = Vec::new();
    let mut offset: usize = 0;
    loop {
        let end = offset.saturating_add(BUFFER_BYTES).min(bytes.len());
        let mut input = PartialBuffer::new(&bytes[offset..end]);
        let mut output = WriteBuffer::new_initialized(&mut scratch);
        let complete = codec
            .step(&mut input, &mut output, offset == bytes.len())
            .map_err(|error| {
                let name = operation.builtin_name();
                super::archive_error_with_source(name, format!("{name}: {error}"), error)
            })?;
        let consumed = input.written().len();
        let produced = output.written_len();
        offset += consumed;
        result.extend_from_slice(output.written());
        checkpoint(&mut cooperation, (consumed + produced).max(1) as u64).await?;
        if complete {
            // The previous single-member GzDecoder ignored trailing members;
            // keep that contract rather than silently switching to MultiGzip.
            return Ok(result);
        }
        if consumed == 0 && produced == 0 {
            let name = operation.builtin_name();
            let error = io::Error::new(io::ErrorKind::InvalidData, "gzip codec made no progress");
            return Err(super::archive_error_with_source(
                name,
                format!("{name}: {error}"),
                error,
            ));
        }
    }
}

pub(super) async fn checkpoint(
    cooperation: &mut Option<&mut CpuCooperationBudget>,
    completed_work: u64,
) -> crate::BuiltinResult<()> {
    if let Some(budget) = cooperation {
        budget
            .checkpoint(completed_work)
            .await
            .map_err(ExecutionServiceError::into_runtime_error)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "codec/tests.rs"]
mod tests;
