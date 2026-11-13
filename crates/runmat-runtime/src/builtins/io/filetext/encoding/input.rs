//! File-owned incremental decoder. A step consumes at most one physical byte;
//! the caller cooperates between steps, including state-only escape sequences.

use super::TextEncoding;
use encoding_rs::{Decoder, DecoderResult};
use runmat_value::Utf16CodeUnit;
use std::io;

mod allocation;
pub(crate) use allocation::TextInputBox;

pub(crate) enum EncodedInputStep {
    Unit { unit: Utf16CodeUnit, bytes: usize },
    Progress { bytes: usize },
    End,
}

pub(crate) struct EncodedInput {
    encoding: TextEncoding,
    decoder: Option<Decoder>,
    pending_byte: Option<u8>,
    pending_unit: Option<Utf16CodeUnit>,
    finished: bool,
    failed: bool,
}

impl EncodedInput {
    pub(crate) fn reset(&mut self, at_start: bool) {
        *self = Self::new(self.encoding, at_start);
    }

    pub(crate) fn allocate(
        runtime: Option<&crate::context::RuntimeContext>,
        encoding: TextEncoding,
        at_start: bool,
    ) -> Result<TextInputBox, crate::context::host_allocation::HostAllocationError> {
        allocation::allocate(runtime, Self::new(encoding, at_start))
    }
    pub(crate) fn new(encoding: TextEncoding, at_start: bool) -> Self {
        Self {
            encoding,
            decoder: match encoding {
                TextEncoding::Encoded(encoding) => Some(if at_start {
                    encoding.new_decoder()
                } else {
                    encoding.new_decoder_without_bom_handling()
                }),
                _ => None,
            },
            pending_byte: None,
            pending_unit: None,
            finished: false,
            failed: false,
        }
    }

    pub(crate) fn step(
        &mut self,
        read: impl FnOnce(&mut [u8]) -> io::Result<usize>,
    ) -> io::Result<EncodedInputStep> {
        if self.failed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "encoded character input is invalid",
            ));
        }
        if let Some(unit) = self.pending_unit.take() {
            return Ok(EncodedInputStep::Unit { unit, bytes: 0 });
        }
        if self.finished {
            return Ok(EncodedInputStep::End);
        }
        let mut byte = [0];
        let (length, bytes) = if let Some(pending) = self.pending_byte.take() {
            byte[0] = pending;
            (1, 0)
        } else {
            let length = read(&mut byte)?;
            (length, length)
        };
        match self.encoding {
            TextEncoding::Bytes | TextEncoding::Ascii => {
                if length == 0 {
                    self.finished = true;
                    return Ok(EncodedInputStep::End);
                }
                if matches!(self.encoding, TextEncoding::Ascii) && byte[0] > 127 {
                    self.failed = true;
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "byte is outside the ASCII range",
                    ));
                }
                Ok(EncodedInputStep::Unit {
                    unit: Utf16CodeUnit::from_raw(u16::from(byte[0])),
                    bytes,
                })
            }
            TextEncoding::Encoded(_) => {
                let mut output = [0u16; 2];
                let (result, consumed, written) = self
                    .decoder
                    .as_mut()
                    .expect("encoded source has decoder")
                    .decode_to_utf16_without_replacement(&byte[..length], &mut output, length == 0);
                if consumed < length {
                    self.pending_byte = Some(byte[0]);
                }
                if matches!(result, DecoderResult::Malformed(_, _)) {
                    self.failed = true;
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "malformed encoded character input",
                    ));
                }
                if length != 0 && consumed == 0 && written == 0 {
                    self.failed = true;
                    return Err(io::Error::other(
                        "character decoder made no progress with a two-unit output window",
                    ));
                }
                if length == 0 && matches!(result, DecoderResult::InputEmpty) {
                    self.finished = true;
                }
                if written == 2 {
                    self.pending_unit = Some(Utf16CodeUnit::from_raw(output[1]));
                }
                if written != 0 {
                    Ok(EncodedInputStep::Unit {
                        unit: Utf16CodeUnit::from_raw(output[0]),
                        bytes,
                    })
                } else if self.finished {
                    Ok(EncodedInputStep::End)
                } else {
                    Ok(EncodedInputStep::Progress { bytes })
                }
            }
        }
    }
}
