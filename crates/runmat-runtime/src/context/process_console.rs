//! Incremental application-pipe text publication on the borrowing caller.
use super::RuntimeContext;
use crate::{console::ConsoleStream, RuntimeError};

/// One decoder per byte stream. Only an incomplete UTF-8 suffix is retained;
/// the process owner keeps the original bytes separately from this text view.
#[derive(Default)]
pub struct ProcessConsoleDecoder {
    tail: [u8; 3],
    tail_len: usize,
}

impl ProcessConsoleDecoder {
    pub fn publish(
        &mut self,
        stream: ConsoleStream,
        bytes: &[u8],
        eof: bool,
        runtime: &RuntimeContext,
    ) -> Result<(), RuntimeError> {
        self.decode(bytes, eof, |text| {
            runtime.record_process_console(stream, text)
        })
    }

    pub(crate) fn decode<E>(
        &mut self,
        mut bytes: &[u8],
        eof: bool,
        mut emit: impl FnMut(&str) -> Result<(), E>,
    ) -> Result<(), E> {
        while self.tail_len != 0 && !bytes.is_empty() {
            let mut joined = [0u8; 4];
            let prefix = self.tail_len;
            joined[..prefix].copy_from_slice(&self.tail[..prefix]);
            let count = (joined.len() - prefix).min(bytes.len());
            joined[prefix..prefix + count].copy_from_slice(&bytes[..count]);
            bytes = &bytes[count..];
            self.tail_len = 0;
            self.decode_contiguous(
                &joined[..prefix + count],
                eof && bytes.is_empty(),
                &mut emit,
            )?;
        }
        if !bytes.is_empty() {
            self.decode_contiguous(bytes, eof, &mut emit)?;
        } else if eof && self.tail_len != 0 {
            self.tail_len = 0;
            emit("\u{fffd}")?;
        }
        Ok(())
    }

    fn decode_contiguous<E>(
        &mut self,
        mut bytes: &[u8],
        eof: bool,
        emit: &mut impl FnMut(&str) -> Result<(), E>,
    ) -> Result<(), E> {
        while !bytes.is_empty() {
            match std::str::from_utf8(bytes) {
                Ok(text) => return emit(text),
                Err(error) => {
                    let (valid, rest) = bytes.split_at(error.valid_up_to());
                    if !valid.is_empty() {
                        emit(std::str::from_utf8(valid).expect("validated UTF-8 prefix"))?;
                    }
                    if let Some(invalid) = error.error_len() {
                        emit("\u{fffd}")?;
                        bytes = &rest[invalid..];
                    } else {
                        if eof {
                            emit("\u{fffd}")?;
                        } else {
                            self.tail[..rest.len()].copy_from_slice(rest);
                            self.tail_len = rest.len();
                        }
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_boundaries_preserve_streaming_lossy_utf8_semantics() {
        for bytes in [
            b"before\xe2\x82\xac\xf0\x9f\x98\x80after".as_slice(),
            b"bad\xff\xe2\x82",
            b"\xe2\x82\xf0\x9f\x98\x80\xc2a\xffend",
        ] {
            for chunk_size in 1..=bytes.len() {
                let mut decoder = ProcessConsoleDecoder::default();
                let mut text = String::new();
                for chunk in bytes.chunks(chunk_size) {
                    decoder
                        .decode(chunk, false, |part| {
                            text.push_str(part);
                            Ok::<_, ()>(())
                        })
                        .unwrap();
                }
                decoder
                    .decode(&[], true, |part| {
                        text.push_str(part);
                        Ok::<_, ()>(())
                    })
                    .unwrap();
                assert_eq!(text, String::from_utf8_lossy(bytes), "chunk={chunk_size}");
                assert_eq!(decoder.tail_len, 0);
            }
        }
    }
}
