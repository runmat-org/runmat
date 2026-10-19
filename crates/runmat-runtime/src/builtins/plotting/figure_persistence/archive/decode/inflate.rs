use super::*;
use flate2::{Decompress, FlushDecompress, Status};

pub(super) async fn decompress_payload<C: StagedWriteCooperation + ?Sized>(
    stored: &[u8],
    decoded_len: usize,
    builtin: &'static str,
    cooperation: &mut C,
) -> BuiltinResult<Vec<u8>> {
    let maximum = ReplayLimits::default().max_scene_payload_bytes;
    cooperation.completed(0).await?;
    let mut decoder = Decompress::new(true);
    let mut payload = Vec::with_capacity(decoded_len.min(maximum));
    let mut input_offset = 0usize;
    let mut output = [0u8; 8 * 1024];
    loop {
        let input_end = input_offset.saturating_add(8 * 1024).min(stored.len());
        let input = &stored[input_offset..input_end];
        cooperation.completed(input.len()).await?;
        let prior_input = decoder.total_in();
        let prior_output = decoder.total_out();
        let flush = if input_end == stored.len() {
            FlushDecompress::Finish
        } else {
            FlushDecompress::None
        };
        let status = decoder
            .decompress(input, &mut output, flush)
            .map_err(|source| {
                persistence_error_with_source(
                    &ERROR_IMPORT,
                    builtin,
                    format!("{builtin}: failed to decompress figure payload"),
                    source,
                )
            })?;
        let consumed =
            usize::try_from(decoder.total_in() - prior_input).map_err(|_| corrupt(builtin))?;
        let produced =
            usize::try_from(decoder.total_out() - prior_output).map_err(|_| corrupt(builtin))?;
        input_offset = input_offset
            .checked_add(consumed)
            .ok_or_else(|| corrupt(builtin))?;
        if payload
            .len()
            .checked_add(produced)
            .is_none_or(|total| total > decoded_len || total > maximum)
        {
            return Err(corrupt(builtin));
        }
        payload.extend_from_slice(&output[..produced]);
        cooperation.completed(produced).await?;

        match status {
            Status::StreamEnd => {
                if input_offset != stored.len() {
                    return Err(corrupt(builtin));
                }
                break;
            }
            Status::Ok | Status::BufError if consumed > 0 || produced > 0 => {}
            Status::Ok | Status::BufError => return Err(corrupt(builtin)),
        }
    }
    if payload.len() != decoded_len {
        return Err(corrupt(builtin));
    }
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::{write::ZlibEncoder, Compression};
    use std::cell::Cell;
    use std::io::Write;
    use std::rc::Rc;

    struct CancelAfterProgress {
        calls: Rc<Cell<usize>>,
    }

    impl StagedWriteCooperation for CancelAfterProgress {
        async fn completed(&mut self, _work: usize) -> BuiltinResult<()> {
            let call = self.calls.get() + 1;
            self.calls.set(call);
            if call == 3 {
                return Err(
                    crate::build_runtime_error("openfig: execution was cancelled")
                        .with_builtin(OPENFIG)
                        .with_identifier("RunMat:Cancelled")
                        .build(),
                );
            }
            Ok(())
        }
    }

    #[test]
    fn incremental_decompression_can_cancel_after_bounded_progress() {
        let payload = (0usize..32 * 1024)
            .map(|index| (index.wrapping_mul(31) % 251) as u8)
            .collect::<Vec<_>>();
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&payload).expect("compress fixture");
        let stored = encoder.finish().expect("finish fixture");
        let calls = Rc::new(Cell::new(0));
        let mut cooperation = CancelAfterProgress {
            calls: calls.clone(),
        };

        let error = futures::executor::block_on(decompress_payload(
            &stored,
            payload.len(),
            OPENFIG,
            &mut cooperation,
        ))
        .expect_err("output checkpoint cancels after one decompression call");

        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(calls.get(), 3);
    }
}
