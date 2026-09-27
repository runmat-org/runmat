//! Pure request geometry shared by public dispatch and backend direct callers.
//! Allocation membership and supported physical representations remain backend
//! responsibilities; validate them before allocating or submitting any work.
use crate::{GpuTensorStorage, ProviderSpectralFrameMode, ProviderSpectralRequest};

/// Validate the complete spectral geometry without provider lookup or effects.
pub fn validate_uniform_spectral_request(
    request: &ProviderSpectralRequest<'_>,
) -> anyhow::Result<()> {
    let invalid = || anyhow::anyhow!("uniform_spectral_estimate: invalid request");
    if request.window.is_empty()
        || request.nfft == 0
        || request.frame_count == 0
        || !request.denominator.is_finite()
        || request.denominator <= 0.0
    {
        return Err(invalid());
    }
    let input_len = if request.input.shape().contains(&0) {
        Some(0)
    } else {
        request
            .input
            .shape()
            .iter()
            .try_fold(1usize, |n, &dim| n.checked_mul(dim))
    };
    if input_len != Some(request.input_len)
        || request.input_complex
            != (request.input.physical().storage() == GpuTensorStorage::ComplexInterleaved)
    {
        return Err(invalid());
    }
    // Every frame owns a full complex transform. Check this even for one-sided
    // output, before a provider can allocate the window or a partial result.
    request
        .nfft
        .checked_mul(request.frame_count)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(invalid)?;
    let required = match request.frame_mode {
        ProviderSpectralFrameMode::Sliding { hop } => {
            if hop == 0 {
                return Err(invalid());
            }
            (request.frame_count - 1)
                .checked_mul(hop)
                .and_then(|offset| offset.checked_add(request.window.len()))
        }
        ProviderSpectralFrameMode::ColumnSliding {
            hop,
            input_rows,
            frames_per_column,
        } => {
            if hop == 0 || input_rows == 0 || frames_per_column == 0 {
                return Err(invalid());
            }
            // A frame must fit its own column, not borrow samples from the next.
            let used_frames = request.frame_count.min(frames_per_column);
            let column_end = (used_frames - 1)
                .checked_mul(hop)
                .and_then(|offset| offset.checked_add(request.window.len()));
            if column_end.is_none_or(|end| end > input_rows) {
                return Err(invalid());
            }
            let last_frame = request.frame_count - 1;
            (last_frame / frames_per_column)
                .checked_mul(input_rows)
                .and_then(|base| {
                    (last_frame % frames_per_column)
                        .checked_mul(hop)
                        .and_then(|offset| base.checked_add(offset))
                })
                .and_then(|offset| offset.checked_add(request.window.len()))
        }
        ProviderSpectralFrameMode::FoldedColumns { input_rows } => {
            // Folding changes the DFT length, not the source column boundary.
            if input_rows == 0 || request.window.len() > input_rows {
                return Err(invalid());
            }
            input_rows.checked_mul(request.frame_count)
        }
    };
    if required.is_none_or(|required| required > request.input_len) {
        return Err(invalid());
    }
    Ok(())
}

#[cfg(test)]
mod tests;
