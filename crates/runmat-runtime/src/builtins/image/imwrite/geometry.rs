use super::*;

/// Checked logical image geometry, before any pixel buffer allocation.
pub(super) fn samples(rows: usize, cols: usize, channels: usize) -> BuiltinResult<usize> {
    rows.checked_mul(cols)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(|| {
            imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, "image dimensions overflow")
        })
}

pub(super) fn pixels(rows: usize, cols: usize) -> BuiltinResult<usize> {
    samples(rows, cols, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_overflow_is_an_image_error_before_allocation() {
        let error = samples(usize::MAX, 2, 4).expect_err("geometry must overflow");
        assert_eq!(error.identifier(), IMWRITE_ERROR_INVALID_IMAGE.identifier);
        let error = samples(usize::MAX / 2, 1, 3).expect_err("channels must overflow");
        assert_eq!(error.identifier(), IMWRITE_ERROR_INVALID_IMAGE.identifier);
        assert_eq!(samples(2, 3, 4).unwrap(), 24);
    }
}
