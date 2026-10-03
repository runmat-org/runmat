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

fn checked_allocation_bytes<T>(elements: usize) -> BuiltinResult<usize> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| {
            imwrite_error_with_detail(
                &IMWRITE_ERROR_INVALID_IMAGE,
                "image allocation size overflows",
            )
        })?;
    if bytes > isize::MAX as usize {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            "image allocation exceeds addressable storage",
        ));
    }
    Ok(bytes)
}

/// Reserve the exact element capacity before image-owned writes or conversion.
pub(super) fn reserve<T>(elements: usize) -> BuiltinResult<Vec<T>> {
    checked_allocation_bytes::<T>(elements)?;
    let mut values = Vec::new();
    values.try_reserve_exact(elements).map_err(|error| {
        imwrite_error_with_source(
            &IMWRITE_ERROR_INVALID_IMAGE,
            "image allocation unavailable",
            error,
        )
    })?;
    Ok(values)
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

    #[test]
    fn native_byte_capacity_rejects_before_reservation() {
        let too_many_u16 = (isize::MAX as usize / std::mem::size_of::<u16>()) + 1;
        let error = reserve::<u16>(too_many_u16).expect_err("unaddressable allocation");
        assert_eq!(error.identifier(), IMWRITE_ERROR_INVALID_IMAGE.identifier);
        let error = reserve::<u16>(usize::MAX).expect_err("byte multiplication overflow");
        assert_eq!(error.identifier(), IMWRITE_ERROR_INVALID_IMAGE.identifier);
        assert!(reserve::<u16>(3).unwrap().capacity() >= 3);
    }
}
