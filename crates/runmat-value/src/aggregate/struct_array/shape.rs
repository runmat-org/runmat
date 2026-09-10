pub(crate) fn total_len(shape: &[usize]) -> Result<usize, String> {
    if shape.len() < 2 {
        return Err("structure array shape must contain at least two dimensions".into());
    }
    shape.iter().try_fold(1usize, |length, extent| {
        length
            .checked_mul(*extent)
            .ok_or_else(|| "structure array shape exceeds platform limits".to_string())
    })
}

pub(crate) fn column_major_strides(shape: &[usize]) -> Result<Vec<usize>, String> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for extent in shape {
        strides.push(stride);
        stride = stride
            .checked_mul(*extent)
            .ok_or_else(|| "structure array shape exceeds platform limits".to_string())?;
    }
    Ok(strides)
}
