//! Pure shape and kernel-point facts shared by the image-filter identities.
//! Public diagnostics remain with each builtin; this module performs no
//! placement or provider work.

#[derive(Clone, Debug)]
pub struct ImfilterKernelPoint {
    pub offsets: Vec<isize>,
    pub value: f64,
}

pub(in crate::builtins::image::filters) fn normalize_shape(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    }
}

pub(in crate::builtins::image::filters) fn extend_shape(
    shape: &[usize],
    rank: usize,
) -> Vec<usize> {
    let mut out = shape.to_vec();
    while out.len() < rank {
        out.push(1);
    }
    out
}

pub(in crate::builtins::image::filters) fn validate_kernel_shape(
    image_shape: &[usize],
    kernel_ext: &[usize],
    builtin: &str,
) -> Result<(), String> {
    for (dim_idx, &ker_dim) in kernel_ext.iter().enumerate() {
        let img_dim = image_shape.get(dim_idx).copied().unwrap_or(1);
        if dim_idx >= image_shape.len() && ker_dim > 1 {
            return Err(format!(
                "{builtin}: filter dimension {} is {}, but the image has no corresponding axis",
                dim_idx + 1,
                ker_dim
            ));
        }
        if img_dim == 0 {
            return Err(format!(
                "{builtin}: image must not have zero-length dimensions"
            ));
        }
    }
    Ok(())
}

pub(in crate::builtins::image::filters) fn filter_output_shape(
    image_shape: &[usize],
    kernel_shape: &[usize],
    shape_mode: runmat_accelerate_api::ImfilterShape,
    builtin: &str,
) -> Result<Vec<usize>, String> {
    use runmat_accelerate_api::ImfilterShape;
    if kernel_shape.contains(&0) {
        return Err(format!(
            "{builtin}: filter must be non-empty along every dimension"
        ));
    }
    let image_norm = normalize_shape(image_shape);
    let kernel_norm = normalize_shape(kernel_shape);
    let rank = image_norm.len().max(kernel_norm.len());
    let image_ext = extend_shape(&image_norm, rank);
    let kernel_ext = extend_shape(&kernel_norm, rank);
    validate_kernel_shape(&image_norm, &kernel_ext, builtin)?;
    let mut output = image_ext
        .iter()
        .zip(&kernel_ext)
        .map(|(&img, &ker)| match shape_mode {
            ImfilterShape::Same => Ok(img),
            ImfilterShape::Full => img
                .checked_add(ker.saturating_sub(1))
                .ok_or_else(|| format!("{builtin}: output shape overflows")),
            ImfilterShape::Valid => Ok(if img >= ker { img - ker + 1 } else { 0 }),
        })
        .collect::<Result<Vec<_>, _>>()?;
    while output.len() > image_norm.len() && output.last() == Some(&1) {
        output.pop();
    }
    if output.is_empty() {
        output.push(1);
    }
    Ok(output)
}
