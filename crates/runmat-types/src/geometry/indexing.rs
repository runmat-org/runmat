/// Allocation-free numeric linear-index geometry. Materialization remains the
/// caller's responsibility, including any execution memory admission.
pub enum NumericLinearShape<'a> {
    Borrowed(&'a [usize]),
    Vector([usize; 2]),
}

impl NumericLinearShape<'_> {
    pub fn as_slice(&self) -> &[usize] {
        match self {
            Self::Borrowed(shape) => shape,
            Self::Vector(shape) => shape,
        }
    }
}

/// A non-scalar vector source supplies orientation only for vector selectors.
/// Matrix selectors and scalar/non-vector sources keep the selector shape.
pub fn numeric_linear_shape<'a>(
    source_shape: &[usize],
    selector_shape: &'a [usize],
    selected_len: usize,
) -> NumericLinearShape<'a> {
    let source_is_non_scalar = source_shape.iter().any(|&extent| extent != 1);
    if is_vector_shape(source_shape) && source_is_non_scalar && is_vector_shape(selector_shape) {
        if source_shape.first().copied().unwrap_or(1) == 1 {
            NumericLinearShape::Vector([1, selected_len])
        } else {
            NumericLinearShape::Vector([selected_len, 1])
        }
    } else {
        NumericLinearShape::Borrowed(selector_shape)
    }
}

pub fn is_vector_shape(shape: &[usize]) -> bool {
    let rows = shape.first().copied().unwrap_or(1);
    let cols = shape.get(1).copied().unwrap_or(1);
    shape.iter().skip(2).all(|&extent| extent == 1) && (rows == 1 || cols == 1)
}

pub fn logical_linear_source_is_row(source_shape: &[usize], mask_shape: &[usize]) -> bool {
    is_vector_shape(source_shape)
        && is_vector_shape(mask_shape)
        && source_shape.first().copied().unwrap_or(1) == 1
}
