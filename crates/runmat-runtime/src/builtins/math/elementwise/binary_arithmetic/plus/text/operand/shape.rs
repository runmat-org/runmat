//! Operand geometry borrows array dimensions or stores a scalar/character shape inline.
pub(super) enum Shape<'a> {
    Borrowed(&'a [usize]),
    Matrix([usize; 2]),
}

impl Shape<'_> {
    pub(super) fn dimensions(&self) -> &[usize] {
        match self {
            Self::Borrowed(dimensions) => dimensions,
            Self::Matrix(dimensions) => dimensions,
        }
    }
}
