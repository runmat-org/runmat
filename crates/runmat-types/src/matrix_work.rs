//! Checked base matrix work geometry. No class or provider admission is implied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixWorkGeometry {
    /// Historical matrix calibration unit: one multiply-accumulate term.
    pub multiply_accumulates: u64,
    /// Output publication remains work even when the inner dimension is zero.
    pub output_elements: u64,
}
impl MatrixWorkGeometry {
    pub fn checked(rows: u64, inner: u64, columns: u64) -> Option<Self> {
        let output_elements = rows.checked_mul(columns)?;
        Some(Self {
            multiply_accumulates: output_elements.checked_mul(inner)?,
            output_elements,
        })
    }
}
