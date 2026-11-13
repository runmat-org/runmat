//! Checked allocation extents for the existing dense least-squares solver.
use super::{Complex64, PolyfitKernelError, PolyfitKernelResult, ProviderPolyfitOutputs};

pub(super) struct SolverGeometry {
    pub rows: usize,
    pub cols: usize,
    pub rectangular_len: usize,
    pub square_len: usize,
}

impl SolverGeometry {
    pub fn for_outputs(
        rows: usize,
        degree: usize,
        outputs: ProviderPolyfitOutputs,
    ) -> PolyfitKernelResult<Self> {
        let overflow = || {
            PolyfitKernelError::argument(
                "polyfit: solver geometry exceeds platform allocation limits",
            )
        };
        let cols = degree.checked_add(1).ok_or_else(overflow)?;
        let rectangular_len = rows.checked_mul(cols).ok_or_else(overflow)?;
        let square_len = if outputs.wants_statistics() {
            cols.checked_mul(cols).ok_or_else(overflow)?
        } else {
            0
        };
        // Layout checks byte addressability (including isize::MAX), not an
        // arbitrary degree cap or a promise that the allocator can satisfy it.
        for len in [rows, cols, rectangular_len, square_len] {
            std::alloc::Layout::array::<f64>(len).map_err(|_| overflow())?;
        }
        for len in [rows, cols] {
            std::alloc::Layout::array::<Complex64>(len).map_err(|_| overflow())?;
        }
        Ok(Self {
            rows,
            cols,
            rectangular_len,
            square_len,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn statistics_geometry(rows: usize, degree: usize) -> PolyfitKernelResult<SolverGeometry> {
        SolverGeometry::for_outputs(rows, degree, ProviderPolyfitOutputs::Statistics)
    }

    #[test]
    fn rejects_arithmetic_and_byte_extents_without_allocating() {
        assert!(statistics_geometry(1, usize::MAX).is_err());
        assert!(statistics_geometry(usize::MAX, 1).is_err());
        let square_overflow_cols = 1usize << (usize::BITS / 2);
        assert!(statistics_geometry(1, square_overflow_cols - 1).is_err());
        let real_byte_overflow = isize::MAX as usize / std::mem::size_of::<f64>() + 1;
        assert!(statistics_geometry(real_byte_overflow, 0).is_err());
        let complex_byte_overflow = isize::MAX as usize / std::mem::size_of::<Complex64>() + 1;
        assert!(std::alloc::Layout::array::<f64>(complex_byte_overflow).is_ok());
        assert!(statistics_geometry(complex_byte_overflow, 0).is_err());
        let valid = statistics_geometry(2, 3).unwrap();
        assert_eq!(
            (
                valid.rows,
                valid.cols,
                valid.rectangular_len,
                valid.square_len
            ),
            (2, 4, 8, 16)
        );
    }

    #[test]
    fn coefficient_only_geometry_does_not_admit_unrequested_square_workspace() {
        let cols = 1usize << (usize::BITS / 2);
        let degree = cols - 1;
        assert!(statistics_geometry(1, degree).is_err());
        let coefficient_only =
            SolverGeometry::for_outputs(1, degree, ProviderPolyfitOutputs::Coefficients)
                .expect("rectangular geometry fits when R is not requested");
        assert_eq!(coefficient_only.rectangular_len, cols);
        assert_eq!(coefficient_only.square_len, 0);
    }
}
