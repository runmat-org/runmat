//! Column-major implicit expansion of known concrete dimensions.
use std::fmt;
mod dimensions;
pub use dimensions::ConcreteBroadcastDimensions;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConcreteBroadcastError {
    IncompatibleAxis {
        axis: usize,
        left: usize,
        right: usize,
    },
    ShapeAllocation,
}

impl fmt::Display for ConcreteBroadcastError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompatibleAxis { axis, left, right } => write!(
                formatter,
                "implicit expansion mismatch at dimension {axis}: {left} versus {right}"
            ),
            Self::ShapeAllocation => {
                formatter.write_str("cannot allocate broadcast shape metadata")
            }
        }
    }
}

impl std::error::Error for ConcreteBroadcastError {}

/// Compute the concrete broadcast shape. A one-dimensional shape denotes a
/// row vector when the resulting rank is at least two, matching RunMat's
/// column-major array convention.
pub fn broadcast_concrete_shape(
    left: &[usize],
    right: &[usize],
) -> Result<Vec<usize>, ConcreteBroadcastError> {
    let dimensions = ConcreteBroadcastDimensions::new(left, right)?;
    let rank = dimensions.rank();
    let mut output = Vec::new();
    output
        .try_reserve_exact(rank)
        .map_err(|_| ConcreteBroadcastError::ShapeAllocation)?;
    for axis in 0..rank {
        output.push(
            dimensions
                .dimension(axis)
                .expect("validated broadcast rank"),
        );
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concrete_expansion_preserves_row_shorthand_empty_and_order() {
        assert_eq!(broadcast_concrete_shape(&[2, 1], &[1, 3]).unwrap(), [2, 3]);
        assert_eq!(broadcast_concrete_shape(&[3], &[2, 3]).unwrap(), [2, 3]);
        assert_eq!(broadcast_concrete_shape(&[0, 2], &[1, 2]).unwrap(), [0, 2]);
        assert_eq!(broadcast_concrete_shape(&[], &[2, 3]).unwrap(), [2, 3]);
        assert!(broadcast_concrete_shape(&[], &[]).unwrap().is_empty());
        assert_eq!(
            broadcast_concrete_shape(&[2, 2], &[3, 2]).unwrap_err(),
            ConcreteBroadcastError::IncompatibleAxis {
                axis: 1,
                left: 2,
                right: 3,
            }
        );
    }
}
