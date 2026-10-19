//! Face-side invariant for already positively oriented materialized tetrahedra.
use super::storage::{error, TopologyMutationError};

/// Positive tetrahedra sharing a face must induce opposite face orientations.
/// Positivity was established with exact materialized predicates before this
/// check, so permutation parity gives the side test without another predicate.
pub(super) fn require_opposite_sides(
    left: [u32; 4],
    left_opposite: usize,
    right: [u32; 4],
    right_opposite: usize,
) -> Result<(), TopologyMutationError> {
    if face_parity(left, left_opposite) == face_parity(right, right_opposite) {
        return Err(error(format!(
            "cavity replacement overlaps across a shared face: tetrahedra {left:?} and {right:?}",
        )));
    }
    Ok(())
}

fn face_parity(vertices: [u32; 4], opposite: usize) -> bool {
    let mut inversions = opposite;
    for left in 0..4 {
        for right in (left + 1)..4 {
            if left != opposite && right != opposite && vertices[left] > vertices[right] {
                inversions += 1;
            }
        }
    }
    inversions % 2 == 1
}
