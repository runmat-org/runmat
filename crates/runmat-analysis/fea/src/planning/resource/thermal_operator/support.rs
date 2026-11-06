//! Scalar thermal support from coupling cliques: a node→clique incidence and a marker array.
//!
//! Every volume element, boundary face and interface conductance face couples all of its dofs.
//! The support row of a dof is the union of the cliques that contain it. The builder counts each
//! row's distinct columns with a marker array, allocates the column array once at its exact
//! length, and fills it in a second pass. Its scratch is linear in the number of dofs and clique
//! incidences; it never stores a repeated coupling.
use super::ResourcePlanningError;
use crate::algebra::SparsePattern;

/// Counts that size [`build`] before it allocates anything.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SupportCounts {
    /// Rows (dofs) of the support.
    pub rows: u64,
    /// Sum of the clique widths: each (clique, dof) incidence once.
    pub incidences: u64,
    /// Extra interface cliques held while the support is built, and their summed widths.
    pub extra_cliques: u64,
    pub extra_incidences: u64,
}

/// Requested heap of [`build`] for `counts` and a support of `entries` stored entries: the
/// incidence offsets and clique ids, the marker, the pattern's row offsets and its columns, and
/// the extra cliques (vector headers and dofs). The builder's scratch and the finished pattern
/// coexist only until the builder returns, so this is also the finished pattern's peak.
pub(super) fn construction_bytes(
    counts: SupportCounts,
    entries: u64,
) -> Result<u64, ResourcePlanningError> {
    let word = std::mem::size_of::<usize>() as u64;
    let offsets = mul(add(counts.rows, 1)?, word)?;
    [
        // Incidence offsets and the clique id of each incidence.
        offsets,
        mul(counts.incidences, word)?,
        // Marker.
        mul(counts.rows, word)?,
        // Pattern row offsets and columns (u64).
        mul(add(counts.rows, 1)?, 8)?,
        mul(entries, 8)?,
        // Extra cliques: one vector header and the dofs of each.
        mul(
            counts.extra_cliques,
            std::mem::size_of::<Vec<usize>>() as u64,
        )?,
        mul(counts.extra_incidences, word)?,
    ]
    .into_iter()
    .try_fold(0, add)
}

/// The scalar support of `rows` dofs coupled by `cliques` plus a diagonal on every row.
/// `clique(c)` returns the dofs of clique `c < cliques`; every dof must be below `rows`.
pub(super) fn build<'a>(
    rows: usize,
    cliques: usize,
    clique: impl Fn(usize) -> &'a [usize],
) -> Result<SparsePattern, ResourcePlanningError> {
    // Node→clique incidence in compressed form.
    let mut starts = vec![0_usize; rows + 1];
    for c in 0..cliques {
        for &dof in clique(c) {
            if dof >= rows {
                return Err(ResourcePlanningError::Invalid(
                    "thermal coupling refers to a dof outside the support".into(),
                ));
            }
            starts[dof + 1] += 1;
        }
    }
    for row in 0..rows {
        starts[row + 1] += starts[row];
    }
    let mut incident = vec![0_usize; starts[rows]];
    {
        let mut next = starts.clone();
        for c in 0..cliques {
            for &dof in clique(c) {
                incident[next[dof]] = c;
                next[dof] += 1;
            }
        }
    }
    // The cursor copy above is freed here; count the distinct columns of each row.
    let mut marker = vec![usize::MAX; rows];
    let mut offsets = Vec::with_capacity(rows + 1);
    offsets.push(0_u64);
    let mut total = 0_u64;
    for row in 0..rows {
        marker[row] = row;
        let mut width = 1_u64;
        for &c in &incident[starts[row]..starts[row + 1]] {
            for &column in clique(c) {
                if marker[column] != row {
                    marker[column] = row;
                    width += 1;
                }
            }
        }
        total = add(total, width)?;
        offsets.push(total);
    }
    let length = usize::try_from(total)
        .map_err(|_| ResourcePlanningError::Overflow("thermal sparse entries"))?;
    let mut columns = Vec::with_capacity(length);
    marker.fill(usize::MAX);
    for row in 0..rows {
        let first = columns.len();
        marker[row] = row;
        columns.push(row as u64);
        for &c in &incident[starts[row]..starts[row + 1]] {
            for &column in clique(c) {
                if marker[column] != row {
                    marker[column] = row;
                    columns.push(column as u64);
                }
            }
        }
        columns[first..].sort_unstable();
    }
    drop((starts, incident, marker));
    SparsePattern::try_new(rows as u64, rows as u64, offsets, columns)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))
}

fn add(left: u64, right: u64) -> Result<u64, ResourcePlanningError> {
    left.checked_add(right)
        .ok_or(ResourcePlanningError::Overflow("thermal support construction"))
}
fn mul(left: u64, right: u64) -> Result<u64, ResourcePlanningError> {
    left.checked_mul(right)
        .ok_or(ResourcePlanningError::Overflow("thermal support construction"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cliques_give_the_union_of_their_couplings_once() {
        let cliques: Vec<Vec<usize>> = vec![vec![0, 1, 2], vec![1, 2, 3], vec![2, 1], vec![4]];
        let pattern = build(6, cliques.len(), |c| cliques[c].as_slice()).unwrap();
        assert_eq!(pattern.row_offsets(), &[0, 3, 7, 11, 14, 15, 16]);
        assert_eq!(
            pattern.column_indices(),
            &[0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 4, 5]
        );
        assert!(build(2, 1, |_| &[0, 2][..]).is_err());
    }
}
