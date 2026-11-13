//! Progress comparison for a previously checked, unchanged physical cavity.
use super::*;

pub(super) fn lifting_change(
    topology: &impl crate::cdt::insertion::topology_access::TetrahedralTopology,
    removed: &[usize],
    created: &[[u32; 4]],
    work: &mut InsertionWork<'_>,
) -> Result<PredicateSign, DelaunayInsertionError> {
    let mut old = Vec::with_capacity(removed.len());
    for index in removed {
        work.predicate()?;
        old.push(
            topology
                .tetrahedron(*index as u32)
                .expect("live lifting cavity")
                .vertex_indices
                .map(|vertex| {
                    topology.nodes()[vertex as usize]
                        .coordinates_m
                        .canonical_value()
                }),
        );
    }
    let mut new = Vec::with_capacity(created.len());
    for vertices in created {
        work.predicate()?;
        new.push(vertices.map(|vertex| {
            topology.nodes()[vertex as usize]
                .coordinates_m
                .canonical_value()
        }));
    }
    runmat_meshing_core::quality::predicate::tetrahedral_lifting_change(&old, &new)
        .map_err(super::predicate_error)
}
