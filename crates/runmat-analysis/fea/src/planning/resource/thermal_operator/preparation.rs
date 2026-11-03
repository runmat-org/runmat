//! Count-only admission of thermal support construction before geometry/index materialization.
use super::{AnalysisProjection, ByteCount, ResourcePlanningError, ThermalOperatorResourceFacts};
use runmat_meshing_core::SolverMeshArtifact;
use std::{collections::BTreeSet, mem::size_of};

impl ThermalOperatorResourceFacts {
    /// Additional requested-heap estimate for support derivation, including projection and mesh
    /// validation. Excludes borrowed input, allocator/native/RSS overhead. Exact factorization
    /// work is reserved separately after support; no factorization is run by this estimator.
    pub fn derivation_resident_bytes_estimate(
        mesh: &SolverMeshArtifact,
    ) -> Result<ByteCount, ResourcePlanningError> {
        let projection = AnalysisProjection::derivation_resident_bytes_estimate(mesh)
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        // A finite-conductance interface adds a second-side dof at each conformal interface
        // node, couples the two sides' face dofs, and works on a split copy of the projection.
        let interface_widths = mesh
            .topology
            .boundary_faces
            .iter()
            .filter(|f| f.role == runmat_meshing_core::BoundaryFaceRole::ConformalInterface)
            .map(|f| f.node_ids.len());
        let interface_nodes = interface_widths
            .clone()
            .try_fold(0, |total, width| add(total, count(width)?))?;
        let interface_entries = interface_widths.clone().try_fold(0, |total, width| {
            let width = count(width)?;
            add(total, mul(mul(2, width)?, mul(2, width)?)?)
        })?;
        let projection = if interface_nodes > 0 {
            mul(projection, 2)?
        } else {
            projection
        };
        let nodes = add(count(mesh.topology.nodes.len())?, interface_nodes)?;
        // Every row has a diagonal. Shared couplings may be counted repeatedly; do not allocate
        // a deduplication graph just to estimate the cost of constructing that graph.
        let entries = mesh
            .topology
            .volume_elements
            .iter()
            .map(|e| e.node_ids.len())
            .chain(
                mesh.topology
                    .boundary_faces
                    .iter()
                    .map(|f| f.node_ids.len()),
            )
            .try_fold(add(nodes, interface_entries)?, |total, width| {
                let width = count(width)?;
                add(total, mul(width, width)?)
            })?;
        let rows = mul(nodes, size_of::<BTreeSet<usize>>() as u64)?;
        let row_trees = tree::<usize>(entries)?;
        let offsets = mul(add(nodes, 1)?, size_of::<u64>() as u64)?;
        let columns = mul(entries.max(4), 2 * size_of::<u64>() as u64)?;
        // The shared CoercivityWorkEnvelope counts undirected edges in place; its symbolic
        // analysis (ordering, elimination tree, column counts) holds a few index words per
        // stored entry of the support.
        let coercivity_graph = mul(entries, 4 * size_of::<u64>() as u64)?;
        [
            projection,
            rows,
            row_trees,
            offsets,
            columns,
            coercivity_graph,
            64 * 1024,
        ]
        .into_iter()
        .try_fold(0, add)
        .map(ByteCount)
    }
}
fn count(value: usize) -> Result<u64, ResourcePlanningError> {
    u64::try_from(value).map_err(|_| overflow())
}
fn tree<T>(entries: u64) -> Result<u64, ResourcePlanningError> {
    // A B-tree node holds up to eleven keys and never fewer than five outside the root; a key
    // therefore costs at most a fifth of a leaf (eleven slots, parent link and lengths) plus its
    // share of internal-node edges: under two key widths plus two words. Repeated entries are
    // counted, so unique keys are bounded from above.
    mul(entries, 2 * (size_of::<T>() + 2 * size_of::<usize>()) as u64)
}
fn add(left: u64, right: u64) -> Result<u64, ResourcePlanningError> {
    left.checked_add(right).ok_or_else(overflow)
}
fn mul(left: u64, right: u64) -> Result<u64, ResourcePlanningError> {
    left.checked_mul(right).ok_or_else(overflow)
}
fn overflow() -> ResourcePlanningError {
    ResourcePlanningError::Overflow("thermal support construction")
}
#[cfg(test)]
mod tests {
    use super::*;
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
    #[test]
    fn construction_profile_includes_projection_and_raw_coupling_growth() {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let mut mesh = canonical_tetrahedron_solver_mesh(order);
            let base =
                ThermalOperatorResourceFacts::derivation_resident_bytes_estimate(&mesh).unwrap();
            assert!(
                base.0 > AnalysisProjection::derivation_resident_bytes_estimate(&mesh).unwrap()
            );
            mesh.topology.volume_elements[0].node_ids.extend(100..200);
            assert!(
                ThermalOperatorResourceFacts::derivation_resident_bytes_estimate(&mesh)
                    .unwrap()
                    .0
                    > base.0
            );
        }
        assert!(add(u64::MAX, 1).is_err());
        assert!(mul(u64::MAX, 2).is_err());
    }
}
