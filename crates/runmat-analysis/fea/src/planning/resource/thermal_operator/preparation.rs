//! Admission of thermal support construction before and after the support is built.
use super::{AnalysisProjection, ByteCount, ResourcePlanningError, ThermalOperatorResourceFacts};
use runmat_meshing_core::SolverMeshArtifact;

impl ThermalOperatorResourceFacts {
    /// Additional requested-heap estimate for support derivation, including projection and mesh
    /// validation, before the support exists. Excludes borrowed input, allocator/native/RSS
    /// overhead. The distinct support entries are bounded by the repeated clique couplings; once
    /// the support is built, [`ThermalOperatorResourceFacts::derive`] retains the same law at the
    /// exact entry count.
    pub fn derivation_resident_bytes_estimate(
        mesh: &SolverMeshArtifact,
    ) -> Result<ByteCount, ResourcePlanningError> {
        Self::derivation_resident_bytes(mesh, None)
    }

    /// The support-derivation law: the projection (twice when a conformal interface may split it),
    /// then the larger of the support builder (with its finished pattern) and the finished pattern
    /// plus the symbolic factor analysis that sizes the proving factor. `entries` is the support's
    /// distinct entry count when known; otherwise the repeated clique couplings bound it.
    pub(crate) fn derivation_resident_bytes(
        mesh: &SolverMeshArtifact,
        entries: Option<u64>,
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
        let interface_faces = count(interface_widths.clone().count())?;
        let interface_nodes = interface_widths
            .clone()
            .try_fold(0, |total, width| add(total, count(width)?))?;
        let projection = if interface_nodes > 0 {
            mul(projection, 2)?
        } else {
            projection
        };
        let nodes = add(count(mesh.topology.nodes.len())?, interface_nodes)?;
        let widths = || {
            mesh.topology
                .volume_elements
                .iter()
                .map(|e| e.node_ids.len())
                .chain(mesh.topology.boundary_faces.iter().map(|f| f.node_ids.len()))
        };
        let extra_incidences = mul(interface_nodes, 2)?;
        let incidences = add(
            widths().try_fold(0, |total, width| add(total, count(width)?))?,
            extra_incidences,
        )?;
        let entries = match entries {
            Some(entries) => entries,
            // Every row has a diagonal; each clique couples its dofs pairwise. Repeated
            // couplings are counted repeatedly, which bounds the distinct entries from above.
            None => widths()
                .map(|width| count(width).and_then(|width| mul(width, width)))
                .chain(interface_widths.map(|width| {
                    count(width).and_then(|width| mul(mul(2, width)?, mul(2, width)?))
                }))
                .try_fold(nodes, |total, square| add(total, square?))?
                .min(mul(nodes, nodes)?),
        };
        let builder = super::support::construction_bytes(
            super::support::SupportCounts {
                rows: nodes,
                incidences,
                extra_cliques: interface_faces,
                extra_incidences,
            },
            entries,
        )?;
        let pattern = add(mul(add(nodes, 1)?, 8)?, mul(entries, 8)?)?;
        let symbolic = add(pattern, symbolic_analysis_bytes(nodes, entries)?)?;
        [projection, builder.max(symbolic), 64 * 1024]
            .into_iter()
            .try_fold(0, add)
            .map(ByteCount)
    }
}

/// Requested heap of the symbolic Cholesky analysis behind the proving factor's admission
/// (`CoercivityWorkEnvelope::derive`) on a support of `rows` rows and `entries` stored entries:
/// its usize copy of the pattern and zero values (three words per entry, one per row), and the
/// ordering and elimination-tree analysis of `A + Aᵀ`, which holds the symmetric adjacency and a
/// fixed number of row-length index arrays.
fn symbolic_analysis_bytes(rows: u64, entries: u64) -> Result<u64, ResourcePlanningError> {
    let word = std::mem::size_of::<usize>() as u64;
    add(
        mul(entries, SYMBOLIC_WORDS_PER_ENTRY * word)?,
        mul(add(rows, 1)?, SYMBOLIC_WORDS_PER_ROW * word)?,
    )
}
const SYMBOLIC_WORDS_PER_ENTRY: u64 = 6;
const SYMBOLIC_WORDS_PER_ROW: u64 = 24;
fn count(value: usize) -> Result<u64, ResourcePlanningError> {
    u64::try_from(value).map_err(|_| overflow())
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
