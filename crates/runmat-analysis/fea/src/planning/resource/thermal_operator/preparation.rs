//! Admission of thermal support construction before and after the support is built.
use super::{AnalysisProjection, ByteCount, ResourcePlanningError, ThermalOperatorResourceFacts};
use runmat_meshing_core::SolverMeshArtifact;

impl ThermalOperatorResourceFacts {
    /// Additional requested-heap estimate for support derivation, including projection and mesh
    /// validation, before the support exists. Excludes borrowed input, allocator/native/RSS
    /// overhead. The support entries are bounded by a distinct-pair count over the mesh's
    /// cliques ([`distinct_support_entries`]), which is exact for a mesh without finite-conductance
    /// interfaces; once the support is built, [`ThermalOperatorResourceFacts::derive`] retains
    /// the same law at the exact entry count. The count's scratch lies within the builder's
    /// entry-free floor, `derivation_resident_bytes(mesh, Some(0))`, which callers admit first.
    pub fn derivation_resident_bytes_estimate(
        mesh: &SolverMeshArtifact,
    ) -> Result<ByteCount, ResourcePlanningError> {
        Self::derivation_resident_bytes(mesh, None)
    }

    /// The support-derivation law: the projection (with the split's added nodes when a conformal
    /// interface may split it), then the largest of the split's pair list, the support builder (with its finished pattern) and the finished pattern
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
        // The split rewrites the projection in place (`InterfaceDofs::split_projection`): it adds
        // each second-side node and its neighbor list header, at most one neighbor entry per
        // conductance pair, and holds its exact-length (row, column) pair list until the new
        // adjacency is built. The pair list is freed before the support builder runs.
        let (projection, split_pairs) = if interface_nodes > 0 {
            let coupled = interface_widths.clone().try_fold(0, |total, width| {
                add(total, mul(2, mul(count(width)?, count(width)?)?)?)
            })?;
            let element = mesh
                .topology
                .volume_elements
                .iter()
                .map(|e| count(e.node_ids.len()))
                .try_fold(0, |total, width| {
                    let width = width?;
                    add(total, mul(width, width.saturating_sub(1))?)
                })?;
            let pair = std::mem::size_of::<(usize, usize)>() as u64;
            let node = (std::mem::size_of::<crate::topology::ProjectedNode>()
                + std::mem::size_of::<Vec<usize>>()) as u64;
            (
                add(
                    projection,
                    add(mul(interface_nodes, node)?, mul(coupled, 8)?)?,
                )?,
                mul(add(coupled, element)?, pair)?,
            )
        } else {
            (projection, 0)
        };
        let nodes = add(count(mesh.topology.nodes.len())?, interface_nodes)?;
        let widths = || {
            mesh.topology
                .volume_elements
                .iter()
                .map(|e| e.node_ids.len())
                .chain(
                    mesh.topology
                        .boundary_faces
                        .iter()
                        .map(|f| f.node_ids.len()),
                )
        };
        let extra_incidences = mul(interface_nodes, 2)?;
        let incidences = add(
            widths().try_fold(0, |total, width| add(total, count(width)?))?,
            extra_incidences,
        )?;
        let entries = match entries {
            Some(entries) => entries,
            None => {
                // Each conductance clique couples at most both sides of its face's dofs.
                let conductance = interface_widths.clone().try_fold(0, |total, width| {
                    add(total, mul(mul(2, count(width)?)?, mul(2, count(width)?)?)?)
                })?;
                let mesh_cliques = match distinct_support_entries(mesh)? {
                    Some(entries) => entries,
                    // Node ids the count cannot index: every row has a diagonal and each clique
                    // couples its dofs pairwise; counting repeated couplings repeatedly bounds
                    // the distinct entries from above.
                    None => widths()
                        .map(|width| count(width).and_then(|width| mul(width, width)))
                        .try_fold(nodes, |total, square| add(total, square?))?,
                };
                add(mesh_cliques, conductance)?.min(mul(nodes, nodes)?)
            }
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
        [
            projection,
            builder.max(symbolic).max(split_pairs),
            64 * 1024,
        ]
        .into_iter()
        .try_fold(0, add)
        .map(ByteCount)
    }
}

/// An upper bound on the support entries that the mesh's volume and boundary-face cliques
/// produce, with the split a finite-conductance interface applies: one diagonal per row plus the
/// distinct (row, column) pairs the cliques couple. A split gives each conformal interface node a
/// second-side dof, so a pair on nodes `i` and `j` becomes at most `(1 + [i split]) · (1 + [j
/// split])` pairs; the count weights each distinct pair by that. Without interfaces the bound is
/// the exact entry count of [`super::support::build`] on the projection. Returns `None` when the
/// node ids are not strictly increasing or a clique names an unknown node; the projection then
/// refuses the mesh, and the caller keeps the repeated-coupling bound.
///
/// Scratch: row starts and the node→clique incidence (one word each), a row marker and a split
/// bitset, all within the support builder's entry-free floor.
fn distinct_support_entries(
    mesh: &SolverMeshArtifact,
) -> Result<Option<u64>, ResourcePlanningError> {
    let topology = &mesh.topology;
    let nodes = topology.nodes.as_slice();
    if !nodes
        .windows(2)
        .all(|pair| pair[0].node_id < pair[1].node_id)
    {
        return Ok(None);
    }
    let dense = nodes.first().is_some_and(|first| {
        nodes.last().map(|last| last.node_id - first.node_id) == Some(nodes.len() as u64 - 1)
    });
    let first = nodes.first().map_or(0, |node| node.node_id);
    let index = |id: u64| -> Option<usize> {
        if dense {
            let offset = usize::try_from(id.checked_sub(first)?).ok()?;
            (offset < nodes.len()).then_some(offset)
        } else {
            nodes.binary_search_by_key(&id, |node| node.node_id).ok()
        }
    };
    let elements = topology.volume_elements.len();
    let cliques = elements + topology.boundary_faces.len();
    let clique = |c: usize| -> &[u64] {
        if c < elements {
            &topology.volume_elements[c].node_ids
        } else {
            &topology.boundary_faces[c - elements].node_ids
        }
    };
    let rows = nodes.len();
    let mut split = vec![0_u64; rows.div_ceil(64)];
    for face in topology
        .boundary_faces
        .iter()
        .filter(|f| f.role == runmat_meshing_core::BoundaryFaceRole::ConformalInterface)
    {
        for &id in &face.node_ids {
            let Some(node) = index(id) else {
                return Ok(None);
            };
            split[node / 64] |= 1 << (node % 64);
        }
    }
    let weight = |node: usize| 1 + (split[node / 64] >> (node % 64) & 1);
    // Exclusive prefix of the incidence counts; filling advances each start to its row's end.
    let mut starts = vec![0_usize; rows + 1];
    for c in 0..cliques {
        for &id in clique(c) {
            let Some(node) = index(id) else {
                return Ok(None);
            };
            starts[node + 1] += 1;
        }
    }
    for row in 0..rows {
        starts[row + 1] += starts[row];
    }
    let mut incident = vec![0_usize; starts[rows]];
    for c in 0..cliques {
        for &id in clique(c) {
            let Some(node) = index(id) else {
                return Ok(None);
            };
            incident[starts[node]] = c;
            starts[node] += 1;
        }
    }
    let mut marker = vec![usize::MAX; rows];
    let mut total = 0_u64;
    for row in 0..rows {
        let begin = if row == 0 { 0 } else { starts[row - 1] };
        let row_weight = weight(row);
        marker[row] = row;
        total = add(total, row_weight * row_weight)?;
        for &c in &incident[begin..starts[row]] {
            for &id in clique(c) {
                let Some(column) = index(id) else {
                    return Ok(None);
                };
                if marker[column] != row {
                    marker[column] = row;
                    total = add(total, row_weight * weight(column))?;
                }
            }
        }
    }
    Ok(Some(total))
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
