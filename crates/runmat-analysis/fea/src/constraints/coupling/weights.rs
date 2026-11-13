//! Original conservative subtriangle weights with compact counted node indexes.
use super::*;

pub(super) fn target_weights(
    mesh: &SolverMeshArtifact,
    target: &ResolvedConstraintTarget,
    coordinates: &NodeValues<[f64; 3]>,
    constraint_id: &str,
) -> Result<Vec<ConstraintCouplingWeight>, ConstraintPlanError> {
    if target.matched_entities.is_empty() && target.matched_node_identities.len() == 1 {
        return Ok(vec![ConstraintCouplingWeight {
            node_identity: target.matched_node_identities[0],
            weight: 1.0,
        }]);
    }
    let identities = crate::topology::storage::solver_node_identities(mesh)?;
    let mut measures = NodeValues::derive(
        mesh.topology
            .nodes
            .iter()
            .map(|n| (n.stable_identity, 0.0_f64)),
        mesh.topology.nodes.len(),
    )?;
    for face in mesh.topology.boundary_faces.iter().filter(|face| {
        face.provenance
            .iter()
            .any(|entity| target.matched_entities.contains(entity))
    }) {
        check_cancelled()?;
        let triangles: &[[usize; 3]] = match face.order {
            BoundaryTriangleOrder::Tri3 => &[[0, 1, 2]],
            BoundaryTriangleOrder::Tri6 => &[[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]],
        };
        for indices in triangles {
            let vertices = indices.map(|index| {
                face.node_ids
                    .get(index)
                    .and_then(|id| identities.get(id).copied())
            });
            let [Some(a), Some(b), Some(c)] = vertices else {
                return Err(ConstraintPlanError::Invalid(format!(
                    "constraint `{constraint_id}` coupling facet is malformed"
                )));
            };
            let points = [a, b, c].map(|id| coordinates.get(&id).copied());
            let [Some(pa), Some(pb), Some(pc)] = points else {
                return Err(ConstraintPlanError::Invalid(format!(
                    "constraint `{constraint_id}` coupling facet is malformed"
                )));
            };
            let area = 0.5 * magnitude(cross(subtract(pb, pa), subtract(pc, pa)));
            if !area.is_finite() || area <= 1.0e-18 {
                return Err(ConstraintPlanError::Invalid(format!(
                    "constraint `{constraint_id}` coupling facet has zero area"
                )));
            }
            for node in [a, b, c] {
                *measures.get_mut(&node).ok_or_else(|| {
                    ConstraintPlanError::Invalid("coupling measure node is absent".into())
                })? += area / 3.0;
            }
        }
    }
    // Filter untouched nodes before summation: preserve original selected-node order and arithmetic.
    let total = measures
        .entries()
        .filter(|(_, m)| *m > 0.0)
        .map(|(_, m)| *m)
        .sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(ConstraintPlanError::Invalid(format!(
            "constraint `{constraint_id}` coupling target has no measurable boundary"
        )));
    }
    let count = measures.entries().filter(|(_, m)| *m > 0.0).count();
    let mut weights = reserve(count)?;
    weights.extend(measures.into_entries().filter(|(_, m)| *m > 0.0).map(
        |(node_identity, measure)| ConstraintCouplingWeight {
            node_identity,
            weight: measure / total,
        },
    ));
    Ok(weights)
}
