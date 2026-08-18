use std::cmp::Ordering;

use super::{MeshingContractError, PersistentEntityKind, SolverMeshNode, SolverNodeExactParameter};

const MAX_EXACT_PARAMETERS_PER_NODE: usize = 64;

pub(super) fn validate_node_exact_parameters(
    node: &SolverMeshNode,
) -> Result<(), MeshingContractError> {
    if node.exact_parameters.len() > MAX_EXACT_PARAMETERS_PER_NODE
        || node
            .exact_parameters
            .windows(2)
            .any(|pair| canonical_cmp(&pair[0], &pair[1]) != Ordering::Less)
    {
        return Err(invalid(
            "exact node parameters must be bounded, unique, and canonically ordered",
        ));
    }
    for parameter in &node.exact_parameters {
        match parameter {
            SolverNodeExactParameter::Curve {
                source_edge_id,
                parameter,
            } => {
                if source_edge_id.kind != PersistentEntityKind::Edge
                    || !parameter.is_finite()
                    || !node.provenance.contains(source_edge_id)
                {
                    return Err(invalid(
                        "curve parameter must be finite and bind a provenance edge",
                    ));
                }
                source_edge_id.validate()?;
            }
            SolverNodeExactParameter::Surface {
                source_face_id,
                chart_id,
                evaluator_uv,
            } => {
                if source_face_id.kind != PersistentEntityKind::Face
                    || *chart_id == super::StableDigest::ZERO
                    || evaluator_uv.iter().any(|value| !value.is_finite())
                    || !node.provenance.contains(source_face_id)
                {
                    return Err(invalid(
                        "surface parameter must be finite and bind a provenance face and chart",
                    ));
                }
                source_face_id.validate()?;
            }
        }
    }
    Ok(())
}

fn canonical_cmp(left: &SolverNodeExactParameter, right: &SolverNodeExactParameter) -> Ordering {
    match (left, right) {
        (
            SolverNodeExactParameter::Curve {
                source_edge_id: left,
                ..
            },
            SolverNodeExactParameter::Curve {
                source_edge_id: right,
                ..
            },
        ) => left.cmp(right),
        (SolverNodeExactParameter::Curve { .. }, SolverNodeExactParameter::Surface { .. }) => {
            Ordering::Less
        }
        (SolverNodeExactParameter::Surface { .. }, SolverNodeExactParameter::Curve { .. }) => {
            Ordering::Greater
        }
        (
            SolverNodeExactParameter::Surface {
                source_face_id: left_face,
                chart_id: left_chart,
                ..
            },
            SolverNodeExactParameter::Surface {
                source_face_id: right_face,
                chart_id: right_chart,
                ..
            },
        ) => (left_face, left_chart).cmp(&(right_face, right_chart)),
    }
}

fn invalid(reason: impl Into<String>) -> MeshingContractError {
    MeshingContractError::invalid("mesh node exact parameters", reason)
}
