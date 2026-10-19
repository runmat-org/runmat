use runmat_execution::Digest;

use crate::topology::AnalysisProjection;

use super::{
    components::connected_components, digest, operator_scale, validate_operator, verify_null_mode,
    MatrixDefiniteness, NullspaceAnchoring, NullspaceComponentEvidence,
    NullspaceEliminationEvidence, NullspaceEvidenceError, NullspaceKind, OperatorRole,
    RealSparseMatrix, CONSTRAINT_DOMAIN,
};

pub(super) fn prove(
    projection: &AnalysisProjection,
    operator: &RealSparseMatrix,
    prescribed: &[bool],
    positive_robin_weight: &[f64],
) -> Result<NullspaceEliminationEvidence, NullspaceEvidenceError> {
    let count = projection.nodes.len();
    if operator.rows() != count
        || operator.columns() != count
        || prescribed.len() != count
        || positive_robin_weight.len() != count
        || positive_robin_weight
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(NullspaceEvidenceError::DimensionMismatch);
    }
    if !matches!(
        operator.role(),
        OperatorRole::Conductivity | OperatorRole::Stiffness
    ) {
        return Err(NullspaceEvidenceError::IncompatibleOperator);
    }
    validate_operator(
        operator,
        operator.role(),
        MatrixDefiniteness::PositiveSemidefinite,
    )?;
    let component_nodes = connected_components(&projection.node_neighbors)?;
    let mut components = Vec::with_capacity(component_nodes.len());
    for (component_index, nodes) in component_nodes.into_iter().enumerate() {
        let prescribed_count = nodes.iter().filter(|node| prescribed[**node]).count();
        let robin_count = nodes
            .iter()
            .filter(|node| positive_robin_weight[**node] > 0.0)
            .count();
        let anchoring = match (prescribed_count, robin_count) {
            (0, 0) => return Err(NullspaceEvidenceError::UnderconstrainedComponent),
            (count, 0) => NullspaceAnchoring::PrescribedDof { count },
            (0, count) => NullspaceAnchoring::PositiveRobin { count },
            (prescribed_count, robin_count) => NullspaceAnchoring::PrescribedAndPositiveRobin {
                prescribed_count,
                robin_count,
            },
        };
        let mut constant_mode = vec![0.0; count];
        for node in &nodes {
            constant_mode[*node] = 1.0;
        }
        let applied = super::spmv(operator, &constant_mode)
            .map_err(|_| NullspaceEvidenceError::DimensionMismatch)?;
        if robin_count == 0 {
            verify_null_mode(operator, &constant_mode, &applied)?;
        } else {
            let energy = constant_mode
                .iter()
                .zip(&applied)
                .map(|(left, right)| left * right)
                .sum::<f64>();
            let diagonal_anchor = nodes
                .iter()
                .map(|node| positive_robin_weight[*node])
                .sum::<f64>();
            let tolerance = operator_scale(operator).max(1.0) * f64::EPSILON * 64.0;
            if !energy.is_finite() || energy <= tolerance || energy + tolerance < diagonal_anchor {
                return Err(NullspaceEvidenceError::MissingPositiveAnchor);
            }
        }
        let mut node_identities = nodes
            .iter()
            .map(|node| projection.nodes[*node].stable_identity)
            .collect::<Vec<_>>();
        node_identities.sort_unstable();
        components.push(NullspaceComponentEvidence {
            component_index,
            node_identities,
            null_mode_count: 1,
            projected_anchor_rank: 1,
            anchoring,
        });
    }
    NullspaceEliminationEvidence::new(
        NullspaceKind::ScalarConstantPerComponent,
        Digest::from_bytes(*projection.mesh_digest.bytes()),
        operator,
        digest(CONSTRAINT_DOMAIN, &(prescribed, positive_robin_weight))?,
        prescribed,
        components,
    )
}
