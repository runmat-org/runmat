use std::collections::BTreeSet;

use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;

use crate::{
    algebra::SparsePattern, constraints::CanonicalStructuralConstraintPlan,
    topology::AnalysisProjection,
};

use super::ResourcePlanningError;

const SYMBOLIC_PREFIX: &[u8] = b"runmat-fea-symbolic-operators/v1\0";
const SYMBOLIC_LIMITS: CanonicalLimits = CanonicalLimits::new(1024 * 1024, 64, 64, 8);
const TRANSLATIONAL_DOF_COUNT: usize = 3;
const REFERENCE_POINT_DOF_COUNT: usize = 6;

/// Material-independent sparse storage bounds for a structural solid solve.
///
/// The stiffness pattern contains every coupling that element assembly may emit, including
/// coefficients that happen to evaluate to zero for a particular material or geometry. Mass and
/// damping follow the currently admitted diagonal storage contracts. This plan is derived before
/// numerical assembly and therefore can be used for allocation admission without doing the work
/// it is intended to guard.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructuralSymbolicOperators {
    stiffness_upper_bound: SparsePattern,
    mass: SparsePattern,
    damping: SparsePattern,
    identity: Digest,
}

impl StructuralSymbolicOperators {
    pub fn derive(
        projection: &AnalysisProjection,
        constraints: &CanonicalStructuralConstraintPlan,
    ) -> Result<Self, ResourcePlanningError> {
        if constraints.topology_digest != projection.mesh_digest {
            return Err(ResourcePlanningError::Invalid(
                "constraint plan and analysis projection name different topology".into(),
            ));
        }
        let mesh_dofs = projection
            .nodes
            .len()
            .checked_mul(TRANSLATIONAL_DOF_COUNT)
            .ok_or(ResourcePlanningError::Overflow("structural mesh DOFs"))?;
        let reference_dofs = constraints
            .reference_points
            .len()
            .checked_mul(REFERENCE_POINT_DOF_COUNT)
            .ok_or(ResourcePlanningError::Overflow(
                "structural reference-point DOFs",
            ))?;
        let total_dofs = mesh_dofs
            .checked_add(reference_dofs)
            .ok_or(ResourcePlanningError::Overflow("structural DOFs"))?;
        if total_dofs == 0 {
            return Err(ResourcePlanningError::Invalid(
                "structural symbolic planning requires at least one DOF".into(),
            ));
        }

        let mut stiffness_rows = (0..total_dofs)
            .map(|_| BTreeSet::<usize>::new())
            .collect::<Vec<_>>();
        for element in &projection.elements {
            for &row_node in &element.node_indices {
                let row_offset = checked_dof_offset(row_node, projection.nodes.len())?;
                for &column_node in &element.node_indices {
                    let column_offset = checked_dof_offset(column_node, projection.nodes.len())?;
                    for row_axis in 0..TRANSLATIONAL_DOF_COUNT {
                        for column_axis in 0..TRANSLATIONAL_DOF_COUNT {
                            stiffness_rows[row_offset + row_axis]
                                .insert(column_offset + column_axis);
                        }
                    }
                }
            }
        }
        let stiffness_upper_bound = pattern_from_rows(stiffness_rows, total_dofs)?;
        let diagonal = diagonal_pattern(total_dofs)?;
        let pattern_identities = (
            stiffness_upper_bound
                .identity()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
            diagonal
                .identity()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
            diagonal
                .identity()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
        );
        let encoded = encode_contract(
            SYMBOLIC_PREFIX,
            "structural-symbolic-operators",
            &pattern_identities,
            SYMBOLIC_LIMITS,
        )
        .map_err(ResourcePlanningError::from)?;
        Ok(Self {
            stiffness_upper_bound,
            mass: diagonal.clone(),
            damping: diagonal,
            identity: Digest::sha256(encoded),
        })
    }

    pub const fn identity(&self) -> Digest {
        self.identity
    }

    pub fn stiffness_upper_bound(&self) -> &SparsePattern {
        &self.stiffness_upper_bound
    }

    pub fn patterns(&self) -> [&SparsePattern; 3] {
        [&self.stiffness_upper_bound, &self.mass, &self.damping]
    }
}

fn checked_dof_offset(
    node_index: usize,
    node_count: usize,
) -> Result<usize, ResourcePlanningError> {
    if node_index >= node_count {
        return Err(ResourcePlanningError::Invalid(
            "projected element references an absent node".into(),
        ));
    }
    node_index
        .checked_mul(TRANSLATIONAL_DOF_COUNT)
        .ok_or(ResourcePlanningError::Overflow("structural DOF offset"))
}

fn diagonal_pattern(dofs: usize) -> Result<SparsePattern, ResourcePlanningError> {
    let rows = (0..dofs)
        .map(|row| BTreeSet::from([row]))
        .collect::<Vec<_>>();
    pattern_from_rows(rows, dofs)
}

fn pattern_from_rows(
    rows: Vec<BTreeSet<usize>>,
    columns: usize,
) -> Result<SparsePattern, ResourcePlanningError> {
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    let mut column_indices = Vec::new();
    offsets.push(0_u64);
    for row in rows {
        for column in row {
            column_indices.push(
                u64::try_from(column)
                    .map_err(|_| ResourcePlanningError::Overflow("sparse column index"))?,
            );
        }
        offsets.push(
            u64::try_from(column_indices.len())
                .map_err(|_| ResourcePlanningError::Overflow("sparse row offset"))?,
        );
    }
    SparsePattern::try_new(
        u64::try_from(offsets.len() - 1)
            .map_err(|_| ResourcePlanningError::Overflow("sparse row count"))?,
        u64::try_from(columns)
            .map_err(|_| ResourcePlanningError::Overflow("sparse column count"))?,
        offsets,
        column_indices,
    )
    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))
}
