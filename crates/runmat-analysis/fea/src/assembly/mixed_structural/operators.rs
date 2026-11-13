use crate::algebra::{MatrixProperties, OperatorRole, RealSparseAssembler, RealSparseMatrix};

use super::super::dofs::{StructuralDofKind, StructuralDofLayout};
use super::LinearAssemblyError;

pub(super) fn embed_solid_matrix(
    matrix: &RealSparseMatrix,
    layout: &StructuralDofLayout,
    solid_node_count: usize,
) -> Result<RealSparseMatrix, LinearAssemblyError> {
    let row_map = solid_row_map(layout, solid_node_count)?;
    if matrix.rows() != row_map.len() || matrix.columns() != row_map.len() {
        return Err(LinearAssemblyError::SparseOperator(
            "solid operator shape does not match canonical solid translations".into(),
        ));
    }
    let mut assembler =
        RealSparseAssembler::square(layout.total_dof_count(), matrix.role(), matrix.properties());
    for contribution in matrix.contributions() {
        assembler.add_contribution(contribution.clone());
    }
    for (source_row, target_row) in row_map.iter().copied().enumerate() {
        for entry in matrix.pattern().row_range(source_row) {
            assembler.add(
                target_row,
                row_map[matrix.pattern().column(entry)],
                matrix.values()[entry],
            );
        }
    }
    assembler.finish().map_err(sparse_error)
}

fn solid_row_map(
    layout: &StructuralDofLayout,
    solid_node_count: usize,
) -> Result<Vec<usize>, LinearAssemblyError> {
    let mut rows = Vec::with_capacity(solid_node_count.saturating_mul(3));
    for node in 0..solid_node_count {
        for kind in [
            StructuralDofKind::Ux,
            StructuralDofKind::Uy,
            StructuralDofKind::Uz,
        ] {
            rows.push(layout.index(node, kind).ok_or_else(|| {
                LinearAssemblyError::SparseOperator(
                    "canonical solid node is missing a translation DOF".into(),
                )
            })?);
        }
    }
    Ok(rows)
}

pub(super) fn sum(
    left: &RealSparseMatrix,
    right: &RealSparseMatrix,
    role: OperatorRole,
    properties: MatrixProperties,
) -> Result<RealSparseMatrix, LinearAssemblyError> {
    RealSparseMatrix::scaled_sum(&[(1.0, left), (1.0, right)], role, properties)
        .map_err(sparse_error)
}

pub(super) fn sparse_error(error: impl std::fmt::Debug) -> LinearAssemblyError {
    LinearAssemblyError::SparseOperator(format!("{error:?}"))
}
