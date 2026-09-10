use std::collections::BTreeMap;

use crate::algebra::{
    ContributionDomain, ContributionId, ContributionProvenance, MatrixProperties, OperatorRole,
    RealSparseAssembler, RealSparseMatrix, RealSparseMatrixError,
};

pub(super) fn empty_rows(dof_count: usize) -> Vec<BTreeMap<usize, f64>> {
    (0..dof_count).map(|_| BTreeMap::new()).collect()
}

pub(super) fn scatter_csr<const N: usize>(
    rows: &mut [BTreeMap<usize, f64>],
    dof_offsets: &[usize],
    element_stiffness: &[[f64; N]; N],
) {
    for (local_row_node, row_offset) in dof_offsets.iter().copied().enumerate() {
        for local_row_axis in 0..3 {
            let local_row = local_row_node * 3 + local_row_axis;
            let global_row = row_offset + local_row_axis;
            for (local_column_node, column_offset) in dof_offsets.iter().copied().enumerate() {
                for local_column_axis in 0..3 {
                    let local_column = local_column_node * 3 + local_column_axis;
                    let global_column = column_offset + local_column_axis;
                    *rows[global_row].entry(global_column).or_insert(0.0) +=
                        element_stiffness[local_row][local_column];
                }
            }
        }
    }
}

pub(super) fn rows_to_sparse(
    rows: Vec<BTreeMap<usize, f64>>,
    role: OperatorRole,
) -> Result<RealSparseMatrix, RealSparseMatrixError> {
    let mut assembler = RealSparseAssembler::square(
        rows.len(),
        role,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    );
    assembler.add_contribution(ContributionProvenance {
        domain: ContributionDomain::Structural,
        contribution_id: ContributionId(
            match role {
                OperatorRole::Stiffness => "solid_element_stiffness",
                OperatorRole::Mass => "solid_element_consistent_mass",
                OperatorRole::Damping => "solid_element_rayleigh_damping",
                _ => "solid_element_operator",
            }
            .to_string(),
        ),
    });
    for (row_index, row) in rows.into_iter().enumerate() {
        for (column, value) in row {
            assembler.add(row_index, column, value);
        }
    }
    assembler.finish()
}
