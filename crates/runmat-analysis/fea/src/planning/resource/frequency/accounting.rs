use super::*;

#[cfg(test)]
mod tests;

pub(super) fn direct_workspace_bytes(
    full_dofs: u64,
    reduced_dofs: u64,
    reduced_nonzeros: u64,
    reduction: &crate::constraints::AffineConstraintReduction,
    restart: u64,
) -> Result<u64, ResourcePlanningError> {
    let block_dimension = multiply(reduced_dofs, 2, "real block dimension")?;
    let reduced_csr = add(
        multiply(
            add(
                multiply(reduced_dofs, 3, "reduced CSR rows")?,
                3,
                "reduced CSR rows",
            )?,
            8,
            "reduced CSR row bytes",
        )?,
        multiply(reduced_nonzeros, 16, "reduced CSR entry bytes")?,
        "reduced CSR bytes",
    )?;
    let reduction_indices = [
        reduction.free_dof_indices.len(),
        reduction.pivot_dof_indices.len(),
        reduction.recovery_row_offsets.len(),
        reduction.recovery_column_indices.len(),
    ]
    .into_iter()
    .try_fold(0_u64, |total, value| {
        add(
            total,
            count(value, "affine reduction indices")?,
            "affine reduction indices",
        )
    })?;
    let reduction_values = add(
        count(
            reduction.recovery_coefficients.len(),
            "affine reduction coefficients",
        )?,
        count(reduction.offset.len(), "affine reduction offsets")?,
        "affine reduction values",
    )?;
    let reduction_bytes = multiply(
        add(
            reduction_indices,
            reduction_values,
            "affine reduction storage",
        )?,
        8,
        "affine reduction bytes",
    )?;
    let krylov_vectors = multiply(
        add(restart, 12, "GMRES and right-Jacobi live vector count")?,
        block_dimension,
        "GMRES live vector values",
    )?;
    let hessenberg = multiply(
        add(restart, 1, "Hessenberg rows")?,
        restart,
        "Hessenberg values",
    )?;
    let small_vectors = add(
        multiply(restart, 3, "GMRES rotations and coefficients")?,
        1,
        "GMRES projected RHS",
    )?;
    let full_order_vectors = multiply(full_dofs, 14, "full-order harmonic work vectors")?;
    let floating_values = add(
        add(krylov_vectors, hessenberg, "direct harmonic workspace")?,
        add(
            small_vectors,
            full_order_vectors,
            "direct harmonic workspace",
        )?,
        "direct harmonic workspace",
    )?;
    add(
        add(reduced_csr, reduction_bytes, "direct harmonic workspace")?,
        multiply(floating_values, 8, "direct harmonic floating workspace")?,
        "direct harmonic workspace",
    )
}

pub(super) fn modal_workspace_bytes(
    full_dofs: u64,
    mode_count: u64,
) -> Result<u64, ResourcePlanningError> {
    let full_by_modes = multiply(full_dofs, mode_count, "modal shape values")?;
    let dense = multiply(mode_count, mode_count, "modal dense values")?;
    let values = add(
        multiply(full_by_modes, 4, "modal projection workspaces")?,
        add(
            multiply(dense, 11, "modal dense workspaces")?,
            add(
                multiply(full_dofs, 14, "modal full-order work vectors")?,
                multiply(mode_count, 8, "modal reduced work vectors")?,
                "modal vector workspaces",
            )?,
            "modal dense and vector workspaces",
        )?,
        "modal harmonic workspace",
    )?;
    multiply(values, 8, "modal harmonic workspace bytes")
}
