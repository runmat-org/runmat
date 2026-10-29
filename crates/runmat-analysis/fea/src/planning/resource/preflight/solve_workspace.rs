//! Solver-specific constraint transforms and solve workspace.
use super::*;

pub(super) fn solve_policy(
    preconditioner: SpdPreconditionerKind,
    path: StaticSolvePath,
    dofs: u64,
    constraint_rows: u64,
    enforcement: ConstraintEnforcement,
    stiffness_nonzeros: u64,
    admission: SolverAdmissionResourcePolicy,
    transpose_index_bytes: u64,
) -> Result<SolveResourcePolicy, ResourcePlanningError> {
    // Multiplier enforcement is solved on the affine-reduced SPD system (one canonical path).
    // The elastic rigidity path always solves with AMG-preconditioned CG.
    let preconditioner = match path {
        StaticSolvePath::ElasticRigidityMultigrid => SpdPreconditionerKind::Amg,
        StaticSolvePath::ProvingFactorization => preconditioner,
    };
    let storage = match preconditioner {
        // The direct factor is the coercivity proof; its bytes are the admission envelope's
        // factorization charge.
        SpdPreconditionerKind::Jacobi
        | SpdPreconditionerKind::Auto
        | SpdPreconditionerKind::Direct => PreconditionerStorage::Diagonal,
        SpdPreconditionerKind::Amg => {
            let bytes = runmat_sparse_solvers::estimate::amg_bytes(
                dofs as usize,
                stiffness_nonzeros as usize,
                3,
            );
            PreconditionerStorage::SparseCsr {
                nonzero_upper_bound: Count(stiffness_nonzeros.saturating_mul(3)),
                resident_bytes_upper_bound: ByteCount(bytes),
            }
        }
        SpdPreconditionerKind::Ilu0 => {
            let row_offsets = checked_bytes(
                dofs.checked_add(1)
                    .ok_or(ResourcePlanningError::Overflow("ILU row offsets"))?,
                8,
                "ILU row offsets",
            )?;
            let entries = checked_bytes(stiffness_nonzeros, 16, "ILU columns and values")?;
            PreconditionerStorage::SparseCsr {
                nonzero_upper_bound: Count(stiffness_nonzeros),
                resident_bytes_upper_bound: ByteCount(checked_add(
                    row_offsets,
                    entries,
                    "ILU resident bytes",
                )?),
            }
        }
    };
    Ok(SolveResourcePolicy::Iterative {
        vector_count: Count(structural_solve_vector_count(
            dofs,
            constraint_rows,
            enforcement,
        )?),
        scalar: NumericalScalar::RealF64,
        preconditioner: storage,
        admission,
        transpose_index_bytes_upper_bound: ByteCount(transpose_index_bytes),
        path,
    })
}

pub(in crate::planning::resource) fn saddle_gmres_workspace_bytes(
    dofs: u64,
    constraint_rows: u64,
) -> Result<u64, ResourcePlanningError> {
    let vector_count = structural_solve_vector_count(
        dofs,
        constraint_rows,
        ConstraintEnforcement::LagrangeMultiplier,
    )?;
    checked_bytes(
        vector_count
            .checked_mul(dofs)
            .ok_or(ResourcePlanningError::Overflow(
                "saddle GMRES workspace values",
            ))?,
        8,
        "saddle GMRES workspace bytes",
    )
}

/// Bounds the retained restarted-GMRES basis and saddle work in units of full physical vectors.
/// The sixteen auxiliary saddle vectors cover RHS/initial/solution/residual, an in-flight
/// Arnoldi vector, operator output, constraint transforms and temporary reductions. Two full
/// physical vectors cover scoped CSR chunk output plus its ordered merge. Hessenberg, Givens,
/// and projected-RHS scalars are charged separately before rounding up to physical vectors.
fn structural_solve_vector_count(
    dofs: u64,
    constraint_rows: u64,
    enforcement: ConstraintEnforcement,
) -> Result<u64, ResourcePlanningError> {
    if enforcement != ConstraintEnforcement::LagrangeMultiplier {
        return Ok(STRUCTURAL_SOLVE_VECTOR_COUNT);
    }
    if dofs == 0 {
        return Err(ResourcePlanningError::Invalid(
            "multiplier solve requires physical DOFs".into(),
        ));
    }
    let saddle = dofs
        .checked_add(constraint_rows)
        .ok_or(ResourcePlanningError::Overflow("saddle dimension"))?;
    let restart = saddle.min(32);
    let basis_and_auxiliary =
        saddle
            .checked_mul(restart + 17)
            .ok_or(ResourcePlanningError::Overflow(
                "GMRES basis and work vectors",
            ))?;
    let scalar_work = (restart + 1)
        .checked_mul(restart + 6)
        .ok_or(ResourcePlanningError::Overflow("GMRES Hessenberg work"))?;
    let spmv_work = dofs
        .checked_mul(2)
        .ok_or(ResourcePlanningError::Overflow("scoped saddle SpMV work"))?;
    let values = basis_and_auxiliary
        .checked_add(scalar_work)
        .and_then(|sum| sum.checked_add(spmv_work))
        .ok_or(ResourcePlanningError::Overflow("multiplier solve values"))?;
    Ok(values.div_ceil(dofs).max(STRUCTURAL_SOLVE_VECTOR_COUNT))
}

pub(super) fn affine_transpose_index_peak_bytes(
    reduction: &AffineConstraintReduction,
) -> Result<u64, ResourcePlanningError> {
    let entries = u64::try_from(reduction.recovery_column_indices.len())
        .map_err(|_| ResourcePlanningError::Overflow("affine transpose entries"))?;
    let columns = u64::try_from(reduction.reduced_dof_count)
        .map_err(|_| ResourcePlanningError::Overflow("affine transpose columns"))?;
    checked_add(
        checked_bytes(entries, 16, "affine transpose row-entry pairs")?,
        checked_bytes(
            columns
                .checked_add(1)
                .ok_or(ResourcePlanningError::Overflow("affine transpose offsets"))?,
            16,
            "affine transpose offsets and construction cursors",
        )?,
        "affine transpose index peak bytes",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplier_preflight_plans_the_reduced_spd_solve() {
        let admission = SolverAdmissionResourcePolicy {
            dimension: Count(8),
            factor_nonzero_upper_bound: Count(36),
            update_count_upper_bound: Count(100),
            resident_bytes_upper_bound: ByteCount(1_000),
        };
        let policy = solve_policy(
            SpdPreconditionerKind::Ilu0,
            StaticSolvePath::ProvingFactorization,
            12,
            7,
            ConstraintEnforcement::LagrangeMultiplier,
            144,
            admission.clone(),
            0,
        )
        .unwrap();
        let SolveResourcePolicy::Iterative {
            vector_count,
            preconditioner,
            admission: planned_admission,
            ..
        } = policy
        else {
            panic!("multiplier solve is planned as the affine-reduced SPD solve");
        };
        assert_eq!(planned_admission, admission);
        assert_eq!(vector_count, Count(101));
        assert!(matches!(
            preconditioner,
            PreconditionerStorage::SparseCsr { .. }
        ));
    }

    #[test]
    fn multiplier_gmres_basis_is_charged_above_scoped_pcg() {
        assert_eq!(
            structural_solve_vector_count(12, 7, ConstraintEnforcement::LagrangeMultiplier)
                .unwrap(),
            101
        );
        assert_eq!(
            structural_solve_vector_count(12, 7, ConstraintEnforcement::AffineReduction).unwrap(),
            STRUCTURAL_SOLVE_VECTOR_COUNT
        );
        assert!(structural_solve_vector_count(
            u64::MAX,
            1,
            ConstraintEnforcement::LagrangeMultiplier
        )
        .is_err());
    }

    #[test]
    fn affine_transpose_index_charges_entries_offsets_and_build_cursors() {
        let reduction = AffineConstraintReduction {
            full_dof_count: 4,
            reduced_dof_count: 2,
            free_dof_indices: vec![0, 1],
            pivot_dof_indices: vec![2, 3],
            recovery_row_offsets: vec![0, 1, 2, 4, 5],
            recovery_column_indices: vec![0, 1, 0, 1, 1],
            recovery_coefficients: vec![1.0; 5],
            offset: vec![0.0; 4],
        };
        reduction.validate().unwrap();
        assert_eq!(
            affine_transpose_index_peak_bytes(&reduction).unwrap(),
            5 * 16 + 3 * 16
        );
    }
}
