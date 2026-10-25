//! Resident storage retained by a selected numerical solver and its proof.

use super::{add, bytes, sum};
use crate::planning::resource::contract::{
    ByteCount, PreconditionerStorage, ResourcePlanningError, SolveResourcePolicy,
    SolverAdmissionResourcePolicy, StructuralOperatorResourceDemand,
};

pub(super) fn solve_storage(
    demand: &StructuralOperatorResourceDemand,
) -> Result<(ByteCount, ByteCount), ResourcePlanningError> {
    match demand.solve {
        SolveResourcePolicy::Iterative {
            ref preconditioner,
            ref admission,
            ..
        } => {
            let value = match preconditioner {
                PreconditionerStorage::None => 0,
                PreconditionerStorage::Diagonal => bytes(
                    demand.sparse.rows.0,
                    demand.solve.scalar_bytes(),
                    "diagonal preconditioner",
                )?,
                PreconditionerStorage::SparseCsr {
                    nonzero_upper_bound,
                    resident_bytes_upper_bound,
                } => {
                    let payload_lower_bound = sum(
                        &[
                            bytes(
                                add(demand.sparse.rows.0, 1, "preconditioner row offsets")?,
                                8,
                                "preconditioner row offsets",
                            )?,
                            bytes(nonzero_upper_bound.0, 8, "preconditioner column indices")?,
                            bytes(
                                nonzero_upper_bound.0,
                                demand.solve.scalar_bytes(),
                                "preconditioner values",
                            )?,
                        ],
                        "sparse preconditioner CSR payload",
                    )?;
                    if nonzero_upper_bound.0 == 0
                        || resident_bytes_upper_bound.0 < payload_lower_bound
                    {
                        return Err(ResourcePlanningError::Invalid(
                            "sparse preconditioner count and resident bound must be nonzero".into(),
                        ));
                    }
                    resident_bytes_upper_bound.0
                }
            };
            match demand.solve.static_solve_path() {
                Some(crate::planning::resource::StaticSolvePath::ElasticRigidityMultigrid) => {
                    // No global factor: the admission carries only the proof's inventory.
                    if admission.dimension.0 == 0
                        || admission.dimension.0 > demand.sparse.rows.0
                        || admission.factor_nonzero_upper_bound.0 != 0
                        || admission.resident_bytes_upper_bound.0 == 0
                        || !matches!(preconditioner, PreconditionerStorage::SparseCsr { .. })
                    {
                        return Err(ResourcePlanningError::Invalid(
                            "elastic rigidity admission must reserve its proof inventory, no factor, and a multigrid preconditioner".into(),
                        ));
                    }
                }
                _ => validate_solver_admission(
                    admission,
                    demand.sparse.rows.0,
                    demand.solve.scalar_bytes(),
                )?,
            }
            Ok((ByteCount(value), admission.resident_bytes_upper_bound))
        }
        SolveResourcePolicy::SaddlePointGmres { ref admission, .. } => {
            validate_solver_admission(admission, demand.sparse.rows.0, 8)?;
            Ok((ByteCount(0), admission.resident_bytes_upper_bound))
        }
        SolveResourcePolicy::DirectFactorization {
            factor_nonzero_upper_bound,
            factor_bytes_upper_bound,
            ..
        } => {
            let payload_lower_bound = sum(
                &[
                    bytes(
                        add(demand.sparse.rows.0, 1, "factor row offsets")?,
                        8,
                        "factor row offsets",
                    )?,
                    bytes(factor_nonzero_upper_bound.0, 8, "factor column indices")?,
                    bytes(
                        factor_nonzero_upper_bound.0,
                        demand.solve.scalar_bytes(),
                        "factor values",
                    )?,
                ],
                "factorization sparse payload",
            )?;
            if factor_nonzero_upper_bound.0 == 0 || factor_bytes_upper_bound.0 < payload_lower_bound
            {
                return Err(ResourcePlanningError::Invalid(
                    "factorization count and resident bound must be nonzero".into(),
                ));
            }
            Ok((ByteCount(0), factor_bytes_upper_bound))
        }
        SolveResourcePolicy::GeneralSparse(_)
        | SolveResourcePolicy::NoSolve { .. }
        | SolveResourcePolicy::RestartedGmres { .. }
        | SolveResourcePolicy::ModalDenseHarmonic { .. }
        | SolveResourcePolicy::TransientBlock { .. }
        | SolveResourcePolicy::SymmetricSubspaceIteration { .. } => {
            Ok((ByteCount(0), ByteCount(0)))
        }
    }
}

fn validate_solver_admission(
    admission: &SolverAdmissionResourcePolicy,
    physical_dimension: u64,
    scalar_bytes: u64,
) -> Result<(), ResourcePlanningError> {
    let factor_payload_lower_bound = sum(
        &[
            bytes(
                admission.factor_nonzero_upper_bound.0,
                scalar_bytes,
                "solver-admission factor values",
            )?,
            bytes(
                admission.factor_nonzero_upper_bound.0,
                8,
                "solver-admission factor indices",
            )?,
            bytes(
                add(admission.dimension.0, 1, "solver-admission rows")?,
                8,
                "solver-admission row offsets",
            )?,
        ],
        "solver-admission factor payload",
    )?;
    if admission.dimension.0 == 0
        || admission.dimension.0 > physical_dimension
        || admission.factor_nonzero_upper_bound.0 < admission.dimension.0
        || admission.resident_bytes_upper_bound.0 < factor_payload_lower_bound
    {
        return Err(ResourcePlanningError::Invalid(
            "solver-admission fill, work, and resident bounds must cover the symbolic factor"
                .into(),
        ));
    }
    Ok(())
}
