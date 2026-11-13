use super::*;
impl<'a> PreparedSolidContact<'a> {
    pub(in crate::solve::nonlinear::solid) fn prepare(
        basis: &CanonicalNonlinearSolidBasis,
        reduction: &AffineConstraintReduction,
        input: SolidContactCoupling<'a>,
        policy: SolidNewtonPolicy,
    ) -> Result<Self, SolidNewtonError> {
        if crate::is_fea_cancelled() {
            return Err(SolidNewtonError::Cancelled);
        }
        let backtracks = if policy.line_search {
            policy.max_backtracks
        } else {
            0
        };
        let trials =
            policy
                .max_iterations
                .checked_mul(backtracks.checked_add(1).ok_or_else(|| {
                    SolidNewtonError::Invalid("contact trial count overflow".into())
                })?)
                .and_then(|n| n.checked_add(1))
                .and_then(|n| n.checked_mul(policy.max_cutbacks.checked_add(1)?))
                .and_then(|n| n.checked_add(3))
                .ok_or_else(|| SolidNewtonError::Invalid("contact trial count overflow".into()))?;
        if trials as u64 > input.maximum_geometry_trials_per_advance {
            return Err(SolidNewtonError::Contact(ContactOperatorError::Budget));
        }
        basis
            .validate_contact_layout(input.operator)
            .map_err(body_error)?;
        let support = crate::constraints::AffineSparseSupportPlan::prepare(
            reduction,
            basis.contact_pattern(),
            input.operator.pattern(),
            crate::constraints::AffineSupportBudget {
                maximum_resident_bytes: input.maximum_additional_resident_bytes,
                maximum_scalar_operations: input.maximum_symbolic_scalar_operations,
            },
        )
        .map_err(support_error)?;
        let full = basis
            .contact_pattern()
            .union_statistics(input.operator.pattern())
            .map_err(constraint_error)?;
        let reduced = support.statistics();
        let required = resources::additional_peak_counts(input.operator, full, reduced)?
            .checked_add(support.planning_resident_bytes())
            .ok_or_else(|| {
                SolidNewtonError::Invalid("contact preparation bytes overflow".into())
            })?;
        if required > input.maximum_additional_resident_bytes {
            return Err(SolidNewtonError::Contact(ContactOperatorError::Budget));
        }
        let solve_resources = if reduced.rows > 0 {
            Some(
                crate::planning::resource::GeneralSparseSolveResourceDemand::derive(
                    reduced,
                    input.solve_policy,
                    crate::planning::resource::Count(1),
                    input.solve_admission,
                )?,
            )
        } else {
            None
        };
        // All exact support/solver counts and construction buffers are admitted
        // before either combined or reduced CSR columns are allocated.
        let pattern = basis
            .contact_pattern()
            .union(input.operator.pattern())
            .map_err(constraint_error)?;
        let reduced_pattern = support.materialize().map_err(support_error)?;
        Ok(Self {
            input,
            pattern,
            reduced_pattern,
            solve_resources,
        })
    }
}

fn support_error(error: crate::constraints::AffineSupportError) -> SolidNewtonError {
    use crate::constraints::AffineSupportError;
    match error {
        AffineSupportError::Cancelled => SolidNewtonError::Cancelled,
        AffineSupportError::Budget => SolidNewtonError::Contact(ContactOperatorError::Budget),
        other => constraint_error(other),
    }
}
