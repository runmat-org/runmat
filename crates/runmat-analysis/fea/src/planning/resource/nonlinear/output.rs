//! Numeric history uses structural codec profiling, not the worst-density map
//! multiplier applied to already over-reserved typed heap capacity.
use super::ResourcePlanningError;
pub(super) fn projection(
    dofs: usize,
    elements: usize,
    accepted_cap: usize,
    point_history_bytes: u64,
) -> Result<(u64, u64), ResourcePlanningError> {
    let overflow = || ResourcePlanningError::Overflow("nonlinear content profile");
    // Raw event source: two DOF histories, stress/total-strain/plastic tensors, alpha/mises,
    // five scalar histories. Twelve additional nodes per step cover nested arrays,
    // including empty contact vectors. Initial and accepted carrier arrays and fixed scalar inventory
    // are included. A container node costs no more than a profiled scalar node;
    // fixed map keys/enum spellings are reserved separately below.
    let per_step = dofs
        .checked_mul(2)
        .and_then(|n| elements.checked_mul(20).and_then(|e| n.checked_add(e)))
        .and_then(|n| n.checked_add(17))
        .ok_or_else(overflow)?;
    let count = per_step
        .checked_mul(accepted_cap)
        .and_then(|n| dofs.checked_mul(2).and_then(|d| n.checked_add(d)))
        .and_then(|n| n.checked_add(elements))
        .and_then(|n| n.checked_add(128))
        .ok_or_else(overflow)?;
    let profile = super::super::numeric_profile::numeric_inventory_profile(
        u64::try_from(count).map_err(|_| overflow())?,
    )?;
    // Point histories contain typed identity maps rather than pure numeric arrays.
    // Keep their conservative wire/materialization allowance, plus fixed source
    // keys and solver metadata. This does not profile diagnostics/public UI fields.
    let metadata_wire = point_history_bytes
        .checked_mul(4)
        .and_then(|n| n.checked_add(65_536))
        .ok_or_else(overflow)?;
    let metadata_resident =
        runmat_canonical_codec::projected_canonical_materialization_resident_bytes_estimate(
            metadata_wire,
            1,
        )
        .ok_or_else(overflow)?;
    Ok((
        profile
            .encoded_bytes_upper_bound
            .checked_add(metadata_wire)
            .ok_or_else(overflow)?,
        profile
            .materialization_resident_bytes_estimate
            .checked_add(metadata_resident)
            .ok_or_else(overflow)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assembly::nonlinear_solid::resources::NonlinearSolidStorageDemand,
        solve::nonlinear::{
            solid::continuum_qualification::fixture, solve_nonlinear_system, NonlinearSolveOptions,
        },
        topology::{CanonicalSolidIntegrationPlan, CanonicalSolidIntegrationRule},
        ComputeBackend,
    };
    use runmat_analysis_core::FormulationRequirement;
    use runmat_canonical_codec::{profile_serialized, CanonicalLimits};

    #[test]
    fn output_projection_covers_actual_complete_driver_content_for_both_orders() {
        for quadratic in [false, true] {
            let (model, mesh, summary) = fixture::prepare(1, quadratic);
            let options = NonlinearSolveOptions {
                formulation_requirement: FormulationRequirement::Experimental,
                solver_mesh: Some(mesh.clone()),
                increment_count: 4,
                max_accepted_increments: 8,
                tolerance: 1e-9,
                linear_tolerance: 1e-12,
                ..NonlinearSolveOptions::default()
            };
            let result =
                solve_nonlinear_system(&model, &summary, &options, ComputeBackend::Cpu).unwrap();
            let plan = CanonicalSolidIntegrationPlan::derive(
                &mesh,
                CanonicalSolidIntegrationRule::Keast15,
            )
            .unwrap();
            let dofs = summary.dof_count;
            let storage = NonlinearSolidStorageDemand::derive(
                &plan,
                summary.structural_property_map_identity.as_ref().unwrap(),
                dofs,
                dofs * dofs,
                8,
                true,
            )
            .unwrap();
            let (wire, resident) =
                projection(dofs, plan.elements().len(), 8, storage.point_history_bytes).unwrap();
            // The full result additionally includes diagnostics excluded from the
            // numerical content digest; testing it is a stronger bounded witness.
            let actual = profile_serialized(
                &result,
                CanonicalLimits::new(1024 * 1024 * 1024, 100_000_000, 65_536, 32),
            )
            .unwrap();
            assert!(wire >= actual.encoded_bytes_upper_bound);
            assert!(resident >= actual.materialization_resident_bytes_estimate);
            let larger =
                projection(dofs, plan.elements().len(), 16, storage.point_history_bytes).unwrap();
            assert!(larger.0 > wire && larger.1 > resident);
        }
    }
}
