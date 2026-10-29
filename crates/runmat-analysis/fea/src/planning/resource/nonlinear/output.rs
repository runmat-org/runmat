//! Numeric history uses structural codec profiling, not the worst-density map
//! multiplier applied to already over-reserved typed heap capacity.
use super::ResourcePlanningError;
use crate::planning::resource::{ByteCount, Count, NonlinearHistoryLimits};

pub(super) struct NonlinearOutputProjection {
    /// Wire bound of the complete result, including point-history metadata.
    pub wire: u64,
    /// Construction residency of the canonical content identity.
    pub resident: u64,
    pub history_limits: NonlinearHistoryLimits,
}

pub(super) fn projection(
    dofs: usize,
    elements: usize,
    accepted_cap: usize,
    point_history_bytes: u64,
) -> Result<NonlinearOutputProjection, ResourcePlanningError> {
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
    // Point histories carry typed identity maps rather than pure numeric arrays. Their wire
    // allowance is four in-memory copies plus fixed keys; decoding them materializes the same
    // typed records (`point_history_bytes` measures their in-memory size), so construction holds
    // the wire buffer and one typed copy per wire copy, not a generic per-byte value tree.
    let metadata_wire = point_history_bytes
        .checked_mul(4)
        .and_then(|n| n.checked_add(65_536))
        .ok_or_else(overflow)?;
    let metadata_resident = metadata_wire.checked_mul(2).ok_or_else(overflow)?;
    let wire = profile
        .encoded_bytes_upper_bound
        .checked_add(metadata_wire)
        .ok_or_else(overflow)?;
    let resident = profile
        .materialization_resident_bytes_estimate
        .checked_add(metadata_resident)
        .ok_or_else(overflow)?;
    // Every point-history metadata item occupies at least one wire byte.
    let collection_items = u64::try_from(count)
        .ok()
        .and_then(|n| n.checked_add(metadata_wire))
        .ok_or_else(overflow)?;
    Ok(NonlinearOutputProjection {
        wire,
        resident,
        history_limits: NonlinearHistoryLimits {
            encoded_bytes: ByteCount(wire),
            collection_items: Count(collection_items),
        },
    })
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
    use runmat_canonical_codec::profile_serialized;

    #[test]
    fn output_projection_covers_actual_complete_driver_content_for_both_orders() {
        for quadratic in [false, true] {
            let (model, mesh, summary) = fixture::prepare(1, quadratic);
            let options = NonlinearSolveOptions {
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
            let projected =
                projection(dofs, plan.elements().len(), 8, storage.point_history_bytes).unwrap();
            let (wire, resident) = (projected.wire, projected.resident);
            // The plan's history limits admit the actual event source and its digest.
            crate::solve::nonlinear::nonlinear_solver_content_digest(
                &result,
                projected.history_limits,
            )
            .unwrap();
            // The full result additionally includes diagnostics excluded from the
            // numerical content digest; testing it is a stronger bounded witness.
            let actual =
                profile_serialized(&result, projected.history_limits.canonical().unwrap()).unwrap();
            assert!(wire >= actual.encoded_bytes_upper_bound);
            // Results are encoded directly into reserved output; no value tree is built.
            let encoding = runmat_canonical_codec::profile_canonical_encoding(
                &result,
                projected.history_limits.canonical().unwrap(),
            )
            .unwrap();
            assert!(
                resident >= encoding.resident_workspace_bytes_estimate,
                "{resident} < {}",
                encoding.resident_workspace_bytes_estimate
            );
            let larger =
                projection(dofs, plan.elements().len(), 16, storage.point_history_bytes).unwrap();
            assert!(larger.wire > wire && larger.resident > resident);
            assert!(
                larger.history_limits.collection_items.0
                    > projected.history_limits.collection_items.0
            );
        }
    }
}
