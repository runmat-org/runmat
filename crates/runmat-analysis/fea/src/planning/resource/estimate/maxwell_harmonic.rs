use super::host_phase::{build_host_phase_resource_plan, HostPhaseAccounting};
use super::*;
use crate::planning::resource::{MaxwellGmresResourceEnvelope, MaxwellHarmonicResourceDemand};

/// Direct-factorization envelope of the shared sparse layer (count model; the exact symbolic
/// size is checked at solve time). The struct keeps its historical name in the demand contract.
pub(in crate::planning::resource) fn krylov_envelope(
    dimension: usize,
    nonzeros: u64,
) -> Result<MaxwellGmresResourceEnvelope, ResourcePlanningError> {
    if dimension == 0 {
        return Ok(MaxwellGmresResourceEnvelope::default());
    }
    let nonzeros = usize::try_from(nonzeros)
        .ok()
        .filter(|&nonzeros| dimension <= isize::MAX as usize / 1024 && nonzeros <= isize::MAX as usize / 1024)
        .ok_or(ResourcePlanningError::Overflow("direct factor dimension"))?;
    let bytes = runmat_sparse_solvers::estimate::direct_reservation_bytes(dimension, nonzeros, true);
    Ok(MaxwellGmresResourceEnvelope {
        dimension: Count(dimension as u64),
        restart: Count(0),
        maximum_iterations: Count(1),
        resident_bytes: ByteCount(bytes),
        operator_applications: Count(16),
        dot_calls: Count(0),
        axpy_calls: Count(0),
    })
}

pub(super) fn build_maxwell_harmonic_plan(
    input: FeaResourcePlanningInput,
    demand: &MaxwellHarmonicResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let dimension = mul(
        demand.free_edge_count.0,
        2,
        "Maxwell complex block dimension",
    )?;
    let host_dimension = usize::try_from(dimension)
        .map_err(|_| ResourcePlanningError::Overflow("Maxwell host dimension"))?;
    if demand.node_count.0 == 0
        || demand.element_count.0 == 0
        || demand.edge_count_upper_bound.0
            != mul(demand.element_count.0, 6, "Maxwell edge upper bound")?
        || demand.quadrature_sample_count.0
            != mul(demand.element_count.0, 15, "Maxwell sample count")?
        || demand.free_edge_count > demand.edge_count_upper_bound
        || demand.real_block_dimension.0 != dimension
        || demand.block_nonzeros.0 > mul(demand.element_count.0, 144, "Maxwell block support")?
        || (dimension == 0) != (demand.block_nonzeros.0 == 0)
        || demand.krylov != krylov_envelope(host_dimension, demand.block_nonzeros.0)?
        || demand.numerical_reservation_bytes.0 == 0
        || demand.numerical_reservation_bytes < demand.krylov.resident_bytes
        || demand.system_plan_identity == Digest::sha256([])
        || demand.state_profiles_resident_bytes.0 == 0
        || demand.receipt_output.encoded_payload_bytes_upper_bound.0
            != crate::execution::MAXWELL_RECEIPT_MAX_ENCODED_BYTES
        || demand.inputs.decoded_resident_bytes_admission_budget
            < demand.inputs.encoded_payload_bytes
        || [
            &demand.result_output,
            &demand.receipt_output,
            &demand.state_profiles_output,
        ]
        .iter()
        .any(|o| {
            o.construction_resident_bytes_admission_budget < o.transport_closure_bytes_upper_bound
        })
        || !input.worker.accelerators.is_empty()
    {
        return Err(ResourcePlanningError::Invalid(
            "Maxwell exact sparse/GMRES or prepared host facts are inconsistent".into(),
        ));
    }
    let retained = ByteCount(add(
        demand.inputs.transport_closure_bytes.0,
        implicit_resource_plan_input_facts()?
            .transport_closure_bytes
            .0,
        "Maxwell retained input transport",
    )?);
    build_host_phase_resource_plan(
        input,
        HostPhaseAccounting::host_only(
            demand.topology_digest,
            &demand.inputs,
            ByteCount(add(
                demand.numerical_reservation_bytes.0,
                demand.state_profiles_resident_bytes.0,
                "Maxwell profiles resident",
            )?),
            retained,
            &[
                &demand.result_output,
                &demand.receipt_output,
                &demand.state_profiles_output,
            ],
        ),
    )
}
