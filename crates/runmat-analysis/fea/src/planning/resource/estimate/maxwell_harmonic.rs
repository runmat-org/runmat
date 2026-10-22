use super::host_phase::{build_host_phase_resource_plan, HostPhaseAccounting};
use super::*;
use crate::planning::resource::{MaxwellDirectFactorEnvelope, MaxwellHarmonicResourceDemand};

/// Same dimension/restart/counter policy as the general physical Maxwell solve.
/// Representative tolerance affects stopping only, never its allocation bounds.
/// Direct-factor envelope for a complex operator of this dimension and support.
pub(in crate::planning::resource) fn factor_envelope(
    dimension: u64,
    nonzeros: u64,
) -> Result<MaxwellDirectFactorEnvelope, ResourcePlanningError> {
    if dimension == 0 {
        return Ok(MaxwellDirectFactorEnvelope::default());
    }
    let fill = (dimension as f64).powf(4.0 / 3.0) * 16.0;
    if !fill.is_finite() || fill > (u64::MAX / 64) as f64 {
        return Err(ResourcePlanningError::Overflow("Maxwell factor fill"));
    }
    let entries = (fill as u64).max(nonzeros);
    Ok(MaxwellDirectFactorEnvelope {
        dimension: Count(dimension),
        nonzeros: Count(nonzeros),
        factor_entries_upper_bound: Count(entries),
        resident_bytes: ByteCount(mul(entries, 2 * 16, "Maxwell factor bytes")?),
    })
}

pub(super) fn build_maxwell_harmonic_plan(
    input: FeaResourcePlanningInput,
    demand: &MaxwellHarmonicResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let dimension = demand.complex_dimension.0;
    if demand.node_count.0 == 0
        || demand.element_count.0 == 0
        || demand.edge_count_upper_bound.0
            != mul(demand.element_count.0, 6, "Maxwell edge upper bound")?
        || demand.quadrature_sample_count.0
            != mul(demand.element_count.0, 15, "Maxwell sample count")?
        || demand.free_edge_count > demand.edge_count_upper_bound
        || dimension < demand.free_edge_count.0
        || demand.nonzeros.0 > mul(demand.element_count.0, 36, "Maxwell support")?
        || (dimension == 0) != (demand.nonzeros.0 == 0)
        || demand.factorization != factor_envelope(dimension, demand.nonzeros.0)?
        || demand.numerical_reservation_bytes.0 == 0
        || demand.numerical_reservation_bytes < demand.factorization.resident_bytes
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
            "Maxwell exact sparse/factorization or prepared host facts are inconsistent".into(),
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
