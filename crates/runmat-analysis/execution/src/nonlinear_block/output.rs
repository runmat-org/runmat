//! Shared codec materialization supplements actual object transport staging.
use super::*;
use runmat_analysis_fea::planning::resource::PreparedOutputResourceFacts;

pub(super) fn project(
    kind: Kind,
    bytes: u64,
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<PreparedOutputResourceFacts> {
    let mut profiler = AnalysisTransportProfiler::new(policy.chunk_bytes())?;
    profiler.push(bytes)?;
    let profile = profiler.finish(kind)?;
    crate::thermal_execution::require_inventory(&profile, policy.inventory_limits())?;
    let mut output = projected_transport_output_resource_facts(&profile)?;
    let encoding = if kind == Kind::NonlinearSolidBlockArtifact {
        runmat_analysis_fea::planning::resource::nonlinear_solid_block_output_construction_bytes(
            bytes,
        )
        .ok()
    } else {
        runmat_canonical_codec::projected_canonical_encoding_resident_bytes_estimate(bytes, 1)
    };
    let materialization = encoding
        .and_then(|n| n.checked_add(output.transport_closure_bytes_upper_bound.0))
        .ok_or_else(|| invalid("nonlinear output construction overflow"))?;
    output.construction_resident_bytes_admission_budget.0 = output
        .construction_resident_bytes_admission_budget
        .0
        .max(materialization);
    Ok(output)
}
