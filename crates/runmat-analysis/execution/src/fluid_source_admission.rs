//! Shared streaming source authentication and pre-decode admission for Fluid.
use crate::{AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects};
use runmat_analysis_fea::{
    execution::AnalysisNumericalObjectKind,
    planning::resource::{FeaResourceDemand, FeaResourcePlan},
};
use runmat_execution_artifact::object::{validate_inventory, ObjectInventoryLimits};

pub(crate) use crate::portable_source_admission::authenticate_source_objects as authenticate_fluid_source_objects;

/// Admit exact original wire counts before decoding workload or mesh.
/// Semantic identities and numerical allocation must still be rederived after decoding.
pub fn require_fluid_source_decode_budget(
    inputs: &[PreparedAnalysisObjects],
    plan: &FeaResourcePlan,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<()> {
    require_original_fluid_decode_budget(
        inputs,
        plan,
        limits,
        AnalysisNumericalObjectKind::FluidSteadyWorkload,
    )
}
pub fn require_fluid_transient_source_decode_budget(
    inputs: &[PreparedAnalysisObjects],
    plan: &FeaResourcePlan,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<()> {
    require_original_fluid_decode_budget(
        inputs,
        plan,
        limits,
        AnalysisNumericalObjectKind::FluidTransientBlockWorkload,
    )
}
fn require_original_fluid_decode_budget(
    inputs: &[PreparedAnalysisObjects],
    plan: &FeaResourcePlan,
    limits: ObjectInventoryLimits,
    kind: AnalysisNumericalObjectKind,
) -> AnalysisBridgeResult<()> {
    plan.validate_identity()?;
    use runmat_analysis_fea::planning::resource::FeaResourcePhase;
    let (workload, topology_digest, actual_decode_resident_bytes, source_inputs) =
        match (&plan.input.demand, kind) {
            (
                FeaResourceDemand::FluidSteadySolve(d),
                AnalysisNumericalObjectKind::FluidSteadyWorkload,
            ) if plan.input.phase == FeaResourcePhase::FluidSteadySolve(d.projection.workload) => (
                d.projection.workload,
                d.projection.topology_digest,
                d.projection.actual_decode_resident_bytes,
                &d.inputs,
            ),
            (
                FeaResourceDemand::FluidTransientBlock(d),
                AnalysisNumericalObjectKind::FluidTransientBlockWorkload,
            ) if plan.input.phase
                == FeaResourcePhase::FluidTransientBlock(d.projection.workload) =>
            {
                (
                    d.projection.workload,
                    d.projection.topology_digest,
                    d.projection.actual_decode_resident_bytes,
                    &d.inputs,
                )
            }
            _ => {
                return Err(AnalysisBridgeError::Identity(
                    "Fluid task resource family or phase differs",
                ))
            }
        };
    let kinds = [
        kind,
        AnalysisNumericalObjectKind::Mesh,
        AnalysisNumericalObjectKind::ResourcePlan,
    ];
    if inputs.len() != kinds.len()
        || kinds.iter().any(|kind| {
            inputs
                .iter()
                .filter(|input| input.manifest.kind == *kind)
                .count()
                != 1
        })
    {
        return Err(AnalysisBridgeError::Identity(
            "Fluid task requires exact original input kinds",
        ));
    }
    let mut encoded = 0_u64;
    for input in inputs {
        validate_inventory(&input.objects, limits)?;
        authenticate_fluid_source_objects(input, input.manifest.kind)?;
        match input.manifest.kind {
            kind if kind == kinds[0] && input.manifest.logical_identity != workload.digest() => {
                return Err(AnalysisBridgeError::Identity(
                    "Fluid workload closure differs from admitted source",
                ));
            }
            AnalysisNumericalObjectKind::Mesh
                if input.manifest.logical_identity != topology_digest =>
            {
                return Err(AnalysisBridgeError::Identity(
                    "Fluid mesh closure differs from admitted source",
                ));
            }
            _ => {}
        }
        if input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan {
            if crate::decode_admitted_resource_plan(input, limits)? != *plan {
                return Err(AnalysisBridgeError::Identity(
                    "Fluid immutable resource input differs from admitted plan",
                ));
            }
        } else {
            encoded = encoded
                .checked_add(input.manifest.encoded_length)
                .ok_or_else(|| {
                    AnalysisBridgeError::Invalid("Fluid source length overflow".into())
                })?;
        }
    }
    if encoded != source_inputs.encoded_payload_bytes.0
        || encoded > source_inputs.decoded_resident_bytes_admission_budget.0
    {
        return Err(AnalysisBridgeError::Identity(
            "Fluid original source exceeds admitted decode budget",
        ));
    }
    // Raw authenticated wire count bounds the join itself. The shared wire profiler
    // must then rederive serde allocation authority before any typed source decode;
    // a self-hashed carried budget is not evidence of its actual expansion cost.
    let facts = crate::prepared_input_resource_facts_without_resource_plan(inputs)?;
    if &facts != source_inputs
        || facts.decoded_resident_bytes_admission_budget < actual_decode_resident_bytes
    {
        return Err(AnalysisBridgeError::Identity(
            "Fluid actual source decode profile differs from admitted budget",
        ));
    }
    Ok(())
}
