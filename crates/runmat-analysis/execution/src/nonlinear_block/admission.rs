//! Stream-authenticate source closure and rederive decode cost before serde joins.
use super::*;
use runmat_analysis_fea::planning::resource::FeaResourceDemand;
use runmat_execution_artifact::object::{validate_inventory, ObjectInventoryLimits};

pub fn require_nonlinear_solid_source_decode_budget(
    inputs: &[PreparedAnalysisObjects],
    plan: &FeaResourcePlan,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<()> {
    plan.validate_identity()?;
    let FeaResourceDemand::StructuralNonlinearPipeline(demand) = &plan.input.demand else {
        return Err(invalid(
            "nonlinear block requires original nonlinear resource demand",
        ));
    };
    let block = demand
        .block_transport
        .as_ref()
        .ok_or_else(|| invalid("nonlinear block transport authority missing"))?;
    let required = [
        Kind::NonlinearSolidWorkload,
        Kind::Mesh,
        Kind::NonlinearSolidBlockRequest,
        Kind::ResourcePlan,
    ];
    let previous_count = inputs
        .iter()
        .filter(|input| input.manifest.kind == Kind::NonlinearSolidBlockArtifact)
        .count();
    if previous_count > 1
        || inputs.len() != required.len() + previous_count
        || required.iter().any(|kind| {
            inputs
                .iter()
                .filter(|input| input.manifest.kind == *kind)
                .count()
                != 1
        })
    {
        return Err(invalid("nonlinear block immutable input kinds differ"));
    }
    for input in inputs {
        validate_inventory(&input.objects, limits)?;
        crate::portable_source_admission::authenticate_source_objects(input, input.manifest.kind)?;
        if input.manifest.kind == Kind::ResourcePlan
            && decode_admitted_resource_plan(input, limits)? != *plan
        {
            return Err(invalid("nonlinear block carried plan differs"));
        }
        if input.manifest.kind == Kind::NonlinearSolidBlockRequest
            && input.manifest.logical_identity != block.request
        {
            return Err(invalid("nonlinear block request differs from admission"));
        }
        if input.manifest.kind == Kind::Mesh
            && input.manifest.logical_identity != demand.operator.topology_digest
        {
            return Err(invalid("nonlinear block mesh differs from admission"));
        }
    }
    let facts = prepared_input_resource_facts_without_resource_plan(inputs)?;
    if facts != block.inputs {
        return Err(invalid(
            "nonlinear block decoded source exceeds original profile",
        ));
    }
    Ok(())
}
