//! Authenticate original transport and raw decode reservation before typed physics inputs.
use super::*;
use runmat_analysis_fea::planning::resource::FeaResourceDemand;
use runmat_execution_artifact::object::ObjectInventoryLimits;

pub fn require_nonlinear_recovery_decode_budget(
    inputs: &[PreparedAnalysisObjects],
    plan: &FeaResourcePlan,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<()> {
    plan.validate_identity()?;
    let FeaResourceDemand::NonlinearFieldRecovery(demand) = &plan.input.demand else {
        return Err(invalid(
            "original nonlinear recovery resource family differs",
        ));
    };
    let kinds = [
        Kind::FieldRecoveryRequest,
        Kind::NonlinearSolidWorkload,
        Kind::Mesh,
        Kind::NonlinearSolidBlockArtifact,
        Kind::ResourcePlan,
    ];
    if inputs.len() != kinds.len()
        || kinds
            .iter()
            .any(|kind| inputs.iter().filter(|i| i.manifest.kind == *kind).count() != 1)
    {
        return Err(invalid("nonlinear recovery admitted input kinds differ"));
    }
    for input in inputs {
        runmat_execution_artifact::object::validate_inventory(&input.objects, limits)?;
        crate::portable_source_admission::authenticate_source_objects(input, input.manifest.kind)?;
        if input.manifest.kind == Kind::ResourcePlan
            && decode_admitted_resource_plan(input, limits)? != *plan
        {
            return Err(invalid("original recovery carried admission differs"));
        }
        if input.manifest.kind == Kind::FieldRecoveryRequest
            && input.manifest.logical_identity != demand.request.digest()
        {
            return Err(invalid("original recovery request identity differs"));
        }
    }
    if prepared_input_resource_facts_without_resource_plan(inputs)? != demand.inputs {
        return Err(invalid(
            "nonlinear recovery actual source decode profile differs",
        ));
    }
    Ok(())
}
