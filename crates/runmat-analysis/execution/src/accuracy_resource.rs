//! Pre-execution accuracy sizing reused by task preparation and native admission.
mod source_admission;
mod thermal;
pub use source_admission::validate_admitted_accuracy_source;
pub(crate) use source_admission::validate_derived_accuracy_source;
#[cfg(test)]
mod tests;
use crate::{
    prepared_input_resource_facts_without_resource_plan, projected_output_resource_facts,
    projected_output_set_resource_facts, AnalysisBridgeError, AnalysisBridgeResult,
    PreparedAnalysisObjects,
};
use runmat_analysis_fea::{
    execution::{AccuracyEvaluationRequest, AccuracySourceReceipt},
    planning::resource::{
        build_accuracy_resource_plan_with_source_workspace, FeaResourcePlan,
        PreparedPhaseResourceFacts,
    },
    results::accuracy::accuracy_evaluation_resource_profile,
};

pub fn build_prepared_accuracy_resource_plan(
    request: &AccuracyEvaluationRequest,
    receipt: &AccuracySourceReceipt,
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<FeaResourcePlan> {
    if receipt.source_result() != request.source_result {
        return Err(AnalysisBridgeError::Identity(
            "accuracy resource source differs from request",
        ));
    }
    let evidence = receipt.evidence(request)?;
    let profile = accuracy_evaluation_resource_profile(request, &evidence)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let workspace = if matches!(receipt, AccuracySourceReceipt::FluidSteady(_)) {
        let original =
            inputs
                .iter()
                .filter(|input| {
                    matches!(input.manifest.kind,
            runmat_analysis_fea::execution::AnalysisNumericalObjectKind::FluidSteadyWorkload
            | runmat_analysis_fea::execution::AnalysisNumericalObjectKind::Mesh
            | runmat_analysis_fea::execution::AnalysisNumericalObjectKind::SolverResult)
                })
                .collect::<Vec<_>>();
        let input_facts = prepared_input_resource_facts_without_resource_plan(inputs)?;
        let mut available = request
            .plan
            .resources
            .resident_byte_budget()
            .checked_sub(input_facts.decoded_resident_bytes_admission_budget.0)
            .ok_or_else(|| {
                AnalysisBridgeError::Invalid("fluid accuracy decode exceeds source ceiling".into())
            })?;
        for input in inputs.iter().filter(|i| {
            i.manifest.kind
                == runmat_analysis_fea::execution::AnalysisNumericalObjectKind::ResourcePlan
        }) {
            authenticate_accuracy_objects(input)?;
            let carried = FeaResourcePlan::from_canonical_bytes(&input.payload())?;
            let runmat_analysis_fea::planning::resource::FeaResourceDemand::AccuracyEvaluation(
                demand,
            ) = carried.input.demand
            else {
                return Err(AnalysisBridgeError::Identity(
                    "fluid accuracy carried resource family",
                ));
            };
            available = available.min(
                demand
                    .evaluator_workspace_bytes_upper_bound
                    .0
                    .checked_sub(profile.workspace_bytes_upper_bound)
                    .ok_or_else(|| {
                        AnalysisBridgeError::Invalid(
                            "fluid accuracy evaluator workspace underflows".into(),
                        )
                    })?,
            );
        }
        Some(crate::accuracy_source_context::fluid::resource_workspace(
            request, &original, available,
        )?)
    } else if matches!(receipt, AccuracySourceReceipt::Thermal(_))
        && request.source_observations.is_some()
    {
        Some(thermal::workspace(
            request,
            &evidence,
            inputs,
            PreparedPhaseResourceFacts {
                inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
                output: projected_output_resource_facts(profile.output_encoded_bytes_upper_bound)?,
                output_set: projected_output_set_resource_facts(),
            },
        )?)
    } else {
        None
    };
    Ok(build_accuracy_resource_plan_with_source_workspace(
        request,
        &evidence,
        PreparedPhaseResourceFacts {
            inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
            output: projected_output_resource_facts(profile.output_encoded_bytes_upper_bound)?,
            output_set: projected_output_set_resource_facts(),
        },
        workspace,
    )?)
}

/// Authenticate all prepared source chunks before joining/decode. Receipt outputs
/// have result namespace; original requests/workloads/layouts remain driver inputs.
pub(crate) fn authenticate_accuracy_objects(
    input: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<()> {
    use runmat_analysis_fea::execution::AnalysisNumericalObjectKind as Kind;
    if input.manifest.kind != Kind::NumericalReceipt {
        return crate::portable_source_admission::authenticate_source_objects(
            input,
            input.manifest.kind,
        );
    }
    use runmat_execution::Digest;
    input.manifest.validate()?;
    if input.manifest.chunks.len().checked_add(1) != Some(input.objects.len())
        || input.objects.iter().any(|o| {
            o.descriptor.namespace != runmat_execution_artifact::ObjectNamespace::ResultValue
        })
    {
        return Err(AnalysisBridgeError::Identity(
            "accuracy receipt namespace/inventory differs",
        ));
    }
    let root = &input.objects[0];
    if root.descriptor != input.root
        || root.bytes.len() as u64 != input.root.encoded_length
        || root.bytes != input.manifest.canonical_bytes()?
        || Digest::sha256(&root.bytes) != input.root.digest
    {
        return Err(AnalysisBridgeError::Identity(
            "accuracy receipt immutable root differs",
        ));
    }
    let mut count = 0u64;
    for (object, chunk) in input.objects[1..].iter().zip(&input.manifest.chunks) {
        if object.bytes.len() as u64 != chunk.encoded_length
            || object.descriptor.encoded_length != chunk.encoded_length
            || object.descriptor.digest != chunk.digest
            || Digest::sha256(&object.bytes) != chunk.digest
        {
            return Err(AnalysisBridgeError::Identity(
                "accuracy receipt immutable chunk differs",
            ));
        }
        count = count
            .checked_add(chunk.encoded_length)
            .ok_or_else(|| AnalysisBridgeError::Invalid("accuracy receipt wire overflow".into()))?;
    }
    if count != input.manifest.encoded_length
        || Digest::sha256_parts(input.objects[1..].iter().map(|o| &o.bytes))
            != input.manifest.payload_digest
    {
        return Err(AnalysisBridgeError::Identity(
            "accuracy receipt immutable payload differs",
        ));
    }
    Ok(())
}

/// Actual authenticated wire/decode profile admission before typed source decoding.
pub fn require_accuracy_source_decode_budget(
    inputs: &[PreparedAnalysisObjects],
    plan: &FeaResourcePlan,
    limits: runmat_execution_artifact::object::ObjectInventoryLimits,
) -> AnalysisBridgeResult<()> {
    use runmat_analysis_fea::{
        execution::AnalysisNumericalObjectKind as Kind, planning::resource::FeaResourceDemand,
    };
    plan.validate_identity()?;
    let FeaResourceDemand::AccuracyEvaluation(demand) = &plan.input.demand else {
        return Err(AnalysisBridgeError::Identity(
            "accuracy source resource family differs",
        ));
    };
    for input in inputs {
        runmat_execution_artifact::object::validate_inventory(&input.objects, limits)?;
        authenticate_accuracy_objects(input)?;
        if input.manifest.kind == Kind::ResourcePlan {
            let decoded = FeaResourcePlan::from_canonical_bytes(&input.payload())?;
            if decoded != *plan || input.manifest.logical_identity != plan.identity.digest() {
                return Err(AnalysisBridgeError::Identity(
                    "accuracy carried resource input differs",
                ));
            }
        }
    }
    let facts = prepared_input_resource_facts_without_resource_plan(inputs)?;
    if facts != demand.inputs {
        return Err(AnalysisBridgeError::Identity(
            "accuracy exact source decode profiles differ",
        ));
    }
    Ok(())
}
