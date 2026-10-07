//! Original native solve closure for adaptive decisions; no solver or scheduler lives here.
use crate::{
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload, PreparedAnalysisObjects,
};
use runmat_analysis_core::{AccuracyPlan, ObservationId};
use runmat_analysis_fea::adaptation::{
    build_structural_adaptation_iteration_from_solved_results, StructuralAdaptationSolvedIteration,
    StructuralAdaptationSolvedIterationInput, StructuralAdaptationSolvedSourceInput,
    StructuralAdaptationSourceLimits, StructuralRecoveryEstimatorOptions,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind as Kind, AnalysisWorkKind, NumericalReceiptIdentity,
    SolverResultIdentity, StructuralLinearNumericalReceiptArtifact, StructuralLinearWorkload,
    StructuralLinearWorkloadResult,
};
use runmat_execution::Digest;
use runmat_meshing_core::{
    CanonicalMeshingContract, SolverMeshAdaptationLineage, SolverMeshArtifact,
    SolverMeshTransferMap,
};
use serde::{Deserialize, Serialize};

mod source;

/// All immutable original objects are required; an imported summary is insufficient.
#[derive(Clone, Copy)]
pub struct StructuralAdaptationNativeSource<'a> {
    pub host: &'a AnalysisHostWorkload,
    pub invocation: &'a PreparedAnalysisObjects,
    pub mesh: &'a PreparedAnalysisObjects,
    pub result: &'a PreparedAnalysisObjects,
    pub receipt: &'a PreparedAnalysisObjects,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralAdaptationNativeIteration {
    pub schema_version: u16,
    pub source_result: SolverResultIdentity,
    pub target_result: SolverResultIdentity,
    pub source_receipt: NumericalReceiptIdentity,
    pub target_receipt: NumericalReceiptIdentity,
    pub solved: StructuralAdaptationSolvedIteration,
}

#[derive(Clone, Copy)]
pub struct StructuralAdaptationNativeIterationInput<'a> {
    pub source: StructuralAdaptationNativeSource<'a>,
    pub target: StructuralAdaptationNativeSource<'a>,
    pub transfer_map: &'a SolverMeshTransferMap,
    pub lineage: &'a SolverMeshAdaptationLineage,
    pub observation_id: &'a ObservationId,
    pub accuracy: &'a AccuracyPlan,
    pub source_limits: StructuralAdaptationSourceLimits,
    /// Aggregate shared decoder inventory for both original source closures, before any join.
    pub maximum_source_decode_bytes: u64,
    pub estimator_options: StructuralRecoveryEstimatorOptions,
    pub previous: Option<&'a StructuralAdaptationNativeIteration>,
}

pub fn evaluate_original_structural_adaptation(
    input: StructuralAdaptationNativeIterationInput<'_>,
) -> AnalysisBridgeResult<StructuralAdaptationNativeIteration> {
    if runmat_analysis_fea::is_fea_cancelled() {
        return Err(AnalysisBridgeError::Invalid(
            "adaptive source decoding cancelled".into(),
        ));
    }
    // Authenticate chunk descriptors before the streaming structural profiler inspects them.
    for source in [input.source, input.target] {
        for (kind, objects) in [
            (Kind::SolverInvocation, source.invocation),
            (Kind::Mesh, source.mesh),
            (Kind::SolverResult, source.result),
        ] {
            crate::portable_source_admission::authenticate_source_objects(objects, kind)?;
        }
        crate::accuracy_resource::authenticate_accuracy_objects(source.receipt)?;
    }
    let inputs = [input.source, input.target].into_iter().flat_map(|source| {
        [
            source.invocation,
            source.mesh,
            source.result,
            source.receipt,
        ]
    });
    let facts = crate::object::prepared_input_resource_facts_from_refs(inputs)?;
    if facts.decoded_resident_bytes_admission_budget.0 > input.maximum_source_decode_bytes {
        return Err(AnalysisBridgeError::Invalid(
            "original adaptive source decoder inventory exceeds its ceiling".into(),
        ));
    }
    let source = source::decode(input.source)?;
    let target = source::decode(input.target)?;
    if input.source.host.workload.study != input.target.host.workload.study
        || input.source.host.workload.step != input.target.host.workload.step
        || input.source.host.artifact_access != input.target.host.artifact_access
    {
        return Err(AnalysisBridgeError::Identity(
            "adaptive solves have different authored studies, steps or artifact access",
        ));
    }
    if let Some(previous) = input.previous {
        if previous.schema_version != 1
            || previous.target_result != source.identity
            || previous.target_receipt != source.receipt.identity()
        {
            return Err(AnalysisBridgeError::Identity(
                "adaptive source differs from the preceding original native target",
            ));
        }
    }
    let source_model = source.workload.solver_model()?;
    let target_model = target.workload.solver_model()?;
    let original = |model, mesh, run| StructuralAdaptationSolvedSourceInput {
        model,
        mesh,
        run,
        observation_id: input.observation_id,
        limits: input.source_limits,
    };
    let solved = build_structural_adaptation_iteration_from_solved_results(
        StructuralAdaptationSolvedIterationInput {
            source: original(&source_model, &source.mesh, &source.result.run),
            target: original(&target_model, &target.mesh, &target.result.run),
            transfer_map: input.transfer_map,
            lineage: input.lineage,
            estimator_options: input.estimator_options,
            previous: input.previous.map(|p| &p.solved),
        },
        input.accuracy,
    )
    .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
    Ok(StructuralAdaptationNativeIteration {
        schema_version: 1,
        source_result: source.identity,
        target_result: target.identity,
        source_receipt: source.receipt.identity(),
        target_receipt: target.receipt.identity(),
        solved,
    })
}

impl StructuralAdaptationNativeIteration {
    /// Requires original source objects; decoding alone does not validate provenance.
    pub fn validate_against(
        &self,
        input: StructuralAdaptationNativeIterationInput<'_>,
    ) -> AnalysisBridgeResult<()> {
        if *self != evaluate_original_structural_adaptation(input)? {
            return Err(AnalysisBridgeError::Identity(
                "adaptive native iteration differs from its original solve closures",
            ));
        }
        Ok(())
    }
}
