use std::collections::BTreeSet;

use runmat_analysis_core::ProcedureStep;
use runmat_analysis_fea::execution::{
    AccuracyEvaluationRequest, AnalysisNumericalObjectKind, AnalysisWorkKind, SequenceIdentity,
    StepIdentity, StructuralLinearNumericalReceiptArtifact,
};
use runmat_analysis_fea::results::accuracy::{
    evaluate_accuracy, publication_gate, AccuracyArtifact,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ObjectNamespace, ProgramExecutionResponse};

use crate::{
    import_analysis_objects, import_analysis_output_set, prepare_analysis_objects,
    prepare_analysis_program, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisHostWorkload, AnalysisWorkloadInput,
    AnalysisWorkloadRequest, ImportedObservations, PreparedAnalysisObjects,
    PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};

pub struct StructuralAccuracyPreparation {
    pub solve_host: AnalysisHostWorkload,
    pub source_numerical_receipt: PreparedAnalysisObjects,
    pub source_observations: Option<ImportedObservations>,
    pub sequence: SequenceIdentity,
    pub step: ProcedureStep,
    pub formulation_requirement: runmat_analysis_core::FormulationRequirement,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedStructuralAccuracy {
    pub request: AccuracyEvaluationRequest,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedAccuracy {
    pub artifact: AccuracyArtifact,
    pub objects: PreparedAnalysisObjects,
}

pub fn prepare_structural_accuracy(
    preparation: StructuralAccuracyPreparation,
) -> AnalysisBridgeResult<PreparedStructuralAccuracy> {
    preparation.solve_host.validate()?;
    if preparation.solve_host.workload.kind != AnalysisWorkKind::Solve
        || preparation.source_numerical_receipt.manifest.kind
            != AnalysisNumericalObjectKind::NumericalReceipt
        || preparation.chunk_bytes == 0
    {
        return Err(AnalysisBridgeError::Invalid(
            "accuracy evaluation requires one completed solve and bounded chunks".into(),
        ));
    }
    let producer = StepIdentity::from_solver_contract(preparation.sequence, &preparation.step)?;
    if producer != preparation.solve_host.workload.step {
        return Err(AnalysisBridgeError::Identity(
            "accuracy procedure step differs from its solver workload",
        ));
    }
    let numerical_receipt = StructuralLinearNumericalReceiptArtifact::from_canonical_bytes(
        &preparation.source_numerical_receipt.payload(),
    )?;
    if preparation
        .source_numerical_receipt
        .manifest
        .logical_identity
        != numerical_receipt.identity().digest()
        || numerical_receipt.source_workload != preparation.solve_host.workload.identity
    {
        return Err(AnalysisBridgeError::Identity(
            "accuracy numerical receipt differs from its canonical solve",
        ));
    }
    let source_result = numerical_receipt.source_result;
    let observations = preparation
        .source_observations
        .as_ref()
        .map(|observations| {
            if observations.output.kind != AnalysisNumericalObjectKind::Observation {
                return Err(AnalysisBridgeError::Invalid(
                    "accuracy observation input has the wrong numerical kind".into(),
                ));
            }
            observations
                .manifest
                .validate()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
            if observations.manifest.identity.digest() != observations.output.logical_identity
                || observations.manifest.producer != producer
                || observations.manifest.source_result != source_result
            {
                return Err(AnalysisBridgeError::Identity(
                    "accuracy observations differ from their producing solve",
                ));
            }
            Ok(observations.manifest.clone())
        })
        .transpose()?;
    let request = AccuracyEvaluationRequest::new(
        producer,
        source_result,
        observations.as_ref(),
        preparation.formulation_requirement,
        preparation.step.accuracy.clone(),
    )?;
    let request_bytes = request.canonical_bytes()?;
    let request_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::AccuracyEvaluationRequest,
        request.identity().digest(),
        &request_bytes,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let mut inputs = vec![
        (request_objects, ValueRefKind::DriverObject),
        (
            preparation.source_numerical_receipt,
            ValueRefKind::ResultObject,
        ),
    ];
    if let Some(observations) = preparation.source_observations {
        let bytes = observations
            .manifest
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        inputs.push((
            prepare_analysis_objects(
                AnalysisNumericalObjectKind::ObservationCatalog,
                observations.manifest.identity.digest(),
                &bytes,
                preparation.chunk_bytes,
                ObjectNamespace::InputValue,
                preparation.inventory_limits,
            )?,
            ValueRefKind::DriverObject,
        ));
    }
    let workload_inputs = inputs
        .iter()
        .map(|(input, reference_kind)| AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: *reference_kind,
        })
        .collect();
    let inputs = inputs
        .into_iter()
        .map(|(input, _)| input)
        .collect::<Vec<_>>();
    let workload = AnalysisWorkloadRequest::new(
        preparation.solve_host.workload.study,
        preparation.solve_host.workload.step,
        preparation.solve_host.workload.execution_plan,
        AnalysisWorkKind::EvaluateAccuracy,
        workload_inputs,
        vec![AnalysisNumericalObjectKind::AccuracyReport],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )?;
    let host = AnalysisHostWorkload::new(workload, preparation.solve_host.artifact_access)?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedStructuralAccuracy {
        request,
        host,
        program,
        inputs,
    })
}

pub fn import_structural_accuracy(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralAccuracy,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedAccuracy> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [output] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "accuracy evaluation must return exactly one report root".into(),
        ));
    };
    let imported = import_analysis_objects(
        source,
        &output.root,
        AnalysisNumericalObjectKind::AccuracyReport,
        output.logical_identity,
        output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let mut expected =
        imported.unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    expected.push(output_set.root);
    if output_set.result_objects != expected {
        return Err(AnalysisBridgeError::Identity(
            "accuracy result inventory differs from its canonical closure",
        ));
    }
    let artifact = AccuracyArtifact::from_canonical_bytes(&imported.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let expected_evaluation = evaluate_accuracy(&prepared.request.plan, artifact.evidence())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let expected_gate = publication_gate(
        &prepared.request.plan,
        prepared.request.formulation_requirement,
        &expected_evaluation,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if artifact.identity().digest() != output.logical_identity
        || artifact.evaluation_request != prepared.request.identity()
        || artifact.producer != prepared.request.producer
        || artifact.source_result != prepared.request.source_result
        || artifact.source_observations != prepared.request.source_observations
        || artifact.formulation_requirement != prepared.request.formulation_requirement
        || artifact.evaluation() != &expected_evaluation
        || artifact.gate() != &expected_gate
    {
        return Err(AnalysisBridgeError::Identity(
            "accuracy artifact differs from its admitted plan, evidence, or sources",
        ));
    }
    Ok(ImportedAccuracy {
        artifact,
        objects: imported,
    })
}
