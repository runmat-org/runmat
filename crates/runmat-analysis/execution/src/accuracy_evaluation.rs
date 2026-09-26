use std::collections::BTreeSet;

use runmat_analysis_core::ProcedureStep;
use runmat_analysis_fea::execution::{
    AccuracyEvaluationRequest, AccuracySourceKind, AccuracySourceReceipt,
    AnalysisNumericalObjectKind, AnalysisWorkKind, SequenceIdentity, StepIdentity,
};
use runmat_analysis_fea::results::accuracy::{
    evaluate_accuracy, publication_gate, AccuracyArtifact,
};
use runmat_analysis_fea::results::observation::ObservationPublicationCatalog;
use runmat_execution::value::ValueRefKind;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ObjectNamespace, ProgramExecutionResponse};

use crate::{
    import_analysis_objects, import_analysis_output_set, prepare_analysis_objects,
    prepare_analysis_program, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisHostWorkload, AnalysisWorkloadInput,
    AnalysisWorkloadRequest, PreparedAnalysisObjects, PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};

pub struct AccuracyPreparation {
    pub solve_host: AnalysisHostWorkload,
    pub source_kind: AccuracySourceKind,
    pub source_numerical_receipt: PreparedAnalysisObjects,
    pub source_observations: Option<ObservationPublicationCatalog>,
    /// Original numerical intent and topology, not copies reconstructed from a result catalog.
    /// Thermal evaluations require exactly ThermalWorkload and Mesh; other families currently
    /// retain their existing receipt contract and do not accept context inputs.
    pub source_context_inputs: Vec<PreparedAnalysisObjects>,
    pub sequence: SequenceIdentity,
    pub step: ProcedureStep,
    pub formulation_requirement: runmat_analysis_core::FormulationRequirement,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedAccuracy {
    pub request: AccuracyEvaluationRequest,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: runmat_analysis_fea::planning::resource::FeaResourcePlan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedAccuracy {
    pub artifact: AccuracyArtifact,
    pub objects: PreparedAnalysisObjects,
}

pub fn prepare_accuracy(
    preparation: AccuracyPreparation,
) -> AnalysisBridgeResult<PreparedAccuracy> {
    preparation.solve_host.validate()?;
    let expected_work_kind = match preparation.source_kind {
        AccuracySourceKind::Thermal => AnalysisWorkKind::SolveThermal,
        AccuracySourceKind::StructuralLinear => AnalysisWorkKind::Solve,
        AccuracySourceKind::StructuralModalResponseSpectrum => {
            AnalysisWorkKind::AssembleStructuralModalResponseSpectrumResult
        }
        AccuracySourceKind::StructuralRandomResponse => {
            AnalysisWorkKind::AssembleStructuralRandomResponseResult
        }
    };
    if preparation.solve_host.workload.kind != expected_work_kind
        || preparation.source_numerical_receipt.manifest.kind
            != AnalysisNumericalObjectKind::NumericalReceipt
        || preparation.chunk_bytes == 0
    {
        return Err(AnalysisBridgeError::Invalid(
            "accuracy evaluation requires a matching completed numerical source and bounded chunks"
                .into(),
        ));
    }
    let producer = StepIdentity::from_solver_contract(preparation.sequence, &preparation.step)?;
    if producer != preparation.solve_host.workload.step {
        return Err(AnalysisBridgeError::Identity(
            "accuracy procedure step differs from its solver workload",
        ));
    }
    let numerical_receipt = AccuracySourceReceipt::from_canonical_bytes(
        preparation.source_kind,
        &preparation.source_numerical_receipt.payload(),
    )?;
    if preparation
        .source_numerical_receipt
        .manifest
        .logical_identity
        != numerical_receipt.identity().digest()
        || !receipt_matches_solve(&numerical_receipt, &preparation.solve_host)
    {
        return Err(AnalysisBridgeError::Identity(
            "accuracy numerical receipt differs from its canonical solve",
        ));
    }
    let source_result = numerical_receipt.source_result();
    if preparation.source_observations.is_some() == preparation.step.observations.is_empty() {
        return Err(AnalysisBridgeError::Invalid(
            "accuracy observation publication presence differs from authored declarations".into(),
        ));
    }
    if let Some(observations) = &preparation.source_observations {
        observations.validate_authored_step(&preparation.step)?;
        observations.validate_source_result(producer, preparation.source_kind, source_result)?;
    }
    let request = AccuracyEvaluationRequest::new(
        producer,
        preparation.source_kind,
        source_result,
        preparation.source_observations.as_ref(),
        preparation.formulation_requirement,
        preparation.step.accuracy.clone(),
    )?;
    crate::validate_accuracy_source_context(
        &request,
        &numerical_receipt,
        preparation.source_observations.as_ref(),
        &preparation.source_context_inputs.iter().collect::<Vec<_>>(),
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
    inputs.extend(
        preparation
            .source_context_inputs
            .into_iter()
            .map(|input| (input, ValueRefKind::DriverObject)),
    );
    if let Some(observations) = preparation.source_observations {
        let bytes = observations
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        inputs.push((
            prepare_analysis_objects(
                AnalysisNumericalObjectKind::ObservationCatalog,
                observations.identity().digest(),
                &bytes,
                preparation.chunk_bytes,
                ObjectNamespace::InputValue,
                preparation.inventory_limits,
            )?,
            ValueRefKind::DriverObject,
        ));
    }
    let mut workload_inputs: Vec<AnalysisWorkloadInput> = inputs
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
    let mut inputs = inputs
        .into_iter()
        .map(|(input, _)| input)
        .collect::<Vec<_>>();
    let resource_plan =
        crate::build_prepared_accuracy_resource_plan(&request, &numerical_receipt, &inputs)?;
    let plan_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        resource_plan.identity.digest(),
        &resource_plan.canonical_bytes()?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    workload_inputs.push(AnalysisWorkloadInput {
        kind: plan_objects.manifest.kind,
        logical_identity: plan_objects.manifest.logical_identity,
        payload_digest: plan_objects.manifest.payload_digest,
        manifest_digest: plan_objects.root.digest,
        manifest_encoded_length: plan_objects.root.encoded_length,
        reference_kind: ValueRefKind::DriverObject,
    });
    inputs.push(plan_objects);
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
    Ok(PreparedAccuracy {
        request,
        host,
        program,
        inputs,
        resource_plan,
    })
}

fn receipt_matches_solve(receipt: &AccuracySourceReceipt, host: &AnalysisHostWorkload) -> bool {
    match receipt {
        AccuracySourceReceipt::Thermal(thermal) => {
            let inputs_of_kind = |kind| {
                host.workload
                    .inputs
                    .iter()
                    .filter(move |input| input.kind == kind)
            };
            let mut workloads = inputs_of_kind(AnalysisNumericalObjectKind::ThermalWorkload);
            let mut meshes = inputs_of_kind(AnalysisNumericalObjectKind::Mesh);
            // Thermal receipts bind numerical intent, not the transport/placement host identity.
            matches!(workloads.next(), Some(input) if input.logical_identity == thermal.source_workload.digest())
                && workloads.next().is_none()
                && matches!(meshes.next(), Some(input) if input.logical_identity == thermal.receipt().topology_digest)
                && meshes.next().is_none()
                && thermal.receipt().producer == host.workload.step
        }
        AccuracySourceReceipt::StructuralLinear(_)
        | AccuracySourceReceipt::StructuralModalResponseSpectrum(_)
        | AccuracySourceReceipt::StructuralRandomResponse(_) => {
            receipt.source_workload() == host.workload.identity
        }
    }
}

pub fn import_accuracy(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedAccuracy,
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
        || artifact.source_kind != prepared.request.source_kind
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
