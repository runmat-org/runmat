use super::*;

pub(in crate::analysis) struct PreparedStudyCompletion {
    pub(in crate::analysis) thermal_prefix: Option<Box<ThermalPrefixCompletion>>,
    pub(in crate::analysis::structural_study) context: OperationContext,
    pub(in crate::analysis::structural_study) study_id: String,
    pub(in crate::analysis::structural_study) study_fingerprint: String,
    pub(in crate::analysis::structural_study) authored_study_digest: runmat_execution::Digest,
    pub(in crate::analysis::structural_study) model: AnalysisModel,
    pub(in crate::analysis::structural_study) steps: Vec<PreparedStudyStep>,
    pub(in crate::analysis::structural_study) operation_sequence: Vec<String>,
    pub(in crate::analysis::structural_study) solver_mesh_artifact_path: Option<String>,
    pub(in crate::analysis::structural_study) meshing_evidence_artifact_path: Option<String>,
}

pub(in crate::analysis) fn study_completion_from_prepared(
    prepared: PreparedStudySequence,
) -> PreparedStudyCompletion {
    let authored_study_digest = crate::analysis::analysis_study_content_digest(&prepared.spec);
    PreparedStudyCompletion {
        thermal_prefix: None,
        context: prepared.context,
        study_id: prepared.spec.study_id,
        study_fingerprint: prepared.study_fingerprint,
        authored_study_digest,
        model: prepared.model,
        steps: prepared.steps,
        operation_sequence: prepared.operation_sequence,
        solver_mesh_artifact_path: prepared.solver_mesh_artifact_path,
        meshing_evidence_artifact_path: prepared.meshing_evidence_artifact_path,
    }
}

pub(in crate::analysis) struct CompletedStudyStep {
    pub(in crate::analysis) step_id: ProcedureStepId,
    pub(in crate::analysis) solver_result_identity: SolverResultIdentity,
    pub(in crate::analysis) run: CompletedStudyStepRun,
    pub(in crate::analysis) resolved_solver_options: AnalysisSolverOptions,
}

pub(in crate::analysis) struct ThermalPrefixCompletion {
    pub authored: PreparedStudyStep,
    pub completed: CompletedStudyStep,
    pub states: Vec<CanonicalSemanticState>,
    pub observation_source: crate::analysis::ThermalStudyObjectSource,
    pub thermal: Box<crate::analysis::ThermalStudyCompletion>,
}

pub(in crate::analysis) enum CompletedStudyStepRun {
    Standard(Box<OperationEnvelope<AnalysisRunResult>>),
    OriginalFields {
        run: Box<OperationEnvelope<AnalysisRunResult>>,
        source: Box<runmat_analysis_fea::results::field::OriginalFieldSourcePlan>,
    },
    AcousticFrequency(Box<crate::analysis::study_publication::CompletedAcousticFrequencyStep>),
    StructuralFrequency(Box<CompletedStructuralFrequencyStep>),
    StructuralResponseSpectrum(Box<CompletedStructuralResponseSpectrumStep>),
    StructuralRandomResponse(Box<CompletedStructuralRandomResponseStep>),
}

/// Family-neutral singleton carrier. Numerical evidence remains with its physics owner.
pub(in crate::analysis) fn standalone_completion(
    context: OperationContext,
    study: runmat_analysis_fea::study::AnalysisStudyDefinition<'_>,
    execution: AnalysisExecutionStep,
    mesh: crate::analysis::meshing::ResolvedStudyMesh,
) -> Result<PreparedStudyCompletion, String> {
    let compiled = study.compiled();
    let [step] = compiled.model.sequence.steps.as_slice() else {
        return Err("standalone completion requires one authored procedure step".into());
    };
    if execution.step_id != step.step_id {
        return Err("standalone execution differs from the compiled step".into());
    }
    let operation = crate::analysis::execution_dispatch::execution_operation(&execution, step)
        .ok_or("standalone completion execution options differ from authored procedure")?;
    let definition = study.digest();
    Ok(PreparedStudyCompletion {
        thermal_prefix: None,
        context,
        study_id: compiled.study_id.clone(),
        study_fingerprint: definition.to_string(),
        authored_study_digest: definition,
        model: compiled.model.clone(),
        steps: vec![PreparedStudyStep {
            authored_execution: execution,
            procedure_step: step.clone(),
        }],
        operation_sequence: vec![operation.version().into()],
        solver_mesh_artifact_path: mesh.artifact_path,
        meshing_evidence_artifact_path: mesh.evidence_path,
    })
}

/// Reuse the shared thermal carrier for a genuine singleton study, without a structural suffix.
pub(in crate::analysis) fn standalone_thermal_completion(
    context: OperationContext,
    authored_study_digest: runmat_execution::Digest,
    thermal: crate::analysis::ThermalStudyCompletion,
    source: crate::analysis::ThermalStudyObjectSource,
    run: OperationEnvelope<AnalysisRunResult>,
    options: crate::analysis::AnalysisThermalRunOptions,
    mesh: crate::analysis::meshing::ResolvedStudyMesh,
) -> PreparedStudyCompletion {
    let step = thermal.compiled.model.sequence.steps[0].clone();
    let execution = AnalysisExecutionStep {
        step_id: step.step_id.clone(),
        backend: thermal.workload.backend,
        accelerator_admission: Default::default(),
        solver_options: AnalysisSolverOptions::ThermalDiffusion(options.clone()),
    };
    PreparedStudyCompletion {
        context,
        study_id: thermal.compiled.study_id.clone(),
        study_fingerprint: authored_study_digest.to_string(),
        authored_study_digest,
        model: thermal.compiled.model.clone(),
        steps: Vec::new(),
        operation_sequence: vec![
            crate::analysis::execution_dispatch::AnalysisOperation::Thermal
                .version()
                .into(),
        ],
        solver_mesh_artifact_path: mesh.artifact_path,
        meshing_evidence_artifact_path: mesh.evidence_path,
        thermal_prefix: Some(Box::new(ThermalPrefixCompletion {
            authored: PreparedStudyStep {
                authored_execution: execution,
                procedure_step: step.clone(),
            },
            completed: CompletedStudyStep {
                step_id: step.step_id,
                solver_result_identity: thermal.manifest.identity(),
                run: CompletedStudyStepRun::Standard(Box::new(run)),
                resolved_solver_options: AnalysisSolverOptions::ThermalDiffusion(options),
            },
            states: thermal.states.clone(),
            observation_source: source,
            thermal: Box::new(thermal),
        })),
    }
}
