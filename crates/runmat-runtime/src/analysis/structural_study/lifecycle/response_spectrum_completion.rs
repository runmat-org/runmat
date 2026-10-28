use super::*;
use runmat_analysis_fea::admission::{admit_formulation, FormulationAdmissionRequest};

pub(super) fn complete_structural_response_spectrum(
    intent: PreparedStructuralStudyIntent,
    study_model: runmat_analysis_core::AnalysisModel,
    persisted_mesh: crate::analysis::meshing::ResolvedStudyMesh,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    authored_workloads: BTreeMap<
        runmat_analysis_core::ProcedureStepId,
        runmat_analysis_fea::execution::StructuralStepWorkload,
    >,
    numerical: crate::analysis::StructuralStudyModalResponseSpectrumNumericalResult,
    completion_identity: AnalysisStudyCompletionIdentity,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    let [modal_step, response_step] = intent.steps.as_slice() else {
        return Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "modal response-spectrum completion requires exactly two authored steps",
        ));
    };
    let modal_run_id = completion_identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Structural,
        intent.authored_study_digest,
        &modal_step.procedure_step.step_id,
    );
    let response_run_id = completion_identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Structural,
        intent.authored_study_digest,
        &response_step.procedure_step.step_id,
    );
    complete_response_spectrum_study(
        super::modal_completion::prepared_completion(intent, study_model, persisted_mesh),
        solver_mesh,
        authored_workloads,
        numerical,
        modal_run_id,
        response_run_id,
        preparation,
    )
}

pub(in crate::analysis) fn complete_response_spectrum_study(
    mut prepared: PreparedStudyCompletion,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    mut authored_workloads: BTreeMap<
        runmat_analysis_core::ProcedureStepId,
        runmat_analysis_fea::execution::StructuralStepWorkload,
    >,
    numerical: crate::analysis::StructuralStudyModalResponseSpectrumNumericalResult,
    modal_run_id: String,
    response_run_id: String,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    let [modal_step, response_step] = prepared.steps.as_slice() else {
        return Err(completion_error(
            &prepared,
            "modal response-spectrum completion requires exactly two authored steps",
        ));
    };
    let modal_step = modal_step.clone();
    let response_step = response_step.clone();
    if numerical.modal_step_id != modal_step.procedure_step.step_id
        || numerical.response_step_id != response_step.procedure_step.step_id
    {
        return Err(completion_error(
            &prepared,
            "modal response-spectrum result step order differs from the authored sequence",
        ));
    }
    let Some(runmat_analysis_fea::execution::StructuralStepWorkload::ModalExtraction {
        workload: modal_workload,
    }) = authored_workloads.remove(&numerical.modal_step_id)
    else {
        return Err(completion_error(
            &prepared,
            "modal response-spectrum workload is missing its modal producer",
        ));
    };
    let Some(runmat_analysis_fea::execution::StructuralStepWorkload::ModalResponseSpectrum {
        template,
    }) = authored_workloads.remove(&numerical.response_step_id)
    else {
        return Err(completion_error(
            &prepared,
            "modal response-spectrum workload is missing its response consumer",
        ));
    };
    if !authored_workloads.is_empty() {
        return Err(completion_error(
            &prepared,
            "modal response-spectrum workload contains an unauthored step",
        ));
    }

    let AnalysisSolverOptions::StructuralModes(mut modal_options) =
        modal_step.authored_execution.solver_options.clone()
    else {
        return Err(completion_error(
            &prepared,
            "modal response-spectrum producer has non-modal solver options",
        ));
    };
    let AnalysisSolverOptions::StructuralResponseSpectrum(mut response_options) =
        response_step.authored_execution.solver_options.clone()
    else {
        return Err(completion_error(
            &prepared,
            "modal response-spectrum consumer has incompatible solver options",
        ));
    };

    let mut modal_workload = *modal_workload;
    modal_workload
        .replace_model(prepared.model.clone())
        .map_err(|error| completion_error(&prepared, error.to_string()))?;
    let recovered_modal =
        runmat_analysis_fea::pipeline::modal::recover_completed_structural_modal_run(
            &modal_workload,
            &solver_mesh,
            &numerical.modal_result,
            &numerical.modal_numerical_receipt,
        )
        .map_err(|error| completion_error(&prepared, error.to_string()))?;
    modal_options.solver_mesh_artifact_path = prepared.solver_mesh_artifact_path.clone();
    let modal_run = analysis_admit_modal_run_with_id(
        recovered_modal,
        modal_workload.backend(),
        modal_options.clone(),
        prepared.context.clone(),
        Some(&solver_mesh),
        Some(&modal_run_id),
    )?;
    let modal_result_identity = numerical.modal_numerical_receipt.source_result();
    let expected_basis_states = super::modal_completion::modal_basis_states_for_completion(
        &prepared,
        &solver_mesh,
        &modal_workload,
        modal_result_identity,
        numerical.modal_result.basis(),
    )?;
    if expected_basis_states.as_slice() != [numerical.basis_state.clone()] {
        return Err(completion_error(
            &prepared,
            "published modal basis differs from the admitted producer result",
        ));
    }

    let mut template = *template;
    template.model = prepared.model.clone();
    let topology_digest =
        runmat_execution::Digest::from_bytes(*solver_mesh.canonical_digest.bytes());
    let transition = runmat_analysis_fea::execution::CanonicalStateTransition::bind_upstream(
        &prepared.model,
        &template.step_id,
        &numerical.basis_state,
        runmat_analysis_core::StateUse::ModalBasis,
        topology_digest,
    )
    .map_err(|error| completion_error(&prepared, error.to_string()))?;
    let response_workload = template
        .bind(transition)
        .map_err(|error| completion_error(&prepared, error.to_string()))?;
    if response_workload != numerical.resolved_response_workload {
        return Err(completion_error(
            &prepared,
            "resolved response-spectrum workload differs from the admitted state transition",
        ));
    }
    numerical
        .response_result
        .validate_for(&response_workload)
        .map_err(|error| completion_error(&prepared, error.to_string()))?;
    let response_result_identity = numerical.response_numerical_receipt.source_result();
    numerical
        .response_numerical_receipt
        .validate_against(response_result_identity, &numerical.response_result)
        .map_err(|error| completion_error(&prepared, error.to_string()))?;

    let declaration = response_step
        .procedure_step
        .state_outputs
        .iter()
        .find(|output| output.kind == runmat_analysis_core::StateKind::StructuralResponseEnvelope)
        .ok_or_else(|| completion_error(&prepared, "response-envelope declaration is absent"))?;
    let sequence = runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
        &prepared.model.sequence,
    )
    .map_err(|error| completion_error(&prepared, error.to_string()))?;
    let producer = runmat_analysis_fea::execution::StepIdentity::from_solver_contract(
        sequence,
        &response_step.procedure_step,
    )
    .map_err(|error| completion_error(&prepared, error.to_string()))?;
    let expected_envelope = runmat_analysis_fea::execution::CanonicalSemanticState::new(
        runmat_analysis_fea::execution::CanonicalSemanticStateInput {
            producer,
            declaration: declaration.clone(),
            topology_digest,
            source_result: response_result_identity,
            payload: runmat_analysis_fea::execution::CanonicalSemanticStatePayload::StructuralResponseEnvelope {
                envelope: Box::new(
                    runmat_analysis_fea::execution::StructuralResponseEnvelope::from_result(
                        &numerical.response_result,
                    )
                    .map_err(|error| completion_error(&prepared, error.to_string()))?,
                ),
            },
        },
    )
    .map_err(|error| completion_error(&prepared, error.to_string()))?;
    if expected_envelope != numerical.envelope_state {
        return Err(completion_error(
            &prepared,
            "published response envelope differs from the admitted response result",
        ));
    }

    let field_recovery = numerical.field_recovery.clone();
    field_recovery
        .validate_for_result(numerical.response_result.numerical())
        .map_err(|error| completion_error(&prepared, error.to_string()))?;
    let constraint_plan =
        runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
            &prepared.model,
            Some(&solver_mesh),
        )
        .map_err(|error| completion_error(&prepared, error.to_string()))?;
    let prepared_response = runmat_analysis_fea::prepare_structural_modal_response_spectrum(
        &response_workload,
        &solver_mesh,
        &numerical.basis_state,
        &constraint_plan,
        &[],
    )
    .map_err(|error| completion_error(&prepared, error.to_string()))?;
    let expected = runmat_analysis_fea::recover_structural_modal_response_spectrum_fields(
        &response_workload,
        &prepared_response,
        numerical.response_result.numerical(),
    )
    .map_err(|error| completion_error(&prepared, error.to_string()))?;
    if field_recovery != expected {
        return Err(completion_error(
            &prepared,
            "response-spectrum field recovery differs from its exact physical inputs",
        ));
    }

    let modal_identity = &modal_run.data.run.identity;
    let mut response_realization = modal_identity.realization().clone();
    response_realization.temporals = vec![
        runmat_analysis_core::TemporalImplementation::StructuralModalResponseSpectrumCombination,
    ];
    let response_identity = admit_formulation(FormulationAdmissionRequest {
        family: response_step.procedure_step.family,
        procedure: response_step.procedure_step.procedure,
        solution_space: response_step.procedure_step.solution_space,
        realization: response_realization,
    })
    .map_err(|error| completion_error(&prepared, error.to_string()))?
    .into_identity();
    response_options.solver_mesh_artifact_path = prepared.solver_mesh_artifact_path.clone();
    // Authored outputs observe the step's recovered statistical fields on the canonical mesh.
    let observations = numerical.response_observations.as_ref();
    if observations.is_some() == response_step.procedure_step.observations.is_empty()
        || observations.map(|catalog| catalog.identity())
            != numerical.response_publication.source_observations
    {
        return Err(completion_error(
            &prepared,
            "response-spectrum observations differ from authored outputs or their publication",
        ));
    }
    if let Some(catalog) = observations {
        catalog
            .validate_source_result(
                producer,
                runmat_analysis_fea::execution::AccuracySourceKind::StructuralModalResponseSpectrum,
                response_result_identity,
            )
            .and_then(|()| {
                catalog.validate_for(
                    &response_step.procedure_step,
                    runmat_analysis_fea::execution::ObservationHistorySourceContext::VolumeMesh {
                        mesh: &solver_mesh,
                    },
                )
            })
            .map_err(|error| completion_error(&prepared, error.to_string()))?;
    }
    let accuracy_request = runmat_analysis_fea::execution::AccuracyEvaluationRequest::new(
        producer,
        runmat_analysis_fea::execution::AccuracySourceKind::StructuralModalResponseSpectrum,
        response_result_identity,
        observations,
        response_step.procedure_step.accuracy.clone(),
    )
    .map_err(|error| completion_error(&prepared, error.to_string()))?;
    let accuracy_source =
        runmat_analysis_fea::execution::AccuracySourceReceipt::StructuralModalResponseSpectrum(
            numerical.response_numerical_receipt.clone(),
        );
    let evidence = accuracy_source
        .evidence(&accuracy_request)
        .map_err(|error| completion_error(&prepared, error.to_string()))?;
    let expected_accuracy = runmat_analysis_fea::results::accuracy::evaluate_accuracy_artifact(
        &accuracy_request,
        evidence,
    )
    .map_err(|error| completion_error(&prepared, error.to_string()))?;
    if numerical.response_accuracy != expected_accuracy {
        return Err(completion_error(
            &prepared,
            "response-spectrum accuracy artifact differs from its exact numerical source",
        ));
    }
    super::accuracy_completion::validate_publication_chain(
        response_workload.identity().digest(),
        &runmat_analysis_fea::planning::resource::FeaResourcePhase::StructuralModalResponseSpectrumAssembly(
            numerical.response_result.assembly().identity(),
        ),
        &numerical.response_accuracy,
        &numerical.response_publication_baseline_usage,
        &numerical.response_resource_usage,
        &numerical.response_publication,
    )
    .map_err(|error| completion_error(&prepared, error))?;
    let accuracy = numerical.response_accuracy;
    let completed_accuracy = super::accuracy_completion::classify_accuracy(&accuracy);
    let persisted = super::statistical_persistence::admit_statistical_persistence(
        super::statistical_persistence::StatisticalPersistenceRequest {
            identity: &response_identity,
            run_id: &response_run_id,
            backend: response_workload.backend(),
            solver_method:
                runmat_analysis_fea::SolverMethod::StructuralModalResponseSpectrumCombination,
            solver_options: &AnalysisSolverOptions::StructuralResponseSpectrum(
                response_options.clone(),
            ),
            solver_mesh: &solver_mesh,
            accuracy: &completed_accuracy,
            fields: runmat_analysis_fea::results::statistical_fields::response_spectrum_fields(
                &field_recovery,
            ),
        },
    )
    .map_err(|error| completion_error(&prepared, error))?;
    let response_run = super::super::publication::CompletedStructuralResponseSpectrumStep {
        identity: response_identity,
        run_id: response_run_id,
        run_status: completed_accuracy.run_status,
        numerically_acceptable: completed_accuracy.numerically_acceptable,
        solver_convergence: completed_accuracy.solver_convergence,
        result_quality: completed_accuracy.result_quality,
        quality_reasons: completed_accuracy.quality_reasons,
        provenance: persisted.provenance,
        numerical_result: crate::analysis::StructuralResponseSpectrumResultsData {
            numerical_receipt: numerical.response_numerical_receipt,
            accuracy,
            resource_usage: numerical.response_resource_usage,
            publication: numerical.response_publication,
        },
        persistence: persisted.run,
    };

    prepared.steps[0].authored_execution.solver_options =
        AnalysisSolverOptions::StructuralModes(modal_options.clone());
    prepared.steps[1].authored_execution.solver_options =
        AnalysisSolverOptions::StructuralResponseSpectrum(response_options.clone());
    let mut states = expected_basis_states;
    states.push(numerical.envelope_state);
    complete_study_run_prepared(
        prepared,
        vec![
            CompletedStudyStep {
                step_id: numerical.modal_step_id,
                solver_result_identity: modal_result_identity,
                run: CompletedStudyStepRun::Standard(Box::new(modal_run)),
                resolved_solver_options: AnalysisSolverOptions::StructuralModes(modal_options),
            },
            CompletedStudyStep {
                step_id: numerical.response_step_id,
                solver_result_identity: response_result_identity,
                run: CompletedStudyStepRun::StructuralResponseSpectrum(Box::new(response_run)),
                resolved_solver_options: AnalysisSolverOptions::StructuralResponseSpectrum(
                    response_options,
                ),
            },
        ],
        states,
        numerical.frozen_project_digest,
        preparation,
    )
}

fn completion_error(
    prepared: &PreparedStudyCompletion,
    message: impl Into<String>,
) -> OperationErrorEnvelope {
    operation_error(
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        &prepared.context,
        OperationErrorSpec {
            error_code: "RM.FEA.RUN_STUDY.RESPONSE_SPECTRUM_RESULT_INVALID",
            error_type: OperationErrorType::Contract,
            retryable: false,
            severity: OperationErrorSeverity::Fatal,
        },
        message,
        BTreeMap::from([("study_id".to_string(), prepared.study_id.clone())]),
    )
}
