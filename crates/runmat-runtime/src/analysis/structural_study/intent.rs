//! Immutable structural study intent validation and continuation identity.
use super::*;
impl PreparedStructuralStudyIntent {
    pub fn authored_study_digest(&self) -> runmat_execution::Digest {
        self.authored_study_digest
    }

    pub fn study_id(&self) -> &str {
        &self.study_id
    }

    pub fn operation_context(&self) -> &OperationContext {
        &self.context
    }

    pub fn model(&self) -> &AnalysisModel {
        &self.model
    }

    pub fn workloads(&self) -> StructuralStudyWorkloads {
        self.workloads.clone()
    }

    pub fn operation_sequence(&self) -> &[String] {
        &self.operation_sequence
    }

    pub fn meshing_request(&self) -> Option<&runmat_meshing_core::MeshingRequest> {
        self.meshing_request.as_ref()
    }

    pub fn geometry_unit_evidence(&self) -> &runmat_geometry_core::physical::LengthUnitEvidence {
        &self.geometry_unit_evidence
    }

    pub fn meshing_contacts(&self) -> &[crate::meshing::MeshingContactDefinition] {
        &self.meshing_contacts
    }

    pub fn geometry_source_digest(&self) -> runmat_execution::Digest {
        self.geometry_source_digest
    }

    pub fn continuation_bytes(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        let bytes = serde_json::to_vec(&StructuralStudyIntentContinuation {
            schema_version: 2,
            study_id: self.study_id.clone(),
            geometry_source_digest: self.geometry_source_digest,
            geometry_unit_evidence: self.geometry_unit_evidence.clone(),
            meshing_request: self.meshing_request.clone(),
            meshing_contacts: self.meshing_contacts.clone(),
            context: self.context.clone(),
            authored_study_digest: self.authored_study_digest,
            intent_digest: self.intent_digest,
            model: self.model.clone(),
            steps: self.steps.clone(),
            workloads: self.workloads.clone(),
            operation_sequence: self.operation_sequence.clone(),
        })
        .map_err(|error| error.to_string())?;
        if bytes.len() > STRUCTURAL_STUDY_INTENT_MAX_BYTES {
            return Err("structural study intent continuation exceeds its bound".into());
        }
        Ok(bytes)
    }

    pub fn from_continuation_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.is_empty() || bytes.len() > STRUCTURAL_STUDY_INTENT_MAX_BYTES {
            return Err("structural study intent continuation exceeds its bound".into());
        }
        let continuation: StructuralStudyIntentContinuation =
            serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
        if continuation.schema_version != 2 {
            return Err("unsupported structural study intent continuation schema".into());
        }
        let intent = Self {
            study_id: continuation.study_id,
            geometry_source_digest: continuation.geometry_source_digest,
            geometry_unit_evidence: continuation.geometry_unit_evidence,
            meshing_request: continuation.meshing_request,
            meshing_contacts: continuation.meshing_contacts,
            context: continuation.context,
            authored_study_digest: continuation.authored_study_digest,
            intent_digest: continuation.intent_digest,
            model: continuation.model,
            steps: continuation.steps,
            workloads: continuation.workloads,
            operation_sequence: continuation.operation_sequence,
        };
        intent.validate()?;
        Ok(intent)
    }

    pub(in crate::analysis) fn validate(&self) -> Result<(), String> {
        self.geometry_unit_evidence
            .validate()
            .map_err(|error| error.to_string())?;
        if !self.geometry_unit_evidence.physical_scale_resolved() {
            return Err(
                "structural study intent requires resolved source-unit interpretation".into(),
            );
        }

        if let Some(request) = &self.meshing_request {
            request.validate().map_err(|error| error.to_string())?;
        }
        if self.study_id.trim().is_empty()
            || self.geometry_source_digest.bytes() == &[0_u8; 32]
            || self.authored_study_digest.bytes() == &[0_u8; 32]
            || self.meshing_contacts
                != crate::analysis::meshing::meshing_contact_definitions_for_model(&self.model)
                    .collect::<Vec<_>>()
        {
            return Err("structural study intent semantics are inconsistent".into());
        }
        if self.steps.len() != self.model.sequence.steps.len() {
            return Err("structural study intent does not cover every sequence step".into());
        }
        self.workloads
            .validate()
            .map_err(|error| error.to_string())?;
        if self
            .workloads
            .authored_model()
            .map_err(|error| error.to_string())?
            != &self.model
        {
            return Err("structural study workload model differs from its intent model".into());
        }
        self.validate_workload_alignment()?;
        for (index, step) in self.steps.iter().enumerate() {
            let authored = self
                .model
                .sequence
                .steps
                .get(index)
                .ok_or_else(|| "structural study intent step order is invalid".to_string())?;
            if step.procedure_step != *authored
                || step.authored_execution.step_id != authored.step_id
            {
                return Err("structural study intent step semantics are inconsistent".into());
            }
        }
        let semantic_digest = self.semantic_digest()?;
        if self.intent_digest != semantic_digest {
            return Err("structural study intent digest differs from its identity".into());
        }
        Ok(())
    }

    fn validate_workload_alignment(&self) -> Result<(), String> {
        match &self.workloads {
            StructuralStudyWorkloads::HeterogeneousSequence { workloads } => {
                if workloads.len() != self.steps.len() {
                    return Err(
                        "heterogeneous structural workloads do not cover the prepared intent steps"
                            .into(),
                    );
                }
                for step in &self.steps {
                    let workload =
                        workloads.get(&step.procedure_step.step_id).ok_or_else(|| {
                            "heterogeneous structural workload is missing its step".to_string()
                        })?;
                    match (workload, &step.authored_execution.solver_options) {
                        (runmat_analysis_fea::execution::StructuralStepWorkload::Thermal { workload }, crate::analysis::AnalysisSolverOptions::ThermalDiffusion(options))
                            if workload.step_id == step.procedure_step.step_id
                                && workload.backend == step.authored_execution.backend
                                && options.residual_warn_threshold == workload.residual_target
                                && options.snapshot_count == match workload.procedure { runmat_analysis_fea::ThermalProcedureControl::Steady => 1, runmat_analysis_fea::ThermalProcedureControl::BackwardEuler { snapshot_count } => snapshot_count }
                                && options.solver_mesh_artifact_path.is_none() => {},
                        (
                            runmat_analysis_fea::execution::StructuralStepWorkload::Linear {
                                workload,
                            },
                            crate::analysis::AnalysisSolverOptions::StructuralLinear(options),
                        ) if workload.step_id == step.procedure_step.step_id
                            && linear_workload_matches_prepared_execution(
                                workload,
                                &step.authored_execution,
                                options,
                            ) => {}
                        (
                            runmat_analysis_fea::execution::StructuralStepWorkload::ModalExtraction {
                                workload,
                            },
                            crate::analysis::AnalysisSolverOptions::StructuralModes(options),
                        ) if workload.step_id() == &step.procedure_step.step_id
                            && workload.backend() == step.authored_execution.backend
                            && workload.observation_execution()
                                == ObservationExecutionPolicy::scalable()
                            && workload.truncation() == options.truncation
                            && workload.convergence() == options.convergence
                            && workload.residual_vector_directions()
                                == options.residual_vector_directions
                            && options.solver_mesh_artifact_path.is_none() => {}
                        (
                            runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedModal {
                                template,
                            },
                            crate::analysis::AnalysisSolverOptions::StructuralModes(options),
                        ) if template.step_id == step.procedure_step.step_id
                            && template.backend == step.authored_execution.backend
                            && template.observation_execution
                                == ObservationExecutionPolicy::scalable()
                            && template.truncation == options.truncation
                            && template.convergence == options.convergence
                            && template.residual_vector_directions
                                == options.residual_vector_directions
                            && options.solver_mesh_artifact_path.is_none() => {}
                        (
                            runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedFrequency {
                                template,
                            },
                            crate::analysis::AnalysisSolverOptions::StructuralFrequency(options),
                        ) if template.step_id == step.procedure_step.step_id
                            && template.backend == step.authored_execution.backend
                            && template.observation_execution
                                == ObservationExecutionPolicy::scalable()
                            && match template.solver {
                                runmat_analysis_fea::execution::StructuralPrestressedFrequencySolverTemplate::Direct { policy } => {
                                    step.procedure_step.solution_space
                                        == runmat_analysis_core::SolutionSpace::Direct
                                        && matches!(
                                            options.solver,
                                            crate::analysis::AnalysisStructuralFrequencySolver::Direct {
                                                policy: authored_policy,
                                            } if authored_policy == policy
                                        )
                                }
                                runmat_analysis_fea::execution::StructuralPrestressedFrequencySolverTemplate::ModalReduced => {
                                    step.procedure_step.solution_space
                                        == runmat_analysis_core::SolutionSpace::ModalReduced
                                        && options.solver
                                            == crate::analysis::AnalysisStructuralFrequencySolver::ModalReduced
                                }
                            }
                            && options.solver_mesh_artifact_path.is_none() => {}
                        (
                            runmat_analysis_fea::execution::StructuralStepWorkload::Buckling {
                                template,
                            },
                            crate::analysis::AnalysisSolverOptions::StructuralBuckling(options),
                        ) if template.step_id == step.procedure_step.step_id
                            && template.backend == step.authored_execution.backend
                            && template.observation_execution
                                == ObservationExecutionPolicy::scalable()
                            && template.solver == options.solver
                            && options.solver_mesh_artifact_path.is_none() => {}
                        (
                            runmat_analysis_fea::execution::StructuralStepWorkload::ModalTransient {
                                template,
                                block_plan,
                            },
                            crate::analysis::AnalysisSolverOptions::StructuralTransient(options),
                        ) if template.step_id == step.procedure_step.step_id
                            && template.backend == step.authored_execution.backend
                            && template.time_step_policy.interval_count()
                                == options.time_step_policy.interval_count()
                            && template.residual_target == options.residual_target
                            && block_plan.maximum_intervals_per_block()
                                == options.maximum_intervals_per_block
                            && options.solver_mesh_artifact_path.is_none() => {}
                        (
                            runmat_analysis_fea::execution::StructuralStepWorkload::ModalFrequency {
                                template,
                            },
                            crate::analysis::AnalysisSolverOptions::StructuralFrequency(options),
                        ) if template.step_id == step.procedure_step.step_id
                            && template.backend == step.authored_execution.backend
                            && template.observation_execution
                                == ObservationExecutionPolicy::scalable()
                            && step.procedure_step.solution_space
                                == runmat_analysis_core::SolutionSpace::ModalReduced
                            && options.solver
                                == crate::analysis::AnalysisStructuralFrequencySolver::ModalReduced
                            && options.solver_mesh_artifact_path.is_none() => {}
                        (
                            runmat_analysis_fea::execution::StructuralStepWorkload::ModalResponseSpectrum {
                                template,
                            },
                            crate::analysis::AnalysisSolverOptions::StructuralResponseSpectrum(options),
                        ) if template.step_id == step.procedure_step.step_id
                            && template.backend == step.authored_execution.backend
                            && template.observation_execution
                                == ObservationExecutionPolicy::scalable()
                            && options.precision_mode == crate::analysis::PrecisionMode::Fp64
                            && options.solver_mesh_artifact_path.is_none() => {}
                        (
                            runmat_analysis_fea::execution::StructuralStepWorkload::RandomResponse {
                                template,
                            },
                            crate::analysis::AnalysisSolverOptions::StructuralRandomResponse(options),
                        ) if template.step_id == step.procedure_step.step_id
                            && template.backend == step.authored_execution.backend
                            && template.observation_execution
                                == ObservationExecutionPolicy::scalable()
                            && template.integration_policy == options.integration
                            && template.maximum_intervals_per_task
                                == options.maximum_intervals_per_task
                            && options.precision_mode == crate::analysis::PrecisionMode::Fp64
                            && options.solver_mesh_artifact_path.is_none() => {}
                        _ => {
                            return Err(
                                "heterogeneous structural workload differs from its prepared execution"
                                    .into(),
                            )
                        }
                    }
                }
            }
            StructuralStudyWorkloads::LinearSequence { workloads } => {
                if workloads.len() != self.steps.len() {
                    return Err(
                        "structural linear workloads do not cover the prepared intent steps".into(),
                    );
                }
                for step in &self.steps {
                    let crate::analysis::AnalysisSolverOptions::StructuralLinear(options) =
                        &step.authored_execution.solver_options
                    else {
                        return Err(
                            "structural linear workload has non-linear authored options".into()
                        );
                    };
                    let workload =
                        workloads.get(&step.procedure_step.step_id).ok_or_else(|| {
                            "structural linear workload is missing its step".to_string()
                        })?;
                    if !linear_workload_matches_prepared_execution(
                        workload,
                        &step.authored_execution,
                        options,
                    ) {
                        return Err(
                            "structural linear workload differs from its prepared execution".into(),
                        );
                    }
                }
            }
            StructuralStudyWorkloads::StructuralTransient {
                workload,
                block_plan,
            } => {
                let [step] = self.steps.as_slice() else {
                    return Err(
                        "direct structural transient requires exactly one prepared step".into(),
                    );
                };
                let crate::analysis::AnalysisSolverOptions::StructuralTransient(options) =
                    &step.authored_execution.solver_options
                else {
                    return Err(
                        "direct structural transient workload has non-transient authored options"
                            .into(),
                    );
                };
                let expected_time_step_policy = match options.time_step_policy {
                    crate::analysis::AnalysisStructuralTimeStepPolicy::Fixed { interval_count } => {
                        runmat_analysis_fea::solve::transient::StructuralTimeStepPolicy::Fixed {
                            interval_count,
                        }
                    }
                };
                if workload.step_id != step.procedure_step.step_id
                    || workload.backend != step.authored_execution.backend
                    || workload.solver
                        != runmat_analysis_fea::execution::StructuralTransientSolverPlan::Direct
                    || workload.time_step_policy != expected_time_step_policy
                    || workload.residual_target != options.residual_target
                    || block_plan.maximum_intervals_per_block()
                        != options.maximum_intervals_per_block
                    || options.solver_mesh_artifact_path.is_some()
                {
                    return Err(
                        "direct structural transient workload differs from its prepared execution"
                            .into(),
                    );
                }
            }
            StructuralStudyWorkloads::StructuralFrequency { workload } => {
                let [step] = self.steps.as_slice() else {
                    return Err(
                        "direct structural frequency execution requires exactly one prepared step"
                            .into(),
                    );
                };
                let crate::analysis::AnalysisSolverOptions::StructuralFrequency(options) =
                    &step.authored_execution.solver_options
                else {
                    return Err(
                        "structural frequency workload has non-frequency authored options".into(),
                    );
                };
                let runmat_analysis_fea::execution::StructuralFrequencySolverPlan::Direct {
                    policy,
                } = workload.solver()
                else {
                    return Err(
                        "standalone structural frequency execution requires a direct solver".into(),
                    );
                };
                if workload.step_id() != &step.procedure_step.step_id
                    || workload.backend() != step.authored_execution.backend
                    || workload.observation_execution() != ObservationExecutionPolicy::scalable()
                    || !matches!(
                        options.solver,
                        crate::analysis::AnalysisStructuralFrequencySolver::Direct {
                            policy: authored_policy,
                        } if authored_policy == *policy
                    )
                    || options.solver_mesh_artifact_path.is_some()
                {
                    return Err(
                        "structural frequency workload differs from its prepared execution".into(),
                    );
                }
            }
            StructuralStudyWorkloads::StructuralModal { workload } => {
                let [step] = self.steps.as_slice() else {
                    return Err(
                        "structural modal extraction requires exactly one prepared step".into(),
                    );
                };
                let crate::analysis::AnalysisSolverOptions::StructuralModes(options) =
                    &step.authored_execution.solver_options
                else {
                    return Err("structural modal workload has non-modal authored options".into());
                };
                if workload.step_id() != &step.procedure_step.step_id
                    || workload.backend() != step.authored_execution.backend
                    || workload.observation_execution() != ObservationExecutionPolicy::scalable()
                    || workload.truncation() != options.truncation
                    || workload.convergence() != options.convergence
                    || workload.residual_vector_directions() != options.residual_vector_directions
                    || options.solver_mesh_artifact_path.is_some()
                {
                    return Err(
                        "structural modal workload differs from its prepared execution".into(),
                    );
                }
            }
        }
        Ok(())
    }

    pub(in crate::analysis) fn semantic_digest(&self) -> Result<runmat_execution::Digest, String> {
        let bytes = serde_json::to_vec(&(
            &self.study_id,
            self.geometry_source_digest,
            &self.geometry_unit_evidence,
            &self.meshing_request,
            &self.meshing_contacts,
            &self.model,
            &self.steps,
            &self.workloads,
            &self.operation_sequence,
        ))
        .map_err(|error| error.to_string())?;
        Ok(runmat_execution::Digest::sha256(&bytes))
    }
}
