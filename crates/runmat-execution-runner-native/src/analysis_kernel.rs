use runmat_analysis_execution::{
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisKernel, AnalysisKernelInvocation,
    AnalysisKernelOutput,
};
use runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan;
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, CanonicalSemanticState, ConstraintPlanIdentity,
    SemanticStatePublicationRequest, SolverResultIdentity, StructuralLinearWorkload,
    StructuralLinearWorkloadResult,
};
use runmat_analysis_fea::{
    run_linear_static_with_prepared_constraint_plan, LinearStaticSolveOptions,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeAnalysisKernel;

impl AnalysisKernel for NativeAnalysisKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        match invocation.kind {
            AnalysisWorkKind::Solve => execute_structural_linear(invocation),
            AnalysisWorkKind::RecoverField => execute_state_publication(invocation),
            _ => Err(AnalysisBridgeError::Invalid(
                "native analysis kernel does not yet implement this work kind".into(),
            )),
        }
    }
}

fn execute_structural_linear(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let solve_invocation =
        unique_input(&invocation, AnalysisNumericalObjectKind::SolverInvocation)?;
    let mesh = unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    let constraint_plan = unique_input(&invocation, AnalysisNumericalObjectKind::ConstraintPlan)?;
    let solve = StructuralLinearWorkload::from_canonical_bytes(&solve_invocation.payload())?;
    let expected_inputs = if solve.initial_guess_transition.is_some() {
        4
    } else {
        3
    };
    if invocation.inputs.len() != expected_inputs
        || invocation.host.workload.outputs != vec![AnalysisNumericalObjectKind::SolverResult]
    {
        return Err(AnalysisBridgeError::Invalid(
            "structural linear solve inputs differ from its invocation and solver-result contract"
                .into(),
        ));
    }
    let initial_displacement = solve
        .initial_guess_transition
        .as_ref()
        .map(|transition| {
            if transition.consumer != invocation.host.workload.step {
                return Err(AnalysisBridgeError::Identity(
                    "initial-guess transition targets a different solver step",
                ));
            }
            let state = unique_input(&invocation, AnalysisNumericalObjectKind::SemanticState)?;
            if state.manifest.logical_identity != transition.source_state.digest() {
                return Err(AnalysisBridgeError::Identity(
                    "initial-guess state manifest differs from its transition",
                ));
            }
            let state = CanonicalSemanticState::from_canonical_bytes(&state.payload())?;
            if state.identity != transition.source_state
                || state.state_id != transition.state_id
                || state.kind != runmat_analysis_core::StateKind::StructuralPrimary
                || state.topology_digest != mesh.manifest.logical_identity
                || transition.target_topology_digest != mesh.manifest.logical_identity
            {
                return Err(AnalysisBridgeError::Identity(
                    "initial-guess state differs from its transition or solver topology",
                ));
            }
            let field = state
                .fields
                .iter()
                .find(|field| {
                    field.field_id == runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT
                })
                .ok_or_else(|| {
                    AnalysisBridgeError::Invalid(
                        "structural-primary state has no displacement field".into(),
                    )
                })?;
            field.as_host_f64().map(<[f64]>::to_vec).ok_or_else(|| {
                AnalysisBridgeError::Invalid(
                    "structural-primary displacement is not host-readable".into(),
                )
            })
        })
        .transpose()?;
    let solver_model = solve.solver_model()?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let constraint_plan =
        CanonicalStructuralConstraintPlan::from_canonical_bytes(&constraint_plan.payload())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    constraint_plan
        .validate_against(&solver_model, &mesh)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let expected_constraint_identity = ConstraintPlanIdentity::new(
        invocation.host.workload.step,
        runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes()),
        constraint_plan
            .content_digest()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    )?;
    if constraint_plan_manifest_identity(&invocation)? != expected_constraint_identity.digest() {
        return Err(AnalysisBridgeError::Identity(
            "constraint-plan manifest differs from the compiled plan",
        ));
    }
    let run = run_linear_static_with_prepared_constraint_plan(
        &solver_model,
        solve.backend,
        LinearStaticSolveOptions {
            formulation_requirement: solve.formulation_requirement,
            preconditioner_kind: solve.preconditioner,
            algebra_backend_kind: solve.algebra_backend,
            solver_mesh_artifact_path: None,
            solver_mesh: Some(mesh),
            require_solver_mesh_for_solid: solve.require_solver_mesh_for_solid,
            thermo_mechanical_context: solve.thermo_mechanical_context,
            electro_thermal_context: solve.electro_thermal_context,
            initial_displacement,
        },
        &constraint_plan,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let payload = StructuralLinearWorkloadResult {
        schema_version: 1,
        run,
    }
    .canonical_bytes()?;
    let result_identity = SolverResultIdentity::new(
        invocation.host.workload.identity,
        runmat_execution::Digest::sha256(&payload),
    )?;
    Ok(vec![AnalysisKernelOutput {
        kind: AnalysisNumericalObjectKind::SolverResult,
        logical_identity: result_identity.digest(),
        payload,
    }])
}

fn constraint_plan_manifest_identity(
    invocation: &AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<runmat_execution::Digest> {
    Ok(
        unique_input(invocation, AnalysisNumericalObjectKind::ConstraintPlan)?
            .manifest
            .logical_identity,
    )
}

fn execute_state_publication(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let publication = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StatePublicationRequest,
    )?;
    let solver_result = unique_input(&invocation, AnalysisNumericalObjectKind::SolverResult)?;
    if invocation.inputs.len() != 2
        || invocation.host.workload.outputs != vec![AnalysisNumericalObjectKind::SemanticState]
    {
        return Err(AnalysisBridgeError::Invalid(
            "state publication requires one publication request, one solver result, and one semantic-state output"
                .into(),
        ));
    }
    let publication =
        SemanticStatePublicationRequest::from_canonical_bytes(&publication.payload())?;
    if solver_result.manifest.logical_identity != publication.source_result.digest() {
        return Err(AnalysisBridgeError::Identity(
            "state publication received a substituted solver result",
        ));
    }
    let solver_result =
        StructuralLinearWorkloadResult::from_canonical_bytes(&solver_result.payload())?;
    let state = CanonicalSemanticState::from_publication(&publication, &solver_result.run.fields)?;
    Ok(vec![AnalysisKernelOutput {
        kind: AnalysisNumericalObjectKind::SemanticState,
        logical_identity: state.identity.digest(),
        payload: state.canonical_bytes()?,
    }])
}

fn unique_input<'a>(
    invocation: &'a AnalysisKernelInvocation<'_>,
    kind: AnalysisNumericalObjectKind,
) -> AnalysisBridgeResult<&'a runmat_analysis_execution::PreparedAnalysisObjects> {
    let mut matches = invocation
        .inputs
        .iter()
        .filter(|input| input.manifest.kind == kind);
    let input = matches.next().ok_or_else(|| {
        AnalysisBridgeError::Invalid(format!("analysis workload is missing its {kind:?} input"))
    })?;
    if matches.next().is_some() {
        return Err(AnalysisBridgeError::Invalid(format!(
            "analysis workload has multiple {kind:?} inputs"
        )));
    }
    Ok(input)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use runmat_analysis_execution::{
        AnalysisCapabilityRequirement, AnalysisHostWorkload, AnalysisWorkloadInput,
        AnalysisWorkloadRequest,
    };
    use runmat_analysis_fea::execution::{
        AccuracyIdentity, ConstraintPlanIdentity, DiscretizationIdentity, ExecutionPlanIdentity,
        NumericalPlanIdentity, OperatorIdentity, ProblemIdentity, SequenceIdentity, StepIdentity,
        StudyIdentity,
    };
    use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
    use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
    use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
    use runmat_analysis_fea::ComputeBackend;
    use runmat_execution::value::ValueRefKind;
    use runmat_execution::Digest;
    use runmat_execution_artifact::object::ObjectInventoryLimits;
    use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
    use runmat_meshing_core::{
        ElementOrder, PersistentEntityId, PersistentEntityKind, StableDigest,
    };

    use super::*;

    fn digest(label: &str) -> Digest {
        Digest::sha256(label)
    }

    fn execution_plan(step: StepIdentity) -> ExecutionPlanIdentity {
        let constraints =
            ConstraintPlanIdentity::new(step, digest("mesh"), digest("constraints")).unwrap();
        let problem =
            ProblemIdentity::new(step, digest("model"), None, constraints, Vec::new()).unwrap();
        let discretization =
            DiscretizationIdentity::new(problem, digest("mesh"), digest("formulation")).unwrap();
        let operator = OperatorIdentity::new(discretization, digest("operator")).unwrap();
        let accuracy = AccuracyIdentity::new(step, digest("accuracy")).unwrap();
        let numerical = NumericalPlanIdentity::new(operator, digest("solver"), accuracy).unwrap();
        ExecutionPlanIdentity::new(numerical, digest("execution")).unwrap()
    }

    #[test]
    fn native_kernel_executes_path_free_structural_linear_workload() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        let producer_step = runmat_analysis_core::ProcedureStep::structural_static("preload");
        let state_id = producer_step.state_outputs[0].state_id.clone();
        let mut consumer_step = runmat_analysis_core::ProcedureStep::structural_static("branch");
        consumer_step
            .state_inputs
            .push(runmat_analysis_core::StateInput {
                input_id: "warm-start".into(),
                required_kind: runmat_analysis_core::StateKind::StructuralPrimary,
                source: runmat_analysis_core::StateSource::Upstream {
                    producer_step: producer_step.step_id.clone(),
                    state_id: state_id.clone(),
                },
                use_as: runmat_analysis_core::StateUse::InitialGuess,
                conflict: runmat_analysis_core::InitialStateConflictPolicy::Reject,
            });
        let authored_transition = runmat_analysis_core::StateTransition {
            transition_id: runmat_analysis_core::TransitionId("warm-start".into()),
            producer_step: producer_step.step_id.clone(),
            consumer_step: consumer_step.step_id.clone(),
            state_id,
            method: runmat_analysis_core::StateTransitionMethod::InitialGuessOnly,
            topology: runmat_analysis_core::TopologyTransitionPolicy::RequireIdentical,
            reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
        };
        model.sequence = runmat_analysis_core::AnalysisSequence {
            sequence_id: "stateful-sequence".into(),
            steps: vec![producer_step, consumer_step],
            dependencies: vec![runmat_analysis_core::SequenceDependency {
                predecessor: authored_transition.producer_step.clone(),
                successor: authored_transition.consumer_step.clone(),
                reason: runmat_analysis_core::SequenceDependencyReason::StateTransfer,
            }],
            transitions: vec![authored_transition.clone()],
            branches: Vec::new(),
        };
        let step_id = model.sequence.steps[0].step_id.clone();
        let solve = StructuralLinearWorkload {
            schema_version: 1,
            model,
            step_id,
            backend: ComputeBackend::Cpu,
            formulation_requirement: runmat_runtime::analysis::FormulationRequirement::Experimental,
            preconditioner: SpdPreconditionerKind::Jacobi,
            algebra_backend: LinearAlgebraBackendKind::CpuReference,
            require_solver_mesh_for_solid: false,
            thermo_mechanical_context: None,
            electro_thermal_context: None,
            initial_guess_transition: None,
        };
        let state = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::SolverInvocation,
            digest("state"),
            &solve.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let mut mesh =
            runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        for (face_index, face) in mesh.topology.boundary_faces.iter_mut().enumerate() {
            face.provenance.push(PersistentEntityId {
                kind: PersistentEntityKind::Face,
                source_topology_id: if face_index == 0 { "root" } else { "tip" }.into(),
                assembly_path: vec!["root".into()],
            });
            face.provenance.sort();
        }
        if let Some(assignment) = solve.model.material_assignments.first() {
            let region = PersistentEntityId {
                kind: PersistentEntityKind::Region,
                source_topology_id: assignment.region_id.clone(),
                assembly_path: vec!["root".into()],
            };
            mesh.topology.volume_elements[0].region_id = region.clone();
            mesh.topology.regions[0].region_id = region;
        }
        mesh.canonical_digest = StableDigest::ZERO;
        mesh.seal_canonical_digest().unwrap();
        let constraint_plan =
            CanonicalStructuralConstraintPlan::compile(&solve.model, &mesh).unwrap();
        let constraint_plan_bytes = constraint_plan.canonical_bytes().unwrap();
        let constraint_plan_digest = constraint_plan.content_digest().unwrap();
        let mesh = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::Mesh,
            Digest::from_bytes(*mesh.canonical_digest.bytes()),
            &mesh.canonical_encode().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let study = StudyIdentity::new(digest("study"), None).unwrap();
        let sequence = SequenceIdentity::from_authored(study, &solve.model.sequence).unwrap();
        let step = StepIdentity::from_authored(sequence, solve.procedure_step().unwrap()).unwrap();
        let constraint_plan = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::ConstraintPlan,
            ConstraintPlanIdentity::new(
                step,
                mesh.manifest.logical_identity,
                constraint_plan_digest,
            )
            .unwrap()
            .digest(),
            &constraint_plan_bytes,
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let inputs = vec![state, mesh, constraint_plan];
        let workload_inputs = inputs
            .iter()
            .map(|input| AnalysisWorkloadInput {
                kind: input.manifest.kind,
                logical_identity: input.manifest.logical_identity,
                payload_digest: input.manifest.payload_digest,
                manifest_digest: input.root.digest,
                manifest_encoded_length: input.root.encoded_length,
                reference_kind: ValueRefKind::DriverObject,
            })
            .collect();
        let execution_plan = execution_plan(step);
        let workload = AnalysisWorkloadRequest::new(
            study,
            step,
            execution_plan,
            AnalysisWorkKind::Solve,
            workload_inputs,
            vec![AnalysisNumericalObjectKind::SolverResult],
            BTreeSet::from([AnalysisCapabilityRequirement::Host {
                abi: runmat_analysis_execution::ANALYSIS_HOST_ABI.into(),
            }]),
        )
        .unwrap();
        let host = AnalysisHostWorkload::new(
            workload,
            ArtifactAccess {
                authorization_scope: "study:test".into(),
                encryption_context: digest("encryption"),
            },
        )
        .unwrap();
        let outputs = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::Solve,
                host: &host,
                inputs: &inputs,
            })
            .unwrap();
        let result =
            StructuralLinearWorkloadResult::from_canonical_bytes(&outputs[0].payload).unwrap();
        assert!(!result.run.fields_are_empty());
        assert!(result
            .run
            .diagnostics
            .iter()
            .all(|diagnostic| !diagnostic.message.contains("artifact_path")));

        let mut substituted_plan = inputs.clone();
        substituted_plan[2].manifest.logical_identity = digest("substituted-constraint-plan");
        assert!(NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::Solve,
                host: &host,
                inputs: &substituted_plan,
            })
            .is_err());

        let solver_result = runmat_analysis_execution::prepare_analysis_objects(
            outputs[0].kind,
            outputs[0].logical_identity,
            &outputs[0].payload,
            4096,
            ObjectNamespace::ResultValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let publication = SemanticStatePublicationRequest::structural_linear(
            step,
            solve.model.sequence.steps[0].state_outputs[0].clone(),
            inputs[1].manifest.logical_identity,
            SolverResultIdentity::new(
                host.workload.identity,
                solver_result.manifest.payload_digest,
            )
            .unwrap(),
        )
        .unwrap();
        let publication_bytes = publication.canonical_bytes().unwrap();
        let publication = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::StatePublicationRequest,
            Digest::sha256(&publication_bytes),
            &publication_bytes,
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let state_inputs = vec![publication, solver_result];
        let state_workload_inputs = state_inputs
            .iter()
            .map(|input| AnalysisWorkloadInput {
                kind: input.manifest.kind,
                logical_identity: input.manifest.logical_identity,
                payload_digest: input.manifest.payload_digest,
                manifest_digest: input.root.digest,
                manifest_encoded_length: input.root.encoded_length,
                reference_kind: ValueRefKind::DriverObject,
            })
            .collect();
        let state_host = AnalysisHostWorkload::new(
            AnalysisWorkloadRequest::new(
                study,
                step,
                execution_plan,
                AnalysisWorkKind::RecoverField,
                state_workload_inputs,
                vec![AnalysisNumericalObjectKind::SemanticState],
                BTreeSet::from([AnalysisCapabilityRequirement::Host {
                    abi: runmat_analysis_execution::ANALYSIS_HOST_ABI.into(),
                }]),
            )
            .unwrap(),
            host.artifact_access.clone(),
        )
        .unwrap();
        let state_outputs = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::RecoverField,
                host: &state_host,
                inputs: &state_inputs,
            })
            .unwrap();
        let state =
            CanonicalSemanticState::from_canonical_bytes(&state_outputs[0].payload).unwrap();
        assert_eq!(
            state.fields[0].field_id,
            runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT
        );
        assert_eq!(state.topology_digest, inputs[1].manifest.logical_identity);

        let consumer_step = solve
            .model
            .sequence
            .steps
            .iter()
            .find(|candidate| candidate.step_id.as_str() == "branch")
            .unwrap();
        let consumer = StepIdentity::from_authored(sequence, consumer_step).unwrap();
        let transition = runmat_analysis_fea::execution::CanonicalStateTransition::new(
            runmat_analysis_fea::execution::CanonicalStateTransitionInput {
                sequence,
                producer: step,
                consumer,
                authored: &authored_transition,
                source: &state,
                target_topology_digest: inputs[1].manifest.logical_identity,
            },
        )
        .unwrap();
        let mut warm_solve = solve.clone();
        warm_solve.step_id = consumer_step.step_id.clone();
        warm_solve.initial_guess_transition = Some(transition);
        let warm_invocation = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::SolverInvocation,
            digest("warm-invocation"),
            &warm_solve.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let warm_state = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::SemanticState,
            state.identity.digest(),
            &state.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::ResultValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let warm_constraint_plan = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::ConstraintPlan,
            ConstraintPlanIdentity::new(
                consumer,
                inputs[1].manifest.logical_identity,
                constraint_plan_digest,
            )
            .unwrap()
            .digest(),
            &constraint_plan_bytes,
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let warm_inputs = vec![
            warm_invocation,
            inputs[1].clone(),
            warm_constraint_plan,
            warm_state,
        ];
        let warm_workload_inputs = warm_inputs
            .iter()
            .map(|input| AnalysisWorkloadInput {
                kind: input.manifest.kind,
                logical_identity: input.manifest.logical_identity,
                payload_digest: input.manifest.payload_digest,
                manifest_digest: input.root.digest,
                manifest_encoded_length: input.root.encoded_length,
                reference_kind: ValueRefKind::DriverObject,
            })
            .collect();
        let warm_host = AnalysisHostWorkload::new(
            AnalysisWorkloadRequest::new(
                study,
                consumer,
                execution_plan,
                AnalysisWorkKind::Solve,
                warm_workload_inputs,
                vec![AnalysisNumericalObjectKind::SolverResult],
                BTreeSet::from([AnalysisCapabilityRequirement::Host {
                    abi: runmat_analysis_execution::ANALYSIS_HOST_ABI.into(),
                }]),
            )
            .unwrap(),
            host.artifact_access.clone(),
        )
        .unwrap();
        let warm_outputs = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::Solve,
                host: &warm_host,
                inputs: &warm_inputs,
            })
            .unwrap();
        let warm_result =
            StructuralLinearWorkloadResult::from_canonical_bytes(&warm_outputs[0].payload).unwrap();
        assert_eq!(
            result
                .run
                .field(runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT),
            warm_result
                .run
                .field(runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        );
        assert!(warm_result.run.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "FEA_CONVERGENCE" && diagnostic.message.contains("iterations=0 ")
        }));

        let mut substituted = warm_inputs.clone();
        substituted[2].manifest.logical_identity = digest("substituted-state");
        assert!(NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::Solve,
                host: &warm_host,
                inputs: &substituted,
            })
            .is_err());
    }
}
