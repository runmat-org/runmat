// Source-bound structural worker dispatch; resource admission precedes state decoding.
use super::*;
use runmat_analysis_fea::execution::StructuralLinearStateInput;
use runmat_analysis_fea::post::fields::structural_static_observable_field_definitions;
use runmat_analysis_fea::results::observation::ObservationLimits;

pub(crate) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    crate::serial::finish_inline(execute_with_capacity(invocation, None))
}

pub(crate) async fn execute_with_capacity(
    invocation: AnalysisKernelInvocation<'_>,
    capacity: Option<&dyn AsyncCapacityOwner>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let phase = runmat_analysis_fea::progress::phase_timer("structural_kernel.decode_inputs");
    let solve_invocation =
        unique_input(&invocation, AnalysisNumericalObjectKind::SolverInvocation)?;
    let mesh_input = optional_unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    let constraint_plan = unique_input(&invocation, AnalysisNumericalObjectKind::ConstraintPlan)?;
    let resource_plan_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    let solve = StructuralLinearWorkload::from_canonical_bytes(&solve_invocation.payload())?;
    let expected_step = runmat_analysis_fea::execution::StepIdentity::from_solver_contract(
        runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
            &solve.model.sequence,
        )?,
        solve.procedure_step()?,
    )?;
    if invocation.host.workload.step != expected_step {
        return Err(AnalysisBridgeError::Identity(
            "structural task names a different authored procedure step",
        ));
    }
    let expected_inputs =
        3 + usize::from(mesh_input.is_some()) + usize::from(solve.state_input.is_some());
    if invocation.inputs.len() != expected_inputs
        || invocation.host.workload.outputs
            != vec![
                AnalysisNumericalObjectKind::SolverResult,
                AnalysisNumericalObjectKind::NumericalReceipt,
                AnalysisNumericalObjectKind::StatePublicationProfiles,
            ]
    {
        return Err(AnalysisBridgeError::Invalid(
            "structural linear solve inputs differ from its invocation and solver-result contract"
                .into(),
        ));
    }
    let mesh = mesh_input
        .map(|input| {
            SolverMeshArtifact::canonical_decode(&input.payload())
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
        })
        .transpose()?;
    if let (Some(input), Some(mesh)) = (mesh_input, mesh.as_ref()) {
        if runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes())
            != input.manifest.logical_identity
        {
            return Err(AnalysisBridgeError::Identity(
                "structural mesh content differs from its input identity",
            ));
        }
    }
    let constraint_plan =
        CanonicalStructuralConstraintPlan::from_canonical_bytes(&constraint_plan.payload())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let topology_digest = constraint_plan.topology.discretization_digest();
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&resource_plan_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if resource_plan_input.manifest.logical_identity != resource_plan.identity.digest() {
        return Err(AnalysisBridgeError::Identity(
            "resource-plan manifest differs from its canonical identity",
        ));
    }
    let all_input_facts = prepared_input_resource_facts(invocation.inputs)?;
    let compiler_limit =
        runmat_analysis_fea::planning::resource::constraint_compilation_available_bytes(
            resource_plan.request.memory_bytes.min(
                solve
                    .procedure_step()?
                    .accuracy
                    .resources
                    .resident_byte_budget(),
            ),
            &all_input_facts,
            resource_plan.input.worker.fixed_memory_bytes,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    // Validate from the unchanged authored constraints; solver_model only rewrites sequence
    // execution controls. Release the fresh validation result before copying that model.
    drop(phase);
    let phase =
        runmat_analysis_fea::progress::phase_timer("structural_kernel.constraint_compilation");
    let compiled = CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
        &solve.model,
        mesh.as_ref(),
        compiler_limit,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if compiled.plan() != &constraint_plan {
        return Err(AnalysisBridgeError::Identity(
            "constraint plan differs from original bounded compilation",
        ));
    }
    drop(phase);
    let phase =
        runmat_analysis_fea::progress::phase_timer("structural_kernel.resource_plan_rederivation");
    let field_definitions = structural_static_observable_field_definitions()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if let Some(capacity) = capacity {
        let work =
            runmat_analysis_fea::structural_linear_preparation_work_upper_bound(mesh.as_ref())?;
        capacity.ensure_capacity(work).await.map_err(|cause| {
            AnalysisBridgeError::FeaRun(runmat_analysis_fea::FeaRunError::Usage(cause))
        })?;
    }
    let observations =
        runmat_analysis_fea::results::observation::resolve_structural_observation_plan(
            solve.procedure_step()?,
            &solve.model,
            mesh.as_ref(),
            &constraint_plan,
            &field_definitions,
            ObservationLimits::default(),
        )
        .map_err(AnalysisBridgeError::from)?;
    let accelerator_requests = resource_plan.request.accelerators.clone();
    let local_execution =
        runmat_analysis_fea::planning::resource::LocalExecutionPlan::from_cpu_millicores(
            resource_plan.request.cpu_millicores,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let prepared_inputs = prepared_input_resource_facts_without_resource_plan(invocation.inputs)?;
    let expected_resource_plan =
        runmat_analysis_fea::planning::resource::build_structural_resource_plan_from_compilation(
            mesh.as_ref(),
            &compiled,
            &observations,
            &solve,
            &accelerator_requests,
            local_execution,
            Some(&prepared_inputs),
            // The submitted plan's recorded solve path is re-derived, never re-selected: the
            // machine memory that chose it is provenance, not an input.
            runmat_analysis_fea::planning::resource::StaticSolveSelection::Recorded(
                runmat_analysis_fea::planning::resource::recorded_static_solve_path(&resource_plan)
                    .ok_or(AnalysisBridgeError::Identity(
                        "structural resource plan records no solve path",
                    ))?,
            ),
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if expected_resource_plan != resource_plan {
        return Err(AnalysisBridgeError::Identity(
            "resource plan does not converge from authoritative structural inputs",
        ));
    }
    drop(compiled);
    let solver_model = if matches!(
        solve.state_input,
        Some(StructuralLinearStateInput::ThermalPrestress(_))
    ) {
        std::borrow::Cow::Borrowed(&solve.model)
    } else {
        std::borrow::Cow::Owned(solve.solver_model()?)
    };
    let expected_constraint_identity = ConstraintPlanIdentity::new(
        invocation.host.workload.step,
        topology_digest,
        constraint_plan
            .content_digest()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    )?;
    if constraint_plan_manifest_identity(&invocation)? != expected_constraint_identity.digest() {
        return Err(AnalysisBridgeError::Identity(
            "constraint-plan manifest differs from the compiled plan",
        ));
    }
    let initial_displacement = solve
        .state_input
        .as_ref()
        .filter(|input| matches!(input, StructuralLinearStateInput::InitialGuess(_)))
        .map(|input| {
            let transition = input.transition();
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
            let state = decode_prepared_semantic_state(state)?;
            let bound = crate::bind_upstream_state_transition(
                &solve.model,
                &solve.step_id,
                &state,
                runmat_analysis_core::StateUse::InitialGuess,
                topology_digest,
            )?;
            if &bound != transition {
                return Err(AnalysisBridgeError::Identity(
                    "initial-guess source does not bind the admitted transition",
                ));
            }
            if state.identity != transition.source_state
                || state.state_id != transition.state_id
                || state.kind != runmat_analysis_core::StateKind::StructuralPrimary
                || state.topology_digest != topology_digest
                || transition.target_topology_digest != topology_digest
            {
                return Err(AnalysisBridgeError::Identity(
                    "initial-guess state differs from its transition or solver topology",
                ));
            }
            let field = state
                .payload
                .fields()
                .ok_or_else(|| {
                    AnalysisBridgeError::Invalid(
                        "structural-primary state has no field payload".into(),
                    )
                })?
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
    drop(phase);
    let phase = runmat_analysis_fea::progress::phase_timer("structural_kernel.run");
    let solve_options = LinearStaticSolveOptions {
        preconditioner_kind: solve.preconditioner,
        algebra_backend_kind: solve.algebra_backend,
        solver_mesh_artifact_path: None,
        solver_mesh: mesh.clone(),
        require_solver_mesh_for_solid: solve.require_solver_mesh_for_solid,
        initial_displacement,
        local_execution,
    };
    let static_admission = (solve.backend == runmat_analysis_fea::ComputeBackend::Cpu).then_some(
        runmat_analysis_fea::StaticResourceAdmission {
            plan: &resource_plan,
            observation_execution: solve.observation_execution,
            prepared_inputs: Some(&prepared_inputs),
        },
    );
    let publishes_linearization = solve
        .procedure_step()?
        .state_outputs
        .iter()
        .any(|output| output.kind == runmat_analysis_core::StateKind::GeometricStiffness);
    let (run, linearization) = if let Some(StructuralLinearStateInput::ThermalPrestress(
        transition,
    )) = &solve.state_input
    {
        let source_input = unique_input(&invocation, AnalysisNumericalObjectKind::SemanticState)?;
        if transition.consumer != invocation.host.workload.step
            || source_input.manifest.logical_identity != transition.source_state.digest()
        {
            return Err(AnalysisBridgeError::Identity(
                "thermal source targets a different worker or state",
            ));
        }
        let source = decode_prepared_semantic_state(source_input)?;
        let bound = crate::bind_upstream_state_transition(
            &solve.model,
            &solve.step_id,
            &source,
            runmat_analysis_core::StateUse::Prestress,
            topology_digest,
        )?;
        if &bound != transition {
            return Err(AnalysisBridgeError::Identity(
                "thermal source does not bind the admitted transition",
            ));
        }
        let execution = runmat_analysis_fea::run_linear_static_from_thermal_state_with_prepared_constraint_plan(
            &solve.model,
            solve.backend,
            solve_options,
            runmat_analysis_fea::PreparedThermoelasticStaticInput {
                consumer_step_id: &solve.step_id,
                source: &source,
                constraint_plan: &constraint_plan,
                admitted: static_admission,
            },
        ).map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        if &execution.thermal_transition != transition {
            return Err(AnalysisBridgeError::Identity(
                "thermal execution transition differs from admitted workload",
            ));
        }
        (execution.run, Some(execution.linearization))
    } else if publishes_linearization {
        let execution = if let Some(capacity) = capacity {
            runmat_analysis_fea::run_linear_static_with_prepared_constraint_plan_and_linearization_tet4_owned(
                &solver_model,
                solve.backend,
                solve_options,
                runmat_analysis_fea::StaticLinearizationInput {
                    constraint_plan: &constraint_plan,
                    topology_digest,
                },
                static_admission,
                usize::MAX,
                |_phase, work| async move {
                    capacity.ensure_capacity(work).await.map_err(|cause| {
                        AnalysisBridgeError::FeaRun(runmat_analysis_fea::FeaRunError::Usage(cause))
                    })
                },
            ).await?
        } else {
            runmat_analysis_fea::run_linear_static_with_prepared_constraint_plan_and_linearization(
                &solver_model,
                solve.backend,
                solve_options,
                &constraint_plan,
                topology_digest,
                static_admission,
            )?
        };
        (execution.run, Some(execution.linearization))
    } else {
        (
            if let Some(capacity) = capacity {
                runmat_analysis_fea::run_linear_static_with_prepared_constraint_plan_tet4_owned(
                    &solver_model,
                    solve.backend,
                    solve_options,
                    &constraint_plan,
                    static_admission,
                    usize::MAX,
                    |_phase, work| async move {
                        capacity.ensure_capacity(work).await.map_err(|cause| {
                            AnalysisBridgeError::FeaRun(runmat_analysis_fea::FeaRunError::Usage(
                                cause,
                            ))
                        })
                    },
                )
                .await?
            } else {
                run_linear_static_with_prepared_constraint_plan(
                    &solver_model,
                    solve.backend,
                    solve_options,
                    &constraint_plan,
                    static_admission,
                )?
            },
            None,
        )
    };
    let constraint_result = run.structural_constraint_result.as_deref().cloned().ok_or(
        AnalysisBridgeError::Identity("structural solve omitted its canonical constraint result"),
    )?;
    let result = match linearization {
        Some(linearization) => StructuralLinearWorkloadResult::with_linearization(
            run,
            constraint_result,
            linearization,
        )?,
        None => StructuralLinearWorkloadResult::new(run, constraint_result)?,
    };
    drop(phase);
    let phase = runmat_analysis_fea::progress::phase_timer("structural_kernel.result_encoding");
    let payload = result.canonical_bytes()?;
    let numerical_receipt =
        StructuralLinearNumericalReceiptArtifact::new(invocation.host.workload.identity, &result)?;
    let result_identity = result.result_identity(invocation.host.workload.identity)?;
    let profiles = crate::structural_state_publication_profiles(
        &solve.model,
        solve.procedure_step()?,
        result_identity,
        runmat_execution::Digest::sha256(&payload),
        crate::StructuralStateProfileResult::Linear(&result),
    )?;
    drop(phase);
    Ok(vec![
        AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::SolverResult,
            result_identity.digest(),
            payload,
        ),
        AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::NumericalReceipt,
            numerical_receipt.identity().digest(),
            numerical_receipt.canonical_bytes()?,
        ),
        AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::StatePublicationProfiles,
            profiles.identity(),
            profiles.canonical_bytes()?,
        ),
    ])
}
