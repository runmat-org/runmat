use runmat_analysis_core::{
    validate_model, AnalysisField, AnalysisModel, AnalysisProcedure, StructuralConstraintKind,
};
mod preflight;
mod prepared;
mod reference_diagnostics;
mod solve;
mod static_diagnostics;
use reference_diagnostics::{
    scalar_field_value, structural_beam_closed_form_diagnostic,
    structural_moment_balance_diagnostic, structural_reference_kinematics_diagnostic,
};
mod static_thermomechanical;
mod thermal_state;
use crate::{
    algebra::apply_k,
    assembly::{
        assemble_structural_system_with_execution, solver_solid::SolverSolidAssemblyError,
        AssemblySummary, LinearAssemblyError, StructuralFormulationRequirement,
    },
    constraints::{CanonicalStructuralConstraintPlan, CanonicalStructuralConstraintResult},
    contracts::{
        ComputeBackend, FeaRunError, FeaRunResult, LinearStaticSolveOptions,
        FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_EQUATION_SCALE,
        FEA_FIELD_STRUCTURAL_REACTION_MOMENT, FEA_FIELD_STRUCTURAL_RESIDUAL_NORM,
        FEA_FIELD_STRUCTURAL_STRESS, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
    },
    diagnostics::{
        builders::{extend_common_run_diagnostics, CommonRunDiagnosticInputs},
        FeaDiagnostic, FeaDiagnosticSeverity,
    },
    excitation::{
        realize_step_excitations, ExcitationSamplePoint, RealizedExcitationKind,
        RealizedExcitations,
    },
    pipeline::electro_thermal::recover_electro_thermal_fields,
    post::{
        fields::{
            recover_result_fields, structural_field_recovery_metrics,
            take_structural_observable_fields,
        },
        recover_connector_observable_fields, recover_constraint_group_observable_fields,
    },
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
    results::{
        field::{RecoveredFieldCatalog, StructuralRecoveredFieldTopology},
        observation::{evaluate_structural_observations, ObservationLimits},
    },
};
pub use prepared::{
    run_linear_static_with_prepared_constraint_plan,
    run_linear_static_with_prepared_constraint_plan_and_linearization, LinearStaticExecution,
    StaticResourceAdmission,
};
pub use thermal_state::{run_linear_static_from_thermal_state, ThermoelasticStaticResult};

const VECTOR_COMPONENT_COUNT: usize = 3;

pub fn run_linear_static(
    model: &AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaRunResult, FeaRunError> {
    run_linear_static_with_options(model, backend, LinearStaticSolveOptions::default())
}

pub fn run_linear_static_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: LinearStaticSolveOptions,
) -> Result<FeaRunResult, FeaRunError> {
    run_linear_static_with_constraint_plan(model, backend, options, None, None, None)
        .map(|result| result.run)
}

struct LinearStaticComputation {
    thermal_transition: Option<crate::execution::CanonicalStateTransition>,
    run: FeaRunResult,
    summary: AssemblySummary,
    solution: Vec<f64>,
}

fn run_linear_static_with_constraint_plan(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: LinearStaticSolveOptions,
    constraint_plan: Option<&CanonicalStructuralConstraintPlan>,
    admitted: Option<StaticResourceAdmission<'_>>,
    thermal_source: Option<&crate::execution::CanonicalSemanticState>,
) -> Result<LinearStaticComputation, FeaRunError> {
    let execution_step = super::admission::require_structural(
        model,
        AnalysisProcedure::Static,
        options.formulation_requirement,
    )?;
    if thermal_source.is_none()
        && execution_step
            .state_inputs
            .iter()
            .any(|input| input.required_kind == runmat_analysis_core::StateKind::Thermal)
    {
        return Err(FeaRunError::InvalidModel(
            "static step declares a thermal state input; its retained source must be supplied"
                .into(),
        ));
    }
    emit_phase(
        "fea.run_linear_static",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Started,
        "validating FEA model",
        Some(0),
        Some(5),
    );
    check_cancelled("fea.run_linear_static")?;
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;
    emit_phase(
        "fea.run_linear_static",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Completed,
        "FEA model validation complete",
        Some(1),
        Some(5),
    );
    check_cancelled("fea.run_linear_static")?;

    let preassembly = preflight::prepare_static_preassembly(
        model,
        execution_step,
        backend,
        &options,
        constraint_plan,
        admitted,
    )?;
    let assembly_constraint_plan = constraint_plan.or(preassembly.constraints.as_ref());

    let excitations =
        realize_step_excitations(model, execution_step, ExcitationSamplePoint::Static)
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let thermal_transfer = thermal_source
        .map(|source| {
            let mesh = options.solver_mesh.as_ref().ok_or_else(|| {
                FeaRunError::InvalidModel(
                    "thermal state consumption requires a canonical solver mesh".into(),
                )
            })?;
            crate::execution::BoundThermalStateTransfer::bind(
                model,
                &execution_step.step_id,
                source,
                mesh,
            )
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))
        })
        .transpose()?;
    emit_phase(
        "fea.run_linear_static",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Started,
        "assembling linear system",
        Some(1),
        Some(5),
    );
    let summary = assemble_structural_system_with_execution(
        model,
        &excitations,
        options.solver_mesh.clone(),
        if options.solver_mesh.is_some() || preflight::model_has_explicit_structural_elements(model)
        {
            StructuralFormulationRequirement::PhysicalDiscretization
        } else {
            StructuralFormulationRequirement::ExperimentalSyntheticOperator
        },
        assembly_constraint_plan,
        options.local_execution,
        thermal_transfer
            .as_ref()
            .map(|transfer| transfer.nodal_temperature_k.as_slice()),
    )
    .map_err(|err| match err {
        LinearAssemblyError::SolidOperator(SolverSolidAssemblyError::Cancelled)
        | LinearAssemblyError::Structural(crate::assembly::StructuralAssemblyError::Cancelled) => {
            FeaRunError::Cancelled
        }
        other => FeaRunError::Assembly(other.to_string()),
    })?;
    // Element recovery now owns the temperatures it needs. Release the aligned global
    // field before solving; only the canonical transition must survive into the result.
    let thermal_transition = thermal_transfer.map(|transfer| transfer.transition);
    preflight::verify_executed_preassembly(
        model,
        &options,
        assembly_constraint_plan,
        &preassembly,
        &summary,
    )?;
    let identity = super::admission::admit_structural(
        &summary,
        execution_step,
        options.formulation_requirement,
    )?;
    super::validate_rotational_dof_targets(model, &summary)?;
    if options.require_solver_mesh_for_solid
        && options.solver_mesh.is_none()
        && !preflight::model_has_explicit_structural_elements(model)
    {
        return Err(FeaRunError::InvalidModel(
            "linear static solid continuum studies require a canonical solver mesh".to_string(),
        ));
    }
    let compiled_constraint_plan;
    let executed_constraint_plan = match assembly_constraint_plan {
        Some(plan) => plan,
        None => {
            compiled_constraint_plan = CanonicalStructuralConstraintPlan::compile_for_model(
                model,
                options.solver_mesh.as_ref(),
            )
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
            &compiled_constraint_plan
        }
    };
    emit_phase(
        "fea.run_linear_static",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Completed,
        "linear system assembly complete",
        Some(2),
        Some(5),
    );
    check_cancelled("fea.run_linear_static")?;
    emit_phase(
        "fea.run_linear_static",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Started,
        "solving linear system",
        Some(2),
        Some(5),
    );
    let solve_result =
        solve::solve_static_system(&summary, &options, preassembly.solver_admission.as_ref())?;
    let static_balance = solve::recover_static_balance(&summary, &solve_result.solution, &options)?;
    let structural_constraint_result =
        CanonicalStructuralConstraintResult::recover_for_model_from_solver_state(
            executed_constraint_plan,
            model,
            options.solver_mesh.as_ref(),
            &solve_result.solution,
            &static_balance.reaction,
        )
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    structural_constraint_result
        .validate_against_model(
            executed_constraint_plan,
            model,
            options.solver_mesh.as_ref(),
        )
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let structural_connector_result =
        crate::connectors::CanonicalStructuralConnectorResult::recover_static(
            match summary.structural_connector_plan.clone() {
                Some(plan) => plan,
                None if model.connectors.is_empty() => {
                    crate::connectors::CanonicalStructuralConnectorPlan::empty(
                        executed_constraint_plan.topology,
                        summary.structural_node_count,
                    )
                    .map_err(|error| FeaRunError::Assembly(error.to_string()))?
                }
                None => {
                    return Err(FeaRunError::Assembly(
                        "structural assembly did not publish a canonical connector plan".into(),
                    ))
                }
            },
            &summary.structural_dof_layout,
            &solve_result.solution,
        )
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    structural_connector_result
        .plan
        .validate_against_authored(&model.connectors)
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    emit_phase(
        "fea.run_linear_static",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Completed,
        "linear solve complete",
        Some(3),
        Some(5),
    );
    check_cancelled("fea.run_linear_static")?;

    emit_phase(
        "fea.run_linear_static",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Started,
        "recovering result fields",
        Some(3),
        Some(5),
    );
    let mut fields = recover_result_fields(&summary, &solve_result, options.local_execution)
        .map_err(|error| match error {
            crate::post::StructuralFieldRecoveryError::Cancelled => FeaRunError::Cancelled,
            other => FeaRunError::Assembly(other.to_string()),
        })?;
    let thermo_mechanical_fields =
        static_thermomechanical::recover(&summary, &fields, solve_result.residual_norm)?;
    fields.extend(thermo_mechanical_fields.temperature_snapshots);
    fields.extend(thermo_mechanical_fields.thermal_strain_snapshots);
    fields.extend(thermo_mechanical_fields.thermal_stress_snapshots);
    fields.extend(thermo_mechanical_fields.displacement_snapshots);
    fields.extend(thermo_mechanical_fields.von_mises_snapshots);
    fields.extend(thermo_mechanical_fields.coupling_residual_snapshots);
    let thermo_mechanical_diagnostics = thermo_mechanical_fields.diagnostics;
    let electro_thermal_fields = recover_electro_thermal_fields(
        summary.electro_thermal.as_ref(),
        &[1.0],
        &[solve_result.residual_norm],
        summary.dof_count,
        options.solver_mesh.as_ref(),
    );
    let electro_thermal_diagnostics = electro_thermal_fields.diagnostics;
    fields.extend(electro_thermal_fields.static_fields);
    fields.extend(electro_thermal_fields.temperature_snapshots);
    fields.extend(electro_thermal_fields.thermal_residual_snapshots);
    let observations = if execution_step.observations.is_empty() {
        Vec::new()
    } else {
        let mesh = options.solver_mesh.as_ref();
        let mut recovered = take_structural_observable_fields(&mut fields)
            .map_err(|error| FeaRunError::Observation(error.to_string()))?;
        recovered.extend(
            recover_constraint_group_observable_fields(&structural_constraint_result)
                .map_err(|error| FeaRunError::Observation(error.to_string()))?,
        );
        recovered.extend(
            recover_connector_observable_fields(&structural_connector_result)
                .map_err(|error| FeaRunError::Observation(error.to_string()))?,
        );
        let mut topology = StructuralRecoveredFieldTopology::derive(model, mesh)
            .map_err(|error| FeaRunError::Observation(error.to_string()))?;
        topology.entity_counts.constraint_groups = structural_constraint_result.groups.len();
        topology.entity_counts.connectors = structural_connector_result.responses.len();
        let catalog = RecoveredFieldCatalog::new_for_topology(
            topology.identity.discretization_digest(),
            topology.entity_counts,
            recovered,
        )
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
        let definitions = catalog
            .fields()
            .iter()
            .map(|field| field.definition().clone())
            .collect::<Vec<_>>();
        let plan =
            crate::results::observation::resolve_structural_observation_plan_with_connectors(
                execution_step,
                model,
                mesh,
                executed_constraint_plan,
                &structural_connector_result.plan,
                &definitions,
                ObservationLimits::default(),
            )
            .map_err(|error| FeaRunError::Observation(error.to_string()))?;
        let observations = evaluate_structural_observations(&plan, model, mesh, &catalog)
            .map_err(|error| FeaRunError::Observation(error.to_string()))?;
        fields.extend(
            catalog
                .into_fields()
                .into_iter()
                .map(|field| field.into_storage()),
        );
        observations
    };
    emit_phase(
        "fea.run_linear_static",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Completed,
        "result field recovery complete",
        Some(4),
        Some(5),
    );
    check_cancelled("fea.run_linear_static")?;
    let solver_device_apply_k_ratio = if solve_result.device_apply_k_attempt_count == 0 {
        0.0
    } else {
        solve_result.device_apply_k_count as f64 / solve_result.device_apply_k_attempt_count as f64
    };

    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_CONVERGENCE".to_string(),
        severity: if solve_result.converged {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "iterations={} residual_norm={} converged={} solver_method={} preconditioner={}",
            solve_result.iterations,
            solve_result.residual_norm,
            solve_result.converged,
            solve_result.solver_method,
            solve_result.preconditioner,
        ),
    }];
    if let Some(path) = options.solver_mesh_artifact_path.as_ref() {
        let mesh_summary = options
            .solver_mesh
            .as_ref()
            .map(|mesh| {
                format!(
                    " solver_mesh_node_count={} solver_mesh_volume_element_count={} solver_mesh_boundary_face_count={}",
                    mesh.topology.nodes.len(),
                    mesh.topology.volume_elements.len(),
                    mesh.topology.boundary_faces.len()
                )
            })
            .unwrap_or_default();
        diagnostics.push(FeaDiagnostic {
            code: "FEA_SOLVER_MESH_REFERENCE".to_string(),
            severity: FeaDiagnosticSeverity::Info,
            message: format!("solver_mesh_artifact_path={path}{mesh_summary}"),
        });
    }
    if let Some(diagnostic) = static_diagnostics::solid_assembly(&summary) {
        diagnostics.push(diagnostic);
    }
    if let (Some(residual_norm), Some(equation_scale)) = (
        scalar_field_value(&fields, FEA_FIELD_STRUCTURAL_RESIDUAL_NORM),
        scalar_field_value(&fields, FEA_FIELD_STRUCTURAL_EQUATION_SCALE),
    ) {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_STRUCTURAL_RESIDUAL".to_string(),
            severity: if residual_norm <= 1.0e-6 {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!(
                "normalized_residual_norm={} equation_scale={}",
                residual_norm, equation_scale
            ),
        });
    }
    if let Some(total_strain_energy) =
        scalar_field_value(&fields, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
    {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_STRUCTURAL_ENERGY".to_string(),
            severity: if total_strain_energy.is_finite() && total_strain_energy >= 0.0 {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!("total_strain_energy={total_strain_energy}"),
        });
        diagnostics.push(static_diagnostics::known_answer(
            total_strain_energy,
            &static_balance.evidence,
        ));
        if let Some(reference_diagnostic) = structural_reference_kinematics_diagnostic(
            model,
            &summary,
            &solve_result.solution,
            &fields,
        ) {
            diagnostics.push(reference_diagnostic);
        }
    }
    let recovery_metrics = structural_field_recovery_metrics(&summary, &solve_result.solution)
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    diagnostics.push(FeaDiagnostic {
        code: "FEA_STRUCTURAL_FIELD_RECOVERY".to_string(),
        severity: if recovery_metrics.recovery_element_count > 0
            && (recovery_metrics.active_stiffness_edge_count > 0
                || recovery_metrics.solver_mesh_element_count > 0)
        {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "basis={} active_stiffness_edge_count={} constrained_edge_count={} recovery_element_count={} solver_mesh_node_count={} solver_mesh_element_count={} max_edge_displacement_jump={} max_edge_strain_norm={} mean_edge_stiffness_ratio={} mean_edge_length_m={} strain_component_coverage_ratio={} element_geometry_node_count={} element_geometry_edge_count={} element_geometry_coverage_ratio={}",
            recovery_metrics.basis,
            recovery_metrics.active_stiffness_edge_count,
            recovery_metrics.constrained_edge_count,
            recovery_metrics.recovery_element_count,
            recovery_metrics.solver_mesh_node_count,
            recovery_metrics.solver_mesh_element_count,
            recovery_metrics.max_edge_displacement_jump,
            recovery_metrics.max_edge_strain_norm,
            recovery_metrics.mean_edge_stiffness_ratio,
            recovery_metrics.mean_edge_length_m,
            recovery_metrics.strain_component_coverage_ratio,
            recovery_metrics.element_geometry_node_count,
            recovery_metrics.element_geometry_edge_count,
            recovery_metrics.element_geometry_coverage_ratio
        ),
    });
    if let Some(moment_balance) = structural_moment_balance_diagnostic(model, &excitations, &fields)
    {
        diagnostics.push(moment_balance);
    }
    if let Some(beam_closed_form) =
        structural_beam_closed_form_diagnostic(model, &excitations, &fields)
    {
        diagnostics.push(beam_closed_form);
    }
    extend_common_run_diagnostics(
        &mut diagnostics,
        CommonRunDiagnosticInputs {
            model,
            summary: &summary,
        },
    );
    diagnostics.extend(thermo_mechanical_diagnostics);
    diagnostics.extend(electro_thermal_diagnostics);
    diagnostics.extend(solve_result.diagnostics);

    emit_phase(
        "fea.run_linear_static",
        FeaProgressPhase::Complete,
        FeaProgressStatus::Completed,
        "FEA linear static run complete",
        Some(5),
        Some(5),
    );

    let run = FeaRunResult {
        identity,
        requested_backend: backend,
        execution_backend: crate::contracts::ExecutionBackend::from_solver_backend(
            solve_result.solver_backend,
        ),
        solver_backend: solve_result.solver_backend,
        solver_device_apply_k_ratio,
        solver_method: solve_result.solver_method,
        preconditioner: solve_result.preconditioner,
        solver_host_sync_count: solve_result.host_sync_count,
        accelerator_evidence: solve_result.accelerator_evidence.into_iter().collect(),
        diagnostics,
        spatial_projection_evidence: summary.spatial_projection_evidence.clone(),
        structural_static_balance: Some(Box::new(static_balance.evidence)),
        structural_constraint_result: Some(Box::new(structural_constraint_result)),
        structural_connector_result: Some(Box::new(structural_connector_result)),
        observations,
        observation_histories: Vec::new(),
        series_reductions: Vec::new(),
        fields,
    };
    run.validate_solver_formulation()?;
    Ok(LinearStaticComputation {
        thermal_transition,
        run,
        summary,
        solution: solve_result.solution,
    })
}

fn field_snapshot(fields: &[AnalysisField], field_id: &str) -> Vec<AnalysisField> {
    fields
        .iter()
        .find(|field| field.field_id == field_id)
        .cloned()
        .into_iter()
        .collect()
}

fn element_count_for_dofs(dof_count: usize) -> usize {
    dof_count.div_ceil(VECTOR_COMPONENT_COUNT).max(1)
}
