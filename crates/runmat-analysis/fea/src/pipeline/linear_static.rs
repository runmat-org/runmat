use runmat_analysis_core::{
    validate_model, AnalysisField, AnalysisModel, AnalysisProcedure, StructuralConstraintKind,
    StructuralOperatorImplementation,
};
use runmat_execution::Digest;

use crate::{
    algebra::apply_k,
    assembly::{
        assemble_structural_system, assemble_structural_system_with_constraint_plan,
        AssemblySummary, StructuralFormulationRequirement,
    },
    constraints::CanonicalStructuralConstraintPlan,
    contracts::{
        ComputeBackend, FeaRunError, FeaRunResult, LinearStaticSolveOptions,
        FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_EQUATION_SCALE,
        FEA_FIELD_STRUCTURAL_REACTION_MOMENT, FEA_FIELD_STRUCTURAL_RESIDUAL_NORM,
        FEA_FIELD_STRUCTURAL_STRESS, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
        FEA_FIELD_STRUCTURAL_VON_MISES,
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
    pipeline::thermo_mechanical::recover_thermo_mechanical_snapshots,
    post::{
        fields::{
            recover_result_fields, structural_field_recovery_metrics,
            take_structural_observable_fields,
        },
        recover_constraint_group_observable_fields,
    },
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
    results::{
        field::{RecoveredFieldCatalog, StructuralRecoveredFieldTopology},
        observation::{
            evaluate_structural_observations, resolve_structural_observation_plan,
            ObservationLimits,
        },
    },
    solve::{backend::build_backend, linear::solve_linear_system},
};

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
    run_linear_static_with_constraint_plan(model, backend, options, None).map(|result| result.run)
}

pub struct LinearStaticExecution {
    pub run: FeaRunResult,
    pub linearization: super::structural_linearization::StructuralLinearizationArtifact,
}

/// Executes a linear-static solve with a topology-resolved constraint plan supplied by the
/// portable execution boundary. Inline callers use [`run_linear_static_with_options`], which
/// compiles the same plan before assembly.
pub fn run_linear_static_with_prepared_constraint_plan(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: LinearStaticSolveOptions,
    constraint_plan: &CanonicalStructuralConstraintPlan,
) -> Result<FeaRunResult, FeaRunError> {
    run_linear_static_with_constraint_plan(model, backend, options, Some(constraint_plan))
        .map(|result| result.run)
}

/// Executes the preload solve and publishes the exact initial-stress tangent from the same
/// assembly and solution. This prevents downstream stability procedures from reconstructing or
/// relabeling a base state.
pub fn run_linear_static_with_prepared_constraint_plan_and_linearization(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: LinearStaticSolveOptions,
    constraint_plan: &CanonicalStructuralConstraintPlan,
    topology_digest: Digest,
) -> Result<LinearStaticExecution, FeaRunError> {
    let computation =
        run_linear_static_with_constraint_plan(model, backend, options, Some(constraint_plan))?;
    let linearization =
        super::structural_linearization::StructuralLinearizationArtifact::from_solution(
            &computation.summary,
            computation.run.identity.clone(),
            topology_digest,
            &computation.solution,
        )
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    Ok(LinearStaticExecution {
        run: computation.run,
        linearization,
    })
}

struct LinearStaticComputation {
    run: FeaRunResult,
    summary: AssemblySummary,
    solution: Vec<f64>,
}

fn run_linear_static_with_constraint_plan(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: LinearStaticSolveOptions,
    constraint_plan: Option<&CanonicalStructuralConstraintPlan>,
) -> Result<LinearStaticComputation, FeaRunError> {
    let execution_step = super::admission::require_structural(
        model,
        AnalysisProcedure::Static,
        options.formulation_requirement,
    )?;
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

    let excitations =
        realize_step_excitations(model, execution_step, ExcitationSamplePoint::Static)
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;

    emit_phase(
        "fea.run_linear_static",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Started,
        "assembling linear system",
        Some(1),
        Some(5),
    );
    let summary = match constraint_plan {
        Some(plan) => assemble_structural_system_with_constraint_plan(
            model,
            &excitations,
            options.solver_mesh.clone(),
            StructuralFormulationRequirement::ExperimentalSyntheticOperator,
            plan,
        ),
        None => assemble_structural_system(
            model,
            &excitations,
            options.solver_mesh.clone(),
            StructuralFormulationRequirement::ExperimentalSyntheticOperator,
        ),
    }
    .map_err(|err| FeaRunError::Assembly(err.to_string()))?;
    let identity = super::admission::admit_structural(
        &summary,
        execution_step,
        options.formulation_requirement,
    )?;
    super::validate_rotational_dof_targets(model, &summary)?;
    let solver_mesh_present = options.solver_mesh.is_some();
    if options.require_solver_mesh_for_solid
        && !solver_mesh_present
        && !model_has_explicit_structural_elements(model)
    {
        return Err(FeaRunError::InvalidModel(
            "linear static solid continuum studies require a canonical solver mesh".to_string(),
        ));
    }
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
    let algebra_backend = build_backend(options.algebra_backend_kind);
    if options.initial_displacement.as_ref().is_some_and(|values| {
        values.len() != summary.dof_count || values.iter().any(|value| !value.is_finite())
    }) {
        return Err(FeaRunError::InvalidModel(format!(
            "initial displacement state must contain exactly {} finite solver DOFs",
            summary.dof_count
        )));
    }
    let solve_result = solve_linear_system(
        &summary,
        options.preconditioner_kind,
        options.algebra_backend_kind,
        algebra_backend.as_ref(),
        options.initial_displacement.as_deref(),
    );
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
    let mut fields = recover_result_fields(&summary, &solve_result)
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let thermo_mechanical_fields = recover_thermo_mechanical_snapshots(
        summary.thermo_mechanical.as_ref(),
        &[1.0],
        &field_snapshot(&fields, FEA_FIELD_STRUCTURAL_DISPLACEMENT),
        &field_snapshot(&fields, FEA_FIELD_STRUCTURAL_VON_MISES),
        &[solve_result.residual_norm],
        element_count_for_dofs(summary.dof_count),
    );
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
        let mesh = options.solver_mesh.as_ref().ok_or_else(|| {
            FeaRunError::Observation(
                "spatial observations require the canonical solver mesh used by assembly".into(),
            )
        })?;
        let compiled_constraint_plan;
        let observation_constraint_plan = if let Some(plan) = constraint_plan {
            plan
        } else {
            compiled_constraint_plan =
                CanonicalStructuralConstraintPlan::compile_for_model(model, Some(mesh))
                    .map_err(|error| FeaRunError::Observation(error.to_string()))?;
            &compiled_constraint_plan
        };
        let constraint_result =
            crate::constraints::CanonicalStructuralConstraintResult::recover_for_model(
                observation_constraint_plan,
                model,
                Some(mesh),
                &fields,
            )
            .map_err(|error| FeaRunError::Observation(error.to_string()))?;
        let mut recovered = take_structural_observable_fields(&mut fields)
            .map_err(|error| FeaRunError::Observation(error.to_string()))?;
        recovered.extend(
            recover_constraint_group_observable_fields(&constraint_result)
                .map_err(|error| FeaRunError::Observation(error.to_string()))?,
        );
        let mut topology = StructuralRecoveredFieldTopology::derive(model, Some(mesh))
            .map_err(|error| FeaRunError::Observation(error.to_string()))?;
        topology.entity_counts.constraint_groups = constraint_result.groups.len();
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
        let plan = resolve_structural_observation_plan(
            execution_step,
            model,
            Some(mesh),
            observation_constraint_plan,
            &definitions,
            ObservationLimits::default(),
        )
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
        let observations = evaluate_structural_observations(&plan, model, Some(mesh), &catalog)
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
    if let Some(diagnostic) = structural_solid_assembly_diagnostic(&summary) {
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
        diagnostics.push(structural_linear_known_answer_diagnostic(
            total_strain_energy,
            summary.operator.load(),
            &solve_result.solution,
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
        observations,
        observation_histories: Vec::new(),
        series_reductions: Vec::new(),
        fields,
    };
    Ok(LinearStaticComputation {
        run,
        summary,
        solution: solve_result.solution,
    })
}

fn model_has_explicit_structural_elements(model: &AnalysisModel) -> bool {
    model
        .structural
        .as_ref()
        .is_some_and(|structural| !structural.elements.is_empty())
}

fn structural_solid_assembly_diagnostic(summary: &AssemblySummary) -> Option<FeaDiagnostic> {
    if summary.structural_solid_element_count == 0 {
        return None;
    }

    let stiffness_nnz = summary.operator.operators().stiffness().values().len();
    let rhs_nonzero_count = summary
        .operator
        .load()
        .iter()
        .filter(|value| value.abs() > 0.0)
        .count();
    let has_solid_stiffness = stiffness_nnz > 0 && summary.structural_solid_element_count > 0;
    let has_solid_mass = !summary.operator.operators().mass().values().is_empty()
        && summary.structural_solid_total_mass_kg.is_finite()
        && summary.structural_solid_total_mass_kg > 0.0;
    let basis = match summary
        .structural_formulations
        .iter()
        .map(|value| value.operator)
        .find(|operator| {
            matches!(
                operator,
                StructuralOperatorImplementation::Tet4LinearElasticity
                    | StructuralOperatorImplementation::Tet10LinearElasticity
            )
        }) {
        Some(StructuralOperatorImplementation::Tet4LinearElasticity) => "tet4_linear_elasticity",
        Some(StructuralOperatorImplementation::Tet10LinearElasticity) => "tet10_linear_elasticity",
        _ => "non_solid",
    };
    let storage = "sparse_csr";

    Some(FeaDiagnostic {
        code: "FEA_STRUCTURAL_SOLID_ASSEMBLY".to_string(),
        severity: if has_solid_stiffness && has_solid_mass {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "basis={} stiffness_storage={} mass_storage={} solid_node_count={} solid_element_count={} solid_total_mass_kg={} dof_count={} constrained_dof_count={} stiffness_nnz={} rhs_nonzero_count={} stiffness_present={} mass_present={}",
            basis,
            storage,
            "sparse_csr",
            summary.structural_node_count,
            summary.structural_solid_element_count,
            summary.structural_solid_total_mass_kg,
            summary.dof_count,
            summary.constrained_dof_count,
            stiffness_nnz,
            rhs_nonzero_count,
            has_solid_stiffness,
            has_solid_mass,
        ),
    })
}

fn structural_linear_known_answer_diagnostic(
    total_strain_energy: f64,
    rhs: &[f64],
    displacement: &[f64],
) -> FeaDiagnostic {
    let external_work = rhs
        .iter()
        .zip(displacement.iter())
        .map(|(force, displacement)| force * displacement)
        .sum::<f64>();
    let work_energy_ratio = if external_work.abs() > 1.0e-12 {
        (2.0 * total_strain_energy) / external_work
    } else if total_strain_energy.abs() <= 1.0e-12 {
        1.0
    } else {
        0.0
    };
    let work_energy_residual_ratio = (work_energy_ratio - 1.0).abs();
    let known_answer_coverage_ratio = if rhs.len() == displacement.len()
        && !rhs.is_empty()
        && total_strain_energy.is_finite()
        && external_work.is_finite()
    {
        1.0
    } else {
        0.0
    };
    let severity = if known_answer_coverage_ratio >= 1.0
        && total_strain_energy >= 0.0
        && work_energy_residual_ratio <= 1.0e-8
    {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };

    FeaDiagnostic {
        code: "FEA_STRUCTURAL_LINEAR_KNOWN_ANSWER".to_string(),
        severity,
        message: format!(
            "identity=linear_work_energy external_work={} total_strain_energy={} work_energy_ratio={} work_energy_residual_ratio={} known_answer_coverage_ratio={}",
            external_work,
            total_strain_energy,
            work_energy_ratio,
            work_energy_residual_ratio,
            known_answer_coverage_ratio
        ),
    }
}

fn structural_reference_kinematics_diagnostic(
    model: &AnalysisModel,
    summary: &crate::assembly::AssemblySummary,
    solution: &[f64],
    fields: &[AnalysisField],
) -> Option<FeaDiagnostic> {
    let (case, primary_component, primary_label) = match model.model_id.0.as_str() {
        "structural_axial_bar_reference" => ("axial_bar_tension", 0, "x"),
        "structural_beam_bending_reference" => ("beam_transverse_bending", 1, "y"),
        _ => return None,
    };
    let displacement = field_values(fields, FEA_FIELD_STRUCTURAL_DISPLACEMENT)?;
    let stress = field_values(fields, FEA_FIELD_STRUCTURAL_STRESS)?;
    let primary_displacement =
        vector_component_peak(displacement, VECTOR_COMPONENT_COUNT, primary_component);
    let transverse_displacement = (0..VECTOR_COMPONENT_COUNT)
        .filter(|component| *component != primary_component)
        .map(|component| vector_component_peak(displacement, VECTOR_COMPONENT_COUNT, component))
        .fold(0.0_f64, f64::max);
    let transverse_displacement_leakage_ratio =
        transverse_displacement / primary_displacement.max(1.0e-18);
    let primary_stress = tensor_component_peak(stress, primary_component);
    let max_stress = stress
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let primary_stress_component_ratio = primary_stress / max_stress.max(1.0e-18);
    let applied = apply_k(&summary.operator, solution);
    let free_residual_norm = applied
        .iter()
        .zip(summary.operator.load())
        .zip(summary.operator.constrained_dofs())
        .filter(|(_, constrained)| !**constrained)
        .map(|((lhs, rhs), _)| (lhs - rhs) * (lhs - rhs))
        .sum::<f64>()
        .sqrt();
    let load_norm = summary
        .operator
        .load()
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
        .max(1.0);
    let normalized_free_residual = free_residual_norm / load_norm;
    let directional_reference_coverage_ratio = if primary_displacement.is_finite()
        && primary_displacement > 0.0
        && transverse_displacement_leakage_ratio.is_finite()
        && primary_stress_component_ratio.is_finite()
        && primary_stress_component_ratio > 0.0
    {
        1.0
    } else {
        0.0
    };
    let severity = if directional_reference_coverage_ratio >= 1.0
        && transverse_displacement_leakage_ratio <= 0.1
        && primary_stress_component_ratio >= 0.5
        && normalized_free_residual <= 1.0e-7
    {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };

    Some(FeaDiagnostic {
        code: "FEA_STRUCTURAL_REFERENCE_KINEMATICS".to_string(),
        severity,
        message: format!(
            "case={} primary_component={} primary_displacement_m={} transverse_displacement_leakage_ratio={} primary_stress_component_ratio={} directional_reference_coverage_ratio={} normalized_free_equilibrium_residual={}",
            case,
            primary_label,
            primary_displacement,
            transverse_displacement_leakage_ratio,
            primary_stress_component_ratio,
            directional_reference_coverage_ratio,
            normalized_free_residual
        ),
    })
}

fn structural_beam_closed_form_diagnostic(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    fields: &[AnalysisField],
) -> Option<FeaDiagnostic> {
    let case = match model.model_id.0.as_str() {
        "structural_beam_cantilever_end_moment_reference" => "cantilever_end_moment",
        "structural_beam_torsion_reference" => "cantilever_torsion",
        "structural_beam_force_and_moment_reference" => "cantilever_force_and_moment",
        _ => return None,
    };
    let structural = model.structural.as_ref()?;
    if structural.nodes.len() != 2 {
        return None;
    }
    let section = structural.beam_sections.first()?;
    let material = model.materials.first()?;
    let youngs_modulus_pa = material.mechanical.youngs_modulus_pa;
    let shear_modulus_pa =
        youngs_modulus_pa / (2.0 * (1.0 + material.mechanical.poisson_ratio)).max(1.0e-9);
    let length_m = distance3(
        structural.nodes[0].coordinates_m,
        structural.nodes[1].coordinates_m,
    );
    if length_m <= 0.0 {
        return None;
    }

    let mut tip_force_y_n = 0.0_f64;
    let mut tip_torque_x_n_m = 0.0_f64;
    let mut tip_moment_z_n_m = 0.0_f64;
    for load in excitations.iter() {
        if load.target.solver_region_id() != Some("node:2") {
            continue;
        }
        match load.kind {
            RealizedExcitationKind::Force(value) => {
                tip_force_y_n += value.real[1];
            }
            RealizedExcitationKind::Moment(value) => {
                tip_torque_x_n_m += value.real[0];
                tip_moment_z_n_m += value.real[2];
            }
            _ => {}
        }
    }

    let eiz = youngs_modulus_pa * section.iz_m4;
    let gj = shear_modulus_pa * section.torsion_j_m4;
    if eiz <= 0.0 || gj <= 0.0 {
        return None;
    }

    let node_count = structural.nodes.len();
    let mut expected_displacement = vec![0.0_f64; node_count * VECTOR_COMPONENT_COUNT];
    let mut expected_rotation = vec![0.0_f64; node_count * VECTOR_COMPONENT_COUNT];
    let tip_offset = VECTOR_COMPONENT_COUNT;
    expected_displacement[tip_offset + 1] = tip_force_y_n * length_m.powi(3) / (3.0 * eiz)
        + tip_moment_z_n_m * length_m.powi(2) / (2.0 * eiz);
    expected_rotation[tip_offset] = tip_torque_x_n_m * length_m / gj;
    expected_rotation[tip_offset + 2] =
        tip_force_y_n * length_m.powi(2) / (2.0 * eiz) + tip_moment_z_n_m * length_m / eiz;

    let displacement = field_values(fields, FEA_FIELD_STRUCTURAL_DISPLACEMENT)?;
    let rotation = field_values(fields, crate::contracts::FEA_FIELD_STRUCTURAL_ROTATION)?;
    let displacement_error_ratio = vector_relative_error(displacement, &expected_displacement);
    let rotation_error_ratio = vector_relative_error(rotation, &expected_rotation);
    let torsion_error_ratio = if tip_torque_x_n_m.abs() > 0.0 {
        let observed = rotation.get(tip_offset).copied().unwrap_or(0.0);
        let expected = expected_rotation[tip_offset];
        (observed - expected).abs() / expected.abs().max(1.0e-18)
    } else {
        0.0
    };
    let coverage_ratio = if displacement_error_ratio.is_finite()
        && rotation_error_ratio.is_finite()
        && torsion_error_ratio.is_finite()
    {
        1.0
    } else {
        0.0
    };
    let severity = if coverage_ratio >= 1.0
        && displacement_error_ratio <= 1.0e-6
        && rotation_error_ratio <= 1.0e-6
        && torsion_error_ratio <= 1.0e-6
    {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };

    Some(FeaDiagnostic {
        code: "FEA_STRUCTURAL_BEAM_CLOSED_FORM".to_string(),
        severity,
        message: format!(
            "case={} structural_beam_closed_form_rotation_error_ratio={} structural_beam_closed_form_displacement_error_ratio={} structural_beam_closed_form_torsion_error_ratio={} structural_beam_closed_form_coverage_ratio={} length_m={} tip_force_y_n={} tip_torque_x_n_m={} tip_moment_z_n_m={}",
            case,
            rotation_error_ratio,
            displacement_error_ratio,
            torsion_error_ratio,
            coverage_ratio,
            length_m,
            tip_force_y_n,
            tip_torque_x_n_m,
            tip_moment_z_n_m
        ),
    })
}

fn structural_moment_balance_diagnostic(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    fields: &[AnalysisField],
) -> Option<FeaDiagnostic> {
    let requested = requested_moment_vector(model, excitations);
    let requested_norm = vector3_norm(requested);
    if requested_norm <= 0.0 {
        return None;
    }
    let reaction_values = field_values(fields, FEA_FIELD_STRUCTURAL_REACTION_MOMENT)?;

    let reaction = vector3_sum(reaction_values);
    let reaction_norm = vector3_norm(reaction);
    let realized_norm = reaction_norm;
    let balance = [
        reaction[0] + requested[0],
        reaction[1] + requested[1],
        reaction[2] + requested[2],
    ];
    let balance_residual_ratio = vector3_norm(balance) / requested_norm.max(1.0e-18);
    let realization_ratio = realized_norm / requested_norm.max(1.0e-18);

    Some(FeaDiagnostic {
        code: "FEA_STRUCTURAL_MOMENT_BALANCE".to_string(),
        severity: if balance_residual_ratio <= 1.0e-6 && realization_ratio.is_finite() {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "structural_reaction_moment_norm_n_m={} structural_moment_requested_norm_n_m={} structural_moment_realized_norm_n_m={} structural_moment_realization_ratio={} structural_moment_balance_residual_ratio={}",
            reaction_norm,
            requested_norm,
            realized_norm,
            realization_ratio,
            balance_residual_ratio
        ),
    })
}

fn requested_moment_vector(model: &AnalysisModel, excitations: &RealizedExcitations) -> [f64; 3] {
    let mut out = [0.0_f64; 3];
    for load in excitations.iter() {
        match load.kind {
            RealizedExcitationKind::Moment(value) => {
                for (total, value) in out.iter_mut().zip(value.real) {
                    *total += value;
                }
            }
            RealizedExcitationKind::Force(value) => {
                let Some(region_id) = load.target.solver_region_id() else {
                    continue;
                };
                if let Some(force_moment) =
                    structural_force_moment_about_reaction_origin(model, region_id, value.real)
                {
                    out[0] += force_moment[0];
                    out[1] += force_moment[1];
                    out[2] += force_moment[2];
                }
            }
            _ => {}
        }
    }
    out
}

fn structural_force_moment_about_reaction_origin(
    model: &AnalysisModel,
    region_id: &str,
    force: [f64; 3],
) -> Option<[f64; 3]> {
    let structural = model.structural.as_ref()?;
    let origin = model
        .constraints
        .iter()
        .filter_map(|constraint| constraint.structural())
        .find_map(|constraint| {
            if !matches!(constraint.kind, StructuralConstraintKind::Fixed { .. }) {
                return None;
            }
            let target = constraint.kind.target()?;
            structural_node_coordinates(structural, &target.region_id)
        })
        .or_else(|| structural.nodes.first().map(|node| node.coordinates_m))?;
    let target = structural_node_coordinates(structural, region_id)?;
    let r = [
        target[0] - origin[0],
        target[1] - origin[1],
        target[2] - origin[2],
    ];
    Some([
        r[1] * force[2] - r[2] * force[1],
        r[2] * force[0] - r[0] * force[2],
        r[0] * force[1] - r[1] * force[0],
    ])
}

fn structural_node_coordinates(
    structural: &runmat_analysis_core::StructuralModel,
    region_id: &str,
) -> Option<[f64; 3]> {
    let node_id = region_id.strip_prefix("node:")?.parse::<u32>().ok()?;
    structural
        .nodes
        .iter()
        .find(|node| node.node_id == node_id)
        .map(|node| node.coordinates_m)
}

fn vector3_sum(values: &[f64]) -> [f64; 3] {
    let mut out = [0.0_f64; 3];
    for components in values.chunks_exact(VECTOR_COMPONENT_COUNT) {
        out[0] += components[0];
        out[1] += components[1];
        out[2] += components[2];
    }
    out
}

fn vector3_norm(values: [f64; 3]) -> f64 {
    (values[0] * values[0] + values[1] * values[1] + values[2] * values[2]).sqrt()
}

fn distance3(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt()
}

fn vector_relative_error(observed: &[f64], expected: &[f64]) -> f64 {
    let count = observed.len().max(expected.len());
    if count == 0 {
        return 0.0;
    }
    let mut max_error = 0.0_f64;
    let mut max_expected = 0.0_f64;
    for index in 0..count {
        let observed_value = observed.get(index).copied().unwrap_or(0.0);
        let expected_value = expected.get(index).copied().unwrap_or(0.0);
        max_error = max_error.max((observed_value - expected_value).abs());
        max_expected = max_expected.max(expected_value.abs());
    }
    max_error / max_expected.max(1.0e-18)
}

fn field_values<'a>(fields: &'a [AnalysisField], field_id: &str) -> Option<&'a [f64]> {
    fields
        .iter()
        .find(|field| field.field_id == field_id)
        .and_then(AnalysisField::as_host_f64)
}

fn vector_component_peak(values: &[f64], component_count: usize, component: usize) -> f64 {
    values
        .chunks_exact(component_count)
        .map(|components| components.get(component).copied().unwrap_or(0.0).abs())
        .fold(0.0_f64, f64::max)
}

fn tensor_component_peak(values: &[f64], component: usize) -> f64 {
    values
        .chunks_exact(6)
        .map(|components| components.get(component).copied().unwrap_or(0.0).abs())
        .fold(0.0_f64, f64::max)
}

fn scalar_field_value(
    fields: &[runmat_analysis_core::AnalysisField],
    field_id: &str,
) -> Option<f64> {
    fields
        .iter()
        .find(|field| field.field_id == field_id)
        .and_then(runmat_analysis_core::AnalysisField::as_host_f64)
        .and_then(|values| values.first().copied())
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
