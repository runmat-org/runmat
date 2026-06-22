use runmat_analysis_core::{
    validate_model, AnalysisField, AnalysisModel, BoundaryConditionKind, LoadKind,
};

use crate::{
    assembly::assemble_linear_system,
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
    pipeline::electro_thermal::recover_electro_thermal_fields,
    pipeline::thermo_mechanical::recover_thermo_mechanical_snapshots,
    post::fields::{
        recover_result_fields, recover_structural_stress_from_displacement,
        structural_field_recovery_metrics,
    },
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
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

    emit_phase(
        "fea.run_linear_static",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Started,
        "assembling linear system",
        Some(1),
        Some(5),
    );
    let summary = assemble_linear_system(
        model,
        options.prep_context.clone(),
        options.thermo_mechanical_context,
        options.electro_thermal_context,
    );
    super::reject_moment_loads_without_rotational_dofs(model, &summary)?;
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
    let solve_result = solve_linear_system(
        &summary,
        options.preconditioner_kind,
        options.algebra_backend_kind,
        algebra_backend.as_ref(),
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
    let mut fields = recover_result_fields(&summary, &solve_result);
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
    );
    let electro_thermal_diagnostics = electro_thermal_fields.diagnostics;
    fields.extend(electro_thermal_fields.static_fields);
    fields.extend(electro_thermal_fields.temperature_snapshots);
    fields.extend(electro_thermal_fields.thermal_residual_snapshots);
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
            &summary.operator.rhs,
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
    let recovery_metrics = structural_field_recovery_metrics(&summary, &solve_result.solution);
    diagnostics.push(FeaDiagnostic {
        code: "FEA_STRUCTURAL_FIELD_RECOVERY".to_string(),
        severity: if recovery_metrics.active_stiffness_edge_count > 0
            && recovery_metrics.recovery_element_count > 0
        {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "basis={} active_stiffness_edge_count={} prep_recovery_edge_count={} constrained_edge_count={} recovery_element_count={} max_edge_displacement_jump={} max_edge_strain_norm={} mean_edge_stiffness_ratio={} mean_edge_length_m={} strain_component_coverage_ratio={} element_geometry_node_count={} element_geometry_edge_count={} element_geometry_coverage_ratio={}",
            recovery_metrics.basis,
            recovery_metrics.active_stiffness_edge_count,
            recovery_metrics.prep_recovery_edge_count,
            recovery_metrics.constrained_edge_count,
            recovery_metrics.recovery_element_count,
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
    if let Some(moment_balance) = structural_moment_balance_diagnostic(model, &fields) {
        diagnostics.push(moment_balance);
    }
    if let Some(beam_closed_form) = structural_beam_closed_form_diagnostic(model, &fields) {
        diagnostics.push(beam_closed_form);
    }
    extend_common_run_diagnostics(
        &mut diagnostics,
        CommonRunDiagnosticInputs {
            model,
            summary: &summary,
            prep_context: options.prep_context.clone(),
            iteration_metric: solve_result.iterations as f64,
            residual_metric: solve_result.residual_norm,
            requested_preconditioner: options.preconditioner_kind.as_str(),
            effective_preconditioner: &solve_result.preconditioner,
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

    Ok(FeaRunResult {
        backend,
        solver_backend: solve_result.solver_backend,
        solver_device_apply_k_ratio,
        solver_method: solve_result.solver_method,
        preconditioner: solve_result.preconditioner,
        solver_host_sync_count: solve_result.host_sync_count,
        diagnostics,
        fields,
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
    let direct_reference = solve_tridiagonal_reference(summary)?;
    let displacement_error_ratio = vector_relative_error(solution, &direct_reference);
    let expected_stress = recover_structural_stress_from_displacement(summary, &direct_reference);
    let stress_error_ratio = vector_relative_error(stress, &expected_stress);
    let closed_form_reference_coverage_ratio = if displacement_error_ratio.is_finite()
        && stress_error_ratio.is_finite()
        && !direct_reference.is_empty()
        && !expected_stress.is_empty()
    {
        1.0
    } else {
        0.0
    };
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
        && closed_form_reference_coverage_ratio >= 1.0
        && transverse_displacement_leakage_ratio <= 0.1
        && primary_stress_component_ratio >= 0.5
        && displacement_error_ratio <= 1.0e-7
        && stress_error_ratio <= 1.0e-7
    {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };

    Some(FeaDiagnostic {
        code: "FEA_STRUCTURAL_REFERENCE_KINEMATICS".to_string(),
        severity,
        message: format!(
            "case={} primary_component={} primary_displacement_m={} transverse_displacement_leakage_ratio={} primary_stress_component_ratio={} directional_reference_coverage_ratio={} closed_form_displacement_error_ratio={} closed_form_stress_error_ratio={} closed_form_reference_coverage_ratio={}",
            case,
            primary_label,
            primary_displacement,
            transverse_displacement_leakage_ratio,
            primary_stress_component_ratio,
            directional_reference_coverage_ratio,
            displacement_error_ratio,
            stress_error_ratio,
            closed_form_reference_coverage_ratio
        ),
    })
}

fn structural_beam_closed_form_diagnostic(
    model: &AnalysisModel,
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
    for load in &model.loads {
        if load.region_id != "node:2" {
            continue;
        }
        match load.kind {
            LoadKind::Force { fy, .. } => {
                tip_force_y_n += fy;
            }
            LoadKind::Moment { mx, mz, .. } => {
                tip_torque_x_n_m += mx;
                tip_moment_z_n_m += mz;
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
    fields: &[AnalysisField],
) -> Option<FeaDiagnostic> {
    let requested = requested_moment_vector(model);
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

fn requested_moment_vector(model: &AnalysisModel) -> [f64; 3] {
    let mut out = [0.0_f64; 3];
    for load in &model.loads {
        match load.kind {
            LoadKind::Moment { mx, my, mz } => {
                out[0] += mx;
                out[1] += my;
                out[2] += mz;
            }
            LoadKind::Force { fx, fy, fz } => {
                if let Some(force_moment) = structural_force_moment_about_reaction_origin(
                    model,
                    &load.region_id,
                    [fx, fy, fz],
                ) {
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
        .boundary_conditions
        .iter()
        .find_map(|bc| {
            if !matches!(bc.kind, BoundaryConditionKind::Fixed) {
                return None;
            }
            structural_node_coordinates(structural, &bc.region_id)
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

fn solve_tridiagonal_reference(summary: &crate::assembly::AssemblySummary) -> Option<Vec<f64>> {
    let n = summary.operator.dof_count;
    if n == 0
        || summary.operator.stiffness_diag.len() != n
        || summary.operator.rhs.len() != n
        || summary.operator.stiffness_upper.len() != n.saturating_sub(1)
    {
        return None;
    }
    let mut upper = vec![0.0_f64; n.saturating_sub(1)];
    let mut rhs = vec![0.0_f64; n];
    let first_pivot = *summary.operator.stiffness_diag.first()?;
    if !first_pivot.is_finite() || first_pivot.abs() <= 1.0e-24 {
        return None;
    }
    if n > 1 {
        upper[0] = -summary.operator.stiffness_upper[0] / first_pivot;
    }
    rhs[0] = summary.operator.rhs[0] / first_pivot;
    for row in 1..n {
        let lower = -summary.operator.stiffness_upper[row - 1];
        let pivot = summary.operator.stiffness_diag[row] - lower * upper[row - 1];
        if !pivot.is_finite() || pivot.abs() <= 1.0e-24 {
            return None;
        }
        if row < n - 1 {
            upper[row] = -summary.operator.stiffness_upper[row] / pivot;
        }
        rhs[row] = (summary.operator.rhs[row] - lower * rhs[row - 1]) / pivot;
    }
    let mut solution = vec![0.0_f64; n];
    solution[n - 1] = rhs[n - 1];
    for row in (0..n - 1).rev() {
        solution[row] = rhs[row] - upper[row] * solution[row + 1];
    }
    Some(solution)
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
