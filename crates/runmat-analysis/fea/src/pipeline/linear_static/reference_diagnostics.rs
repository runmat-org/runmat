//! Independent reference and equilibrium diagnostics for static structural fixtures.

use super::*;

pub(super) fn structural_reference_kinematics_diagnostic(
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
        .zip(&summary.physical_load)
        .zip(summary.operator.constrained_dofs())
        .filter(|(_, constrained)| !**constrained)
        .map(|((lhs, rhs), _)| (lhs - rhs) * (lhs - rhs))
        .sum::<f64>()
        .sqrt();
    let load_norm = summary
        .physical_load
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

pub(super) fn structural_beam_closed_form_diagnostic(
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
    let mechanical = material.require_mechanical().ok()?;
    let youngs_modulus_pa = mechanical.youngs_modulus_pa.canonical_value();
    let shear_modulus_pa =
        youngs_modulus_pa / (2.0 * (1.0 + mechanical.poisson_ratio.canonical_value())).max(1.0e-9);
    let length_m = distance3(
        structural.nodes[0].coordinates_m.canonical_value(),
        structural.nodes[1].coordinates_m.canonical_value(),
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

    let eiz = youngs_modulus_pa * section.iz_m4.canonical_value();
    let gj = shear_modulus_pa * section.torsion_j_m4.canonical_value();
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

pub(super) fn structural_moment_balance_diagnostic(
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
        .or_else(|| {
            structural
                .nodes
                .first()
                .map(|node| node.coordinates_m.canonical_value())
        })?;
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
        .map(|node| node.coordinates_m.canonical_value())
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

pub(super) fn scalar_field_value(
    fields: &[runmat_analysis_core::AnalysisField],
    field_id: &str,
) -> Option<f64> {
    fields
        .iter()
        .find(|field| field.field_id == field_id)
        .and_then(runmat_analysis_core::AnalysisField::as_host_f64)
        .and_then(|values| values.first().copied())
}
