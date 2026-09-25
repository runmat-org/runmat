use runmat_analysis_core::StructuralOperatorImplementation;

use crate::{
    assembly::AssemblySummary,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    solve::StructuralStaticBalanceEvidence,
};

pub(super) fn known_answer(
    total_strain_energy: f64,
    balance: &StructuralStaticBalanceEvidence,
) -> FeaDiagnostic {
    let total_external_work = balance.applied_load_work_joules + balance.constraint_work_joules;
    let work_energy_ratio = if total_external_work.abs() > 1.0e-12 {
        (2.0 * total_strain_energy) / total_external_work
    } else if total_strain_energy.abs() <= 1.0e-12 {
        1.0
    } else {
        0.0
    };
    let work_energy_residual_ratio = (work_energy_ratio - 1.0).abs();
    let known_answer_coverage_ratio =
        if total_strain_energy.is_finite() && total_external_work.is_finite() && balance.accepted {
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
            "identity=linear_work_energy applied_load_work_joules={} constraint_work_joules={} total_external_work_joules={} total_strain_energy={} work_energy_ratio={} work_energy_residual_ratio={} equilibrium_relative_error={} balance_accepted={} known_answer_coverage_ratio={}",
            balance.applied_load_work_joules,
            balance.constraint_work_joules,
            total_external_work,
            total_strain_energy,
            work_energy_ratio,
            work_energy_residual_ratio,
            balance.equilibrium_relative_error,
            balance.accepted,
            known_answer_coverage_ratio
        ),
    }
}

pub(super) fn solid_assembly(summary: &AssemblySummary) -> Option<FeaDiagnostic> {
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
