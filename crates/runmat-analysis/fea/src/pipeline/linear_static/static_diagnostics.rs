use runmat_analysis_core::StructuralOperatorImplementation;

use crate::{
    assembly::AssemblySummary,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
};

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
