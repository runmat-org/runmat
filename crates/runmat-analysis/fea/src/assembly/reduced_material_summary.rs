//! Legacy scalar summary retained for reduced recovery and diagnostics. Physical element
//! operators take coefficients from compiled element owners; this summary must not qualify
//! heterogeneous constitutive recovery for production.

use runmat_analysis_core::AnalysisModel;

use super::{mean, StructuralMaterialSummary};

pub(super) fn structural_material_summary(model: &AnalysisModel) -> StructuralMaterialSummary {
    let avg_youngs_modulus = mean(model.materials.iter().filter_map(|material| {
        material
            .require_mechanical()
            .ok()
            .map(|mechanical| mechanical.youngs_modulus_pa.canonical_value())
    }))
    .unwrap_or(1.0e9);
    let avg_poisson_ratio = mean(model.materials.iter().filter_map(|material| {
        material
            .require_mechanical()
            .ok()
            .map(|mechanical| mechanical.poisson_ratio.canonical_value())
    }))
    .unwrap_or(0.3);
    let shear_modulus_pa = avg_youngs_modulus / (2.0 * (1.0 + avg_poisson_ratio));
    let avg_density_kg_per_m3 = mean(model.materials.iter().filter_map(|material| {
        material
            .mass_density_kg_per_m3
            .map(|quantity| quantity.canonical_value())
    }))
    .unwrap_or(7850.0);
    let lame_lambda_pa = avg_youngs_modulus * avg_poisson_ratio
        / ((1.0 + avg_poisson_ratio) * (1.0 - 2.0 * avg_poisson_ratio));
    StructuralMaterialSummary {
        youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(avg_youngs_modulus),
        poisson_ratio: runmat_types::physical::PoissonRatio::new(avg_poisson_ratio),
        density_kg_per_m3: runmat_types::physical::MassDensity::new(avg_density_kg_per_m3),
        lame_lambda_pa,
        shear_modulus_pa: runmat_types::physical::ShearModulus::new(shear_modulus_pa),
    }
}
