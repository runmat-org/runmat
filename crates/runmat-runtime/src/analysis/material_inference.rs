use runmat_analysis_core::{
    MaterialMechanicalModel, MaterialModel, MaterialProvenance, MaterialThermalModel,
};
use runmat_geometry_core::GeometryAsset;

pub(super) fn infer_material_models(geometry: &GeometryAsset) -> Vec<MaterialModel> {
    let mut materials = Vec::new();
    for evidence in &geometry.source_geometry.material_evidence {
        let value = evidence.value.to_ascii_lowercase();
        let (material_id, name, youngs_modulus_pa, poisson_ratio) = if value.contains("aluminum") {
            ("mat_aluminum", "Aluminum", 69e9, 0.33)
        } else if value.contains("steel") {
            ("mat_steel", "Steel", 200e9, 0.30)
        } else if value.contains("polymer") || value.contains("plastic") {
            ("mat_polymer", "Polymer", 3.2e9, 0.37)
        } else {
            ("mat_inferred", "Inferred Material", 100e9, 0.32)
        };

        if materials
            .iter()
            .any(|material: &MaterialModel| material.material_id == material_id)
        {
            continue;
        }
        materials.push(material_model(
            material_id,
            name,
            youngs_modulus_pa,
            poisson_ratio,
        ));
    }

    if materials.is_empty() {
        materials.push(material_model(
            "mat_default_steel",
            "Steel (Default)",
            200e9,
            0.3,
        ));
    }
    materials
}

fn material_model(
    material_id: &str,
    name: &str,
    youngs_modulus_pa: f64,
    poisson_ratio: f64,
) -> MaterialModel {
    MaterialModel {
        material_id: material_id.to_string(),
        name: name.to_string(),
        provenance: MaterialProvenance::Authored,
        mass_density_kg_per_m3: Some(7850.0),
        mechanical: Some(MaterialMechanicalModel {
            youngs_modulus_pa,
            poisson_ratio,
        }),
        damping: None,
        thermal: Some(MaterialThermalModel {
            reference_temperature_k: 293.15,
            modulus_temp_coeff_per_k: -2.5e-4,
            conductivity_w_per_mk: 45.0,
            specific_heat_j_per_kgk: 500.0,
            expansion_coefficient_per_k: 1.2e-5,
        }),
        acoustic: None,
        electrical: None,
        plastic: None,
    }
}
