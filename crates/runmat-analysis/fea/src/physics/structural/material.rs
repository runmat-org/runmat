use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisModel, MechanicalDampingModel};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuralElementMaterial {
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
    pub density_kg_per_m3: f64,
    pub damping: MechanicalDampingModel,
}

impl StructuralElementMaterial {
    fn from_model(material: &runmat_analysis_core::MaterialModel) -> Self {
        Self {
            youngs_modulus_pa: material.mechanical.youngs_modulus_pa,
            poisson_ratio: material.mechanical.poisson_ratio,
            density_kg_per_m3: material.mechanical.density_kg_per_m3,
            damping: material.mechanical.damping,
        }
    }

    pub fn shear_modulus_pa(self) -> f64 {
        self.youngs_modulus_pa / (2.0 * (1.0 + self.poisson_ratio))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructuralMaterialInventory {
    default: Option<StructuralElementMaterial>,
    by_region: BTreeMap<String, StructuralElementMaterial>,
}

impl StructuralMaterialInventory {
    pub fn from_model(model: &AnalysisModel) -> Self {
        let materials = model
            .materials
            .iter()
            .map(|material| {
                (
                    material.material_id.clone(),
                    StructuralElementMaterial::from_model(material),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let default = match model.materials.as_slice() {
            [material] => Some(StructuralElementMaterial::from_model(material)),
            _ => None,
        };
        let by_region = model
            .material_assignments
            .iter()
            .filter_map(|assignment| {
                materials
                    .get(&assignment.assigned_material_id)
                    .copied()
                    .map(|material| (assignment.region_id.clone(), material))
            })
            .collect();
        Self { default, by_region }
    }

    pub fn resolve(&self, region_id: &str) -> Option<StructuralElementMaterial> {
        self.by_region.get(region_id).copied().or(self.default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{fixture_model, FixtureId};
    use runmat_analysis_core::{
        EvidenceConfidence, MaterialAssignment, MechanicalDampingModel, RayleighMassCoefficient,
        RayleighStiffnessCoefficient,
    };

    #[test]
    fn inventory_uses_exact_region_assignment_and_only_unambiguous_default() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.materials[0].mechanical.damping = MechanicalDampingModel::Rayleigh {
            mass_coefficient: RayleighMassCoefficient::new(0.3),
            stiffness_coefficient: RayleighStiffnessCoefficient::new(0.004),
        };
        let single = StructuralMaterialInventory::from_model(&model);
        assert_eq!(
            single
                .resolve("unassigned")
                .expect("single default")
                .damping,
            model.materials[0].mechanical.damping
        );

        let mut second = model.materials[0].clone();
        second.material_id = "second".into();
        second.mechanical.youngs_modulus_pa *= 0.5;
        model.materials.push(second);
        model.material_assignments = vec![MaterialAssignment {
            region_id: "chosen".into(),
            expected_material_id: "second".into(),
            assigned_material_id: "second".into(),
            confidence: EvidenceConfidence::Verified,
        }];
        let multiple = StructuralMaterialInventory::from_model(&model);
        assert_eq!(
            multiple
                .resolve("chosen")
                .expect("assigned material")
                .youngs_modulus_pa,
            model.materials[1].mechanical.youngs_modulus_pa
        );
        assert_eq!(multiple.resolve("unassigned"), None);
    }
}
