use std::collections::BTreeMap;

use runmat_analysis_core::{
    AnalysisModel, MechanicalDampingModel, PropertyRegionScope, PropertyRegionTarget,
    StructuralElementKind, StructuralModel,
};
use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::SolverMeshArtifact;

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
    beams: Vec<(PropertyRegionScope, StructuralElementMaterial)>,
    shells: Vec<(PropertyRegionScope, StructuralElementMaterial)>,
    solids: Vec<(PropertyRegionScope, StructuralElementMaterial)>,
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
        let mut inventory = Self {
            beams: Vec::new(),
            shells: Vec::new(),
            solids: Vec::new(),
        };
        for assignment in &model.material_assignments {
            let Some(material) = materials.get(&assignment.assigned_material_id).copied() else {
                continue;
            };
            match &assignment.target {
                PropertyRegionTarget::Beam(scope) => {
                    inventory.beams.push((scope.clone(), material));
                }
                PropertyRegionTarget::Shell(scope) => {
                    inventory.shells.push((scope.clone(), material));
                }
                PropertyRegionTarget::Solid(scope) => {
                    inventory.solids.push((scope.clone(), material));
                }
                PropertyRegionTarget::Cell(_) => {}
            }
        }
        inventory
    }

    pub fn resolve_beam(&self, region_id: &str) -> Option<StructuralElementMaterial> {
        resolve_explicit(region_id, &self.beams)
    }

    pub fn resolve_shell(&self, region_id: &str) -> Option<StructuralElementMaterial> {
        resolve_explicit(region_id, &self.shells)
    }

    pub fn resolve_solid(
        &self,
        region: &PersistentEntityId,
    ) -> Result<Option<StructuralElementMaterial>, usize> {
        let mut matches = self.solids.iter().filter_map(|(scope, material)| {
            scope_matches_persistent(scope, region).then_some(*material)
        });
        let first = matches.next();
        let extra = matches.count();
        if extra == 0 {
            Ok(first)
        } else {
            Err(extra + 1)
        }
    }

    pub fn validate_explicit_coverage(&self, structural: &StructuralModel) -> Result<(), String> {
        let mut matched_beams = vec![false; self.beams.len()];
        let mut matched_shells = vec![false; self.shells.len()];
        for element in &structural.elements {
            let (kind, assignments, matched) = match &element.kind {
                StructuralElementKind::Beam(_) => ("beam", &self.beams, &mut matched_beams),
                StructuralElementKind::Shell(_) => ("shell", &self.shells, &mut matched_shells),
            };
            let matching = assignments
                .iter()
                .enumerate()
                .filter_map(|(index, (scope, _))| {
                    scope_matches_explicit(scope, &element.region_id).then_some(index)
                })
                .collect::<Vec<_>>();
            if matching.len() != 1 {
                return Err(format!(
                    "{kind} element `{}` in region `{}` requires exactly one compatible material assignment; found {}",
                    element.element_id, element.region_id, matching.len()
                ));
            }
            matched[matching[0]] = true;
        }
        reject_unmatched("beam", &self.beams, &matched_beams)?;
        reject_unmatched("shell", &self.shells, &matched_shells)
    }

    pub fn validate_solid_coverage(&self, mesh: &SolverMeshArtifact) -> Result<(), String> {
        let mut matched = vec![false; self.solids.len()];
        for element in &mesh.topology.volume_elements {
            let matching = self
                .solids
                .iter()
                .enumerate()
                .filter_map(|(index, (scope, _))| {
                    scope_matches_persistent(scope, &element.region_id).then_some(index)
                })
                .collect::<Vec<_>>();
            if matching.len() != 1 {
                return Err(format!(
                    "solid element {} in region `{}` requires exactly one solid material assignment; found {}",
                    element.element_id,
                    element.region_id.source_topology_id,
                    matching.len()
                ));
            }
            matched[matching[0]] = true;
        }
        if let Some((index, _)) = matched.iter().enumerate().find(|(_, found)| !**found) {
            return Err(format!(
                "solid material assignment for `{}` selects no solver element",
                scope_label(&self.solids[index].0)
            ));
        }
        Ok(())
    }
}

fn reject_unmatched(
    kind: &str,
    assignments: &[(PropertyRegionScope, StructuralElementMaterial)],
    matched: &[bool],
) -> Result<(), String> {
    if let Some((index, _)) = matched.iter().enumerate().find(|(_, found)| !**found) {
        Err(format!(
            "{kind} material assignment for `{}` selects no {kind} element",
            scope_label(&assignments[index].0)
        ))
    } else {
        Ok(())
    }
}

fn resolve_explicit(
    region_id: &str,
    assignments: &[(PropertyRegionScope, StructuralElementMaterial)],
) -> Option<StructuralElementMaterial> {
    let mut matches = assignments.iter().filter_map(|(scope, material)| {
        scope_matches_explicit(scope, region_id).then_some(*material)
    });
    let first = matches.next();
    debug_assert!(
        matches.next().is_none(),
        "material coverage must be validated before resolution"
    );
    first
}

fn scope_matches_explicit(scope: &PropertyRegionScope, region_id: &str) -> bool {
    match scope {
        PropertyRegionScope::All => true,
        PropertyRegionScope::Region(selection) => selection.region_id() == region_id,
    }
}

fn scope_matches_persistent(scope: &PropertyRegionScope, region: &PersistentEntityId) -> bool {
    match scope {
        PropertyRegionScope::All => true,
        PropertyRegionScope::Region(selection) => {
            region.matches_analysis_selection(selection.region_id())
        }
    }
}

fn scope_label(scope: &PropertyRegionScope) -> &str {
    scope.selected_region_id().unwrap_or("<all>")
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
    fn inventory_requires_an_exact_region_assignment_even_for_one_material() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.materials[0].mechanical.damping = MechanicalDampingModel::Rayleigh {
            mass_coefficient: RayleighMassCoefficient::new(0.3),
            stiffness_coefficient: RayleighStiffnessCoefficient::new(0.004),
        };
        let single = StructuralMaterialInventory::from_model(&model);
        let assigned_region = PersistentEntityId {
            kind: runmat_geometry_core::PersistentEntityKind::Region,
            source_topology_id: model.material_assignments[0]
                .selected_region_id()
                .expect("fixture uses selected material region")
                .to_owned(),
            assembly_path: vec!["root".into()],
        };
        assert_eq!(
            single.resolve_solid(&assigned_region),
            Ok(Some(StructuralElementMaterial::from_model(
                &model.materials[0]
            )))
        );

        let mut second = model.materials[0].clone();
        second.material_id = "second".into();
        second.mechanical.youngs_modulus_pa *= 0.5;
        model.materials.push(second);
        model.material_assignments = vec![MaterialAssignment {
            target: runmat_analysis_core::PropertyRegionTarget::beam("chosen"),
            expected_material_id: "second".into(),
            assigned_material_id: "second".into(),
            confidence: EvidenceConfidence::Verified,
        }];
        let multiple = StructuralMaterialInventory::from_model(&model);
        assert_eq!(
            multiple
                .resolve_beam("chosen")
                .expect("assigned material")
                .youngs_modulus_pa,
            model.materials[1].mechanical.youngs_modulus_pa
        );
        assert_eq!(multiple.resolve_beam("unassigned"), None);
    }

    #[test]
    fn explicit_all_beams_scope_is_typed_and_must_select_real_beam_elements() {
        let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        model.material_assignments[0].target =
            runmat_analysis_core::PropertyRegionTarget::all_beams();
        let structural = model.structural.as_ref().expect("beam fixture topology");
        let inventory = StructuralMaterialInventory::from_model(&model);
        inventory
            .validate_explicit_coverage(structural)
            .expect("all-beams scope should cover the fixture beam");
        assert!(inventory.resolve_beam("beam_span").is_some());

        model.material_assignments[0].target =
            runmat_analysis_core::PropertyRegionTarget::all_shells();
        let stale = StructuralMaterialInventory::from_model(&model);
        assert!(stale
            .validate_explicit_coverage(structural)
            .expect_err("a shell material scope cannot cover a beam")
            .contains("beam_1"));
    }
}
