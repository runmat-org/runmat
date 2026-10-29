use super::*;
use crate::fixtures::{fixture_model, FixtureId};
use runmat_analysis_core::{
    AnalysisSequence, DynamicViscosity, MaterialFluidModel, ProcedureStep, PropertyRegionTarget,
};
use runmat_meshing_core::fixtures::three_region_tetrahedron_solver_mesh;

fn fixture() -> (AnalysisModel, SolverMeshArtifact) {
    let mesh = three_region_tetrahedron_solver_mesh();
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.geometry_revision = mesh.geometry.geometry_revision as u32;
    model.sequence =
        AnalysisSequence::from_steps("flow", vec![ProcedureStep::fluid_static("steady")]);
    model.constraints.clear();
    model.excitations.clear();
    model.cfd = None;
    model.materials[0].fluid = Some(MaterialFluidModel::IncompressibleNewtonian {
        dynamic_viscosity: DynamicViscosity::new(0.001),
    });
    let template = model.material_assignments[0].clone();
    model.material_assignments = mesh
        .topology
        .volume_elements
        .iter()
        .enumerate()
        .map(|(index, element)| {
            let mut material = model.materials[0].clone();
            material.material_id = format!("fluid-{index}");
            let mut assignment = template.clone();
            assignment.target =
                PropertyRegionTarget::cell(element.region_id.analysis_selection_id().unwrap());
            assignment.expected_material_id = material.material_id.clone();
            assignment.assigned_material_id = material.material_id.clone();
            model.materials.push(material);
            assignment
        })
        .collect();
    (model, mesh)
}

#[test]
fn exact_cell_inventory_admits_same_coefficients_from_distinct_material_ids() {
    let (model, mesh) = fixture();
    let map = compile_property_map(&model, Some(&mesh)).unwrap();
    assert_eq!(map.cell_materials().len(), 3);
    assert_eq!(map.solid_materials().len(), 0);
    let domain = derive_canonical_fluid_material(&model, &mesh, &map).unwrap();
    assert_eq!(domain.dynamic_viscosity.canonical_value(), 0.001);
    assert_eq!(
        domain.density,
        model.materials[0].require_mass_density().unwrap()
    );
    assert_eq!(domain.turbulence_intensity.canonical_value(), 0.0);
    let wire = serde_json::to_vec(&map).unwrap();
    let restored: CompiledPropertyMap = serde_json::from_slice(&wire).unwrap();
    assert_eq!(restored, map);
    restored.validate_integrity().unwrap();
}

#[test]
fn cell_coverage_rejects_missing_overlapping_unused_and_solid_owners() {
    let (model, mesh) = fixture();
    let mut missing = model.clone();
    missing.material_assignments.pop();
    assert!(matches!(
        compile_property_map(&missing, Some(&mesh)),
        Err(PropertyCompilationError::InvalidCoverage { role: "cell", .. })
    ));
    let mut overlap = model.clone();
    overlap
        .material_assignments
        .push(overlap.material_assignments[0].clone());
    assert!(matches!(
        compile_property_map(&overlap, Some(&mesh)),
        Err(PropertyCompilationError::InvalidCoverage { .. })
    ));
    let mut unused = model.clone();
    let mut extra = unused.material_assignments[0].clone();
    extra.target = PropertyRegionTarget::cell(runmat_meshing_core::fixtures::fixture_region(
        "absent-region",
    ));
    unused.material_assignments.push(extra);
    assert!(matches!(
        compile_property_map(&unused, Some(&mesh)),
        Err(PropertyCompilationError::UnusedAssignment { .. })
    ));
    let mut mixed = model.clone();
    mixed.material_assignments[0].target = PropertyRegionTarget::all_solids();
    assert!(matches!(
        compile_property_map(&mixed, Some(&mesh)),
        Err(PropertyCompilationError::UnsupportedMixedVolumeRoles)
    ));
}

#[test]
fn material_authority_rejects_heterogeneity_missing_law_and_stale_domain() {
    let (model, mesh) = fixture();
    let mut heterogeneous = model.clone();
    heterogeneous.materials[2].fluid = Some(MaterialFluidModel::IncompressibleNewtonian {
        dynamic_viscosity: DynamicViscosity::new(0.002),
    });
    let map = compile_property_map(&heterogeneous, Some(&mesh)).unwrap();
    assert!(matches!(
        derive_canonical_fluid_material(&heterogeneous, &mesh, &map),
        Err(FluidMaterialCompilationError::HeterogeneousCoefficients { .. })
    ));
    let mut absent = model.clone();
    absent.materials[1].fluid = None;
    let map = compile_property_map(&absent, Some(&mesh)).unwrap();
    assert!(matches!(
        derive_canonical_fluid_material(&absent, &mesh, &map),
        Err(FluidMaterialCompilationError::Material(_))
    ));
    let map = compile_property_map(&model, Some(&mesh)).unwrap();
    let mut stale = model.clone();
    let mut domain = derive_canonical_fluid_material(&model, &mesh, &map).unwrap();
    domain.dynamic_viscosity = DynamicViscosity::new(0.5);
    stale.cfd = Some(domain);
    assert!(matches!(
        derive_canonical_fluid_material(&stale, &mesh, &map),
        Err(FluidMaterialCompilationError::RetainedDomainMismatch)
    ));
}

#[test]
fn source_bound_map_cannot_be_substituted_and_coefficients_change_identity() {
    let (model, mesh) = fixture();
    let original = compile_property_map(&model, Some(&mesh)).unwrap();
    let mut changed = model.clone();
    for material in &mut changed.materials {
        material.fluid = Some(MaterialFluidModel::IncompressibleNewtonian {
            dynamic_viscosity: DynamicViscosity::new(0.003),
        });
    }
    let replacement = compile_property_map(&changed, Some(&mesh)).unwrap();
    assert_ne!(original.identity, replacement.identity);
    assert!(matches!(
        derive_canonical_fluid_material(&model, &mesh, &replacement),
        Err(FluidMaterialCompilationError::SourceSubstitution)
    ));
}
