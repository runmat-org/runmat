use super::*;
use crate::{
    properties::compile_solid_property_map,
    topology::{test_meshes::slab_mesh, AnalysisProjection},
};

fn assembly() -> AcousticAssembly {
    let mesh = slab_mesh(2);
    let mut model = super::super::super::tests::model();
    model.material_assignments[0].target = runmat_analysis_core::PropertyRegionTarget::solid(
        runmat_meshing_core::fixtures::fixture_solid("slab"),
    );
    let properties = compile_solid_property_map(&model, &mesh).unwrap();
    AcousticAssembly::assemble(
        AnalysisProjection::derive(&mesh).unwrap(),
        &properties,
        runmat_types::physical::AngularFrequency::new(1.0),
    )
    .unwrap()
}

#[test]
fn incident_plane_rejects_equal_impedance_but_different_material_and_nonparallel_union() {
    let mut a = assembly();
    let left = a
        .faces(&RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("slab_left"),
        ))
        .unwrap();
    assert!(left.len() > 1);
    a.validate_incident_plane(&left).unwrap();
    let owner = left.last().unwrap().adjacent_element_indices[0];
    assert_ne!(owner, left[0].adjacent_element_indices[0]);
    a.density = runmat_types::physical::Quantity::new({
        let mut values = a.density.canonical_values().clone();
        values[owner] *= 2.0;
        values
    });
    a.compressibility = runmat_types::physical::Quantity::new({
        let mut values = a.compressibility.canonical_values().clone();
        values[owner] *= 2.0;
        values
    }); // Y unchanged, c halved: genuinely different medium.
    assert!(matches!(
        a.validate_incident_plane(&left),
        Err(AcousticHarmonicError::Unsupported(_))
    ));
    let a = assembly();
    let mut union = a
        .faces(&RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("slab_left"),
        ))
        .unwrap();
    union.extend(
        a.faces(&RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("slab_right"),
        ))
        .unwrap(),
    );
    assert!(matches!(
        a.validate_incident_plane(&union),
        Err(AcousticHarmonicError::Unsupported(_))
    ));
}

#[test]
fn incident_admittance_rejects_missing_multiple_and_foreign_owners() {
    let a = assembly();
    let face = a
        .faces(&RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("slab_left"),
        ))
        .unwrap()
        .remove(0);
    for owners in [vec![], vec![0, 0], vec![usize::MAX]] {
        let mut invalid = face.clone();
        invalid.adjacent_element_indices = owners;
        assert!(a.radiation_admittance(&invalid).is_err());
    }
}
