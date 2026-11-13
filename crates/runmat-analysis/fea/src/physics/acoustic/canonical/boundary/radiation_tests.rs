use super::*;
use crate::{
    properties::compile_solid_property_map,
    topology::{test_meshes::slab_mesh, AnalysisProjection},
};
use runmat_analysis_core::PropertyRegionTarget;

fn assembly() -> AcousticAssembly {
    let mesh = slab_mesh(2);
    let mut model = super::super::tests::model();
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    let properties = compile_solid_property_map(&model, &mesh).unwrap();
    AcousticAssembly::assemble(
        AnalysisProjection::derive(&mesh).unwrap(),
        &properties,
        runmat_types::physical::AngularFrequency::new(1.0),
    )
    .unwrap()
}

#[test]
fn radiation_uses_unique_face_owner_not_first_medium() {
    let mut a = assembly();
    let face = a
        .faces(&RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("slab_right"),
        ))
        .unwrap()
        .remove(0);
    let [owner] = face.adjacent_element_indices.as_slice() else {
        panic!("unique owner")
    };
    assert_ne!(
        *owner, 0,
        "fixture must distinguish first material from face owner"
    );
    a.density = a
        .density
        .clone()
        .try_map_canonical_storage(|mut values| {
            values.fill(3.0);
            values
        })
        .unwrap();
    a.compressibility = a
        .compressibility
        .clone()
        .try_map_canonical_storage(|mut values| {
            values.fill(1.0 / 12.0);
            values
        })
        .unwrap();
    a.density = runmat_types::physical::Quantity::new({
        let mut values = a.density.canonical_values().clone();
        values[*owner] = 5.0;
        values
    });
    a.compressibility = runmat_types::physical::Quantity::new({
        let mut values = a.compressibility.canonical_values().clone();
        values[*owner] = 1.0 / 80.0;
        values
    }); // c=4, Z=20.
    assert!((a.radiation_admittance(&face).unwrap() - 1.0 / 20.0).abs() < 1e-15);
    a.density = runmat_types::physical::Quantity::new({
        let mut values = a.density.canonical_values().clone();
        values[0] = 123.0;
        values
    });
    assert!((a.radiation_admittance(&face).unwrap() - 1.0 / 20.0).abs() < 1e-15);
}

#[test]
fn radiation_rejects_missing_multiple_foreign_or_invalid_owners() {
    let mut a = assembly();
    let face = a
        .faces(&RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("slab_right"),
        ))
        .unwrap()
        .remove(0);
    let owner = face.adjacent_element_indices[0];
    let mut interior = face.clone();
    interior.role = BoundaryFaceRole::ConformalInterface;
    assert!(a.radiation_admittance(&interior).is_err());
    for owners in [vec![], vec![owner, owner], vec![usize::MAX]] {
        let mut invalid = face.clone();
        invalid.adjacent_element_indices = owners;
        assert!(matches!(
            a.radiation_admittance(&invalid),
            Err(AcousticHarmonicError::Boundary(_))
        ));
    }
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        a.density = runmat_types::physical::Quantity::new({
            let mut values = a.density.canonical_values().clone();
            values[owner] = invalid;
            values
        });
        assert!(a.radiation_admittance(&face).is_err());
    }
    a.density = runmat_types::physical::Quantity::new({
        let mut values = a.density.canonical_values().clone();
        values[owner] = 3.0;
        values
    });
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        a.compressibility = runmat_types::physical::Quantity::new({
            let mut values = a.compressibility.canonical_values().clone();
            values[owner] = invalid;
            values
        });
        assert!(a.radiation_admittance(&face).is_err());
    }
    assert!(a
        .faces(&RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("absent")
        ))
        .is_err());
}
