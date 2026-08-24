#![cfg(feature = "occt-native")]

use runmat_geometry_core::{
    ExactMassPropertiesEvaluator, GeometryEvaluationControl, GeometryEvaluationError, UnitSystem,
};
use runmat_geometry_io::import::GeometryImportError;
use runmat_geometry_io::{
    import_exact_cad, ExactCadImportOptions, GeometryFormat, GeometryImportContext,
    OcctExactEvaluator,
};

const NIST_SEEKER_TABLE: &[u8] = include_bytes!("fixtures/nist_seeker_table_acis.step");
const ANSYS_WIREFRAME: &[u8] = include_bytes!("fixtures/ansys_sample.iges");

struct Unlimited;

impl GeometryEvaluationControl for Unlimited {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }

    fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }

    fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }

    fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
}

#[test]
fn public_industrial_step_imports_exact_topology_and_mass_properties() {
    let options = ExactCadImportOptions {
        source_units: UnitSystem::Millimeter,
        ..ExactCadImportOptions::default()
    };
    let imported = import_exact_cad(
        "nist_seeker_table_acis.step",
        NIST_SEEKER_TABLE,
        GeometryFormat::Step,
        &options,
        &GeometryImportContext::new(),
    )
    .expect("public-domain NIST ACIS STEP must import");

    assert_eq!(imported.topology.bodies.len(), 1);
    assert_eq!(imported.topology.solids.len(), 1);
    assert_eq!(imported.topology.shells.len(), 1);
    assert_eq!(imported.topology.faces.len(), 6);
    assert_eq!(imported.topology.edges.len(), 12);
    assert_eq!(imported.topology.vertices.len(), 8);
    assert_eq!(imported.topology.regions.len(), 1);

    let evaluator = OcctExactEvaluator::new(&imported).expect("evaluator must open");
    let mass = evaluator
        .mass_properties(
            &imported.topology.bodies[0].mass_properties_evaluator_id,
            &Unlimited,
        )
        .expect("kernel mass properties must evaluate");
    assert!((mass.volume_m3 - 0.011_076_007_2).abs() < 1.0e-12);
    assert!((mass.surface_area_m2 - 2.269_505_468_8).abs() < 1.0e-12);
    assert!(mass.centroid_m[0].abs() < 1.0e-12);
    assert!(mass.centroid_m[1].abs() < 1.0e-12);
    assert!((mass.centroid_m[2] + 0.005).abs() < 1.0e-12);
    imported.build_closure().expect("closure must validate");
}

#[test]
fn public_iges_wireframe_is_read_then_rejected_by_typed_admission() {
    let options = ExactCadImportOptions {
        source_units: UnitSystem::Millimeter,
        ..ExactCadImportOptions::default()
    };
    let error = import_exact_cad(
        "ansys_sample.iges",
        ANSYS_WIREFRAME,
        GeometryFormat::Iges,
        &options,
        &GeometryImportContext::new(),
    )
    .expect_err("a wireframe has no solver-mesh boundary topology");

    assert!(matches!(
        error,
        GeometryImportError::InvalidGeometry(reason)
            if reason == "OCCT exact shape has incomplete boundary topology"
    ));
}
