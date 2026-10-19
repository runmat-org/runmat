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

#[test]
fn concurrent_step_imports_keep_independent_units_and_physical_properties() {
    let start = std::sync::Barrier::new(6);
    std::thread::scope(|scope| {
        for worker in 0..6 {
            let start = &start;
            scope.spawn(move || {
                start.wait();
                for iteration in 0..6 {
                    let units = [UnitSystem::Meter, UnitSystem::Millimeter, UnitSystem::Inch]
                        [(worker + iteration) % 3];
                    let imported = import_exact_cad(
                        "nist_seeker_table_acis.step",
                        NIST_SEEKER_TABLE,
                        GeometryFormat::Step,
                        &ExactCadImportOptions {
                            source_units: units,
                            ..ExactCadImportOptions::default()
                        },
                        &GeometryImportContext::new(),
                    )
                    .unwrap();
                    let evaluator = OcctExactEvaluator::new(&imported).unwrap();
                    let mass = evaluator
                        .mass_properties(
                            &imported.topology.bodies[0].mass_properties_evaluator_id,
                            &Unlimited,
                        )
                        .unwrap();
                    assert!((mass.volume_m3 - 0.011_076_007_2).abs() < 1.0e-12);
                    assert!((mass.centroid_m[2] + 0.005).abs() < 1.0e-12);
                }
            });
        }
    });
}

#[test]
fn declared_step_units_preserve_physical_scale_across_requested_coordinate_units() {
    let unspecified = import_exact_cad(
        "nist_seeker_table_acis.step",
        NIST_SEEKER_TABLE,
        GeometryFormat::Step,
        &ExactCadImportOptions {
            source_units: UnitSystem::Unspecified,
            ..ExactCadImportOptions::default()
        },
        &GeometryImportContext::new(),
    )
    .unwrap_err();
    assert!(matches!(
        unspecified,
        GeometryImportError::InvalidOptions(_)
    ));
    // This STEP's native dimensions are independently fixed by the public corpus
    // reference below. Caller coordinate units must not reinterpret that shape.
    for units in [UnitSystem::Meter, UnitSystem::Millimeter, UnitSystem::Inch] {
        let imported = import_exact_cad(
            "nist_seeker_table_acis.step",
            NIST_SEEKER_TABLE,
            GeometryFormat::Step,
            &ExactCadImportOptions {
                source_units: units,
                ..ExactCadImportOptions::default()
            },
            &GeometryImportContext::new(),
        )
        .unwrap();
        let evaluator = OcctExactEvaluator::new(&imported).unwrap();
        let mass = evaluator
            .mass_properties(
                &imported.topology.bodies[0].mass_properties_evaluator_id,
                &Unlimited,
            )
            .unwrap();
        assert!(
            (mass.volume_m3 - 0.011_076_007_2).abs() < 1.0e-12,
            "{units:?}: {mass:?}"
        );
        assert!((mass.surface_area_m2 - 2.269_505_468_8).abs() < 1.0e-11);
        assert!((mass.centroid_m[2] + 0.005).abs() < 1.0e-12);
        imported.build_closure().unwrap();
    }
}

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
    OcctExactEvaluator::from_closure(
        &imported.representation,
        imported.meters_per_source_unit,
        &imported.topology,
        &imported.evaluators,
    )
    .expect("the distributed exact-geometry closure must reconstruct its evaluator");
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

#[test]
#[ignore = "requires RUNMAT_CAD_BRACKET_STEP pointing to the original sandbox STEP asset"]
fn sandbox_bracket_samples_stay_inside_original_edge_domains() {
    use runmat_geometry_core::ExactCurveEvaluator;
    use sha2::{Digest, Sha256};
    let path = std::env::var("RUNMAT_CAD_BRACKET_STEP").expect("explicit external corpus path");
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "723b72c4b36e7056f0d246fd528479f213d7073b1480e74fffcb7e78eb53ce3c"
    );
    let imported = import_exact_cad(
        &path,
        &bytes,
        GeometryFormat::Step,
        &ExactCadImportOptions::default(),
        &GeometryImportContext::new(),
    )
    .unwrap();
    let evaluator = OcctExactEvaluator::new(&imported).unwrap();
    let mut checked = 0;
    for edge in &imported.topology.edges {
        if edge.is_degenerate {
            continue;
        }
        let range = evaluator.parameter_range(&edge.curve_evaluator_id).unwrap();
        for sample in 0..9 {
            let fraction = f64::from(sample) / 8.0;
            let parameter = range.parameter_at_fraction(fraction).unwrap();
            assert!(parameter >= range.start && parameter <= range.end,
                "edge {:?}, sample {sample}, range [{:.17e}, {:.17e}], parameter {:.17e}, bits [{:x}, {:x}, {:x}]",
                edge.id, range.start, range.end, parameter,
                range.start.to_bits(), range.end.to_bits(), parameter.to_bits());
            evaluator
                .derivatives(&edge.curve_evaluator_id, parameter, &Unlimited)
                .unwrap();
            checked += 1;
        }
        if range.end.next_up().is_finite() {
            assert!(
                evaluator
                    .derivatives(&edge.curve_evaluator_id, range.end.next_up(), &Unlimited)
                    .is_err(),
                "native edge guard must still reject an actually external parameter"
            );
        }
    }
    assert!(checked > 0);
}
