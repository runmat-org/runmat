#![cfg(feature = "occt-native")]

use runmat_geometry_core::{
    ExactMassPropertiesEvaluator, GeometryEvaluationControl, GeometryEvaluationError, LengthUnit,
};
use runmat_geometry_io::import::GeometryImportError;
use runmat_geometry_io::{
    import_exact_cad, ExactCadImportOptions, GeometryFormat, GeometryImportContext,
    OcctExactEvaluator, SourceUnitsAuthority,
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
                    let units = [LengthUnit::Meter, LengthUnit::Millimeter, LengthUnit::Inch]
                        [(worker + iteration) % 3];
                    let imported = import_exact_cad(
                        "nist_seeker_table_acis.step",
                        NIST_SEEKER_TABLE,
                        GeometryFormat::Step,
                        &ExactCadImportOptions {
                            source_units: Some(units),
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
                    assert!((mass.volume_m3.canonical_value() - 0.011_076_007_2).abs() < 1.0e-12);
                    assert!((mass.centroid_m.canonical_value()[2] + 0.005).abs() < 1.0e-12);
                }
            });
        }
    });
}

// witness: FEA-MESH-GEOMETRY-001/geometry_load_units_option @
//   shared_infrastructure/mesh_independent/source_check; features: occt-native; coverage:
//   partial;
//   boundary: Declared STEP units preserve physical scale across requested coordinate units; mixed
//   or missing unit rejection is not asserted here.
#[test]
fn declared_step_units_preserve_physical_scale_across_requested_coordinate_units() {
    for units in [LengthUnit::Meter, LengthUnit::Millimeter, LengthUnit::Inch] {
        let imported = import_exact_cad(
            "nist_seeker_table_acis.step",
            NIST_SEEKER_TABLE,
            GeometryFormat::Step,
            &ExactCadImportOptions {
                source_units: Some(units),
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
            (mass.volume_m3.canonical_value() - 0.011_076_007_2).abs() < 1.0e-12,
            "{units:?}: {mass:?}"
        );
        assert!((mass.surface_area_m2.canonical_value() - 2.269_505_468_8).abs() < 1.0e-11);
        assert!((mass.centroid_m.canonical_value()[2] + 0.005).abs() < 1.0e-12);
        imported.build_closure().unwrap();
    }
}

/// The round-pipe flow fixture (radius 0.05, length 0.5) with its unit header relabelled as
/// millimetres, as the fixture shipped before its generator was fixed.
fn pipe_labelled_millimetres() -> Vec<u8> {
    let metres = include_str!("../../../../docs/simulation/examples/geometry/pipe.step");
    assert!(metres.contains("SI_UNIT($,.METRE.)"));
    metres
        .replace("SI_UNIT($,.METRE.)", "SI_UNIT(.MILLI.,.METRE.)")
        .into_bytes()
}

fn pipe_volume_m3(options: &ExactCadImportOptions) -> (f64, runmat_geometry_io::ImportedExactCad) {
    let imported = import_exact_cad(
        "pipe.step",
        &pipe_labelled_millimetres(),
        GeometryFormat::Step,
        options,
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
    (mass.volume_m3.canonical_value(), imported)
}

// witness: FEA-MESH-GEOMETRY-001/geometry_load_units_option @
//   shared_infrastructure/mesh_independent/source_check; features: occt-native; coverage:
//   partial;
//   boundary: An authored metre unit over a STEP file declaring millimetres reads the coordinates
//   in metres, records the declared unit as provenance in the geometry document, and survives
//   closure; without authority the declaration governs.
#[test]
fn authored_step_units_override_a_different_declaration_and_record_it() {
    let pipe_m3 = std::f64::consts::PI * 0.05 * 0.05 * 0.5;
    let (authored_volume, authored) = pipe_volume_m3(&ExactCadImportOptions {
        source_units: Some(LengthUnit::Meter),
        source_units_authority: SourceUnitsAuthority::Authored,
        ..ExactCadImportOptions::default()
    });
    assert!(
        (authored_volume / pipe_m3 - 1.0).abs() < 1.0e-9,
        "{authored_volume} vs {pipe_m3}"
    );
    assert_eq!(
        authored.source_unit_evidence.uniform_unit(),
        Some(LengthUnit::Meter)
    );
    assert_eq!(
        authored.source_unit_evidence.overridden_declared_unit(),
        Some(LengthUnit::Millimeter)
    );
    assert!((authored.kernel_units.meters_per_unit() - 1.0).abs() < 1.0e-15);
    let document = authored.geometry_document().unwrap();
    assert_eq!(
        document.source.unit_evidence.overridden_declared_unit(),
        Some(LengthUnit::Millimeter)
    );
    authored.build_closure().unwrap();

    // Without authority the caller's unit is only a fallback: the millimetre declaration governs.
    let (declared_volume, declared) = pipe_volume_m3(&ExactCadImportOptions {
        source_units: Some(LengthUnit::Meter),
        ..ExactCadImportOptions::default()
    });
    assert!((declared_volume / (pipe_m3 * 1.0e-9) - 1.0).abs() < 1.0e-9);
    assert_eq!(
        declared.source_unit_evidence.overridden_declared_unit(),
        None
    );

    // An authored unit equal to the declaration is not an override.
    let (same_volume, same) = pipe_volume_m3(&ExactCadImportOptions {
        source_units: Some(LengthUnit::Millimeter),
        source_units_authority: SourceUnitsAuthority::Authored,
        ..ExactCadImportOptions::default()
    });
    assert!((same_volume / (pipe_m3 * 1.0e-9) - 1.0).abs() < 1.0e-9);
    assert_eq!(same.source_unit_evidence.overridden_declared_unit(), None);
}

// witness: FEA-MESH-GEOMETRY-001/geometry_load_units_option @
//   shared_infrastructure/mesh_independent/source_check; features: occt-native; coverage:
//   partial;
//   boundary: An authored metre unit over an IGES file whose global section declares millimetres
//   reads the coordinates in metres and records the declared unit; without authority the
//   declaration governs.
#[test]
fn authored_iges_units_override_a_different_declaration_and_record_it() {
    let import = |options: &ExactCadImportOptions| {
        let imported = import_exact_cad(
            "box_metre_values_mm_label.igs",
            include_bytes!("fixtures/box_metre_values_mm_label.igs"),
            GeometryFormat::Iges,
            options,
            &GeometryImportContext::new(),
        )
        .unwrap();
        let volume = OcctExactEvaluator::new(&imported)
            .unwrap()
            .mass_properties(
                &imported.topology.bodies[0].mass_properties_evaluator_id,
                &Unlimited,
            )
            .unwrap()
            .volume_m3;
        (volume.canonical_value(), imported)
    };
    let box_m3 = 0.5 * 0.1 * 0.02;
    let (authored_volume, authored) = import(&ExactCadImportOptions {
        source_units: Some(LengthUnit::Meter),
        source_units_authority: SourceUnitsAuthority::Authored,
        ..ExactCadImportOptions::default()
    });
    assert!(
        (authored_volume / box_m3 - 1.0).abs() < 1.0e-9,
        "{authored_volume} vs {box_m3}"
    );
    assert_eq!(
        authored.source_unit_evidence.overridden_declared_unit(),
        Some(LengthUnit::Millimeter)
    );
    authored.build_closure().unwrap();
    let (declared_volume, declared) = import(&ExactCadImportOptions {
        source_units: Some(LengthUnit::Meter),
        ..ExactCadImportOptions::default()
    });
    assert!((declared_volume / (box_m3 * 1.0e-9) - 1.0).abs() < 1.0e-9);
    assert_eq!(
        declared.source_unit_evidence.overridden_declared_unit(),
        None
    );
}

fn step_volume_m3(
    name: &str,
    bytes: &[u8],
    options: &ExactCadImportOptions,
) -> (f64, runmat_geometry_io::ImportedExactCad) {
    let imported = import_exact_cad(
        name,
        bytes,
        GeometryFormat::Step,
        options,
        &GeometryImportContext::new(),
    )
    .unwrap();
    let volume = OcctExactEvaluator::new(&imported)
        .unwrap()
        .mass_properties(
            &imported.topology.bodies[0].mass_properties_evaluator_id,
            &Unlimited,
        )
        .unwrap()
        .volume_m3
        .canonical_value();
    (volume, imported)
}

const PIPE_METRE_UNIT: &str = "#114 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.) );";

// witness: FEA-MESH-GEOMETRY-001/geometry_load_units_option @
//   shared_infrastructure/mesh_independent/source_check; features: occt-native; coverage:
//   partial;
//   boundary: STEP files declaring centimetres (SI prefix) or feet (conversion-based unit with a
//   304.8 mm factor) import at their declared scale, and an authored metre unit over either reads
//   the coordinates in metres and records the declared unit.
#[test]
fn centimetre_and_foot_step_declarations_resolve_and_accept_authored_overrides() {
    let pipe = include_str!("../../../../docs/simulation/examples/geometry/pipe.step");
    assert!(pipe.contains(PIPE_METRE_UNIT));
    let centimetres = pipe.replacen(
        PIPE_METRE_UNIT,
        "#114 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.CENTI.,.METRE.) );",
        1,
    );
    let feet = pipe.replacen(
        PIPE_METRE_UNIT,
        "#114 = ( CONVERSION_BASED_UNIT('FOOT',#900) LENGTH_UNIT() NAMED_UNIT(#901) );\n\
         #900 = LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(304.8),#902);\n\
         #901 = DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.);\n\
         #902 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );",
        1,
    );
    let pipe_m3 = std::f64::consts::PI * 0.05 * 0.05 * 0.5;
    for (name, bytes, meters_per_unit) in [
        ("centimetres.step", centimetres, 0.01),
        ("feet.step", feet, 0.3048),
    ] {
        let (declared_volume, _) =
            step_volume_m3(name, bytes.as_bytes(), &ExactCadImportOptions::default());
        let scaled = pipe_m3 * meters_per_unit * meters_per_unit * meters_per_unit;
        assert!(
            (declared_volume / scaled - 1.0).abs() < 1.0e-9,
            "{name}: {declared_volume} vs {scaled}"
        );
        let (authored_volume, authored) = step_volume_m3(
            name,
            bytes.as_bytes(),
            &ExactCadImportOptions {
                source_units: Some(LengthUnit::Meter),
                source_units_authority: SourceUnitsAuthority::Authored,
                ..ExactCadImportOptions::default()
            },
        );
        assert!(
            (authored_volume / pipe_m3 - 1.0).abs() < 1.0e-9,
            "{name}: {authored_volume} vs {pipe_m3}"
        );
        let declared = authored
            .source_unit_evidence
            .overridden_declared_unit()
            .map(LengthUnit::meters_per_unit);
        assert!(
            (declared.unwrap() / meters_per_unit - 1.0).abs() < 1.0e-12,
            "{name}: {declared:?}"
        );
        authored.build_closure().unwrap();
    }
}

// witness: FEA-MESH-GEOMETRY-001/geometry_load_units_option @
//   shared_infrastructure/mesh_independent/source_check; features: occt-native; coverage:
//   partial;
//   boundary: Authored units over a STEP file whose global contexts declare several different
//   length units, or no length unit, are refused with GEOMETRY_UNITS_UNRESOLVED instead of
//   skipping the rescale. Both files are synthetic variants of the shipped fixtures.
#[test]
fn authored_units_refuse_step_files_without_one_declared_unit() {
    let tube = include_str!("../../../../docs/simulation/examples/geometry/compliant-tube.step");
    let mixed = tube.replacen(
        "#136 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.) );",
        "#136 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );",
        1,
    );
    assert_ne!(mixed, tube);
    let pipe = include_str!("../../../../docs/simulation/examples/geometry/pipe.step");
    let undeclared = pipe.replacen(
        PIPE_METRE_UNIT,
        "#114 = ( NAMED_UNIT(*) SI_UNIT($,.METRE.) );",
        1,
    );
    assert_ne!(undeclared, pipe);
    for (name, bytes) in [("mixed.step", mixed), ("undeclared.step", undeclared)] {
        let refused = import_exact_cad(
            name,
            bytes.as_bytes(),
            GeometryFormat::Step,
            &ExactCadImportOptions {
                source_units: Some(LengthUnit::Meter),
                source_units_authority: SourceUnitsAuthority::Authored,
                ..ExactCadImportOptions::default()
            },
            &GeometryImportContext::new(),
        )
        .unwrap_err();
        assert!(
            matches!(refused, GeometryImportError::UnresolvedUnits(_)),
            "{name}: {refused}"
        );
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

// witness: FEA-MESH-GEOMETRY-001/exact_formats @
//   shared_infrastructure/mesh_independent/source_check; features: occt-native; coverage:
//   partial;
//   boundary: A public NIST ACIS STEP part imports with one body, solid, shell, six faces, twelve
//   edges and eight vertices and kernel mass-property volume equal to the reference within 1e-12;
//   IGES and BREP and the rest of the corpus are not asserted here.
#[test]
fn public_industrial_step_imports_exact_topology_and_mass_properties() {
    let options = ExactCadImportOptions {
        source_units: Some(LengthUnit::Millimeter),
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
    assert!((mass.volume_m3.canonical_value() - 0.011_076_007_2).abs() < 1.0e-12);
    assert!((mass.surface_area_m2.canonical_value() - 2.269_505_468_8).abs() < 1.0e-12);
    assert!(mass.centroid_m.canonical_value()[0].abs() < 1.0e-12);
    assert!(mass.centroid_m.canonical_value()[1].abs() < 1.0e-12);
    assert!((mass.centroid_m.canonical_value()[2] + 0.005).abs() < 1.0e-12);
    imported.build_closure().expect("closure must validate");
    OcctExactEvaluator::from_closure(
        &imported.representation,
        imported.kernel_units.meters_per_unit(),
        &imported.topology,
        &imported.evaluators,
    )
    .expect("the distributed exact-geometry closure must reconstruct its evaluator");
}

#[test]
fn public_iges_wireframe_is_read_then_rejected_by_typed_admission() {
    let options = ExactCadImportOptions {
        source_units: Some(LengthUnit::Millimeter),
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
        hex::encode(Sha256::digest(&bytes)),
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

/// One solid ("Pack negative perimeter cable") carried out of an exported EV battery pack. The
/// writer stamps a 0.9 nm tolerance on every edge, while its swept B-spline pcurves sit up to
/// 23 nm from their 3D curves. Import rebuilds those pcurves and admits each remaining gap, far
/// within the request's 0.1 mm geometric tolerance, by raising the edge's tolerance to the
/// measured gap. Every such edge is recorded, and the file passes incidence validation.
#[test]
fn exported_pack_cable_with_nanometre_pcurve_gaps_imports() {
    const CABLE: &[u8] = include_bytes!("fixtures/battery_pack_perimeter_cable.step");
    let imported = import_exact_cad(
        "battery_pack_perimeter_cable.step",
        CABLE,
        GeometryFormat::Step,
        &ExactCadImportOptions {
            source_units: Some(LengthUnit::Millimeter),
            ..ExactCadImportOptions::default()
        },
        &GeometryImportContext::new(),
    )
    .unwrap();
    let tolerances: Vec<f64> = imported
        .topology
        .edges
        .iter()
        .map(|edge| edge.tolerance_m.canonical_value())
        .collect();
    let written = tolerances.iter().copied().fold(f64::INFINITY, f64::min);
    let raised = tolerances.iter().copied().fold(0.0_f64, f64::max);
    assert!(written < 1.0e-9, "written edge tolerance {written:e} m");
    assert!(
        raised < 1.0e-6,
        "raised edge tolerance {raised:e} m against written {written:e} m"
    );
    assert!(!imported.edge_tolerance_admissions.is_empty());
    for admission in &imported.edge_tolerance_admissions {
        assert!(admission.admitted, "{}", admission.describe());
        assert!(admission.gap_before_m > admission.written_tolerance_m);
        assert!(admission.gap_after_m <= admission.gap_before_m);
        assert!(admission.admitted_tolerance_m <= raised);
    }
    assert_eq!(imported.edge_tolerance_defects().count(), 0);
    assert_eq!(imported.topology.solids.len(), 1);
}
