//! The drone's landing skid exports exact STEP whose every edge lies on its faces within the
//! edge's own tolerance, at the thin rail depths the touchdown study compares (2.0 and 2.5 mm).
//! At those depths the struts' base plane falls inside the band of the 1.2 mm sole fillet, and
//! the fillet's tangent edge on the rail's rear end once exported 26 µm off its surface with a
//! 1.9 µm tolerance, which the FEA mesher refuses.

use std::path::PathBuf;

struct Unlimited;

impl runmat_geometry_core::GeometryEvaluationControl for Unlimited {
    fn checkpoint(&self) -> Result<(), runmat_geometry_core::GeometryEvaluationError> {
        Ok(())
    }

    fn consume_iterations(&self, _count: u64) -> Result<(), runmat_geometry_core::GeometryEvaluationError> {
        Ok(())
    }

    fn consume_search_work(&self, _count: u64) -> Result<(), runmat_geometry_core::GeometryEvaluationError> {
        Ok(())
    }

    fn consume_allocation_bytes(&self, _count: u64) -> Result<(), runmat_geometry_core::GeometryEvaluationError> {
        Ok(())
    }
}

fn skid_at(rail_thickness_mm: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../runmat-cad/wasm/tests/drone/mechanical/body/landing_skid.part");
    let text = std::fs::read_to_string(&path).expect("the landing skid is checked in");
    let default = "input rail_thickness : length = 3.5 mm";
    assert!(text.contains(default), "landing_skid.part declares {default:?}");
    text.replace(default, &format!("input rail_thickness : length = {rail_thickness_mm} mm"))
}

#[test]
fn the_thin_skid_exports_every_edge_on_its_faces() {
    runmat_geometry_wasm_host::cad::install_occt();
    for thickness in ["2.0", "2.5"] {
        let text = skid_at(thickness);
        let step = runmat_cad_wasm::exchange::part_step("/p/drone/mechanical/body/landing_skid.part", &text)
            .unwrap_or_else(|e| panic!("{thickness} mm: {e}"))
            .expect("OCCT writes STEP");
        let imported = runmat_geometry_io::import_exact_cad(
            "landing_skid.step",
            &step,
            runmat_geometry_io::GeometryFormat::Step,
            &runmat_geometry_io::ExactCadImportOptions::default(),
            &runmat_geometry_io::GeometryImportContext::new(),
        )
        .unwrap_or_else(|e| panic!("{thickness} mm: the STEP imports back: {e:?}"));
        assert_eq!(imported.topology.bodies.len(), 1, "{thickness} mm: one solid");
        // A request tolerance far below any edge's: each edge is held to its own tolerance, the
        // gap the FEA mesher admits between an edge's curve and its faces.
        runmat_geometry_io::OcctExactEvaluator::new(&imported)
            .unwrap()
            .validate_incidence_consistency(&imported.topology, 1.0e-12, &Unlimited)
            .unwrap_or_else(|e| panic!("{thickness} mm: {e:?}"));
    }
}
