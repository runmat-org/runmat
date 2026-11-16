//! The browser meshing host sends live work counts while a mesh request runs: the same entry
//! the browser worker calls (`execution::execute`), with a page handler that collects events.

use std::sync::{Arc, Mutex};

use runmat_analysis_execution::ImportedSourceMeshingRequest;
use runmat_analysis_fea::{FeaProgressEvent, FeaProgressPhase, FeaProgressStatus};
use runmat_cad::regen::kernel::{ExchangeDocument, ExchangeFormat, ExchangePart};
use runmat_cad_wasm::runtime::regenerate_part;
use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::{LogicalObject, ObjectNamespace};

/// A twelve-sided prism: 14 faces, so the browser executor meshes two face partitions.
const PRISM: &str = r#"runmat 0.7
std 1

part prism {
    sketch s on origin.xy {
        l1 = line(start = [40 mm, 0 mm], end = [34.64 mm, 20 mm])
        l2 = line(start = l1.end, end = [20 mm, 34.64 mm])
        l3 = line(start = l2.end, end = [0 mm, 40 mm])
        l4 = line(start = l3.end, end = [-20 mm, 34.64 mm])
        l5 = line(start = l4.end, end = [-34.64 mm, 20 mm])
        l6 = line(start = l5.end, end = [-40 mm, 0 mm])
        l7 = line(start = l6.end, end = [-34.64 mm, -20 mm])
        l8 = line(start = l7.end, end = [-20 mm, -34.64 mm])
        l9 = line(start = l8.end, end = [0 mm, -40 mm])
        l10 = line(start = l9.end, end = [20 mm, -34.64 mm])
        l11 = line(start = l10.end, end = [34.64 mm, -20 mm])
        l12 = line(start = l11.end, end = l1.start)
    }
    feature body = extrude(profile = sketch.s, distance = 30 mm)
}
"#;

fn prism_step() -> Vec<u8> {
    runmat_geometry_wasm_host::cad::install_occt();
    let part = regenerate_part("/p/prism.part", PRISM).expect("compiles");
    let doc = ExchangeDocument {
        name: "prism".into(),
        parts: vec![ExchangePart {
            name: "prism".into(),
            bodies: part.result.state.bodies.iter().map(|b| &b.handle).collect(),
            color: None,
            material: None,
            face_names: Vec::new(),
            face_colors: Vec::new(),
        }],
        occurrences: Vec::new(),
    };
    part.kernel
        .export(&doc, ExchangeFormat::Ap242)
        .expect("OCCT writes STEP")
}

fn mesh_request(step: Vec<u8>) -> String {
    let expected = Digest::sha256(&step);
    let request = ImportedSourceMeshingRequest {
        source_name: "prism.step".into(),
        source: LogicalObject::new(
            ObjectNamespace::InputValue,
            "input/exact-cad-source",
            "application/octet-stream",
            step,
        )
        .expect("source object"),
        expected_source_digest: expected,
        import_options: Default::default(),
        settings: Default::default(),
        contacts: Vec::new(),
        program_revision: ProgramRevision::new(
            Digest::sha256(b"graph"),
            Digest::sha256(b"source"),
            ProgramEnvironment::new(
                1,
                1,
                Digest::sha256(b"runtime"),
                Digest::sha256(b"catalog"),
                "matlab",
            )
            .expect("environment"),
        )
        .expect("revision"),
        artifact_access: runmat_analysis_execution::checked_fea_artifact_access(
            "test:browser-meshing-progress",
        ),
    };
    serde_json::json!({ "operation": "mesh_imported_source", "request": request }).to_string()
}

fn unit(event: &FeaProgressEvent) -> &str {
    event.metadata.get("unit").map(String::as_str).unwrap_or("")
}

#[test]
fn a_mesh_request_sends_live_counts_while_it_meshes() {
    let json = mesh_request(prism_step());
    let events = Arc::new(Mutex::new(Vec::<FeaProgressEvent>::new()));
    let sink = Arc::clone(&events);
    let result = runmat_geometry_wasm_host::execution::execute(
        &json,
        Some(Arc::new(move |event| sink.lock().unwrap().push(event))),
    )
    .expect("the prism meshes");
    assert!(result.contains("\"artifact\""));
    let collected = events.lock().unwrap().clone();
    // The observer leaves with the request: a later request without a page reaches nothing.
    runmat_geometry_wasm_host::execution::execute(&json, None).expect("meshes again");
    assert_eq!(events.lock().unwrap().len(), collected.len());
    let events = collected;

    assert!(!events.is_empty(), "no progress while meshing");
    for event in &events {
        assert_eq!(event.phase, FeaProgressPhase::MeshPrep);
        assert_eq!(event.status, FeaProgressStatus::Advanced);
        assert_eq!(event.operation, "fea.meshing");
    }
    let faces: Vec<_> = events
        .iter()
        .filter(|e| unit(e) == "surface_faces")
        .collect();
    assert!(!faces.is_empty(), "no face counts: {events:?}");
    // Counts across the two face partitions add up and never go backwards.
    for pair in faces.windows(2) {
        assert!(pair[1].current >= pair[0].current, "{faces:?}");
    }
    let last = faces.last().unwrap();
    assert!(
        last.total.unwrap() > 8,
        "partitions were not added up: {faces:?}"
    );
    assert!(
        events.iter().any(|e| unit(e) == "volume_points"),
        "no volume point counts: {events:?}"
    );
}
