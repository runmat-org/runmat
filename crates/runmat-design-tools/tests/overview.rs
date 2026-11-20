//! `design.overview` end to end (doc 18 §9): every fact on an overview comes from a check that
//! ran, links go both ways (a part's study and drawing, a schematic's board), and the counts are
//! the check's.

use runmat_design_tools::overview::{related_candidates, Level, Relation};
use runmat_design_tools::{call, Files};
use serde_json::{json, Value};

const IO_CIRCUIT: &str = include_str!("../../runmat-circuit/tests/projects/io/io.circuit");
const IO_LAYOUT: &str = include_str!("../../runmat-circuit/tests/projects/io/io.circuit.layout");
const IO_LOCK: &str = include_str!("../../runmat-circuit/tests/projects/io/io.circuit.lock");

const PART: &str = r#"runmat 0.7
std 1

part plate {
    input width : length = 80 mm
    input depth : length = 50 mm
    input thickness : length = 10 mm

    sketch s1 on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])
        l2 = line(start = [input.width, 0 mm], end = [input.width, input.depth])
        l3 = line(start = [input.width, input.depth], end = [0 mm, input.depth])
        l4 = line(start = [0 mm, input.depth], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s1, distance = input.thickness)

    region mount : face = { select = [feature.body.face.start], expect = exactly_one }
}
"#;

const STUDY: &str = r#"geometry plate {
    file = "../parts/plate.part"
    units = mm
    region mount : face { select name "mount"; expect exactly_one }
}

study plate_static {
    geometry = geometry.plate
    analysis static { structural static {} }
}
"#;

fn overview(path: &str, files: &Files) -> Value {
    call("design.overview", json!({ "path": path }), files)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// Every fact names a check that ran, every link a check of the linked file, and the counts
/// come from the check.
fn assert_grounded(out: &Value) {
    let checks: Vec<&str> = out["checks"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .collect();
    for fact in out["state"].as_array().unwrap() {
        let source = fact["source"].as_str().unwrap();
        assert!(
            checks.contains(&source),
            "{fact} names {source}, which didn't run ({checks:?})"
        );
        assert!(!fact["value"].as_str().unwrap().is_empty(), "{fact}");
    }
    for link in out["links"].as_array().unwrap() {
        assert!(checks.contains(&link["source"].as_str().unwrap()), "{link}");
    }
    assert!(checks.contains(&"design.check"), "{checks:?}");
    assert!(out["counts"]["totals"].is_object(), "{}", out["counts"]);
    assert!(out.get("nextSteps").is_none() && out.get("openDecisions").is_none());
}

#[test]
fn a_schematic_overview_reads_erc_its_board_and_its_simulation_sections() {
    let files: Files = [
        ("/b/io.circuit".to_owned(), IO_CIRCUIT.to_owned()),
        ("/b/io.circuit.layout".to_owned(), IO_LAYOUT.to_owned()),
        ("/b/io.circuit.lock".to_owned(), IO_LOCK.to_owned()),
    ]
    .into();
    let out = overview("/b/io.circuit", &files);
    assert_grounded(&out);
    assert_eq!(out["fileState"], "written");
    assert!(
        out["about"].as_str().unwrap().starts_with("Schematic: "),
        "{}",
        out["about"]
    );
    let ids: Vec<&str> = out["state"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|f| f["id"].as_str())
        .collect();
    for id in ["erc", "board_sync", "placement", "drc", "simulation"] {
        assert!(ids.contains(&id), "{id} missing from {ids:?}");
    }
    let links = out["links"].as_array().unwrap();
    assert_eq!(links.len(), 1, "{links:#?}");
    assert_eq!(links[0]["path"], "/b/io.circuit.layout");
    assert_eq!(links[0]["relation"], json!(Relation::Board));
    // The board's overview links back to its schematic.
    let board = overview("/b/io.circuit.layout", &files);
    assert_grounded(&board);
    assert_eq!(board["links"][0]["relation"], json!(Relation::Schematic));
    assert!(board["about"].as_str().unwrap().starts_with("Board: "));
}

#[test]
fn a_part_overview_links_its_study_and_drawing() {
    // The agent scenes' drawing (scripts/site-captures/agent-scenes/files/bracket.drawing), of the plate.
    let drawing = r#"runmat 0.7
std 1

drawing plate {
    format a3 = {
        size = { size = iso, name = A3 },
        title_block = [
            { label = "Part number", source = { source = property, name = part_number } },
            { label = Rev, source = { source = revision } },
            { label = Sheet, source = { source = sheet_number } }
        ]
    }

    sheet "1" = { format = a3, scale = [2, 1], projection = third_angle }

    view front = standard(
        orientation = front,
        sheet = "1",
        model = { part = plate },
        at = [130, 120],
        display = hidden_lines
    )
    view top = projected(
        from = front,
        direction = up,
        sheet = "1",
        model = { part = plate },
        at = [130, 215],
        display = hidden_lines
    )
    view right = projected(
        from = front,
        direction = right,
        sheet = "1",
        model = { part = plate },
        at = [290, 120],
        display = hidden_lines
    )
}
"#;
    let files: Files = [
        ("/parts/plate.part".to_owned(), PART.to_owned()),
        ("/parts/plate.drawing".to_owned(), drawing.to_owned()),
        ("/studies/plate.fea".to_owned(), STUDY.to_owned()),
    ]
    .into();
    let out = overview("/parts/plate.part", &files);
    assert_grounded(&out);
    let links = out["links"].as_array().unwrap();
    let study = links
        .iter()
        .find(|l| l["relation"] == json!(Relation::Study))
        .expect("the study");
    assert_eq!(study["path"], "/studies/plate.fea");
    assert_eq!(study["level"], json!(Level::Ok), "{study}");
    assert!(
        study["status"].as_str().unwrap().contains("plate_static"),
        "{study}"
    );
    let sheet = links
        .iter()
        .find(|l| l["relation"] == json!(Relation::Drawing))
        .expect("the drawing");
    assert_eq!(sheet["path"], "/parts/plate.drawing");
    let inputs = out["state"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["id"] == "inputs")
        .expect("inputs");
    assert_eq!(inputs["value"], "width 80 mm, depth 50 mm, thickness 10 mm");
    // A part with nothing linked has no links and the check's counts.
    let alone: Files = [("/parts/plate.part".to_owned(), PART.to_owned())].into();
    let out = overview("/parts/plate.part", &alone);
    assert_grounded(&out);
    assert_eq!(out["links"], json!([]));
    assert_eq!(out["counts"]["ok"], true, "{}", out["counts"]);
    assert_eq!(out["counts"]["totals"]["bodies"], 1, "{}", out["counts"]);
    // What the part holds, from its index: a machined plate is a written part.
    assert_eq!(out["fileState"], "written");
}

#[test]
fn candidates_leave_out_the_file_and_non_design_files() {
    let project = ["/a.part", "/b.fea", "/c.m", "/d.circuit.lock", "/e.drawing"]
        .map(String::from)
        .to_vec();
    assert_eq!(
        related_candidates("/a.part", &project),
        ["/b.fea", "/e.drawing"]
    );
}
