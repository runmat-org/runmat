//! The command registry's build check: every catalog command runs on every surface its route
//! claims. Each command is run through `commands::run` (the one dispatcher the Desktop host, the
//! agent's tools and the CLI share) on a small file of each editor kind that offers it, with no
//! values and no selection. A refusal (a pick missing, a value out of range) is an answer; a
//! command that reaches a dispatcher with no code for it fails the build. A command with no
//! handler yet is listed in its catalog's `UNBUILT`, so no surface offers it.

use runmat_design_commands::EditorKind;
use runmat_design_tools::commands::{self, Handler, RunRequest, Surface};
use runmat_design_tools::Files;
use serde_json::{json, Value};

const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])\n        l2 = line(start = l1.end, end = [80 mm, 50 mm])\n        l3 = line(start = l2.end, end = [0 mm, 50 mm])\n        l4 = line(start = l3.end, end = l1.start)\n    }\n    feature body = extrude(profile = sketch.s, distance = 10 mm)\n    output plate = { body = feature.body.body, material = \"material.aluminum_6061\" }\n}\n";

const PAIR: &str = "runmat 0.7\nstd 1\n\nassembly pair {\n    use part plate as lower\n    use part plate as upper {\n        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 20 mm] } }\n    }\n}\n";

const SHEET: &str = r#"runmat 0.7
std 1

drawing sheet {
    format a3 = {
        size = { size = iso, name = A3 },
        title_block = [{ label = Sheet, source = { source = sheet_number } }]
    }

    sheet "1" = { format = a3, scale = [1, 1], projection = third_angle }
    sheet "2" = { format = a3, scale = [1, 1], projection = third_angle }

    view front = standard(
        orientation = front,
        sheet = "1",
        model = { part = plate },
        at = [130, 120],
        display = hidden_lines
    )
}
"#;

/// The circuit editors run on the ECO scenario's checked-in project: a schematic with its board
/// and lock (`runmat-circuit/tests/projects/io`).
const IO_CIRCUIT: &str = include_str!("../../runmat-circuit/tests/projects/io/io.circuit");
const IO_LAYOUT: &str = include_str!("../../runmat-circuit/tests/projects/io/io.circuit.layout");
const IO_LOCK: &str = include_str!("../../runmat-circuit/tests/projects/io/io.circuit.lock");

fn files() -> Files {
    [
        ("/p/plate.part", PLATE),
        ("/p/pair.assembly", PAIR),
        ("/p/sheet.drawing", SHEET),
        ("/p/io.circuit", IO_CIRCUIT),
        ("/p/io.circuit.layout", IO_LAYOUT),
        ("/p/io.circuit.lock", IO_LOCK),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v.to_owned()))
    .collect()
}

fn request(command: &str, values: Value, selection: &[&str]) -> RunRequest {
    serde_json::from_value(json!({ "command": command, "values": values, "selection": selection }))
        .unwrap()
}

fn run(path: &str, request: &RunRequest) -> (Value, bool) {
    let files = files();
    commands::take_unhandled();
    let out = commands::run(path, &files[path], &files, request);
    (out, commands::take_unhandled())
}

/// The MCAD and ECAD editors' commands: none reaches a dispatcher without a handler.
#[test]
fn every_design_command_has_a_handler_on_its_surfaces() {
    let mut missing = Vec::new();
    for route in commands::routes() {
        let path = match route.editor {
            EditorKind::Part => "/p/plate.part",
            EditorKind::Assembly => "/p/pair.assembly",
            EditorKind::Drawing => "/p/sheet.drawing",
            EditorKind::Circuit => "/p/io.circuit",
            EditorKind::CircuitLayout => "/p/io.circuit.layout",
            _ => continue,
        };
        if route.handler == Handler::Shell {
            assert_eq!(route.surfaces, [Surface::Desktop], "{}", route.id);
            continue;
        }
        assert!(
            route.surfaces.contains(&Surface::Agent) && route.surfaces.contains(&Surface::Cli),
            "{}",
            route.id
        );
        // A sketch tool runs in the open sketch.
        let base = if route.id == "board.autoroute" {
            // Check this dispatcher on one authored net. Routing every open connection of
            // the full I/O board belongs to routing qualification, not handler inventory.
            // The circuit autoroute_power tests retain routing and clearance assertions.
            json!({ "nets": ["net:/ch[0]/FBK"] })
        } else if route.id.starts_with("sketch.") && route.editor == EditorKind::Part {
            json!({ "sketch": "s" })
        } else {
            json!({})
        };
        let (out, unhandled) = run(path, &request(&route.id, base.clone(), &[]));
        if route.id == "board.autoroute" {
            assert_eq!(out["ok"], true, "{}: {out}", route.editor.as_str());
        }
        if unhandled {
            missing.push(format!("{} {}", route.id, route.editor.as_str()));
        }
        // A choice the form offers (a format, a type) runs too.
        let descriptor = commands::catalog().command(&route.id).unwrap();
        for field in descriptor
            .inputs
            .iter()
            .filter(|f| f.id == "format" || f.id == "type")
        {
            let options: Value = serde_json::to_value(field).unwrap();
            for option in options["options"].as_array().into_iter().flatten() {
                let value = option["value"].clone();
                let mut values = base.clone();
                values[field.id.as_str()] = value.clone();
                let (_, unhandled) = run(path, &request(&route.id, values, &[]));
                if unhandled {
                    missing.push(format!(
                        "{} {} {}={}",
                        route.id,
                        route.editor.as_str(),
                        field.id,
                        value.as_str().unwrap_or("")
                    ));
                }
            }
        }
    }
    // The ledger of commands the catalog offers with no handler yet, one per line. Each line
    // leaves the ledger in the commit that builds its handler (or withdraws it into `UNBUILT`);
    // a command missing a handler that isn't listed fails here, and so does a listed one that
    // now has one, so the ledger is always the exact list of gaps.
    missing.sort();
    missing.dedup();
    let ledger_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/command_gaps.txt");
    if std::env::var_os("RUNMAT_COMMAND_GAPS_WRITE").is_some() {
        std::fs::write(ledger_path, missing.join("\n") + "\n").unwrap();
    }
    let ledger: Vec<String> = std::fs::read_to_string(ledger_path)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(str::to_owned)
        .collect();
    let new: Vec<&String> = missing.iter().filter(|m| !ledger.contains(m)).collect();
    let closed: Vec<&String> = ledger.iter().filter(|l| !missing.contains(l)).collect();
    assert!(
        new.is_empty(),
        "catalog commands with no handler (build one, or list the command in UNBUILT):\n{}",
        new.iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert!(
        closed.is_empty(),
        "these commands have handlers now; take them out of tests/command_gaps.txt:\n{}",
        closed
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
}

fn occurrence(name: &str) -> String {
    let files = files();
    let (index, _) =
        runmat_cad_wasm::runtime::assembly_index("/p/pair.assembly", PAIR, &files).unwrap();
    let uuid = index["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .find(|o| o["name"] == name)
        .unwrap()["uuid"]
        .as_str()
        .unwrap()
        .to_owned();
    format!("occurrence:{uuid}")
}

#[test]
fn bom_lists_the_plate_twice_in_every_type() {
    for kind in ["top", "indented", "parts"] {
        let (out, _) = run(
            "/p/pair.assembly",
            &request("assembly.bom", json!({ "type": kind }), &[]),
        );
        assert_eq!(out["ok"], true, "{kind}: {out}");
        assert_eq!(out["kind"], "table");
        assert_eq!(out["title"], "BOM");
        let rows = out["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 1, "{kind}: {out}");
        assert_eq!(rows[0]["cells"][2], "plate.part", "{out}");
        assert_eq!(rows[0]["cells"][6], "2", "{out}");
        assert_eq!(
            rows[0]["items"].as_array().unwrap().len(),
            2,
            "both occurrences select: {out}"
        );
    }
}

#[test]
fn where_used_lists_every_occurrence_of_the_selected_file() {
    let upper = occurrence("upper");
    let (out, _) = run(
        "/p/pair.assembly",
        &request(
            "assembly.where-used",
            json!({}),
            &[&format!("{upper}/face:0:1")],
        ),
    );
    assert_eq!(out["ok"], true, "{out}");
    assert_eq!(out["rows"].as_array().unwrap().len(), 2, "{out}");
    assert!(
        out["summary"]
            .as_str()
            .unwrap()
            .contains("plate.part is used 2 times"),
        "{out}"
    );
    let (out, _) = run(
        "/p/pair.assembly",
        &request("assembly.where-used", json!({}), &[]),
    );
    assert_eq!(out["ok"], false, "{out}");
}

#[test]
fn a_drawing_exports_as_pdf_dxf_and_dwg() {
    let (pdf, _) = run(
        "/p/sheet.drawing",
        &request(
            "drawing.export",
            json!({ "format": "pdf", "sheets": "all" }),
            &[],
        ),
    );
    assert_eq!(pdf["ok"], true, "{pdf}");
    assert_eq!(pdf["kind"], "file");
    assert_eq!(pdf["mime"], "application/pdf");
    assert_eq!(pdf["name"], "sheet.pdf", "both sheets in one PDF: {pdf}");
    let (dxf, _) = run(
        "/p/sheet.drawing",
        &request(
            "drawing.export",
            json!({ "format": "dxf", "sheets": "current", "sheet": "2" }),
            &[],
        ),
    );
    assert_eq!(dxf["ok"], true, "{dxf}");
    assert_eq!(dxf["name"], "sheet-2.dxf", "{dxf}");
    assert!(dxf["text"].as_str().unwrap().contains("SECTION"), "{dxf}");
    let (dwg, _) = run(
        "/p/sheet.drawing",
        &request(
            "drawing.export",
            json!({ "format": "dwg", "sheets": "all" }),
            &[],
        ),
    );
    assert_eq!(
        dwg["files"].as_array().unwrap().len(),
        2,
        "one DWG per sheet: {dwg}"
    );
}

/// Sketch on a picked plane writes the new sketch (the Desktop's Sketch command form and the agent).
#[test]
fn sketch_on_a_plane_writes_the_new_sketch() {
    let route = commands::route("part.sketch", EditorKind::Part).expect("part.sketch");
    assert_eq!(route.handler, Handler::Edit);
    let (out, unhandled) = run(
        "/p/plate.part",
        &request("part.sketch", json!({ "plane": "origin.xy" }), &[]),
    );
    assert_eq!(out["ok"], true, "{out}");
    assert!(!unhandled);
}

#[test]
fn view_commands_run_only_in_the_desktop() {
    let route = commands::route("view.fit", EditorKind::Part).expect("view.fit");
    assert_eq!(route.handler, Handler::Shell);
    let (out, unhandled) = run("/p/plate.part", &request("view.fit", json!({}), &[]));
    assert_eq!(out["ok"], false);
    assert!(!unhandled);
}

#[test]
fn a_drawing_exports_as_svg_and_an_assembly_as_usd() {
    let (svg, _) = run(
        "/p/sheet.drawing",
        &request(
            "drawing.export",
            json!({ "format": "svg", "sheets": "current", "sheet": "1" }),
            &[],
        ),
    );
    assert_eq!(svg["ok"], true, "{svg}");
    assert_eq!(svg["mime"], "image/svg+xml");
    assert!(
        svg["text"]
            .as_str()
            .unwrap()
            .contains("viewBox=\"0 0 420 297\""),
        "A3 at true size: {svg}"
    );
    let (usd, _) = run(
        "/p/pair.assembly",
        &request("assembly.export", json!({ "format": "usd" }), &[]),
    );
    assert_eq!(usd["ok"], true, "{usd}");
    assert_eq!(usd["name"], "pair.usda");
    assert_eq!(
        usd["text"].as_str().unwrap().matches("def Mesh").count(),
        2,
        "one mesh per placed plate: {usd}"
    );
}

#[test]
fn pack_and_share_copies_the_assembly_and_its_parts_into_the_folder() {
    let (out, _) = run(
        "/p/pair.assembly",
        &request(
            "assembly.pack-and-share",
            json!({ "folder": "pair-pack" }),
            &[],
        ),
    );
    assert_eq!(out["ok"], true, "{out}");
    let mut names: Vec<&str> = out["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["name"].as_str().unwrap())
        .collect();
    names.sort();
    assert_eq!(names, ["pair-pack/pair.assembly", "pair-pack/plate.part"]);
}
