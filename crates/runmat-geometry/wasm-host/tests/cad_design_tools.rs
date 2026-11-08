//! The design tools in the CAD host (`designTool` through `runmat_cad_call`, the browser worker's
//! entry): the agent's tools on parts and assemblies regenerate on OCCT here, as they do on the
//! server. A new tube part is staged as created, an assembly fastens two tubes to a plate with
//! mates, and inspect finds the plate and a tube sunk into it overlapping, with each part's mass
//! from its material.

use serde_json::{json, Value};

const PLATE: &str = r#"runmat 0.7
std 1

part plate {
    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        l2 = line(start = l1.end, end = [80 mm, 50 mm])
        l3 = line(start = l2.end, end = [0 mm, 50 mm])
        l4 = line(start = l3.end, end = l1.start)
    }
    feature body = extrude(profile = sketch.s, distance = 10 mm)
    output plate = { body = feature.body.body, material = "material.aluminum_6061" }
}
"#;

const TUBE: &str = r#"runmat 0.7
std 1

part tube {
    sketch s on origin.yz {
        outer = circle(center = [0 mm, 0 mm], radius = 5 mm)
        inner = circle(center = [0 mm, 0 mm], radius = 4 mm)
    }
    feature body = extrude(profile = sketch.s, distance = 120 mm)
    output tube = { body = feature.body.body, material = "material.aluminum_6061" }
}
"#;

/// Two tubes fastened under the plate (`sink` mm into it): connectors at the plate's underside
/// and at each tube's top, joined by fasten mates.
fn skid(sink: f64) -> String {
    let z = 5.0 - sink;
    format!(
        "runmat 0.7\nstd 1\n\nassembly skid {{\n    use part plate as plate\n    use part tube as left {{\n        fixed = false\n        placement = {{ placement = offset, offset = {{ translation = [0 mm, 10 mm, -20 mm] }} }}\n    }}\n    use part tube as right {{\n        fixed = false\n        placement = {{ placement = offset, offset = {{ translation = [0 mm, 40 mm, -20 mm] }} }}\n    }}\n    connector under_left = {{ occurrence = [{{ name = plate }}], anchor = {{ anchor = explicit, offset = {{ translation = [-20 mm, 10 mm, 0 mm] }} }} }}\n    connector under_right = {{ occurrence = [{{ name = plate }}], anchor = {{ anchor = explicit, offset = {{ translation = [-20 mm, 40 mm, 0 mm] }} }} }}\n    connector left_top = {{ occurrence = [{{ name = left }}], anchor = {{ anchor = explicit, offset = {{ translation = [0 mm, 0 mm, {z} mm] }} }} }}\n    connector right_top = {{ occurrence = [{{ name = right }}], anchor = {{ anchor = explicit, offset = {{ translation = [0 mm, 0 mm, {z} mm] }} }} }}\n    mate left_mount = {{ kind = {{ mate = fasten }}, a = under_left, b = left_top }}\n    mate right_mount = {{ kind = {{ mate = fasten }}, a = under_right, b = right_top }}\n}}\n"
    )
}

fn tool(name: &str, input: Value, files: &[(&str, &str)]) -> Value {
    let files: serde_json::Map<String, Value> = files.iter().map(|(p, t)| (p.to_string(), Value::String(t.to_string()))).collect();
    let request = json!({ "fn": "designTool", "args": [name, input.to_string(), Value::Object(files).to_string()] });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

#[test]
fn a_new_part_stages_as_created_and_checks_on_occt() {
    let out = tool("design.propose_edit", json!({ "path": "/p/tube.part", "text": TUBE }), &[]);
    let data = out.get("data").unwrap_or_else(|| panic!("a staged change, not {out}"));
    assert_eq!(data["files"][0]["created"], true, "{data}");
    assert_eq!(data["ok"], true, "{:#}", data["diagnostics"]);
    let check = tool("design.check", json!({ "path": "/p/tube.part" }), &[("/p/tube.part", TUBE)]);
    assert_eq!(check["data"]["ok"], true, "{check}");
    // A file that isn't there is the tool's own error, not the host's.
    let missing = tool("design.check", json!({ "path": "/p/none.part" }), &[]);
    assert_eq!(missing["error"]["code"], "file_missing", "{missing}");
}

#[test]
fn mates_place_the_tubes_and_inspect_finds_a_sunk_tube_overlapping() {
    let clear = skid(0.0);
    let files = [("/p/plate.part", PLATE), ("/p/tube.part", TUBE), ("/p/skid.assembly", clear.as_str())];
    let check = tool("design.check", json!({ "path": "/p/skid.assembly" }), &files);
    assert_eq!(check["data"]["errors"], 0, "{check:#}");
    let out = tool("design.inspect", json!({ "path": "/p/skid.assembly", "sections": ["mates", "interference", "mass"] }), &files);
    let sections = &out["data"]["sections"];
    assert_eq!(sections["mates"].as_array().map(Vec::len), Some(2), "{out:#}");
    assert_eq!(sections["interference"]["result"]["overlaps"].as_array().map(Vec::len), Some(0), "{out:#}");
    assert_eq!(sections["interference"]["problems"].as_array().map(Vec::len), Some(0), "{out:#}");
    // 40 cm³ of plate and two 3.39 cm³ tubes, all aluminium: 108 g + 2 × 9.2 g.
    let total = sections["mass"]["result"]["total"]["mass"].as_f64().unwrap_or_else(|| panic!("{out:#}"));
    assert!((total - 0.1263).abs() < 0.001, "{total}");

    let sunk = skid(2.0);
    let files = [("/p/plate.part", PLATE), ("/p/tube.part", TUBE), ("/p/skid.assembly", sunk.as_str())];
    let out = tool("design.inspect", json!({ "path": "/p/skid.assembly", "sections": ["interference"] }), &files);
    let overlaps = out["data"]["sections"]["interference"]["result"]["overlaps"].as_array().cloned().unwrap_or_default();
    assert_eq!(overlaps.len(), 2, "{out:#}");
}

#[test]
fn an_overview_here_lists_the_drawing_and_assembly_that_use_a_part_without_regenerating_them() {
    let drawing = "runmat 0.7\nstd 1\n\ndrawing tube {\n    format a3 = { size = { size = iso, name = A3 }, title_block = [{ label = \"Title\", source = { source = literal, text = \"Tube\" } }] }\n    sheet \"1\" = { format = a3, scale = [1, 1], projection = third_angle }\n    view front = standard(orientation = front, sheet = \"1\", model = { part = tube }, at = [200, 150], display = hidden_lines)\n}\n";
    let clear = skid(0.0);
    let files = [("/p/plate.part", PLATE), ("/p/tube.part", TUBE), ("/p/skid.assembly", clear.as_str()), ("/p/tube.drawing", drawing)];
    let out = tool("design.overview", json!({ "path": "/p/tube.part" }), &files);
    let links = out["data"]["links"].as_array().cloned().unwrap_or_else(|| panic!("links, not {out:#}"));
    for user in ["/p/skid.assembly", "/p/tube.drawing"] {
        let link = links.iter().find(|l| l["path"] == user).unwrap_or_else(|| panic!("{user} in {out:#}"));
        assert_eq!(link["status"], runmat_design_tools::overview::NOT_CHECKED_HERE, "{link}");
    }
    // The part itself is checked on OCCT.
    assert!(out["data"].to_string().contains("design.check"), "{out:#}");
}

/// Every file of the camera drone fixture, keyed `/p/drone/<relative path>`.
fn drone() -> Vec<(String, String)> {
    fn walk(dir: &std::path::Path, at: &str, all: &mut Vec<(String, String)>) {
        for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') || name.ends_with(".step") {
                continue;
            }
            if e.path().is_dir() {
                walk(&e.path(), &format!("{at}/{name}"), all);
            } else if let Ok(text) = std::fs::read_to_string(e.path()) {
                all.push((format!("{at}/{name}"), text));
            }
        }
    }
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../runmat-cad/wasm/tests/drone");
    let mut all = Vec::new();
    walk(&dir, "/p/drone", &mut all);
    assert!(all.iter().any(|(p, _)| p == "/p/drone/drone.assembly"), "the drone project");
    all
}

#[test]
fn on_the_drone_a_change_checks_only_its_pairs_and_a_ray_finds_the_lower_shell() {
    let all = drone();
    let files: Vec<(&str, &str)> = all.iter().map(|(p, t)| (p.as_str(), t.as_str())).collect();
    // The skid as the change: only its pairs are checked, and it overlaps nothing on OCCT.
    let out = tool("design.inspect", json!({ "path": "/p/drone/drone.assembly", "sections": ["interference"], "changed": ["skid_left", "skid_right"] }), &files);
    let interference = &out["data"]["sections"]["interference"];
    assert_eq!(interference["result"]["changed"], json!(["skid_left", "skid_right"]), "{out:#}");
    assert_eq!(interference["result"]["overlaps"].as_array().map(Vec::len), Some(0), "{out:#}");
    assert_eq!(interference["problems"].as_array().map(Vec::len), Some(0), "{out:#}");
    // Straight up from under the belly, limited to the lower shell: the first hit enters it,
    // with an outward normal that faces down.
    let out = tool("design.inspect", json!({ "path": "/p/drone/drone.assembly", "sections": ["surface"],
        "ray": { "from": [0.0, 20.0, -200.0], "direction": [0.0, 0.0, 1.0], "occurrences": ["body_bottom"] } }), &files);
    let hits = out["data"]["sections"]["surface"]["hits"].as_array().cloned().unwrap_or_else(|| panic!("hits, not {out:#}"));
    assert!(!hits.is_empty(), "{out:#}");
    assert_eq!(hits[0]["occurrence"], "body_bottom", "{out:#}");
    assert_eq!(hits[0]["entering"], true, "{out:#}");
    assert!(hits[0]["normal"][2].as_f64().unwrap() < -0.5, "{out:#}");
    assert!(hits.iter().all(|h| h["occurrence"] == "body_bottom"), "{out:#}");
}
