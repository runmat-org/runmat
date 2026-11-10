//! The design tools in the CAD host (`designTool` through `runmat_cad_call`, the browser worker's
//! entry): the agent's tools on parts and assemblies regenerate on OCCT here, as they do on the
//! server. A new tube part is staged as created, an assembly fastens two tubes to a plate with
//! mates, and inspect finds the plate and a tube sunk into it overlapping, with each part's mass
//! from its material. design.seat_part seats a block under a plate (its gap and, pressed in, its
//! overlap volume) and, on the drone, a clamp under the belly and a tube in its bore.

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
        "rays": [{ "from": [0.0, 20.0, -200.0], "direction": [0.0, 0.0, 1.0], "occurrences": ["body_bottom"] }] }), &files);
    let hits = out["data"]["sections"]["surface"]["rays"][0]["hits"].as_array().cloned().unwrap_or_else(|| panic!("hits, not {out:#}"));
    assert!(!hits.is_empty(), "{out:#}");
    assert_eq!(hits[0]["occurrence"], "body_bottom", "{out:#}");
    assert_eq!(hits[0]["entering"], true, "{out:#}");
    assert!(hits[0]["normal"][2].as_f64().unwrap() < -0.5, "{out:#}");
    assert!(hits.iter().all(|h| h["occurrence"] == "body_bottom"), "{out:#}");
}

/// A skid clamp: a 12 mm block, 14 wide and 20 tall, with a 8.2 mm bore along it 5 mm above
/// its base.
const CLAMP: &str = r#"runmat 0.7
std 1

part skid_clamp {
    sketch s on origin.yz {
        l1 = line(start = [-7 mm, 0 mm], end = [7 mm, 0 mm])
        l2 = line(start = l1.end, end = [7 mm, 20 mm])
        l3 = line(start = l2.end, end = [-7 mm, 20 mm])
        l4 = line(start = l3.end, end = l1.start)
        bore = circle(center = [0 mm, 5 mm], radius = 4.1 mm)
    }
    feature body = extrude(profile = sketch.s, distance = 12 mm)
    output clamp = { body = feature.body.body, material = "material.aluminum_6061" }
}
"#;

/// A carbon skid tube, 8 mm outside, 100 mm long.
const SKID_TUBE: &str = r#"runmat 0.7
std 1

part skid_tube {
    sketch s on origin.yz {
        outer = circle(center = [0 mm, 0 mm], radius = 4 mm)
        inner = circle(center = [0 mm, 0 mm], radius = 3.4 mm)
    }
    feature body = extrude(profile = sketch.s, distance = 100 mm)
    output tube = { body = feature.body.body, material = "material.aluminum_6061" }
}
"#;

/// The face of `part` whose area is `area` mm² (within 3 %), by the reference design.inspect
/// gives it.
fn face_by_area(path: &str, text: &str, area: f64) -> String {
    let out = tool("design.inspect", json!({ "path": path, "sections": ["faces"] }), &[(path, text)]);
    let faces = out["data"]["sections"]["faces"].as_array().cloned().unwrap_or_else(|| panic!("faces, not {out:#}"));
    let face = faces.iter().find(|f| (f["area_mm2"].as_f64().unwrap() - area).abs() < 0.03 * area).unwrap_or_else(|| panic!("a {area} mm² face in {faces:#?}"));
    face["reference"].as_str().unwrap().to_owned()
}

#[test]
fn on_the_drone_a_clamp_seats_under_the_belly_and_a_tube_in_its_bore_in_two_calls_with_no_interference() {
    // The drone without its landing skids, plus the clamp and tube parts.
    let mut all = drone();
    for (path, text) in all.iter_mut() {
        if path == "/p/drone/drone.assembly" {
            let start = text.find("    // Landing gear:").expect("the skid block");
            let right = start + text[start..].find("as skid_right {").expect("skid_right");
            let end = right + text[right..].find("\n    }\n").expect("its block") + "\n    }\n".len();
            text.replace_range(start..end, "");
            assert!(!text.contains("landing_skid"), "{text}");
        }
    }
    all.push(("/p/drone/mechanical/ground/skid_clamp.part".into(), CLAMP.into()));
    all.push(("/p/drone/mechanical/ground/skid_tube.part".into(), SKID_TUBE.into()));
    let assembly = "/p/drone/drone.assembly";
    let with = |all: &Vec<(String, String)>| -> Vec<(String, String)> { all.clone() };
    let call = |all: &Vec<(String, String)>, input: Value| -> Value {
        let files: Vec<(&str, &str)> = all.iter().map(|(p, t)| (p.as_str(), t.as_str())).collect();
        let out = tool("design.seat_part", input, &files);
        out.get("data").cloned().unwrap_or_else(|| panic!("a seating, not {out:#}"))
    };
    let keep = |all: &mut Vec<(String, String)>, out: &Value| {
        let text = out["files"].as_array().unwrap().iter().find(|f| f["path"] == assembly).unwrap()["text"].as_str().unwrap().to_owned();
        all.iter_mut().find(|(p, _)| p == assembly).unwrap().1 = text;
    };

    // The clamp comes up under the belly at the rear strut's station, beside the ESC heatsink
    // (it ends at y = 28 mm; the clamp is 14 mm wide), its bore running fore-aft along `along`,
    // and stops 0.5 mm short.
    let mut all = with(&all);
    let out = call(&all, json!({ "path": assembly, "component": "clamp_left", "part": "/p/drone/mechanical/ground/skid_clamp.part",
        "target": { "component": "body_bottom", "from": [28.0, 36.0, -100.0], "direction": [0.0, 0.0, 1.0] },
        "own": { "side": "+z" }, "offset": 0.5, "along": [1.0, 0.0, 0.0], "clearance": ["body_bottom"] }));
    let seat = &out["seat"];
    assert_eq!(seat["mate_status"], "satisfied", "{seat:#}");
    assert_eq!(seat["interference"], json!([]), "{seat:#}");
    let gap = seat["clearance"][0]["distance_mm"].as_f64().unwrap_or_else(|| panic!("{seat:#}"));
    assert!((gap - 0.5).abs() < 0.2, "{seat:#}");
    keep(&mut all, &out);

    // The tube in the clamp's bore: one axis, centred on the clamp.
    let bore = face_by_area("/p/drone/mechanical/ground/skid_clamp.part", CLAMP, 2.0 * std::f64::consts::PI * 4.1 * 12.0);
    let outside = face_by_area("/p/drone/mechanical/ground/skid_tube.part", SKID_TUBE, 2.0 * std::f64::consts::PI * 4.0 * 100.0);
    let out = call(&all, json!({ "path": assembly, "component": "tube_left", "part": "/p/drone/mechanical/ground/skid_tube.part",
        "target": { "component": "clamp_left", "face": bore }, "own": { "face": outside } }));
    let seat = &out["seat"];
    assert_eq!(seat["mate_status"], "satisfied", "{seat:#}");
    assert_eq!(seat["interference"], json!([]), "{seat:#}");
    let y = seat["pose"]["translation_mm"][1].as_f64().unwrap();
    assert!((y - 36.0).abs() < 0.5, "{seat:#}");
    keep(&mut all, &out);
    let files: Vec<(&str, &str)> = all.iter().map(|(p, t)| (p.as_str(), t.as_str())).collect();
    let check = tool("design.check", json!({ "path": assembly }), &files);
    assert_eq!(check["data"]["errors"], 0, "{check:#}");
}

/// A 120 mm carbon skid tube, 8 mm outside.
const LONG_TUBE: &str = r#"runmat 0.7
std 1

part skid_tube_long {
    sketch s on origin.yz {
        outer = circle(center = [0 mm, 0 mm], radius = 4 mm)
        inner = circle(center = [0 mm, 0 mm], radius = 3.4 mm)
    }
    feature body = extrude(profile = sketch.s, distance = 120 mm)
    output tube = { body = feature.body.body, material = "material.aluminum_6061" }
}
"#;

/// The drone without its landing skids, plus the clamp and the long tube.
fn skidless_drone() -> Vec<(String, String)> {
    let mut all = drone();
    for (path, text) in all.iter_mut() {
        if path == "/p/drone/drone.assembly" {
            let start = text.find("    // Landing gear:").expect("the skid block");
            let right = start + text[start..].find("as skid_right {").expect("skid_right");
            let end = right + text[right..].find("\n    }\n").expect("its block") + "\n    }\n".len();
            text.replace_range(start..end, "");
        }
    }
    all.push(("/p/drone/mechanical/ground/skid_clamp.part".into(), CLAMP.into()));
    all.push(("/p/drone/mechanical/ground/skid_tube_long.part".into(), LONG_TUBE.into()));
    all
}

/// The landing skid the agent builds: on each side a clamp under the belly fore and aft, the
/// aft one levelled with the fore one, and a straight tube through both. The belly curves, so
/// clamps seated to first contact sit at different heights and a straight tube clips one; on
/// one axis the six parts overlap nothing. Each seating after the first, with the assembly's
/// bodies regenerated once, takes seconds.
#[test]
fn on_the_drone_two_tubes_in_four_levelled_clamps_overlap_nothing_and_each_seating_takes_seconds() {
    let mut all = skidless_drone();
    let assembly = "/p/drone/drone.assembly";
    let mut seconds: Vec<(String, f64)> = Vec::new();
    let mut call = |all: &Vec<(String, String)>, input: Value| -> Value {
        let files: Vec<(&str, &str)> = all.iter().map(|(p, t)| (p.as_str(), t.as_str())).collect();
        let started = std::time::Instant::now();
        let out = tool("design.seat_part", input.clone(), &files);
        seconds.push((input["component"].as_str().unwrap_or_default().to_owned(), started.elapsed().as_secs_f64()));
        out.get("data").cloned().unwrap_or_else(|| panic!("a seating, not {out:#}"))
    };
    let keep = |all: &mut Vec<(String, String)>, out: &Value| {
        let text = out["files"].as_array().unwrap().iter().find(|f| f["path"] == assembly).unwrap()["text"].as_str().unwrap().to_owned();
        all.iter_mut().find(|(p, _)| p == assembly).unwrap().1 = text;
    };
    let clamp = "/p/drone/mechanical/ground/skid_clamp.part";
    let bore = face_by_area(clamp, CLAMP, 2.0 * std::f64::consts::PI * 4.1 * 12.0);
    let outside = face_by_area("/p/drone/mechanical/ground/skid_tube_long.part", LONG_TUBE, 2.0 * std::f64::consts::PI * 4.0 * 120.0);
    let mut heights = Vec::new();
    for (side, y) in [("left", 36.0), ("right", -36.0)] {
        let (fore, aft, tube) = (format!("clamp_fore_{side}"), format!("clamp_aft_{side}"), format!("tube_{side}"));
        let under = |x: f64| json!({ "component": "body_bottom", "from": [x, y, -100.0], "direction": [0.0, 0.0, 1.0] });
        let z = |out: &Value| out["seat"]["pose"]["translation_mm"][2].as_f64().unwrap_or_else(|| panic!("{:#}", out["seat"]));
        let out = call(&all, json!({ "path": assembly, "component": fore, "part": clamp, "target": under(28.0),
            "own": { "side": "+z" }, "offset": 0.5, "along": [1.0, 0.0, 0.0] }));
        assert_eq!(out["seat"]["interference"], json!([]), "{:#}", out["seat"]);
        let mut fore_z = z(&out);
        keep(&mut all, &out);
        let out = call(&all, json!({ "path": assembly, "component": aft, "part": clamp, "target": under(-28.0),
            "own": { "side": "+z" }, "offset": 0.5, "levelWith": fore }));
        let aft_z = z(&out);
        keep(&mut all, &out);
        let notes = out["seat"]["notes"].to_string();
        if notes.contains("would press") {
            // The aft belly is lower: the fore clamp comes down to the aft one's axis instead.
            let out = call(&all, json!({ "path": assembly, "component": fore, "target": under(28.0),
                "own": { "side": "+z" }, "offset": 0.5, "levelWith": aft }));
            assert!(out["seat"]["notes"].to_string().contains("axis: moved"), "{:#}", out["seat"]);
            fore_z = z(&out);
            keep(&mut all, &out);
        } else {
            assert!(notes.contains(&format!("on `{fore}`'s axis: moved")), "{:#}", out["seat"]);
        }
        assert!((fore_z - aft_z).abs() < 0.01, "{side}: fore {fore_z} mm, aft {aft_z} mm");
        heights.push(fore_z);
        // The tube in the fore clamp's bore, then moved along its axis to centre between the
        // clamps (the axis's sign is the contact normal's).
        let out = call(&all, json!({ "path": assembly, "component": tube, "part": "/p/drone/mechanical/ground/skid_tube_long.part",
            "target": { "component": fore, "face": bore }, "own": { "face": outside } }));
        let sign = out["seat"]["contact"]["normal"][0].as_f64().unwrap_or_else(|| panic!("{:#}", out["seat"])).signum();
        keep(&mut all, &out);
        let out = call(&all, json!({ "path": assembly, "component": tube, "target": { "component": fore, "face": bore },
            "own": { "face": outside }, "offset": -28.0 * sign }));
        assert_eq!(out["seat"]["mate_status"], "satisfied", "{:#}", out["seat"]);
        keep(&mut all, &out);
    }
    let files: Vec<(&str, &str)> = all.iter().map(|(p, t)| (p.as_str(), t.as_str())).collect();
    let out = tool("design.inspect", json!({ "path": assembly, "sections": ["interference"],
        "changed": ["clamp_fore_left", "clamp_aft_left", "tube_left", "clamp_fore_right", "clamp_aft_right", "tube_right"] }), &files);
    let interference = &out["data"]["sections"]["interference"];
    assert_eq!(interference["result"]["overlaps"], json!([]), "{out:#}");
    assert_eq!(interference["problems"].as_array().map(Vec::len), Some(0), "{out:#}");
    let check = tool("design.check", json!({ "path": assembly }), &files);
    assert_eq!(check["data"]["errors"], 0, "{check:#}");
    eprintln!("clamp heights {heights:?} mm; seconds per seating {seconds:?}");
    // The first seating regenerates the assembly's bodies; each later one reuses them and takes
    // about 3 s here (the median, so a busy machine's odd slow call doesn't fail it; a seating
    // that regenerates its neighbours again takes about 50 s).
    let mut later: Vec<f64> = seconds[1..].iter().map(|(_, s)| *s).collect();
    later.sort_by(f64::total_cmp);
    let median = later[later.len() / 2];
    assert!(median < 10.0, "the median seating after the first took {median:.1} s: {seconds:?}");
}

#[test]
fn a_block_seated_under_a_plate_stops_its_offset_short_and_pressed_in_reports_the_overlap_volume() {
    const BLOCK: &str = "runmat 0.7\nstd 1\n\npart block {\n    sketch s on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n        l2 = line(start = l1.end, end = [20 mm, 10 mm])\n        l3 = line(start = l2.end, end = [0 mm, 10 mm])\n        l4 = line(start = l3.end, end = l1.start)\n    }\n    feature box = extrude(profile = sketch.s, distance = 5 mm)\n}\n";
    // The plate lifted to z = 100 mm, fixed.
    const RIG: &str = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part plate as plate {\n        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 100 mm] } }\n    }\n}\n";
    let files = [("/p/plate.part", PLATE), ("/p/mech/block.part", BLOCK), ("/p/rig.assembly", RIG)];
    let seat = |offset: f64| -> Value {
        let out = tool("design.seat_part", json!({ "path": "/p/rig.assembly", "component": "foot", "part": "/p/mech/block.part",
            "target": { "component": "plate", "from": [40.0, 25.0, 0.0], "direction": [0.0, 0.0, 1.0] },
            "own": { "side": "+z" }, "offset": offset, "clearance": ["plate"] }), &files);
        out.get("data").map(|d| d["seat"].clone()).unwrap_or_else(|| panic!("a seating, not {out:#}"))
    };
    // 0.5 mm short of the plate's underside: no overlap, a 0.5 mm gap.
    let short = seat(0.5);
    assert_eq!(short["interference"], json!([]), "{short:#}");
    let gap = short["clearance"][0]["distance_mm"].as_f64().unwrap_or_else(|| panic!("{short:#}"));
    assert!((gap - 0.5).abs() < 0.05, "{short:#}");
    assert!(short["summary"].as_str().unwrap().contains("no interference; 0.5 mm from plate"), "{short:#}");
    // Pressed 2 mm in: 20 × 10 × 2 mm of block inside the plate.
    let pressed = seat(-2.0);
    let overlap = &pressed["interference"][0];
    assert_eq!(overlap["with"], "plate", "{pressed:#}");
    assert!((overlap["volume_mm3"].as_f64().unwrap() - 400.0).abs() < 1.0, "{pressed:#}");
    assert!(pressed["summary"].as_str().unwrap().contains("overlaps plate by"), "{pressed:#}");
}

/// A landing-gear leg, sketched square to the tube and extruded 8 mm along it: a 16 × 4 mm pad
/// on top, a web down and 25 mm out toward -y, and a foot 38 mm down holding the tube's 8.4 mm
/// bore (the `bore` region). Its box's top is centred 12.5 mm out from the pad.
const LEG: &str = r#"runmat 0.7
std 1

part skid_leg {
    sketch s on origin.yz {
        l1 = line(start = [8 mm, 0 mm], end = [-8 mm, 0 mm])
        l2 = line(start = l1.end, end = [-33 mm, -30 mm])
        l3 = line(start = l2.end, end = [-33 mm, -46 mm])
        l4 = line(start = l3.end, end = [-17 mm, -46 mm])
        l5 = line(start = l4.end, end = [-17 mm, -30 mm])
        l6 = line(start = l5.end, end = [8 mm, -4 mm])
        l7 = line(start = l6.end, end = l1.start)
        bore = circle(center = [-25 mm, -38 mm], radius = 4.2 mm)
    }
    feature body = extrude(profile = sketch.s, distance = 8 mm)
    output leg = { body = feature.body.body, material = "material.aluminum_6061" }
    region bore : face = { select = [feature.body.face.side(s.bore)], expect = at_least_one }
}
"#;

/// A 160 mm carbon skid tube, 8 mm outside.
const SKID_160: &str = r#"runmat 0.7
std 1

part skid_tube_160 {
    sketch s on origin.yz {
        outer = circle(center = [0 mm, 0 mm], radius = 4 mm)
        inner = circle(center = [0 mm, 0 mm], radius = 3.4 mm)
    }
    feature body = extrude(profile = sketch.s, distance = 160 mm)
    output tube_160 = { body = feature.body.body, material = "material.aluminum_6061" }
}
"#;

/// The landing gear from symmetric pieces: a left leg at two stations equally fore and aft of
/// the centre of mass (the aft one levelled with the fore one), each right leg the mirror image
/// of its left leg, and each tube through its side's two bores in one call. The six parts
/// overlap nothing, every bore is on its tube's axis, the sides mirror each other across the
/// body's middle plane, the tubes are centred on the centre of mass, and the last seating
/// carries the stance check.
#[test]
fn on_the_drone_mirrored_legs_and_tubes_through_both_bores_are_symmetric_and_centred_on_the_centre_of_mass() {
    let mut all = skidless_drone();
    all.push(("/p/drone/mechanical/ground/skid_leg.part".into(), LEG.into()));
    all.push(("/p/drone/mechanical/ground/skid_tube_160.part".into(), SKID_160.into()));
    let assembly = "/p/drone/drone.assembly";
    let run = |all: &Vec<(String, String)>, name: &str, input: Value| -> Value {
        let files: Vec<(&str, &str)> = all.iter().map(|(p, t)| (p.as_str(), t.as_str())).collect();
        let out = tool(name, input, &files);
        out.get("data").cloned().unwrap_or_else(|| panic!("{name}: {out:#}"))
    };
    let keep = |all: &mut Vec<(String, String)>, out: &Value| {
        let text = out["files"].as_array().unwrap().iter().find(|f| f["path"] == assembly).unwrap()["text"].as_str().unwrap().to_owned();
        all.iter_mut().find(|(p, _)| p == assembly).unwrap().1 = text;
    };
    let before = run(&all, "design.inspect", json!({ "path": assembly, "sections": ["stance"], "contact": ["body_bottom"] }));
    let com_x = before["sections"]["stance"]["center_of_mass"]["point_mm"][0].as_f64().unwrap_or_else(|| panic!("{before:#}"));
    // The feet hold the tubes and reach 4 mm below them, so the feet carry the craft over a
    // 76 mm support along x. The legs drop 38 mm, which keeps the camera over 20 mm clear and
    // the centre of mass low enough to tip past 30° over the support's ends.
    let (fore, aft) = (com_x + 34.0, com_x - 34.0);
    let leg = "/p/drone/mechanical/ground/skid_leg.part";
    // The pads (16 mm wide) at y = 38 mm clear the heatsink's belly plate, whose edge is at
    // y = 28 mm, by 2 mm.
    let under = |x: f64| json!({ "component": "body_bottom", "from": [x, 38.0, -100.0], "direction": [0.0, 0.0, 1.0] });
    let seat = |all: &mut Vec<(String, String)>, input: Value| -> Value {
        let out = run(all, "design.seat_part", input);
        assert_eq!(out["seat"]["interference"], json!([]), "{:#}", out["seat"]);
        assert_eq!(out["seat"]["mate_status"], "satisfied", "{:#}", out["seat"]);
        keep(all, &out);
        out["seat"].clone()
    };
    // Turned half round from along +x so the web runs outboard (+y): the bore 25 mm outside the
    // pad's middle, which meets the belly.
    seat(&mut all, json!({ "path": assembly, "component": "leg_fore_left", "part": leg, "target": under(fore),
        "own": { "side": "+z" }, "offset": 0.5, "along": [1.0, 0.0, 0.0], "rotation": 180.0 }));
    let aft_seat = seat(&mut all, json!({ "path": assembly, "component": "leg_aft_left", "part": leg, "target": under(aft),
        "own": { "side": "+z" }, "offset": 0.5, "along": [1.0, 0.0, 0.0], "rotation": 180.0, "levelWith": "leg_fore_left" }));
    if aft_seat["notes"].to_string().contains("would press") {
        seat(&mut all, json!({ "path": assembly, "component": "leg_fore_left", "target": under(fore),
            "own": { "side": "+z" }, "offset": 0.5, "along": [1.0, 0.0, 0.0], "rotation": 180.0, "levelWith": "leg_aft_left" }));
    }
    for station in ["fore", "aft"] {
        let s = seat(&mut all, json!({ "path": assembly, "component": format!("leg_{station}_right"), "part": leg,
            "target": { "component": "body_bottom" }, "mirror": format!("leg_{station}_left") }));
        assert!(s["notes"].to_string().contains("the mirror image of"), "{s:#}");
    }
    let tube = "/p/drone/mechanical/ground/skid_tube_160.part";
    let left = seat(&mut all, json!({ "path": assembly, "component": "tube_left", "part": tube, "through": ["leg_fore_left.bore", "leg_aft_left"] }));
    let right = seat(&mut all, json!({ "path": assembly, "component": "tube_right", "part": tube, "through": ["leg_fore_right", "leg_aft_right"],
        "contact": ["tube_*", "leg_*"] }));
    for s in [&left, &right] {
        let rows = s["through"].as_array().unwrap_or_else(|| panic!("{s:#}"));
        assert_eq!(rows.len(), 2, "{s:#}");
        for row in rows {
            assert!(row["off_axis_mm"].as_f64().unwrap() < 0.01, "{s:#}");
            assert!((row["station_mm"].as_f64().unwrap().abs() - 34.0).abs() < 0.01, "{s:#}");
        }
    }
    // Symmetric about the body's middle plane (y = 0) and centred on the centre of mass.
    let (l, r) = (&left["contact"]["point_mm"], &right["contact"]["point_mm"]);
    let at = |v: &Value, k: usize| v[k].as_f64().unwrap();
    assert!((at(l, 1) - 63.0).abs() < 0.5, "the left tube 63 mm out, 25 mm outboard of the pads: {l}");
    assert!((at(l, 0) - at(r, 0)).abs() < 0.01 && (at(l, 1) + at(r, 1)).abs() < 0.05 && (at(l, 2) - at(r, 2)).abs() < 0.01, "left {l}, right {r}");
    let stance = &right["stance"];
    let com_after = at(&stance["center_of_mass"]["point_mm"], 0);
    assert!((at(l, 0) - com_x).abs() < 2.0 && (at(l, 0) - com_after).abs() < 2.0, "tube middle x {} mm, centre of mass {com_x} mm before, {com_after} mm after", at(l, 0));
    // Standing on its feet and tubes, the camera clear, tipping past 30° on every edge.
    assert_eq!(stance["ok"], true, "{stance:#}");
    assert!(stance["support"]["min_tip_angle_deg"].as_f64().unwrap() >= 30.0, "{stance:#}");
    assert!(right["summary"].as_str().unwrap().contains("Stance"), "{right:#}");
    let files: Vec<(&str, &str)> = all.iter().map(|(p, t)| (p.as_str(), t.as_str())).collect();
    let changed = ["leg_fore_left", "leg_aft_left", "leg_fore_right", "leg_aft_right", "tube_left", "tube_right"];
    let out = tool("design.inspect", json!({ "path": assembly, "sections": ["interference"], "changed": changed }), &files);
    assert_eq!(out["data"]["sections"]["interference"]["result"]["overlaps"], json!([]), "{out:#}");
    let check = tool("design.check", json!({ "path": assembly }), &files);
    assert_eq!(check["data"]["errors"], 0, "{check:#}");
    eprintln!("left {l}, right {r}; aft leg notes {}; stance {}", aft_seat["notes"], stance["summary"]);
}
