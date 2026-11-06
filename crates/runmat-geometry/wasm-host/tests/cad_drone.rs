//! The camera drone showcase (`runmat-cad/wasm/tests/drone/`, in module folders) on OCCT, through `runmat_cad_call`
//! (the Desktop and browser worker's entry): every part, assembly and drawing compiles with no
//! error, every part has a body and every feature regenerates, and the top assembly places every
//! occurrence. `RUNMAT_DRONE_SRC=<folder>` compiles another copy of the project instead.

use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn entry(path: &str, text: &str, files: &str) -> Value {
    let request = serde_json::json!({ "fn": "compile", "args": [path, text, files, "{}"] });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

fn project() -> BTreeMap<String, String> {
    let dir = std::env::var("RUNMAT_DRONE_SRC").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../runmat-cad/wasm/tests/drone")
    });
    let mut all = BTreeMap::new();
    read_tree(&dir, "/p/drone", &mut all);
    assert!(all.contains_key("/p/drone/drone.assembly"), "the drone project is checked in at {dir:?}");
    all
}

/// Every file under `dir`, keyed `<at>/<relative path>`: the project's folders are its modules.
fn read_tree(dir: &std::path::Path, at: &str, all: &mut BTreeMap<String, String>) {
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if e.path().is_dir() {
            read_tree(&e.path(), &format!("{at}/{name}"), all);
        } else if let Ok(text) = std::fs::read_to_string(e.path()) {
            all.insert(format!("{at}/{name}"), text);
        }
    }
}

fn errors(v: &Value) -> Vec<String> {
    v["diagnostics"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter(|d| d["severity"] == "error")
        .map(|d| d["message"].as_str().unwrap_or("").to_string())
        .collect()
}

#[test]
fn the_drone_compiles_part_by_part_and_as_one_assembly() {
    let all = project();
    let files_json = serde_json::to_string(&all).unwrap();
    let only = std::env::var("RUNMAT_DRONE_ONLY").ok();
    let mut failures = Vec::new();
    for path in all.keys() {
        let cad = [".part", ".assembly", ".drawing"].iter().any(|x| path.ends_with(x));
        if !cad || only.as_ref().is_some_and(|o| !path.contains(o.as_str())) {
            continue;
        }
        let started = std::time::Instant::now();
        let v = entry(path, &all[path], &files_json);
        let errs = errors(&v);
        let warnings: Vec<String> = v["diagnostics"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .iter()
            .filter(|d| d["severity"] != "error")
            .map(|d| d["message"].as_str().unwrap_or("").to_string())
            .collect();
        eprintln!("{path}: {:.2?}, {} errors, warnings {warnings:?}", started.elapsed(), errs.len());
        if !errs.is_empty() {
            failures.push(format!("{path}: {errs:?}"));
            continue;
        }
        if path.ends_with(".part") {
            let data = &v["display"]["data"];
            let problems = data["problems"].as_array().map(|p| p.len()).unwrap_or(0);
            let bodies = data["bodies"].as_array().map(|b| b.len()).unwrap_or(0);
            let failed: Vec<String> = v["projection"]["data"]["index"]["features"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .iter()
                .filter(|f| f["status"].as_str().is_some_and(|s| s != "ok" && s != "regenerated"))
                .map(|f| format!("{} {}", f["id"], f["status"]))
                .collect();
            let skeleton = path.ends_with("skeleton.part");
            // Each drone part is one solid: a second body is a feature left out of the union
            // (STEP then carries overlapping solids, which the FEA rejects as interpenetrating).
            if problems > 0 || (bodies != 1 && !skeleton) || !failed.is_empty() {
                failures.push(format!("{path}: {bodies} bodies, problems {:?}, features {failed:?}", data["problems"]));
            }
        }
        if path.ends_with(".assembly") {
            let failed: Vec<String> = v["projection"]["data"]["occurrences"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .iter()
                .filter(|o| o["state"] == "failed")
                .map(|o| o["path"].as_str().unwrap_or("").to_string())
                .collect();
            let d = &v["display"]["data"];
            eprintln!(
                "  {} occurrences, bounds {}",
                d["occurrences"].as_array().map(|o| o.len()).unwrap_or(0),
                d["bounds"]
            );
            if !failed.is_empty() {
                failures.push(format!("{path}: unplaced {failed:?}"));
            }
        }
        if let Ok(dump) = std::env::var("RUNMAT_DRONE_DUMP") {
            let name = path.rsplit('/').next().unwrap();
            std::fs::create_dir_all(&dump).ok();
            std::fs::write(format!("{dump}/{name}.json"), serde_json::to_string_pretty(&v["projection"]).unwrap()).ok();
            std::fs::write(format!("{dump}/{name}.display.json"), serde_json::to_string(&v["display"]).unwrap()).ok();
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The arm's production drawing reads the part: dimensions between planes and between hole
/// axes, tolerances from the PMI after the values, the hole callout from the hole feature, the
/// feature control frames and finish marks, and the revision from the revision table.
#[test]
fn the_arm_drawing_dimensions_read_the_part() {
    let all = project();
    let files_json = serde_json::to_string(&all).unwrap();
    let path = "/p/drone/drawings/arm.drawing";
    let v = entry(path, &all[path], &files_json);
    assert!(errors(&v).is_empty(), "{:?}", errors(&v));
    let text = &v["projection"]["data"]["text"];
    for (name, want) in [
        // The arm at its defaults (the front arm, 88.2 mm long): the knuckle and pad heights,
        // the hinge bore to the motor axis and the motor's bolt circle with their location
        // tolerances, the bore's diameter with its size tolerance, the motor holes' callout
        // from the hole feature, the frames from the PMI and the finish marks.
        ("knuckle_height", "13"),
        ("pad_thickness", "8"),
        ("pin_to_motor", "88.2 ±0.1"),
        // Overall, over the R7 knuckle and the R9 pad: 7 + 88.2 + 9, a reference dimension.
        ("overall_length", "(104.2)"),
        ("bolt_circle", "12 ±0.05"),
        ("pin_bore", "⌀4.1 +0.018/0"),
        ("motor_holes", "4× ⌀2.2 THRU"),
        ("bore_square", "⊥|⌀0.02|A"),
        ("motor_position", "⌖|⌀0.1 Ⓜ|A|B|C"),
        ("seat_parallel", "∥|0.03|C"),
        ("seat_finish", "Ra 0.8"),
        ("bore_finish", "Ra 0.8 reamed"),
    ] {
        assert_eq!(text[name], want, "{name}: {text}");
    }
    // The title block's revision is the revision table's newest row.
    let block = &v["projection"]["data"]["title_blocks"]["1"];
    assert!(block.as_array().unwrap().iter().any(|r| r[0] == "Rev" && r[1] == "C"), "{block}");
    // GD&T frames, datum symbols and finish marks carry the model point they point at.
    for name in ["bore_square", "motor_position", "datum_a", "datum_b", "datum_c", "seat_finish"] {
        assert!(v["projection"]["data"]["anchors"][name].is_array(), "{name} has no leader anchor");
    }
    // The overall length's extension lines leave the knuckle's and the pad's far sides (x = -7
    // and 97.2), not their axes.
    let ends = &v["projection"]["data"]["anchors"]["overall_length"];
    let xs: Vec<f64> = ends.as_array().unwrap().iter().map(|p| p[0].as_f64().unwrap()).collect();
    assert!((xs[0] + 7.0).abs() < 1e-3 && (xs[1] - 97.2).abs() < 1e-3, "overall_length anchors {ends}");
    // The bore's frame leader ends on the R2.05 bore, not the R7 knuckle round it.
    let r = v["projection"]["data"]["radii"]["bore_square"].as_f64().unwrap_or(0.0);
    assert!((r - 2.05).abs() < 1e-6, "bore_square radius {r}");
}

/// The hinge mount's lug profiles are fully defined before they extrude (the Desktop's sketch
/// readout shows 0 degrees of freedom): two horizontal lines tangent to the R7.5 end arc, a
/// vertical back edge 14 mm from the pin axis, and the R2 pin bore concentric with the arc.
#[test]
fn the_hinge_lug_sketches_are_fully_defined() {
    let all = project();
    let files_json = serde_json::to_string(&all).unwrap();
    let path = "/p/drone/mechanical/arm/hinge_mount.part";
    let v = entry(path, &all[path], &files_json);
    assert!(errors(&v).is_empty(), "{:?}", errors(&v));
    let sketches = v["display"]["data"]["sketches"].as_array().cloned().unwrap_or_default();
    for id in ["upper_s", "lower_s"] {
        let s = sketches.iter().find(|s| s["id"] == id).unwrap_or_else(|| panic!("{id} in {sketches:?}"));
        assert_eq!(s["status"], "well_defined", "{id}: dof {}, redundant {}, conflicting {}", s["dof"], s["redundant"], s["conflicting"]);
        assert_eq!(s["dof"], 0, "{id}: {}", s["dof"]);
        assert!(s["redundant"].as_array().is_none_or(|r| r.is_empty()), "{id}: redundant {}", s["redundant"]);
    }
}

/// The motor pod's mount pattern is fully defined: the 5 mm bore fixed at the motor axis, the
/// four 2.2 mm screw holes on the 12 mm bolt circle along the axes, and the phase-lead slot at
/// 225 degrees between two of them, dimensioned by diameters and its R1.2 ends.
#[test]
fn the_motor_mount_sketch_is_fully_defined() {
    let all = project();
    let files_json = serde_json::to_string(&all).unwrap();
    let path = "/p/drone/mechanical/propulsion/motor_pod.part";
    let v = entry(path, &all[path], &files_json);
    assert!(errors(&v).is_empty(), "{:?}", errors(&v));
    let sketches = v["display"]["data"]["sketches"].as_array().cloned().unwrap_or_default();
    let s = sketches.iter().find(|s| s["id"] == "mount_s").unwrap_or_else(|| panic!("mount_s in {sketches:?}"));
    assert_eq!(s["status"], "well_defined", "dof {}, redundant {}, conflicting {}", s["dof"], s["redundant"], s["conflicting"]);
    assert_eq!(s["dof"], 0);
}

/// The study geometry as exact AP242: the arm with its motor pod, and the ESC heatsink with the
/// gap pad and the ESC board on it. Each product is one solid, and each face in a region travels
/// under the region's name, which is what the studies' `select name` reads (`root`,
/// `motor_seat`, `fins`). `RUNMAT_DRONE_STEP_DIR=<folder>` writes the files there (the example's
/// geometry/).
#[test]
fn the_study_assemblies_export_with_their_region_names() {
    runmat_geometry_wasm_host::cad::install_occt();
    let all = project();
    for (file, products, faces) in [
        ("arm_study", &["'arm'", "'motor_pod'"][..], &["ADVANCED_FACE('root'", "ADVANCED_FACE('motor_seat'"][..]),
        ("esc_thermal", &["'esc_heatsink'", "'thermal_pad'", "'esc_board_thermal'"][..], &["ADVANCED_FACE('fins'", "ADVANCED_FACE('pad_seat'"][..]),
    ] {
        let path = format!("/p/drone/simulation/{file}.assembly");
        let step = runmat_cad_wasm::exchange::assembly_step(&path, &all[&path], &all)
            .unwrap_or_else(|e| panic!("{file}: {e}"))
            .expect("OCCT writes STEP");
        let text = String::from_utf8(step).expect("STEP is text");
        if let Ok(dir) = std::env::var("RUNMAT_DRONE_STEP_DIR") {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(format!("{dir}/{file}.step"), &text).unwrap();
        }
        for p in products {
            assert!(text.contains(&format!("PRODUCT({p}")), "{file}: no product {p}");
        }
        for f in faces {
            assert!(text.contains(f), "{file}: no {f}");
        }
        let solids = text.matches("MANIFOLD_SOLID_BREP(").count();
        assert_eq!(solids, products.len(), "{file}: {solids} solids for {} products", products.len());
    }
}

/// No two parts share material, in flight or stowed: the hinges sit in their shell seats, the
/// battery's notched case in its bay, the skids on the belly, the sensors and the gimbal's yaw
/// motor in their openings, each pod on its arm's pad; folded, the front arms ride over the rear
/// arms and every blade lies clear along its arm. Screws in their tapped holes overlap the
/// thread's depth by design and are left out of the clash list; each stack screw's overlap with
/// the heatsink is read as its engaged thread length, which must reach 1.5 × D = 4.5 mm.
#[test]
fn the_drone_has_no_interference_in_flight_or_folded() {
    runmat_geometry_wasm_host::cad::install_occt();
    let all = project();
    let mut problems = Vec::new();
    let mut engagements = 0;
    for file in ["drone", "drone_folded"] {
        let path = format!("/p/drone/{file}.assembly");
        let started = std::time::Instant::now();
        let r = runmat_cad_wasm::evaluate::assembly_interference(&path, &all[&path], &all, None)
            .unwrap_or_else(|e| panic!("{file}: {e}"));
        eprintln!("{file}: {} parts, {} candidate pairs, {:.2?}", r.parts, r.candidates, started.elapsed());
        assert!(r.errors.is_empty(), "{file}: {:?}", r.errors);
        for o in &r.overlaps {
            eprintln!("  {} / {}: {:.3} mm^3", o.a, o.b, o.volume * 1e9);
            let screw = o.a.contains("screw") || o.b.contains("screw");
            if !screw {
                problems.push(format!("{file}: {} / {} share {:.3} mm^3", o.a, o.b, o.volume * 1e9));
            } else if o.a.contains("heatsink") || o.b.contains("heatsink") {
                // A stack screw in its tapped standoff overlaps the thread: the Ø3 shank in the
                // Ø2.5 tap drill, 2.16 mm² over the engaged length. M3 into 6061 needs at least
                // 1.5 × D = 4.5 mm of thread.
                let engaged = o.volume * 1e9 / THREAD_ANNULUS_MM2;
                eprintln!("    engaged {engaged:.2} mm");
                if engaged < 4.5 {
                    problems.push(format!("{file}: {} / {} engage {engaged:.2} mm of thread (needs 4.5 mm)", o.a, o.b));
                }
                engagements += 1;
            }
        }
    }
    // Four stack screws, in flight and folded.
    assert_eq!(engagements, 8, "stack screws in the heatsink's standoffs");
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// The Ø3 M3 shank inside the Ø2.5 tap drill: π (1.5² − 1.25²) mm².
const THREAD_ANNULUS_MM2: f64 = std::f64::consts::PI * (1.5 * 1.5 - 1.25 * 1.25);

/// The drone hovers level on equal thrust: its centre of gravity sits within 3 mm of the four
/// props' thrust centre (the motor axes' centroid in plan), so no motor runs a large differential
/// to hold it, and neighbouring props keep a tip gap of 10 % of their radius. Prints each part's
/// mass and CG for the balance notes.
#[test]
fn the_cg_sits_over_the_thrust_centre() {
    runmat_geometry_wasm_host::cad::install_occt();
    let all = project();
    let path = "/p/drone/drone.assembly";
    let r = runmat_cad_wasm::mass::assembly_mass(path, &all[path], &all, None).expect("mass properties");
    let mm = |v: f64| v * 1000.0;
    for p in &r.parts {
        eprintln!(
            "{:<44} {:>7.1} g  ({:>7.1}, {:>6.1}, {:>5.1}) mm",
            p.path,
            p.mass.unwrap_or(0.0) * 1000.0,
            mm(p.center[0]),
            mm(p.center[1]),
            mm(p.center[2])
        );
    }
    let motors: Vec<[f64; 3]> = r.parts.iter().filter(|p| p.path.ends_with("/motor/motor")).map(|p| p.center).collect();
    assert_eq!(motors.len(), 4, "four motors");
    let thrust = [0, 1].map(|k| motors.iter().map(|c| c[k]).sum::<f64>() / 4.0);
    // Neighbouring props clear each other in flight by at least 10 % of their radius: the closest
    // two motor axes less one 174 mm prop diameter (tips at 9 mm pivot + 78 mm blade = 87 mm).
    let radius = 87.0;
    let closest = motors
        .iter()
        .enumerate()
        .flat_map(|(i, a)| motors[i + 1..].iter().map(move |b| mm(((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt())))
        .fold(f64::MAX, f64::min);
    eprintln!("closest motor axes {closest:.1} mm, tip gap {:.1} mm", closest - 2.0 * radius);
    assert!(closest - 2.0 * radius >= 0.1 * radius, "prop tips {:.1} mm apart", closest - 2.0 * radius);
    let cg = r.total.center;
    eprintln!(
        "total {:.1} g, CG ({:.1}, {:.1}, {:.1}) mm, thrust centre ({:.1}, {:.1}) mm",
        r.total.mass * 1000.0,
        mm(cg[0]),
        mm(cg[1]),
        mm(cg[2]),
        mm(thrust[0]),
        mm(thrust[1])
    );
    assert!(r.problems.is_empty(), "{:?}", r.problems);
    let off = (mm(cg[0] - thrust[0]), mm(cg[1] - thrust[1]));
    assert!(off.0.abs() < 3.0 && off.1.abs() < 1.0, "CG off the thrust centre by ({:.1}, {:.1}) mm", off.0, off.1);
}
