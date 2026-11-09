//! A generic airliner, authored as `.part` and `.assembly` text and compiled through the
//! including crate's `super::entry(path, text, files_json) -> String`: the OCCT host's
//! `runmat_cad_call` in `runmat-geometry-wasm-host` (the grid test double cannot build its
//! lofted and partly revolved parts, so the suite runs on OCCT only): fuselage sections of
//! frames, stringers, skin panels and windows, lofted wings, an empennage, landing gear and
//! engine nacelles, with flush rivets patterned along the structure. A skeleton part publishes the frames the major assemblies
//! attach to, and the main sizes are inputs.
//!
//! `RUNMAT_AIRCRAFT_DIR=<folder>` writes the files there (the Desktop's shoot scenes load them).

use serde_json::Value;
use std::collections::BTreeMap;

use super::aircraft_text::files;

fn compile(path: &str, all: &BTreeMap<String, String>) -> Value {
    let files_json = serde_json::to_string(all).unwrap();
    let out = super::entry(path, &all[path], &files_json);
    serde_json::from_str(&out).expect("json")
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
fn the_aircraft_compiles_part_by_part_and_as_one_assembly() {
    let started = std::time::Instant::now();
    let all: BTreeMap<String, String> = files()
        .into_iter()
        .map(|(n, t)| (format!("/p/aircraft/{n}"), t))
        .collect();
    if let Ok(dir) = std::env::var("RUNMAT_AIRCRAFT_DIR") {
        std::fs::create_dir_all(&dir).unwrap();
        for (path, text) in &all {
            let file = std::path::Path::new(&dir).join(path.strip_prefix("/p/aircraft/").unwrap());
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(file, text).unwrap();
        }
    }
    let mut failures = Vec::new();
    for path in all.keys() {
        let v = compile(path, &all);
        let errs = errors(&v);
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
                .filter(|f| {
                    f["status"]
                        .as_str()
                        .is_some_and(|s| s != "ok" && s != "regenerated")
                })
                .map(|f| format!("{} {}", f["id"], f["status"]))
                .collect();
            let skeleton = path.ends_with("skeleton.part");
            if problems > 0 || (bodies == 0 && !skeleton) || !failed.is_empty() {
                failures.push(format!(
                    "{path}: {bodies} bodies, problems {:?}, features {failed:?}",
                    data["problems"]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));

    eprintln!("aircraft parts: {:.2?}", started.elapsed());
    let started = std::time::Instant::now();
    let top = compile("/p/aircraft/airliner.assembly", &all);
    eprintln!("airliner assembly compile: {:.2?}", started.elapsed());
    eprintln!("airliner diagnostics: {}", top["diagnostics"]);
    // What the Outline shows: every occurrence placed (no red ✕), no error on the assembly.
    assert!(errors(&top).is_empty(), "{:?}", errors(&top));
    let failed: Vec<String> = top["projection"]["data"]["occurrences"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter(|o| o["state"] == "failed")
        .map(|o| o["path"].as_str().unwrap_or("").to_string())
        .collect();
    assert!(failed.is_empty(), "unplaced occurrences: {failed:?}");
    // The Outline's markers: `(f)` only on an occurrence fixed in space by its
    // own placement; attached ones and pattern copies are fully defined.
    let rows = top["projection"]["data"]["occurrences"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let state = |path: &str| {
        rows.iter()
            .find(|o| o["path"] == path)
            .map(|o| o["state"].as_str().unwrap_or("").to_string())
            .unwrap_or_else(|| {
                panic!(
                    "no occurrence {path} in {:?}",
                    rows.iter()
                        .take(12)
                        .map(|o| o["path"].clone())
                        .collect::<Vec<_>>()
                )
            })
    };
    assert_eq!(state("skeleton"), "fixed");
    assert_eq!(state("wing"), "fully_defined");
    assert_eq!(state("fuselage/forward/frame@frames[2]"), "fully_defined");
    let fixed: Vec<&str> = rows
        .iter()
        .filter(|o| o["state"] == "fixed")
        .filter_map(|o| o["path"].as_str())
        .collect();
    assert!(
        fixed.len() * 10 < rows.len(),
        "{} of {} occurrences fixed: {:?}",
        fixed.len(),
        rows.len(),
        fixed
            .iter()
            .step_by(fixed.len() / 40 + 1)
            .collect::<Vec<_>>()
    );
    let display = &top["display"]["data"];
    let occurrences = display["occurrences"]
        .as_array()
        .map(|o| o.len())
        .unwrap_or(0);
    eprintln!(
        "airliner: {occurrences} occurrences of {} prototypes, bounds {}",
        display["prototypes"]
            .as_array()
            .map(|p| p.len())
            .unwrap_or(0),
        display["bounds"]
    );
    assert!(occurrences >= 2000, "{occurrences} occurrences");
    // True proportions: 38 m long, a 34.8 m span over the winglets, 12.3 m from the tyres to
    // the fin tip.
    let b = &display["bounds"];
    let ext = |k: usize| b["max"][k].as_f64().unwrap() - b["min"][k].as_f64().unwrap();
    assert!((ext(0) - 38.0).abs() < 0.5, "length {}", ext(0));
    assert!((ext(1) - 34.8).abs() < 0.6, "span {}", ext(1));
    assert!((ext(2) - 12.3).abs() < 0.5, "height {}", ext(2));
    // Configured parts draw their own configuration: every rib station and both gear legs'
    // tyres are their own prototype.
    let protos = display["prototypes"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let of = |file: &str| {
        protos
            .iter()
            .filter(|p| {
                p["file"]
                    .as_str()
                    .is_some_and(|f| f == file || f.ends_with(&format!("/{file}")))
            })
            .count()
    };
    assert!(of("rib.part") >= 20, "{} rib prototypes", of("rib.part"));
    assert_eq!(of("tyre.part"), 2, "tyre prototypes");
    assert!(
        protos.iter().any(|p| p["material"] == "glass"),
        "window panes carry their material"
    );
    // Each part draws in its own color (its finish), apart from its material: dark tinted
    // windshield, a near-white fuselage, a darker wing grey.
    let appearance = |file: &str| -> Vec<f64> {
        let p = protos
            .iter()
            .find(|p| {
                p["file"]
                    .as_str()
                    .is_some_and(|f| f.ends_with(&format!("/{file}")) || f == file)
            })
            .unwrap_or_else(|| panic!("no prototype of {file}"));
        p["appearance"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    };
    // It opens on its saved hero view.
    let views = top["projection"]["data"]["views"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        views
            .iter()
            .any(|v| v["name"] == "hero" && v["default"] == true),
        "views {views:?}"
    );
    let glass = appearance("cockpit_window_front.part");
    assert!(
        glass[0] < 0.15 && glass[3] > 0.99,
        "cockpit glazing {glass:?}"
    );
    let skin = appearance("skin_panel.part");
    let wing = appearance("wing_box_skin_upper.part");
    assert!(
        skin[0] > 0.85 && wing[0] < skin[0] - 0.08,
        "fuselage {skin:?}, wing {wing:?}"
    );
    the_parts_sit_where_they_did(&rows);
}

/// Each part occurrence of the airliner, grouped by its design and hand (`frame`,
/// `wing_slat|port`), with its world transform: the rotation's rows, then the translation (mm).
/// A port part is `<name>|port`, whether it is its own file (`wing_slat_port.part`) or the
/// starboard part in its `side = -1` configuration (`wing_slat.part`, placed by a mirror).
fn part_placements(rows: &[Value]) -> BTreeMap<String, Vec<[f64; 12]>> {
    let mut out: BTreeMap<String, Vec<[f64; 12]>> = BTreeMap::new();
    for row in rows {
        let def = &row["definition"];
        if def["kind"] != "part" {
            continue;
        }
        let file = def["file"].as_str().unwrap_or("");
        let stem = file.rsplit('/').next().unwrap_or(file);
        let mut name = def["part"]
            .as_str()
            .unwrap_or(stem.split('.').next().unwrap_or(stem))
            .to_string();
        let config = row["configuration"].to_string();
        let mut hand = "";
        if let Some(n) = name.strip_suffix("_port") {
            name = n.to_string();
            hand = "|port";
        } else if (config.contains("side") && config.contains("-1")) || config == "\"port\"" {
            hand = "|port";
        }
        let w = &row["world"];
        let r = |i: usize, j: usize| w["rotation"][i][j].as_f64().unwrap();
        let t = |i: usize| w["translation"][i].as_f64().unwrap();
        out.entry(format!("{name}{hand}")).or_default().push([
            r(0, 0),
            r(0, 1),
            r(0, 2),
            r(1, 0),
            r(1, 1),
            r(1, 2),
            r(2, 0),
            r(2, 1),
            r(2, 2),
            t(0),
            t(1),
            t(2),
        ]);
    }
    for list in out.values_mut() {
        // Rounded to a micrometre for the order only; equality is checked at 1e-9 below.
        let key = |m: &[f64; 12]| {
            m.iter()
                .map(|v| (v * 1e3).round() as i64)
                .collect::<Vec<_>>()
        };
        list.sort_by_key(key);
    }
    out
}

/// The airliner resolves to the same parts in the same places as the flat project it was made
/// from: the same occurrences of each design and hand, with world transforms equal within
/// 1e-9 (`occurrences.json`, written by the flat generator with
/// `RUNMAT_AIRCRAFT_OCCURRENCES_WRITE=1`). The folders, the mirrored port side and the patterns
/// are a different way to write the same aircraft.
fn the_parts_sit_where_they_did(rows: &[Value]) {
    // The port cockpit panes are the starboard designs' `port` configurations, under the port
    // parts' own numbers.
    for (design, number) in [
        ("cockpit_window_front", "FUS-WS-032"),
        ("cockpit_window_side", "FUS-WS-034"),
        ("cockpit_window_aft", "FUS-WS-036"),
    ] {
        let port: Vec<&Value> = rows
            .iter()
            .filter(|r| {
                r["definition"]["file"]
                    .as_str()
                    .is_some_and(|f| f.ends_with(&format!("{design}.part")))
                    && r["configuration"] == "port"
            })
            .collect();
        assert!(!port.is_empty(), "{design} has no port occurrence");
        assert!(
            port.iter().all(|r| r["part_number"] == number),
            "{design}'s port hand is {number}: {:?}",
            port.iter().map(|r| &r["part_number"]).collect::<Vec<_>>()
        );
    }
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../runmat-cad/wasm/tests/aircraft/occurrences.json"
    );
    let now = part_placements(rows);
    if std::env::var("RUNMAT_AIRCRAFT_OCCURRENCES_WRITE").is_ok() {
        std::fs::write(path, serde_json::to_string(&now).unwrap() + "\n").unwrap();
        return;
    }
    let before: BTreeMap<String, Vec<[f64; 12]>> =
        serde_json::from_str(&std::fs::read_to_string(path).expect("occurrences.json")).unwrap();
    let names = |m: &BTreeMap<String, Vec<[f64; 12]>>| {
        m.iter()
            .map(|(k, v)| format!("{k} × {}", v.len()))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        names(&now),
        names(&before),
        "the designs placed, and how often"
    );
    let mut worst = (0.0f64, String::new());
    for (name, list) in &before {
        for (a, b) in list.iter().zip(&now[name]) {
            for (x, y) in a.iter().zip(b) {
                let d = (x - y).abs() / if x.abs() > 1.0 { x.abs() } else { 1.0 };
                if d > worst.0 {
                    worst = (d, format!("{name}: {a:?} became {b:?}"));
                }
            }
        }
    }
    assert!(
        worst.0 <= 1e-9,
        "a part moved by {:e}: {}",
        worst.0,
        worst.1
    );
}

/// The airliner example in runmat-examples (`aerospace/airliner`) holds exactly the files this
/// generator writes, byte for byte, plus its project files (`runmat.toml`, `example.json`,
/// `.artifacts/`). `RUNMAT_EXAMPLES_DIR` names the runmat-examples checkout; by default it is
/// the sibling of this repository.
#[test]
fn the_runmat_examples_airliner_holds_the_generated_files() {
    let root = std::env::var("RUNMAT_EXAMPLES_DIR")
        .unwrap_or_else(|_| format!("{}/../../../../runmat-examples", env!("CARGO_MANIFEST_DIR")));
    let dir = std::path::Path::new(&root).join("aerospace/airliner");
    if !dir.is_dir() {
        eprintln!("no runmat-examples checkout at {root}; set RUNMAT_EXAMPLES_DIR to check the airliner example");
        return;
    }
    let generated: BTreeMap<String, String> = files().into_iter().collect();
    // Every design file in the example's folders, by its path in the project.
    fn designs(root: &std::path::Path, at: &std::path::Path, out: &mut Vec<String>) {
        for e in std::fs::read_dir(at).unwrap().filter_map(|e| e.ok()) {
            let path = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            if path.is_dir() {
                designs(root, &path, out);
            } else if name.ends_with(".part") || name.ends_with(".assembly") {
                out.push(
                    path.strip_prefix(root)
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
    }
    let mut on_disk = Vec::new();
    designs(&dir, &dir, &mut on_disk);
    on_disk.sort();
    assert_eq!(
        on_disk,
        generated.keys().cloned().collect::<Vec<_>>(),
        "the example's design files"
    );
    for (name, text) in &generated {
        let disk = std::fs::read_to_string(dir.join(name)).unwrap();
        assert!(
            &disk == text,
            "{name} differs from the generator; rerun with RUNMAT_AIRCRAFT_DIR={}",
            dir.display()
        );
    }
    for project in ["runmat.toml", "example.json", ".artifacts/workspace.json"] {
        assert!(dir.join(project).is_file(), "the example has {project}");
    }
}
