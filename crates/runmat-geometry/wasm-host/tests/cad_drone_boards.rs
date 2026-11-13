//! The camera drone's boards in its assembly, on OCCT through `runmat_cad_call` (the Desktop and
//! browser worker's entry): `use circuit esc` and `use circuit flight_controller` draw each
//! board's 3D model with its components at the occurrence's placement.
//! `RUNMAT_DRONE_SRC=<folder>` compiles another copy of the project instead.

use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn entry(path: &str, text: &str, files: &str) -> Value {
    let request = serde_json::json!({ "fn": "compile", "args": [path, text, files, "{}"] });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

fn project() -> BTreeMap<String, String> {
    let dir = std::env::var("RUNMAT_DRONE_SRC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../runmat-cad/wasm/tests/drone")
        });
    let mut all = BTreeMap::new();
    read_tree(&dir, "/p/drone", &mut all);
    all
}

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

/// The prototype and placement (metres) of occurrence `name` in an assembly display.
fn placed<'a>(display: &'a Value, name: &str) -> (&'a Value, [f64; 3]) {
    let occ = display["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .find(|o| o["path"] == name)
        .unwrap_or_else(|| panic!("no occurrence `{name}`"));
    let t = occ["transform"].as_array().unwrap();
    let at = [9, 10, 11].map(|k| t[k].as_f64().unwrap());
    let p = occ["prototype"].as_u64().unwrap() as usize;
    (&display["prototypes"][p], at)
}

/// Both boards draw their 3D model, components and all, where the assembly places them: the ESC
/// on the heatsink's standoffs at z = 10 mm and the flight controller 11.6 mm above it. No
/// occurrence draws without its model.
#[test]
fn the_drone_draws_both_boards_with_their_components_at_their_placements() {
    let all = project();
    let path = "/p/drone/drone.assembly";
    let v = entry(path, &all[path], &serde_json::to_string(&all).unwrap());
    let unbuilt: Vec<&str> = v["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|d| d["message"].as_str())
        .filter(|m| m.contains("draws as its box") || m.contains("draws nothing"))
        .collect();
    assert!(unbuilt.is_empty(), "{unbuilt:?}");
    let display = &v["display"]["data"];
    for (name, z, components) in [("esc", 0.010, 20), ("flight_controller", 0.0216, 30)] {
        let (prototype, at) = placed(display, name);
        assert_eq!(prototype["shape"], "mesh", "{name}: {}", prototype["shape"]);
        let bodies = prototype["part"]["bodies"].as_array().unwrap();
        let references: Vec<&str> = bodies
            .iter()
            .filter_map(|b| b["reference"].as_str())
            .collect();
        for layer in ["Board edge", "Solder mask", "Copper", "Silkscreen"] {
            assert!(references.contains(&layer), "{name}: {references:?}");
        }
        // The component bodies (packages, chip parts, connectors, by family) stand on the board:
        // at least a box's 12 triangles for each of `components`.
        const BOARD: [&str; 9] = [
            "Board edge",
            "Solder mask",
            "Copper",
            "Pads",
            "Silkscreen",
            "Drills",
            "Leads",
            "Terminals",
            "Pins",
        ];
        let parts: Vec<&Value> = bodies
            .iter()
            .filter(|b| b["reference"].as_str().is_some_and(|r| !BOARD.contains(&r)))
            .collect();
        let triangles: u64 = parts
            .iter()
            .map(|b| b["triangle_count"].as_u64().unwrap())
            .sum();
        assert!(
            parts.len() >= 2 && triangles >= 12 * components,
            "{name}: {triangles} component triangles in {references:?}"
        );
        assert!(
            (at[0] - 0.025).abs() < 1e-6 && at[1].abs() < 1e-6 && (at[2] - z).abs() < 1e-6,
            "{name} at {at:?}"
        );
        let row = v["projection"]["data"]["occurrences"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["path"] == name)
            .unwrap();
        assert!(row.get("problem").is_none(), "{row}");
    }
}

/// The `service` exploded view lifts each board with its stack: naming the pattern seeds
/// `stack_screw` and `spacer` moves all four of each (the `stack` grid's copies with their seed),
/// so no screw is left floating at the assembled height.
#[test]
fn the_service_explode_lifts_each_board_with_every_screw_and_spacer() {
    let all = project();
    let path = "/p/drone/drone.assembly";
    let v = entry(path, &all[path], &serde_json::to_string(&all).unwrap());
    let index = &v["projection"]["data"];
    let views = index["exploded"].as_array().unwrap();
    let view = views
        .iter()
        .find(|w| w["name"] == "service")
        .expect("the service view");
    assert!(
        view["diagnostics"]
            .as_array()
            .is_none_or(|d| d.iter().all(|d| d["severity"] != "error")),
        "{}",
        view["diagnostics"]
    );
    let assembled: std::collections::BTreeMap<String, f64> = index["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|r| {
            Some((
                r["path"].as_str()?.to_owned(),
                r["world"]["translation"][2].as_f64()?,
            ))
        })
        .collect();
    let lift = |path: &str| -> f64 {
        let o = view["occurrences"]
            .as_array()
            .unwrap()
            .iter()
            .find(|o| o["path"] == path)
            .unwrap_or_else(|| panic!("`{path}` doesn't move: {}", view["occurrences"]));
        o["transform"][11].as_f64().unwrap() - assembled[path]
    };
    let fc = lift("flight_controller");
    let esc = lift("esc");
    assert!(fc > 0.01 && esc > 0.01 && fc > esc, "fc {fc}, esc {esc}");
    for (seed, with) in [("stack_screw", fc), ("spacer", esc)] {
        let members: Vec<&String> = assembled
            .keys()
            .filter(|p| *p == seed || p.starts_with(&format!("{seed}@")))
            .collect();
        assert_eq!(members.len(), 4, "{seed}: {members:?}");
        for m in members {
            assert!(
                (lift(m) - with).abs() < 1e-9,
                "{m} lifts {} not {with}",
                lift(m)
            );
        }
    }
}
