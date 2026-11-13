//! The agent scenes' checked facts (doc 18 §3, §8): what `design.check`, `design.inspect` and
//! `design.erc` return for every file a scene's guide turn reads, each scene guide's
//! completion status, and the staged change of a scene whose turn proposes one, through the real
//! compilers.
//!
//! The scenes live in the Desktop at `desktop/scripts/site-captures/agent-scenes/`: the scene
//! files under `files/`, the recorded guide turns in `guide-turns.json`, and these facts in
//! `facts.json`. The I/O board scenes read the I/O project of runmat-circuit's tests, the board the
//! Desktop's design fixtures show. The harness's `scene_turns` test serves these facts to each
//! recorded turn and fails when the turn names something they don't contain.
//!
//! Set `RUNMAT_AGENT_SCENE_FACTS_OUT=1` to write `facts.json`; with it unset, the test compares
//! against the sibling Desktop checkout when there is one.

use runmat_design_tools::guides::{catalog, status};
use runmat_design_tools::{call, Files};
use serde_json::{json, Map, Value};
use std::path::Path;

const IO_CIRCUIT: &str = include_str!("../../runmat-circuit/tests/projects/io/io.circuit");
const IO_LAYOUT: &str = include_str!("../../runmat-circuit/tests/projects/io/io.circuit.layout");
const IO_LOCK: &str = include_str!("../../runmat-circuit/tests/projects/io/io.circuit.lock");
// The ECO scene's project: the daq showcase board with a schematic change waiting for it
// (each channel's anti-alias capacitor CF goes from 1 nF to 2.2 nF).
const DAQ_ECO_CIRCUIT: &str =
    include_str!("../../runmat-circuit/tests/projects/daq/daq-eco.circuit");
const DAQ_LAYOUT: &str = include_str!("../../runmat-circuit/tests/projects/daq/daq.circuit.layout");
const DAQ_LOCK: &str = include_str!("../../runmat-circuit/tests/projects/daq/daq.circuit.lock");

/// The sections a circuit guide turn reads when the whole inspection doesn't fit.
const CIRCUIT_GUIDE_SECTIONS: &[&str] = &["erc_power", "unplaced", "eco", "rooms", "nets", "bom"];

fn scenes_dir() -> std::path::PathBuf {
    std::env::var_os("RUNMAT_PRIVATE_CAD_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../runmat-private-cad")
        })
        .join("desktop/scripts/site-captures/agent-scenes")
}

/// The scene workspace: every scene file at the root, as the capture creates them (with the
/// drone's flight controller, a snapshot of runmat-cad-wasm's drone board), plus the I/O project
/// and the ECO scene's daq project (its edited schematic under the name `daq.circuit`).
fn workspace(dir: &Path) -> Files {
    let mut files: Files = [
        ("/io.circuit".to_owned(), IO_CIRCUIT.to_owned()),
        ("/io.circuit.layout".to_owned(), IO_LAYOUT.to_owned()),
        ("/io.circuit.lock".to_owned(), IO_LOCK.to_owned()),
        ("/daq.circuit".to_owned(), DAQ_ECO_CIRCUIT.to_owned()),
        ("/daq.circuit.layout".to_owned(), DAQ_LAYOUT.to_owned()),
        ("/daq.circuit.lock".to_owned(), DAQ_LOCK.to_owned()),
    ]
    .into();
    let mut names: Vec<_> = std::fs::read_dir(dir.join("files"))
        .expect("scene files")
        .map(|e| e.unwrap().path())
        .collect();
    names.sort();
    for path in names {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        files.insert(format!("/{name}"), std::fs::read_to_string(&path).unwrap());
    }
    files
}

fn facts(dir: &Path) -> String {
    let files = workspace(dir);
    let turns: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("guide-turns.json")).unwrap())
            .unwrap();
    let guides = catalog();
    let mut targets = Map::new();
    let mut statuses = Map::new();
    let mut staged_changes = Map::new();
    for scene in turns["scenes"].as_array().unwrap() {
        let path = scene["path"].as_str().unwrap();
        let Some(guide) = guides.guide(scene["guide"].as_str().unwrap()) else {
            continue;
        };
        if guide.completion.tool != "design.check" {
            continue;
        }
        if !targets.contains_key(path) {
            let read = |tool: &str| match call(tool, json!({ "path": path }), &files) {
                Ok(v) => v,
                Err(e) => json!({ "error": e.code, "message": e.message }),
            };
            let mut facts = Map::new();
            facts.insert("check".into(), read("design.check"));
            facts.insert("inspect".into(), read("design.inspect"));
            if path.ends_with(".circuit") {
                facts.insert("erc".into(), read("design.erc"));
                // A board's whole inspection outgrows a guide turn's context; the turn reads
                // these sections instead (ah-core design_guide_turn CIRCUIT_GUIDE_SECTIONS).
                let sections = json!(CIRCUIT_GUIDE_SECTIONS);
                let result = call(
                    "design.inspect",
                    json!({ "path": path, "sections": sections }),
                    &files,
                )
                .unwrap_or_else(|e| json!({ "error": e.code, "message": e.message }));
                facts.insert(
                    "inspect_sections".into(),
                    json!({ "sections": sections, "result": result }),
                );
            }
            // Where every design guide stands on the target, so a scene asks only what's open.
            let mut standing = Map::new();
            for g in &guides.guides {
                if g.completion.tool != "design.check" {
                    continue;
                }
                if let Ok(s) = status(g, path, &files) {
                    standing.insert(
                        g.id.clone(),
                        json!({ "open": s.open, "complete": s.complete }),
                    );
                }
            }
            facts.insert("guides".into(), Value::Object(standing));
            targets.insert(path.to_owned(), Value::Object(facts));
        }
        // A scene whose turn proposes a change: the host stages it with `design.propose_edit`,
        // with the input the harness builds from the proposal (numbers stay numbers).
        if let Some(change) = scene["response"]["proposed_change"].as_object() {
            let mut input = json!({ "path": change["path"] });
            if let Some(command) = change.get("command").filter(|c| !c.is_null()) {
                let values: Map<String, Value> = change["values"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|v| {
                        let text = v["value"].as_str().unwrap_or_default();
                        let value = text
                            .trim()
                            .parse::<f64>()
                            .ok()
                            .and_then(serde_json::Number::from_f64)
                            .map(Value::Number)
                            .unwrap_or_else(|| Value::String(text.to_owned()));
                        (v["field"].as_str().unwrap_or_default().to_owned(), value)
                    })
                    .collect();
                input["command"] = command.clone();
                input["values"] = Value::Object(values);
                input["selection"] = change.get("selection").cloned().unwrap_or(json!([]));
            }
            if let Some(text) = change.get("text").filter(|t| !t.is_null()) {
                input["text"] = text.clone();
            }
            let staged = match call("design.propose_edit", input.clone(), &files) {
                Ok(v) => v,
                Err(e) => json!({ "error": e.code, "message": e.message }),
            };
            staged_changes.insert(
                scene["scene"].as_str().unwrap().to_owned(),
                json!({ "input": input, "staged": staged }),
            );
        }
        let s = status(guide, path, &files).expect("a design guide's status");
        statuses.insert(
            scene["scene"].as_str().unwrap().to_owned(),
            serde_json::to_value(s).unwrap(),
        );
    }
    let mut text = serde_json::to_string_pretty(&json!({
        "about": "Generated by runmat-design-tools tests/agent_scene_facts.rs (RUNMAT_AGENT_SCENE_FACTS_OUT=1): design.check, design.inspect and design.erc of every scene target through the real compilers, and each scene guide's completion status. The harness's scene_turns test serves these to the recorded guide turns and fails when a turn names something they don't contain.",
        "targets": targets,
        "status": statuses,
        "staged": staged_changes,
    }))
    .unwrap();
    text.push('\n');
    text
}

#[test]
fn agent_scene_facts_are_current() {
    let dir = scenes_dir();
    if !dir.join("guide-turns.json").exists() {
        return;
    }
    let text = facts(&dir);
    let again = facts(&dir);
    if text != again {
        if let Ok(dump) = std::env::var("RUNMAT_AGENT_SCENE_FACTS_DUMP") {
            std::fs::write(format!("{dump}/a.json"), &text).unwrap();
            std::fs::write(format!("{dump}/b.json"), &again).unwrap();
        }
        panic!("the scene facts differ between two runs");
    }
    let out = dir.join("facts.json");
    if std::env::var_os("RUNMAT_AGENT_SCENE_FACTS_OUT").is_some() {
        std::fs::write(&out, &text).unwrap();
        return;
    }
    assert_eq!(
        std::fs::read_to_string(&out).unwrap_or_default(),
        text,
        "{} is stale: rerun with RUNMAT_AGENT_SCENE_FACTS_OUT=1",
        out.display()
    );
}

/// The files whose overview a scene shows: every guide scene's design target (its overview
/// collapses to the summary line above the card) and the two overview scenes, the daq schematic
/// and the bracket part with its study and drawing.
const OVERVIEW_PATHS: &[&str] = &[
    "/bracket.drawing",
    "/bracket.part",
    "/daq.circuit",
    "/flight_controller.circuit",
    "/hook-mount.assembly",
    "/io.circuit",
    "/wall_bracket.part",
];

/// `design.overview` of each scene file over the scene workspace, as the server answers it: the
/// workspace holds the project's other design files and studies, so the links that point at a
/// file are found.
fn overviews(dir: &Path) -> String {
    let files = workspace(dir);
    let mut out = Map::new();
    for path in OVERVIEW_PATHS {
        let value = call("design.overview", json!({ "path": path }), &files)
            .unwrap_or_else(|e| panic!("{path}: {e}"));
        out.insert((*path).to_owned(), value);
    }
    let mut text = serde_json::to_string_pretty(&json!({
        "about": "Generated by runmat-design-tools tests/agent_scene_facts.rs (RUNMAT_AGENT_SCENE_FACTS_OUT=1): design.overview of each agent scene file through the real compilers. The Desktop's agent scenes show these overviews; every fact names the check it came from.",
        "overviews": out,
    }))
    .unwrap();
    text.push('\n');
    text
}

#[test]
fn agent_scene_overviews_are_current() {
    let dir = scenes_dir();
    if !dir.join("guide-turns.json").exists() {
        return;
    }
    let text = overviews(&dir);
    let out = dir.join("../../../src/generated/agent-scene-overviews.json");
    if std::env::var_os("RUNMAT_AGENT_SCENE_FACTS_OUT").is_some() {
        std::fs::write(&out, &text).unwrap();
        return;
    }
    assert_eq!(
        std::fs::read_to_string(&out).unwrap_or_default(),
        text,
        "{} is stale: rerun with RUNMAT_AGENT_SCENE_FACTS_OUT=1",
        out.display()
    );
}
