//! Every occurrence an assembly's projection lists has its own id: pattern copies, mirrored
//! copies (one per instance of a mirrored pattern) and the occurrences nested inside them. The
//! ids come from the occurrence's path, so a second compile and an edit elsewhere in the
//! project keep them. Runs over every assembly of the checked-in corpus (the airliner, the
//! drone and the simulation examples) on the grid test double: only the structure is read.

use runmat_cad_wasm::runtime::{assembly_index, Files};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[path = "aircraft_text/mod.rs"]
mod aircraft_text;

fn read_tree(dir: &Path, at: &str, all: &mut Files) {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
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

fn here() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn airliner() -> Files {
    aircraft_text::files()
        .into_iter()
        .map(|(n, t)| (format!("/p/aircraft/{n}"), t))
        .collect()
}

/// The corpus as projects: the airliner, the drone, and each simulation example.
fn corpus() -> Vec<(String, Files)> {
    let mut out = vec![("airliner".to_string(), airliner())];
    let mut drone = Files::new();
    read_tree(&here().join("drone"), "/p/drone", &mut drone);
    out.push(("drone".into(), drone));
    let mut sims: Vec<_> = std::fs::read_dir(here().join("sim"))
        .unwrap()
        .flatten()
        .filter(|e| e.path().is_dir())
        .collect();
    sims.sort_by_key(|e| e.file_name());
    for e in sims {
        let name = e.file_name().to_string_lossy().into_owned();
        let mut files = Files::new();
        read_tree(&e.path(), &format!("/p/{name}"), &mut files);
        out.push((format!("sim/{name}"), files));
    }
    out
}

/// Each occurrence's id by its path, failing on any id two occurrences share.
fn ids(path: &str, files: &Files) -> BTreeMap<String, String> {
    let (index, _) = assembly_index(path, &files[path], files)
        .unwrap_or_else(|d| panic!("{path} does not parse: {d:?}"));
    let rows = index["occurrences"].as_array().cloned().unwrap_or_default();
    assert!(!rows.is_empty(), "{path} lists no occurrence");
    let mut by_path = BTreeMap::new();
    let mut by_id: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for r in &rows {
        let id = match &r["uuid"] {
            Value::String(s) => s.clone(),
            v => v.to_string(),
        };
        let p = r["path"].as_str().unwrap_or("").to_string();
        by_id.entry(id.clone()).or_default().push(p.clone());
        by_path.insert(p, id);
    }
    let shared: Vec<_> = by_id
        .iter()
        .filter(|(_, ps)| ps.len() > 1)
        .map(|(id, ps)| {
            format!(
                "{id}: {} occurrences, e.g. {:?}",
                ps.len(),
                &ps[..ps.len().min(3)]
            )
        })
        .collect();
    assert!(
        shared.is_empty(),
        "{path}: occurrences share ids:\n{}",
        shared.join("\n")
    );
    assert_eq!(by_path.len(), rows.len(), "{path}: two rows share a path");
    by_path
}

#[test]
fn every_corpus_occurrence_has_its_own_id_and_keeps_it_on_a_second_compile() {
    let mut assemblies = 0;
    let mut occurrences = 0;
    for (project, files) in corpus() {
        for path in files.keys().filter(|p| p.ends_with(".assembly")) {
            let first = ids(path, &files);
            let second = ids(path, &files);
            assert_eq!(
                first, second,
                "{project}: {path} ids change between compiles"
            );
            assemblies += 1;
            occurrences += first.len();
        }
    }
    eprintln!("{assemblies} assemblies, {occurrences} occurrences");
    assert!(
        assemblies >= 20,
        "only {assemblies} corpus assemblies found"
    );
}

#[test]
fn each_mirrored_rivet_of_the_airliner_wing_is_its_own_occurrence() {
    let files = airliner();
    let wing = ids("/p/aircraft/wing/wing.assembly", &files);
    let port: Vec<_> = wing
        .iter()
        .filter(|(p, _)| p.contains("@rivets_port["))
        .collect();
    // Four mirrored rows (seed + instances each), every image its own occurrence.
    assert!(port.len() > 400, "{} port rivets", port.len());
    let starboard = wing
        .keys()
        .filter(|p| p.starts_with("rivet_") && !p.contains("@rivets_port["))
        .count();
    assert_eq!(
        port.len(),
        starboard,
        "every starboard rivet has one port image"
    );

    // From the top: the port rivets under `wing` keep distinct ids there too.
    let top = ids("/p/aircraft/airliner.assembly", &files);
    let n = top.keys().filter(|p| p.contains("@rivets_port[")).count();
    assert_eq!(n, port.len());
}

#[test]
fn an_edit_elsewhere_keeps_every_occurrence_id() {
    let mut files = airliner();
    let path = "/p/aircraft/wing/wing.assembly";
    let before = ids(path, &files);
    let text = files[path].clone();
    // A second copy of the first rivet's block, as a new occurrence `spare_rivet`.
    let start = text
        .find("    use part rivet")
        .expect("the wing places rivets");
    let end = start + text[start..].find("\n    }\n").expect("the rivet's block") + 7;
    let spare = text[start..end].replacen("rivet_front_upper", "spare_rivet", 1);
    assert!(spare.contains("spare_rivet"), "{spare}");
    let mut edited = String::new();
    for l in text.lines() {
        if l.trim_start().starts_with("pattern front_upper_cap") {
            // One rivet fewer in this row.
            let (head, tail) = l.rsplit_once("count = ").expect("a counted row");
            let n: u32 = tail.trim_end_matches(')').parse().expect("a count");
            edited += &format!("{head}count = {})\n", n - 1);
        } else {
            edited += l;
            edited += "\n";
        }
    }
    edited.insert_str(
        edited.find("    mirror ").expect("the wing mirrors"),
        &spare,
    );
    files.insert(path.into(), edited);
    let after = ids(path, &files);
    assert!(
        after.contains_key("spare_rivet"),
        "{:?}",
        after.keys().take(5).collect::<Vec<_>>()
    );
    let mut kept = 0;
    for (p, id) in &before {
        if let Some(now) = after.get(p) {
            assert_eq!(now, id, "{p} changed id after an unrelated edit");
            kept += 1;
        }
    }
    // Only the dropped instance and its mirror image left.
    assert_eq!(kept, before.len() - 2, "{} of {} kept", kept, before.len());
}

/// `RUNMAT_EXPLODE_CHECK=<folder>:<top file>` compiles another project's top assembly and checks
/// that each of its exploded views names only what the assembly holds and moves distinct
/// occurrences (for an example edited outside this repository). Without it, nothing runs. On the
/// grid test double, parts it cannot build have no bounding box, so a radial step may report them
/// on its centre; those notes are printed, not failed.
#[test]
fn an_outside_projects_exploded_views_evaluate_cleanly() {
    let Ok(spec) = std::env::var("RUNMAT_EXPLODE_CHECK") else {
        return;
    };
    let (dir, top) = spec.split_once(':').expect("<folder>:<top file>");
    let mut files = Files::new();
    read_tree(Path::new(dir), "/p/x", &mut files);
    let path = format!("/p/x/{top}");
    let (index, _) = assembly_index(&path, &files[&path], &files).expect("parses");
    let views = index["exploded"].as_array().cloned().unwrap_or_default();
    assert!(!views.is_empty(), "{path} has no exploded view");
    for v in &views {
        let moved = v["occurrences"].as_array().map(|a| a.len()).unwrap_or(0);
        let ids: std::collections::BTreeSet<_> = v["occurrences"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| o["uuid"].to_string())
            .collect();
        eprintln!(
            "{}: {} steps, {moved} moved, {} notes",
            v["name"],
            v["steps"].as_array().map(|a| a.len()).unwrap_or(0),
            v["diagnostics"].as_array().map(|a| a.len()).unwrap_or(0)
        );
        assert_eq!(ids.len(), moved, "{}: moved ids repeat", v["name"]);
        let named_wrong: Vec<_> = v["diagnostics"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|d| d.as_str())
            .filter(|d| !d.contains("sits on the centre of radial step"))
            .collect();
        assert!(named_wrong.is_empty(), "{}: {named_wrong:?}", v["name"]);
    }
}
