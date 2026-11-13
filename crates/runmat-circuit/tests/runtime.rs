//! The document runtime end to end on a real project: `io.circuit`, `io.circuit.layout` and
//! `io.circuit.lock` printed from the multi-channel ECO scenario, then opened, compiled,
//! reviewed and edited through `runmat_circuit::runtime` only, as the Desktop does.
//!
//! The three files are checked in at `tests/projects/io/` and in the Desktop at
//! `desktop/src/app/components/circuit/fixtures/io/`. Set `RUNMAT_CIRCUIT_PROJECTS=1` to
//! rewrite both; with it unset, the test fails when either copy is stale.

use runmat_circuit::fixture::eco_scenario;
use runmat_circuit::index::{CircuitIndex, EcoIndex};
use runmat_circuit::runtime::{
    self, CompileOptions, EcoSettings, EditRequest, Files, LAYOUT_SUFFIX,
};
use runmat_circuit::syntax::board::{parse_board, print_board};
use runmat_circuit::syntax::print_circuit;
use runmat_design_syntax::document::UseItem;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::path::PathBuf;

const CIRCUIT: &str = "/p/io.circuit";
const LAYOUT: &str = "/p/io.circuit.layout";
const LOCK: &str = "/p/io.circuit.lock";

/// The three files, printed from the scenario.
fn project_files() -> [(&'static str, String); 3] {
    let mut sc = eco_scenario();
    // Each file is named for its declaration.
    sc.design.name = "io".into();
    sc.board.name = "io".into();
    let circuit = print_circuit(&sc.design).expect("the scenario prints");
    let uses = vec![UseItem {
        symbol: Some(("circuit".into(), "io".into())),
        from: String::new(),
        alias: None,
        revision: None,
        span: Default::default(),
    }];
    let layout = print_board(&sc.board, &uses);
    [
        ("io.circuit", circuit),
        ("io.circuit.layout", layout),
        ("io.circuit.lock", sc.lock.to_text()),
    ]
}

fn check_in(dir: &PathBuf, files: &[(&str, String)], write: bool) {
    if write {
        std::fs::create_dir_all(dir).unwrap();
    }
    for (name, text) in files {
        let path = dir.join(name);
        if write {
            std::fs::write(&path, text).unwrap();
        } else {
            let checked_in = std::fs::read_to_string(&path).unwrap_or_default();
            assert!(
                &checked_in == text,
                "{} is stale; regenerate with RUNMAT_CIRCUIT_PROJECTS=1",
                path.display()
            );
        }
    }
}

#[test]
fn the_checked_in_project_is_current() {
    let files = project_files();
    let write = std::env::var("RUNMAT_CIRCUIT_PROJECTS").is_ok();
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/projects/io");
    check_in(&here, &files, write);
    let desktop = std::env::var_os("RUNMAT_PRIVATE_CAD_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../runmat-private-cad")
        })
        .join("desktop/src/app/components/circuit/fixtures");
    if desktop.exists() {
        check_in(&desktop.join("io"), &files, write);
    }
}

fn files() -> Files {
    project_files()
        .into_iter()
        .map(|(name, text)| (format!("/p/{name}"), text))
        .collect()
}

fn compile(path: &str, files: &Files, options: &CompileOptions) -> CircuitIndex {
    let out = runtime::compile(path, &files[path], files, options);
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    out.index.expect("compiles")
}

/// The lines that can apply together: every line whose status is not an error, without the
/// lines that need one that is.
fn applicable(index: &EcoIndex) -> BTreeSet<String> {
    let rows: Vec<_> = index.groups.iter().flat_map(|g| g.rows.iter()).collect();
    let mut on: BTreeSet<String> = rows
        .iter()
        .filter(|r| r.status != "error")
        .map(|r| r.id.clone())
        .collect();
    loop {
        let before = on.len();
        let keep: BTreeSet<String> = rows
            .iter()
            .filter(|r| on.contains(&r.id) && r.requires.iter().all(|q| on.contains(q)))
            .map(|r| r.id.clone())
            .collect();
        on = keep;
        if on.len() == before {
            return on;
        }
    }
}

fn request(command: &str, values: Value) -> EditRequest {
    EditRequest {
        command: command.into(),
        values: values.as_object().cloned().unwrap_or_default(),
        selection: Vec::new(),
        options: CompileOptions::default(),
    }
}

fn changed(out: &runtime::EditOutput, path: &str) -> runtime::FileChange {
    out.files
        .iter()
        .find(|f| f.path == path)
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "{path} changed: {:?}",
                out.files.iter().map(|f| &f.path).collect::<Vec<_>>()
            )
        })
}

#[test]
fn either_file_opens_the_same_circuit_and_board() {
    let files = files();
    assert_eq!(
        runtime::dependencies(LAYOUT, &files[LAYOUT]),
        // The layout names its circuit by symbol: the host lists the folder, and the circuit
        // it reads there names its lock.
        vec!["/p/".to_owned()]
    );
    assert_eq!(
        runtime::dependencies(CIRCUIT, &files[CIRCUIT]),
        vec![LAYOUT.to_owned(), LOCK.to_owned()]
    );
    let from_layout = compile(LAYOUT, &files, &CompileOptions::default());
    let from_circuit = compile(CIRCUIT, &files, &CompileOptions::default());
    assert_eq!(from_layout, from_circuit);
    assert!(from_layout.board.is_some());
    assert!(
        from_layout.eco.total > 0,
        "the third channel waits for the board"
    );
    assert!(
        from_layout.back_annotation.total > 0,
        "board edits wait for the schematic"
    );

    // Without its layout, a circuit is the schematic alone.
    let mut alone = files.clone();
    alone.remove(LAYOUT);
    let schematic = compile(CIRCUIT, &alone, &CompileOptions::default());
    assert!(schematic.board.is_none() && schematic.eco.total == 0);
}

#[test]
fn errors_are_diagnostics_in_their_own_file() {
    let mut files = files();
    files.insert(
        CIRCUIT.into(),
        files[CIRCUIT].replacen("component", "componnet", 1),
    );
    let out = runtime::compile(LAYOUT, &files[LAYOUT], &files, &CompileOptions::default());
    assert!(out.index.is_none());
    assert!(!out.diagnostics.is_empty() && out.diagnostics.iter().all(|d| d.path == CIRCUIT));

    let mut missing = self::files();
    missing.remove(CIRCUIT);
    let out = runtime::compile(
        LAYOUT,
        &missing[LAYOUT],
        &missing,
        &CompileOptions::default(),
    );
    // The layout names its circuit by symbol: with no file of the folder declaring it, the
    // error is on the layout's `use circuit` line.
    assert_eq!(out.diagnostics[0].message, "no `circuit io` in this folder");
    assert_eq!(out.diagnostics[0].path, LAYOUT);
}

#[test]
fn the_review_is_recomputed_with_the_runs_ignore_settings() {
    let files = files();
    let full = compile(LAYOUT, &files, &CompileOptions::default()).eco;
    let room_rows = |i: &EcoIndex| {
        i.groups
            .iter()
            .flat_map(|g| &g.rows)
            .filter(|r| r.kind.contains("room"))
            .count()
    };
    assert!(room_rows(&full) > 0, "the new channel gets a room");
    let mut run = EcoSettings::default();
    run.ignore.insert("rooms".into(), true);
    let options = CompileOptions {
        eco: Some(run),
        ..CompileOptions::default()
    };
    let reviewed = runtime::review(LAYOUT, &files[LAYOUT], &files, &options)
        .eco
        .unwrap();
    assert_eq!(room_rows(&reviewed), 0);
    assert!(reviewed.total < full.total);
    // The command's values carry the same settings.
    let from_values = EcoSettings::from_values(
        json!({ "ignore_rooms": true, "ignored_parameters": ["datasheet", " "] })
            .as_object()
            .unwrap(),
    );
    assert_eq!(from_values.ignore.get("rooms"), Some(&true));
    assert_eq!(
        from_values.ignored_parameters,
        Some(vec!["datasheet".to_owned()])
    );
}

#[test]
fn update_board_applies_as_minimal_edits_and_the_board_takes_the_lines() {
    let files = files();
    let before = compile(LAYOUT, &files, &CompileOptions::default());
    let on = applicable(&before.eco);
    assert!(!on.is_empty());
    let changes: Vec<String> = on.iter().map(|id| format!("eco:{id}")).collect();
    let out = runtime::edit(
        LAYOUT,
        &files[LAYOUT],
        &files,
        &request("schematic.update-board", json!({ "changes": changes })),
    )
    .unwrap_or_else(|e| panic!("{}", e.message));
    assert_eq!(out.files.len(), 1);
    // The summary is a sentence a reviewer reads: counts agree with their nouns.
    assert!(out.summary.starts_with("Applied "), "{}", out.summary);
    assert!(!out.summary.contains("(s)"), "{}", out.summary);
    let layout = changed(&out, LAYOUT);
    assert!(!layout.reprinted, "minimal edits");
    let statements = files[LAYOUT].lines().count();
    assert!(
        layout.edits.len() < statements / 4,
        "{} edits for {statements} lines",
        layout.edits.len()
    );
    // The edits applied to the old text give the new text.
    let mut text = files[LAYOUT].encode_utf16().collect::<Vec<u16>>();
    for e in layout.edits.iter().rev() {
        text.splice(e.start..e.end, e.text.encode_utf16());
    }
    let mut sorted = layout.edits.clone();
    sorted.sort_by_key(|e| e.start);
    assert_eq!(sorted, layout.edits, "edits come in source order");
    assert_eq!(String::from_utf16(&text).unwrap(), layout.text);

    let mut next = files.clone();
    next.insert(LAYOUT.into(), layout.text.clone());
    let after = compile(LAYOUT, &next, &CompileOptions::default());
    let pending: BTreeSet<String> = after
        .eco
        .groups
        .iter()
        .flat_map(|g| g.rows.iter().map(|r| r.id.clone()))
        .collect();
    let still: Vec<&String> = on.iter().filter(|id| pending.contains(*id)).collect();
    assert!(
        still.is_empty(),
        "applied lines are no longer pending: {still:?}"
    );
    assert!(after.eco.total < before.eco.total);

    // The same run from the schematic writes the board file.
    let from_circuit = runtime::edit(
        CIRCUIT,
        &files[CIRCUIT],
        &files,
        &request(
            "schematic.update-board",
            json!({ "changes": on.iter().map(|id| format!("eco:{id}")).collect::<Vec<_>>() }),
        ),
    )
    .unwrap_or_else(|e| panic!("{}", e.message));
    assert_eq!(changed(&from_circuit, LAYOUT).text, layout.text);
}

#[test]
fn a_failing_line_refuses_the_whole_apply() {
    let files = files();
    let index = compile(LAYOUT, &files, &CompileOptions::default());
    let rows: Vec<_> = index.eco.groups.iter().flat_map(|g| &g.rows).collect();
    let failing = rows
        .iter()
        .find(|r| r.status == "error")
        .expect("the scenario has a conflict");
    let err = runtime::edit(
        LAYOUT,
        &files[LAYOUT],
        &files,
        &request(
            "schematic.update-board",
            json!({ "changes": [format!("eco:{}", failing.id)] }),
        ),
    )
    .unwrap_err();
    assert!(
        err.message.contains("nothing was applied"),
        "{}",
        err.message
    );
    let none = runtime::edit(
        LAYOUT,
        &files[LAYOUT],
        &files,
        &request("schematic.update-board", json!({})),
    )
    .unwrap_err();
    assert!(none.message.contains("select at least one"));
}

#[test]
fn back_annotation_writes_the_schematic_and_the_board() {
    let files = files();
    let before = compile(LAYOUT, &files, &CompileOptions::default());
    let on = applicable(&before.back_annotation);
    assert!(!on.is_empty());
    let changes: Vec<String> = on.iter().map(|id| format!("eco-back:{id}")).collect();
    let out = runtime::edit(
        LAYOUT,
        &files[LAYOUT],
        &files,
        &request("board.update-schematic", json!({ "changes": changes })),
    )
    .unwrap_or_else(|e| panic!("{}", e.message));
    let mut next = files.clone();
    for f in &out.files {
        assert!(
            f.path == LAYOUT || f.path == CIRCUIT || f.path == LOCK,
            "{}",
            f.path
        );
        if f.path == CIRCUIT {
            assert!(!f.reprinted, "the schematic takes minimal edits");
        }
        next.insert(f.path.clone(), f.text.clone());
    }
    assert!(
        out.files.iter().any(|f| f.path != LAYOUT),
        "the schematic side changed"
    );
    let after = compile(LAYOUT, &next, &CompileOptions::default());
    let pending: BTreeSet<String> = after
        .back_annotation
        .groups
        .iter()
        .flat_map(|g| g.rows.iter().map(|r| r.id.clone()))
        .collect();
    assert!(on.iter().all(|id| !pending.contains(id)), "{pending:?}");
}

#[test]
fn copy_room_formats_and_board_commands_run_on_the_text() {
    let files = files();
    let board = parse_board(LAYOUT, &files[LAYOUT]).board.unwrap();
    let source = board.rooms[0].name.clone();
    let out = runtime::edit(
        LAYOUT,
        &files[LAYOUT],
        &files,
        &request(
            "board.copy-room-formats",
            json!({ "source": [format!("room:{source}")] }),
        ),
    )
    .unwrap_or_else(|e| panic!("{}", e.message));
    assert!(out.summary.contains(&source), "{}", out.summary);
    assert!(!changed(&out, LAYOUT).reprinted);

    let fp = board
        .footprints
        .values()
        .find(|f| !f.locked)
        .unwrap()
        .reference
        .clone();
    let mut moved = request("board.move", json!({ "dx": 1.5, "dy": 0 }));
    moved.selection = vec![format!("component:{fp}")];
    let out = runtime::edit(LAYOUT, &files[LAYOUT], &files, &moved)
        .unwrap_or_else(|e| panic!("{}", e.message));
    let layout = changed(&out, LAYOUT);
    assert_eq!(layout.edits.len(), 1, "one statement changes");
    assert!(
        layout.edits[0]
            .text
            .starts_with(&format!("footprint {fp} ")),
        "{}",
        layout.edits[0].text
    );

    // Flip puts the part on the bottom where it stands; flipping again brings it back.
    let mut flip = request("board.flip", json!({}));
    flip.selection = vec![format!("component:{fp}")];
    let out = runtime::edit(LAYOUT, &files[LAYOUT], &files, &flip)
        .unwrap_or_else(|e| panic!("{}", e.message));
    let flipped = changed(&out, LAYOUT);
    assert_eq!(flipped.edits.len(), 1, "one statement changes");
    assert!(
        flipped.edits[0].text.contains("side = bottom"),
        "{}",
        flipped.edits[0].text
    );
    let mut once = files.clone();
    once.insert(LAYOUT.into(), out.files[0].text.clone());
    let at = |files: &runtime::Files| {
        let b = parse_board(LAYOUT, &files[LAYOUT]).board.unwrap();
        let f = b.footprint_by_ref(&fp).unwrap();
        (f.at, f.rotation, f.side)
    };
    assert_eq!(
        (at(&once).0, at(&once).1),
        (at(&files).0, at(&files).1),
        "flips in place"
    );
    let back = runtime::edit(LAYOUT, &once[LAYOUT], &once, &flip).unwrap();
    let mut twice = once.clone();
    twice.insert(LAYOUT.into(), back.files[0].text.clone());
    assert_eq!(at(&twice), at(&files));

    let mut locked = request("board.lock", json!({}));
    locked.selection = vec![format!("component:{fp}")];
    let out = runtime::edit(LAYOUT, &files[LAYOUT], &files, &locked).unwrap();
    assert!(changed(&out, LAYOUT).edits[0]
        .text
        .contains("locked = true"));

    let route = runtime::edit(
        LAYOUT,
        &files[LAYOUT],
        &files,
        &request("board.route", json!({})),
    )
    .unwrap_err();
    assert!(route.message.contains("board view"));
    runtime::take_unhandled();
    let unknown = runtime::edit(
        LAYOUT,
        &files[LAYOUT],
        &files,
        &request("board.nonsense", json!({})),
    )
    .unwrap_err();
    assert_eq!(
        unknown.message,
        "`board.nonsense` has no handler for this file"
    );
    assert!(unknown.diagnostics.is_empty());
    assert!(runtime::take_unhandled());
    assert!(LAYOUT.ends_with(LAYOUT_SUFFIX));
}

#[test]
fn drawn_board_commands_take_canvas_points() {
    let files = files();
    let board = parse_board(LAYOUT, &files[LAYOUT]).board.unwrap();
    let pad = board
        .footprints
        .values()
        .flat_map(|f| {
            f.pad_nets
                .keys()
                .map(move |p| format!("pad:{}.{p}", f.reference))
        })
        .next()
        .unwrap();
    let run = |command: &str, values: Value, selection: Vec<String>| {
        let mut r = request(command, values);
        r.selection = selection;
        runtime::edit(LAYOUT, &files[LAYOUT], &files, &r)
            .unwrap_or_else(|e| panic!("{command}: {}", e.message))
    };
    let added = |out: &runtime::EditOutput, word: &str| {
        let layout = changed(out, LAYOUT);
        assert!(!layout.reprinted, "minimal edits");
        assert!(
            layout.edits.iter().any(|e| e.text.contains(word)),
            "{word}: {:?}",
            layout.edits
        );
    };
    // Route from a pad: the net comes from the pad, the points from the clicks.
    let out = run(
        "board.route",
        json!({ "width": 0.2, "layer": "Top" }),
        vec![
            pad.clone(),
            "point:5,5".into(),
            "point:8,5".into(),
            "point:10,7".into(),
        ],
    );
    added(&out, "track");
    // The points field wins over the selection.
    let out = run(
        "board.place-via",
        json!({ "points": [[6.0, 6.0]], "size": "0.3/0.6" }),
        vec![],
    );
    added(&out, "via");
    let square = json!([[2.0, 2.0], [12.0, 2.0], [12.0, 9.0], [2.0, 9.0]]);
    let out = run(
        "board.pour",
        json!({ "action": "draw", "points": square, "layer": "Bottom", "net": "net:GND" }),
        vec![],
    );
    added(&out, "pour");
    let out = run(
        "board.keepout",
        json!({ "points": square, "applies": "vias" }),
        vec![],
    );
    added(&out, "keepout");
    let out = run(
        "board.room",
        json!({ "name": "drawn", "points": square }),
        vec![],
    );
    added(&out, "room");
    let out = run(
        "board.dimension",
        json!({ "points": [[0.0, 0.0], [20.0, 0.0], [10.0, -3.0]] }),
        vec![],
    );
    added(&out, "dimension");
    // Too few points say what to click.
    let mut r = request(
        "board.pour",
        json!({ "action": "draw", "points": [[0.0, 0.0], [1.0, 1.0]] }),
    );
    r.selection = vec![];
    let err = runtime::edit(LAYOUT, &files[LAYOUT], &files, &r).unwrap_err();
    assert!(err.message.contains("three points"), "{}", err.message);
}

/// `board.capability` loads a board-house set by name (`limits = { capability = hdi }`); a
/// value written beside it overrides the set, the rules table says which values came from it,
/// a set's minimum applies in DRC, and an unknown name is refused.
#[test]
fn a_board_house_capability_set_loads_its_limits_and_written_values_override_it() {
    use runmat_circuit::capability::Capability;
    let mut files = files();
    let out = runtime::edit(
        LAYOUT,
        &files[LAYOUT],
        &files,
        &request("board.capability", json!({ "capability": "hdi" })),
    )
    .unwrap_or_else(|e| panic!("{}", e.message));
    let text = changed(&out, LAYOUT).text;
    let limits_line = |t: &str| {
        t.lines()
            .find(|l| l.trim_start().starts_with("limits ="))
            .unwrap_or("")
            .trim()
            .to_owned()
    };
    assert_eq!(limits_line(&text), "limits = { capability = hdi }");
    let board = parse_board(LAYOUT, &text).board.unwrap();
    assert_eq!(board.limits, Capability::Hdi.limits());
    // A written track width overrides the set; the rest come from it.
    let written = text.replace(
        "limits = { capability = hdi }",
        "limits = { capability = hdi, track = 0.2 mm }",
    );
    let board = parse_board(LAYOUT, &written).board.unwrap();
    assert_eq!(board.limits.min_track, runmat_circuit::geom::mm(0.2));
    assert_eq!(
        board.limits.min_clearance,
        Capability::Hdi.limits().min_clearance
    );
    // It prints back as written: the set by name and the one value that differs.
    let reprinted = print_board(&board, &[]);
    assert_eq!(
        limits_line(&reprinted),
        "limits = { capability = hdi, track = 0.2 }"
    );
    files.insert(LAYOUT.into(), written);
    let index = compile(LAYOUT, &files, &CompileOptions::default());
    let limits = &index.board.as_ref().unwrap().rules[0];
    assert_eq!(
        limits.values["via_diameter"], "0.30 mm (hdi capability)",
        "{limits:?}"
    );
    assert_eq!(limits.values["track"], "0.20 mm", "{limits:?}");
    // The standard set's larger minimums find more on the same board than the HDI set's.
    let standard =
        files[LAYOUT].replace("capability = hdi, track = 0.2 mm", "capability = standard");
    files.insert(LAYOUT.into(), standard);
    let index = compile(LAYOUT, &files, &CompileOptions::default());
    let drc = &index.board.as_ref().unwrap().drc;
    let hdi_drc = {
        let mut f = files.clone();
        f.insert(
            LAYOUT.into(),
            f[LAYOUT].replace("capability = standard", "capability = hdi"),
        );
        compile(LAYOUT, &f, &CompileOptions::default())
            .board
            .unwrap()
            .drc
    };
    assert!(
        drc.len() > hdi_drc.len(),
        "standard {} findings, hdi {}",
        drc.len(),
        hdi_drc.len()
    );
    let refused = runtime::edit(
        LAYOUT,
        &files[LAYOUT],
        &files,
        &request("board.capability", json!({ "capability": "prototype" })),
    );
    assert!(refused.is_err_and(|e| e.message.contains("prototype")));
    let bad = files[LAYOUT].replace("capability = standard", "capability = prototype");
    let parsed = parse_board(LAYOUT, &bad);
    assert!(
        parsed
            .diagnostics
            .iter()
            .any(|d| d.message.contains("not a board-house capability")),
        "{:?}",
        parsed.diagnostics
    );
}

/// `board.layer-stack` writes its dielectric and reports the width that meets its target
/// impedance on each copper layer, from the stackup it writes; an unknown dielectric is refused.
#[test]
fn the_layer_stack_takes_its_dielectric_and_reports_widths_for_its_target_impedance() {
    let files = files();
    let out = runtime::edit(
        LAYOUT,
        &files[LAYOUT],
        &files,
        &request(
            "board.layer-stack",
            json!({ "layers": 4, "material": "Polyimide", "thickness": 1.0, "target_z": 50 }),
        ),
    )
    .unwrap_or_else(|e| panic!("{}", e.message));
    let text = &changed(&out, LAYOUT).text;
    let board = parse_board(LAYOUT, text).board.unwrap();
    assert_eq!(board.copper_count(), 4);
    let dielectrics: Vec<_> = board
        .stackup
        .layers
        .iter()
        .filter(|l| matches!(l.kind, runmat_circuit::layout::StackKind::Dielectric { .. }))
        .collect();
    assert!(
        dielectrics.iter().all(|l| l.material.name == "Polyimide"),
        "{dielectrics:?}"
    );
    // Each width is the one the stackup's closed form gives for 50 Ω.
    for (i, name) in board.stackup.copper_names().iter().enumerate() {
        let g = runmat_circuit::impedance::TraceGeometry::of(&board.stackup, i as u32).unwrap();
        let w = g.width_for(50.0).unwrap() as f64 / 1e6;
        assert!(
            out.summary.contains(&format!("{name} {w:.3} mm")),
            "{name}: {}",
            out.summary
        );
    }
    assert!(
        out.summary.contains("50 Ω: F.Cu")
            && out.summary.contains("(microstrip)")
            && out.summary.contains("(stripline)"),
        "{}",
        out.summary
    );
    let refused = runtime::edit(
        LAYOUT,
        &files[LAYOUT],
        &files,
        &request("board.layer-stack", json!({ "material": "balsa" })),
    );
    assert!(refused.is_err_and(|e| e.message.contains("balsa")));
}

/// `board.diff-pair` routes a declared pair or says why not, and refuses a net in no pair;
/// `board.tune-length` meanders a net to its target and refuses to shorten one.
#[test]
fn the_pair_and_tuning_commands_run_on_the_layout() {
    let files = files();
    let refused = runtime::edit(
        LAYOUT,
        &files[LAYOUT],
        &files,
        &request("board.diff-pair", json!({ "pair": "net:GND" })),
    );
    assert!(refused.is_err_and(|e| e.message.contains("no differential pair")));
    // The io board's AIN01 pair: routed, or refused by DRC or the router with nothing written.
    match runtime::edit(
        LAYOUT,
        &files[LAYOUT],
        &files,
        &request(
            "board.diff-pair",
            json!({ "pair": "net:AIN0", "width": 0.15, "gap": 0.15 }),
        ),
    ) {
        Ok(out) => assert!(
            out.summary.starts_with("Routed AIN01 on F.Cu"),
            "{}",
            out.summary
        ),
        Err(e) => assert!(
            e.message.contains("DRC") || e.message.contains("no path"),
            "{}",
            e.message
        ),
    }
    // The net with the longest straight track gets 1 mm more.
    let board = parse_board(LAYOUT, &files[LAYOUT]).board.unwrap();
    let net = board
        .tracks
        .values()
        .filter(|t| t.net.is_some())
        .max_by_key(|t| {
            let (a, b) = t.endpoints();
            a.dist(b) as i64
        })
        .and_then(|t| t.net)
        .unwrap();
    let name = board.nets[net.0 as usize].name.clone();
    let before = runmat_circuit::pair::net_track_length(&board, net);
    let target = before + 1.0;
    let out = runtime::edit(
        LAYOUT,
        &files[LAYOUT],
        &files,
        &request(
            "board.tune-length",
            json!({ "net": format!("net:{name}"), "target": target, "tolerance": 0.05 }),
        ),
    )
    .unwrap_or_else(|e| panic!("{name}: {}", e.message));
    let tuned = parse_board(LAYOUT, &changed(&out, LAYOUT).text)
        .board
        .unwrap();
    let after = runmat_circuit::pair::net_track_length(&tuned, tuned.net(&name).unwrap());
    assert!(
        (after - target).abs() < 0.05,
        "{name}: {before} → {after}, target {target}; {}",
        out.summary
    );
    assert!(
        out.summary.starts_with(&format!("Tuned {name}")),
        "{}",
        out.summary
    );
    let shorter = runtime::edit(
        LAYOUT,
        &files[LAYOUT],
        &files,
        &request(
            "board.tune-length",
            json!({ "net": format!("net:{name}"), "target": 1.0 }),
        ),
    );
    assert!(shorter.is_err_and(|e| e.message.contains("longer than")));
}
