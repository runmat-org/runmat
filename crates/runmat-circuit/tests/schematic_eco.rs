//! A wire drawn on the schematic canvas reaches the board through Update board (ECO): on a
//! project whose board is in sync, the wire's `.circuit` edit shows as pending lines, applying
//! them writes the layout, and nothing stays pending.

use runmat_circuit::index::{CircuitIndex, EcoIndex};
use runmat_circuit::runtime::{self, CompileOptions, EditRequest, Files};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::path::PathBuf;

const CIRCUIT: &str = "/p/io.circuit";
const LAYOUT: &str = "/p/io.circuit.layout";

fn files() -> Files {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/projects/io");
    ["io.circuit", "io.circuit.layout", "io.circuit.lock"]
        .iter()
        .map(|n| {
            (
                format!("/p/{n}"),
                std::fs::read_to_string(dir.join(n)).unwrap(),
            )
        })
        .collect()
}

fn compile(path: &str, files: &Files) -> CircuitIndex {
    let out = runtime::compile(path, &files[path], files, &CompileOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    out.index.expect("compiles")
}

fn request(command: &str, values: Value) -> EditRequest {
    EditRequest {
        command: command.into(),
        values: values.as_object().cloned().unwrap_or_default(),
        selection: Vec::new(),
        options: CompileOptions::default(),
    }
}

/// The lines that can apply together (no error, nothing they need is an error).
fn applicable(index: &EcoIndex) -> BTreeSet<String> {
    let rows: Vec<_> = index.groups.iter().flat_map(|g| g.rows.iter()).collect();
    let mut on: BTreeSet<String> = rows
        .iter()
        .filter(|r| r.status != "error")
        .map(|r| r.id.clone())
        .collect();
    loop {
        let keep: BTreeSet<String> = rows
            .iter()
            .filter(|r| on.contains(&r.id) && r.requires.iter().all(|q| on.contains(q)))
            .map(|r| r.id.clone())
            .collect();
        if keep.len() == on.len() {
            return keep;
        }
        on = keep;
    }
}

fn pending(index: &EcoIndex) -> BTreeSet<String> {
    index
        .groups
        .iter()
        .flat_map(|g| g.rows.iter().map(|r| r.id.clone()))
        .collect()
}

/// Applies every line that can apply; returns the files with the layout written.
fn update_board(files: &Files) -> Files {
    let on = applicable(&compile(LAYOUT, files).eco);
    if on.is_empty() {
        return files.clone();
    }
    let changes: Vec<String> = on.iter().map(|id| format!("eco:{id}")).collect();
    let out = runtime::edit(
        CIRCUIT,
        &files[CIRCUIT],
        files,
        &request("schematic.update-board", json!({ "changes": changes })),
    )
    .unwrap_or_else(|e| panic!("{}", e.message));
    let mut next = files.clone();
    for f in out.files {
        next.insert(f.path, f.text);
    }
    next
}

#[test]
fn a_canvas_wire_goes_to_the_board_through_update_board() {
    // The board first takes every line it can, so what follows is the wire's alone.
    let synced = update_board(&files());
    let before = pending(&compile(LAYOUT, &synced).eco);
    // Two pins on no net yet, on one sheet.
    let index = compile(CIRCUIT, &synced);
    let page = index
        .schematic
        .sheets
        .iter()
        .find(|p| {
            p.symbols
                .iter()
                .flat_map(|s| &s.pins)
                .filter(|p| p.net.is_none())
                .count()
                >= 2
        })
        .expect("a page with two free pins");
    let free: Vec<String> = page
        .symbols
        .iter()
        .flat_map(|s| {
            s.pins
                .iter()
                .filter(|p| p.net.is_none())
                .map(move |p| format!("pin:{}.{}", s.designator, p.name))
        })
        .take(2)
        .collect();
    let wired = runtime::edit(
        CIRCUIT,
        &synced[CIRCUIT],
        &synced,
        &request(
            "schematic.place-wire",
            json!({ "from": [free[0]], "to": [free[1]] }),
        ),
    )
    .unwrap_or_else(|e| panic!("{}", e.message));
    let mut next = synced.clone();
    for f in wired.files {
        next.insert(f.path, f.text);
    }
    // The wire is pending on the board.
    let after_wire = pending(&compile(LAYOUT, &next).eco);
    let new: Vec<&String> = after_wire.difference(&before).collect();
    assert!(
        !new.is_empty(),
        "the wire shows as ECO lines: {after_wire:?} vs {before:?}"
    );
    // Update board takes it, and nothing new stays pending.
    let applied = update_board(&next);
    assert_ne!(applied[LAYOUT], next[LAYOUT], "the layout is written");
    let left = pending(&compile(LAYOUT, &applied).eco);
    assert!(
        new.iter().all(|id| !left.contains(*id)),
        "applied: {left:?}"
    );
}
