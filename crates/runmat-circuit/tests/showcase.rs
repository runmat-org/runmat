//! The showcase board, `tests/projects/daq`: a three-channel, 16-bit data-acquisition board
//! (op-amp driver, SAR ADC and reference per channel, simultaneous-sampling SPI into a BGA-256
//! FPGA, USB-C power with two buck regulators) on six layers, placed, fanned out, routed and
//! poured with the product's own board commands.
//!
//! The project is checked in at `tests/projects/daq/` (with `daq_drc.circuit.layout` and
//! `daq-erc.circuit`, the DRC and ERC views' seeded variants, and `daq-eco.circuit`, the ECO scene's
//! edited schematic). Set `RUNMAT_SHOWCASE_BUILD=1` to
//! rebuild it from `tests/showcase/start.circuit.layout` with the command script in
//! `showcase/build.rs`; run it in release (`cargo test --release`), where it takes seconds.

mod showcase {
    pub mod build;
}

use std::path::PathBuf;

fn here() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/projects/daq")
}

#[test]
fn rebuild_when_asked() {
    if std::env::var("RUNMAT_SHOWCASE_BUILD").is_err() {
        return;
    }
    showcase::build::build(&here());
}

fn files() -> runmat_circuit::runtime::Files {
    ["daq.circuit", "daq.circuit.layout", "daq.circuit.lock"]
        .iter()
        .map(|name| {
            (
                format!("/p/{name}"),
                std::fs::read_to_string(here().join(name)).expect("the showcase is checked in"),
            )
        })
        .collect()
}

/// Every component of the schematic is on the board, the board passes DRC with no finding,
/// nothing is left unrouted, ERC is clean and the board is in sync with the schematic.
#[test]
fn the_showcase_board_is_placed_routed_and_clean() {
    use runmat_circuit::runtime::{compile, CompileOptions};
    let files = files();
    let layout = "/p/daq.circuit.layout";
    let out = compile(layout, &files[layout], &files, &CompileOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    let index = out.index.expect("compiles");
    let board = index.board.as_ref().expect("a board");
    let placed: std::collections::BTreeSet<&str> = board
        .footprints
        .iter()
        .map(|f| f.reference.as_str())
        .collect();
    let missing: Vec<&str> = index
        .components
        .iter()
        .map(|c| c.designator.as_str())
        .filter(|d| !placed.contains(d))
        .collect();
    assert!(missing.is_empty(), "not on the board: {missing:?}");
    assert!(
        index.components.len() >= 70,
        "{} components",
        index.components.len()
    );
    let outline = runmat_circuit::syntax::board::parse_board(layout, &files[layout])
        .board
        .expect("reads")
        .outline
        .bbox();
    for f in &board.footprints {
        let at = runmat_circuit::geom::Point::mm(f.x_mm, f.y_mm);
        assert!(
            outline.contains(at),
            "{} at {at} is off the board",
            f.reference
        );
    }
    let drc: Vec<String> = board
        .drc
        .iter()
        .map(|r| format!("{} {} {:?}", r.code, r.message, r.items))
        .collect();
    assert!(drc.is_empty(), "DRC: {drc:#?}");
    assert!(
        board.unrouted.is_empty(),
        "unrouted: {:?}",
        board.unrouted.len()
    );
    let erc: Vec<String> = index
        .erc
        .iter()
        .map(|r| format!("{} {} {:?}", r.code, r.message, r.items))
        .collect();
    assert!(erc.is_empty(), "ERC: {erc:#?}");
    assert_eq!(
        index.eco.total, 0,
        "the board is in sync with the schematic"
    );
    // Six copper layers, 1.6 mm, routed on both outer layers, BGA fanned out, poured planes.
    assert_eq!(
        board.layers.iter().filter(|l| l.kind == "copper").count(),
        6
    );
    assert!(board.tracks.len() > 150 && board.vias.len() > 100);
    assert!(board.zones.len() >= 7);
}

/// `schematic.annotate` records every designator in the lock (a project without one gets it),
/// and with `reset` renumbers only the compiler-numbered parts, keeping each prefix's set.
#[test]
fn annotate_records_designators_and_reset_renumbers_the_numbered_parts() {
    use runmat_circuit::runtime::{compile, edit, CompileOptions, EditRequest};
    use std::collections::BTreeMap;
    let mut files = files();
    let circuit = "/p/daq.circuit";
    let lock = "/p/daq.circuit.lock";
    let designators = |files: &runmat_circuit::runtime::Files| -> BTreeMap<String, String> {
        let out = compile(circuit, &files[circuit], files, &CompileOptions::default());
        out.index
            .expect("compiles")
            .components
            .into_iter()
            .map(|c| (c.path, c.designator))
            .collect()
    };
    let before = designators(&files);
    files.remove(lock);
    let request = |reset: bool| EditRequest {
        command: "schematic.annotate".into(),
        values: serde_json::json!({ "reset": reset, "order": "down" })
            .as_object()
            .cloned()
            .unwrap(),
        selection: Vec::new(),
        options: CompileOptions::default(),
    };
    let out = edit(circuit, &files[circuit], &files, &request(false)).expect("annotates");
    let written = out
        .files
        .iter()
        .find(|f| f.path == lock)
        .expect("a lock is written");
    assert!(written.text.contains("designator="), "{}", written.text);
    files.insert(lock.into(), written.text.clone());
    assert_eq!(designators(&files), before, "the lock keeps the numbering");
    // Again: nothing to change.
    let again = edit(circuit, &files[circuit], &files, &request(false)).expect("annotates");
    assert!(again.files.is_empty());

    let out = edit(circuit, &files[circuit], &files, &request(true)).expect("renumbers");
    assert!(out.summary.starts_with("Annotated"), "{}", out.summary);
    if let Some(f) = out.files.iter().find(|f| f.path == lock) {
        files.insert(lock.into(), f.text.clone());
    }
    let after = designators(&files);
    let prefix = |d: &str| d.trim_end_matches(|c: char| c.is_ascii_digit()).to_owned();
    for (path, d) in &before {
        let numbered = path.contains(']');
        if numbered {
            assert_eq!(prefix(&after[path]), prefix(d), "{path}");
        } else {
            assert_eq!(&after[path], d, "{path} is named in the source");
        }
    }
    let mut a: Vec<&String> = before.values().collect();
    let mut b: Vec<&String> = after.values().collect();
    a.sort();
    b.sort();
    assert_eq!(a, b, "the same designators, reassigned");
}

/// The DRC variant opens on its own: its layout names `daq.circuit`, compiles to a board, and
/// reports the seeded violations (the Desktop's DRC picture opens this file).
#[test]
fn the_drc_variant_compiles_to_a_board_with_findings() {
    use runmat_circuit::runtime::{compile, CompileOptions};
    let mut files = files();
    let path = "/p/daq_drc.circuit.layout";
    files.insert(
        path.into(),
        std::fs::read_to_string(here().join("daq_drc.circuit.layout")).expect("checked in"),
    );
    let out = compile(path, &files[path], &files, &CompileOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    let index = out.index.expect("compiles");
    let board = index.board.as_ref().expect("a board");
    assert!(!board.drc.is_empty(), "the seeded violations are found");
}

/// A room's interior carries only its own circuits: a track more than 1 mm inside a room is on a
/// net with a pad in that room, so a chain between channels runs in the streets between rooms
/// and around a channel it doesn't serve.
#[test]
fn tracks_inside_a_room_belong_to_its_pads() {
    use runmat_circuit::geom::{Point, Rect};
    let files = files();
    let layout = "/p/daq.circuit.layout";
    let board = runmat_circuit::syntax::board::parse_board(layout, &files[layout])
        .board
        .expect("reads");
    let pads = board.all_pads();
    let inset = 1_000_000;
    let mut through = Vec::new();
    for room in &board.rooms {
        let b = room.outline.bbox();
        let inner = Rect::new(
            Point::new(b.min.x + inset, b.min.y + inset),
            Point::new(b.max.x - inset, b.max.y - inset),
        );
        for t in board.tracks.values() {
            let Some(net) = t.net else { continue };
            if pads
                .iter()
                .any(|p| p.net == Some(net) && b.contains(p.center))
            {
                continue;
            }
            let (a, z) = t.endpoints();
            let steps = (a.dist(z) / 250_000.0).ceil().max(1.0) as i64;
            let inside = (0..=steps).any(|k| {
                let p = Point::new(a.x + (z.x - a.x) * k / steps, a.y + (z.y - a.y) * k / steps);
                inner.contains(p)
            });
            if inside {
                through.push(format!("{} through {}", board.net_name(net), room.name));
            }
        }
    }
    through.sort();
    through.dedup();
    assert!(through.is_empty(), "{through:#?}");
}

/// The ECO scene's schematic gives each channel's anti-alias capacitor CF a 2.2 nF part. Update
/// board stages exactly those changes (the part and its capacitance, in each channel), and
/// applying them leaves the board clean: DRC finds nothing, nothing is unrouted, and the board
/// is in sync again.
#[test]
fn the_eco_variant_stages_value_changes_that_keep_the_board_clean() {
    use runmat_circuit::runtime::{compile, edit, CompileOptions, EditRequest};
    let mut files = files();
    let circuit = "/p/daq.circuit";
    files.insert(
        circuit.into(),
        std::fs::read_to_string(here().join("daq-eco.circuit")).expect("checked in"),
    );
    let layout = "/p/daq.circuit.layout";
    let out = compile(layout, &files[layout], &files, &CompileOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    let index = out.index.expect("compiles");
    let rows: Vec<(String, String)> = index
        .eco
        .groups
        .iter()
        .flat_map(|g| g.rows.iter())
        .map(|r| (r.id.clone(), r.status.clone()))
        .collect();
    let want: Vec<String> = (0..3)
        .map(|k| format!("change_value:/ch[{k}]/CF"))
        .chain((0..3).map(|k| format!("change_parameter:/ch[{k}]/CF#capacitance")))
        .collect();
    assert_eq!(
        rows.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>(),
        want,
        "{rows:?}"
    );
    let values = serde_json::json!({
        "changes": want.iter().map(|id| format!("eco:{id}")).collect::<Vec<_>>().join(",")
    });
    let request = EditRequest {
        command: "schematic.update-board".into(),
        values: values.as_object().cloned().unwrap(),
        selection: vec![],
        options: CompileOptions::default(),
    };
    let applied = edit(circuit, &files[circuit], &files, &request).expect("applies");
    assert_eq!(
        applied.summary,
        "Applied C4, C10, C16: 1 nF ±5% → 2.2 nF ±5%. No tracks or vias were removed."
    );
    for f in applied.files {
        files.insert(f.path, f.text);
    }
    let out = compile(layout, &files[layout], &files, &CompileOptions::default());
    let index = out.index.expect("compiles");
    let board = index.board.as_ref().expect("a board");
    assert!(board.drc.is_empty(), "DRC: {:?}", board.drc.len());
    assert!(
        board.unrouted.is_empty(),
        "unrouted: {}",
        board.unrouted.len()
    );
    assert_eq!(index.eco.total, 0, "the board is in sync after the update");
}

/// The 3D board keeps each component as its own face referenced `component:<designator>`, and
/// each net's outer copper as `net:<name>`, the
/// ID the schematic and board tabs select it by, so selections cross-probe into 3D.
#[test]
fn the_3d_board_references_each_component_by_its_selection_id() {
    let files = files();
    let layout = "/p/daq.circuit.layout";
    let board = runmat_circuit::syntax::board::parse_board(layout, &files[layout]).board.expect("reads");
    let model = runmat_circuit::board3d::board_model(&board);
    let refs = |material: &str| -> Vec<String> {
        model
            .bodies
            .iter()
            .filter(|b| b.reference == material)
            .flat_map(|b| b.faces.iter().filter_map(|f| f.reference.clone()))
            .collect()
    };
    let packages = refs("Packages");
    assert!(!packages.is_empty() && packages.iter().all(|r| r.starts_with("component:")), "{packages:?}");
    for f in board.footprints.values().filter(|f| f.package.body.is_some()) {
        let id = format!("component:{}", f.reference);
        assert!(model.bodies.iter().any(|b| b.faces.iter().any(|x| x.reference.as_deref() == Some(id.as_str()))), "{id} has no face in 3D");
    }
    // Shared geometry stays one face named by its material, and faces number from 0 per body.
    assert_eq!(refs("Board edge"), ["Board edge"]);
    assert_eq!(refs("Silkscreen"), ["Silkscreen"]);
    assert!(model.bodies.iter().all(|b| b.faces.iter().enumerate().all(|(i, f)| f.index == i)));
    assert!(refs("Pads").iter().filter(|r| r.starts_with("component:")).count() >= board.footprints.len() / 2);
    // Outer-layer copper is a face per net, so a selected net lights in 3D.
    let copper = refs("Copper");
    let routed: std::collections::BTreeSet<String> = board
        .tracks
        .values()
        .filter(|t| t.layer == 0 || t.layer + 1 == board.copper_count())
        .filter_map(|t| t.net.map(|n| format!("net:{}", board.nets[n.0 as usize].name)))
        .collect();
    assert!(!routed.is_empty());
    for net in &routed {
        assert!(copper.contains(net), "{net} has no copper face in 3D: {copper:?}");
    }
}

/// `board.stitch` on the showcase board: stitching vias through the top GND pour, and a shield
/// row beside a signal net's tracks in the GND pours. Each run places vias, and the board stays
/// DRC-clean and fully routed with them.
#[test]
fn stitching_and_shield_vias_land_where_drc_passes() {
    use runmat_circuit::runtime::{compile, edit, CompileOptions, EditRequest};
    let files = files();
    let layout = "/p/daq.circuit.layout";
    let board = runmat_circuit::syntax::board::parse_board(layout, &files[layout]).board.unwrap();
    let top = 0u8;
    let pour = board.zones.values().find(|z| z.layers == [top] && board.nets[z.net.unwrap().0 as usize].name == "GND").unwrap();
    // A signal net with routed tracks.
    let signal = board
        .tracks
        .values()
        .filter_map(|t| t.net)
        .map(|n| board.nets[n.0 as usize].name.clone())
        .find(|n| n.starts_with("SPI") || n.contains("SCK") || n.contains("SCLK"))
        .unwrap_or_else(|| board.nets[board.tracks.values().find_map(|t| t.net).unwrap().0 as usize].name.clone());
    let vias = board.vias.len();
    for (values, label) in [
        (serde_json::json!({ "mode": "stitch", "target": format!("zone:{}", pour.id.0), "pitch": 2.5, "via": "0.3/0.6" }), "stitch"),
        (serde_json::json!({ "mode": "shield", "target": format!("net:{signal}"), "shield_net": "GND", "pitch": 2.0, "via": "0.3/0.6" }), "shield"),
    ] {
        let request: EditRequest = serde_json::from_value(serde_json::json!({ "command": "board.stitch", "values": values })).unwrap();
        let out = edit(layout, &files[layout], &files, &request).unwrap_or_else(|e| panic!("{label}: {}", e.message));
        eprintln!("{label}: {}", out.summary);
        let mut edited = files.clone();
        let text = out.files.iter().find(|f| f.path == layout).unwrap().text.clone();
        edited.insert(layout.into(), text.clone());
        write_when_asked(&format!("daq-{label}.circuit.layout"), &text);
        let after = runmat_circuit::syntax::board::parse_board(layout, &text).board.unwrap();
        let placed = after.vias.len() - vias;
        assert!(placed >= 4, "{label}: {placed} via(s): {}", out.summary);
        assert!(out.summary.contains(&format!("Placed {placed} GND via(s)")), "{}", out.summary);
        let index = compile(layout, &text, &edited, &CompileOptions::default()).index.expect("compiles");
        let b = index.board.as_ref().unwrap();
        let drc: Vec<String> = b.drc.iter().map(|r| format!("{} {} {:?}", r.code, r.message, r.items)).collect();
        assert!(drc.is_empty(), "{label} DRC: {drc:#?}");
        assert!(b.unrouted.is_empty(), "{label}: {} unrouted", b.unrouted.len());
    }
}

/// `board.teardrops` on the showcase board adds a teardrop pour at track ends on pads and vias,
/// the board stays DRC-clean and routed, and Remove takes every one out again.
#[test]
fn teardrops_are_added_where_drc_passes_and_removed_again() {
    use runmat_circuit::runtime::{compile, edit, CompileOptions, EditRequest};
    let files = files();
    let layout = "/p/daq.circuit.layout";
    let zones = runmat_circuit::syntax::board::parse_board(layout, &files[layout]).board.unwrap().zones.len();
    let run = |text: &str, action: &str| {
        let request: EditRequest =
            serde_json::from_value(serde_json::json!({ "command": "board.teardrops", "values": { "action": action, "all": true } })).unwrap();
        let mut f = files.clone();
        f.insert(layout.into(), text.to_owned());
        let out = edit(layout, text, &f, &request).unwrap_or_else(|e| panic!("{action}: {}", e.message));
        eprintln!("{action}: {}", out.summary);
        (out.files.iter().find(|c| c.path == layout).unwrap().text.clone(), out.summary)
    };
    let (added, summary) = run(&files[layout], "add");
    write_when_asked("daq-teardrops.circuit.layout", &added);
    let board = runmat_circuit::syntax::board::parse_board(layout, &added).board.unwrap();
    let drops = board.zones.values().filter(|z| z.name == "teardrop").count();
    assert!(drops >= 20 && board.zones.len() == zones + drops, "{drops} teardrops: {summary}");
    assert!(summary.starts_with(&format!("Added {drops} teardrop(s)")), "{summary}");
    let mut edited = files.clone();
    edited.insert(layout.into(), added.clone());
    let index = compile(layout, &added, &edited, &CompileOptions::default()).index.expect("compiles");
    let b = index.board.as_ref().unwrap();
    let drc: Vec<String> = b.drc.iter().map(|r| format!("{} {} {:?}", r.code, r.message, r.items)).collect();
    assert!(drc.is_empty(), "DRC: {drc:#?}");
    assert!(b.unrouted.is_empty());
    let (removed, summary) = run(&added, "remove");
    assert_eq!(summary, format!("Removed {drops} teardrop(s)"));
    let board = runmat_circuit::syntax::board::parse_board(layout, &removed).board.unwrap();
    assert_eq!(board.zones.len(), zones);
}

/// With `RUNMAT_BOARD_TOOLS_OUT` set to a folder, a board-tool test writes its edited layout
/// there, for the pictures.
fn write_when_asked(name: &str, text: &str) {
    if let Ok(dir) = std::env::var("RUNMAT_BOARD_TOOLS_OUT") {
        std::fs::write(PathBuf::from(dir).join(name), text).expect("write the edited layout");
    }
}
