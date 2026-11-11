//! The guide catalog: declarations sound against the command catalog, the Desktop's
//! `design-guides.json` current, and the design guides' completion checks run through the real
//! compilers on fixture files.
//!
//! The Desktop checks the catalog in at `desktop/src/generated/design-guides.json`. Set
//! `RUNMAT_DESIGN_GUIDES_OUT` to that path to write it; with it unset, the test compares against
//! the sibling Desktop checkout when there is one.

use runmat_design_tools::guides::{catalog, status};
use runmat_design_tools::{dependency_closure, Files};

const IO_CIRCUIT: &str = include_str!("../../runmat-circuit/tests/projects/io/io.circuit");
const IO_LAYOUT: &str = include_str!("../../runmat-circuit/tests/projects/io/io.circuit.layout");
const IO_LOCK: &str = include_str!("../../runmat-circuit/tests/projects/io/io.circuit.lock");

/// A plate with its outline only: the requirements guide has its envelope and nothing more.
const PLATE: &str = r#"runmat 0.7
std 1

part plate {
    input width : length = 80 mm
    input depth : length = 50 mm
    input thickness : length = 10 mm

    sketch s1 on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])
        l2 = line(start = [input.width, 0 mm], end = [input.width, input.depth])
        l3 = line(start = [input.width, input.depth], end = [0 mm, input.depth])
        l4 = line(start = [0 mm, input.depth], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s1, distance = input.thickness)
}
"#;

/// The same plate with its mounting face named and published: every requirement met.
const PLATE_DONE: &str = r#"runmat 0.7
std 1

part plate {
    input width : length = 80 mm
    input depth : length = 50 mm
    input thickness : length = 10 mm

    sketch s1 on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])
        l2 = line(start = [input.width, 0 mm], end = [input.width, input.depth])
        l3 = line(start = [input.width, input.depth], end = [0 mm, input.depth])
        l4 = line(start = [0 mm, input.depth], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s1, distance = input.thickness)

    region mount : face = { select = [feature.body.face.start], expect = exactly_one }
    publish mount = region(region = mount)
}
"#;

fn command_ids() -> Vec<String> {
    let mut c = runmat_cad::commands::catalog();
    c.merge(runmat_circuit::commands::catalog());
    c.commands.iter().map(|x| x.id.to_string()).collect()
}

#[test]
fn guides_name_only_commands_the_catalog_has() {
    let ids = command_ids();
    let ids: Vec<&str> = ids.iter().map(String::as_str).collect();
    let problems = catalog().lint(&ids);
    assert!(problems.is_empty(), "{problems:#?}");
}

#[test]
fn design_guides_json_is_current() {
    let json = catalog().to_json();
    assert_eq!(json, catalog().to_json(), "deterministic");
    if let Ok(out) = std::env::var("RUNMAT_DESIGN_GUIDES_OUT") {
        std::fs::write(&out, &json).expect("write design-guides.json");
        return;
    }
    let sibling = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../runmat-private-cad/desktop/src/generated/design-guides.json");
    if let Ok(checked_in) = std::fs::read_to_string(&sibling) {
        assert_eq!(
            checked_in,
            json,
            "design-guides.json is stale: rerun with RUNMAT_DESIGN_GUIDES_OUT={}",
            sibling.display()
        );
    }
}

#[test]
fn a_part_with_only_its_outline_leaves_interfaces_and_published_values_open() {
    let c = catalog();
    let g = c.guide("part_from_requirements").unwrap();
    let files: Files = [("/p/plate.part".to_owned(), PLATE.to_owned())].into();
    let s = status(g, "/p/plate.part", &files).unwrap();
    assert!(s.check_ok, "{}", s.summary);
    assert!(!s.complete);
    assert_eq!(s.open, ["interfaces", "published"]);
    assert_eq!((s.settled, s.total), (4, 6));
    assert_eq!(s.summary, "New part from requirements: 4 of 6 decided");

    let files: Files = [("/p/plate.part".to_owned(), PLATE_DONE.to_owned())].into();
    let s = status(g, "/p/plate.part", &files).unwrap();
    assert!(s.complete, "{s:?}");
    assert!(s.open.is_empty());
}

#[test]
fn a_part_that_does_not_compile_is_not_complete() {
    let c = catalog();
    let g = c.guide("manufacturable").unwrap();
    let broken = PLATE.replace("distance = input.thickness", "distance = input.missing");
    let files: Files = [("/p/plate.part".to_owned(), broken)].into();
    let s = status(g, "/p/plate.part", &files).unwrap();
    assert!(!s.check_ok);
    assert!(!s.complete);
}

/// The I/O schematic compiles with every block and net, but ERC finds power pins no source
/// drives, so the rails stay open; its board leaves the third channel's parts unplaced until it
/// takes the schematic's changes, so placement and the ECO stay open.
#[test]
fn the_io_schematic_leaves_its_rails_open_and_its_board_its_placement_and_changes() {
    let all: Files = [
        ("/b/io.circuit".to_owned(), IO_CIRCUIT.to_owned()),
        ("/b/io.circuit.layout".to_owned(), IO_LAYOUT.to_owned()),
        ("/b/io.circuit.lock".to_owned(), IO_LOCK.to_owned()),
    ]
    .into();
    let c = catalog();
    let files = dependency_closure(
        "/b/io.circuit",
        IO_CIRCUIT,
        &all.keys().cloned().collect::<Vec<_>>(),
        |p| all.get(p).cloned(),
        64,
    );
    let s = status(
        c.guide("schematic_capture").unwrap(),
        "/b/io.circuit",
        &files,
    )
    .unwrap();
    assert!(s.check_ok && !s.complete, "{s:?}");
    assert_eq!(s.open, ["rails"]);
    let files = dependency_closure(
        "/b/io.circuit.layout",
        IO_LAYOUT,
        &all.keys().cloned().collect::<Vec<_>>(),
        |p| all.get(p).cloned(),
        64,
    );
    let s = status(
        c.guide("board_bring_up").unwrap(),
        "/b/io.circuit.layout",
        &files,
    )
    .unwrap();
    assert!(s.check_ok && !s.complete, "{s:?}");
    assert_eq!(s.open, ["placement"]);
    let s = status(
        c.guide("engineering_change").unwrap(),
        "/b/io.circuit.layout",
        &files,
    )
    .unwrap();
    assert_eq!(s.open, ["differences"], "{s:?}");
}

#[test]
fn guides_that_write_fea_are_checked_by_the_runtime() {
    let c = catalog();
    let e = status(
        c.guide("simulation_setup").unwrap(),
        "/s/a.fea",
        &Files::new(),
    )
    .unwrap_err();
    assert_eq!(e.code, "unsupported");
}
