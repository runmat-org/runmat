//! Real KiCad schematics by other authors (`tests/kicad/third_party`, licenses in its README):
//! each hierarchy is read into RunMat's model, exported back to KiCad and read again with the
//! same connectivity, and its parts match the board drawn from it. Imported with its board as
//! RunMat sources (a `part` declaration per symbol and value, the board's footprints as their
//! packages), each project compiles with the parts, designators and nets KiCad gave it.

use runmat_circuit::elab::{elaborate, ElabOptions, Netlist};
use runmat_circuit::kicad::sch::read_hierarchy;
use runmat_circuit::kicad::sch_export::{export, SchExportOptions};
use runmat_circuit::kicad::{pcb, sexpr};
use std::collections::{BTreeMap, BTreeSet};

mod import_check;

struct Project {
    name: &'static str,
    root: &'static str,
    sheets: &'static [(&'static str, &'static str)],
    board: &'static str,
}

const PROJECTS: [Project; 2] = [
    Project {
        name: "esp32-flight-controller",
        root: "flight_controller_rev2.kicad_sch",
        sheets: &[(
            "flight_controller_rev2.kicad_sch",
            include_str!(
                "kicad/third_party/esp32-flight-controller/flight_controller_rev2.kicad_sch"
            ),
        )],
        board: include_str!(
            "kicad/third_party/esp32-flight-controller/flight_controller_rev2.kicad_pcb"
        ),
    },
    Project {
        name: "ottercast-audio-v2",
        root: "OtterCastAudioV2.kicad_sch",
        sheets: &[
            (
                "OtterCastAudioV2.kicad_sch",
                include_str!("kicad/third_party/ottercast-audio-v2/OtterCastAudioV2.kicad_sch"),
            ),
            (
                "Audio.kicad_sch",
                include_str!("kicad/third_party/ottercast-audio-v2/Audio.kicad_sch"),
            ),
            (
                "MIPI.kicad_sch",
                include_str!("kicad/third_party/ottercast-audio-v2/MIPI.kicad_sch"),
            ),
            (
                "Power.kicad_sch",
                include_str!("kicad/third_party/ottercast-audio-v2/Power.kicad_sch"),
            ),
            (
                "WifiBT.kicad_sch",
                include_str!("kicad/third_party/ottercast-audio-v2/WifiBT.kicad_sch"),
            ),
        ],
        board: include_str!("kicad/third_party/ottercast-audio-v2/OtterCastAudioV2.kicad_pcb"),
    },
];

fn files(p: &Project) -> BTreeMap<String, String> {
    p.sheets
        .iter()
        .map(|(n, t)| (n.to_string(), t.to_string()))
        .collect()
}

fn netlist(files: &BTreeMap<String, String>, root: &str, name: &str) -> Netlist {
    let imp = read_hierarchy(files, root).unwrap_or_else(|e| panic!("{name}: {e}"));
    elaborate(
        &imp.design,
        &imp.pool,
        &ElabOptions::default(),
        Some(&imp.lock),
    )
}

/// Nets as sets of (designator, pad), for nets joining two or more pads.
fn partition(nl: &Netlist) -> BTreeSet<BTreeSet<(String, String)>> {
    nl.nets
        .iter()
        .map(|n| {
            n.pins
                .iter()
                .flat_map(|&(c, p)| {
                    let comp = &nl.components[c];
                    let pin = &comp.pins[p];
                    let pads = if pin.pads.is_empty() {
                        vec![pin.pin.clone()]
                    } else {
                        pin.pads.clone()
                    };
                    pads.into_iter()
                        .map(move |pad| (comp.designator.clone(), pad))
                })
                .collect::<BTreeSet<_>>()
        })
        .filter(|s| s.len() > 1)
        .collect()
}

#[test]
fn third_party_schematics_parse_and_write_token_for_token() {
    for p in &PROJECTS {
        for (file, text) in p.sheets {
            let tree = sexpr::parse(text).unwrap_or_else(|e| panic!("{}/{file}: {e}", p.name));
            assert!(
                sexpr::parse(&sexpr::write(&tree)).unwrap() == tree,
                "{}/{file}: the written file differs",
                p.name
            );
        }
    }
}

#[test]
fn third_party_schematics_read_with_their_parts_and_nets() {
    for p in &PROJECTS {
        let nl = netlist(&files(p), p.root, p.name);
        assert!(
            nl.components.len() >= 30,
            "{}: {} components",
            p.name,
            nl.components.len()
        );
        assert!(
            partition(&nl).len() >= 20,
            "{}: {} nets",
            p.name,
            partition(&nl).len()
        );
        // Every part placed on the board is in the schematic, under its designator.
        let board = pcb::read(p.board)
            .unwrap_or_else(|e| panic!("{}: {e}", p.name))
            .board;
        let parts: BTreeSet<&str> = nl
            .components
            .iter()
            .map(|c| c.designator.as_str())
            .collect();
        let missing: Vec<&str> = board
            .footprints
            .values()
            .map(|f| f.reference.as_str())
            .filter(|r| !r.is_empty() && !r.starts_with(['#', 'G', 'H', 'M']) && !parts.contains(r))
            .collect();
        assert!(
            missing.is_empty(),
            "{}: on the board, not in the schematic: {missing:?}",
            p.name
        );
    }
}

/// The board's nets as sets of (designator, pad), for nets joining two or more pads of parts.
fn board_partition(board: &runmat_circuit::layout::Board) -> BTreeSet<BTreeSet<(String, String)>> {
    let mut nets: BTreeMap<u32, BTreeSet<(String, String)>> = BTreeMap::new();
    for f in board.footprints.values() {
        for (pad, net) in &f.pad_nets {
            nets.entry(net.0)
                .or_default()
                .insert((f.reference.clone(), pad.clone()));
        }
    }
    nets.into_values().filter(|s| s.len() > 1).collect()
}

#[test]
fn third_party_schematics_read_with_the_nets_of_the_board_drawn_from_them() {
    // KiCad wrote each board's pad nets from its schematic, so the two partitions match: a
    // reader that joins two signals (or splits one) fails here. Compared over the parts on the
    // board (a schematic newer than its board has parts the board lacks), leaving out a part
    // the import reports as split (its units on several sheets: one part in KiCad, a part per
    // sheet in RunMat).
    for p in &PROJECTS {
        let imp = read_hierarchy(&files(p), p.root).unwrap_or_else(|e| panic!("{}: {e}", p.name));
        let nl = netlist(&files(p), p.root, p.name);
        let board = pcb::read(p.board)
            .unwrap_or_else(|e| panic!("{}: {e}", p.name))
            .board;
        let on_board: BTreeSet<String> = board
            .footprints
            .values()
            .map(|f| f.reference.clone())
            .collect();
        let split: BTreeSet<String> = imp
            .warnings
            .iter()
            .filter(|w| w.contains("units are on several sheets"))
            .filter_map(|w| w.split(':').next().map(str::to_string))
            .collect();
        // The split part's other sheets get fresh designators; its pads are left out by the
        // board's name and by those designators.
        let fresh: BTreeSet<String> = nl
            .components
            .iter()
            .filter(|c| split.contains(c.path.leaf()))
            .map(|c| c.designator.clone())
            .collect();
        let keep = |s: &BTreeSet<(String, String)>| -> BTreeSet<(String, String)> {
            s.iter()
                .filter(|(d, _)| on_board.contains(d) && !split.contains(d) && !fresh.contains(d))
                .cloned()
                .collect()
        };
        let ours: BTreeSet<_> = partition(&nl)
            .iter()
            .map(keep)
            .filter(|s| s.len() > 1)
            .collect();
        let theirs: BTreeSet<_> = board_partition(&board)
            .iter()
            .map(keep)
            .filter(|s| s.len() > 1)
            .collect();
        // Each differing net, beside the board net it shares most pads with.
        let mut diffs = Vec::new();
        for n in ours.difference(&theirs) {
            let best = theirs
                .iter()
                .max_by_key(|t| t.intersection(n).count())
                .cloned()
                .unwrap_or_default();
            let extra: Vec<_> = n.difference(&best).collect();
            let missing: Vec<_> = best.difference(n).collect();
            diffs.push(format!("import has {extra:?} more, lacks {missing:?}"));
        }
        for t in theirs.difference(&ours) {
            if !ours.iter().any(|n| n.intersection(t).next().is_some()) {
                diffs.push(format!("board net {t:?} is not in the import"));
            }
        }
        assert!(diffs.is_empty(), "{}: {diffs:#?}", p.name);
    }
}

#[test]
fn third_party_schematics_export_and_read_back_with_the_same_connectivity() {
    for p in &PROJECTS {
        let nl = netlist(&files(p), p.root, p.name);
        let imp = read_hierarchy(&files(p), p.root).unwrap();
        let out = export(
            &nl,
            &imp.pool,
            &SchExportOptions {
                project: p.name.into(),
            },
        );
        let root = format!("{}.kicad_sch", p.name);
        let back = netlist(&out, &root, p.name);
        let designators = |nl: &Netlist| -> BTreeSet<String> {
            nl.components.iter().map(|c| c.designator.clone()).collect()
        };
        assert_eq!(designators(&back), designators(&nl), "{}: parts", p.name);
        assert_eq!(partition(&back), partition(&nl), "{}: nets", p.name);
        // The written hierarchy, with the board, imports to sources that pass the design check.
        let mut given = out.clone();
        given.insert(format!("{}.kicad_pcb", p.name), p.board.to_string());
        let stem = p.name.replace('-', "_");
        let imported =
            runmat_circuit::runtime::import_kicad(&given, Some(&root), &format!("boards/{stem}"))
                .unwrap_or_else(|e| panic!("{}: {e}", p.name));
        let fs: runmat_circuit::runtime::Files = imported
            .files
            .iter()
            .map(|(k, v)| (format!("/p/{k}"), v.clone()))
            .collect();
        // The written schematic keeps every part's footprint name, including parts the board
        // leaves off, so the round trip passes the design check as the first import does.
        let errors = import_check::design_errors(
            &fs,
            &format!("/p/boards/{stem}.circuit"),
            Some(&format!("/p/boards/{stem}.circuit.layout")),
        );
        assert!(
            errors.is_empty(),
            "{}: {errors:#?}\n{:?}",
            p.name,
            imported.report.lines()
        );
    }
}

#[test]
fn third_party_projects_import_to_sources_that_compile_with_the_same_connectivity() {
    use runmat_circuit::runtime::{compile, import_kicad, netlist_and_pool, CompileOptions, Files};
    for p in &PROJECTS {
        let original = netlist(&files(p), p.root, p.name);
        let mut given = files(p);
        given.insert(format!("{}.kicad_pcb", p.name), p.board.to_string());
        let stem = p.name.replace('-', "_");
        let out = import_kicad(&given, Some(p.root), &format!("boards/{stem}"))
            .unwrap_or_else(|e| panic!("{}: {e}", p.name));
        let fs: Files = out
            .files
            .iter()
            .map(|(k, v)| (format!("/p/{k}"), v.clone()))
            .collect();
        let path = format!("/p/boards/{stem}.circuit");
        let text = &fs[&path];
        // Each symbol and value is a part declaration, its package a footprint the board gave.
        assert!(text.contains("part \""), "{}: {text}", p.name);
        let footprints = fs.keys().filter(|k| k.ends_with(".kicad_mod")).count();
        assert!(footprints >= 10, "{}: {footprints} footprints", p.name);
        let (back, _) = netlist_and_pool(&path, text, &fs, &CompileOptions::default())
            .unwrap_or_else(|d| {
                panic!(
                    "{}: {:#?}",
                    p.name,
                    d.iter().map(|d| &d.message).collect::<Vec<_>>()
                )
            });
        let designators = |nl: &Netlist| -> BTreeSet<String> {
            nl.components.iter().map(|c| c.designator.clone()).collect()
        };
        assert_eq!(
            designators(&back),
            designators(&original),
            "{}: parts",
            p.name
        );
        assert_eq!(partition(&back), partition(&original), "{}: nets", p.name);
        // The circuit's own check (its elaboration and ERC) has no error. A differential pair
        // is inferred only from labels of nets (the flight controller's `usb_esd+` label touches
        // nothing), and its usb+/usb- crossover, drawn as a wire ending on the other's middle
        // without a junction, stays two nets as KiCad reads it.
        let layout = format!("/p/boards/{stem}.circuit.layout");
        let errors = import_check::design_errors(&fs, &path, Some(&layout));
        assert!(errors.is_empty(), "{}: {errors:#?}", p.name);
        let compiled = compile(&layout, &fs[&layout], &fs, &CompileOptions::default());
        let errors: Vec<_> = compiled
            .diagnostics
            .iter()
            .filter(|d| d.severity == "error")
            .map(|d| &d.message)
            .collect();
        assert!(errors.is_empty(), "{}: {errors:#?}", p.name);
    }
}
