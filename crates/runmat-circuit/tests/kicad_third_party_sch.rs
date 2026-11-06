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
            .unwrap_or_else(|d| panic!("{}: {:#?}", p.name, d.iter().map(|d| &d.message).collect::<Vec<_>>()));
        let designators = |nl: &Netlist| -> BTreeSet<String> {
            nl.components.iter().map(|c| c.designator.clone()).collect()
        };
        assert_eq!(designators(&back), designators(&original), "{}: parts", p.name);
        assert_eq!(partition(&back), partition(&original), "{}: nets", p.name);
        let layout = format!("/p/boards/{stem}.circuit.layout");
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
