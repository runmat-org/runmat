//! KiCad interchange: lossless round trips on hand-written fixtures (hierarchy, buses,
//! power symbols, multi-unit parts, diff pairs, no-connects, stackup, zones, keepouts,
//! blind and micro vias, back-side footprints, rules, unknown tokens).

use runmat_circuit::board_rules::{props, BItem, BoardGraph, BoardRules, RuleValue};
use runmat_circuit::elab::{elaborate, ElabOptions};
use runmat_circuit::geom::{mm, Point};
use runmat_circuit::kicad::{dru, footprint, pcb, sch, sexpr};
use runmat_circuit::layout::*;
use runmat_circuit::library::footprint as gen;
use std::collections::{BTreeMap, BTreeSet};

mod import_check;

const BOARD: &str = include_str!("kicad/board.kicad_pcb");
const ROOT: &str = include_str!("kicad/root.kicad_sch");
const CHILD: &str = include_str!("kicad/power.kicad_sch");
const RULES: &str = include_str!("kicad/rules.kicad_dru");
const SYMS: &str = include_str!("kicad/device.kicad_sym");

#[test]
fn pcb_unedited_round_trip_is_token_identical() {
    let doc = pcb::read(BOARD).unwrap();
    assert!(doc.warnings.is_empty(), "{:?}", doc.warnings);
    let out = doc.write();
    assert_eq!(sexpr::parse(&out).unwrap(), sexpr::parse(BOARD).unwrap());
    // And again from the written text.
    let again = pcb::read(&out).unwrap().write();
    assert_eq!(again, out);
}

#[test]
fn pcb_semantics() {
    let doc = pcb::read(BOARD).unwrap();
    let b = &doc.board;
    assert_eq!(b.copper_count(), 4);
    assert_eq!(
        b.stackup.copper_names(),
        vec!["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]
    );
    let d1 = b.stackup.dielectric_below(0).unwrap();
    assert_eq!(d1.material.dk, 4.4);
    assert_eq!(d1.thickness, mm(0.2104));
    assert_eq!(b.stackup.finish, "ENIG");
    assert_eq!(b.nets.len(), 5);
    // Outline with the circular cutout.
    assert_eq!(b.outline.holes.len(), 1);
    assert!(
        (b.outline.area() - (50.0 * 30.0 - std::f64::consts::PI * 1.5 * 1.5) * 1e12).abs()
            < 0.01e12
    );
    // R1 at (20, 15) rotated 90° counter-clockwise on screen: pad 1 at local (−0.825, 0)
    // turns to point down the screen, to KiCad (20, 15.825), i.e. y = −15.825 mm here.
    let r1 = b.footprint_by_ref("R1").unwrap();
    assert_eq!(r1.rotation, 90.0);
    let pads = r1.pads(4);
    let p1 = pads.iter().find(|p| p.name == "1").unwrap();
    assert_eq!(p1.center, Point::mm(20.0, -15.825));
    assert_eq!(b.net_name(p1.net.unwrap()), "SIG");
    assert_eq!(
        r1.properties.get("MPN").map(|s| s.as_str()),
        Some("RC0603FR-0710KL")
    );
    assert_eq!(
        r1.component.as_deref(),
        Some("/aaaaaaaa-0000-0000-0000-000000000001")
    );
    // C1 is on the back: its pads are on B.Cu and pad 1 sits at KiCad x = 30 − 0.95.
    let c1 = b.footprint_by_ref("C1").unwrap();
    assert_eq!(c1.side, Side::Bottom);
    let cp = c1.pads(4);
    assert!(cp.iter().all(|p| p.layers == [3].into()));
    assert_eq!(
        cp.iter().find(|p| p.name == "1").unwrap().center,
        Point::mm(29.05, -15.0)
    );
    // Through-hole and NPTH.
    let j1 = b.footprint_by_ref("J1").unwrap();
    let jp = j1.pads(4);
    assert_eq!(jp[0].layers.len(), 4);
    assert!(jp
        .iter()
        .any(|p| p.kind == runmat_circuit::library::PadKind::Npth));
    // Tracks, arc, vias.
    assert_eq!(b.tracks.len(), 4);
    assert!(b
        .tracks
        .values()
        .any(|t| matches!(t.geom, TrackGeom::Arc { .. })));
    let kinds: Vec<ViaKind> = b.vias.values().map(|v| v.kind(4)).collect();
    assert_eq!(
        kinds,
        vec![ViaKind::Through, ViaKind::Blind, ViaKind::Micro]
    );
    // Zone with fill and keepout.
    let z = b.zones.values().next().unwrap();
    assert_eq!(z.priority, 1);
    assert_eq!(z.layers, vec![1]);
    assert_eq!(z.clearance, Some(mm(0.3)));
    assert_eq!(z.thermal_gap, mm(0.4));
    assert!(matches!(z.islands, IslandPolicy::RemoveBelow(_)));
    assert_eq!(z.fill[&1].polys.len(), 1);
    let k = b.keepouts.values().next().unwrap();
    assert!(k.rules.tracks && k.rules.vias && !k.rules.pads && k.rules.pours);
    assert_eq!(k.layers, vec![0, 3]);
}

#[test]
fn pcb_edit_in_place_keeps_unknown_tokens() {
    let mut doc = pcb::read(BOARD).unwrap();
    let r1 = doc.board.footprint_by_ref("R1").unwrap().id;
    doc.board.footprints.get_mut(&r1).unwrap().at = Point::mm(22.0, -16.0);
    let t = *doc.board.tracks.keys().next().unwrap();
    doc.board.tracks.get_mut(&t).unwrap().width = mm(0.4);
    let v = *doc.board.vias.keys().last().unwrap();
    doc.board.vias.remove(&v);
    let sig = doc.board.net("SIG");
    doc.board
        .add_track(sig, 2, mm(0.2), Point::mm(5.0, -5.0), Point::mm(10.0, -5.0));
    let out = doc.write();
    // Unmodelled tokens survive: text, dimension, group, future token, pad pintype, uuids.
    for needle in [
        "RunMat test board",
        "(dimension",
        "(group",
        "future_token_from_kicad_11",
        "pintype",
        "11111111-1111-1111-1111-111111111117",
        "pcbplotparams",
        "sheetname",
        "(remove_unused_layers no)",
    ] {
        assert!(out.contains(needle), "lost `{needle}`");
    }
    let back = pcb::read(&out).unwrap();
    let r1b = back.board.footprint_by_ref("R1").unwrap();
    assert_eq!(r1b.at, Point::mm(22.0, -16.0));
    assert_eq!(r1b.pads(4).len(), 2);
    assert_eq!(back.board.tracks.len(), 5);
    assert!(back.board.tracks.values().any(|t| t.width == mm(0.4)));
    assert!(back.board.tracks.values().any(|t| t.layer == 2));
    assert_eq!(back.board.vias.len(), 2);
    // Edited R1 keeps its uuid and MPN property.
    assert!(out.contains("11111111-1111-1111-1111-111111111111"));
    assert_eq!(
        r1b.properties.get("MPN").map(|s| s.as_str()),
        Some("RC0603FR-0710KL")
    );
}

#[test]
fn runmat_board_exports_and_reimports() {
    let pool = runmat_circuit::fixture::standard_pool();
    let p = runmat_circuit::fixture::LargeParams {
        channels: 6,
        copper_layers: 6,
        route_every: 1,
        pours: true,
    };
    let (_, _, _, mut board) = runmat_circuit::fixture::large(&p);
    let _ = pool;
    let (mut rules, _) = BoardRules::new(&board.limits, &[]);
    rules.rebuild(&BoardGraph::new(&board, &[]));
    runmat_circuit::zone::fill_zones(&mut board, &rules, &[]);
    let text = pcb::write_board(&board);
    let back = pcb::read(&text).unwrap();
    let b2 = &back.board;
    assert_eq!(b2.copper_count(), board.copper_count());
    assert_eq!(
        b2.nets.iter().map(|n| &n.name).collect::<Vec<_>>(),
        board.nets.iter().map(|n| &n.name).collect::<Vec<_>>()
    );
    assert_eq!(b2.footprints.len(), board.footprints.len());
    assert_eq!(b2.tracks.len(), board.tracks.len());
    assert_eq!(b2.vias.len(), board.vias.len());
    assert_eq!(b2.zones.len(), board.zones.len());
    assert_eq!(b2.outline, board.outline);
    assert!((b2.stackup.thickness() - board.stackup.thickness()).abs() < 1000);
    // Every pad lands where it was, with its net.
    let key = |b: &Board| {
        let mut v: Vec<(String, String, Point, Option<String>)> = b
            .footprints
            .values()
            .flat_map(|f| {
                f.pads(b.copper_count()).into_iter().map(move |p| {
                    (
                        f.reference.clone(),
                        p.name.clone(),
                        p.center,
                        p.net.map(|n| b.net_name(n).to_string()),
                    )
                })
            })
            .collect();
        v.sort();
        v
    };
    assert_eq!(key(b2), key(&board));
    // Written again unchanged: identical.
    assert_eq!(back.write(), text);
}

#[test]
fn footprint_library_round_trip() {
    for s in [
        "R0402",
        "soic8",
        "qfn32_5mm_p0.5_ep3.3",
        "pinheader2x05_p2.54",
        "bga256_16x16_p1.0_ball0.5",
    ] {
        let p = gen::from_string(s).unwrap();
        let text = footprint::write_mod(&p);
        let back = footprint::read_mod(&text).unwrap();
        assert_eq!(back.pads.len(), p.pads.len(), "{s}");
        for (a, b) in p.pads.iter().zip(&back.pads) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.at, b.at, "{s} pad {}", a.name);
            assert_eq!(a.shape.size(), b.shape.size());
            assert_eq!(a.kind, b.kind);
        }
        assert_eq!(back.courtyard.len(), p.courtyard.len());
        assert_eq!(footprint::write_mod(&back), text, "{s} is a fixed point");
    }
}

#[test]
fn schematic_hierarchy_imports_and_elaborates() {
    let files: BTreeMap<String, String> = [
        ("root.kicad_sch".to_string(), ROOT.to_string()),
        ("power.kicad_sch".to_string(), CHILD.to_string()),
    ]
    .into();
    let imp = sch::read_hierarchy(&files, "root.kicad_sch").unwrap();
    assert!(imp.warnings.is_empty(), "{:?}", imp.warnings);
    let nl = elaborate(
        &imp.design,
        &imp.pool,
        &ElabOptions::default(),
        Some(&imp.lock),
    );
    assert!(nl.diagnostics.is_empty(), "{:#?}", nl.diagnostics);
    let net = |r: &str, p: &str| {
        nl.net_of(r, p)
            .map(|n| n.name.clone())
            .unwrap_or_else(|| format!("<none {r}.{p}>"))
    };
    // Global label and power symbols join across sheets.
    assert_eq!(net("R1", "1"), "VCC");
    assert_eq!(net("U1", "8"), "VCC");
    assert_eq!(net("R3", "2"), "VCC");
    assert_eq!(net("R1", "2"), "GND");
    assert_eq!(net("U1", "4"), "GND");
    assert_eq!(net("R5", "2"), "GND");
    // Local labels.
    assert_eq!(net("R2", "1"), "SIG");
    assert_eq!(net("U1", "3"), "SIG");
    // Sheet pin ↔ hierarchical label.
    assert_eq!(net("R2", "2"), "/power/IN");
    assert_eq!(net("R3", "1"), "/power/IN");
    // Bus through the sheet: D[0..1] members by name.
    assert_eq!(net("J2", "1"), "D0");
    assert_eq!(net("R4", "1"), "D0");
    assert_eq!(net("R5", "1"), "D1");
    // Op-amp feedback through wires and a junction; multi-unit symbol is one component.
    assert_eq!(net("U1", "1"), net("U1", "2"));
    assert_eq!(
        nl.components
            .iter()
            .filter(|c| c.designator == "U1")
            .count(),
        1
    );
    // Power symbols are not components.
    assert!(nl.components.iter().all(|c| !c.designator.starts_with('#')));
    // Diff pair by naming convention.
    assert_eq!(nl.diff_pairs.len(), 1);
    assert_eq!(nl.nets[nl.diff_pairs[0].p].name, "USB_P");
    // No-connect becomes a no-ERC marker on R4.2.
    let (ci, r4) = nl.component("R4").unwrap();
    let pi = r4.pins.iter().position(|p| p.pin == "2").unwrap();
    assert!(nl.no_erc.contains(&(ci, pi)));
    // KiCad's per-instance references come through the lock section.
    assert!(nl
        .components
        .iter()
        .any(|c| c.designator == "R3" && c.path.to_string() == "/power/R3"));
    // Footprint field became the package reference.
    assert!(
        matches!(&imp.design.sheets["root.kicad_sch"].components.iter().find(|c| c.name == "R1").unwrap().part, runmat_circuit::ir::PartRef::Entity { package: Some(p), .. } if p == "Resistor_SMD:R_0603_1608Metric")
    );
}

#[test]
fn schematic_and_symbol_files_round_trip() {
    for src in [ROOT, CHILD, SYMS] {
        let t = sexpr::parse(src).unwrap();
        assert_eq!(sexpr::parse(&sexpr::write(&t)).unwrap(), t);
    }
    // Back-annotation edits one property in place.
    let mut t = sexpr::parse(ROOT).unwrap();
    assert!(sch::set_symbol_property(&mut t, "R1", "Value", "22k"));
    let out = sexpr::write(&t);
    assert!(out.contains("\"22k\""));
    let files: BTreeMap<String, String> = [
        ("root.kicad_sch".to_string(), out),
        ("power.kicad_sch".to_string(), CHILD.to_string()),
    ]
    .into();
    let imp = sch::read_hierarchy(&files, "root.kicad_sch").unwrap();
    let r1 = imp.design.sheets["root.kicad_sch"]
        .components
        .iter()
        .find(|c| c.name == "R1")
        .unwrap();
    assert_eq!(r1.properties["Value"], "22k");
    // Symbol library: multi-unit entity with three gates.
    let syms = sch::read_symbol_lib(SYMS).unwrap();
    assert_eq!(syms.len(), 2);
    let mut pool = runmat_circuit::library::Pool::new();
    for s in &syms {
        sch::lib_into_pool(s, &mut pool);
    }
    let amp = pool.entity("Amp:LM358").unwrap();
    assert_eq!(amp.gates.len(), 3);
    assert_eq!(amp.prefix, "U");
    let pins = pool.pins(amp, None);
    assert_eq!(pins.len(), 8);
    assert!(pins
        .iter()
        .any(|p| p.pin == "8" && p.kind == runmat_circuit::library::PinType::PowerIn));
}

#[test]
fn design_rules_round_trip_and_apply() {
    let imp = dru::read(RULES, "board.kicad_dru", 2).unwrap();
    assert_eq!(
        dru::write(&imp).replace(char::is_whitespace, ""),
        RULES.replace(char::is_whitespace, "")
    );
    assert_eq!(imp.unsupported.len(), 1, "{:?}", imp.unsupported);
    assert_eq!(imp.unsupported[0].0, "exotic");
    assert_eq!(imp.rules.len(), 5);
    // Apply to the fixture board: give SIG the HV class and check the clearance it gets.
    let mut doc = pcb::read(BOARD).unwrap();
    let sig = doc.board.net("SIG").unwrap();
    doc.board.nets[sig.0 as usize].class = Some("HV".into());
    let b = &doc.board;
    let (mut rules, errs) = BoardRules::new(&b.limits, &imp.rules);
    assert!(errs.is_empty(), "{errs:?}");
    rules.rebuild(&BoardGraph::new(b, &[]));
    let sig_track = b.tracks.values().find(|t| t.net == Some(sig)).unwrap().id;
    let other = b.tracks.values().find(|t| t.net != Some(sig)).unwrap().id;
    assert_eq!(
        rules.pair_len(
            BItem::Track(sig_track),
            BItem::Track(other),
            props::CLEARANCE
        ),
        mm(1.5)
    );
    assert_eq!(
        rules.pair_len(BItem::Track(other), BItem::Track(other), props::CLEARANCE),
        b.limits.min_clearance
    );
    let g = BoardGraph::new(b, &[]);
    let via = *b.vias.keys().next().unwrap();
    assert_eq!(
        rules.value(&g, BItem::Via(via), props::ANNULAR_MIN),
        Some(RuleValue::Len(mm(0.15)))
    );
}

#[test]
fn fresh_schematic_export_reimports_with_the_same_connectivity() {
    use runmat_circuit::kicad::sch_export::{export, SchExportOptions};
    let p = runmat_circuit::fixture::LargeParams {
        channels: 24,
        copper_layers: 6,
        route_every: 0,
        pours: false,
    };
    let (pool, _, nl, _) = runmat_circuit::fixture::large(&p);
    let files = export(
        &nl,
        &pool,
        &SchExportOptions {
            project: "large".into(),
        },
    );
    // One file per sheet instance: an instance's second page (the FPGA's unused units) sits
    // right of its first on the same paper.
    let pages = runmat_circuit::schematic::layout(&nl);
    let drawn = runmat_circuit::schematic::merge_pages(&pages);
    assert_eq!(files.len(), drawn.sheets.len());
    let spare: Vec<_> = pages.sheets.iter().filter(|s| s.page == 2).collect();
    assert_eq!(spare.len(), 1, "the root's unused FPGA units are a page of their own");
    let root = &drawn.sheets[0];
    let first = &pages.sheets[0];
    assert_eq!(root.frame, "User");
    assert!((root.width_mm - first.width_mm - spare[0].width_mm).abs() < 1e-6);
    assert_eq!(root.symbols.len(), first.symbols.len() + spare[0].symbols.len());
    assert!(files.len() >= 25, "{} files", files.len());
    assert_eq!(
        files,
        export(
            &nl,
            &pool,
            &SchExportOptions {
                project: "large".into(),
            }
        )
    );
    let imp = sch::read_hierarchy(&files, "large.kicad_sch").unwrap();
    assert!(imp.warnings.is_empty(), "{:?}", imp.warnings);
    let back = elaborate(
        &imp.design,
        &imp.pool,
        &ElabOptions::default(),
        Some(&imp.lock),
    );
    // Same partition of (designator, pad) into nets.
    let partition = |nl: &runmat_circuit::elab::Netlist| {
        let mut sets: BTreeSet<BTreeSet<(String, String)>> = BTreeSet::new();
        for n in &nl.nets {
            let s: BTreeSet<(String, String)> = n
                .pins
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
                .collect();
            if s.len() > 1 {
                sets.insert(s);
            }
        }
        sets
    };
    let (a, b) = (partition(&nl), partition(&back));
    assert_eq!(a.len(), b.len());
    assert_eq!(a, b);
    let designators: BTreeSet<_> = nl.components.iter().map(|c| c.designator.clone()).collect();
    let back_d: BTreeSet<_> = back
        .components
        .iter()
        .map(|c| c.designator.clone())
        .collect();
    assert_eq!(designators, back_d);
    assert!(
        drawn
            .sheets
            .iter()
            .any(|s| s.labels.iter().any(|l| l.kind == "power")),
        "the large design draws power ports"
    );
    // The export draws the canvas's schematic: each page's wire segments and labels are the
    // layout's, flipped to KiCad's y-down page.
    for sheet in &drawn.sheets {
        let file = if sheet.path == "/" {
            "large.kicad_sch".to_owned()
        } else {
            format!(
                "large-{}.kicad_sch",
                sheet
                    .path
                    .trim_start_matches('/')
                    .replace('/', "_")
                    .replace(['[', ']'], "_")
            )
        };
        let tree = runmat_circuit::kicad::sexpr::parse(&files[&file]).unwrap();
        let key = |x: f64, y: f64| ((x * 1.0e3).round() as i64, (y * 1.0e3).round() as i64);
        let mut want: BTreeSet<((i64, i64), (i64, i64))> = BTreeSet::new();
        for w in &sheet.wires {
            for s in w.points.windows(2) {
                let a = key(s[0][0], sheet.height_mm - s[0][1]);
                let b = key(s[1][0], sheet.height_mm - s[1][1]);
                want.insert((a.min(b), a.max(b)));
            }
        }
        let got: BTreeSet<((i64, i64), (i64, i64))> = tree
            .children("wire")
            .map(|w| {
                let pts = runmat_circuit::kicad::pts_of(w.child("pts"));
                // `pts_of` reads KiCad's y-down page into y-up points.
                let a = key(pts[0].x as f64 * 1e-6, -(pts[0].y as f64) * 1e-6);
                let b = key(pts[1].x as f64 * 1e-6, -(pts[1].y as f64) * 1e-6);
                (a.min(b), a.max(b))
            })
            .collect();
        assert_eq!(got, want, "{file}");
        let labels = ["label", "global_label", "hierarchical_label"]
            .iter()
            .map(|h| tree.children(h).count())
            .sum::<usize>();
        let power: Vec<_> = sheet.labels.iter().filter(|l| l.kind == "power").collect();
        assert_eq!(labels, sheet.labels.len() - power.len(), "{file}");
        assert_eq!(
            tree.children("sheet").count(),
            sheet.sheet_symbols.len(),
            "{file}"
        );
        // Power ports are power symbols, never global labels: one `power:<rail>` symbol each,
        // valued with the rail, and one library symbol per rail.
        let power_symbols: Vec<_> = tree
            .children("symbol")
            .filter(|s| s.child_arg("lib_id").is_some_and(|l| l.starts_with("power:")))
            .collect();
        assert_eq!(power_symbols.len(), power.len(), "{file}");
        for (symbol, label) in power_symbols.iter().zip(&power) {
            let value = symbol
                .children("property")
                .find(|p| p.arg(0) == Some("Value"))
                .and_then(|p| p.arg(1));
            assert_eq!(value, Some(label.net.as_str()), "{file}");
        }
        assert!(tree
            .children("global_label")
            .all(|l| !power.iter().any(|p| Some(p.net.as_str()) == l.arg(0))));
        let rails: BTreeSet<&str> = power.iter().map(|l| l.net.as_str()).collect();
        let power_libs = tree
            .child("lib_symbols")
            .map(|ls| ls.children("symbol").filter(|s| s.child("power").is_some()).count())
            .unwrap_or(0);
        assert_eq!(power_libs, rails.len(), "{file}");
        assert_eq!(
            tree.children("symbol").count(),
            sheet.symbols.len() + power.len(),
            "{file}"
        );
    }
}

#[test]
fn a_large_part_exports_as_the_units_the_canvas_draws() {
    use runmat_circuit::kicad::sch_export::{export, SchExportOptions};
    let p = runmat_circuit::fixture::LargeParams {
        channels: 2,
        copper_layers: 4,
        route_every: 0,
        pours: false,
    };
    let (pool, _, nl, _) = runmat_circuit::fixture::large(&p);
    let fpga = nl
        .components
        .iter()
        .find(|c| c.pins.len() > runmat_circuit::schematic::SPLIT_ABOVE)
        .expect("the large design has an FPGA");
    let units = runmat_circuit::schematic::units(fpga);
    let drawn = runmat_circuit::schematic::layout(&nl);
    let placed: Vec<_> = drawn
        .sheets
        .iter()
        .flat_map(|s| s.symbols.iter())
        .filter(|s| s.designator == fpga.designator)
        .collect();
    assert_eq!(placed.len(), units.len());
    let files = export(
        &nl,
        &pool,
        &SchExportOptions {
            project: "large".into(),
        },
    );
    let all: String = files.values().cloned().collect();
    for u in &units {
        assert!(placed
            .iter()
            .any(|s| s.unit.as_deref() == Some(u.name.as_str())));
        if let Some(label) = &u.label {
            assert!(all.contains(&format!("(unit_name \"{label}\")")), "{label}");
        }
    }
    // Each unit is placed once, as its own KiCad unit of the one symbol.
    let mut seen: BTreeSet<u64> = BTreeSet::new();
    for text in files.values() {
        let tree = sexpr::parse(text).unwrap();
        for sym in tree.children("symbol") {
            let reference = sym
                .children("property")
                .find(|p| p.arg(0) == Some("Reference"))
                .and_then(|p| p.arg(1));
            if reference == Some(fpga.designator.as_str()) {
                let unit = sym
                    .child_arg("unit")
                    .and_then(|u| u.parse::<f64>().ok())
                    .unwrap() as u64;
                assert!(seen.insert(unit), "unit {unit} twice");
            }
        }
    }
    assert_eq!(seen, (1..=units.len() as u64).collect());
}

#[test]
fn custom_pad_primitives_union_with_the_anchor() {
    use runmat_circuit::library::PadShape;
    let src = r#"(footprint "T:Custom" (layer "F.Cu")
      (pad "1" smd custom (at 0 0) (size 1 1) (layers "F.Cu" "F.Mask")
        (options (clearance outline) (anchor rect))
        (primitives
          (gr_line (start 0 0) (end 3 0) (width 0.4))
          (gr_circle (center 5 0) (end 6 0) (width 0.2))
          (gr_arc (start 0 -1) (mid -1 -2) (end 0 -3) (width 0.2))
          (gr_rect (start -0.5 2) (end 0.5 3) (width 0) (fill yes))
          (gr_poly (pts (xy 2 1) (xy 3 1) (xy 2.5 1.6)) (width 0.1))
        ))
    )"#;
    let node = sexpr::parse(src).unwrap();
    let fp = footprint::read(&node, false);
    let PadShape::Custom { points, holes } = &fp.package.pads[0].shape else {
        panic!("{:?}", fp.package.pads[0].shape)
    };
    // The stroked circle is a ring: its inside stays open (a hole) since nothing covers it,
    // but it is a separate island from the anchor, so the largest island is kept.
    let r = runmat_circuit::geom::Rect::from_points(points.iter().copied());
    // Anchor (−0.5…0.5) joined by the line to x = 3.2; the arc, rectangle, polygon and ring
    // do not touch it and are separate islands (the largest island is the pad).
    assert!(r.max.x >= mm(3.1) && r.min.x <= mm(-0.49), "{r:?}");
    assert!(holes.is_empty());
    // A filled anchor with a stroked circle around its centre leaves no hole; one beside it,
    // overlapping the anchor, keeps its own hole.
    let src2 = r#"(footprint "T:Ring" (layer "F.Cu")
      (pad "1" smd custom (at 0 0) (size 0.5 0.5) (layers "F.Cu")
        (options (anchor circle))
        (primitives (gr_circle (center 1 0) (end 2 0) (width 0.6)))))"#;
    let fp2 = footprint::read(&sexpr::parse(src2).unwrap(), false);
    let PadShape::Custom { holes, .. } = &fp2.package.pads[0].shape else {
        panic!()
    };
    assert_eq!(holes.len(), 1, "the ring keeps its open centre");
    // Writing a holed custom pad keeps the hole (keyhole contour) and a dot anchor.
    let out = sexpr::write(&footprint::write(&fp2.package, None));
    assert!(
        out.contains("(anchor circle)") && out.contains("(size 0.01 0.01)"),
        "{out}"
    );
    let back = footprint::read(&sexpr::parse(&out).unwrap(), false);
    let PadShape::Custom { holes: h2, .. } = &back.package.pads[0].shape else {
        panic!()
    };
    assert_eq!(h2.len(), 1);
}

/// Text, graphics, dimensions and fiducials: exported, re-imported and edited in place.
#[test]
fn board_text_graphics_dimensions_and_fiducials_map_both_ways() {
    use runmat_circuit::library::Graphic;
    let p = runmat_circuit::fixture::LargeParams {
        channels: 1,
        copper_layers: 2,
        route_every: 0,
        pours: false,
    };
    let (_, _, _, mut board) = runmat_circuit::fixture::large(&p);
    let id = board.alloc();
    board.texts.insert(
        id,
        BoardText {
            id,
            layer: Layer::Silk(Side::Top),
            at: Point::new(mm(5.0), mm(6.0)),
            text: "REV A".into(),
            size: mm(1.2),
            thickness: mm(0.18),
            angle: 90.0,
            mirrored: false,
        },
    );
    let id = board.alloc();
    board.texts.insert(
        id,
        BoardText {
            id,
            layer: Layer::Silk(Side::Bottom),
            at: Point::new(mm(5.0), mm(9.0)),
            text: "BOTTOM".into(),
            size: mm(1.0),
            thickness: mm(0.15),
            angle: 0.0,
            mirrored: true,
        },
    );
    let id = board.alloc();
    board.graphics.insert(
        id,
        BoardGraphic {
            id,
            layer: Layer::Fab(Side::Top),
            graphic: Graphic::Rect {
                a: Point::new(0, 0),
                b: Point::new(mm(3.0), mm(2.0)),
                width: mm(0.1),
                fill: false,
            },
        },
    );
    let id = board.alloc();
    board.graphics.insert(
        id,
        BoardGraphic {
            id,
            layer: Layer::User(2),
            graphic: Graphic::Line {
                points: vec![Point::new(0, 0), Point::new(mm(4.0), 0)],
                width: mm(0.2),
            },
        },
    );
    let id = board.alloc();
    board.dimensions.insert(
        id,
        Dimension {
            id,
            layer: Layer::User(0),
            a: Point::new(0, 0),
            b: Point::new(mm(20.0), 0),
            offset: mm(3.0),
            precision: 3,
        },
    );
    let id = board.alloc();
    board.fiducials.insert(
        id,
        Fiducial {
            id,
            name: "FID1".into(),
            at: Point::new(mm(2.0), mm(2.0)),
            side: Side::Top,
            diameter: mm(1.0),
            opening: mm(2.0),
            locked: true,
        },
    );

    let text = pcb::write_board(&board);
    assert!(
        text.contains("(gr_text \"REV A\""),
        "{}",
        &text[..400.min(text.len())]
    );
    assert!(
        text.contains("(gr_rect") && text.contains("\"User.2\"") && text.contains("(dimension")
    );
    assert!(text.contains("Fiducial:Fiducial_1mm_Mask2mm"));
    let back = pcb::read(&text).unwrap();
    let b = &back.board;
    let texts: Vec<_> = b
        .texts
        .values()
        .map(|t| {
            (
                t.layer,
                t.at,
                t.text.clone(),
                t.size,
                t.thickness,
                t.angle,
                t.mirrored,
            )
        })
        .collect();
    let want: Vec<_> = board
        .texts
        .values()
        .map(|t| {
            (
                t.layer,
                t.at,
                t.text.clone(),
                t.size,
                t.thickness,
                t.angle,
                t.mirrored,
            )
        })
        .collect();
    assert_eq!(texts, want);
    let graphics: Vec<_> = b
        .graphics
        .values()
        .map(|g| (g.layer, g.graphic.clone()))
        .collect();
    assert_eq!(
        graphics,
        board
            .graphics
            .values()
            .map(|g| (g.layer, g.graphic.clone()))
            .collect::<Vec<_>>()
    );
    let dims: Vec<_> = b
        .dimensions
        .values()
        .map(|d| (d.layer, d.a, d.b, d.offset, d.precision))
        .collect();
    assert_eq!(
        dims,
        board
            .dimensions
            .values()
            .map(|d| (d.layer, d.a, d.b, d.offset, d.precision))
            .collect::<Vec<_>>()
    );
    let fids: Vec<_> = b
        .fiducials
        .values()
        .map(|f| {
            (
                f.name.clone(),
                f.at,
                f.side,
                f.diameter,
                f.opening,
                f.locked,
            )
        })
        .collect();
    assert_eq!(
        fids,
        board
            .fiducials
            .values()
            .map(|f| (
                f.name.clone(),
                f.at,
                f.side,
                f.diameter,
                f.opening,
                f.locked
            ))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        b.footprints.len(),
        board.footprints.len(),
        "a fiducial is not a part"
    );
    assert_eq!(back.write(), text, "unchanged: identical");

    // An edit lands on the original node and keeps the tokens RunMat doesn't model.
    let marked = text.replacen(
        "(gr_text \"REV A\"",
        "(gr_text \"REV A\" (custom_token kept)",
        1,
    );
    let mut doc = pcb::read(&marked).unwrap();
    let tid = *doc
        .board
        .texts
        .iter()
        .find(|(_, t)| t.text == "REV A")
        .unwrap()
        .0;
    doc.board.texts.get_mut(&tid).unwrap().at = Point::new(mm(7.0), mm(6.0));
    let fid = *doc.board.fiducials.keys().next().unwrap();
    doc.board.fiducials.get_mut(&fid).unwrap().at = Point::new(mm(3.0), mm(3.0));
    let did = *doc.board.dimensions.keys().next().unwrap();
    doc.board.dimensions.get_mut(&did).unwrap().precision = 1;
    let out = doc.write();
    assert!(out.contains("(custom_token kept)"));
    let again = pcb::read(&out).unwrap();
    assert!(again
        .board
        .texts
        .values()
        .any(|t| t.text == "REV A" && t.at == Point::new(mm(7.0), mm(6.0))));
    assert_eq!(
        again.board.fiducials.values().next().unwrap().at,
        Point::new(mm(3.0), mm(3.0))
    );
    assert_eq!(again.board.dimensions.values().next().unwrap().precision, 1);
    // A deleted text is dropped.
    let mut doc = pcb::read(&text).unwrap();
    doc.board.texts.retain(|_, t| t.text != "BOTTOM");
    assert!(!doc.write().contains("BOTTOM"));
}

/// PCB-03: an imported project keeps its net classes and rules. The project file's classes
/// become circuit net classes and layout `net_class` parameters, each net takes its class by
/// the file's assignments and patterns, the Default class sets the board's limits, and the
/// custom rules file's rules become layout rules (its one rule on an area condition is
/// reported as not carried).
#[test]
fn an_imported_project_keeps_its_net_classes_and_rules() {
    use runmat_circuit::runtime::import_kicad;
    let files: BTreeMap<String, String> = [
        ("root.kicad_sch", ROOT),
        ("power.kicad_sch", CHILD),
        ("root.kicad_pcb", BOARD),
        ("root.kicad_pro", include_str!("kicad/root.kicad_pro")),
        ("root.kicad_dru", RULES),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    let out = import_kicad(&files, Some("root.kicad_sch"), "boards/io").unwrap();
    let r = &out.report;
    assert_eq!(r.carried["net classes"], 3, "{:?}", r.lines());
    assert_eq!(r.carried["design rules"], 4, "{:?}", r.lines());
    assert_eq!(r.carried["nets with a class"], 4, "{:?}", r.lines());
    assert!(r.not_carried.keys().any(|k| k.starts_with("design rule exotic")), "{:?}", r.lines());
    assert!(r.not_carried.keys().any(|k| k == "net class Default: its diff_gap"), "{:?}", r.lines());

    let layout_text = &out.files["boards/io.circuit.layout"];
    let board = runmat_circuit::syntax::board::parse_board("io.circuit.layout", layout_text).board.unwrap();
    assert_eq!(board.limits.min_clearance, mm(0.2));
    assert_eq!(board.limits.min_track, mm(0.25));
    assert_eq!(board.limits.min_via_diameter, mm(0.8));
    let class_of = |n: &str| board.net(n).and_then(|id| board.nets[id.0 as usize].class.clone());
    assert_eq!(class_of("SIG").as_deref(), Some("HV"));
    assert_eq!(class_of("VCC").as_deref(), Some("power"));
    assert_eq!(class_of("USB_P").as_deref(), Some("usb"));
    assert_eq!(class_of("GND"), None);
    let hv = &board.net_class_rules["HV"].decls;
    assert!(hv.contains(&(props::CLEARANCE.to_string(), RuleValue::Len(mm(1.5)))), "{hv:?}");
    let usb = &board.net_class_rules["usb"].decls;
    assert!(usb.contains(&(props::DIFF_GAP.to_string(), RuleValue::Len(mm(0.15)))), "{usb:?}");
    let ids: Vec<&str> = board.rules.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, ["HV clearance", "power width", "usb pairs", "via ring"]);

    // The circuit declares the classes and puts its nets in them.
    let circuit = &out.files["boards/io.circuit"];
    for class in ["HV", "power", "usb"] {
        assert!(circuit.contains(&format!("net_class {class}")), "{circuit}");
    }
    for line in ["net VCC = { class = power, global = true }", "net SIG = { class = HV }", "net USB_P = { class = usb }", "net USB_N = { class = usb }"] {
        assert!(circuit.contains(line), "{line}\n{circuit}");
    }
    assert!(circuit.contains("net_class HV = { clearance = 0.0015 m, track_width = 0.0005 m }"), "{circuit}");
    // It reads back: nothing but the two parts the fixture's board leaves off (reported above
    // as footprints not carried, so they have no package) is in error.
    let parsed = runmat_circuit::syntax::parse_circuit("io.circuit", circuit);
    let errors: Vec<_> = parsed.diagnostics.iter().map(|d| d.message.as_str()).filter(|m| !m.ends_with("needs its `package`")).collect();
    assert!(errors.is_empty(), "{errors:#?}");
    // The design check (elaboration and ERC) finds no error beyond those two parts.
    let fs: runmat_circuit::runtime::Files = out.files.iter().map(|(k, v)| (format!("/p/{k}"), v.clone())).collect();
    let errors: Vec<_> = import_check::design_errors(&fs, "/p/boards/io.circuit", Some("/p/boards/io.circuit.layout"))
        .into_iter()
        .filter(|m| !m.ends_with("needs its `package`"))
        .collect();
    assert!(errors.is_empty(), "{errors:#?}");
}

/// A board alone imports with an empty circuit stub, so its layout passes the design check;
/// the missing schematic is reported.
#[test]
fn a_board_alone_imports_to_a_layout_that_compiles() {
    use runmat_circuit::runtime::import_kicad;
    let files: BTreeMap<String, String> = [("root.kicad_pcb".to_string(), BOARD.to_string())].into_iter().collect();
    let out = import_kicad(&files, None, "boards/io").unwrap();
    assert!(out.report.not_carried.keys().any(|k| k.starts_with("the schematic")), "{:?}", out.report.lines());
    let fs: runmat_circuit::runtime::Files = out.files.iter().map(|(k, v)| (format!("/p/{k}"), v.clone())).collect();
    let errors = import_check::design_errors(&fs, "/p/boards/io.circuit", Some("/p/boards/io.circuit.layout"));
    assert!(errors.is_empty(), "{errors:#?}");
}
