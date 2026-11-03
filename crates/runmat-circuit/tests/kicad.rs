//! KiCad interchange: lossless round trips on hand-written fixtures (hierarchy, buses,
//! power symbols, multi-unit parts, diff pairs, no-connects, stackup, zones, keepouts,
//! blind and micro vias, back-side footprints, rules, unknown tokens).

use runmat_circuit::board_rules::{props, BItem, BoardGraph, BoardRules, RuleValue};
use runmat_circuit::elab::{elaborate, ElabOptions};
use runmat_circuit::geom::{mm, Point};
use runmat_circuit::kicad::{dru, footprint, pcb, sch, sexpr};
use runmat_circuit::layout::*;
use runmat_circuit::library::footprint as gen;
use std::collections::BTreeMap;

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
