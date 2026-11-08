//! High-speed routing: impedance-driven widths per layer (PCB-39), matched-length groups in
//! length and delay with skew and tuning patterns (PCB-40), lengths through series parts
//! (PCB-41).

use runmat_circuit::board_rules::{props, BoardGraph, BoardRules, RuleSpec, RuleValue};
use runmat_circuit::drc::Drc;
use runmat_circuit::geom::{mm, Point, Rect};
use runmat_circuit::high_speed::{self, delay::net_timing, groups, ImpedanceTarget};
use runmat_circuit::impedance::TraceGeometry;
use runmat_circuit::layout::*;
use runmat_circuit::library::footprint;
use runmat_circuit::pair::{self, Pattern};
use runmat_circuit::zone::rect_outline;

fn board() -> Board {
    let outline = rect_outline(Rect::new(Point::ORIGIN, Point::mm(60.0, 40.0)));
    Board::new("hs", outline, Stackup::standard(4, mm(1.6), 1.0, 0.5))
}

fn class(b: &mut Board, name: &str, decls: Vec<(&str, RuleValue)>) {
    b.net_class_rules.insert(
        name.into(),
        NetClassRules { decls: decls.into_iter().map(|(k, v)| (k.to_owned(), v)).collect(), ..Default::default() },
    );
}

fn checked(b: &Board, pairs: &[(NetId, NetId)]) -> (BoardRules, Drc) {
    let (mut r, errs) = BoardRules::for_board(b);
    assert!(errs.is_empty(), "{errs:?}");
    r.rebuild(&BoardGraph::new(b, pairs));
    let drc = Drc::run(b, &r, pairs, &[]);
    (r, drc)
}

/// Each net's length and delay under the board's own rules.
fn timing(b: &Board) -> std::collections::BTreeMap<NetId, high_speed::delay::Timing> {
    let (mut r, _) = BoardRules::for_board(b);
    let g = BoardGraph::new(b, &[]);
    r.rebuild(&g);
    net_timing(b, &r, &g)
}

fn codes<'a>(drc: &'a Drc, code: &str) -> Vec<&'a runmat_circuit::drc::DrcFinding> {
    drc.findings().into_iter().filter(|f| f.code == code).collect()
}

/// PCB-39: a class's 50 Ω target gives each copper layer its own width; the rules hand it to
/// tracks on that layer, and DRC flags a track at the wrong width for its layer.
#[test]
fn an_impedance_class_sets_each_layers_width_and_drc_holds_tracks_to_it() {
    let mut b = board();
    let rf = b.add_net("RF", Some("rf"));
    class(&mut b, "rf", vec![(props::IMPEDANCE, RuleValue::Int(50))]);
    let (r, _) = checked(&b, &[]);
    let n = b.copper_count();
    let widths: Vec<_> = (0..n).map(|l| r.width_on(&b, Some(rf), l, mm(0.2))).collect();
    for (l, w) in widths.iter().enumerate() {
        let z = TraceGeometry::of(&b.stackup, l as u32).unwrap().single_ended(*w as f64 * 1e-9);
        assert!((z - 50.0).abs() < 0.5, "layer {l}: {w} nm gives {z} Ω");
    }
    // Microstrip on the outer layers is wider than stripline inside.
    assert!(widths[0] > widths[1] && widths[0] == widths[3], "{widths:?}");
    // A track at its layer's width passes; the outer width on an inner layer is off target.
    b.add_track(Some(rf), 0, widths[0], Point::mm(5.0, 10.0), Point::mm(25.0, 10.0));
    b.add_track(Some(rf), 1, widths[0], Point::mm(5.0, 20.0), Point::mm(25.0, 20.0));
    let (_, drc) = checked(&b, &[]);
    let found = codes(&drc, "impedance");
    assert_eq!(found.len(), 1, "{found:#?}");
    assert_eq!(found[0].layer, Some(1));
    assert!(found[0].message.contains("In1.Cu"), "{}", found[0].message);
    // A track-width rule on In1.Cu reads the class's solved width.
    let w = codes(&drc, "track-width");
    assert!(w.iter().all(|f| f.layer != Some(0)), "{w:#?}");
}

/// PCB-39: a differential class solves a width and gap per layer; Diff pair routes at them
/// by default, the pair checks clean, and a pair routed at a hand-picked gap is flagged with
/// its differential impedance.
#[test]
fn a_differential_class_solves_width_and_gap_and_drc_checks_the_pair() {
    let mut b = board();
    let (p, n) = (b.add_net("USB_DP", Some("usb")), b.add_net("USB_DN", Some("usb")));
    class(&mut b, "usb", vec![(props::DIFF_IMPEDANCE, RuleValue::Int(90))]);
    let t = ImpedanceTarget::of(&b.net_class_rules["usb"].decls).unwrap();
    let layers = high_speed::controlled_layers(&b.stackup, &t, &b.limits);
    let top = &layers[0];
    assert!((top.z_diff.unwrap() - 90.0).abs() < 1.0, "{top:?}");
    let (r, _) = checked(&b, &[(p, n)]);
    assert_eq!(r.pair_on(&b, p, 0), (top.width, top.gap));
    assert_eq!(r.pair_on(&b, p, 1), (layers[1].width, layers[1].gap));

    // Two parallel runs on F.Cu at the solved width and gap: clean.
    let (w, s) = (top.width.unwrap(), top.gap.unwrap());
    let pitch = w + s;
    b.add_track(Some(p), 0, w, Point::mm(5.0, 10.0), Point::mm(35.0, 10.0));
    b.add_track(Some(n), 0, w, Point::new(mm(5.0), mm(10.0) + pitch), Point::new(mm(35.0), mm(10.0) + pitch));
    let (_, drc) = checked(&b, &[(p, n)]);
    assert!(codes(&drc, "diff-impedance").is_empty(), "{:#?}", codes(&drc, "diff-impedance"));
    assert!(codes(&drc, "diff-pair-gap").is_empty());
    assert_eq!(drc.impedance_targets.get("USB_DP/USB_DN"), Some(&90.0));

    // The same pair at half the width, at the solved gap: its differential impedance is high.
    let ids: Vec<_> = b.tracks.keys().copied().collect();
    for id in ids {
        b.tracks.remove(&id);
    }
    let half = w / 2;
    b.add_track(Some(p), 0, half, Point::mm(5.0, 10.0), Point::mm(35.0, 10.0));
    b.add_track(Some(n), 0, half, Point::new(mm(5.0), mm(10.0) + half + s), Point::new(mm(35.0), mm(10.0) + half + s));
    let (_, drc) = checked(&b, &[(p, n)]);
    let z = codes(&drc, "diff-impedance");
    assert_eq!(z.len(), 1, "{:#?}", drc.findings());
    assert!(z[0].actual.unwrap() > 99.0 && z[0].rule == "net_class.usb", "{:?}", z[0]);
    assert!(codes(&drc, "diff-pair-gap").is_empty());
    assert!(z[0].message.contains("target 90 Ω"), "{}", z[0].message);
}

/// PCB-39: the autorouter draws an impedance class's net at each layer's own width.
#[test]
fn the_autorouter_draws_each_layer_at_its_impedance_width() {
    let mut b = board();
    let rf = b.add_net("RF", Some("rf"));
    class(&mut b, "rf", vec![(props::IMPEDANCE, RuleValue::Num(50.0))]);
    let pkg = footprint::from_string("R0603").unwrap();
    for (r, x) in [("R1", 8.0), ("R2", 50.0)] {
        let id = b.place(r, pkg.clone(), Point::mm(x, 20.0), 0.0, Side::Top);
        b.footprints.get_mut(&id).unwrap().pad_nets.insert("1".into(), rf);
    }
    let (r, _) = checked(&b, &[]);
    let top = r.width_on(&b, Some(rf), 0, 0);
    let mut opts = runmat_circuit::autoroute::RouteOptions::new(vec![rf], vec![0], mm(0.2));
    opts.layer_widths.insert(rf, (0..4).map(|l| r.width_on(&b, Some(rf), l, mm(0.2))).collect());
    let report = runmat_circuit::autoroute::autoroute(&mut b, &opts);
    assert_eq!(report.routed, 1, "{report:?}");
    let widths: std::collections::BTreeSet<_> = b.tracks.values().filter(|t| t.net == Some(rf)).map(|t| t.width).collect();
    // Necked down into the 0603 pads, else at the 50 Ω width.
    assert!(widths.contains(&top), "{widths:?} vs {top}");
}

/// PCB-40/41: a byte lane matched in delay. Members on a stripline layer are slower per mm
/// than on microstrip, so equal copper is not equal delay; DRC reports the members off the
/// target in ps, and tuning with each pattern brings a short member in.
#[test]
fn a_delay_group_compares_members_in_picoseconds_and_tuning_brings_them_in() {
    let mut b = board();
    let nets: Vec<NetId> = (0..3).map(|i| b.add_net(&format!("DQ{i}"), None)).collect();
    b.rules.push(
        RuleSpec::single("byte0", ":net(DQ*)")
            .set(props::MATCH_GROUP, RuleValue::Text("byte0".into()))
            .set(props::MATCH_TOLERANCE, RuleValue::Time(2.0)),
    );
    // DQ0: 40 mm on F.Cu; DQ1: 40 mm on In1.Cu (slower); DQ2: 30 mm on F.Cu.
    let w = mm(0.2);
    b.add_track(Some(nets[0]), 0, w, Point::mm(5.0, 5.0), Point::mm(45.0, 5.0));
    b.add_track(Some(nets[1]), 1, w, Point::mm(5.0, 15.0), Point::mm(45.0, 15.0));
    b.add_track(Some(nets[2]), 0, w, Point::mm(5.0, 30.0), Point::mm(35.0, 30.0));
    let (r, drc) = checked(&b, &[]);
    let g = BoardGraph::new(&b, &[]);
    let tm = timing(&b);
    let gs = groups::groups(&b, &r, &g, &tm);
    assert_eq!(gs.len(), 1);
    let grp = &gs[0];
    assert_eq!(grp.basis, groups::Basis::Delay);
    assert!(!grp.written);
    // The stripline member sets the target.
    let d: Vec<f64> = grp.members.iter().map(|m| m.value).collect();
    assert!(d[1] > d[0] + 20.0 && (grp.target - d[1]).abs() < 1e-9, "{d:?}");
    assert_eq!(grp.members[1].status, groups::Status::Matched);
    assert_eq!(grp.members[0].status, groups::Status::Short);
    let found = codes(&drc, "length-match");
    assert_eq!(found.len(), 2, "{found:#?}");
    assert!(found.iter().all(|f| f.message.contains(" ps") && f.message.contains("longest member's target")));

    // Tune DQ0 and DQ2 to the target delay, each with a different pattern.
    let model = runmat_circuit::high_speed::delay::DelayModel::new(&b.stackup);
    for (net, pattern) in [(nets[0], Pattern::Accordion), (nets[2], Pattern::Sawtooth)] {
        let per_mm = model.ps_per_mm(0, w);
        let now = timing(&b)[&net];
        let target_mm = now.length * 1e-6 + (grp.target - now.delay) / per_mm;
        let t = pair::tune(&mut b, net, target_mm, 1.0 / per_mm, &[], pattern, 0.0).unwrap();
        assert!(t.after > t.before, "{t:?}");
    }
    let (_, drc) = checked(&b, &[]);
    assert!(codes(&drc, "length-match").is_empty(), "{:#?}", codes(&drc, "length-match"));
    assert!(drc.findings().iter().all(|f| f.code != "clearance"), "{:#?}", drc.findings());
}

/// PCB-40: a written length target; a member over it is `Long`, and trombone tuning only adds
/// length.
#[test]
fn a_length_group_reads_its_written_target_and_tolerance() {
    let mut b = board();
    let a = b.add_net("A0", None);
    let c = b.add_net("A1", None);
    b.rules.push(
        RuleSpec::single("addr", ":net(A*)")
            .set(props::MATCH_GROUP, RuleValue::Text("addr".into()))
            .set(props::MATCH_TARGET, RuleValue::Len(mm(30.0)))
            .set(props::MATCH_TOLERANCE, RuleValue::Len(mm(0.5))),
    );
    b.add_track(Some(a), 0, mm(0.2), Point::mm(5.0, 5.0), Point::mm(35.0, 5.0));
    b.add_track(Some(c), 0, mm(0.2), Point::mm(5.0, 20.0), Point::mm(40.0, 20.0));
    let (r, drc) = checked(&b, &[]);
    let gs = groups::groups(&b, &r, &BoardGraph::new(&b, &[]), &timing(&b));
    let m = &gs[0].members;
    assert_eq!((gs[0].basis, gs[0].written, gs[0].target), (groups::Basis::Length, true, 30.0));
    assert_eq!((m[0].status, m[1].status), (groups::Status::Matched, groups::Status::Long));
    let found = codes(&drc, "length-match");
    assert_eq!(found.len(), 1);
    assert!(found[0].message.contains("over the target 30.000 mm by 5.000 mm"), "{}", found[0].message);
    // The marker sits on the member's own copper (its longest track's middle), not at the
    // board origin.
    assert_eq!(found[0].at, Point::mm(22.5, 20.0));
    assert!(pair::tune(&mut b, c, 30.0, 0.5, &[], Pattern::Trombone, 0.0).unwrap_err().contains("longer"));
}

/// PCB-40: in-pair skew as a delay: the same copper length on different layers is skew in ps.
#[test]
fn in_pair_skew_in_picoseconds_counts_each_layers_velocity() {
    let mut b = board();
    let (p, n) = (b.add_net("CLK_P", None), b.add_net("CLK_N", None));
    b.rules.push(RuleSpec::single("clk", ":net(CLK_P)").set(props::SKEW_MAX, RuleValue::Time(1.0)));
    b.add_track(Some(p), 0, mm(0.2), Point::mm(5.0, 5.0), Point::mm(35.0, 5.0));
    b.add_track(Some(n), 1, mm(0.2), Point::mm(5.0, 6.0), Point::mm(35.0, 6.0));
    let (_, drc) = checked(&b, &[(p, n)]);
    let skew = codes(&drc, "skew");
    assert_eq!(skew.len(), 1, "{skew:#?}");
    // 30 mm at about 6.8 − 5.9 ps/mm.
    assert!((15.0..40.0).contains(&skew[0].actual.unwrap()), "{:?}", skew[0]);
    assert_eq!(skew[0].at, Point::mm(20.0, 5.0));
}

/// PCB-41: a signal through a series resistor is one tuned signal: both nets' copper plus the
/// part's delay.
#[test]
fn a_signal_through_a_series_resistor_counts_both_sides_and_the_parts_delay() {
    let mut b = board();
    let dq = b.add_net("DQ0", None);
    let dqr = b.add_net("DQ0_R", None);
    let pkg = footprint::from_string("R0402").unwrap();
    let rs = b.place("R21", pkg.clone(), Point::mm(30.0, 10.0), 0.0, Side::Top);
    {
        let f = b.footprints.get_mut(&rs).unwrap();
        f.pad_nets.insert("1".into(), dq);
        f.pad_nets.insert("2".into(), dqr);
    }
    // A receiver pad on DQ0_R (a second two-pad part with one pad on the net).
    let rx = b.place("U2", pkg, Point::mm(50.0, 10.0), 0.0, Side::Top);
    b.footprints.get_mut(&rx).unwrap().pad_nets.insert("1".into(), dqr);
    b.add_track(Some(dq), 0, mm(0.2), Point::mm(5.0, 10.0), Point::mm(29.5, 10.0));
    b.add_track(Some(dqr), 0, mm(0.2), Point::mm(30.5, 10.0), Point::mm(49.5, 10.0));
    b.rules.push(RuleSpec::single("dq", ":net(DQ0)").set(props::THROUGH_PARTS, RuleValue::Bool(true)));
    b.rules.push(RuleSpec::single("rs", "footprint#R21").set(props::THROUGH_DELAY, RuleValue::Time(8.0)));
    let (r, _) = checked(&b, &[]);
    let g = BoardGraph::new(&b, &[]);
    let t = timing(&b);
    let sig = high_speed::signals::signal_of(&b, &r, &g, dq, &t);
    assert_eq!(sig.nets, vec![dq, dqr]);
    assert_eq!(sig.parts, vec!["R21".to_string()]);
    assert!((sig.timing.length - (t[&dq].length + t[&dqr].length)).abs() < 1.0);
    assert!((sig.timing.delay - (t[&dq].delay + t[&dqr].delay + 8.0)).abs() < 1e-9);
}

/// PCB-42: the board study's `pcb` source carries each controlled line's cross-section, with
/// its stack between the planes and the closed form to compare a field solve with.
#[test]
fn the_pcb_source_carries_each_controlled_lines_cross_section() {
    let mut b = board();
    let rf = b.add_net("RF", Some("rf"));
    class(&mut b, "rf", vec![(props::IMPEDANCE, RuleValue::Int(50))]);
    let (r, _) = checked(&b, &[]);
    let w = r.width_on(&b, Some(rf), 1, 0);
    b.add_track(Some(rf), 1, w, Point::mm(5.0, 10.0), Point::mm(25.0, 10.0));
    let src = runmat_circuit::fea::pcb_source(&b, None, None, &Default::default());
    assert_eq!(src.lines.len(), 1, "{:#?}", src.lines);
    let line = &src.lines[0];
    assert_eq!((line.region.as_str(), line.kind), ("pcb/hs/line/RF/In1.Cu", "stripline"));
    assert_eq!((line.plane_above.as_deref(), line.plane_below.as_deref()), (Some("F.Cu"), Some("In2.Cu")));
    assert_eq!(line.stack.first().unwrap().kind, "copper");
    assert_eq!(line.stack.last().unwrap().kind, "copper");
    assert!((line.z0 - 50.0).abs() < 0.5 && line.gap.is_none() && (line.length - 0.02).abs() < 1e-9, "{line:?}");
}

/// A through via joining F.Cu to In1.Cu: the barrel past In1.Cu down to the bottom copper is a
/// stub. `stub_max` flags it; `backdrill` removes it from the bottom face, down to In1.Cu,
/// leaving `backdrill_stub`; the stub's delay leaves the net with it, its length never counted.
#[test]
fn a_via_stub_past_its_connected_layer_is_checked_and_back_drilled() {
    use runmat_circuit::high_speed::stubs::{via_stubs, Face};
    let mut b = board();
    let sig = b.add_net("SIG", None);
    let at = Point::mm(20.0, 10.0);
    b.add_track(Some(sig), 0, mm(0.2), Point::mm(5.0, 10.0), at);
    b.add_track(Some(sig), 1, mm(0.2), at, Point::mm(40.0, 10.0));
    let v = b.add_via(Some(sig), at, mm(0.6), mm(0.3));
    b.rules.push(RuleSpec::single("sig", ":net(SIG)").set(props::STUB_MAX, RuleValue::Len(mm(0.3))));
    let s = b.stackup.clone();
    // From In1.Cu's centre to the bottom copper's outer face (the board less the bottom mask).
    let expected = s.thickness() - mm(0.02) - s.copper_z(1).unwrap();
    let (r, drc) = checked(&b, &[]);
    let g = BoardGraph::new(&b, &[]);
    let st = via_stubs(&b, &r, &g);
    assert_eq!(st.len(), 1);
    assert_eq!((st[0].via, st[0].connected, st[0].stubs.len()), (v, (0, 1), 1));
    let stub = &st[0].stubs[0];
    assert_eq!((stub.face, stub.drilled, stub.length, stub.open_to_surface), (Face::Bottom, expected, expected, true));
    assert!(stub.drilled > mm(1.0), "{stub:?}");
    let found = codes(&drc, "via-stub");
    assert_eq!(found.len(), 1, "{found:#?}");
    assert_eq!(found[0].at, at);
    assert!(found[0].message.contains("past In1.Cu") && found[0].message.contains("back-drill it"), "{}", found[0].message);
    let before = timing(&b)[&sig];
    // The signal runs 15 + 20 mm of track and the via's F.Cu–In1.Cu span; the stub is no length.
    assert!((before.length - (mm(35.0) + s.span_length(0, 1)) as f64).abs() < 1.0, "{before:?}");

    // Back-drilled from the bottom, stopping 0.2 mm past In1.Cu, 0.2 mm over the drill.
    b.rules.push(RuleSpec::single("bd", ":net(SIG)").set(props::BACKDRILL, RuleValue::Bool(true)));
    let (r, drc) = checked(&b, &[]);
    let g = BoardGraph::new(&b, &[]);
    let st = via_stubs(&b, &r, &g);
    let bd = st[0].stubs[0].backdrill.clone().expect("back-drilled");
    assert_eq!((bd.face, bd.must_not_cut, bd.diameter, bd.residual, bd.depth), (Face::Bottom, 1, mm(0.5), mm(0.2), expected - mm(0.2)));
    assert_eq!(st[0].length, mm(0.2));
    assert!(codes(&drc, "via-stub").is_empty(), "{:#?}", codes(&drc, "via-stub"));
    // The board projection lists the via with its stub, limit and back-drill.
    let rows = runmat_circuit::index::stub_rows(&b, &r, &g);
    assert_eq!(rows.len(), 1);
    assert_eq!((rows[0].net.as_str(), rows[0].connected.as_str(), rows[0].stub_mm, rows[0].max_mm, rows[0].status.as_str()), ("SIG", "F.Cu-In1.Cu", 0.2, Some(0.3), "ok"));
    assert_eq!((rows[0].backdrills[0].face.as_str(), rows[0].backdrills[0].must_not_cut.as_str(), rows[0].backdrills[0].diameter_mm), ("bottom", "In1.Cu", 0.5));
    let after = timing(&b)[&sig];
    assert_eq!(after.length, before.length);
    // About 1.0 mm of stub at ~6.9 ps/mm, half of it seen: ~3.5 ps, down to ~0.7 ps at 0.2 mm.
    assert!((2.4..3.2).contains(&(before.delay - after.delay)), "{before:?} {after:?}");

    // The drill files carry it as a controlled-depth, non-plated drill from the bottom through
    // layers 3 and 4, and the fabrication drawing's notes state its depth and must-not-cut layer.
    let files = runmat_circuit::fab::excellon(&b);
    let bd_file = &files["hs-Backdrill-bottom-3-4.drl"];
    assert!(bd_file.contains("TF.FileFunction,NonPlated,3,4,Blind") && bd_file.contains("BackDrill"), "{bd_file}");
    assert!(bd_file.contains(&format!("depth {:.3} mm from the bottom face", (expected - mm(0.2)) as f64 * 1e-6)) && bd_file.contains("T1C0.500") && bd_file.contains("X20.0000Y10.0000"), "{bd_file}");
    assert!(files.contains_key("hs-PTH-1-4.drl"), "{:?}", files.keys());
    let notes = runmat_circuit::fab::drawing::backdrill_notes(&b);
    assert_eq!(notes, vec![format!(
        "Back-drill 1 hole from the bottom face with a 0.500 mm drill, {:.3} mm deep; must not cut layer 2 (In1.Cu); stub left at most 0.200 mm",
        (expected - mm(0.2)) as f64 * 1e-6
    )]);
}

/// A buried via's stub is inside the board: back-drill can't reach it, and DRC says so.
#[test]
fn a_buried_stub_cannot_be_back_drilled() {
    use runmat_circuit::high_speed::stubs::via_stubs;
    let outline = rect_outline(Rect::new(Point::ORIGIN, Point::mm(60.0, 40.0)));
    let mut b = Board::new("hs6", outline, Stackup::standard(6, mm(1.6), 1.0, 0.5));
    let sig = b.add_net("SIG", None);
    let at = Point::mm(20.0, 10.0);
    b.add_track(Some(sig), 1, mm(0.2), Point::mm(5.0, 10.0), at);
    b.add_track(Some(sig), 2, mm(0.2), at, Point::mm(40.0, 10.0));
    let v = b.add_via(Some(sig), at, mm(0.6), mm(0.3));
    {
        let via = b.vias.get_mut(&v).unwrap();
        (via.from, via.to) = (1, 4);
    }
    b.rules.push(
        RuleSpec::single("sig", ":net(SIG)")
            .set(props::STUB_MAX, RuleValue::Len(mm(0.3)))
            .set(props::BACKDRILL, RuleValue::Bool(true)),
    );
    let (r, drc) = checked(&b, &[]);
    let st = via_stubs(&b, &r, &BoardGraph::new(&b, &[]));
    assert_eq!(st[0].connected, (1, 2));
    assert!(!st[0].stubs[0].open_to_surface && st[0].stubs[0].backdrill.is_none(), "{st:?}");
    let found = codes(&drc, "via-stub");
    assert_eq!(found.len(), 1);
    assert!(found[0].message.contains("inside the board"), "{}", found[0].message);
}

/// A stitching via joining an F.Cu track to a B.Cu pour of its net has no stub: the pour
/// connects it at the bottom.
#[test]
fn a_pour_of_the_net_connects_its_via() {
    use runmat_circuit::geom::poly::Polygon;
    use runmat_circuit::high_speed::stubs::{connected_layers, via_stubs};
    let mut b = board();
    let gnd = b.add_net("GND", None);
    let at = Point::mm(20.0, 10.0);
    b.add_track(Some(gnd), 0, mm(0.3), Point::mm(5.0, 10.0), at);
    let v = b.add_via(Some(gnd), at, mm(0.6), mm(0.3));
    let id = b.alloc();
    let square = vec![Point::mm(10.0, 0.0), Point::mm(30.0, 0.0), Point::mm(30.0, 20.0), Point::mm(10.0, 20.0)];
    b.zones.insert(id, Zone::new(id, "gnd", Some(gnd), vec![3], Polygon { outer: square, holes: vec![] }));
    assert_eq!(connected_layers(&b, &b.vias[&v]), vec![0, 3]);
    let (r, _) = checked(&b, &[]);
    assert!(via_stubs(&b, &r, &BoardGraph::new(&b, &[])).is_empty());
}

fn pour(b: &mut Board, net: NetId, layer: u32, x0: f64, x1: f64, holes: Vec<Vec<Point>>) {
    use runmat_circuit::geom::poly::Polygon;
    let id = b.alloc();
    let outer = vec![Point::mm(x0, 1.0), Point::mm(x1, 1.0), Point::mm(x1, 39.0), Point::mm(x0, 39.0)];
    // A plane with no pads on it here: keep its fill (a board's planes reach their nets' vias).
    let mut z = Zone::new(id, "", Some(net), vec![layer], Polygon { outer, holes });
    z.islands = IslandPolicy::Keep;
    b.zones.insert(id, z);
}

fn filled(b: &mut Board) -> Drc {
    let (mut r, errs) = BoardRules::for_board(b);
    assert!(errs.is_empty(), "{errs:?}");
    r.rebuild(&BoardGraph::new(b, &[]));
    runmat_circuit::zone::fill_zones(b, &r, &[]);
    checked(b, &[]).1
}

/// A 50 Ω line on F.Cu over In1.Cu: crossing from the ground pour to the supply pour is a
/// split; running over a 2 mm hole in the ground pour is a gap; `return_path = false` turns the
/// check off, and a net with no impedance target isn't checked.
#[test]
fn a_controlled_line_over_a_plane_split_or_gap_is_a_return_path_finding() {
    let hole = || vec![Point::mm(24.0, 18.0), Point::mm(24.0, 22.0), Point::mm(26.0, 22.0), Point::mm(26.0, 18.0)];
    let w = mm(0.3);
    let mut b = board();
    let rf = b.add_net("RF", Some("rf"));
    let (gnd, vcc) = (b.add_net("GND", None), b.add_net("VCC", None));
    class(&mut b, "rf", vec![(props::IMPEDANCE, RuleValue::Int(50))]);
    pour(&mut b, gnd, 1, 1.0, 30.0, vec![]);
    pour(&mut b, vcc, 1, 31.0, 59.0, vec![]);
    b.add_track(Some(rf), 0, w, Point::mm(10.0, 20.0), Point::mm(50.0, 20.0));
    let drc = filled(&mut b);
    let found = codes(&drc, "return-path");
    assert_eq!(found.len(), 1, "{found:#?}");
    assert!(found[0].message.contains("crosses a split in its reference plane In1.Cu (`GND` to `VCC`)"), "{}", found[0].message);
    // Where the ground pour ends under the track.
    assert!((found[0].at.x - mm(30.0)).abs() < mm(0.6) && found[0].at.y == mm(20.0), "{:?}", found[0].at);

    // One ground pour with a 2 mm hole under the line: a gap, not a split.
    let mut b = board();
    let rf = b.add_net("RF", Some("rf"));
    let gnd = b.add_net("GND", None);
    class(&mut b, "rf", vec![(props::IMPEDANCE, RuleValue::Int(50))]);
    pour(&mut b, gnd, 1, 1.0, 59.0, vec![hole()]);
    b.add_track(Some(rf), 0, w, Point::mm(10.0, 20.0), Point::mm(50.0, 20.0));
    let drc = filled(&mut b);
    let found = codes(&drc, "return-path");
    assert_eq!(found.len(), 1, "{found:#?}");
    assert!(found[0].message.contains("over a gap in its reference plane In1.Cu"), "{}", found[0].message);
    // The 2 mm hole, give or take a sample step and the fill's edge.
    assert!((mm(1.8) as f64..mm(3.2) as f64).contains(&found[0].actual.unwrap()), "{:?}", found[0].actual);

    b.net_class_rules.get_mut("rf").unwrap().decls.push((props::RETURN_PATH.into(), RuleValue::Bool(false)));
    assert!(codes(&filled(&mut b), "return-path").is_empty());
    let mut b2 = board();
    let gnd = b2.add_net("GND", None);
    let sig = b2.add_net("SIG", None);
    pour(&mut b2, gnd, 1, 1.0, 59.0, vec![hole()]);
    b2.add_track(Some(sig), 0, w, Point::mm(10.0, 20.0), Point::mm(50.0, 20.0));
    assert!(codes(&filled(&mut b2), "return-path").is_empty());
}
