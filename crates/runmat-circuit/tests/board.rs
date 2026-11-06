//! Physical design: zone fill, DRC (full and incremental), routing primitives.

use runmat_circuit::board_rules::{props, BItem, BoardGraph, BoardRules, RuleSpec, RuleValue};
use runmat_circuit::drc::Drc;
use runmat_circuit::fixture::{self, standard_pool, LargeParams};
use runmat_circuit::geom::poly::{rect_ring, Polygon};
use runmat_circuit::geom::{mm, Point, Rect};
use runmat_circuit::layout::*;
use runmat_circuit::library::footprint;
use runmat_circuit::route;
use runmat_circuit::zone::{fill_zones, rect_outline};
use runmat_rules::{SourceKey, SourceOrder};

fn rules(board: &Board, specs: &[RuleSpec]) -> BoardRules {
    let (mut r, errs) = BoardRules::new(&board.limits, specs);
    assert!(errs.is_empty(), "{errs:?}");
    r.rebuild(&BoardGraph::new(board, &[]));
    r
}

fn small_board() -> Board {
    let outline = rect_outline(Rect::new(Point::ORIGIN, Point::mm(40.0, 30.0)));
    let mut b = Board::new("t", outline, Stackup::standard(4, mm(1.6), 1.0, 0.5));
    b.add_net("GND", Some("power"));
    b.add_net("SIG", None);
    b.add_net("VCC", Some("power"));
    b
}

fn src(o: u32) -> (SourceKey, SourceOrder) {
    (
        SourceKey::new("board.circuit", o, o + 20),
        SourceOrder {
            file_rank: 1,
            offset: o,
        },
    )
}

#[test]
fn zone_fill_clearance_spokes_islands() {
    let mut b = small_board();
    let gnd = b.net("GND");
    let sig = b.net("SIG");
    let r = footprint::from_string("R0603").unwrap();
    let r1 = b.place("R1", r, Point::mm(10.0, 10.0), 0.0, Side::Top);
    b.footprints
        .get_mut(&r1)
        .unwrap()
        .pad_nets
        .insert("1".into(), gnd.unwrap());
    b.footprints
        .get_mut(&r1)
        .unwrap()
        .pad_nets
        .insert("2".into(), sig.unwrap());
    let p2 = b.footprints[&r1].pads(4)[1].center;
    b.add_track(sig, 0, mm(0.25), p2, Point::mm(20.0, 10.0));
    // A closed SIG loop isolates a pocket of pour with no GND copper inside.
    let sq = [
        Point::mm(25.0, 15.0),
        Point::mm(35.0, 15.0),
        Point::mm(35.0, 25.0),
        Point::mm(25.0, 25.0),
    ];
    for i in 0..4 {
        b.add_track(sig, 0, mm(0.3), sq[i], sq[(i + 1) % 4]);
    }
    let z = b.alloc();
    b.zones.insert(
        z,
        Zone::new(
            z,
            "GND",
            gnd,
            vec![0],
            rect_outline(Rect::new(Point::mm(1.0, 1.0), Point::mm(39.0, 29.0))),
        ),
    );
    let (s0, o0) = src(0);
    let r = rules(
        &b,
        &[RuleSpec::pair("zone-clear", "zone", "copper")
            .at(s0, o0)
            .set(props::CLEARANCE, RuleValue::Len(mm(0.3)))],
    );
    let reps = fill_zones(&mut b, &r, &[]);
    assert_eq!(reps.len(), 1);
    let rep = &reps[0];
    assert!(rep.islands_removed >= 1, "{rep:?}");
    let fill = &b.zones[&z].fill[&0];
    assert_eq!(
        fill.contains(Point::mm(30.0, 20.0)),
        Some(false),
        "isolated pocket removed"
    );
    assert_eq!(fill.contains(Point::mm(5.0, 25.0)), Some(true));
    // Clearance around the SIG pad: nothing within 0.3 mm.
    let pad = &b.footprints[&r1].pads(4)[1];
    let bb = pad.prim.bbox();
    assert_eq!(
        fill.contains(Point::new(bb.max.x + mm(0.25), pad.center.y)),
        Some(false)
    );
    assert_eq!(
        fill.contains(Point::new(bb.max.x + mm(0.6), pad.center.y + mm(1.5))),
        Some(true)
    );
    // The GND pad is joined by thermal spokes.
    let (reached, total) = rep.spokes[&(r1, 0)];
    assert_eq!(total, 4);
    assert!(reached >= 3, "{reached}");
    // The fill is DRC-clean against other nets and the edge.
    let drc = Drc::run(&b, &r, &[], &reps);
    let bad: Vec<_> = drc
        .findings()
        .into_iter()
        .filter(|f| f.items.contains(&BItem::Zone(z)) && f.code != "min-spokes")
        .collect();
    assert!(bad.is_empty(), "{bad:#?}");
    // A second pour of another net with lower priority keeps clear of the first.
    let vcc = b.net("VCC");
    let z2 = b.alloc();
    let mut zz = Zone::new(
        z2,
        "VCC",
        vcc,
        vec![0],
        rect_outline(Rect::new(Point::mm(15.0, 1.0), Point::mm(39.0, 12.0))),
    );
    zz.priority = -1;
    b.zones.insert(z2, zz);
    let r = rules(&b, &[]);
    let reps = fill_zones(&mut b, &r, &[]);
    assert_eq!(reps.len(), 2);
    let drc = Drc::run(&b, &r, &[], &reps);
    assert_eq!(drc.count("clearance"), 0, "{:#?}", drc.findings());
}

#[test]
fn drc_checks_and_explain() {
    let mut b = small_board();
    let gnd = b.net("GND");
    let sig = b.net("SIG");
    let vcc = b.net("VCC");
    // Clearance: VCC (power) track 0.2 mm from a SIG track; the power rule wants 0.3 mm.
    b.add_track(vcc, 0, mm(0.3), Point::mm(5.0, 5.0), Point::mm(15.0, 5.0));
    let t_sig = b.add_track(sig, 0, mm(0.2), Point::mm(5.0, 5.45), Point::mm(15.0, 5.45));
    // Width: a SIG track thinner than the board minimum.
    let thin = b.add_track(sig, 1, mm(0.1), Point::mm(5.0, 20.0), Point::mm(15.0, 20.0));
    // Annular ring and hole-to-hole: two vias with small rings, 0.1 mm apart.
    let v1 = b.add_via(gnd, Point::mm(20.0, 20.0), mm(0.4), mm(0.3));
    let v2 = b.add_via(gnd, Point::mm(20.0, 20.4), mm(0.6), mm(0.3));
    // Edge clearance: a track running 0.1 mm from the edge.
    let edge_t = b.add_track(sig, 0, mm(0.2), Point::mm(1.0, 0.2), Point::mm(10.0, 0.2));
    // Courtyards overlapping, silk crossing a pad.
    let c = footprint::from_string("C0805").unwrap();
    let ca = b.place("C1", c.clone(), Point::mm(30.0, 10.0), 0.0, Side::Top);
    let cb = b.place("C2", c, Point::mm(31.5, 10.0), 0.0, Side::Top);
    // Unrouted: GND pads of C1 and C2 are not joined (the vias are separate too).
    b.footprints
        .get_mut(&ca)
        .unwrap()
        .pad_nets
        .insert("1".into(), gnd.unwrap());
    b.footprints
        .get_mut(&cb)
        .unwrap()
        .pad_nets
        .insert("1".into(), gnd.unwrap());
    // Keepout: no vias.
    let k = b.alloc();
    b.keepouts.insert(
        k,
        Keepout {
            id: k,
            name: "mounting".into(),
            outline: rect_outline(Rect::new(Point::mm(19.0, 19.0), Point::mm(21.0, 21.0))),
            layers: vec![],
            rules: KeepoutRules {
                vias: true,
                ..Default::default()
            },
        },
    );
    // Blind vias forbidden for power nets.
    let blind = b.alloc();
    b.vias.insert(
        blind,
        Via {
            id: blind,
            net: vcc,
            at: Point::mm(25.0, 25.0),
            diameter: mm(0.6),
            drill: mm(0.3),
            from: 0,
            to: 1,
            micro: false,
            tented: true,
            locked: false,
        },
    );
    let mut specs = Vec::new();
    let (s, o) = src(10);
    specs.push(
        RuleSpec::pair("power-clearance", "copper:net-class(power)", "copper")
            .at(s, o)
            .set(props::CLEARANCE, RuleValue::Len(mm(0.3))),
    );
    let (s, o) = src(40);
    specs.push(
        RuleSpec::single("power-vias", "via:net-class(power)")
            .at(s, o)
            .set(
                props::VIA_TYPES,
                RuleValue::ViaKinds(vec![ViaKind::Through]),
            ),
    );
    let (s, o) = src(70);
    specs.push(
        RuleSpec::single("fine-ring", "via")
            .at(s, o)
            .set(props::ANNULAR_MIN, RuleValue::Len(mm(0.1))),
    );
    let r = rules(&b, &specs);
    let drc = Drc::run(&b, &r, &[], &[]);
    let has = |code: &str, it: BItem| {
        drc.findings()
            .iter()
            .any(|f| f.code == code && f.items.contains(&it))
    };
    assert!(has("clearance", BItem::Track(t_sig)), "{}", drc.digest());
    let cl = drc
        .findings()
        .into_iter()
        .find(|f| f.code == "clearance" && f.items.contains(&BItem::Track(t_sig)))
        .unwrap();
    assert_eq!(cl.rule, "power-clearance");
    assert_eq!(cl.source.start, 10);
    assert!(
        (cl.actual.unwrap() - mm(0.2) as f64).abs() < 1000.0,
        "{cl:?}"
    );
    assert!(has("track-width", BItem::Track(thin)));
    assert!(has("annular-ring", BItem::Via(v1)));
    assert!(!has("annular-ring", BItem::Via(v2)));
    assert!(has("hole-to-hole", BItem::Via(v1)));
    assert!(has("edge-clearance", BItem::Track(edge_t)));
    assert!(has("courtyard-overlap", BItem::Footprint(ca)));
    assert!(has("keepout", BItem::Via(v1)));
    assert!(has("via-type", BItem::Via(blind)));
    assert!(has("unrouted", BItem::Net(gnd.unwrap().0)));
    // Explain the clearance between the two tracks: power rule wins over the default.
    let e = r.pair.explain(
        BItem::Track(t_sig),
        BItem::Track(*b.tracks.keys().next().unwrap()),
        props::CLEARANCE,
    );
    assert_eq!(e.effective.unwrap().value, RuleValue::Len(mm(0.3)));
    let e2 = r.single.explain(
        &BoardGraph::new(&b, &[]),
        BItem::Via(blind),
        props::VIA_TYPES,
    );
    assert!(format!("{e2}").contains("power-vias"));
}

#[test]
fn diff_pairs_lengths_and_skew() {
    let mut b = small_board();
    let p = b.add_net("USB_DP", Some("usb"));
    let n = b.add_net("USB_DN", Some("usb"));
    // Coupled for 10 mm at 0.15 mm gap (0.2 mm tracks, centres 0.35 mm apart); then P runs on
    // alone for 6 mm.
    b.add_track(
        Some(p),
        0,
        mm(0.2),
        Point::mm(5.0, 10.0),
        Point::mm(15.0, 10.0),
    );
    b.add_track(
        Some(n),
        0,
        mm(0.2),
        Point::mm(5.0, 10.35),
        Point::mm(15.0, 10.35),
    );
    b.add_track(
        Some(p),
        0,
        mm(0.2),
        Point::mm(15.0, 10.0),
        Point::mm(21.0, 10.0),
    );
    let specs = vec![RuleSpec::single("usb", "net.usb")
        .set(props::DIFF_GAP, RuleValue::Len(mm(0.2)))
        .set(props::DIFF_UNCOUPLED_MAX, RuleValue::Len(mm(3.0)))
        .set(props::SKEW_MAX, RuleValue::Len(mm(1.0)))
        .set(props::LENGTH_MAX, RuleValue::Len(mm(12.0)))
        .set(props::IMPEDANCE, RuleValue::Num(90.0))];
    let pairs = vec![(p, n)];
    let (mut r, errs) = BoardRules::new(&b.limits, &specs);
    assert!(errs.is_empty());
    r.rebuild(&BoardGraph::new(&b, &pairs));
    let drc = Drc::run(&b, &r, &pairs, &[]);
    let codes: Vec<&str> = drc.findings().iter().map(|f| f.code.as_str()).collect();
    assert!(codes.contains(&"diff-pair-gap"), "{codes:?}");
    assert!(codes.contains(&"diff-pair-uncoupled"), "{codes:?}");
    assert!(codes.contains(&"skew"), "{codes:?}");
    assert!(codes.contains(&"length"), "{codes:?}");
    let unc = drc
        .findings()
        .into_iter()
        .find(|f| f.code == "diff-pair-uncoupled")
        .unwrap();
    assert!(
        (unc.actual.unwrap() - mm(6.0) as f64).abs() < mm(0.01) as f64,
        "{unc:?}"
    );
    assert_eq!(drc.impedance_targets.get("USB_DP"), Some(&90.0));
}

#[test]
fn rigid_flex_region_and_bend() {
    let mut b = small_board();
    let sig = b.net("SIG");
    b.stackups.insert("flex".into(), {
        let mut f = Stackup::flex(2, 0.5, mm(0.05));
        f.name = "flex".into();
        f
    });
    b.regions.push(StackRegion {
        name: "tail".into(),
        outline: rect_outline(Rect::new(Point::mm(30.0, 0.0), Point::mm(40.0, 30.0))),
        stackup: "flex".into(),
        bends: vec![Bend {
            a: Point::mm(35.0, 0.0),
            b: Point::mm(35.0, 30.0),
            radius: mm(1.0),
            angle: 90.0,
            keepout: mm(1.0),
        }],
    });
    // An inner-layer track in the flex tail (the flex stack has only F.Cu and B.Cu).
    let t = b.add_track(sig, 1, mm(0.2), Point::mm(32.0, 5.0), Point::mm(33.0, 5.0));
    let v = b.add_via(sig, Point::mm(35.2, 20.0), mm(0.6), mm(0.3));
    let r = rules(&b, &[]);
    let drc = Drc::run(&b, &r, &[], &[]);
    let has = |code: &str, it: BItem| {
        drc.findings()
            .iter()
            .any(|f| f.code == code && f.items.contains(&it))
    };
    assert!(
        has("layer-not-in-region", BItem::Track(t)),
        "{}",
        drc.digest()
    );
    assert!(has("via-in-bend", BItem::Via(v)));
    assert!(has("layer-not-in-region", BItem::Via(v)));
}

#[test]
fn incremental_drc_equals_full() {
    let pool = standard_pool();
    let p = LargeParams {
        channels: 24,
        copper_layers: 6,
        route_every: 1,
        pours: false,
    };
    let d = fixture::large_design(&p);
    let nl = runmat_circuit::elab::elaborate(&d, &pool, &Default::default(), None);
    let mut b = fixture::large_board(&pool, &nl, &p);
    let specs = vec![RuleSpec::pair("power", "copper:net-class(power)", "copper")
        .set(props::CLEARANCE, RuleValue::Len(mm(0.25)))];
    let mut r = rules(&b, &specs);
    let mut drc = Drc::run(&b, &r, &[], &[]);
    let mut rng = 0x2545F4914F6CDD1Du64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for step in 0..60 {
        let tracks: Vec<ItemId> = b.tracks.keys().copied().collect();
        let vias: Vec<ItemId> = b.vias.keys().copied().collect();
        let fps: Vec<ItemId> = b.footprints.keys().copied().collect();
        let (changed, removed): (Vec<BItem>, Vec<BItem>) = match next() % 5 {
            0 => {
                let t = tracks[(next() as usize) % tracks.len()];
                let tr = b.tracks.get_mut(&t).unwrap();
                if let TrackGeom::Segment { a, b: e } = &mut tr.geom {
                    let dx = (next() % 2000) as i64 * 1000 - 1_000_000;
                    *a = Point::new(a.x + dx, a.y);
                    *e = Point::new(e.x + dx, e.y + 300_000);
                }
                (vec![BItem::Track(t)], vec![])
            }
            1 => {
                let v = vias[(next() as usize) % vias.len()];
                b.vias.remove(&v);
                (vec![], vec![BItem::Via(v)])
            }
            2 => {
                let f = fps[(next() as usize) % fps.len()];
                let fp = b.footprints.get_mut(&f).unwrap();
                fp.at = fp.at + Point::new(mm(0.7), mm(-0.4));
                fp.rotation = (fp.rotation + 90.0) % 360.0;
                (vec![BItem::Footprint(f)], vec![])
            }
            3 => {
                let t = tracks[(next() as usize) % tracks.len()];
                let tr = b.tracks[&t].clone();
                let id = b.add_track(
                    tr.net,
                    tr.layer,
                    mm(0.1),
                    tr.endpoints().0 + Point::mm(0.1, 0.1),
                    tr.endpoints().1 + Point::mm(0.3, 0.0),
                );
                (vec![BItem::Track(id)], vec![])
            }
            _ => {
                let t = tracks[(next() as usize) % tracks.len()];
                b.tracks.remove(&t);
                (vec![], vec![BItem::Track(t)])
            }
        };
        let g = BoardGraph::new(&b, &[]);
        let mut all_changed = changed.clone();
        for c in &changed {
            if let BItem::Footprint(f) = c {
                for k in 0..b.footprints[f].package.pads.len() {
                    all_changed.push(BItem::Pad(*f, k as u32));
                }
            }
        }
        r.single.items_changed(&g, &all_changed, &removed);
        r.pair.items_changed(&g, &all_changed, &removed);
        drc.update(&b, &r, &changed, &removed, None);
        let fresh = Drc::run(&b, &rules(&b, &specs), &[], &[]);
        assert_eq!(drc.digest(), fresh.digest(), "step {step}");
    }
    assert!(
        drc.count("clearance") > 0,
        "edits should have produced violations"
    );
}

#[test]
fn ratsnest_walkaround_insert_and_length() {
    let mut b = small_board();
    let sig = b.net("SIG");
    let gnd = b.net("GND");
    let r0603 = footprint::from_string("R0603").unwrap();
    let a = b.place("R1", r0603.clone(), Point::mm(5.0, 15.0), 0.0, Side::Top);
    let c = b.place("R2", r0603.clone(), Point::mm(35.0, 15.0), 0.0, Side::Top);
    let e = b.place("R3", r0603, Point::mm(20.0, 25.0), 0.0, Side::Top);
    for f in [a, c, e] {
        b.footprints
            .get_mut(&f)
            .unwrap()
            .pad_nets
            .insert("2".into(), sig.unwrap());
    }
    // A GND wall between R1 and R2.
    b.add_track(gnd, 0, mm(0.5), Point::mm(20.0, 5.0), Point::mm(20.0, 20.0));
    let rn = route::ratsnest(&b);
    let sig_wires: Vec<_> = rn.iter().filter(|w| Some(w.net) == sig).collect();
    assert_eq!(sig_wires.len(), 2, "three islands need two airwires");
    let mut r = rules(&b, &[]);
    let pa = b.footprints[&a].pads(4)[1].center;
    let pc = b.footprints[&c].pads(4)[1].center;
    let path = route::walkaround(&b, &r, sig, 0, mm(0.2), pa, pc).expect("a path around the wall");
    assert!(path.len() > 2, "{path:?}");
    let mut drc = Drc::run(&b, &r, &[], &[]);
    for w in path.windows(2) {
        route::insert_track(&mut b, &mut r, &mut drc, sig, 0, mm(0.2), w[0], w[1])
            .expect("walkaround path is DRC-clean");
    }
    // Straight through the wall is refused and rolled back.
    let n_tracks = b.tracks.len();
    let err = route::insert_track(
        &mut b,
        &mut r,
        &mut drc,
        sig,
        0,
        mm(0.2),
        Point::mm(18.0, 12.0),
        Point::mm(22.0, 12.0),
    )
    .unwrap_err();
    assert!(err.iter().any(|f| f.code == "clearance"));
    assert_eq!(b.tracks.len(), n_tracks);
    let fresh = Drc::run(&b, &rules(&b, &[]), &[], &[]);
    assert_eq!(drc.digest(), fresh.digest());
    // Tuning length R1.2 → R2.2 is at least the straight distance.
    let l = route::trace_length(&b, ("R1", "2"), ("R2", "2")).unwrap();
    assert!(l > pa.dist(pc), "{l}");
    // Through a via: R3.2 down to B.Cu and back up.
    let pe = b.footprints[&e].pads(4)[1].center;
    let v1 = pe + Point::mm(0.0, -2.0);
    let v2 = Point::mm(30.0, 23.0);
    route::insert_track(&mut b, &mut r, &mut drc, sig, 0, mm(0.2), pe, v1).unwrap();
    route::insert_via(&mut b, &mut r, &mut drc, sig, v1, mm(0.6), mm(0.3), (0, 3)).unwrap();
    route::insert_track(&mut b, &mut r, &mut drc, sig, 3, mm(0.2), v1, v2).unwrap();
    route::insert_via(&mut b, &mut r, &mut drc, sig, v2, mm(0.6), mm(0.3), (0, 3)).unwrap();
    route::insert_track(&mut b, &mut r, &mut drc, sig, 0, mm(0.2), v2, pc).unwrap();
    let l2 = route::trace_length(&b, ("R3", "2"), ("R2", "2")).unwrap();
    let flat = pe.dist(v1) + v1.dist(v2) + v2.dist(pc);
    let span = 2.0 * b.stackup.span_length(0, 3) as f64;
    assert!((l2 - flat - span).abs() < 2.0, "{l2} vs {}", flat + span);
    assert!(
        route::ratsnest(&b).iter().all(|w| Some(w.net) != sig),
        "SIG fully routed"
    );
    let _ = Polygon::from_ring(rect_ring(Rect::new(Point::ORIGIN, Point::mm(1.0, 1.0))));
}

#[test]
fn interactive_router_plans_drc_clean_tracks() {
    use runmat_circuit::router::{self, Corners, Geometry, WalkMode};
    let mut b = small_board();
    let sig = b.net("SIG");
    let gnd = b.net("GND");
    let r0603 = footprint::from_string("R0603").unwrap();
    let a = b.place("R1", r0603.clone(), Point::mm(5.0, 15.0), 0.0, Side::Top);
    let c = b.place("R2", r0603, Point::mm(35.0, 12.0), 0.0, Side::Top);
    for f in [a, c] {
        b.footprints
            .get_mut(&f)
            .unwrap()
            .pad_nets
            .insert("2".into(), sig.unwrap());
    }
    // A GND wall and a via field between the pads.
    b.add_track(gnd, 0, mm(0.5), Point::mm(20.0, 4.0), Point::mm(20.0, 22.0));
    for i in 0..6 {
        b.add_via(gnd, Point::mm(27.0, 8.0 + i as f64 * 1.2), mm(0.6), mm(0.3));
    }
    let pa = b.footprints[&a].pads(4)[1].center;
    let pc = b.footprints[&c].pads(4)[1].center;
    let specs = vec![RuleSpec::pair("power", "copper:net-class(power)", "copper")
        .set(props::CLEARANCE, RuleValue::Len(mm(0.3)))];
    for geometry in [Geometry::Octilinear, Geometry::Orthogonal] {
        for (mode, corners) in [
            (WalkMode::Shortest, Corners::Sharp),
            (WalkMode::HugBest, Corners::Mitre(mm(0.5))),
            (WalkMode::Hug { clockwise: false }, Corners::Arc(mm(1.0))),
        ] {
            let mut bb = b.clone();
            let mut r = rules(&bb, &specs);
            let plan = router::plan(&bb, &r, sig, 0, mm(0.2), pa, pc, geometry, mode, corners)
                .unwrap_or_else(|| panic!("{geometry:?} {mode:?}"));
            assert!(plan
                .path
                .windows(2)
                .all(|w| router::allowed(w[0], w[1], geometry)));
            assert!(plan.length >= pa.dist(pc));
            let mut drc = Drc::run(&bb, &r, &[], &[]);
            for g in &plan.tracks {
                let id = bb.alloc();
                bb.tracks.insert(
                    id,
                    Track {
                        id,
                        net: sig,
                        layer: 0,
                        width: mm(0.2),
                        geom: g.clone(),
                        locked: false,
                    },
                );
                let gr = BoardGraph::new(&bb, &[]);
                r.single.items_changed(&gr, &[BItem::Track(id)], &[]);
                r.pair.items_changed(&gr, &[BItem::Track(id)], &[]);
                drc.update(&bb, &r, &[BItem::Track(id)], &[], None);
            }
            let errors: Vec<_> = drc
                .findings()
                .into_iter()
                .filter(|f| {
                    f.code != "unrouted" && f.severity == runmat_circuit::elab::Severity::Error
                })
                .collect();
            assert!(errors.is_empty(), "{geometry:?} {mode:?}: {errors:?}");
            assert!(
                route::ratsnest(&bb).iter().all(|w| Some(w.net) != sig),
                "{geometry:?} {mode:?}: SIG connected"
            );
        }
    }
}

/// A land pattern spaces its own pads: the board's clearance applies between them only
/// through a rule written for them, and only pads that touch are a finding. A generated
/// DFN keeps its pins clear of the exposed pad. Each clearance finding names where its limit
/// came from: the layout's `limits` value, or RunMat's default rule set.
#[test]
fn a_footprints_own_pads_are_spaced_by_its_land_pattern() {
    use runmat_circuit::library::footprint::{dfn, join_to_tab, qfn, Density, EP_GAP};
    // The ESC's power FET: 3.3 mm DFN-8, 0.65 mm pitch, 2.4 x 1.8 mm exposed pad; the drain
    // pins 5 to 8 are one copper with the drain tab.
    let mut fet = dfn(8, (3.3, 3.3), 0.65, Some((2.4, 1.8)), Density::B);
    join_to_tab(&mut fet, &["5", "6", "7", "8"]);
    let mut b = small_board();
    let (gnd, vcc) = (b.net("GND").unwrap(), b.net("VCC").unwrap());
    let q = b.place("Q1", fet, Point::mm(10.0, 10.0), 0.0, Side::Top);
    let fp = b.footprints.get_mut(&q).unwrap();
    for pin in ["1", "2", "3", "4"] {
        fp.pad_nets.insert(pin.into(), gnd);
    }
    for pin in ["5", "6", "7", "8", "9"] {
        fp.pad_nets.insert(pin.into(), vcc);
    }
    let pads = b.footprints[&q].pads(4);
    let ep = pads.iter().find(|p| p.thermal_pad).expect("an exposed pad");
    for p in pads.iter().filter(|p| !p.thermal_pad) {
        let gap = p.prim.gap(&ep.prim);
        if ["5", "6", "7", "8"].contains(&p.name.as_str()) {
            assert!(gap < 0.0, "drain pin {} doesn't join the tab", p.name);
        } else {
            assert!(gap >= mm(EP_GAP) as f64 - 1.0, "pin {} is {:.3} mm from the exposed pad", p.name, gap * 1e-6);
        }
    }
    // The drain pins on the tab's net are no finding; the source and gate pins clear it.
    let drc = Drc::run(&b, &rules(&b, &[]), &[], &[]);
    assert!(!drc.findings().iter().any(|f| f.code == "clearance" && f.items.iter().any(|i| matches!(i, BItem::Pad(id, _) if *id == q))), "{}", drc.digest());
    // A 0.4 mm pitch QFN with unconnected pins: its pads are closer than a 0.3 mm board
    // clearance, and no finding.
    b.limits.min_clearance = mm(0.3);
    let u = b.place("U1", qfn(24, 4.0, 0.4, None, Density::C), Point::mm(25.0, 15.0), 0.0, Side::Top);
    let pad_findings = |drc: &Drc| -> Vec<String> {
        drc.findings()
            .into_iter()
            .filter(|f| f.code == "clearance" && f.items.iter().all(|i| matches!(i, BItem::Pad(..))))
            .map(|f| format!("{:?} {}", f.items, f.message))
            .collect()
    };
    let drc = Drc::run(&b, &rules(&b, &[]), &[], &[]);
    assert!(pad_findings(&drc).is_empty(), "{:?}", pad_findings(&drc));
    // A rule written for the QFN's pads holds them to it, and names itself.
    let (s, o) = src(90);
    let spec = RuleSpec::pair("fine-pads", "pad", "pad")
        .at(s, o)
        .set(props::CLEARANCE, RuleValue::Len(mm(0.35)));
    let drc = Drc::run(&b, &rules(&b, &[spec]), &[], &[]);
    let fine: Vec<_> = drc
        .findings()
        .into_iter()
        .filter(|f| f.code == "clearance" && f.items.iter().all(|i| matches!(i, BItem::Pad(id, _) if *id == u)))
        .collect();
    assert!(!fine.is_empty(), "{}", drc.digest());
    assert!(fine.iter().all(|f| f.rule == "fine-pads" && f.source.start == 90));
    // Pads of one footprint that touch on different nets are a finding (a shorted pattern).
    let mut shorted = dfn(8, (3.3, 3.3), 0.65, None, Density::B);
    shorted.pads[0].at = shorted.pads[1].at;
    let s1 = b.place("Q2", shorted, Point::mm(10.0, 22.0), 0.0, Side::Top);
    let fp = b.footprints.get_mut(&s1).unwrap();
    fp.pad_nets.insert("1".into(), gnd);
    fp.pad_nets.insert("2".into(), vcc);
    let drc = Drc::run(&b, &rules(&b, &[]), &[], &[]);
    let touching: Vec<_> = drc
        .findings()
        .into_iter()
        .filter(|f| f.code == "clearance" && f.items == vec![BItem::Pad(s1, 0), BItem::Pad(s1, 1)])
        .collect();
    assert_eq!(touching.len(), 1, "{}", drc.digest());
    assert!(touching[0].message.contains("touch"), "{:?}", touching[0]);
    // Provenance: with no `limits` written the limit is RunMat's default rule set; with one,
    // the finding points at its text.
    let sig = b.net("SIG");
    b.add_track(sig, 0, mm(0.2), Point::mm(2.0, 2.0), Point::mm(8.0, 2.0));
    let t = b.add_track(b.net("GND"), 0, mm(0.2), Point::mm(2.0, 2.3), Point::mm(8.0, 2.3));
    let track_finding = |b: &Board| {
        let drc = Drc::run(b, &rules(b, &[]), &[], &[]);
        drc.findings()
            .into_iter()
            .find(|f| f.code == "clearance" && f.items.contains(&BItem::Track(t)))
            .cloned()
            .expect("the tracks are too close")
    };
    let f = track_finding(&b);
    assert_eq!((f.rule.as_str(), f.source.file.as_str()), ("default.clearance", ""));
    b.limits.sources.insert("clearance".into(), SourceKey::new("/p/t.circuit.layout", 300, 304));
    let f = track_finding(&b);
    assert_eq!(f.rule, "limits.clearance");
    assert_eq!((f.source.start, f.source.end), (300, 304));
}

#[test]
fn a_net_class_clearance_reaches_drc_and_names_its_line() {
    use runmat_circuit::syntax::board::{parse_board, print_board};
    // VCC (class power, 0.2 mm) runs 0.17 mm from DAT (class fine, 0.1 mm), and DAT runs
    // 0.12 mm from SIG (no class). The board's own clearance is 0.15 mm.
    let mut b = small_board();
    let dat = Some(b.add_net("DAT", Some("fine")));
    let (vcc, sig) = (b.net("VCC"), b.net("SIG"));
    let w = mm(0.3);
    b.add_track(vcc, 0, w, Point::mm(5.0, 5.0), Point::mm(15.0, 5.0));
    b.add_track(dat, 0, w, Point::mm(5.0, 5.47), Point::mm(15.0, 5.47));
    b.add_track(sig, 0, w, Point::mm(5.0, 5.89), Point::mm(15.0, 5.89));
    for (class, clearance) in [("power", 0.2), ("fine", 0.1)] {
        b.net_class_rules.insert(
            class.into(),
            NetClassRules {
                decls: vec![(props::CLEARANCE.into(), RuleValue::Len(mm(clearance)))],
                ..Default::default()
            },
        );
    }
    // Read back from its text, the board keeps where each net class is written.
    let text = print_board(&b, &[]);
    let parsed = parse_board("t.circuit.layout", &text);
    assert!(parsed.diagnostics.is_empty(), "{:#?}", parsed.diagnostics);
    let b = parsed.board.expect("the printed board reads back");
    let power = &b.net_class_rules["power"].source;
    assert!(text[power.start as usize..].starts_with("net_class power"), "{power:?}");
    let (mut r, errs) = BoardRules::for_board(&b);
    assert!(errs.is_empty(), "{errs:?}");
    r.rebuild(&BoardGraph::new(&b, &[]));
    let drc = Drc::run(&b, &r, &[], &[]);
    let clearance: Vec<_> = drc.findings().into_iter().filter(|f| f.code == "clearance").collect();
    // The board default allows 0.17 mm; the power class flags it, and the larger class value
    // holds between two classes. The fine class relaxes DAT to SIG to 0.1 mm.
    assert_eq!(clearance.len(), 1, "{clearance:#?}");
    let f = &clearance[0];
    assert_eq!(f.rule, "net_class.power");
    assert_eq!(&f.source, power);
    // Without the net classes, only the 0.12 mm gap is a finding.
    let mut plain = b.clone();
    plain.net_class_rules.clear();
    let drc = Drc::run(&plain, &rules(&plain, &plain.rules), &[], &[]);
    let gaps: Vec<_> = drc.findings().into_iter().filter(|f| f.code == "clearance").collect();
    assert_eq!(gaps.len(), 1, "{gaps:#?}");
    assert!(BoardRules::is_fallback(&gaps[0].rule));
}

#[test]
fn a_gap_equal_to_its_rule_passes_and_one_nanometre_under_fails() {
    // A 7 mil rule (177.8 µm) and two pairs of 0.3 mm tracks: one exactly 7 mil apart, one 1 nm
    // closer.
    let mut b = small_board();
    b.limits.min_clearance = 177_800;
    let (vcc, sig) = (b.net("VCC"), b.net("SIG"));
    let w = mm(0.3);
    for (x0, gap) in [(2_000_000i64, 177_800i64), (20_000_000, 177_799)] {
        let y0 = 5_000_000;
        b.add_track(vcc, 0, w, Point::new(x0, y0), Point::new(x0 + 10_000_000, y0));
        let y1 = y0 + w + gap;
        b.add_track(sig, 0, w, Point::new(x0, y1), Point::new(x0 + 10_000_000, y1));
    }
    let drc = Drc::run(&b, &rules(&b, &[]), &[], &[]);
    let gaps: Vec<_> = drc.findings().into_iter().filter(|f| f.code == "clearance").collect();
    assert_eq!(gaps.len(), 1, "{gaps:#?}");
    assert!(gaps[0].at.x >= 20_000_000, "the 1 nm short pair: {:?}", gaps[0].at);
    // The message never reads the gap equal to its limit.
    assert_eq!(gaps[0].message, "clearance 0.177799 mm < 0.177800 mm");
}

/// A net's impedance target is checked against the stackup's closed forms: on the 4-layer
/// 1.6 mm stack, a 0.15 mm top track is far above 50 Ω (one finding naming the width that hits
/// it), and a track of that width passes.
#[test]
fn an_impedance_target_is_checked_against_the_stackup() {
    use runmat_circuit::impedance::TraceGeometry;
    let mut b = small_board();
    let rf = Some(b.add_net("RF", Some("rf")));
    b.net_class_rules.insert(
        "rf".into(),
        NetClassRules { decls: vec![(props::IMPEDANCE.into(), RuleValue::Num(50.0))], ..Default::default() },
    );
    let fit = TraceGeometry::of(&b.stackup, 0).unwrap().width_for(50.0).expect("a 50 ohm width");
    b.add_track(rf, 0, mm(0.15), Point::mm(5.0, 10.0), Point::mm(15.0, 10.0));
    b.add_track(rf, 0, fit, Point::mm(5.0, 20.0), Point::mm(15.0, 20.0));
    let (mut r, errs) = BoardRules::for_board(&b);
    assert!(errs.is_empty(), "{errs:?}");
    r.rebuild(&BoardGraph::new(&b, &[]));
    let drc = Drc::run(&b, &r, &[], &[]);
    let found: Vec<_> = drc.findings().into_iter().filter(|f| f.code == "impedance").collect();
    assert_eq!(found.len(), 1, "{found:#?}");
    let f = &found[0];
    assert_eq!(f.rule, "net_class.rf");
    assert!(f.actual.unwrap() > 55.0 && f.required == Some(50.0), "{f:?}");
    assert!(f.message.contains("F.Cu at 0.150 mm") && f.message.contains("gives 50 Ω"), "{}", f.message);
    assert_eq!(drc.impedance_targets.get("RF"), Some(&50.0));
}
