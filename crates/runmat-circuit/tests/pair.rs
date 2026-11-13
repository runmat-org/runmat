//! Coupled routing (runmat_circuit::pair): a differential pair routed as one around an
//! obstacle at its gap, and a net's length tuned with meanders, both passing DRC.

use runmat_circuit::board_rules::{props, BItem, BoardGraph, BoardRules, RuleSpec, RuleValue};
use runmat_circuit::drc::Drc;
use runmat_circuit::geom::{mm, Point, Rect};
use runmat_circuit::layout::*;
use runmat_circuit::library::footprint;
use runmat_circuit::pair::{net_track_length, offset, route_pair, tune_length};
use runmat_circuit::route;
use runmat_circuit::zone::{fill_zones, rect_outline};

/// A 40 × 30 mm board with a USB-style pair between two 0603 footprints turned upright (one
/// pad per net at each end), a GND track across the straight line between them, and a
/// 0.15 mm gap rule on the pair's class.
fn pair_board() -> (Board, NetId, NetId) {
    let outline = rect_outline(Rect::new(Point::ORIGIN, Point::mm(40.0, 30.0)));
    let mut b = Board::new("t", outline, Stackup::standard(4, mm(1.6), 1.0, 0.5));
    let gnd = b.add_net("GND", Some("power"));
    let p = b.add_net("USB_DP", Some("usb"));
    let n = b.add_net("USB_DN", Some("usb"));
    for (r, x) in [("R1", 6.0), ("R2", 34.0)] {
        let id = b.place(
            r,
            footprint::from_string("R0603").unwrap(),
            Point::mm(x, 15.0),
            90.0,
            Side::Top,
        );
        let f = b.footprints.get_mut(&id).unwrap();
        f.pad_nets.insert("1".into(), p);
        f.pad_nets.insert("2".into(), n);
    }
    b.add_track(
        Some(gnd),
        0,
        mm(0.3),
        Point::mm(20.0, 8.0),
        Point::mm(20.0, 22.0),
    );
    b.rules
        .push(RuleSpec::single("usb", "net.usb").set(props::DIFF_GAP, RuleValue::Len(mm(0.15))));
    (b, p, n)
}

fn drc(b: &Board, pairs: &[(NetId, NetId)]) -> Vec<String> {
    let mut b = b.clone();
    let (mut rules, _) = BoardRules::for_board(&b);
    rules.rebuild(&BoardGraph::new(&b, pairs));
    let fills = fill_zones(&mut b, &rules, pairs);
    Drc::run(&b, &rules, pairs, &fills)
        .findings()
        .iter()
        .map(|f| format!("{} {}", f.code, f.message))
        .collect()
}

#[test]
fn an_offset_path_stays_parallel_at_the_distance_through_45_degree_bends() {
    let path = [
        Point::mm(0.0, 0.0),
        Point::mm(10.0, 0.0),
        Point::mm(15.0, 5.0),
        Point::mm(25.0, 5.0),
    ];
    let left = offset(&path, mm(0.5) as f64);
    assert_eq!(left.len(), path.len());
    // Each offset segment is parallel to its own and 0.5 mm from it.
    for i in 0..3 {
        let (a, b) = (path[i], path[i + 1]);
        let (c, d) = (left[i], left[i + 1]);
        let (ux, uy) = ((b.x - a.x) as f64, (b.y - a.y) as f64);
        let (vx, vy) = ((d.x - c.x) as f64, (d.y - c.y) as f64);
        assert!(
            (ux * vy - uy * vx).abs() / (ux.hypot(uy) * vx.hypot(vy)) < 1e-4,
            "segment {i} not parallel"
        );
        let dist = ((c.x - a.x) as f64 * -uy + (c.y - a.y) as f64 * ux) / ux.hypot(uy);
        assert!(
            (dist - 0.5e6).abs() < 2_000.0,
            "segment {i} is {dist} nm off"
        );
    }
}

#[test]
fn a_pair_routes_around_an_obstacle_at_its_gap_and_its_length_tunes() {
    let (mut b, p, n) = pair_board();
    let pairs = [(p, n)];
    let routed = route_pair(&mut b, p, n, 0, mm(0.2), mm(0.15)).unwrap_or_else(|e| panic!("{e}"));
    assert!(!routed.p_tracks.is_empty() && !routed.n_tracks.is_empty());
    // Both nets are joined end to end and the board is DRC clean, gap rule included.
    let open: Vec<_> = route::ratsnest(&b)
        .into_iter()
        .filter(|a| a.net == p || a.net == n)
        .collect();
    assert!(open.is_empty(), "{open:?}");
    let findings = drc(&b, &pairs);
    assert!(findings.is_empty(), "{findings:?}");
    // The pair went around the GND track, so it is longer than the straight 28 mm.
    assert!(
        routed.p_length > 29.0 && routed.n_length > 29.0,
        "{routed:?}"
    );
    assert!(routed.skew() < 1.0, "{routed:?}");
    // Tuning the shorter net to the longer one's length plus 1.5 mm adds meanders that land
    // within tolerance and keep the board clean.
    let (short, long) = if routed.p_length < routed.n_length {
        (p, routed.n_length)
    } else {
        (n, routed.p_length)
    };
    let target = long + 1.5;
    let tuned = tune_length(&mut b, short, target, 0.05, &pairs).unwrap_or_else(|e| panic!("{e}"));
    assert!((tuned.after - target).abs() < 0.05, "{tuned:?}");
    assert!((net_track_length(&b, short) - tuned.after).abs() < 1e-9);
    let findings = drc(&b, &pairs);
    assert!(findings.is_empty(), "{findings:?}");
    let open: Vec<_> = route::ratsnest(&b)
        .into_iter()
        .filter(|a| a.net == short)
        .collect();
    assert!(open.is_empty(), "{open:?}");
    // Meanders only add length.
    let err = tune_length(&mut b, short, 10.0, 0.05, &pairs).unwrap_err();
    assert!(err.contains("longer than"), "{err}");
}

#[test]
fn a_pair_with_no_way_through_is_refused_and_adds_nothing() {
    let (mut b, p, n) = pair_board();
    let gnd = b.net("GND").unwrap();
    // A GND wall from edge to edge.
    b.add_track(
        Some(gnd),
        0,
        mm(0.3),
        Point::mm(20.0, 0.5),
        Point::mm(20.0, 29.5),
    );
    let tracks = b.tracks.len();
    let err = route_pair(&mut b, p, n, 0, mm(0.2), mm(0.15)).unwrap_err();
    assert!(err.contains("no path"), "{err}");
    assert_eq!(b.tracks.len(), tracks);
    let _ = BItem::Edge;
}

/// The pair board with its left end a four-pad connector that carries each net on two pads
/// (as a USB-C receptacle carries D+ and D-), each net's two pads tied by a short track.
fn connector_pair_board() -> (Board, NetId, NetId) {
    let (mut b, p, n) = pair_board();
    let r1 = b.footprint_by_ref("R1").map(|f| f.id).unwrap();
    b.footprints.remove(&r1);
    let id = b.place(
        "J1",
        footprint::from_string("SMDConn4_P1").unwrap(),
        Point::mm(6.0, 15.0),
        90.0,
        Side::Top,
    );
    let f = b.footprints.get_mut(&id).unwrap();
    for (pad, net) in [("1", p), ("2", p), ("3", n), ("4", n)] {
        f.pad_nets.insert(pad.into(), net);
    }
    let at = |b: &Board, name: &str| {
        b.all_pads()
            .into_iter()
            .find(|q| q.footprint == id && q.name == name)
            .unwrap()
            .center
    };
    let (a1, a2, a3, a4) = (at(&b, "1"), at(&b, "2"), at(&b, "3"), at(&b, "4"));
    b.add_track(Some(p), 0, mm(0.2), a1, a2);
    b.add_track(Some(n), 0, mm(0.2), a3, a4);
    (b, p, n)
}

#[test]
fn a_part_that_carries_a_pair_net_on_several_pads_is_one_end_of_the_pair() {
    let (mut b, p, n) = connector_pair_board();
    let routed = route_pair(&mut b, p, n, 0, mm(0.2), mm(0.15)).unwrap_or_else(|e| panic!("{e}"));
    assert!(!routed.p_tracks.is_empty() && !routed.n_tracks.is_empty());
    // The pair joins the connector to R2 and the board stays DRC clean.
    let open: Vec<_> = route::ratsnest(&b)
        .into_iter()
        .filter(|a| a.net == p || a.net == n)
        .collect();
    assert!(open.is_empty(), "{open:?}");
    let findings = drc(&b, &[(p, n)]);
    assert!(findings.is_empty(), "{findings:?}");
}

#[test]
fn a_pair_net_on_three_parts_is_refused_as_not_point_to_point() {
    let (mut b, p, n) = connector_pair_board();
    let id = b.place(
        "R3",
        footprint::from_string("R0603").unwrap(),
        Point::mm(20.0, 26.0),
        0.0,
        Side::Top,
    );
    b.footprints
        .get_mut(&id)
        .unwrap()
        .pad_nets
        .insert("1".into(), p);
    let err = route_pair(&mut b, p, n, 0, mm(0.2), mm(0.15)).unwrap_err();
    assert!(err.contains("USB_DP") && err.contains("3 part(s)"), "{err}");
}
