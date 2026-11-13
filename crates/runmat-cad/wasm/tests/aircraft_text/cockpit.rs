//! The cockpit behind the windshield: its floor (which runs aft under the forward galley to the
//! first barrel section), the two pilot seats, the instrument panel with six display units
//! under the glareshield, the centre pedestal, the overhead panel, the two control columns and
//! the bulkhead with its door at x 3600.

use super::nose::half_width;
use super::text::{mm, offset, part, polygon, rect, use_, HEADER};
use super::{FLOOR_Z, NOSE};

/// Where the instrument panel's face, the pilots' seats and the bulkhead stand, aft of the tip.
const PANEL_X: f64 = 1_560.0;
const SEAT_X: f64 = 2_420.0;
const PILOT_Y: f64 = 540.0;
const BULKHEAD_X: f64 = 3_600.0;
const PANEL_TOP: f64 = 400.0;

/// The floor from where the nose is 700 mm wide at floor height to the barrel: its outline
/// follows the nose 30 mm inside the skin.
fn floor() -> String {
    let xs: Vec<f64> = (0..=24)
        .map(|k| 700.0 + (NOSE - 700.0) * k as f64 / 24.0)
        .collect();
    let z = FLOOR_Z - 5.0;
    let mut pts: Vec<(f64, f64)> = xs
        .iter()
        .map(|&x| (x, half_width(x, z, 30.0).max(40.0)))
        .collect();
    pts.extend(
        xs.iter()
            .rev()
            .map(|&x| (x, -half_width(x, z, 30.0).max(40.0))),
    );
    part(
        "cockpit_floor",
        "FD-FL-300",
        "composite",
        "The flight deck and forward cabin floor, from the rudder pedals aft to the first barrel section.",
        &format!(
            "    feature datum = plane.offset(base = origin.xy, distance = {})\n{}    feature floor = extrude(profile = sketch.s, distance = 10 mm)\n",
            mm(FLOOR_Z - 10.0),
            polygon("s", "plane.datum", &pts)
        ),
        "floor",
    )
}

/// The bulkhead behind the pilots: the nose section 25 mm inside the skin above the floor,
/// 40 mm thick, with the cockpit door's opening.
fn bulkhead() -> String {
    let z0 = FLOOR_Z;
    let (zc, b, _) = super::nose::section(BULKHEAD_X);
    let top = zc + b - 25.0;
    let mut pts = Vec::new();
    for k in 0..=16 {
        let z = z0 + (top - 5.0 - z0) * k as f64 / 16.0;
        pts.push((half_width(BULKHEAD_X, z, 25.0), z));
    }
    let mut port: Vec<(f64, f64)> = pts.iter().rev().map(|&(y, z)| (-y, z)).collect();
    pts.append(&mut port);
    part(
        "cockpit_bulkhead",
        "FD-BH-310",
        "composite",
        "The flight deck bulkhead with the cockpit door's opening, 800 mm wide and 1900 mm tall.",
        &format!(
            "    feature datum = plane.offset(base = origin.yz, distance = {})\n{}    feature wall = extrude(profile = sketch.s, distance = 40 mm)\n{}    feature doorway = extrude(profile = sketch.d, distance = 40 mm, mode = cut)\n",
            mm(BULKHEAD_X),
            polygon("s", "plane.datum", &pts),
            rect("d", "plane.datum", -400.0, z0, 400.0, z0 + 1_900.0)
        ),
        "wall",
    )
}

/// The instrument panel: a flat panel across the flight deck from the floor to under the
/// glareshield, 120 mm deep, with the pedestal's footwell gaps either side.
fn instrument_panel() -> String {
    let half = half_width(PANEL_X, PANEL_TOP, 60.0).min(half_width(PANEL_X, FLOOR_Z + 200.0, 60.0));
    part(
        "instrument_panel",
        "FD-IP-320",
        "aluminum_6061",
        "The main instrument panel across the flight deck, carrying the six display units.",
        &format!(
            "    feature datum = plane.offset(base = origin.yz, distance = {})\n{}    feature panel = extrude(profile = sketch.s, distance = 120 mm)\n",
            mm(PANEL_X),
            rect("s", "plane.datum", -half, FLOOR_Z + 250.0, half, PANEL_TOP)
        ),
        "panel",
    )
}

/// The glareshield over the panel: its side profile (x, z) rounded at the front edge.
fn glareshield() -> String {
    let half = half_width(PANEL_X - 120.0, PANEL_TOP + 60.0, 50.0);
    let pts: Vec<(f64, f64)> = [
        (PANEL_X - 260.0, PANEL_TOP),
        (PANEL_X + 140.0, PANEL_TOP),
        (PANEL_X + 140.0, PANEL_TOP + 45.0),
        (PANEL_X - 200.0, PANEL_TOP + 80.0),
    ]
    .iter()
    .map(|&(x, z)| (z, x))
    .collect();
    part(
        "glareshield",
        "FD-GS-330",
        "polycarbonate",
        "The glareshield over the instrument panel, carrying the flight control unit.",
        &format!(
            "{}    feature shield = extrude(profile = sketch.s, distance = {}, symmetric)\n",
            super::text::outline("s", "origin.zx", &pts, 15.0),
            mm(2.0 * half)
        ),
        "shield",
    )
}

fn display_unit() -> String {
    part(
        "display_unit",
        "FD-DU-340",
        "glass",
        "A display unit: a 200 mm square screen standing 8 mm proud of the panel face.",
        &format!(
            "{}    feature screen = extrude(profile = sketch.s, distance = 8 mm)\n",
            rect("s", "origin.yz", -100.0, -100.0, 100.0, 100.0)
        ),
        "screen",
    )
}

/// The centre pedestal between the seats: its side profile (x, z), the throttle quadrant's
/// slope at the front.
fn pedestal() -> String {
    let pts: Vec<(f64, f64)> = [
        (PANEL_X + 120.0, FLOOR_Z),
        (SEAT_X + 100.0, FLOOR_Z),
        (SEAT_X + 100.0, FLOOR_Z + 560.0),
        (PANEL_X + 480.0, FLOOR_Z + 640.0),
        (PANEL_X + 120.0, PANEL_TOP - 120.0),
    ]
    .iter()
    .map(|&(x, z)| (z, x))
    .collect();
    part(
        "pedestal",
        "FD-PD-350",
        "aluminum_6061",
        "The centre pedestal: throttle quadrant and radio panels between the pilot seats.",
        &format!(
            "{}    feature body = extrude(profile = sketch.s, distance = 300 mm, symmetric)\n",
            super::text::outline("s", "origin.zx", &pts, 30.0)
        ),
        "body",
    )
}

/// The overhead panel in the flight deck's roof, between the windshield brow and the seats: its
/// side profile (x, z) follows the roof 90 mm inside the skin and stands 110 mm deep, 760 mm
/// wide across the centre line.
fn overhead_panel() -> String {
    let (x0, x1) = (PANEL_X + 520.0, SEAT_X + 260.0);
    let roof = |x: f64| {
        let (zc, b, _) = super::nose::section(x);
        zc + b - 90.0
    };
    let xs: Vec<f64> = (0..=6).map(|k| x0 + (x1 - x0) * k as f64 / 6.0).collect();
    let mut pts: Vec<(f64, f64)> = xs.iter().map(|&x| (roof(x), x)).collect();
    pts.extend(xs.iter().rev().map(|&x| (roof(x) - 110.0, x)));
    part(
        "overhead_panel",
        "FD-OH-370",
        "aluminum_6061",
        "The overhead panel in the flight deck roof: the systems switches over the pilots' heads.",
        &format!(
            "{}    feature body = extrude(profile = sketch.s, distance = 760 mm, symmetric)\n",
            polygon("s", "origin.zx", &pts)
        ),
        "body",
    )
}

/// A control column: a post from the floor, leaning aft, with a yoke across its top.
fn control_column() -> String {
    let post: Vec<(f64, f64)> = [(0.0, 0.0), (60.0, 0.0), (130.0, 640.0), (70.0, 640.0)]
        .iter()
        .map(|&(x, z)| (z, x))
        .collect();
    let yoke = [
        (-170.0, 600.0),
        (170.0, 600.0),
        (170.0, 700.0),
        (130.0, 700.0),
        (130.0, 640.0),
        (-130.0, 640.0),
        (-130.0, 700.0),
        (-170.0, 700.0),
    ];
    part(
        "control_column",
        "FD-CC-360",
        "aluminum_7075",
        "A control column: a post from the floor, leaning aft, with the yoke across its top.",
        &format!(
            "{}    feature post = extrude(profile = sketch.p, distance = 60 mm, symmetric)\n    feature yoke_at = plane.offset(base = origin.yz, distance = 80 mm)\n{}    feature yoke = extrude(profile = sketch.y, distance = 40 mm, mode = add)\n",
            polygon("p", "origin.zx", &post),
            polygon("y", "plane.yoke_at", &yoke)
        ),
        "post",
    )
}

pub fn parts() -> Vec<(String, String)> {
    vec![
        ("cockpit_floor.part".into(), floor()),
        ("cockpit_bulkhead.part".into(), bulkhead()),
        ("instrument_panel.part".into(), instrument_panel()),
        ("glareshield.part".into(), glareshield()),
        ("display_unit.part".into(), display_unit()),
        ("pedestal.part".into(), pedestal()),
        ("control_column.part".into(), control_column()),
        ("overhead_panel.part".into(), overhead_panel()),
    ]
}

/// `cockpit.assembly`, in fuselage coordinates.
pub fn cockpit() -> String {
    let mut s = format!(
        "{HEADER}// The flight deck: floor, bulkhead and door, instrument panel and glareshield, six display\n// units, the centre pedestal, two control columns and the two pilot seats.\nassembly cockpit {{\n"
    );
    s += &use_("part", "cockpit_floor", "floor", &[]);
    s += &use_("part", "cockpit_bulkhead", "bulkhead", &[]);
    s += &use_("part", "instrument_panel", "panel", &[]);
    s += &use_("part", "glareshield", "glareshield", &[]);
    s += &use_("part", "pedestal", "pedestal", &[]);
    s += &use_("part", "overhead_panel", "overhead_panel", &[]);
    // The screens: each pilot's primary flight and navigation displays, and the engine and
    // system displays one above the other in the middle.
    let screens = [
        (-780.0, 230.0),
        (-540.0, 230.0),
        (540.0, 230.0),
        (780.0, 230.0),
        (0.0, 280.0),
        (0.0, 60.0),
    ];
    for (k, (y, z)) in screens.iter().enumerate() {
        s += &use_(
            "part",
            "display_unit",
            &format!("display{}", k + 1),
            &[offset(PANEL_X + 120.0, *y, *z)],
        );
    }
    for (side, y) in [("captain", -PILOT_Y), ("first_officer", PILOT_Y)] {
        s += &use_(
            "part",
            "control_column",
            &format!("column_{side}"),
            &[offset(PANEL_X + 180.0, y, FLOOR_Z)],
        );
        s += &use_(
            "part",
            "seat_frame",
            &format!("seat_frame_{side}"),
            &[offset(SEAT_X, y, FLOOR_Z + 40.0)],
        );
        s += &use_(
            "part",
            "pilot_seat_cushion",
            &format!("seat_{side}"),
            &[offset(SEAT_X, y, FLOOR_Z + 40.0)],
        );
    }
    s += "}\n";
    s
}
