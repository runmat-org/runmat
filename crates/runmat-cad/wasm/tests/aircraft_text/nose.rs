//! The nose and the cockpit glazing. The nose is a drooped shell, 4 mm thick, lofted through
//! elliptical sections: its tip sits 420 mm below the fuselage axis, its upper line climbs
//! through the windshield at about 33° and its lower line falls to the belly, all three
//! tangent to the barrel at `NOSE`. The glazing is six panes in openings cut through the
//! shell: two front panes either side of a centre post, then a sliding side window and a
//! smaller aft window each side, with skin-colored posts between them. Each pane sits 1 mm
//! inside the skin's outer surface and is 15 to 20 mm thick, so no pane face lies on a skin
//! face.
//!
//! Glazing layout, from the pilots' design eye (x 2350, y ±540, z 800, 1.2 m above the cockpit
//! floor at z −400): the front panes' sill lies about 17° below the horizon and their brow about
//! 15° above it; the posts between panes are 110 to 140 mm wide on the skin.

use super::text::{mm, outline, part, pt, use_};
use super::{NOSE, RADIUS};

/// Where the nose tip sits below the fuselage axis.
const TIP_Z: f64 = -420.0;
/// The nose skin's thickness, measured across each section.
const SKIN: f64 = 4.0;
/// Loft stations, aft of the tip: close together at the tip, where the sections grow fastest.
const STATIONS: [f64; 16] = [
    2.0, 10.0, 30.0, 70.0, 140.0, 250.0, 400.0, 600.0, 900.0, 1_300.0, 1_800.0, 2_400.0, 3_000.0,
    3_700.0, 4_400.0, NOSE,
];
/// Points on each section's closed spline.
const SECTION_POINTS: usize = 24;

/// `(1 - (1 - u)^p)^(1/p)`: 0 at the tip with a vertical tangent, 1 at `u = 1` with a level one.
fn superellipse(u: f64, p: f64) -> f64 {
    let u = u.clamp(0.0, 1.0);
    (1.0 - (1.0 - u).powf(p)).max(0.0).powf(1.0 / p)
}

/// The nose section `x` aft of the tip: the centre's height, the half-height and the half-width.
pub fn section(x: f64) -> (f64, f64, f64) {
    let u = x / NOSE;
    let top = TIP_Z + (RADIUS - TIP_Z) * superellipse(u, 1.4);
    let bottom = TIP_Z - (RADIUS + TIP_Z) * superellipse(u, 1.6);
    let half_width = RADIUS * superellipse(u, 1.8);
    ((top + bottom) / 2.0, (top - bottom) / 2.0, half_width)
}

/// The nose's half-width at height `z`, `x` aft of the tip, `inset` inside its outer surface
/// (0 above or below the section).
pub fn half_width(x: f64, z: f64, inset: f64) -> f64 {
    let (zc, b, a) = section(x);
    let q = 1.0 - ((z - zc) / (b - inset)).powi(2);
    if q > 0.0 { (a - inset) * q.sqrt() } else { 0.0 }
}

/// The section `x` aft of the tip, shrunk by `inset` all round, as points on the yz plane
/// (u = y, v = z), starting at the top.
fn section_points(x: f64, inset: f64) -> Vec<(f64, f64)> {
    let (zc, b, a) = section(x);
    (0..SECTION_POINTS)
        .map(|k| {
            let t = std::f64::consts::TAU * k as f64 / SECTION_POINTS as f64;
            ((a - inset) * t.sin(), zc + (b - inset) * t.cos())
        })
        .collect()
}

/// A loft through closed elliptical sections at `stations`: section `x` is the nose section at
/// `x - setback`, shrunk by `inset`. Its planes and sketches are named after `feature`.
fn loft(feature: &str, stations: &[f64], setback: f64, inset: f64, mode: &str) -> String {
    let mut s = String::new();
    let mut profiles = Vec::new();
    for (k, x) in stations.iter().enumerate() {
        let plane = format!("{feature}_st{k}");
        let sketch = format!("{feature}_s{k}");
        s += &format!("    feature {plane} = plane.offset(base = origin.yz, distance = {})\n", mm(*x));
        let pts: Vec<String> = section_points(x - setback, inset).iter().map(|p| pt(p.0, p.1)).collect();
        s += &format!("    sketch {sketch} on plane.{plane} {{\n        c = spline(points = [{}], closed)\n    }}\n", pts.join(", "));
        profiles.push(format!("sketch.{sketch}"));
    }
    s += &format!("    feature {feature} = loft(profiles = [{}]{mode})\n", profiles.join(", "));
    s
}

/// One pane's opening, the outline it is cut to and how far its prism runs.
struct Pane {
    /// The part's name.
    name: &'static str,
    number: &'static str,
    comment: &'static str,
    /// The outline's plane and its corners in that plane's (u, v).
    plane: Plane,
    corners: Vec<(f64, f64)>,
    /// The loft stations that cover the pane.
    span: (f64, f64),
}

#[derive(Clone, Copy)]
enum Plane {
    /// Front view, (y, z): a prism along x, from x 300 to 2500.
    Front,
    /// Side view, (z, x): a prism along y, to starboard or to port.
    Side { port: bool },
}

/// The front pane to starboard in front view, (y, z); the port pane mirrors it. The centre
/// post between them is 120 mm wide.
const FRONT: [(f64, f64); 4] = [(60.0, 300.0), (60.0, 900.0), (800.0, 810.0), (940.0, 400.0)];
/// The side windows in side view, (x, z): the sliding window, then the smaller aft window,
/// their brows falling aft.
const SIDE: [(f64, f64); 4] = [(2_010.0, 500.0), (2_090.0, 960.0), (2_540.0, 920.0), (2_540.0, 520.0)];
const AFT: [(f64, f64); 4] = [(2_670.0, 530.0), (2_670.0, 910.0), (3_100.0, 840.0), (3_100.0, 580.0)];

fn panes() -> Vec<Pane> {
    let zx = |c: &[(f64, f64)]| c.iter().map(|&(x, z)| (z, x)).collect::<Vec<_>>();
    vec![
        Pane {
            name: "cockpit_window_front",
            number: "FUS-WS-031",
            comment: "The starboard front windshield pane, beside the centre post.",
            plane: Plane::Front,
            corners: FRONT.to_vec(),
            span: (400.0, 2_400.0),
        },
        Pane {
            name: "cockpit_window_front_port",
            number: "FUS-WS-032",
            comment: "The port front windshield pane, beside the centre post.",
            plane: Plane::Front,
            corners: FRONT.iter().map(|&(y, z)| (-y, z)).collect(),
            span: (400.0, 2_400.0),
        },
        Pane {
            name: "cockpit_window_side",
            number: "FUS-WS-033",
            comment: "The starboard sliding window, aft of the windshield.",
            plane: Plane::Side { port: false },
            corners: zx(&SIDE),
            span: (1_300.0, 3_000.0),
        },
        Pane {
            name: "cockpit_window_side_port",
            number: "FUS-WS-034",
            comment: "The port sliding window, aft of the windshield.",
            plane: Plane::Side { port: true },
            corners: zx(&SIDE),
            span: (1_300.0, 3_000.0),
        },
        Pane {
            name: "cockpit_window_aft",
            number: "FUS-WS-035",
            comment: "The starboard aft side window, tapering aft.",
            plane: Plane::Side { port: false },
            corners: zx(&AFT),
            span: (1_800.0, 3_700.0),
        },
        Pane {
            name: "cockpit_window_aft_port",
            number: "FUS-WS-036",
            comment: "The port aft side window, tapering aft.",
            plane: Plane::Side { port: true },
            corners: zx(&AFT),
            span: (1_800.0, 3_700.0),
        },
    ]
}

/// The prism along a pane's outline, cutting or intersecting (`mode`).
fn prism(pane: &Pane, sketch: &str, feature: &str, mode: &str) -> String {
    match pane.plane {
        Plane::Front => format!(
            "    feature {sketch}_at = plane.offset(base = origin.yz, distance = 300 mm)\n{}    feature {feature} = extrude(profile = sketch.{sketch}, distance = 2200 mm, mode = {mode})\n",
            outline(sketch, &format!("plane.{sketch}_at"), &pane.corners, 70.0)
        ),
        Plane::Side { port } => format!(
            "    feature {sketch}_at = plane.offset(base = origin.zx, distance = {})\n{}    feature {feature} = extrude(profile = sketch.{sketch}, distance = 2200 mm, mode = {mode})\n",
            if port { "-2200 mm" } else { "0 mm" },
            outline(sketch, &format!("plane.{sketch}_at"), &pane.corners, 70.0)
        ),
    }
}

fn nose() -> String {
    let mut body = loft("skin", &STATIONS, 0.0, 0.0, "");
    // The inner surface: each section 8 mm further aft, shrunk by the skin, so the wall stays
    // at least the skin's thickness where the nose turns towards the tip.
    body += &loft("hollow", &STATIONS[1..], 8.0, SKIN, ", mode = cut");
    for (k, pane) in panes().iter().enumerate() {
        body += &prism(pane, &format!("o{k}"), &format!("opening{k}"), "cut");
    }
    part(
        "nose",
        "FUS-NS-010",
        "composite",
        "The nose: a drooped shell lofted through elliptical sections, tangent to the barrel, with openings for the cockpit glazing.",
        &body,
        "skin",
    )
}

fn pane(p: &Pane) -> String {
    let stations: Vec<f64> = STATIONS.iter().copied().filter(|x| *x >= p.span.0 && *x <= p.span.1).collect();
    // 1 mm inside the skin's outer surface; the inner face 10 mm further aft and 18 mm in.
    let mut body = loft("glass", &stations, 0.0, 1.0, "");
    body += &loft("inside", &stations, 10.0, 18.0, ", mode = cut");
    body += &prism(p, "o", "pane", "intersect");
    // The intersect trims the glass body in place.
    part(p.name, p.number, "glass", p.comment, &body, "glass")
}

pub fn parts() -> Vec<(String, String)> {
    let mut out = vec![("nose.part".to_string(), nose())];
    for p in panes() {
        out.push((format!("{}.part", p.name), pane(&p)));
    }
    out
}

/// The nose's occurrences in `fuselage.assembly`: the shell and its six panes.
pub fn uses() -> String {
    let mut s = use_("part", "nose", "nose", &[]);
    for p in panes() {
        let alias = p.name.trim_start_matches("cockpit_");
        s += &use_("part", p.name, alias, &[]);
    }
    s
}
