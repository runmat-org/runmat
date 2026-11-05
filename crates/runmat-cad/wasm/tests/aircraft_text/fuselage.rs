//! The fuselage: an elliptical nose with its windshield, three barrel sections (frames,
//! stringers, skin panels, flush rivet rings and a row of cabin windows each side), an upswept
//! tail cone and a wing-to-body fairing.

use super::text::{loop_sketch, mm, offset, offset_rot, part, polygon, pt, rect, use_, Curve, HEADER};
use super::{FRAME_PITCH, NOSE, RADIUS, SECTION, SKIN_PANELS, STRINGERS, TAIL};

/// The nose's radius at `x` aft of its tip: a quarter ellipse, blunt at the tip and tangent to
/// the barrel at `NOSE`.
pub fn nose_radius(x: f64) -> f64 {
    let s = (1.0 - x / NOSE).clamp(0.0, 1.0);
    RADIUS * (1.0 - s.powf(2.2)).max(0.0).powf(1.0 / 2.2)
}

fn nose_profile(x0: f64, x1: f64, lift: f64, n: usize) -> Vec<(f64, f64)> {
    (0..=n)
        .map(|k| {
            let x = x0 + (x1 - x0) * k as f64 / n as f64;
            (x, nose_radius(x) + lift)
        })
        .collect()
}

fn nose() -> String {
    // Stations closer together at the tip, where the profile turns fastest.
    let pts: Vec<(f64, f64)> = (0..=16)
        .map(|k| {
            let t = k as f64 / 16.0;
            let x = NOSE * (1.0 - (t * std::f64::consts::FRAC_PI_2).cos());
            (x, nose_radius(x))
        })
        .collect();
    part(
        "nose",
        "FUS-NS-010",
        "composite",
        "The nose: a blunt quarter-ellipse profile turned about the fuselage axis, tangent to the barrel.",
        &format!(
            "{}    feature dome = revolve(profile = sketch.s, axis = origin.x, angle = 360 deg)\n",
            loop_sketch("s", "origin.xy", &[Curve::Spline(pts), Curve::Line(vec![(NOSE, 0.0)])])
        ),
        "dome",
    )
}

/// A corner of the windshield's outline rounded by a short spline: from `r` before `p` (towards
/// `a`) through a point inside the corner to `r` after it (towards `b`).
fn rounded(a: (f64, f64), p: (f64, f64), b: (f64, f64), r: f64) -> Vec<(f64, f64)> {
    let unit = |q: (f64, f64)| {
        let (du, dv) = (q.0 - p.0, q.1 - p.1);
        let l = du.hypot(dv);
        (du / l, dv / l)
    };
    let (ua, ub) = (unit(a), unit(b));
    vec![
        (p.0 + r * ua.0, p.1 + r * ua.1),
        (p.0 + 0.3 * r * (ua.0 + ub.0), p.1 + 0.3 * r * (ua.1 + ub.1)),
        (p.0 + r * ub.0, p.1 + r * ub.1),
    ]
}

/// The windshield: one tinted pane flush with the nose (0.5 mm proud, so it shades over the
/// skin), a shell of the nose's own profile cut to the side-view outline of the cockpit
/// glazing. The outline's sill falls aft from the crown and its brow falls more slowly, so the
/// front panes meet on the centerline and the side panes taper aft, as on a narrow-body.
fn windshield() -> String {
    let (x0, x1) = (1_900.0, 4_700.0);
    let outer = nose_profile(x0, x1, 0.5, 14);
    let mut inner = nose_profile(x0, x1, -6.0, 14);
    inner.reverse();
    // Side view, (z, x): forward sill, aft sill, aft brow, forward brow.
    let corners = [(1_736.0, 2_000.0), (1_150.0, 4_400.0), (1_560.0, 4_150.0), (2_125.0, 2_000.0)];
    let mut outline = Vec::new();
    for k in 0..4 {
        let (a, p, b) = (corners[(k + 3) % 4], corners[k], corners[(k + 1) % 4]);
        outline.push(Curve::Spline(rounded(a, p, b, 110.0)));
        outline.push(Curve::Line(vec![]));
    }
    part(
        "cockpit_windshield",
        "FUS-WS-030",
        "glass",
        "The cockpit windshield: a shell of the nose's profile cut to the glazing's side-view outline.",
        &format!(
            "{}    feature shell = revolve(profile = sketch.s, axis = origin.x, angle = 360 deg)\n{}    feature pane = extrude(profile = sketch.o, distance = 4400 mm, symmetric, mode = intersect)\n",
            loop_sketch("s", "origin.xy", &[Curve::Spline(outer), Curve::Line(vec![]), Curve::Spline(inner), Curve::Line(vec![])]),
            loop_sketch("o", "origin.zx", &outline)
        ),
        // The intersect trims the shell's body in place.
        "shell",
    )
}

fn frame() -> String {
    part(
        "frame",
        "FUS-FR-100",
        "aluminum_7075",
        "A fuselage frame: a ring 147 mm deep and 40 mm wide under the skin.",
        &format!(
            "{}    feature ring = revolve(profile = sketch.s, axis = origin.x, angle = 360 deg)\n",
            rect("s", "origin.xy", 0.0, RADIUS - 150.0, 40.0, RADIUS - 3.0)
        ),
        "ring",
    )
}

fn stringer() -> String {
    part(
        "stringer",
        "FUS-ST-200",
        "aluminum_7075",
        "A stringer the length of one barrel section, under the skin at the crown.",
        &format!(
            "{}    feature body = extrude(profile = sketch.s, distance = {})\n",
            rect("s", "origin.yz", -12.0, RADIUS - 43.0, 12.0, RADIUS - 3.0),
            mm(SECTION)
        ),
        "body",
    )
}

fn skin_panel() -> String {
    let sweep = 360.0 / SKIN_PANELS as f64;
    part(
        "skin_panel",
        "FUS-SK-300",
        "aluminum_2024",
        &format!("A skin panel: {sweep}° of the barrel, one section long, 3 mm thick."),
        &format!(
            "{}    feature panel = revolve(profile = sketch.s, axis = origin.x, angle = {sweep} deg)\n",
            rect("s", "origin.xy", 0.0, RADIUS - 3.0, SECTION, RADIUS)
        ),
        "panel",
    )
}

/// A cabin window: a rounded pane 230 mm along the fuselage and 250 mm round it, its outer
/// face 4 mm proud of the skin at the centre (the skin's curve buries its upper and lower edges).
fn cabin_window() -> String {
    let (w, h, r) = (115.0, 125.0, 80.0);
    // On the zx plane: u = z (round the barrel), v = x (along it); extruded radially along y.
    let body = format!(
        "    sketch s on origin.zx {{\n        l1 = line(start = {}, end = {})\n        a1 = arc(center = {}, start = l1.end, end = {})\n        l2 = line(start = a1.end, end = {})\n        a2 = arc(center = {}, start = l2.end, end = {})\n        l3 = line(start = a2.end, end = {})\n        a3 = arc(center = {}, start = l3.end, end = {})\n        l4 = line(start = a3.end, end = {})\n        a4 = arc(center = {}, start = l4.end, end = l1.start)\n    }}\n    feature pane = extrude(profile = sketch.s, distance = 12 mm)\n",
        pt(-h + r, -w),
        pt(h - r, -w),
        pt(h - r, -w + r),
        pt(h, -w + r),
        pt(h, w - r),
        pt(h - r, w - r),
        pt(h - r, w),
        pt(-h + r, w),
        pt(-h + r, w - r),
        pt(-h, w - r),
        pt(-h, -w + r),
        pt(-h + r, -w + r)
    );
    part("cabin_window", "FUS-WN-040", "glass", "A cabin window pane, flat, extruded outward through the skin.", &body, "pane")
}

fn rivet() -> String {
    part(
        "rivet",
        "MS20426AD6",
        "aluminum_2117",
        "A countersunk rivet, axis along z: its head sits flush, 1 mm under the skin's outer face.",
        &format!(
            "{}    feature body = revolve(profile = sketch.s, axis = origin.z, angle = 360 deg)\n",
            polygon("s", "origin.yz", &[(0.0, -16.0), (2.4, -16.0), (2.4, -3.6), (5.0, -1.0), (0.0, -1.0)])
        ),
        "body",
    )
}

/// The tail cone: circles lofted aft from the barrel, closing and rising to the APU exhaust.
fn tail_cone() -> String {
    let stations = [(0.0, 0.0, RADIUS), (3_000.0, 120.0, 1_760.0), (5_600.0, 520.0, 1_150.0), (TAIL, 980.0, 320.0)];
    let mut body = String::new();
    for (k, (x, zc, r)) in stations.iter().enumerate() {
        if k > 0 {
            body += &format!("    feature st{k} = plane.offset(base = origin.yz, distance = {})\n", mm(*x));
        }
        let plane = if k == 0 { "origin.yz".to_string() } else { format!("plane.st{k}") };
        body += &format!(
            "    sketch s{k} on {plane} {{\n        c = circle(center = {}, radius = {})\n    }}\n",
            pt(0.0, *zc),
            mm(*r)
        );
    }
    body += "    feature cone = loft(profiles = [sketch.s0, sketch.s1, sketch.s2, sketch.s3])\n";
    part("tail_cone", "FUS-TC-020", "aluminum_2024", "The tail cone: circles lofted aft from the barrel, closing and sweeping up to the APU exhaust.", &body, "cone")
}

/// The wing-to-body fairing: a long flattened blister under the wing box.
fn fairing() -> String {
    let len = 11_000.0;
    let pts: Vec<(f64, f64)> = (0..=14)
        .map(|k| {
            let t = k as f64 / 14.0;
            let x = len * t;
            (x, 1_350.0 * (4.0 * t * (1.0 - t)).powf(0.45))
        })
        .collect();
    part(
        "fairing",
        "FUS-FR-050",
        "composite",
        "The wing-to-body fairing: a long blister turned about its own axis under the wing box.",
        &format!(
            "{}    feature blister = revolve(profile = sketch.s, axis = origin.x, angle = 360 deg)\n",
            loop_sketch("s", "origin.xy", &[Curve::Spline(pts), Curve::Line(vec![])])
        ),
        "blister",
    )
}

pub fn parts() -> Vec<(String, String)> {
    vec![
        ("nose.part".into(), nose()),
        ("cockpit_windshield.part".into(), windshield()),
        ("frame.part".into(), frame()),
        ("stringer.part".into(), stringer()),
        ("skin_panel.part".into(), skin_panel()),
        ("cabin_window.part".into(), cabin_window()),
        ("rivet.part".into(), rivet()),
        ("tail_cone.part".into(), tail_cone()),
        ("fairing.part".into(), fairing()),
    ]
}

pub fn rivet_ring() -> String {
    format!(
        "{HEADER}// A ring of flush rivets through the skin onto one frame, one at each stringer.\nassembly rivet_ring {{\n{}    pattern around = circular(seeds = [rivet], axis_point = [0 mm, 0 mm, 0 mm], axis = [1, 0, 0], count = {STRINGERS}, angle = 360 deg)\n}}\n",
        use_("part", "rivet", "rivet", &[offset(20.0, 0.0, RADIUS)])
    )
}

pub fn fuselage_section() -> String {
    let frames = (SECTION / FRAME_PITCH) as usize + 1;
    let windows = (SECTION / FRAME_PITCH) as usize;
    // The window belt: pane centres 350 mm above the axis, each turned to face outward.
    let zc = 350.0;
    let tilt = (zc / RADIUS).asin().to_degrees();
    // The pane's inner face sits 8 mm inside the skin, its outer face 4 mm proud.
    let rho = RADIUS + 4.0 - 12.0;
    let (yc, zw) = (rho * tilt.to_radians().cos(), rho * tilt.to_radians().sin());
    let half_panel = 180.0 / SKIN_PANELS as f64;
    let mut s = format!(
        "{HEADER}// One barrel section: {frames} frames at the frame pitch, {STRINGERS} stringers, {SKIN_PANELS} skin panels,\n// a ring of flush rivets at every frame and a row of cabin windows each side between the frames.\nassembly fuselage_section {{\n    input frame_pitch : length = {}\n\n",
        mm(FRAME_PITCH)
    );
    s += &use_("part", "frame", "frame", &[]);
    s += &use_("part", "stringer", "stringer", &[]);
    s += &use_("part", "skin_panel", "panel", &[offset_rot(0.0, 0.0, 0.0, [1.0, 0.0, 0.0], half_panel)]);
    s += &use_("assembly", "rivet_ring", "rivets", &[]);
    // The pane faces +y; turned about x it faces the belt's normal on each side.
    s += &use_("part", "cabin_window", "window_starboard", &[offset_rot(FRAME_PITCH / 2.0, yc, zw, [1.0, 0.0, 0.0], tilt)]);
    s += &use_("part", "cabin_window", "window_port", &[offset_rot(FRAME_PITCH / 2.0, -yc, zw, [1.0, 0.0, 0.0], 180.0 - tilt)]);
    s += &format!("\n    pattern frames = linear(seeds = [frame], direction = [1, 0, 0], spacing = input.frame_pitch, count = {frames})\n");
    s += &format!("    pattern stringers = circular(seeds = [stringer], axis_point = [0 mm, 0 mm, 0 mm], axis = [1, 0, 0], count = {STRINGERS}, angle = 360 deg)\n");
    s += &format!("    pattern panels = circular(seeds = [panel], axis_point = [0 mm, 0 mm, 0 mm], axis = [1, 0, 0], count = {SKIN_PANELS}, angle = 360 deg)\n");
    s += &format!("    pattern rivet_rings = linear(seeds = [rivets], direction = [1, 0, 0], spacing = input.frame_pitch, count = {frames})\n");
    s += &format!("    pattern windows = linear(seeds = [window_starboard, window_port], direction = [1, 0, 0], spacing = input.frame_pitch, count = {windows})\n");
    s += "}\n";
    s
}

pub fn fuselage() -> String {
    let mut s = format!("{HEADER}// The fuselage: the nose and its cockpit windows, three barrel sections, the tail cone and the\n// wing-to-body fairing, along x.\nassembly fuselage {{\n");
    s += &use_("part", "nose", "nose", &[]);
    s += &use_("part", "cockpit_windshield", "windshield", &[]);
    for (k, name) in ["forward", "center", "aft"].iter().enumerate() {
        s += &use_("assembly", "fuselage_section", name, &[offset(NOSE + k as f64 * SECTION, 0.0, 0.0)]);
    }
    s += &use_("part", "tail_cone", "tail_cone", &[offset(NOSE + 3.0 * SECTION, 0.0, 0.0)]);
    s += &use_("part", "fairing", "fairing", &[offset(super::wing::ROOT_X - 1_200.0, 0.0, -1_250.0)]);
    s += "}\n";
    s
}
