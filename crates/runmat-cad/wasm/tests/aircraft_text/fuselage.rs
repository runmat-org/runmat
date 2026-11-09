//! The fuselage (the nose and its glazing are in `nose.rs`): three barrel sections (frames,
//! stringers, skin panels, flush rivet rings and a row of cabin windows each side), an upswept
//! tail cone and a wing-to-body fairing.

use super::text::{
    loop_sketch, mm, offset, offset_rot, part, polygon, pt, rect, use_, Curve, HEADER,
};
use super::{FRAME_PITCH, NOSE, RADIUS, SECTION, SKIN_PANELS, STRINGERS, TAIL};

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

/// Half a cabin window's length along the fuselage.
const WINDOW_HALF_LENGTH: f64 = 115.0;

/// A cabin window: a rounded pane 230 mm along the fuselage and 250 mm round it, its outer
/// face 4 mm proud of the skin at the centre (the skin's curve buries its upper and lower edges).
fn cabin_window() -> String {
    let (w, h, r) = (WINDOW_HALF_LENGTH, 125.0, 80.0);
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
    part(
        "cabin_window",
        "FUS-WN-040",
        "glass",
        "A cabin window pane, flat, extruded outward through the skin.",
        &body,
        "pane",
    )
}

fn rivet() -> String {
    part(
        "rivet",
        "MS20426AD6",
        "aluminum_2117",
        "A countersunk rivet, axis along z: its head sits flush, 1 mm under the skin's outer face.",
        &format!(
            "{}    feature body = revolve(profile = sketch.s, axis = origin.z, angle = 360 deg)\n",
            polygon(
                "s",
                "origin.yz",
                &[
                    (0.0, -16.0),
                    (2.4, -16.0),
                    (2.4, -3.6),
                    (5.0, -1.0),
                    (0.0, -1.0)
                ]
            )
        ),
        "body",
    )
}

/// The tail cone: circles lofted aft from the barrel, closing and rising to the APU exhaust.
fn tail_cone() -> String {
    let stations = [
        (0.0, 0.0, RADIUS),
        (3_000.0, 120.0, 1_760.0),
        (5_600.0, 520.0, 1_150.0),
        (TAIL, 980.0, 320.0),
    ];
    let mut body = String::new();
    for (k, (x, zc, r)) in stations.iter().enumerate() {
        if k > 0 {
            body += &format!(
                "    feature st{k} = plane.offset(base = origin.yz, distance = {})\n",
                mm(*x)
            );
        }
        let plane = if k == 0 {
            "origin.yz".to_string()
        } else {
            format!("plane.st{k}")
        };
        // The skin's outer circle and, 5 mm inside it, the inner one.
        for (tag, radius) in [("s", *r), ("i", *r - 5.0)] {
            body += &format!(
                "    sketch {tag}{k} on {plane} {{\n        c = circle(center = {}, radius = {})\n    }}\n",
                pt(0.0, *zc),
                mm(radius)
            );
        }
    }
    body += "    feature cone = loft(profiles = [sketch.s0, sketch.s1, sketch.s2, sketch.s3])\n";
    // A 5 mm skin, open at the barrel joint and at the APU exhaust: the inner loft is cut away.
    body += "    feature hollow = loft(profiles = [sketch.i0, sketch.i1, sketch.i2, sketch.i3], mode = cut)\n";
    part("tail_cone", "FUS-TC-020", "aluminum_2024", "The tail cone: a 5 mm skin lofted aft through circles from the barrel, closing and sweeping up to the APU exhaust.", &body, "cone")
}

/// Where the aft pressure bulkhead's rim sits, aft of the barrel's end (behind the aft galleys),
/// and its rim radius, inside the tail cone's skin there.
const BULKHEAD_AT: f64 = 700.0;
const BULKHEAD_RIM: f64 = RADIUS - 30.0;

/// The aft pressure bulkhead: a 4 mm spherical dome, convex aft, 800 mm deep, closing the
/// pressurized cabin inside the tail cone, with a small fitting hole at its apex.
fn aft_pressure_bulkhead() -> String {
    let (depth, wall, apex) = (800.0, 4.0, 40.0);
    let rho = (depth * depth + BULKHEAD_RIM * BULKHEAD_RIM) / (2.0 * depth);
    let c = depth - rho;
    // Points on the sphere of radius `r` about (c, 0), from the rim plane x = 0 to the apex hole.
    let arc = |r: f64| -> Vec<(f64, f64)> {
        let (t0, t1) = ((r * r - c * c).sqrt().atan2(-c), (apex / r).asin());
        (0..=12)
            .map(|k| t0 + (t1 - t0) * k as f64 / 12.0)
            .map(|t| (c + r * t.cos(), r * t.sin()))
            .collect()
    };
    let outer = arc(rho);
    let mut inner = arc(rho - wall);
    inner.reverse();
    part(
        "aft_pressure_bulkhead",
        "FUS-PB-060",
        "aluminum_2024",
        "The aft pressure bulkhead: a 4 mm spherical dome, convex aft, closing the cabin inside the tail cone.",
        &format!(
            "{}    feature dome = revolve(profile = sketch.s, axis = origin.x, angle = 360 deg)\n",
            loop_sketch("s", "origin.xy", &[Curve::Spline(outer), Curve::Line(vec![]), Curve::Spline(inner), Curve::Line(vec![])])
        ),
        "dome",
    )
}

/// The wing-to-body fairing: a long flattened blister under the wing box, a 20 mm shell, cut
/// back to the fuselage skin so nothing of it stands inside the cabin or the hold.
fn fairing() -> String {
    let len = 11_000.0;
    let r = |t: f64| 1_350.0 * (4.0 * t * (1.0 - t)).powf(0.45);
    let pts: Vec<(f64, f64)> = (0..=14)
        .map(|k| k as f64 / 14.0)
        .map(|t| (len * t, r(t)))
        .collect();
    let inner: Vec<(f64, f64)> = (0..=14)
        .map(|k| 0.03 + 0.94 * k as f64 / 14.0)
        .map(|t| (len * t, (r(t) - 20.0).max(5.0)))
        .collect();
    // The fairing's frame sits 1250 mm below the fuselage axis.
    let axis_z = 1_250.0;
    part(
        "fairing",
        "FUS-FR-050",
        "composite",
        "The wing-to-body fairing: a long blister shell under the wing box, cut back to the fuselage skin.",
        &format!(
            "{}    feature blister = revolve(profile = sketch.s, axis = origin.x, angle = 360 deg)\n{}    feature hollow = revolve(profile = sketch.i, axis = origin.x, angle = 360 deg, mode = cut)\n    sketch body_line on origin.yz {{\n        c = circle(center = {}, radius = {})\n    }}\n    feature skin_line = extrude(profile = sketch.body_line, distance = {}, mode = cut)\n",
            loop_sketch("s", "origin.xy", &[Curve::Spline(pts), Curve::Line(vec![])]),
            loop_sketch("i", "origin.xy", &[Curve::Spline(inner), Curve::Line(vec![(0.97 * len, 0.0), (0.03 * len, 0.0)])]),
            pt(0.0, axis_z),
            mm(RADIUS - 2.0),
            mm(len)
        ),
        "blister",
    )
}

pub fn parts() -> Vec<(String, String)> {
    vec![
        ("frame.part".into(), frame()),
        ("stringer.part".into(), stringer()),
        ("skin_panel.part".into(), skin_panel()),
        ("cabin_window.part".into(), cabin_window()),
        ("rivet.part".into(), rivet()),
        ("tail_cone.part".into(), tail_cone()),
        ("fairing.part".into(), fairing()),
        ("aft_pressure_bulkhead.part".into(), aft_pressure_bulkhead()),
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
    s += &use_(
        "part",
        "skin_panel",
        "panel",
        &[offset_rot(0.0, 0.0, 0.0, [1.0, 0.0, 0.0], half_panel)],
    );
    s += &use_("assembly", "rivet_ring", "rivets", &[]);
    // The pane faces +y; turned about x it faces the belt's normal on each side.
    // Two window rows from the middle of the section, one running aft and one forward, so no
    // pattern seed sits at a door (a door's windows are suppressed instances).
    let mid = windows / 2;
    for (suffix, bay) in [("", mid), ("_fwd", mid - 1)] {
        let x = FRAME_PITCH / 2.0 + bay as f64 * FRAME_PITCH;
        s += &use_(
            "part",
            "cabin_window",
            &format!("window_starboard{suffix}"),
            &[offset_rot(x, yc, zw, [1.0, 0.0, 0.0], tilt)],
        );
        s += &use_(
            "part",
            "cabin_window",
            &format!("window_port{suffix}"),
            &[offset_rot(x, -yc, zw, [1.0, 0.0, 0.0], 180.0 - tilt)],
        );
    }
    let (cabin_uses, cabin_patterns) = super::cabin::section_uses(frames, windows);
    s += &cabin_uses;
    s += &format!("\n    pattern frames = linear(seeds = [frame], direction = [1, 0, 0], spacing = input.frame_pitch, count = {frames})\n");
    s += &format!("    pattern stringers = circular(seeds = [stringer], axis_point = [0 mm, 0 mm, 0 mm], axis = [1, 0, 0], count = {STRINGERS}, angle = 360 deg)\n");
    s += &format!("    pattern panels = circular(seeds = [panel], axis_point = [0 mm, 0 mm, 0 mm], axis = [1, 0, 0], count = {SKIN_PANELS}, angle = 360 deg)\n");
    s += &format!("    pattern rivet_rings = linear(seeds = [rivets], direction = [1, 0, 0], spacing = input.frame_pitch, count = {frames})\n");
    s += &format!("    pattern windows = linear(seeds = [window_starboard, window_port], direction = [1, 0, 0], spacing = input.frame_pitch, count = {})\n", windows - mid);
    s += &format!("    pattern windows_fwd = linear(seeds = [window_starboard_fwd, window_port_fwd], direction = [-1, 0, 0], spacing = input.frame_pitch, count = {mid})\n");
    s += &cabin_patterns;
    s += "}\n";
    s
}

/// The steps of a pattern in the barrel section starting at `x0` (`count` items `pitch` apart,
/// the first centred `first` aft of `x0`, each `half` long either side) that a passenger door
/// spans.
fn at_doors(x0: f64, first: f64, pitch: f64, count: usize, half: f64) -> Vec<usize> {
    let doors = super::cabin::passenger_door_spans();
    (0..count)
        .filter(|j| {
            let c = x0 + first + *j as f64 * pitch;
            doors.iter().any(|(a, b)| c + half > *a && c - half < *b)
        })
        .collect()
}

/// Overrides that leave out the cabin windows and overhead bins at the passenger doors, in the
/// barrel section starting at `x0`.
fn door_overrides(x0: f64) -> Vec<String> {
    let windows = at_doors(
        x0,
        FRAME_PITCH / 2.0,
        FRAME_PITCH,
        (SECTION / FRAME_PITCH) as usize,
        WINDOW_HALF_LENGTH,
    );
    let (bin_first, bin_pitch, bin_count, bin_half) = super::cabin::bin_layout();
    let bins = at_doors(x0, bin_first, bin_pitch, bin_count, bin_half);
    let mut out = Vec::new();
    let rows = [
        (
            "windows",
            ["window_starboard", "window_port"],
            windows,
            (SECTION / FRAME_PITCH) as usize,
        ),
        ("bins", ["bin_starboard", "bin_port"], bins, bin_count),
    ];
    for (pattern, seeds, steps, count) in rows {
        // Each row runs aft from item count/2 and forward from the item ahead of it.
        let mid = count / 2;
        for j in steps {
            let (suffix, step) = if j >= mid {
                ("", j - mid)
            } else {
                ("_fwd", mid - 1 - j)
            };
            assert!(step > 0, "a door covers the seed of `{pattern}{suffix}`");
            for seed in seeds {
                out.push(format!("override \"{seed}{suffix}@{pattern}{suffix}[{step}]\" {{\n            suppressed = true\n        }}"));
            }
        }
    }
    out
}

pub fn fuselage() -> String {
    let mut s = format!("{HEADER}// The fuselage: the nose and its cockpit glazing, three barrel sections, the tail cone and the\n// wing-to-body fairing, along x.\nassembly fuselage {{\n");
    s += &super::nose::uses();
    for (k, name) in ["forward", "center", "aft"].iter().enumerate() {
        let x0 = NOSE + k as f64 * SECTION;
        let mut lines = vec![offset(x0, 0.0, 0.0)];
        lines.extend(door_overrides(x0));
        s += &use_("assembly", "fuselage_section", name, &lines);
    }
    s += &use_("assembly", "cockpit", "cockpit", &[]);
    s += &use_("assembly", "cabin", "cabin", &[]);
    s += &use_("assembly", "hold", "hold", &[]);
    s += &use_(
        "part",
        "tail_cone",
        "tail_cone",
        &[offset(NOSE + 3.0 * SECTION, 0.0, 0.0)],
    );
    s += &use_(
        "part",
        "aft_pressure_bulkhead",
        "aft_pressure_bulkhead",
        &[offset(NOSE + 3.0 * SECTION + BULKHEAD_AT, 0.0, 0.0)],
    );
    s += &use_(
        "part",
        "fairing",
        "fairing",
        &[offset(super::wing::ROOT_X - 1_200.0, 0.0, -1_250.0)],
    );
    s += "}\n";
    s
}
