//! The wings: a swept, tapered wing with dihedral on cambered NACA sections, lofted root to
//! tip. Each wing is a slat and fixed leading edge, upper and lower box skins over two spars and
//! ribs at the rib pitch, trailing-edge flaps and an aileron, a winglet, flush rivet rows along
//! the spar caps, and an engine nacelle on its pylon.
//!
//! Wing coordinates: the origin is the leading edge on the aircraft's centreline (the skeleton's
//! `wing_root_*` frame), x aft, y outboard (to starboard), z up.

use super::airfoil::{nose, run, Naca, Placed};
use super::text::{inputs, loop_sketch, mirror_port, mm, num, offset, part, sided, use_, with_sided, Curve, HEADER, SIDE_INPUT};
use super::RIB_PITCH;

/// Where the wing frame sits in the aircraft: x of the leading edge at the centreline, z.
pub const ROOT_X: f64 = 12_600.0;
pub const ROOT_Z: f64 = -1_250.0;
/// The semispan from the centreline, and the structural root station inside the fuselage.
pub const SEMISPAN: f64 = 17_000.0;
pub const ROOT_STATION: f64 = 1_400.0;
pub const ROOT_CHORD: f64 = 6_400.0;
pub const TIP_CHORD: f64 = 1_550.0;
/// Leading-edge sweep 27° and dihedral 5.5°, as slopes.
pub const LE_SLOPE: f64 = 0.509_525;
pub const DIHEDRAL_SLOPE: f64 = 0.096_289;
pub const ROOT_THICKNESS: f64 = 0.145;
pub const TIP_THICKNESS: f64 = 0.105;
/// Spar stations, fractions of chord.
pub const FRONT: f64 = 0.15;
pub const REAR: f64 = 0.65;
const SKIN: f64 = 4.0;

pub fn chord(y: f64) -> f64 {
    ROOT_CHORD - (ROOT_CHORD - TIP_CHORD) * y / SEMISPAN
}
pub fn x_le(y: f64) -> f64 {
    LE_SLOPE * y
}
pub fn z_le(y: f64) -> f64 {
    DIHEDRAL_SLOPE * y
}
pub fn section(y: f64) -> Naca {
    Naca { m: 0.02, p: 0.4, t: ROOT_THICKNESS - (ROOT_THICKNESS - TIP_THICKNESS) * y / SEMISPAN }
}
/// The section at `y`, mapped into its zx sketch plane (u = z, v = x).
fn placed(y: f64) -> Placed {
    Placed { chord: chord(y), le: (z_le(y), x_le(y)), chord_along_v: true, flip: false }
}

/// What a lofted wing piece holds of each section.
#[derive(Clone, Copy)]
enum Piece {
    /// The nose round to `FRONT` on both surfaces, closed at the front spar.
    Nose,
    /// A skin between the spars, `SKIN` thick, upper or lower.
    Box { up: bool },
    /// From the rear spar to the trailing edge.
    Tail,
}

fn piece_curves(piece: Piece, y: f64) -> Vec<Curve> {
    let f = section(y);
    let at = placed(y);
    let m = |v: Vec<(f64, f64)>| v.into_iter().map(|p| at.map(p)).collect::<Vec<_>>();
    match piece {
        Piece::Nose => vec![Curve::Spline(m(nose(&f, FRONT, 10))), Curve::Line(vec![])],
        Piece::Box { up } => {
            let depth = SKIN / chord(y);
            vec![
                Curve::Spline(m(run(&f, FRONT, REAR, up, 0.0, 8))),
                Curve::Line(vec![]),
                Curve::Spline(m(run(&f, REAR, FRONT, up, depth, 8))),
                Curve::Line(vec![]),
            ]
        }
        Piece::Tail => vec![
            Curve::Spline(m(run(&f, REAR, 1.0, true, 0.0, 8))),
            Curve::Line(vec![]),
            Curve::Spline(m(run(&f, 1.0, REAR, false, 0.0, 8))),
            Curve::Line(vec![]),
        ],
    }
}

/// A piece lofted between stations `span = (y0, y1)`, starboard positive; its `side` input set
/// to -1 builds the port piece.
fn lofted(name: &str, number: &str, material: &str, comment: &str, piece: Piece, span: (f64, f64)) -> String {
    let (y0, y1) = span;
    let mut body = SIDE_INPUT.to_string();
    for (k, y) in [y0, y1].into_iter().enumerate() {
        body += &format!("    feature st{k} = plane.offset(base = origin.zx, distance = {})\n", sided(y));
        body += &loop_sketch(&format!("s{k}"), &format!("plane.st{k}"), &piece_curves(piece, y));
    }
    body += "    feature piece = loft(profiles = [sketch.s0, sketch.s1])\n";
    part(name, number, material, comment, &body, "piece")
}

/// A spar web at `at` of chord, 30 mm thick, between the box skins, root to tip.
fn spar(name: &str, number: &str, at: f64) -> String {
    let mut body = SIDE_INPUT.to_string();
    for (k, y) in [ROOT_STATION, SEMISPAN].into_iter().enumerate() {
        let f = section(y);
        let depth = SKIN / chord(y);
        let p = placed(y);
        let half = 15.0 / chord(y);
        let up = p.map(f.inside(at, true, depth));
        let lo = p.map(f.inside(at, false, depth));
        let (u0, u1) = (lo.0, up.0);
        let (v0, v1) = (p.map((at - half, 0.0)).1, p.map((at + half, 0.0)).1);
        body += &format!("    feature st{k} = plane.offset(base = origin.zx, distance = {})\n", sided(y));
        body += &super::text::rect(&format!("s{k}"), &format!("plane.st{k}"), u0, v0, u1, v1);
    }
    body += "    feature web = loft(profiles = [sketch.s0, sketch.s1])\n";
    part(name, number, "aluminum_7075", &format!("A spar web at {} % chord, root to tip.", (at * 100.0).round()), &body, "web")
}

/// How far a rib's caps sit inside the box skins' inner face: deep enough that the rib's spline
/// section never meets the lofted skin between their stations.
const RIB_CLEARANCE: f64 = 12.0;

/// A rib between the spars, 20 mm thick, for any station: its chord and thickness ratio are
/// inputs, so one definition serves every rib (each occurrence sets them).
fn rib() -> String {
    let f = |up: bool| -> Vec<String> {
        let xs = super::airfoil::stations(FRONT + 0.004, REAR - 0.004, 8);
        let xs: Vec<f64> = if up { xs } else { xs.into_iter().rev().collect() };
        xs.into_iter()
            .map(|x| {
                let unit = Naca { m: 0.02, p: 0.4, t: 1.0 };
                let camber = Naca { m: 0.02, p: 0.4, t: 0.0 }.at(x, up).1;
                let thick = unit.at(x, up).1 - camber;
                let s = if up { "-" } else { "+" };
                format!(
                    "[input.chord * ({} + input.thickness * {}) {s} {}, input.chord * {}]",
                    num(camber),
                    num(thick),
                    mm(SKIN + RIB_CLEARANCE),
                    num(x)
                )
            })
            .collect()
    };
    let body = format!(
        "    input chord : length = {}\n    input thickness : real = {}\n\n    sketch s on origin.zx {{\n        c1 = spline(points = [{}])\n        c2 = line(start = c1.end, end = {})\n        c3 = spline(points = [c2.end, {}])\n        c4 = line(start = c3.end, end = c1.start)\n    }}\n    feature web = extrude(profile = sketch.s, distance = 20 mm)\n",
        mm(ROOT_CHORD),
        num(ROOT_THICKNESS),
        f(true).join(", "),
        f(false)[0],
        f(false)[1..].join(", ")
    );
    part("rib", "WNG-RB-400", "aluminum_7075", "A wing rib between the spars, its section scaled to the station's chord and thickness.", &body, "web")
}

/// The winglet: a swept, canted blade on NACA 0010 sections, lofted up from inside the tip.
fn winglet(name: &str) -> String {
    let f = Naca::symmetric(0.10);
    let (y, c0, c1, h) = (SEMISPAN - 90.0, 1_250.0, 420.0, 2_300.0);
    let base = (x_le(SEMISPAN) + 120.0, y);
    let top = (base.0 + h * 0.84, y + h * 0.21);
    let mut body = SIDE_INPUT.to_string();
    for (k, (le, c, z)) in [(base, c0, z_le(SEMISPAN)), (top, c1, z_le(SEMISPAN) + h)].into_iter().enumerate() {
        let at = Placed { chord: c, le, chord_along_v: false, flip: false };
        let m = |v: Vec<(f64, f64)>| v.into_iter().map(|p| at.map(p)).collect::<Vec<_>>();
        body += &format!("    feature st{k} = plane.offset(base = origin.xy, distance = {})\n", mm(z));
        // The sections' y (v) changes sign with the hand.
        body += &with_sided(false, true, || loop_sketch(&format!("s{k}"), &format!("plane.st{k}"), &[Curve::Spline(m(nose(&f, 1.0, 12))), Curve::Line(vec![])]));
    }
    body += "    feature blade = loft(profiles = [sketch.s0, sketch.s1])\n";
    part(name, "WNG-WL-700", "composite", "The winglet: a swept, canted blade lofted up from inside the wing tip.", &body, "blade")
}

/// Spanwise pieces: (file stem, part number, material, comment, piece, from, to).
fn pieces() -> Vec<(&'static str, &'static str, &'static str, &'static str, Piece, f64, f64)> {
    vec![
        ("leading_edge", "WNG-LE-610", "aluminum_2024", "The fixed leading edge inboard of the engine.", Piece::Nose, ROOT_STATION, 5_000.0),
        ("slat", "WNG-SL-620", "aluminum_2024", "The leading-edge slat, engine to tip.", Piece::Nose, 5_000.0, SEMISPAN),
        ("box_skin_upper", "WNG-SK-600", "aluminum_2024", "The upper wing-box skin between the spars, 4 mm thick, root to tip.", Piece::Box { up: true }, ROOT_STATION, SEMISPAN),
        ("box_skin_lower", "WNG-SK-610", "aluminum_2024", "The lower wing-box skin between the spars, 4 mm thick, root to tip.", Piece::Box { up: false }, ROOT_STATION, SEMISPAN),
        ("flap_inboard", "WNG-FL-630", "aluminum_2024", "The inboard flap, aft of the rear spar.", Piece::Tail, ROOT_STATION, 6_400.0),
        ("flap_outboard", "WNG-FL-640", "aluminum_2024", "The outboard flap.", Piece::Tail, 6_400.0, 12_400.0),
        ("aileron", "WNG-AL-650", "aluminum_2024", "The aileron.", Piece::Tail, 12_400.0, 15_700.0),
        ("trailing_edge_tip", "WNG-TE-660", "aluminum_2024", "The fixed trailing edge outboard of the aileron.", Piece::Tail, 15_700.0, SEMISPAN),
    ]
}

/// The handed parts of a wing, by file stem.
fn handed() -> Vec<String> {
    let mut out: Vec<String> = pieces().iter().map(|p| format!("wing_{}", p.0)).collect();
    out.extend(["wing_front_spar", "wing_rear_spar", "wing_winglet"].map(String::from));
    out
}

pub fn parts() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (name, number, material, comment, piece, y0, y1) in pieces() {
        let n = format!("wing_{name}");
        out.push((format!("{n}.part"), lofted(&n, number, material, comment, piece, (y0, y1))));
    }
    out.push(("wing_front_spar.part".into(), spar("wing_front_spar", "WNG-SP-500", FRONT)));
    out.push(("wing_rear_spar.part".into(), spar("wing_rear_spar", "WNG-SP-510", REAR)));
    out.push(("wing_winglet.part".into(), winglet("wing_winglet")));
    out.push(("rib.part".into(), rib()));
    out
}

/// The wing: the starboard wing's slat and leading edge, box skins over the front and rear
/// spars and ribs at the rib pitch, flaps and aileron, the winglet, flush rivet rows along the
/// spar caps and the engine on its pylon, and the port wing as their mirror image about the
/// centre plane (the handed parts in their opposite hand).
pub fn wing() -> String {
    let ribs = ((SEMISPAN - 200.0 - ROOT_STATION) / RIB_PITCH) as usize + 1;
    let mut s = format!(
        "{HEADER}// Both wings: slat and leading edge, box skins over the front and rear spars and {ribs} ribs at the\n// rib pitch, flaps and aileron, the winglet, flush rivet rows along the spar caps, and the engine\n// on its pylon, to starboard; the port wing is their mirror image. The origin is the leading edge\n// at the aircraft's centreline.\nassembly wing {{\n    input rib_pitch : length = {}\n    input root_station : length = {}\n    input semispan : length = {}\n    input root_chord : length = {}\n    input tip_chord : length = {}\n    input root_thickness : real = {}\n    input tip_thickness : real = {}\n\n",
        mm(RIB_PITCH),
        mm(ROOT_STATION),
        mm(SEMISPAN),
        mm(ROOT_CHORD),
        mm(TIP_CHORD),
        num(ROOT_THICKNESS),
        num(TIP_THICKNESS)
    );
    let aliases: Vec<(String, String)> = handed()
        .into_iter()
        .map(|h| {
            let alias = h.trim_start_matches("wing_").to_string();
            (h, alias)
        })
        .collect();
    for (design, alias) in &aliases {
        s += &use_("part", design, alias, &[]);
    }
    // Ribs: each at its station, its section from the planform inputs.
    for k in 0..ribs {
        let y = ROOT_STATION + 100.0 + k as f64 * RIB_PITCH;
        let station = format!("(input.root_station + 100 mm + {k} * input.rib_pitch)");
        let frac = format!("{station} / input.semispan");
        // The skins are lofted straight between the root-station and tip sections, so at a
        // station between them the depth (chord times thickness ratio) runs linearly, not the
        // product of the linear chord and the linear ratio: the rib takes that depth over its
        // chord as its ratio, or its caps would stand up to 24 mm through the skin at midspan.
        let chord_at = |f: &str| format!("(input.root_chord - {f} * (input.root_chord - input.tip_chord))");
        let ratio_at = |f: &str| format!("(input.root_thickness - {f} * (input.root_thickness - input.tip_thickness))");
        let root_frac = "(input.root_station / input.semispan)";
        let root_depth = format!("{} * {}", chord_at(root_frac), ratio_at(root_frac));
        let tip_depth = "input.tip_chord * input.tip_thickness".to_string();
        let along = format!("(({station} - input.root_station) / (input.semispan - input.root_station))");
        let depth = format!("({root_depth} + {along} * ({tip_depth} - {root_depth}))");
        s += &use_(
            "part",
            "rib",
            &format!("rib{}", k + 1),
            &[
                inputs(&[
                    ("chord", format!("input.root_chord - {frac} * (input.root_chord - input.tip_chord)")),
                    ("thickness", format!("{depth} / {}", chord_at(&frac))),
                ]),
                offset(x_le(y), y, z_le(y)),
            ],
        );
    }
    // Rivet rows on the spar caps: straight lines root to tip, one rivet each rivet pitch, on
    // both wings (a mirror repeats occurrences, not patterns, so each side has its rows).
    for (side, suffix) in [(1.0, ""), (-1.0, "_port")] {
    for (k, (at, up)) in [(FRONT + 0.012, true), (FRONT + 0.012, false), (REAR - 0.012, true), (REAR - 0.012, false)].into_iter().enumerate() {
        let point = |y: f64| {
            let p = section(y).at(at, up);
            let c = chord(y);
            [x_le(y) + p.0 * c, side * y, z_le(y) + p.1 * c]
        };
        let (a, b) = (point(ROOT_STATION + 60.0), point(SEMISPAN - 150.0));
        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let count = (len / super::RIVET_PITCH) as usize + 1;
        let place = if up { offset(a[0], a[1], a[2]) } else { super::text::offset_rot(a[0], a[1], a[2], [1.0, 0.0, 0.0], 180.0) };
        s += &use_("part", "rivet", &format!("rivet{k}{suffix}"), &[place]);
        s += &format!(
            "    pattern rivets{k}{suffix} = linear(seeds = [rivet{k}{suffix}], direction = [{}, {}, {}], spacing = {}, count = {count})\n",
            num(d[0] / len),
            num(d[1] / len),
            num(d[2] / len),
            mm(super::RIVET_PITCH)
        );
    }
    }
    let ye = 5_800.0;
    s += &use_("assembly", "nacelle", "engine", &[offset(x_le(ye) - 2_700.0, ye, z_le(ye) - 1_550.0)]);
    // The port wing: each handed part's image in its opposite hand; the ribs (symmetric about
    // their mid-plane, 10 mm off their own origin) and the engine (symmetric about its own
    // y = 0) reflected as they are.
    s += "\n";
    for (design, alias) in &aliases {
        s += &mirror_port(&format!("{alias}_port"), "part", design, &[alias], &[]);
    }
    let rib_seeds: Vec<String> = (1..=ribs).map(|k| format!("rib{k}")).collect();
    s += &format!(
        "    mirror ribs_port = {{\n        seeds = [{}],\n        plane = {{ plane = explicit, point = [0 mm, 0 mm, 0 mm], normal = [0, 1, 0] }},\n        hand = symmetric(point = [0 mm, 10 mm, 0 mm], normal = [0, 1, 0])\n    }}\n",
        rib_seeds.join(", ")
    );
    s += "    mirror engine_port = {\n        seeds = [engine],\n        plane = { plane = explicit, point = [0 mm, 0 mm, 0 mm], normal = [0, 1, 0] },\n        hand = symmetric(point = [0 mm, 0 mm, 0 mm], normal = [0, 1, 0])\n    }\n";
    s += "}\n";
    s
}
