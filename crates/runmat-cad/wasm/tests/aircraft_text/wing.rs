//! The wings: a swept, tapered wing with dihedral on cambered NACA sections, lofted root to
//! tip. Each wing is a slat and fixed leading edge, upper and lower box skins over two spars and
//! ribs at the rib pitch, trailing-edge flaps and an aileron, a winglet, flush rivet rows along
//! the spar caps, and an engine nacelle on its pylon.
//!
//! Wing coordinates: the origin is the leading edge on the aircraft's centreline (the skeleton's
//! `wing_root_*` frame), x aft, y outboard (to starboard), z up.

use super::airfoil::{nose, run, Naca, Placed};
use super::text::{inputs, loop_sketch, mm, num, offset, part, use_, Curve, HEADER};
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
const FRONT: f64 = 0.15;
const REAR: f64 = 0.65;
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
fn section(y: f64) -> Naca {
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

/// A piece lofted between stations `y0` and `y1` (starboard positive; `side` -1 mirrors it).
fn lofted(name: &str, number: &str, material: &str, comment: &str, piece: Piece, y0: f64, y1: f64, side: f64) -> String {
    let mut body = String::new();
    for (k, y) in [y0, y1].into_iter().enumerate() {
        body += &format!("    feature st{k} = plane.offset(base = origin.zx, distance = {})\n", mm(side * y));
        body += &loop_sketch(&format!("s{k}"), &format!("plane.st{k}"), &piece_curves(piece, y));
    }
    body += "    feature piece = loft(profiles = [sketch.s0, sketch.s1])\n";
    part(name, number, material, comment, &body, "piece")
}

/// A spar web at `at` of chord, 30 mm thick, between the box skins, root to tip.
fn spar(name: &str, number: &str, at: f64, side: f64) -> String {
    let mut body = String::new();
    for (k, y) in [ROOT_STATION, SEMISPAN].into_iter().enumerate() {
        let f = section(y);
        let depth = SKIN / chord(y);
        let p = placed(y);
        let half = 15.0 / chord(y);
        let up = p.map(f.inside(at, true, depth));
        let lo = p.map(f.inside(at, false, depth));
        let (u0, u1) = (lo.0, up.0);
        let (v0, v1) = (p.map((at - half, 0.0)).1, p.map((at + half, 0.0)).1);
        body += &format!("    feature st{k} = plane.offset(base = origin.zx, distance = {})\n", mm(side * y));
        body += &super::text::rect(&format!("s{k}"), &format!("plane.st{k}"), u0, v0, u1, v1);
    }
    body += "    feature web = loft(profiles = [sketch.s0, sketch.s1])\n";
    part(name, number, "aluminum_7075", &format!("A spar web at {} % chord, root to tip.", (at * 100.0).round()), &body, "web")
}

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
                    mm(SKIN + 1.0),
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
fn winglet(name: &str, side: f64) -> String {
    let f = Naca::symmetric(0.10);
    let (y, c0, c1, h) = (SEMISPAN - 90.0, 1_250.0, 420.0, 2_300.0);
    let base = (x_le(SEMISPAN) + 120.0, side * y);
    let top = (base.0 + h * 0.84, side * (y + h * 0.21));
    let mut body = String::new();
    for (k, (le, c, z)) in [(base, c0, z_le(SEMISPAN)), (top, c1, z_le(SEMISPAN) + h)].into_iter().enumerate() {
        let at = Placed { chord: c, le, chord_along_v: false, flip: side < 0.0 };
        let m = |v: Vec<(f64, f64)>| v.into_iter().map(|p| at.map(p)).collect::<Vec<_>>();
        body += &format!("    feature st{k} = plane.offset(base = origin.xy, distance = {})\n", mm(z));
        body += &loop_sketch(&format!("s{k}"), &format!("plane.st{k}"), &[Curve::Spline(m(nose(&f, 1.0, 12))), Curve::Line(vec![])]);
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

fn stem(name: &str, side: f64) -> String {
    if side > 0.0 { format!("wing_{name}") } else { format!("wing_{name}_port") }
}

pub fn parts() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for side in [1.0, -1.0] {
        for (name, number, material, comment, piece, y0, y1) in pieces() {
            let n = stem(name, side);
            out.push((format!("{n}.part"), lofted(&n, number, material, comment, piece, y0, y1, side)));
        }
        let (f, r) = (stem("front_spar", side), stem("rear_spar", side));
        out.push((format!("{f}.part"), spar(&f, "WNG-SP-500", FRONT, side)));
        out.push((format!("{r}.part"), spar(&r, "WNG-SP-510", REAR, side)));
        let w = stem("winglet", side);
        out.push((format!("{w}.part"), winglet(&w, side)));
    }
    out.push(("rib.part".into(), rib()));
    out
}

/// A wing assembly: `side` 1 starboard, -1 port (the opposite-hand pieces).
pub fn wing(name: &str, side: f64) -> String {
    let side_name = if side > 0.0 { "starboard" } else { "port" };
    let ribs = ((SEMISPAN - 200.0 - ROOT_STATION) / RIB_PITCH) as usize + 1;
    let mut s = format!(
        "{HEADER}// The {side_name} wing: slat and leading edge, box skins over the front and rear spars and {ribs} ribs at\n// the rib pitch, flaps and aileron, the winglet, flush rivet rows along the spar caps, and the\n// engine on its pylon. The origin is the leading edge at the aircraft's centreline.\nassembly {name} {{\n    input rib_pitch : length = {}\n    input root_station : length = {}\n    input semispan : length = {}\n    input root_chord : length = {}\n    input tip_chord : length = {}\n    input root_thickness : real = {}\n    input tip_thickness : real = {}\n\n",
        mm(RIB_PITCH),
        mm(ROOT_STATION),
        mm(SEMISPAN),
        mm(ROOT_CHORD),
        mm(TIP_CHORD),
        num(ROOT_THICKNESS),
        num(TIP_THICKNESS)
    );
    for (n, ..) in pieces() {
        s += &use_("part", &stem(n, side), n, &[]);
    }
    s += &use_("part", &stem("front_spar", side), "front_spar", &[]);
    s += &use_("part", &stem("rear_spar", side), "rear_spar", &[]);
    s += &use_("part", &stem("winglet", side), "winglet", &[]);
    // Ribs: each at its station, its section from the planform inputs.
    for k in 0..ribs {
        let y = ROOT_STATION + 100.0 + k as f64 * RIB_PITCH;
        let station = format!("(input.root_station + 100 mm + {k} * input.rib_pitch)");
        let frac = format!("{station} / input.semispan");
        let rib_y = if side > 0.0 { y } else { -y - 20.0 };
        s += &use_(
            "part",
            "rib",
            &format!("rib{}", k + 1),
            &[
                inputs(&[
                    ("chord", format!("input.root_chord - {frac} * (input.root_chord - input.tip_chord)")),
                    ("thickness", format!("input.root_thickness - {frac} * (input.root_thickness - input.tip_thickness)")),
                ]),
                offset(x_le(y), rib_y, z_le(y)),
            ],
        );
    }
    // Rivet rows on the spar caps: straight lines root to tip, one rivet each rivet pitch.
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
        s += &use_("part", "rivet", &format!("rivet{k}"), &[place]);
        s += &format!(
            "    pattern rivets{k} = linear(seeds = [rivet{k}], direction = [{}, {}, {}], spacing = {}, count = {count})\n",
            num(d[0] / len),
            num(d[1] / len),
            num(d[2] / len),
            mm(super::RIVET_PITCH)
        );
    }
    let ye = 5_800.0;
    s += &use_("assembly", "nacelle", "engine", &[offset(x_le(ye) - 2_700.0, side * ye, z_le(ye) - 1_550.0)]);
    s += "}\n";
    s
}
