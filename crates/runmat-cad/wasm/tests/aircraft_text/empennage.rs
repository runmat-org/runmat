//! The empennage: a swept vertical fin and rudder, and swept horizontal stabilizers with dihedral
//! and their elevators, on NACA 0010/0011 sections. Coordinates are the tail frame's: the origin
//! at the tail cone's front on the fuselage axis, x aft, y to starboard, z up.

use super::airfoil::{nose, run, Naca, Placed};
use super::text::{loop_sketch, mm, part, use_, Curve, HEADER};

/// Hinge line, fraction of chord: fixed surface ahead, control surface aft.
const HINGE: f64 = 0.70;

struct Surface {
    /// Leading edge at the root and tip, and the chords.
    root_le: (f64, f64, f64),
    tip_le: (f64, f64, f64),
    root_chord: f64,
    tip_chord: f64,
    thickness: f64,
    /// Sections on horizontal planes (a fin) or vertical ones (a stabilizer).
    vertical: bool,
}

impl Surface {
    /// The two sections' planes and placements.
    fn sections(&self, side: f64) -> [(String, Placed); 2] {
        [(self.root_le, self.root_chord), (self.tip_le, self.tip_chord)].map(|(le, c)| {
            if self.vertical {
                // Horizontal planes at z: u = x, v = y.
                (format!("plane.offset(base = origin.xy, distance = {})", mm(le.2)), Placed { chord: c, le: (le.0, le.1), chord_along_v: false, flip: false })
            } else {
                // Vertical planes at y: u = z, v = x.
                (format!("plane.offset(base = origin.zx, distance = {})", mm(side * le.1)), Placed { chord: c, le: (le.2, le.0), chord_along_v: true, flip: false })
            }
        })
    }

    fn piece(&self, name: &str, number: &str, comment: &str, control: bool, side: f64) -> String {
        let f = Naca::symmetric(self.thickness);
        let mut body = String::new();
        for (k, (plane, at)) in self.sections(side).into_iter().enumerate() {
            let m = |v: Vec<(f64, f64)>| v.into_iter().map(|p| at.map(p)).collect::<Vec<_>>();
            let curves = if control {
                vec![
                    Curve::Spline(m(run(&f, HINGE, 1.0, true, 0.0, 6))),
                    Curve::Line(vec![]),
                    Curve::Spline(m(run(&f, 1.0, HINGE, false, 0.0, 6))),
                    Curve::Line(vec![]),
                ]
            } else {
                vec![Curve::Spline(m(nose(&f, HINGE, 12))), Curve::Line(vec![])]
            };
            body += &format!("    feature st{k} = {plane}\n");
            body += &loop_sketch(&format!("s{k}"), &format!("plane.st{k}"), &curves);
        }
        body += "    feature surface = loft(profiles = [sketch.s0, sketch.s1])\n";
        part(name, number, "composite", comment, &body, "surface")
    }
}

fn fin() -> Surface {
    // 38° leading-edge sweep, 6.5 m tall above the root inside the tail cone.
    Surface {
        root_le: (1_700.0, 0.0, 1_500.0),
        tip_le: (1_700.0 + 6_500.0 * 0.781_286, 0.0, 8_000.0),
        root_chord: 5_600.0,
        tip_chord: 1_750.0,
        thickness: 0.11,
        vertical: true,
    }
}

fn stabilizer() -> Surface {
    // 33° sweep and 7° dihedral, root inside the tail cone.
    let span = 5_900.0;
    Surface {
        root_le: (3_500.0, 400.0, 550.0),
        tip_le: (3_500.0 + span * 0.649_408, 400.0 + span, 550.0 + span * 0.122_785),
        root_chord: 3_600.0,
        tip_chord: 1_250.0,
        thickness: 0.10,
        vertical: false,
    }
}

pub fn parts() -> Vec<(String, String)> {
    let mut out = vec![
        ("fin.part".into(), fin().piece("fin", "EMP-VT-700", "The vertical fin ahead of the rudder hinge: NACA 0011, swept 38°.", false, 1.0)),
        ("rudder.part".into(), fin().piece("rudder", "EMP-RD-705", "The rudder, aft of the hinge line at 70 % chord.", true, 1.0)),
    ];
    for (side, suffix) in [(1.0, ""), (-1.0, "_port")] {
        let s = format!("stabilizer{suffix}");
        let e = format!("elevator{suffix}");
        out.push((format!("{s}.part"), stabilizer().piece(&s, "EMP-HT-710", "A horizontal stabilizer ahead of the elevator hinge: NACA 0010, swept 33° with 7° dihedral.", false, side)));
        out.push((format!("{e}.part"), stabilizer().piece(&e, "EMP-EL-715", "An elevator, aft of the hinge line at 70 % chord.", true, side)));
    }
    out
}

pub fn empennage() -> String {
    let mut s = format!("{HEADER}// The empennage: fin and rudder, both stabilizers and their elevators, on the tail frame.\nassembly empennage {{\n");
    for (name, alias) in [
        ("fin", "fin"),
        ("rudder", "rudder"),
        ("stabilizer", "stabilizer_starboard"),
        ("elevator", "elevator_starboard"),
        ("stabilizer_port", "stabilizer_port"),
        ("elevator_port", "elevator_port"),
    ] {
        s += &use_("part", name, alias, &[]);
    }
    s += "}\n";
    s
}
