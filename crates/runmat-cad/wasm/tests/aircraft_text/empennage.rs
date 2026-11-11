//! The empennage: a swept vertical fin and rudder, and swept horizontal stabilizers with dihedral
//! and their elevators, on NACA 0010/0011 sections. Coordinates are the tail frame's: the origin
//! at the tail cone's front on the fuselage axis, x aft, y to starboard, z up.

use super::airfoil::{run, Naca, Placed};
use super::text::{loop_sketch, mirror_port, mm, part, sided, use_, Curve, HEADER, SIDE_INPUT};

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
    /// The section `t` of the way from root (0) to tip (1): its plane and placement.
    fn section_at(&self, t: f64) -> (String, Placed) {
        let lerp = |a: f64, b: f64| a + (b - a) * t;
        let le = (
            lerp(self.root_le.0, self.tip_le.0),
            lerp(self.root_le.1, self.tip_le.1),
            lerp(self.root_le.2, self.tip_le.2),
        );
        let c = lerp(self.root_chord, self.tip_chord);
        if self.vertical {
            // Horizontal planes at z: u = x, v = y.
            (
                format!("plane.offset(base = origin.xy, distance = {})", mm(le.2)),
                Placed {
                    chord: c,
                    le: (le.0, le.1),
                    chord_along_v: false,
                    flip: false,
                },
            )
        } else {
            // Vertical planes at y: u = z, v = x; a stabilizer is handed, so y changes sign with
            // its `side` input.
            (
                format!("plane.offset(base = origin.zx, distance = {})", sided(le.1)),
                Placed {
                    chord: c,
                    le: (le.2, le.0),
                    chord_along_v: true,
                    flip: false,
                },
            )
        }
    }

    /// The root-to-tip distance along the section planes' normal.
    fn span(&self) -> f64 {
        if self.vertical {
            self.tip_le.2 - self.root_le.2
        } else {
            self.tip_le.1 - self.root_le.1
        }
    }

    fn piece(&self, name: &str, number: &str, comment: &str, control: bool) -> String {
        let f = Naca::symmetric(self.thickness);
        let mut body = if self.vertical {
            String::new()
        } else {
            SIDE_INPUT.to_string()
        };
        let sketches = |body: &mut String, tag: &str, ts: [f64; 2], skin: f64| {
            for (k, t) in ts.into_iter().enumerate() {
                let (plane, at) = self.section_at(t);
                let m = |v: Vec<(f64, f64)>| v.into_iter().map(|p| at.map(p)).collect::<Vec<_>>();
                let curves = if control {
                    vec![
                        Curve::Spline(m(run(&f, HINGE, 1.0, true, 0.0, 6))),
                        Curve::Line(vec![]),
                        Curve::Spline(m(run(&f, 1.0, HINGE, false, 0.0, 6))),
                        Curve::Line(vec![]),
                    ]
                } else {
                    // The nose wrap, `skin` mm inside the surface, closed by the hinge line.
                    let d = skin / at.chord;
                    let a = HINGE - d;
                    let mut pts = run(&f, a, 0.0, true, d, 12);
                    pts.extend(run(&f, 0.0, a, false, d, 12).into_iter().skip(1));
                    vec![Curve::Spline(m(pts)), Curve::Line(vec![])]
                };
                *body += &format!("    feature {tag}st{k} = {plane}\n");
                *body += &loop_sketch(&format!("{tag}s{k}"), &format!("plane.{tag}st{k}"), &curves);
            }
        };
        sketches(&mut body, "", [0.0, 1.0], 0.0);
        body += "    feature surface = loft(profiles = [sketch.s0, sketch.s1])\n";
        if !control {
            // The fixed surface is a 3 mm skin round a hollow box with end ribs: the same section
            // 3 mm inside the skin, lofted from 3 mm inside the root to 3 mm inside the tip, is
            // cut away. The control surface aft of the hinge stays solid (its sharp trailing edge
            // leaves no room for a skin).
            let t = 3.0 / self.span();
            sketches(&mut body, "i", [t, 1.0 - t], 3.0);
            body += "    feature hollow = loft(profiles = [sketch.is0, sketch.is1], mode = cut)\n";
        }
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
        tip_le: (
            3_500.0 + span * 0.649_408,
            400.0 + span,
            550.0 + span * 0.122_785,
        ),
        root_chord: 3_600.0,
        tip_chord: 1_250.0,
        thickness: 0.10,
        vertical: false,
    }
}

pub fn parts() -> Vec<(String, String)> {
    vec![
        ("fin.part".into(), fin().piece("fin", "EMP-VT-700", "The vertical fin ahead of the rudder hinge: NACA 0011, swept 38°.", false)),
        ("rudder.part".into(), fin().piece("rudder", "EMP-RD-705", "The rudder, aft of the hinge line at 70 % chord.", true)),
        ("stabilizer.part".into(), stabilizer().piece("stabilizer", "EMP-HT-710", "A horizontal stabilizer ahead of the elevator hinge: NACA 0010, swept 33° with 7° dihedral.", false)),
        ("elevator.part".into(), stabilizer().piece("elevator", "EMP-EL-715", "An elevator, aft of the hinge line at 70 % chord.", true)),
    ]
}

pub fn empennage() -> String {
    let mut s = format!("{HEADER}// The empennage: fin and rudder, the starboard stabilizer and elevator and their mirror images to\n// port, on the tail frame.\nassembly empennage {{\n");
    for (name, alias) in [
        ("fin", "fin"),
        ("rudder", "rudder"),
        ("stabilizer", "stabilizer_starboard"),
        ("elevator", "elevator_starboard"),
    ] {
        s += &use_("part", name, alias, &[]);
    }
    s += "\n";
    s += &mirror_port(
        "stabilizer_port",
        "part",
        "stabilizer",
        &["stabilizer_starboard"],
        &[],
    );
    s += &mirror_port(
        "elevator_port",
        "part",
        "elevator",
        &["elevator_starboard"],
        &[],
    );
    s += "}\n";
    s
}
