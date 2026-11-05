//! The airliner's files: a generic twin-engine narrow-body, 38 m long with a 35 m span.
//! Lengths are written in millimetres; the aircraft's x runs aft from the nose, y to starboard
//! and z up.
//!
//! - `airliner.assembly`: the skeleton, the fuselage, both wings, the empennage and the landing
//!   gear, each attached to a frame `airliner_skeleton.part` publishes.
//! - `fuselage.assembly`: the drooped nose and its six cockpit panes, three barrel sections, the tail cone and
//!   the wing-to-body fairing.
//! - `fuselage_section.assembly`: frames at the frame pitch, stringers, skin panels, flush rivet
//!   rings and cabin windows, and the cabin's floor beams, floor, seat tracks, linings, ceiling
//!   and overhead bins.
//! - `cockpit.assembly`, `cabin.assembly`, `hold.assembly`: the flight deck, the seat rows,
//!   galleys, lavatories and doors, and the cargo holds, centre wing box and gear bays.
//! - `wing_starboard.assembly`, `wing_port.assembly`: lofted slats, box skins, spars, ribs at the
//!   rib pitch, flaps, aileron and winglet on cambered sections, rivet rows along the spar caps,
//!   and an engine (the port wing uses the opposite-hand pieces).
//! - `empennage.assembly`: fin, rudder, stabilizers and elevators.
//! - `nacelle.assembly`, `main_gear.assembly`, `nose_gear.assembly`, `rivet_ring.assembly`.

mod airfoil;
mod cabin;
mod cockpit;
mod empennage;
mod fuselage;
mod hold;
mod nacelle_gear;
mod nose;
mod seats;
mod text;
mod wing;

use text::{attach, mm, use_, HEADER};

/// Fuselage radius, nose and tail cone lengths, barrel section length (mm).
const RADIUS: f64 = 1_980.0;
const NOSE: f64 = 5_200.0;
const SECTION: f64 = 8_000.0;
const TAIL: f64 = 8_800.0;
const FRAME_PITCH: f64 = 500.0;
const RIB_PITCH: f64 = 650.0;
const STRINGERS: usize = 24;
const SKIN_PANELS: usize = 8;
const RIVET_PITCH: f64 = 150.0;
/// The cabin and flight deck floor's top, below the fuselage axis.
const FLOOR_Z: f64 = -400.0;

/// The skeleton's frames: where each major assembly's origin sits.
fn frames() -> Vec<(&'static str, f64, f64, f64)> {
    vec![
        ("fuselage", 0.0, 0.0, 0.0),
        ("wing_root_starboard", wing::ROOT_X, 0.0, wing::ROOT_Z),
        ("wing_root_port", wing::ROOT_X, 0.0, wing::ROOT_Z),
        ("tail", NOSE + 3.0 * SECTION, 0.0, 0.0),
        ("main_gear_starboard", 18_300.0, 3_000.0, -1_150.0),
        ("main_gear_port", 18_300.0, -3_000.0, -1_150.0),
        ("nose_gear", 4_600.0, 0.0, -1_700.0),
    ]
}

fn skeleton() -> String {
    let mut s = format!(
        "{HEADER}// The airliner's skeleton: its main sizes and the frames its assemblies attach to.\npart airliner_skeleton {{\n    input span : length = {}\n    input root_chord : length = {}\n    input tip_chord : length = {}\n    input fuselage_radius : length = {}\n    input frame_pitch : length = {}\n    input rib_pitch : length = {}\n\n",
        mm(2.0 * wing::SEMISPAN),
        mm(wing::ROOT_CHORD),
        mm(wing::TIP_CHORD),
        mm(RADIUS),
        mm(FRAME_PITCH),
        mm(RIB_PITCH)
    );
    for (name, x, y, z) in frames() {
        s += &format!(
            "    feature {name} = frame(frame = at, origin = point({}, {}, {}), z = origin.z, x = origin.x)\n",
            mm(x),
            mm(y),
            mm(z)
        );
    }
    s += "\n";
    for (name, ..) in frames() {
        s += &format!("    publish {name} = datum(datum = {{ datum = feature, feature = {name} }})\n");
    }
    for input in ["span", "root_chord", "tip_chord", "fuselage_radius", "frame_pitch", "rib_pitch"] {
        s += &format!("    publish {input} = value(input = {input})\n");
    }
    s += "}\n";
    s
}

fn airliner() -> String {
    let mut s = format!(
        "{HEADER}// A generic twin-engine narrow-body airliner, 38 m long with a 35 m span.\nassembly airliner {{\n    input span : length = {}\n    input frame_pitch : length = {}\n    input rib_pitch : length = {}\n\n",
        mm(2.0 * wing::SEMISPAN),
        mm(FRAME_PITCH),
        mm(RIB_PITCH)
    );
    s += &use_("part", "airliner_skeleton", "skeleton", &[]);
    s += &use_("assembly", "fuselage", "fuselage", &[attach("fuselage")]);
    for side in ["starboard", "port"] {
        s += &use_("assembly", &format!("wing_{side}"), &format!("wing_{side}"), &[attach(&format!("wing_root_{side}"))]);
    }
    s += &use_("assembly", "empennage", "empennage", &[attach("tail")]);
    s += &use_("assembly", "main_gear", "main_gear_starboard", &[attach("main_gear_starboard")]);
    s += &use_("assembly", "main_gear", "main_gear_port", &[attach("main_gear_port")]);
    s += &use_("assembly", "nose_gear", "nose_gear", &[attach("nose_gear")]);
    // The view it opens on: three-quarters from ahead, a little below, so both engines, both
    // wings and the gear read.
    s += "\n    view hero = { from = [-0.68, -0.62, -0.26], up = [0, 0, 1], default = true }\n";
    s += "}\n";
    s
}

/// Every file of the airliner, by file name.
pub fn files() -> Vec<(String, String)> {
    let mut out = vec![("airliner_skeleton.part".to_string(), skeleton())];
    out.extend(nose::parts());
    out.extend(fuselage::parts());
    out.extend(cabin::parts());
    out.extend(seats::parts());
    out.extend(cockpit::parts());
    out.extend(hold::parts());
    out.push(("cabin.assembly".into(), cabin::cabin()));
    out.push(("cockpit.assembly".into(), cockpit::cockpit()));
    out.push(("hold.assembly".into(), hold::hold()));
    out.extend(wing::parts());
    out.extend(empennage::parts());
    out.extend(nacelle_gear::parts());
    out.push(("rivet_ring.assembly".into(), fuselage::rivet_ring()));
    out.push(("fuselage_section.assembly".into(), fuselage::fuselage_section()));
    out.push(("fuselage.assembly".into(), fuselage::fuselage()));
    out.push(("wing_starboard.assembly".into(), wing::wing("wing_starboard", 1.0)));
    out.push(("wing_port.assembly".into(), wing::wing("wing_port", -1.0)));
    out.push(("empennage.assembly".into(), empennage::empennage()));
    out.push(("nacelle.assembly".into(), nacelle_gear::nacelle()));
    out.push((
        "main_gear.assembly".into(),
        nacelle_gear::gear("main_gear", "A main landing gear leg: oleo strut, axle and two wheels.", 2_500.0, 115.0, 440.0, 170.0, 840.0),
    ));
    out.push((
        "nose_gear.assembly".into(),
        nacelle_gear::gear("nose_gear", "The nose landing gear leg: oleo strut, axle and two wheels.", 2_150.0, 80.0, 270.0, 110.0, 520.0),
    ));
    out.push(("airliner.assembly".into(), airliner()));
    out
}
