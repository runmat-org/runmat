//! Below the cabin floor: the cargo holds' floors with their containers (four forward of the
//! wing, four aft of the main gear bays, a bulk area behind them), the centre wing box through
//! the fuselage, the two main landing gear bays behind it and the holds' sidewall liners.

use super::text::{mm, num, offset, offset_rot, part, polygon, rect, use_, HEADER};
use super::wing::{self, FRONT, REAR, ROOT_STATION, ROOT_X, ROOT_Z};
use super::{FLOOR_Z, RADIUS};

/// The cargo floor's top, and the holds: (x of the first container, containers).
const CARGO_Z: f64 = -1_690.0;
const CONTAINER: f64 = 1_560.0;
const HOLDS: [(&str, f64, usize); 2] = [("forward", 6_300.0, 4), ("aft", 19_900.0, 4)];
/// The cargo floors' extents along x.
const CARGO_FLOORS: [(f64, f64); 2] = [(6_100.0, 12_900.0), (19_700.0, 27_600.0)];
const BAY: (f64, f64) = (17_400.0, 19_600.0);

fn cargo_floor(name: &str, number: &str, length: f64) -> String {
    let half = (((RADIUS - 150.0).powi(2) - CARGO_Z.powi(2)).sqrt() - 10.0).max(300.0);
    part(
        name,
        number,
        "composite",
        "A cargo hold floor with its roller tracks, the length of one hold.",
        &format!(
            "    feature datum = plane.offset(base = origin.xy, distance = {})\n{}    feature floor = extrude(profile = sketch.s, distance = 12 mm)\n",
            mm(CARGO_Z - 12.0),
            rect("s", "plane.datum", 0.0, -half, length, half)
        ),
        "floor",
    )
}

/// The hold liner's sector, from the cargo floor up to under the floor beams, at the cabin
/// lining's radius: (start angle above the horizontal, sweep), in degrees.
fn liner_sector() -> (f64, f64) {
    let r = super::cabin::LINING_IN;
    let low = (CARGO_Z / r).asin().to_degrees();
    let high = ((FLOOR_Z - 170.0) / r).asin().to_degrees();
    (low, high - low)
}

/// A hold's sidewall liner, one side, the length of the hold: an 8 mm panel inside the frames.
fn liner(name: &str, number: &str, length: f64) -> String {
    let (_, sweep) = liner_sector();
    part(
        name,
        number,
        "composite",
        "A cargo hold's sidewall liner, one side, inside the frames from the cargo floor to the floor beams.",
        &format!(
            "{}    feature panel = revolve(profile = sketch.s, axis = origin.x, angle = {} deg)\n",
            rect("s", "origin.xy", 0.0, super::cabin::LINING_IN, length, super::cabin::LINING_OUT),
            num(sweep)
        ),
        "panel",
    )
}

/// A lower-deck container: the full-width shape with its sloped lower corners, 1530 mm long.
fn container() -> String {
    let top = FLOOR_Z - 10.0 - 150.0 - 40.0;
    let pts = [
        (-620.0, CARGO_Z),
        (620.0, CARGO_Z),
        (1_080.0, CARGO_Z + 520.0),
        (1_080.0, top),
        (-1_080.0, top),
        (-1_080.0, CARGO_Z + 520.0),
    ];
    part(
        "cargo_container",
        "FUS-UL-810",
        "aluminum_2024",
        "A lower-deck container: a sloped base for the hold's curve, full width, 1530 mm long.",
        &format!(
            "{}    feature box = extrude(profile = sketch.s, distance = 1530 mm)\n",
            polygon("s", "origin.yz", &pts)
        ),
        "box",
    )
}

/// The wing box's height at the side of the fuselage: the root station's section between the
/// spars, in aircraft z.
fn box_heights() -> (f64, f64) {
    let f = wing::section(ROOT_STATION);
    let c = wing::chord(ROOT_STATION);
    let z0 = ROOT_Z + wing::z_le(ROOT_STATION);
    let up = f.at(FRONT, true).1.min(f.at(REAR, true).1);
    let low = f.at(FRONT, false).1.max(f.at(REAR, false).1);
    (z0 + low * c + 4.0, z0 + up * c - 4.0)
}

/// The spar stations at the side of the fuselage, in aircraft x.
pub fn box_x() -> (f64, f64) {
    let c = wing::chord(ROOT_STATION);
    let x0 = ROOT_X + wing::x_le(ROOT_STATION);
    (x0 + FRONT * c, x0 + REAR * c)
}

/// The centre wing box: a closed box between the spars across the fuselage, 25 mm walls.
fn centre_box() -> String {
    let (x0, x1) = box_x();
    let (z0, z1) = box_heights();
    let w = 25.0;
    part(
        "centre_wing_box",
        "WNG-CB-500",
        "aluminum_7075",
        "The centre wing box: the spars and skins carried across the fuselage between the wing roots.",
        &format!(
            "    feature datum = plane.offset(base = origin.yz, distance = {})\n{}    feature box = extrude(profile = sketch.s, distance = {})\n    feature in_at = plane.offset(base = origin.yz, distance = {})\n{}    feature cell = extrude(profile = sketch.c, distance = {}, mode = cut)\n",
            mm(x0),
            rect("s", "plane.datum", -ROOT_STATION, z0, ROOT_STATION, z1),
            mm(x1 - x0),
            mm(x0 + w),
            rect("c", "plane.in_at", -ROOT_STATION + w, z0 + w, ROOT_STATION - w, z1 - w),
            mm(x1 - x0 - 2.0 * w)
        ),
        "box",
    )
}

/// A main gear bay: a roof and walls, open below, where the leg's wheels stow.
fn gear_bay() -> String {
    let (x0, x1) = BAY;
    let (y, z0, z1) = (1_050.0, -1_550.0, FLOOR_Z - 170.0);
    let w = 15.0;
    part(
        "gear_bay",
        "FUS-GB-820",
        "aluminum_2024",
        "The main gear bays: a roof, side walls and pressure walls fore and aft, open below.",
        &format!(
            "    feature datum = plane.offset(base = origin.yz, distance = {})\n{}    feature bay = extrude(profile = sketch.s, distance = {})\n    feature in_at = plane.offset(base = origin.yz, distance = {})\n{}    feature well = extrude(profile = sketch.c, distance = {}, mode = cut)\n",
            mm(x0),
            rect("s", "plane.datum", -y, z0, y, z1),
            mm(x1 - x0),
            mm(x0 + w),
            rect("c", "plane.in_at", -y + w, z0 - 1.0, y - w, z1 - w),
            mm(x1 - x0 - 2.0 * w)
        ),
        "bay",
    )
}

pub fn parts() -> Vec<(String, String)> {
    let mut out = vec![
        ("cargo_container.part".to_string(), container()),
        ("centre_wing_box.part".into(), centre_box()),
        ("gear_bay.part".into(), gear_bay()),
    ];
    let [(a0, b0), (a1, b1)] = CARGO_FLOORS;
    out.push((
        "cargo_floor.part".into(),
        cargo_floor("cargo_floor", "FUS-CF-800", b0 - a0),
    ));
    out.push((
        "cargo_floor_aft.part".into(),
        cargo_floor("cargo_floor_aft", "FUS-CF-801", b1 - a1),
    ));
    out.push((
        "hold_liner.part".into(),
        liner("hold_liner", "FUS-CL-810", b0 - a0),
    ));
    out.push((
        "hold_liner_aft.part".into(),
        liner("hold_liner_aft", "FUS-CL-811", b1 - a1),
    ));
    out
}

/// `hold.assembly`, in fuselage coordinates.
pub fn hold() -> String {
    let mut s = format!(
        "{HEADER}// Below the cabin floor: the cargo floors and containers, the centre wing box and the main\n// gear bays.\nassembly hold {{\n    input container_pitch : length = {}\n\n",
        mm(CONTAINER)
    );
    s += &use_(
        "part",
        "cargo_floor",
        "cargo_floor_forward",
        &[offset(CARGO_FLOORS[0].0, 0.0, 0.0)],
    );
    s += &use_(
        "part",
        "cargo_floor_aft",
        "cargo_floor_aft",
        &[offset(CARGO_FLOORS[1].0, 0.0, 0.0)],
    );
    for (name, x, _) in HOLDS {
        s += &use_(
            "part",
            "cargo_container",
            &format!("container_{name}"),
            &[offset(x, 0.0, 0.0)],
        );
    }
    let (start, sweep) = liner_sector();
    for ((a, _), part_name) in CARGO_FLOORS.iter().zip(["hold_liner", "hold_liner_aft"]) {
        s += &use_(
            "part",
            part_name,
            &format!("{part_name}_starboard"),
            &[offset_rot(*a, 0.0, 0.0, [1.0, 0.0, 0.0], start)],
        );
        s += &use_(
            "part",
            part_name,
            &format!("{part_name}_port"),
            &[offset_rot(
                *a,
                0.0,
                0.0,
                [1.0, 0.0, 0.0],
                180.0 - start - sweep,
            )],
        );
    }
    s += &use_("part", "centre_wing_box", "centre_wing_box", &[]);
    s += &use_("part", "gear_bay", "gear_bays", &[]);
    s += "\n";
    for (name, _, n) in HOLDS {
        s += &format!("    pattern containers_{name} = linear(seeds = [container_{name}], direction = [1, 0, 0], spacing = input.container_pitch, count = {n})\n");
    }
    s += "}\n";
    s
}
