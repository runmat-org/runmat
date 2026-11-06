//! The cabin: the floor (beams at every frame, panels, seat tracks), the sidewall linings with
//! their window reveals, the ceiling, the overhead bins, the galleys and lavatories at the doors,
//! the passenger and cargo doors, and the seat rows (three business rows, then two economy
//! blocks either side of the overwing exits).
//!
//! Cabin layout along x, in mm aft of the nose tip: cockpit to 3600; forward galley and
//! lavatory 3700 to 4560; forward doors 5250 to 6070, on the barrel just aft of the nose; business rows from 6000; economy from
//! 8900; overwing exits at 16300; aft lavatories, aft doors at 28100 and the aft galley behind
//! them.

use super::seats::{BUSINESS_AISLE, ECONOMY_AISLE};
use super::text::{mm, offset, offset_rot, outline, part, polygon, rect, use_, HEADER};
use super::{FLOOR_Z, FRAME_PITCH, RADIUS, SECTION};

/// The frames' inner radius, and the linings' inner and outer radii.
const FRAME_IN: f64 = RADIUS - 150.0;
pub const LINING_OUT: f64 = RADIUS - 172.0;
pub const LINING_IN: f64 = LINING_OUT - 8.0;
/// The sidewall lining runs from just above the floor to the bins, centred on the window belt.
const WINDOW_Z: f64 = 350.0;
/// Floor beam depth under the floor panels; floor panel thickness.
const BEAM_DEPTH: f64 = 150.0;
const PANEL: f64 = 10.0;

/// Business rows, economy rows forward and aft of the overwing exits: first row's x, count.
pub const BUSINESS_PITCH: f64 = 940.0;
pub const ECONOMY_PITCH: f64 = 790.0;
pub const BUSINESS: (f64, usize) = (6_000.0, 3);
pub const ECONOMY_FORWARD: (f64, usize) = (8_900.0, 9);
pub const ECONOMY_AFT: (f64, usize) = (17_200.0, 13);
/// Doors: (part, x of the forward edge, width, sill z, height, both sides).
const DOORS: [(&str, f64, f64, f64, f64, bool); 5] = [
    ("passenger_door", 5_250.0, 820.0, -390.0, 1_850.0, true),
    ("overwing_exit", 16_300.0, 510.0, 0.0, 920.0, true),
    ("passenger_door_aft", 28_100.0, 820.0, -390.0, 1_850.0, true),
    ("cargo_door", 9_400.0, 1_800.0, -1_620.0, 1_150.0, false),
    ("cargo_door_aft", 23_000.0, 1_800.0, -1_620.0, 1_150.0, false),
];

/// The passenger doors' spans along x, `(forward edge, aft edge)`: the cabin windows they
/// cover are left out.
pub fn passenger_door_spans() -> Vec<(f64, f64)> {
    DOORS.iter().filter(|d| d.0.starts_with("passenger_door")).map(|d| (d.1, d.1 + d.2)).collect()
}

fn half_chord(r: f64, z: f64) -> f64 {
    (r * r - z * z).max(0.0).sqrt()
}

fn floor_beam() -> String {
    let z0 = FLOOR_Z - PANEL - BEAM_DEPTH;
    let span = 2.0 * half_chord(FRAME_IN, z0) - 20.0;
    // An I section on the zx plane (u = z, v = x), 80 mm flanges, 6 mm web.
    let (t, w, f) = (8.0, 3.0, 40.0);
    let z1 = z0 + BEAM_DEPTH;
    let pts = [
        (z0, -f),
        (z0, f),
        (z0 + t, f),
        (z0 + t, w),
        (z1 - t, w),
        (z1 - t, f),
        (z1, f),
        (z1, -f),
        (z1 - t, -f),
        (z1 - t, -w),
        (z0 + t, -w),
        (z0 + t, -f),
    ];
    part(
        "floor_beam",
        "FUS-FB-400",
        "aluminum_7075",
        &format!("A floor beam: an I section {} deep across the fuselage at a frame, under the floor panels.", mm(BEAM_DEPTH)),
        &format!("{}    feature beam = extrude(profile = sketch.s, distance = {}, symmetric)\n", polygon("s", "origin.zx", &pts), mm(span)),
        "beam",
    )
}

fn floor_panel() -> String {
    let half = half_chord(FRAME_IN, FLOOR_Z) - 10.0;
    part(
        "floor_panel",
        "FUS-FP-410",
        "composite",
        "The cabin floor over one barrel section: honeycomb panels on the floor beams.",
        &format!(
            "{}    feature panel = extrude(profile = sketch.s, distance = {})\n",
            rect("s", "origin.xy", 0.0, -half, SECTION, half),
            mm(PANEL)
        ),
        "panel",
    )
}

fn seat_track() -> String {
    part(
        "seat_track",
        "FUS-TR-420",
        "aluminum_7075",
        "A seat track along one barrel section, flush with the floor panels.",
        &format!(
            "{}    feature track = extrude(profile = sketch.s, distance = {})\n",
            rect("s", "origin.yz", -14.0, 0.0, 14.0, 16.0),
            mm(SECTION)
        ),
        "track",
    )
}

/// The sidewall lining over one frame bay: a sector of the lining from just above the floor to
/// the bins, centred on the window belt, with the window's reveal cut through it.
pub fn lining_sector() -> (f64, f64) {
    let centre = (WINDOW_Z / RADIUS).asin().to_degrees();
    let low = ((FLOOR_Z + 10.0) / LINING_OUT).asin().to_degrees();
    (centre, 2.0 * (centre - low))
}

fn sidewall_lining() -> String {
    let (_, sweep) = lining_sector();
    let half = FRAME_PITCH / 2.0 - 2.0;
    // The reveal: the window's outline, 20 mm larger, along y at the sector's middle.
    let mid = (sweep / 2.0).to_radians();
    let (zc, cos) = (LINING_IN * mid.sin(), mid.cos());
    let (h, w) = ((125.0 + 25.0) * cos, 115.0 + 20.0);
    let reveal = [(zc - h, -w), (zc - h, w), (zc + h, w), (zc + h, -w)];
    part(
        "sidewall_lining",
        "CAB-LN-500",
        "polycarbonate",
        &format!("The sidewall lining over one frame bay: {sweep:.1}° of the cabin wall with the window's reveal."),
        &format!(
            "{}    feature panel = revolve(profile = sketch.s, axis = origin.x, angle = {} deg)\n{}    feature reveal = extrude(profile = sketch.r, distance = {}, mode = cut)\n",
            rect("s", "origin.xy", -half, LINING_IN, half, LINING_OUT),
            super::text::num(sweep),
            outline("r", "origin.zx", &reveal, 70.0),
            mm(RADIUS)
        ),
        "panel",
    )
}

/// The ceiling: the lining's crown between the bins over one barrel section.
const CEILING: (f64, f64) = (58.0, 64.0);

fn ceiling() -> String {
    part(
        "ceiling_panel",
        "CAB-LN-510",
        "polycarbonate",
        "The ceiling over one barrel section: the lining's crown between the two rows of bins.",
        &format!(
            "{}    feature panel = revolve(profile = sketch.s, axis = origin.x, angle = {} deg)\n",
            rect("s", "origin.xy", 0.0, LINING_IN, SECTION, LINING_OUT),
            super::text::num(CEILING.1)
        ),
        "panel",
    )
}

/// The starboard overhead bin, one module long, its back on the lining and its door curving
/// down to the aisle; the port bins are the same part turned about z.
const BIN_LENGTH: f64 = 1_590.0;

fn overhead_bin() -> String {
    let (lo, _) = (lining_sector().0 + lining_sector().1 / 2.0, 0.0);
    let top = CEILING.0;
    let at = |deg: f64, r: f64| (r * deg.to_radians().cos(), r * deg.to_radians().sin());
    let (b1, b4) = (at(lo + 0.5, LINING_IN - 4.0), at(top - 0.5, LINING_IN - 4.0));
    let pts = [b1, (b1.0 - 470.0, 1_150.0), (b1.0 - 560.0, 1_250.0), (b1.0 - 585.0, 1_400.0), b4];
    // On the yz plane (u = y, v = z), extruded along x.
    part(
        "overhead_bin",
        "CAB-BN-520",
        "polycarbonate",
        "An overhead bin module: its back on the lining, its door curving down to the aisle.",
        &format!(
            "{}    feature bin = extrude(profile = sketch.s, distance = {}, symmetric)\n",
            outline("s", "origin.yz", &pts, 40.0),
            mm(BIN_LENGTH)
        ),
        "bin",
    )
}

/// A galley or lavatory: a monument from the floor to near the ceiling, inboard of the lining,
/// `depth` long. A galley has three cart bays open to the aisle; a lavatory a door panel.
fn monument(name: &str, number: &str, comment: &str, depth: f64, galley: bool) -> String {
    let base = FLOOR_Z;
    // (y, z) outline on the yz plane; the outboard corners stay inside the lining.
    let pts = [(250.0, base), (1_700.0, base), (1_760.0, 300.0), (1_440.0, 1_000.0), (900.0, 1_480.0), (250.0, 1_480.0)];
    let mut body = format!(
        "{}    feature box = extrude(profile = sketch.s, distance = {})\n",
        polygon("s", "origin.yz", &pts),
        mm(depth)
    );
    if galley {
        // Three cart bays in the aft face (x = depth), 330 mm wide, 1030 mm tall, 600 deep.
        body += &format!("    feature face = plane.offset(base = origin.yz, distance = {})\n", mm(depth - 600.0));
        for k in 0..3 {
            let y0 = 330.0 + k as f64 * 360.0;
            body += &rect(&format!("bay{k}"), "plane.face", y0, base + 20.0, y0 + 330.0, base + 1_050.0);
            body += &format!("    feature cart{k} = extrude(profile = sketch.bay{k}, distance = 700 mm, mode = cut)\n");
        }
        // The worktop recess above the carts.
        body += &format!(
            "{}    feature worktop = extrude(profile = sketch.w, distance = 400 mm, mode = cut)\n",
            rect("w", "plane.face", 330.0, base + 1_100.0, 1_100.0, base + 1_350.0)
        );
    } else {
        // The door: a recessed panel in the aisle face (y = 250).
        body += "    feature door_at = plane.offset(base = origin.zx, distance = 245 mm)\n";
        body += &format!(
            "{}    feature door = extrude(profile = sketch.d, distance = 10 mm, mode = cut)\n",
            rect("d", "plane.door_at", base + 30.0, 80.0, base + 1_850.0, depth - 80.0)
        );
    }
    part(name, number, "polycarbonate", comment, &body, "box")
}

/// A door: the outer skin of the door, 6 mm proud of the fuselage skin, cut to a rounded outline.
/// A door on both sides is handed: its `side` input of -1 builds the port door.
fn door(name: &str, number: &str, width: f64, sill: f64, height: f64, both: bool) -> String {
    let corners = [(sill, 0.0), (sill, width), (sill + height, width), (sill + height, 0.0)];
    let comment = if both {
        format!("A door, {} wide and {} tall: its outer skin, 6 mm proud of the fuselage; to starboard, or to port in the opposite hand.", mm(width), mm(height))
    } else {
        format!("A door to starboard, {} wide and {} tall: its outer skin, 6 mm proud of the fuselage.", mm(width), mm(height))
    };
    // The cutting prism runs from the centre plane out to starboard, or from outside the port
    // skin in to the centre plane.
    let (input, from) = if both {
        (super::text::SIDE_INPUT, format!("(input.side - 1) * {}", mm((RADIUS + 200.0) / 2.0)))
    } else {
        ("", mm(0.0))
    };
    part(
        name,
        number,
        "aluminum_2024",
        &comment,
        &format!(
            "{input}{}    feature side = plane.offset(base = origin.zx, distance = {})\n{}    feature cut = extrude(profile = sketch.o, distance = {}, mode = intersect)\n",
            rect("s", "origin.xy", 0.0, RADIUS + 6.0, width, RADIUS + 9.0) + "    feature skin = revolve(profile = sketch.s, axis = origin.x, angle = 360 deg)\n",
            from,
            outline("o", "plane.side", &corners, 120.0),
            mm(RADIUS + 200.0)
        ),
        "skin",
    )
}

pub fn parts() -> Vec<(String, String)> {
    let mut out = vec![
        ("floor_beam.part".to_string(), floor_beam()),
        ("floor_panel.part".into(), floor_panel()),
        ("seat_track.part".into(), seat_track()),
        ("sidewall_lining.part".into(), sidewall_lining()),
        ("ceiling_panel.part".into(), ceiling()),
        ("overhead_bin.part".into(), overhead_bin()),
        ("galley.part".into(), monument("galley", "CAB-GL-600", "A galley: three cart bays and a worktop open to the aisle.", 860.0, true)),
        ("lavatory.part".into(), monument("lavatory", "CAB-LV-610", "A lavatory: a monument with its door on the aisle.", 900.0, false)),
    ];
    let numbers = ["FUS-DR-700", "FUS-DR-710", "FUS-DR-720", "FUS-DR-730", "FUS-DR-740"];
    for ((name, _, width, sill, height, both), number) in DOORS.iter().zip(numbers) {
        out.push((format!("{name}.part"), door(name, number, *width, *sill, *height, *both)));
    }
    out
}

/// The cabin parts one barrel section carries, for `fuselage_section.assembly`: a floor beam at
/// every frame, the floor panels and four seat tracks, a sidewall lining each side in every bay,
/// the ceiling and five bin modules each side.
/// The overhead bins in a barrel section: (the first bin's centre aft of the section's start,
/// pitch, count, half length).
pub fn bin_layout() -> (f64, f64, usize, f64) {
    (BIN_LENGTH / 2.0 + 5.0, BIN_LENGTH + 10.0, (SECTION / (BIN_LENGTH + 10.0)) as usize, BIN_LENGTH / 2.0)
}

pub fn section_uses(frames: usize, bays: usize) -> (String, String) {
    let (centre, sweep) = lining_sector();
    let mut s = String::new();
    s += &use_("part", "floor_beam", "floor_beam", &[offset(0.0, 0.0, 0.0)]);
    s += &use_("part", "floor_panel", "floor", &[offset(0.0, 0.0, FLOOR_Z - PANEL)]);
    let eco = ECONOMY_AISLE + 1.5 * super::seats::ECONOMY_SEAT;
    for (k, y) in [-eco - 450.0, -eco + 450.0, eco - 450.0, eco + 450.0].iter().enumerate() {
        s += &use_("part", "seat_track", &format!("track{k}"), &[offset(0.0, *y, FLOOR_Z - 14.0)]);
    }
    let start = centre - sweep / 2.0;
    s += &use_("part", "sidewall_lining", "lining_starboard", &[offset_rot(FRAME_PITCH / 2.0, 0.0, 0.0, [1.0, 0.0, 0.0], start)]);
    s += &use_("part", "sidewall_lining", "lining_port", &[offset_rot(FRAME_PITCH / 2.0, 0.0, 0.0, [1.0, 0.0, 0.0], 180.0 - start - sweep)]);
    s += &use_("part", "ceiling_panel", "ceiling", &[offset_rot(0.0, 0.0, 0.0, [1.0, 0.0, 0.0], CEILING.0)]);
    // Bin rows from the middle of the section, aft and forward, like the windows.
    let (first, pitch, bins, _) = bin_layout();
    let mid = bins / 2;
    for (suffix, k) in [("", mid), ("_fwd", mid - 1)] {
        let x = first + k as f64 * pitch;
        s += &use_("part", "overhead_bin", &format!("bin_starboard{suffix}"), &[offset(x, 0.0, 0.0)]);
        s += &use_("part", "overhead_bin", &format!("bin_port{suffix}"), &[offset_rot(x, 0.0, 0.0, [0.0, 0.0, 1.0], 180.0)]);
    }
    let patterns = format!(
        "    pattern floor_beams = linear(seeds = [floor_beam], direction = [1, 0, 0], spacing = input.frame_pitch, count = {frames})\n    pattern linings = linear(seeds = [lining_starboard, lining_port], direction = [1, 0, 0], spacing = input.frame_pitch, count = {bays})\n    pattern bins = linear(seeds = [bin_starboard, bin_port], direction = [1, 0, 0], spacing = {pitch}, count = {})\n    pattern bins_fwd = linear(seeds = [bin_starboard_fwd, bin_port_fwd], direction = [-1, 0, 0], spacing = {pitch}, count = {mid})\n",
        bins - mid,
        pitch = mm(pitch)
    );
    (s, patterns)
}

/// `cabin.assembly`: the seat rows, galleys, lavatories and doors, in fuselage coordinates.
pub fn cabin() -> String {
    let mut s = format!(
        "{HEADER}// The cabin: three business rows at the business pitch, nine economy rows forward of the\n// overwing exits and thirteen aft of them at the economy pitch, the galleys and lavatories at\n// the doors, and the doors.\nassembly cabin {{\n    input business_pitch : length = {}\n    input economy_pitch : length = {}\n\n",
        mm(BUSINESS_PITCH),
        mm(ECONOMY_PITCH)
    );
    let (eco, biz) = (super::seats::ECONOMY_SEAT, super::seats::BUSINESS_SEAT);
    let blocks = [
        ("business_starboard", "seat_cushion_business", BUSINESS, "business_pitch", BUSINESS_AISLE + biz / 2.0, 2, biz),
        ("business_port", "seat_cushion_business", BUSINESS, "business_pitch", -BUSINESS_AISLE - 1.5 * biz, 2, biz),
        ("economy_forward_starboard", "seat_cushion", ECONOMY_FORWARD, "economy_pitch", ECONOMY_AISLE + eco / 2.0, 3, eco),
        ("economy_forward_port", "seat_cushion", ECONOMY_FORWARD, "economy_pitch", -ECONOMY_AISLE - 2.5 * eco, 3, eco),
        ("economy_aft_starboard", "seat_cushion", ECONOMY_AFT, "economy_pitch", ECONOMY_AISLE + eco / 2.0, 3, eco),
        ("economy_aft_port", "seat_cushion", ECONOMY_AFT, "economy_pitch", -ECONOMY_AISLE - 2.5 * eco, 3, eco),
    ];
    let mut patterns = String::new();
    for (name, cushion, (x0, rows), row_pitch, y0, seats, seat_pitch) in blocks {
        let (u, p) = super::seats::Block { name, cushion, x0, y0, z: FLOOR_Z, rows, row_pitch, seats, seat_pitch }.uses();
        s += &u;
        patterns += &p;
    }
    // Monuments: the forward galley to starboard and lavatory to port ahead of the forward
    // doors, lavatories either side ahead of the aft doors and the aft galley behind them.
    let turn = |x: f64| offset_rot(x, 0.0, 0.0, [0.0, 0.0, 1.0], 180.0);
    s += &use_("part", "galley", "galley_forward", &[offset(3_700.0, 0.0, 0.0)]);
    s += &use_("part", "lavatory", "lavatory_forward", &[turn(4_600.0)]);
    s += &use_("part", "lavatory", "lavatory_aft_starboard", &[offset(27_100.0, 0.0, 0.0)]);
    s += &use_("part", "lavatory", "lavatory_aft_port", &[turn(28_000.0)]);
    s += &use_("part", "galley", "galley_aft", &[offset(29_000.0, 0.0, 0.0)]);
    s += &use_("part", "galley", "galley_aft_port", &[turn(29_860.0)]);
    for (name, x, ..) in DOORS {
        s += &use_("part", name, name, &[offset(x, 0.0, 0.0)]);
    }
    // The port doors: mirror images of the starboard ones, in their opposite hand.
    for (name, .., both) in DOORS {
        if both {
            s += &super::text::mirror_port(&format!("{name}_port"), "part", name, &[name], &[]);
        }
    }
    s += "\n";
    s += &patterns;
    s += "}\n";
    s
}
