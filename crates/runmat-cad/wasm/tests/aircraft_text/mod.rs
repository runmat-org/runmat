//! The airliner's files. Lengths are written in millimetres; the aircraft's x runs aft from
//! the nose, y to starboard and z up.
//!
//! - `airliner.assembly`: the skeleton, the fuselage, both wings (the port wing is the
//!   starboard wing's opposite-hand mirror), the empennage and the landing gear, placed on the
//!   frames `airliner_skeleton.part` publishes.
//! - `fuselage.assembly`: the nose, three barrel sections and the tail cone.
//! - `fuselage_section.assembly`: frames at the frame pitch, stringers and skin panels around,
//!   and a ring of rivets at every frame.
//! - `wing_starboard.assembly`, `wing_port.assembly`: spars, ribs at the rib pitch, upper and
//!   lower skins, rivet rows along the spar caps, and a nacelle.
//! - `nacelle.assembly`, `gear.assembly`, `rivet_ring.assembly`: the repeated subassemblies.

/// Semispan, chord and the other main sizes (mm), shared with the skeleton's inputs.
const SEMISPAN: f64 = 15_000.0;
const CHORD: f64 = 3_600.0;
const RADIUS: f64 = 1_950.0;
const SECTION: f64 = 8_000.0;
const FRAME_PITCH: f64 = 500.0;
const RIB_PITCH: f64 = 600.0;
const STRINGERS: usize = 24;
const RIVET_PITCH: f64 = 150.0;
/// Chordwise skin panels on each wing surface.
const SKIN_PANELS: usize = 8;

fn mm(v: f64) -> String {
    let r = (v * 1000.0).round() / 1000.0;
    if r == r.trunc() {
        format!("{} mm", r as i64)
    } else {
        format!("{r} mm")
    }
}

fn pt(u: f64, v: f64) -> String {
    format!("[{}, {}]", mm(u), mm(v))
}

const HEADER: &str = "runmat 0.7\nstd 1\n\n";

/// A closed polygon sketch: `l1 … ln` through the points.
fn polygon(name: &str, plane: &str, pts: &[(f64, f64)]) -> String {
    let mut s = format!("    sketch {name} on {plane} {{\n");
    for i in 0..pts.len() {
        let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
        s += &format!(
            "        l{} = line(start = {}, end = {})\n",
            i + 1,
            pt(a.0, a.1),
            pt(b.0, b.1)
        );
    }
    s += "    }\n";
    s
}

fn rect(name: &str, plane: &str, u0: f64, v0: f64, u1: f64, v1: f64) -> String {
    polygon(name, plane, &[(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
}

/// A symmetric biconvex section of `chord` and `thickness`, leading edge at `(u0, 0)`, chord
/// along +u: an upper arc and a lower arc.
fn lens(name: &str, plane: &str, u0: f64, chord: f64, thickness: f64) -> String {
    let h = thickness / 2.0;
    let r = (chord * chord / 4.0 + h * h) / (2.0 * h);
    let c = u0 + chord / 2.0;
    let (le, te) = (pt(u0, 0.0), pt(u0 + chord, 0.0));
    format!(
        "    sketch {name} on {plane} {{\n        upper = arc(center = {}, start = {te}, end = {le})\n        lower = arc(center = {}, start = {le}, end = {te})\n    }}\n",
        pt(c, h - r),
        pt(c, r - h)
    )
}

fn part(name: &str, number: &str, material: &str, body: &str, last: &str) -> String {
    format!(
        "{HEADER}part {name} {{\n{body}\n    output {name} = {{ body = feature.{last}.body, part_number = \"{number}\", material = \"{material}\" }}\n}}\n"
    )
}

fn offset(x: f64, y: f64, z: f64) -> String {
    format!(
        "placement = {{ placement = offset, offset = {{ translation = [{}, {}, {}] }} }}",
        mm(x),
        mm(y),
        mm(z)
    )
}

fn offset_rot(x: f64, y: f64, z: f64, axis: [i32; 3], deg: f64) -> String {
    format!(
        "placement = {{ placement = offset, offset = {{ translation = [{}, {}, {}], rotation = {{ rotation = axis_angle, axis = [{}, {}, {}], angle = {} deg }} }} }}",
        mm(x),
        mm(y),
        mm(z),
        axis[0],
        axis[1],
        axis[2],
        (deg * 1000.0).round() / 1000.0 + 0.0
    )
}

fn attach(frame: &str) -> String {
    format!(
        "placement = {{ placement = attach, own = {{ anchor = origin }}, to = {{ frame = interface, file = \"airliner_skeleton.part\", name = {frame} }} }}"
    )
}

fn skeleton() -> String {
    // Published frames: where each major assembly's origin sits.
    let frames: [(&str, f64, f64, f64); 6] = [
        ("wing_root_starboard", 12_400.0, 1_200.0, -900.0),
        ("wing_root_port", 12_400.0, -1_200.0, -900.0),
        ("fin_root", 30_200.0, 0.0, 1_200.0),
        ("main_gear_starboard", 15_200.0, 2_900.0, -1_500.0),
        ("main_gear_port", 15_200.0, -2_900.0, -1_500.0),
        ("nose_gear", 3_400.0, 0.0, -1_500.0),
    ];
    let mut s = format!(
        "{HEADER}// The airliner's skeleton: its main sizes and the frames its assemblies attach to.\npart airliner_skeleton {{\n    input span : length = {}\n    input chord : length = {}\n    input fuselage_radius : length = {}\n    input frame_pitch : length = {}\n    input rib_pitch : length = {}\n\n",
        mm(2.0 * (SEMISPAN + 1_200.0)),
        mm(CHORD),
        mm(RADIUS),
        mm(FRAME_PITCH),
        mm(RIB_PITCH)
    );
    for (name, x, y, z) in frames {
        s += &format!(
            "    feature {name} = frame(frame = at, origin = point({}, {}, {}), z = origin.z, x = origin.x)\n",
            mm(x),
            mm(y),
            mm(z)
        );
    }
    s += "\n";
    for (name, ..) in frames {
        s += &format!(
            "    publish {name} = datum(datum = {{ datum = feature, feature = {name} }})\n"
        );
    }
    for input in [
        "span",
        "chord",
        "fuselage_radius",
        "frame_pitch",
        "rib_pitch",
    ] {
        s += &format!("    publish {input} = value(input = {input})\n");
    }
    s += "}\n";
    s
}

fn fuselage_parts() -> Vec<(String, String)> {
    let r = RADIUS;
    vec![
        (
            "frame.part".into(),
            part(
                "frame",
                "FUS-FR-100",
                "aluminum_7075",
                &format!(
                    "    // A fuselage frame: a ring 147 mm deep and 40 mm wide.\n{}    feature ring = revolve(profile = sketch.s, axis = origin.x, angle = 360 deg)\n",
                    rect("s", "origin.xy", 0.0, r - 150.0, 40.0, r - 3.0)
                ),
                "ring",
            ),
        ),
        (
            "stringer.part".into(),
            part(
                "stringer",
                "FUS-ST-200",
                "aluminum_7075",
                &format!(
                    "    // A stringer the length of one barrel section, under the skin at the crown.\n{}    feature body = extrude(profile = sketch.s, distance = {})\n",
                    rect("s", "origin.yz", -12.0, r - 43.0, 12.0, r - 3.0),
                    mm(SECTION)
                ),
                "body",
            ),
        ),
        (
            "skin_panel.part".into(),
            part(
                "skin_panel",
                "FUS-SK-300",
                "aluminum_2024",
                &format!(
                    "    // A skin panel: 15° of the barrel, 3 mm thick.\n{}    feature panel = revolve(profile = sketch.s, axis = origin.x, angle = 15 deg)\n",
                    rect("s", "origin.xy", 0.0, r - 3.0, SECTION, r)
                ),
                "panel",
            ),
        ),
        (
            "nose.part".into(),
            part(
                "nose",
                "FUS-NS-010",
                "composite",
                &format!(
                    "    // The nose: an ogive turned about the fuselage axis, tangent to the barrel at x = 4 m.\n    sketch s on origin.xy {{\n        l1 = line(start = {}, end = {})\n        l2 = line(start = {}, end = {})\n        a1 = arc(center = {}, start = {}, end = {})\n    }}\n    feature dome = revolve(profile = sketch.s, axis = origin.x, angle = 360 deg)\n",
                    pt(0.0, 0.0),
                    pt(4_000.0, 0.0),
                    pt(4_000.0, 0.0),
                    pt(4_000.0, r),
                    pt(4_000.0, r - (4_000.0 * 4_000.0 + r * r) / (2.0 * r)),
                    pt(4_000.0, r),
                    pt(0.0, 0.0)
                ),
                "dome",
            ),
        ),
        (
            "tail_cone.part".into(),
            part(
                "tail_cone",
                "FUS-TC-020",
                "aluminum_2024",
                &format!(
                    "    // The tail cone: the barrel closing to the APU exhaust.\n{}    feature cone = revolve(profile = sketch.s, axis = origin.x, angle = 360 deg)\n",
                    polygon("s", "origin.xy", &[(0.0, 0.0), (6_000.0, 0.0), (6_000.0, 450.0), (0.0, r)])
                ),
                "cone",
            ),
        ),
        (
            "rivet.part".into(),
            part(
                "rivet",
                "NAS1097-6",
                "aluminum_2117",
                &format!(
                    "    // A universal-head rivet, axis along z, head on top.\n{}    feature body = revolve(profile = sketch.s, axis = origin.z, angle = 360 deg)\n",
                    polygon(
                        "s",
                        "origin.yz",
                        &[(0.0, -16.0), (3.0, -16.0), (3.0, 0.0), (5.0, 0.0), (5.0, 3.0), (0.0, 3.0)]
                    )
                ),
                "body",
            ),
        ),
    ]
}

fn wing_parts() -> Vec<(String, String)> {
    let t = 0.12 * CHORD;
    let front = 0.15 * CHORD;
    let rear = 0.65 * CHORD;
    // Half depth of the section at chord station x (upper arc).
    let h = t / 2.0;
    let rr = (CHORD * CHORD / 4.0 + h * h) / (2.0 * h);
    let depth = |x: f64| (rr * rr - (x - CHORD / 2.0).powi(2)).sqrt() - (rr - h);
    let sweep = skin_sweep() / SKIN_PANELS as f64;
    vec![
        (
            "rib.part".into(),
            part(
                "rib",
                "WNG-RB-400",
                "aluminum_7075",
                &format!(
                    "    // A wing rib: the full biconvex section, 20 mm thick.\n{}    feature web = extrude(profile = sketch.s, distance = 20 mm)\n",
                    zx(&lens("s", "origin.zx", 0.0, CHORD, t))
                ),
                "web",
            ),
        ),
        (
            "spar.part".into(),
            part(
                "spar",
                "WNG-SP-500",
                "aluminum_7075",
                &format!(
                    "    // A spar web the depth of the section at its chord station; `input.station` is the\n    // fraction of chord.\n{}    feature web = extrude(profile = sketch.s, distance = {})\n",
                    rect("s", "origin.zx", -depth(front) + 6.0, front - 15.0, depth(front) - 6.0, front + 15.0),
                    mm(SEMISPAN)
                ),
                "web",
            ),
        ),
        (
            "rear_spar.part".into(),
            part(
                "rear_spar",
                "WNG-SP-510",
                "aluminum_7075",
                &format!(
                    "    // The rear spar web at 65 % chord.\n{}    feature web = extrude(profile = sketch.s, distance = {})\n",
                    rect("s", "origin.zx", -depth(rear) + 6.0, rear - 15.0, depth(rear) - 6.0, rear + 15.0),
                    mm(SEMISPAN)
                ),
                "web",
            ),
        ),
        (
            "wing_skin_upper.part".into(),
            part(
                "wing_skin_upper",
                "WNG-SK-600",
                "aluminum_2024",
                &format!(
                    "    // The upper wing skin, 4 mm thick, root to tip: the section's upper arc turned\n    // about its centre (the part origin) through {sweep:.3}°: one of 8 chordwise panels.\n{}    feature skin = revolve(profile = sketch.s, axis = origin.y, angle = {sweep:.3} deg)\n",
                    rect("s", "origin.yz", 0.0, rr - 4.0, SEMISPAN, rr)
                ),
                "skin",
            ),
        ),
        (
            "wing_skin_lower.part".into(),
            part(
                "wing_skin_lower",
                "WNG-SK-610",
                "aluminum_2024",
                &format!(
                    "    // The lower wing skin, 4 mm thick, root to tip: the section's lower arc turned\n    // about its centre (the part origin) through {sweep:.3}°: one of 8 chordwise panels.\n{}    feature skin = revolve(profile = sketch.s, axis = origin.y, angle = {sweep:.3} deg)\n",
                    rect("s", "origin.yz", 0.0, -rr, SEMISPAN, -rr + 4.0)
                ),
                "skin",
            ),
        ),
    ]
}

/// The angle (degrees) one skin arc of the wing section spans about its centre.
fn skin_sweep() -> f64 {
    let h = 0.06 * CHORD;
    let rr = (CHORD * CHORD / 4.0 + h * h) / (2.0 * h);
    2.0 * (CHORD / 2.0 / rr).asin().to_degrees()
}

/// Sketch text written in (x, z) section coordinates, moved to the zx plane's (z, x) order.
fn zx(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(i) = rest.find('[') {
        out += &rest[..i];
        let j = rest[i..].find(']').expect("closing bracket") + i;
        let inner = &rest[i + 1..j];
        let parts: Vec<&str> = inner.split(", ").collect();
        if parts.len() == 2 {
            out += &format!("[{}, {}]", parts[1], parts[0]);
        } else {
            out += &rest[i..=j];
        }
        rest = &rest[j + 1..];
    }
    out += rest;
    // Swapping u and v mirrors the plane: an arc keeps its side by swapping its ends.
    out.lines()
        .map(|l| match (l.find("start = "), l.find(", end = ")) {
            (Some(a), Some(b)) if l.contains("arc(") => {
                let start = &l[a + 8..b];
                let end = &l[b + 8..l.len() - 1];
                format!("{}start = {end}, end = {start})", &l[..a])
            }
            _ => l.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

fn other_parts() -> Vec<(String, String)> {
    vec![
        (
            "fin.part".into(),
            part(
                "fin",
                "EMP-VT-700",
                "composite",
                &format!(
                    "    // The vertical fin: a biconvex section 4.2 m long, 5.6 m tall.\n{}    feature fin = extrude(profile = sketch.s, distance = 5600 mm)\n",
                    lens("s", "origin.xy", 0.0, 4_200.0, 420.0)
                ),
                "fin",
            ),
        ),
        (
            "stabilizer.part".into(),
            part(
                "stabilizer",
                "EMP-HT-710",
                "composite",
                &format!(
                    "    // One horizontal stabilizer: a biconvex section 3 m long, 5.6 m out.\n{}    feature stab = extrude(profile = sketch.s, distance = 5600 mm)\n",
                    zx(&lens("s", "origin.zx", 0.0, 3_000.0, 300.0))
                ),
                "stab",
            ),
        ),
        (
            "gear_strut.part".into(),
            part(
                "gear_strut",
                "LG-ST-800",
                "steel_300m",
                "    // The oleo strut, 1.4 m long; the gear assembly hangs it from the trunnion.\n    sketch s on origin.xy {\n        c = circle(center = [0 mm, 0 mm], radius = 90 mm)\n    }\n    feature strut = extrude(profile = sketch.s, distance = 1400 mm)\n",
                "strut",
            ),
        ),
        (
            "wheel.part".into(),
            part(
                "wheel",
                "LG-WH-810",
                "rubber",
                "    // A tyre: a 200 mm section turned on a 420 mm radius about the axle (y).\n    sketch s on origin.xy {\n        c = circle(center = [420 mm, 0 mm], radius = 200 mm)\n    }\n    feature tyre = revolve(profile = sketch.s, axis = origin.y, angle = 360 deg)\n",
                "tyre",
            ),
        ),
        (
            "axle.part".into(),
            part(
                "axle",
                "LG-AX-820",
                "steel_300m",
                "    // The axle across both wheels.\n    sketch s on origin.zx {\n        c = circle(center = [0 mm, 0 mm], radius = 60 mm)\n    }\n    feature axle = extrude(profile = sketch.s, distance = 900 mm, symmetric)\n",
                "axle",
            ),
        ),
        (
            "cowl.part".into(),
            part(
                "cowl",
                "NAC-CW-900",
                "composite",
                &format!(
                    "    // The nacelle cowl: an annulus 3.6 m long, lip radius 1 m, turned about x.\n{}    feature cowl = revolve(profile = sketch.s, axis = origin.x, angle = 360 deg)\n",
                    polygon(
                        "s",
                        "origin.xy",
                        &[(0.0, 900.0), (3_600.0, 780.0), (3_600.0, 840.0), (1_200.0, 1_060.0), (0.0, 1_000.0)]
                    )
                ),
                "cowl",
            ),
        ),
        (
            "fan.part".into(),
            part(
                "fan",
                "NAC-FN-910",
                "titanium",
                "    // The fan disk behind the inlet.\n    sketch s on origin.yz {\n        c = circle(center = [0 mm, 0 mm], radius = 880 mm)\n    }\n    feature disk = extrude(profile = sketch.s, distance = 120 mm)\n",
                "disk",
            ),
        ),
        (
            "spinner.part".into(),
            part(
                "spinner",
                "NAC-SP-920",
                "composite",
                "    // The spinner cone at the fan's hub.\n    sketch s on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [600 mm, 0 mm])\n        l2 = line(start = [600 mm, 0 mm], end = [600 mm, 300 mm])\n        l3 = line(start = [600 mm, 300 mm], end = [0 mm, 0 mm])\n    }\n    feature cone = revolve(profile = sketch.s, axis = origin.x, angle = 360 deg)\n",
                "cone",
            ),
        ),
        (
            "pylon.part".into(),
            part(
                "pylon",
                "NAC-PY-930",
                "titanium",
                &format!(
                    "    // The pylon from the cowl's crown up to the wing's lower skin.\n{}    feature pylon = extrude(profile = sketch.s, distance = 300 mm, symmetric)\n",
                    zx(&polygon("s", "origin.zx", &[(400.0, 950.0), (3_400.0, 880.0), (3_400.0, 1_850.0), (1_400.0, 1_850.0)]))
                ),
                "pylon",
            ),
        ),
    ]
}

fn rivet_ring() -> String {
    format!(
        "{HEADER}// A ring of rivets through the skin onto one frame, one at each stringer.\nassembly rivet_ring {{\n    use part rivet from \"rivet.part\" as rivet {{\n        {}\n    }}\n    pattern around = circular(seeds = [rivet], axis_point = [0 mm, 0 mm, 0 mm], axis = [1, 0, 0], count = {STRINGERS}, angle = 360 deg)\n}}\n",
        offset(20.0, 0.0, RADIUS)
    )
}

fn fuselage_section() -> String {
    let frames = (SECTION / FRAME_PITCH) as usize + 1;
    format!(
        "{HEADER}// One barrel section: {frames} frames at the frame pitch, {STRINGERS} stringers, 24 skin panels\n// and a ring of rivets at every frame.\nassembly fuselage_section {{\n    input frame_pitch : length = {}\n\n    use part frame from \"frame.part\" as frame\n    use part stringer from \"stringer.part\" as stringer\n    use part skin_panel from \"skin_panel.part\" as panel\n    use assembly rivet_ring from \"rivet_ring.assembly\" as rivets\n\n    pattern frames = linear(seeds = [frame], direction = [1, 0, 0], spacing = input.frame_pitch, count = {frames})\n    pattern stringers = circular(seeds = [stringer], axis_point = [0 mm, 0 mm, 0 mm], axis = [1, 0, 0], count = {STRINGERS}, angle = 360 deg)\n    pattern panels = circular(seeds = [panel], axis_point = [0 mm, 0 mm, 0 mm], axis = [1, 0, 0], count = 24, angle = 360 deg)\n    pattern rivet_rings = linear(seeds = [rivets], direction = [1, 0, 0], spacing = input.frame_pitch, count = {frames})\n}}\n",
        mm(FRAME_PITCH)
    )
}

fn fuselage() -> String {
    format!(
        "{HEADER}// The fuselage: nose, three barrel sections and the tail cone along x.\nassembly fuselage {{\n    use part nose from \"nose.part\" as nose\n    use assembly fuselage_section from \"fuselage_section.assembly\" as forward {{\n        {}\n    }}\n    use assembly fuselage_section from \"fuselage_section.assembly\" as center {{\n        {}\n    }}\n    use assembly fuselage_section from \"fuselage_section.assembly\" as aft {{\n        {}\n    }}\n    use part tail_cone from \"tail_cone.part\" as tail_cone {{\n        {}\n    }}\n}}\n",
        offset(4_000.0, 0.0, 0.0),
        offset(4_000.0 + SECTION, 0.0, 0.0),
        offset(4_000.0 + 2.0 * SECTION, 0.0, 0.0),
        offset(4_000.0 + 3.0 * SECTION, 0.0, 0.0)
    )
}

fn wing(name: &str, s: f64) -> String {
    let t = 0.12 * CHORD;
    let h = t / 2.0;
    let rr = (CHORD * CHORD / 4.0 + h * h) / (2.0 * h);
    let depth = |x: f64| (rr * rr - (x - CHORD / 2.0).powi(2)).sqrt() - (rr - h);
    let ribs = (SEMISPAN / RIB_PITCH) as usize;
    let rivets = (SEMISPAN / RIVET_PITCH) as usize;
    // Parts run root to tip along +y; the port wing places them back from y = 0.
    let base = if s > 0.0 { 0.0 } else { -SEMISPAN };
    let rib0 = if s > 0.0 { 0.0 } else { -20.0 };
    let side = if s > 0.0 { "starboard" } else { "port" };
    let dir = if s > 0.0 { "[0, 1, 0]" } else { "[0, -1, 0]" };
    // Skin panels side by side about each arc's centre, leading edge to trailing edge.
    let mut skins = String::new();
    let panel = skin_sweep() / SKIN_PANELS as f64;
    for k in 0..SKIN_PANELS {
        let a = -skin_sweep() / 2.0 + k as f64 * panel;
        skins += &format!(
            "    use part wing_skin_upper from \"wing_skin_upper.part\" as upper_skin{k} {{\n        {}\n    }}\n",
            offset_rot(CHORD / 2.0, base, -(rr - h), [0, 1, 0], a)
        );
        skins += &format!(
            "    use part wing_skin_lower from \"wing_skin_lower.part\" as lower_skin{k} {{\n        {}\n    }}\n",
            offset_rot(CHORD / 2.0, base, rr - h, [0, 1, 0], -(a + panel))
        );
    }
    let mut rows = String::new();
    for (k, (x, upper)) in [
        (0.15 * CHORD, true),
        (0.15 * CHORD, false),
        (0.65 * CHORD, true),
        (0.65 * CHORD, false),
    ]
    .into_iter()
    .enumerate()
    {
        let z = if upper { depth(x) } else { -depth(x) };
        let place = if upper {
            offset(x, s * RIVET_PITCH / 2.0, z)
        } else {
            offset_rot(x, s * RIVET_PITCH / 2.0, z, [1, 0, 0], 180.0)
        };
        rows += &format!(
            "    use part rivet from \"rivet.part\" as rivet{k} {{\n        {place}\n    }}\n"
        );
        rows += &format!("    pattern rivets{k} = linear(seeds = [rivet{k}], direction = {dir}, spacing = {}, count = {rivets})\n", mm(RIVET_PITCH));
    }
    format!(
        "{HEADER}// The {side} wing box: front and rear spars, {ribs} ribs at the rib pitch, upper and lower skins,\n// rivet rows along the spar caps, and the engine nacelle at 1/3 semispan.\nassembly {name} {{\n    input rib_pitch : length = {}\n\n    use part spar from \"spar.part\" as front_spar {{\n        {}\n    }}\n    use part rear_spar from \"rear_spar.part\" as rear_spar {{\n        {}\n    }}\n{skins}    use part rib from \"rib.part\" as rib {{\n        {}\n    }}\n    pattern ribs = linear(seeds = [rib], direction = {dir}, spacing = input.rib_pitch, count = {})\n{rows}    use assembly nacelle from \"nacelle.assembly\" as engine {{\n        {}\n    }}\n}}\n",
        mm(RIB_PITCH),
        offset(0.0, base, 0.0),
        offset(0.0, base, 0.0),
        offset(0.0, rib0 + s * RIB_PITCH / 2.0, 0.0),
        ribs,
        offset(-2_600.0, s * 4_800.0, -1_900.0)
    )
}

fn nacelle() -> String {
    format!(
        "{HEADER}// An engine nacelle: cowl, fan, spinner and pylon, the inlet at the origin.\nassembly nacelle {{\n    use part cowl from \"cowl.part\" as cowl\n    use part fan from \"fan.part\" as fan {{\n        {}\n    }}\n    use part spinner from \"spinner.part\" as spinner {{\n        {}\n    }}\n    use part pylon from \"pylon.part\" as pylon\n}}\n",
        offset(500.0, 0.0, 0.0),
        offset(-100.0, 0.0, 0.0)
    )
}

fn gear() -> String {
    format!(
        "{HEADER}// A landing gear leg: strut, axle and two wheels.\nassembly gear {{\n    use part gear_strut from \"gear_strut.part\" as strut {{\n        {}\n    }}\n    use part axle from \"axle.part\" as axle {{\n        {}\n    }}\n    use part wheel from \"wheel.part\" as inboard {{\n        {}\n    }}\n    use part wheel from \"wheel.part\" as outboard {{\n        {}\n    }}\n}}\n",
        offset(0.0, 0.0, -1_400.0),
        offset(0.0, 0.0, -1_400.0),
        offset(0.0, -330.0, -1_400.0),
        offset(0.0, 330.0, -1_400.0)
    )
}

fn airliner() -> String {
    format!(
        "{HEADER}// A generic twin-engine airliner, 34 m long with a 34 m span.\nassembly airliner {{\n    input span : length = {}\n    input frame_pitch : length = {}\n\n    use part airliner_skeleton from \"airliner_skeleton.part\" as skeleton\n    use assembly fuselage from \"fuselage.assembly\" as fuselage\n    use assembly wing_starboard from \"wing_starboard.assembly\" as wing_starboard {{\n        {}\n    }}\n    use assembly wing_port from \"wing_port.assembly\" as wing_port {{\n        {}\n    }}\n    use part fin from \"fin.part\" as fin {{\n        {}\n    }}\n    use part stabilizer from \"stabilizer.part\" as stab_starboard {{\n        {}\n    }}\n    use part stabilizer from \"stabilizer.part\" as stab_port {{\n        {}\n    }}\n    use assembly gear from \"gear.assembly\" as main_gear_starboard {{\n        {}\n    }}\n    use assembly gear from \"gear.assembly\" as main_gear_port {{\n        {}\n    }}\n    use assembly gear from \"gear.assembly\" as nose_gear {{\n        {}\n    }}\n}}\n",
        mm(2.0 * (SEMISPAN + 1_200.0)),
        mm(FRAME_PITCH),
        attach("wing_root_starboard"),
        attach("wing_root_port"),
        attach("fin_root"),
        offset(30_000.0, 0.0, 300.0),
        offset(30_000.0, -5_600.0, 300.0),
        attach("main_gear_starboard"),
        attach("main_gear_port"),
        attach("nose_gear")
    )
}

/// Every file of the airliner, by file name.
pub fn files() -> Vec<(String, String)> {
    let mut out = vec![("airliner_skeleton.part".to_string(), skeleton())];
    out.extend(fuselage_parts());
    out.extend(wing_parts());
    out.extend(other_parts());
    out.push(("rivet_ring.assembly".into(), rivet_ring()));
    out.push(("fuselage_section.assembly".into(), fuselage_section()));
    out.push(("fuselage.assembly".into(), fuselage()));
    out.push((
        "wing_starboard.assembly".into(),
        wing("wing_starboard", 1.0),
    ));
    out.push(("wing_port.assembly".into(), wing("wing_port", -1.0)));
    out.push(("nacelle.assembly".into(), nacelle()));
    out.push(("gear.assembly".into(), gear()));
    out.push(("airliner.assembly".into(), airliner()));
    out
}
