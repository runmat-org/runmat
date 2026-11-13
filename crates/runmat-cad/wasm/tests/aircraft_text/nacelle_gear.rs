//! The engine nacelle (cowl with a rounded inlet lip, fan, spinner, core nozzle, exhaust plug
//! and pylon) and the landing gear (oleo struts, axles, hubs and tyres), each a sub-assembly.

use super::text::{inputs, loop_sketch, mm, offset, part, polygon, use_, Curve, HEADER};

fn revolved(name: &str, number: &str, material: &str, comment: &str, curves: &[Curve]) -> String {
    part(
        name,
        number,
        material,
        comment,
        &format!(
            "{}    feature body = revolve(profile = sketch.s, axis = origin.x, angle = 360 deg)\n",
            loop_sketch("s", "origin.xy", curves)
        ),
        "body",
    )
}

pub fn parts() -> Vec<(String, String)> {
    let cowl = revolved(
        "cowl",
        "NAC-CW-900",
        "composite",
        "The nacelle cowl: a rounded inlet lip swelling to 1.1 m and closing over the core, turned about x.",
        &[
            Curve::Spline(vec![
                (180.0, 835.0),
                (50.0, 850.0),
                (0.0, 895.0),
                (30.0, 965.0),
                (300.0, 1_045.0),
                (1_100.0, 1_100.0),
                (2_200.0, 1_075.0),
                (3_000.0, 975.0),
                (3_450.0, 880.0),
            ]),
            Curve::Line(vec![(3_450.0, 820.0)]),
        ],
    );
    let fan = part(
        "fan",
        "NAC-FN-910",
        "titanium",
        "The fan disk behind the inlet.",
        "    sketch s on origin.yz {\n        c = circle(center = [0 mm, 0 mm], radius = 840 mm)\n    }\n    feature disk = extrude(profile = sketch.s, distance = 140 mm)\n",
        "disk",
    );
    let spinner = revolved(
        "spinner",
        "NAC-SP-920",
        "composite",
        "The spinner at the fan's hub.",
        &[
            Curve::Spline(vec![
                (140.0, 0.0),
                (175.0, 120.0),
                (280.0, 230.0),
                (440.0, 300.0),
                (620.0, 320.0),
            ]),
            Curve::Line(vec![(620.0, 0.0)]),
        ],
    );
    let core = revolved(
        "core_nozzle",
        "NAC-CN-940",
        "titanium",
        "The core cowl and its nozzle, aft of the fan cowl.",
        &[
            Curve::Spline(vec![
                (2_900.0, 0.0),
                (2_900.0, 400.0),
                (3_100.0, 790.0),
                (3_700.0, 760.0),
                (4_300.0, 590.0),
            ]),
            Curve::Line(vec![(4_300.0, 0.0)]),
        ],
    );
    let plug = revolved(
        "plug",
        "NAC-PL-950",
        "titanium",
        "The exhaust plug.",
        &[
            Curve::Spline(vec![
                (4_300.0, 420.0),
                (4_650.0, 300.0),
                (4_950.0, 120.0),
                (5_050.0, 0.0),
            ]),
            Curve::Line(vec![(4_300.0, 0.0)]),
        ],
    );
    // The pylon's side outline (x, z): along the cowl's crown, up into the wing box.
    let outline = [
        (700.0, 1_040.0),
        (3_300.0, 860.0),
        (5_000.0, 1_480.0),
        (3_200.0, 1_720.0),
    ];
    let zx: Vec<(f64, f64)> = outline.iter().map(|(x, z)| (*z, *x)).collect();
    let pylon = part(
        "pylon",
        "NAC-PY-930",
        "titanium",
        "The pylon from the cowl's crown up into the wing box, 360 mm wide.",
        &format!(
            "{}    feature pylon = extrude(profile = sketch.s, distance = 360 mm, symmetric)\n",
            polygon("s", "origin.zx", &zx)
        ),
        "pylon",
    );
    // Gear parts, sized per occurrence by their inputs.
    let strut = part(
        "gear_strut",
        "LG-ST-800",
        "steel_300m",
        "An oleo strut hanging from its trunnion: length and radius are inputs.",
        "    input length : length = 2500 mm\n    input radius : length = 110 mm\n\n    sketch s on origin.xy {\n        c = circle(center = [0 mm, 0 mm], radius = input.radius)\n    }\n    feature strut = extrude(profile = sketch.s, distance = input.length)\n",
        "strut",
    );
    let axle = part(
        "axle",
        "LG-AX-820",
        "steel_300m",
        "The axle across both wheels: its length is an input.",
        "    input length : length = 1000 mm\n\n    sketch s on origin.zx {\n        c = circle(center = [0 mm, 0 mm], radius = 70 mm)\n    }\n    feature axle = extrude(profile = sketch.s, distance = input.length, symmetric)\n",
        "axle",
    );
    let tyre = part(
        "tyre",
        "LG-TY-810",
        "rubber",
        "A tyre: its section turned about the axle (y). The rim radius and section are inputs.",
        "    input rim : length = 440 mm\n    input tube : length = 170 mm\n\n    sketch s on origin.xy {\n        c = circle(center = [input.rim, 0 mm], radius = input.tube)\n    }\n    feature tyre = revolve(profile = sketch.s, axis = origin.y, angle = 360 deg)\n",
        "tyre",
    );
    let hub = part(
        "hub",
        "LG-HB-815",
        "aluminum_7075",
        "A wheel hub filling the tyre's rim: its radius is an input.",
        "    input radius : length = 440 mm\n\n    sketch s on origin.zx {\n        c = circle(center = [0 mm, 0 mm], radius = input.radius)\n    }\n    feature hub = extrude(profile = sketch.s, distance = 200 mm, symmetric)\n",
        "hub",
    );
    [
        ("cowl", cowl),
        ("fan", fan),
        ("spinner", spinner),
        ("core_nozzle", core),
        ("plug", plug),
        ("pylon", pylon),
        ("gear_strut", strut),
        ("axle", axle),
        ("tyre", tyre),
        ("hub", hub),
    ]
    .into_iter()
    .map(|(n, t)| (format!("{n}.part"), t))
    .collect()
}

pub fn nacelle() -> String {
    let mut s = format!("{HEADER}// An engine nacelle: cowl, fan, spinner, core nozzle, exhaust plug and pylon; the inlet at the origin.\nassembly nacelle {{\n");
    s += &use_("part", "cowl", "cowl", &[]);
    s += &use_("part", "fan", "fan", &[offset(560.0, 0.0, 0.0)]);
    s += &use_("part", "spinner", "spinner", &[]);
    s += &use_("part", "core_nozzle", "core", &[]);
    s += &use_("part", "plug", "plug", &[]);
    s += &use_("part", "pylon", "pylon", &[]);
    s += "}\n";
    s
}

/// A gear leg hanging from its frame: strut of `length`, the axle at its foot, two wheels of
/// `rim` + `section` radius `track` apart.
pub fn gear(
    name: &str,
    comment: &str,
    length: f64,
    radius: f64,
    rim: f64,
    section: f64,
    track: f64,
) -> String {
    let mut s = format!("{HEADER}// {comment}\nassembly {name} {{\n");
    s += &use_(
        "part",
        "gear_strut",
        "strut",
        &[
            inputs(&[("length", mm(length)), ("radius", mm(radius))]),
            offset(0.0, 0.0, -length),
        ],
    );
    s += &use_(
        "part",
        "axle",
        "axle",
        &[
            inputs(&[("length", mm(track + 2.0 * section))]),
            offset(0.0, 0.0, -length),
        ],
    );
    for (alias, y) in [("inboard", -track / 2.0), ("outboard", track / 2.0)] {
        s += &use_(
            "part",
            "tyre",
            &format!("{alias}_tyre"),
            &[
                inputs(&[("rim", mm(rim)), ("tube", mm(section))]),
                offset(0.0, y, -length),
            ],
        );
        s += &use_(
            "part",
            "hub",
            &format!("{alias}_hub"),
            &[inputs(&[("radius", mm(rim))]), offset(0.0, y, -length)],
        );
    }
    s += "}\n";
    s
}
