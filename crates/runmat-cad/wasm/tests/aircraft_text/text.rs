//! Text helpers shared by the airliner's files: numbers in millimetres, sketches, parts and
//! placements.

pub const HEADER: &str = "runmat 0.7\nstd 1\n\n";

pub fn mm(v: f64) -> String {
    let r = (v * 1000.0).round() / 1000.0 + 0.0;
    if r == r.trunc() {
        format!("{} mm", r as i64)
    } else {
        format!("{r} mm")
    }
}

pub fn num(v: f64) -> String {
    let r = (v * 1e6).round() / 1e6 + 0.0;
    format!("{r}")
}

pub fn pt(u: f64, v: f64) -> String {
    format!("[{}, {}]", mm(u), mm(v))
}

/// A closed polygon sketch: `l1 … ln` through the points.
pub fn polygon(name: &str, plane: &str, pts: &[(f64, f64)]) -> String {
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

/// One closed loop of curves, written in order: each a spline through its points or a line.
/// Consecutive curves share their end points (`next.start = prev.end`).
pub enum Curve {
    Spline(Vec<(f64, f64)>),
    /// Straight lines through these corners, then to the next curve's first point.
    Line(Vec<(f64, f64)>),
}

/// A sketch of one closed loop: splines through the given points, lines between them. Each
/// curve starts where the previous one ended; a `Line` runs to the next curve's first point
/// (the last one closes the loop).
pub fn loop_sketch(name: &str, plane: &str, curves: &[Curve]) -> String {
    let mut s = format!("    sketch {name} on {plane} {{\n");
    let first = |k: usize| -> (f64, f64) {
        match &curves[k % curves.len()] {
            Curve::Spline(p) => p[0],
            Curve::Line(_) => unreachable!("two lines in a row"),
        }
    };
    let mut prev: Option<String> = None;
    for (k, c) in curves.iter().enumerate() {
        let id = format!("c{}", k + 1);
        let last = k + 1 == curves.len();
        match c {
            Curve::Spline(p) => {
                let mut pts: Vec<String> = p.iter().map(|q| pt(q.0, q.1)).collect();
                if let Some(prev) = &prev {
                    pts[0] = format!("{prev}.end");
                }
                if last {
                    *pts.last_mut().unwrap() = "c1.start".into();
                }
                s += &format!("        {id} = spline(points = [{}])\n", pts.join(", "));
            }
            Curve::Line(corners) => {
                let mut start = prev.clone().map(|p| format!("{p}.end")).expect("a line follows a spline");
                for (j, q) in corners.iter().enumerate() {
                    let lid = format!("{id}_{}", j + 1);
                    s += &format!("        {lid} = line(start = {start}, end = {})\n", pt(q.0, q.1));
                    start = format!("{lid}.end");
                }
                let end = if last {
                    "c1.start".to_string()
                } else {
                    let q = first(k + 1);
                    pt(q.0, q.1)
                };
                s += &format!("        {id} = line(start = {start}, end = {end})\n");
            }
        }
        prev = Some(id);
    }
    s += "    }\n";
    s
}

/// A corner of an outline rounded by a short spline: from `r` before `p` (towards `a`) through
/// a point inside the corner to `r` after it (towards `b`).
pub fn rounded(a: (f64, f64), p: (f64, f64), b: (f64, f64), r: f64) -> Vec<(f64, f64)> {
    let unit = |q: (f64, f64)| {
        let (du, dv) = (q.0 - p.0, q.1 - p.1);
        let l = du.hypot(dv);
        (du / l, dv / l)
    };
    let (ua, ub) = (unit(a), unit(b));
    vec![
        (p.0 + r * ua.0, p.1 + r * ua.1),
        (p.0 + 0.3 * r * (ua.0 + ub.0), p.1 + 0.3 * r * (ua.1 + ub.1)),
        (p.0 + r * ub.0, p.1 + r * ub.1),
    ]
}

/// A closed outline through `corners`, each corner rounded over `r`.
pub fn outline(name: &str, plane: &str, corners: &[(f64, f64)], r: f64) -> String {
    let n = corners.len();
    let mut curves = Vec::new();
    for k in 0..n {
        let (a, p, b) = (corners[(k + n - 1) % n], corners[k], corners[(k + 1) % n]);
        curves.push(Curve::Spline(rounded(a, p, b, r)));
        curves.push(Curve::Line(vec![]));
    }
    loop_sketch(name, plane, &curves)
}

pub fn rect(name: &str, plane: &str, u0: f64, v0: f64, u1: f64, v1: f64) -> String {
    polygon(name, plane, &[(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
}

pub fn part(name: &str, number: &str, material: &str, comment: &str, body: &str, last: &str) -> String {
    let color = paint(name).map(|c| format!(", color = \"{c}\"")).unwrap_or_default();
    format!(
        "{HEADER}// {comment}\npart {name} {{\n{body}\n    output {name} = {{ body = feature.{last}.body, part_number = \"{number}\", material = \"{material}\"{color} }}\n}}\n"
    )
}

/// A generic airliner's finish by part, apart from its material: a near-white fuselage and tail,
/// darker wing greys, dark tinted glazing, dark intakes, bare-metal exhaust and gear,
/// primer-green inner structure. No airline's livery.
fn paint(name: &str) -> Option<&'static str> {
    let stem = name.strip_prefix("wing_").unwrap_or(name);
    let stem = stem.strip_suffix("_port").unwrap_or(stem);
    Some(match stem {
        "nose" | "skin_panel" | "tail_cone" => "#F4F5F7",
        "fairing" | "pylon" => "#C7CCD3",
        "cockpit_window_front" | "cockpit_window_side" | "cockpit_window_aft" => "#16202A",
        "cabin_window" => "#252E38",
        "frame" | "stringer" | "rib" | "front_spar" | "rear_spar" => "#A7B596",
        "rivet" => "#BFC4CB",
        "leading_edge" | "slat" | "box_skin_upper" | "box_skin_lower" | "trailing_edge_tip" => "#A8AFB8",
        "flap_inboard" | "flap_outboard" | "aileron" => "#B1B7BF",
        "winglet" | "fin" | "rudder" | "stabilizer" | "elevator" => "#E4E7EB",
        "cowl" => "#E6E9EC",
        "fan" | "spinner" => "#363C44",
        "core_nozzle" | "plug" => "#8A9097",
        "gear_strut" | "axle" | "hub" => "#A2A8AF",
        // The interior: fabric and leather seats, light plastic linings and monuments, a grey
        // carpeted floor, primer-green structure, dark flight-deck panels and screens.
        "seat_cushion" => "#2E4A6B",
        "seat_cushion_business" => "#5A2E3B",
        "pilot_seat_cushion" => "#3A3E44",
        "seat_frame" | "seat_track" => "#8C9299",
        "seat_armrest" => "#4C535B",
        "floor_panel" | "cockpit_floor" => "#6A7078",
        "floor_beam" | "centre_wing_box" | "gear_bay" => "#A7B596",
        "sidewall_lining" | "ceiling_panel" => "#ECEDEF",
        "overhead_bin" => "#E1E4E8",
        "galley" => "#D3D6DA",
        "lavatory" => "#E5E1D8",
        "cockpit_bulkhead" => "#C9CDD2",
        "instrument_panel" | "pedestal" => "#3C4148",
        "glareshield" | "control_column" => "#2A2E33",
        "display_unit" => "#0D1318",
        "cargo_floor" | "cargo_floor_aft" => "#5D6369",
        "cargo_container" => "#B7BDC5",
        "passenger_door" | "passenger_door_aft" | "overwing_exit" | "cargo_door" | "cargo_door_aft" => "#EEF0F2",
        _ => return None,
    })
}

pub fn offset(x: f64, y: f64, z: f64) -> String {
    format!(
        "placement = {{ placement = offset, offset = {{ translation = [{}, {}, {}] }} }}",
        mm(x),
        mm(y),
        mm(z)
    )
}

pub fn offset_rot(x: f64, y: f64, z: f64, axis: [f64; 3], deg: f64) -> String {
    format!(
        "placement = {{ placement = offset, offset = {{ translation = [{}, {}, {}], rotation = {{ rotation = axis_angle, axis = [{}, {}, {}], angle = {} deg }} }} }}",
        mm(x),
        mm(y),
        mm(z),
        num(axis[0]),
        num(axis[1]),
        num(axis[2]),
        num(deg)
    )
}

pub fn attach(frame: &str) -> String {
    format!(
        "placement = {{ placement = attach, own = {{ anchor = origin }}, to = {{ frame = interface, file = \"airliner_skeleton.part\", name = {frame} }} }}"
    )
}

/// An occurrence: `use <kind> <name> from "<name>.<kind>" as <alias> { lines }`.
pub fn use_(kind: &str, name: &str, alias: &str, lines: &[String]) -> String {
    if lines.is_empty() {
        return format!("    use {kind} {name} from \"{name}.{kind}\" as {alias}\n");
    }
    let mut s = format!("    use {kind} {name} from \"{name}.{kind}\" as {alias} {{\n");
    for l in lines {
        s += &format!("        {l}\n");
    }
    s += "    }\n";
    s
}

/// `configuration = { select = inputs, values = { k = v, … } }`.
pub fn inputs(values: &[(&str, String)]) -> String {
    let v: Vec<String> = values.iter().map(|(k, v)| format!("{k} = {v}")).collect();
    format!("configuration = {{ select = inputs, values = {{ {} }} }}", v.join(", "))
}
