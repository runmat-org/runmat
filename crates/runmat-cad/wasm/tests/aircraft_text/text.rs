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

pub fn rect(name: &str, plane: &str, u0: f64, v0: f64, u1: f64, v1: f64) -> String {
    polygon(name, plane, &[(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
}

pub fn part(name: &str, number: &str, material: &str, comment: &str, body: &str, last: &str) -> String {
    format!(
        "{HEADER}// {comment}\npart {name} {{\n{body}\n    output {name} = {{ body = feature.{last}.body, part_number = \"{number}\", material = \"{material}\" }}\n}}\n"
    )
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
