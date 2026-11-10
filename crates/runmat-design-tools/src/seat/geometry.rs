//! The geometry of a seating, in metres: the contact frame on a named face or on a side of a
//! part's box, the rotation that turns one frame onto another, and how far one placed part can
//! travel along a direction before it meets another (rays from each part's vertices onto the
//! other's triangles, both ways, so a curved surface pressing into a flat face is caught as well
//! as a corner meeting a curved surface).

use runmat_cad::display::{FaceDisplay, PartDisplay};
use runmat_cad::refs::history::SurfaceKind;

pub type V3 = [f64; 3];
pub type M3 = [[f64; 3]; 3];

pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
pub fn unit(a: V3) -> Option<V3> {
    let l = norm(a);
    (l > 1e-12).then(|| scale(a, 1.0 / l))
}

/// `v` made square to unit `z` and of unit length, when it isn't along `z`.
pub fn square_to(v: V3, z: V3) -> Option<V3> {
    unit(sub(v, scale(z, dot(v, z))))
}

/// A unit vector square to `z`, along the world axis that leans least on it.
pub fn perpendicular(z: V3) -> V3 {
    let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let least = axes.into_iter().min_by(|a, b| dot(*a, z).abs().total_cmp(&dot(*b, z).abs())).unwrap_or([1.0, 0.0, 0.0]);
    square_to(least, z).unwrap_or([1.0, 0.0, 0.0])
}

pub fn apply(r: &M3, v: V3) -> V3 {
    [dot(r[0], v), dot(r[1], v), dot(r[2], v)]
}

pub fn transpose(r: &M3) -> M3 {
    [[r[0][0], r[1][0], r[2][0]], [r[0][1], r[1][1], r[2][1]], [r[0][2], r[1][2], r[2][2]]]
}

/// The rotation that turns the frame (`from_x`, `from_z`) onto (`to_x`, `to_z`); each `x` is
/// square to its `z`.
pub fn frame_rotation(from_x: V3, from_z: V3, to_x: V3, to_z: V3) -> M3 {
    let columns = |x: V3, z: V3| -> M3 {
        let y = cross(z, x);
        [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]
    };
    let (a, b) = (columns(to_x, to_z), transpose(&columns(from_x, from_z)));
    let mut r = [[0.0; 3]; 3];
    for (i, row) in r.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    r
}

/// `v` turned by `angle` (radians) about unit `axis` (Rodrigues).
pub fn turn(v: V3, axis: V3, angle: f64) -> V3 {
    let (s, c) = angle.sin_cos();
    add(add(scale(v, c), scale(cross(axis, v), s)), scale(axis, dot(axis, v) * (1.0 - c)))
}

/// A rotation as its unit axis and angle (radians, 0 to π).
pub fn axis_angle(r: &M3) -> (V3, f64) {
    let trace = r[0][0] + r[1][1] + r[2][2];
    let angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
    if angle < 1e-9 {
        return ([0.0, 0.0, 1.0], 0.0);
    }
    let skew = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
    if angle < std::f64::consts::PI - 1e-6 {
        return (unit(skew).unwrap_or([0.0, 0.0, 1.0]), angle);
    }
    // Half a turn: the axis is the column of R + I with the largest diagonal.
    let k = (0..3).max_by(|&a, &b| r[a][a].total_cmp(&r[b][b])).unwrap_or(0);
    let column = [r[0][k] + if k == 0 { 1.0 } else { 0.0 }, r[1][k] + if k == 1 { 1.0 } else { 0.0 }, r[2][k] + if k == 2 { 1.0 } else { 0.0 }];
    (unit(column).unwrap_or([0.0, 0.0, 1.0]), angle)
}

/// Where a part meets its neighbour: a plane (a point on it and its outward normal) or an axis
/// (a point on it and its direction).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Contact {
    Plane { origin: V3, normal: V3 },
    Axis { origin: V3, axis: V3 },
}

/// The face of `display` that `pick` names: its reference text, one of its names or regions,
/// or its name for people (`Body side 2`, case-insensitive).
pub fn find_face<'a>(display: &'a PartDisplay, pick: &str) -> Option<&'a FaceDisplay> {
    let pick = pick.trim();
    // `feature.<f>.face[<body>:<face>]`: how design.inspect names a face that has no name.
    let indexed = pick.rsplit_once(".face[").and_then(|(_, rest)| {
        let (body, face) = rest.strip_suffix(']')?.split_once(':')?;
        Some((body.parse::<u32>().ok()?, face.parse::<u32>().ok()?))
    });
    if let Some((body, face)) = indexed {
        let found = display.bodies.iter().filter(|b| b.index == body).flat_map(|b| b.faces.iter()).find(|f| f.index == face);
        if found.is_some() {
            return found;
        }
    }
    faces(display).find(|f| {
        f.reference.as_deref() == Some(pick)
            || f.names.iter().any(|n| n == pick)
            || f.regions.iter().any(|n| n == pick)
            || f.label.as_deref().is_some_and(|l| l.eq_ignore_ascii_case(pick))
    })
}

/// The names a refusal lists: each face's name for people, or its reference text.
pub fn face_names(display: &PartDisplay, at_most: usize) -> Vec<String> {
    faces(display).filter_map(|f| f.label.clone().or_else(|| f.reference.clone())).take(at_most).collect()
}

fn faces(display: &PartDisplay) -> impl Iterator<Item = &FaceDisplay> {
    display.bodies.iter().filter(|b| !b.flat_pattern).flat_map(|b| b.faces.iter())
}

fn face_area(face: &FaceDisplay) -> f64 {
    face.triangles.chunks_exact(3).map(|t| norm(cross(sub(position(face, t[1]), position(face, t[0])), sub(position(face, t[2]), position(face, t[0])))) / 2.0).sum()
}

fn position(face: &FaceDisplay, i: u32) -> V3 {
    let k = i as usize * 3;
    [face.positions[k] as f64, face.positions[k + 1] as f64, face.positions[k + 2] as f64]
}

/// A face's contact, in the part's coordinates: a cylinder's axis (through the point level
/// with the face's middle), else the plane through its centroid along its mean outward normal.
pub fn face_contact(face: &FaceDisplay) -> Option<Contact> {
    let (mut centroid, mut normal, mut area) = ([0.0; 3], [0.0; 3], 0.0);
    for t in face.triangles.chunks_exact(3) {
        let (a, b, c) = (position(face, t[0]), position(face, t[1]), position(face, t[2]));
        let n = cross(sub(b, a), sub(c, a));
        let w = norm(n) / 2.0;
        centroid = add(centroid, scale(add(add(a, b), c), w / 3.0));
        normal = add(normal, n);
        area += w;
    }
    if area <= 0.0 {
        return None;
    }
    let centroid = scale(centroid, 1.0 / area);
    if let Some(SurfaceKind::Cylinder { axis_point, axis_direction, .. }) = face.surface {
        let axis = unit(axis_direction)?;
        let origin = add(axis_point, scale(axis, dot(sub(centroid, axis_point), axis)));
        return Some(Contact::Axis { origin, axis });
    }
    // A gently curved face seats as the plane of its mean normal (the seating then moves to
    // first contact); one that curls round (a cylinder the kernel didn't name) has none.
    if norm(normal) < area {
        return None;
    }
    Some(Contact::Plane { origin: centroid, normal: unit(normal)? })
}

/// The box around a part's bodies, in its coordinates.
pub fn bounds(display: &PartDisplay) -> Option<(V3, V3)> {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for face in faces(display) {
        for p in face.positions.chunks_exact(3) {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k] as f64);
                hi[k] = hi[k].max(p[k] as f64);
            }
        }
    }
    (lo[0] <= hi[0]).then_some((lo, hi))
}

/// A side of the part's box (`+x`, `-x`, `+y`, `-y`, `+z`, `-z`): the middle of that side and
/// the side's outward normal.
pub fn side_contact(display: &PartDisplay, side: &str) -> Option<Contact> {
    let (lo, hi) = bounds(display)?;
    let s = side.trim().to_ascii_lowercase();
    let (sign, axis) = match s.as_bytes() {
        [b'+', a] | [a] => (1.0, *a),
        [b'-', a] => (-1.0, *a),
        _ => return None,
    };
    let k = match axis {
        b'x' => 0,
        b'y' => 1,
        b'z' => 2,
        _ => return None,
    };
    let mut origin = scale(add(lo, hi), 0.5);
    origin[k] = if sign > 0.0 { hi[k] } else { lo[k] };
    let mut normal = [0.0; 3];
    normal[k] = sign;
    Some(Contact::Plane { origin, normal })
}

/// The side of the part's box whose normal, turned by `rotation`, faces `toward` most.
pub fn facing_side(rotation: &M3, toward: V3) -> &'static str {
    const SIDES: [(&str, V3); 6] = [("+x", [1.0, 0.0, 0.0]), ("-x", [-1.0, 0.0, 0.0]), ("+y", [0.0, 1.0, 0.0]), ("-y", [0.0, -1.0, 0.0]), ("+z", [0.0, 0.0, 1.0]), ("-z", [0.0, 0.0, -1.0])];
    SIDES.into_iter().max_by(|a, b| dot(apply(rotation, a.1), toward).total_cmp(&dot(apply(rotation, b.1), toward))).map(|s| s.0).unwrap_or("-z")
}

/// The part's long axis square to `normal`. A part with a round face whose axis lies across
/// the contact (a clamp's bore, a boss) runs along that axis, the largest such face's, so a
/// clamp seated `along` a member holds it along the member. Otherwise it is the box axis of
/// greatest extent that lies across the contact, made square to it.
pub fn long_axis(display: &PartDisplay, normal: V3) -> V3 {
    let round = faces(display)
        .filter_map(|f| match f.surface {
            Some(SurfaceKind::Cylinder { axis_direction, .. }) => unit(axis_direction).filter(|a| dot(*a, normal).abs() < 0.3).map(|a| (a, face_area(f))),
            _ => None,
        })
        .max_by(|a, b| a.1.total_cmp(&b.1));
    if let Some(axis) = round.and_then(|(a, _)| square_to(a, normal)) {
        return axis;
    }
    let Some((lo, hi)) = bounds(display) else { return perpendicular(normal) };
    let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    (0..3)
        .filter(|&k| dot(axes[k], normal).abs() < 0.7)
        .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
        .and_then(|k| square_to(axes[k], normal))
        .unwrap_or_else(|| perpendicular(normal))
}

/// A part's triangles and vertices, placed by `place`.
pub struct Mesh {
    pub points: Vec<V3>,
    pub triangles: Vec<[V3; 3]>,
}

pub fn mesh(display: &PartDisplay, place: impl Fn(V3) -> V3) -> Mesh {
    let mut points = Vec::new();
    let mut triangles = Vec::new();
    for face in faces(display) {
        let placed: Vec<V3> = face.positions.chunks_exact(3).map(|p| place([p[0] as f64, p[1] as f64, p[2] as f64])).collect();
        for t in face.triangles.chunks_exact(3) {
            if let (Some(a), Some(b), Some(c)) = (placed.get(t[0] as usize), placed.get(t[1] as usize), placed.get(t[2] as usize)) {
                triangles.push([*a, *b, *c]);
            }
        }
        points.extend(placed);
    }
    Mesh { points, triangles }
}

/// The ray's distance to the triangle (Möller–Trumbore), when it meets it ahead of the start.
fn meets(from: V3, dir: V3, [a, b, c]: &[V3; 3]) -> Option<f64> {
    let (e1, e2) = (sub(*b, *a), sub(*c, *a));
    let p = cross(dir, e2);
    let det = dot(e1, p);
    if det.abs() < 1e-18 {
        return None;
    }
    let inv = 1.0 / det;
    let s = sub(from, *a);
    let u = dot(s, p) * inv;
    if !(-1e-9..=1.0 + 1e-9).contains(&u) {
        return None;
    }
    let q = cross(s, e1);
    let v = dot(dir, q) * inv;
    if v < -1e-9 || u + v > 1.0 + 1e-9 {
        return None;
    }
    let t = dot(e2, q) * inv;
    (t >= 0.0).then_some(t)
}

/// The most rays cast from one part: every vertex up to this many, an even sample beyond.
const MAX_RAYS: usize = 4000;

/// How far `moving` can travel along unit `a` before it touches `fixed`: the least distance over
/// rays from `moving`'s vertices along `a` onto `fixed`'s triangles and from `fixed`'s vertices
/// along `-a` onto `moving`'s. `None` when the two never meet along `a`. Only what lies in
/// `moving`'s shadow along `a` is tested.
pub fn travel(moving: &Mesh, fixed: &Mesh, a: V3) -> Option<f64> {
    let u = perpendicular(a);
    let v = cross(a, u);
    let uv = |p: V3| (dot(p, u), dot(p, v));
    let (mut lo, mut hi) = ((f64::INFINITY, f64::INFINITY), (f64::NEG_INFINITY, f64::NEG_INFINITY));
    for p in &moving.points {
        let (x, y) = uv(*p);
        lo = (lo.0.min(x), lo.1.min(y));
        hi = (hi.0.max(x), hi.1.max(y));
    }
    let margin = 1e-4;
    let inside = |p: V3| {
        let (x, y) = uv(p);
        x >= lo.0 - margin && x <= hi.0 + margin && y >= lo.1 - margin && y <= hi.1 + margin
    };
    let shadowed = |t: &[V3; 3]| {
        let (mut l, mut h) = ((f64::INFINITY, f64::INFINITY), (f64::NEG_INFINITY, f64::NEG_INFINITY));
        for p in t {
            let (x, y) = uv(*p);
            l = (l.0.min(x), l.1.min(y));
            h = (h.0.max(x), h.1.max(y));
        }
        h.0 >= lo.0 - margin && l.0 <= hi.0 + margin && h.1 >= lo.1 - margin && l.1 <= hi.1 + margin
    };
    let targets: Vec<&[V3; 3]> = fixed.triangles.iter().filter(|t| shadowed(t)).collect();
    let sample = |points: Vec<V3>| -> Vec<V3> {
        let step = points.len().div_ceil(MAX_RAYS).max(1);
        points.into_iter().step_by(step).collect()
    };
    let mut best: Option<f64> = None;
    let mut keep = |t: f64| best = Some(best.map_or(t, |b: f64| b.min(t)));
    for p in sample(moving.points.clone()) {
        for t in targets.iter().filter_map(|tri| meets(p, a, tri)) {
            keep(t);
        }
    }
    let back = scale(a, -1.0);
    for p in sample(fixed.points.iter().copied().filter(|p| inside(*p)).collect()) {
        for t in moving.triangles.iter().filter_map(|tri| meets(p, back, tri)) {
            keep(t);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_turns_onto_another_and_back_to_its_axis_and_angle() {
        let r = frame_rotation([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]);
        let z = apply(&r, [0.0, 0.0, 1.0]);
        let x = apply(&r, [1.0, 0.0, 0.0]);
        assert!(norm(sub(z, [0.0, 0.0, -1.0])) < 1e-12 && norm(sub(x, [0.0, 1.0, 0.0])) < 1e-12);
        let (axis, angle) = axis_angle(&r);
        assert!((angle - std::f64::consts::PI).abs() < 1e-9, "{angle}");
        for p in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            assert!(norm(sub(turn(p, axis, angle), apply(&r, p))) < 1e-9);
        }
        let q = frame_rotation([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let (axis, angle) = axis_angle(&q);
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-9 && norm(sub(axis, [0.0, 0.0, 1.0])) < 1e-9);
    }

    fn square(z: f64, half: f64) -> Mesh {
        let p = [[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]];
        Mesh { points: p.to_vec(), triangles: vec![[p[0], p[1], p[2]], [p[0], p[2], p[3]]] }
    }

    #[test]
    fn travel_is_the_nearest_meeting_either_way() {
        // A small square 3 below a large one moves up 3, and so does a large one under a small
        // one: its corners' rays miss the small square, and the small square's corners, cast
        // back down, meet it.
        assert!((travel(&square(0.0, 1.0), &square(3.0, 5.0), [0.0, 0.0, 1.0]).unwrap() - 3.0).abs() < 1e-12);
        assert!((travel(&square(0.0, 5.0), &square(3.0, 1.0), [0.0, 0.0, 1.0]).unwrap() - 3.0).abs() < 1e-12);
        assert!(travel(&square(0.0, 1.0), &square(3.0, 1.0), [0.0, 0.0, -1.0]).is_none());
    }
}
