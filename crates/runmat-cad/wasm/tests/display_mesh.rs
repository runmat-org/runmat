//! The display meshes the Desktop draws, through the wasm `compile` entry it calls: holes,
//! fillets and bends are watertight, round, well shaped and smoothly shaded, and they hold the
//! same closed-form volume and area as the solid.

use runmat_cad::display::{BodyDisplay, FaceDisplay, PartDisplay};
use std::collections::BTreeMap;
use std::f64::consts::PI;

const MM: f64 = 1e-3;

fn compile(path: &str, text: &str) -> PartDisplay {
    let out = runmat_cad_wasm::compile(path, text, "{}", "{}");
    let v: serde_json::Value = serde_json::from_str(&out).expect("json");
    let diags = v["diagnostics"].as_array().cloned().unwrap_or_default();
    let errors: Vec<_> = diags.iter().filter(|d| d["severity"] == "error").collect();
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(v["display"]["kind"], "part", "{out}");
    let display: PartDisplay =
        serde_json::from_value(v["display"]["data"].clone()).expect("part display");
    assert!(display.problems.is_empty(), "{:?}", display.problems);
    display
}

type P = [f64; 3];

fn pts(f: &FaceDisplay) -> Vec<P> {
    f.positions
        .chunks(3)
        .map(|c| [c[0] as f64, c[1] as f64, c[2] as f64])
        .collect()
}

fn sub(a: P, b: P) -> P {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: P, b: P) -> P {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: P, b: P) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn len(a: P) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: P) -> P {
    let l = len(a);
    [a[0] / l, a[1] / l, a[2] / l]
}

fn tris(f: &FaceDisplay) -> impl Iterator<Item = [usize; 3]> + '_ {
    f.triangles
        .chunks(3)
        .map(|t| [t[0] as usize, t[1] as usize, t[2] as usize])
}

/// Every edge of the body's triangles (merged across faces by exact position) is used by
/// exactly two triangles, once in each direction; no triangle is degenerate.
fn assert_watertight(b: &BodyDisplay) {
    let key = |p: &[f32]| (p[0].to_bits(), p[1].to_bits(), p[2].to_bits());
    let mut ids: BTreeMap<(u32, u32, u32), usize> = BTreeMap::new();
    let mut directed: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for f in &b.faces {
        let local: Vec<usize> = f
            .positions
            .chunks(3)
            .map(|c| {
                let n = ids.len();
                *ids.entry(key(c)).or_insert(n)
            })
            .collect();
        let p = pts(f);
        for t in tris(f) {
            let [a, bb, c] = t.map(|i| p[i]);
            let area2 = len(cross(sub(bb, a), sub(c, a)));
            let longest = len(sub(bb, a)).max(len(sub(c, bb))).max(len(sub(a, c)));
            assert!(
                area2 > 1e-7 * longest * longest,
                "face {}: sliver {a:?} {bb:?} {c:?}",
                f.index
            );
            let g = t.map(|i| local[i]);
            for k in 0..3 {
                *directed.entry((g[k], g[(k + 1) % 3])).or_default() += 1;
            }
        }
    }
    for ((a, b2), n) in &directed {
        assert_eq!(*n, 1, "edge {a}->{b2} used {n} times in one direction");
        assert_eq!(
            directed.get(&(*b2, *a)).copied(),
            Some(1),
            "edge {a}->{b2} has no opposite: the mesh is open there"
        );
    }
}

fn volume(b: &BodyDisplay) -> f64 {
    let mut v = 0.0;
    for f in &b.faces {
        let p = pts(f);
        for t in tris(f) {
            let [a, bb, c] = t.map(|i| p[i]);
            v += dot(a, cross(bb, c)) / 6.0;
        }
    }
    v
}

fn area(b: &BodyDisplay) -> f64 {
    b.faces
        .iter()
        .map(|f| {
            let p = pts(f);
            tris(f)
                .map(|t| {
                    let [a, bb, c] = t.map(|i| p[i]);
                    len(cross(sub(bb, a), sub(c, a))) / 2.0
                })
                .sum::<f64>()
        })
        .sum()
}

/// Area-weighted vertex normals, the way the renderer shades a face.
fn vertex_normals(f: &FaceDisplay) -> Vec<P> {
    let p = pts(f);
    let mut n = vec![[0.0; 3]; p.len()];
    for t in tris(f) {
        let c = cross(sub(p[t[1]], p[t[0]]), sub(p[t[2]], p[t[0]]));
        for i in t {
            n[i] = [n[i][0] + c[0], n[i][1] + c[1], n[i][2] + c[2]];
        }
    }
    n.into_iter().map(unit).collect()
}

/// Neighbouring vertices of a curved face have normals at most 10.5° apart, and no vertex
/// normal is more than 10.5° from the normal of a triangle it belongs to: the face shades as
/// one smooth surface.
fn assert_smooth(p: &[P], n: &[P], f: &FaceDisplay) {
    for t in tris(f) {
        let c = unit(cross(sub(p[t[1]], p[t[0]]), sub(p[t[2]], p[t[0]])));
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let ang = dot(n[a], n[b]).clamp(-1.0, 1.0).acos();
            assert!(
                ang <= 10.5f64.to_radians(),
                "{}: normals jump {ang} rad",
                names(f)
            );
            let off = dot(n[a], c).clamp(-1.0, 1.0).acos();
            assert!(
                off <= 10.5f64.to_radians(),
                "{}: normal {off} rad off its triangle",
                names(f)
            );
        }
    }
}

fn names(f: &FaceDisplay) -> String {
    f.names.join(" ")
}

const PLATE_WITH_HOLE: &str = r#"runmat 0.7
std 1

part plate {
    sketch s1 on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        l2 = line(start = [80 mm, 0 mm], end = [80 mm, 50 mm])
        l3 = line(start = [80 mm, 50 mm], end = [0 mm, 50 mm])
        l4 = line(start = [0 mm, 50 mm], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s1, distance = 10 mm)
    feature h = hole(
        face = feature.body.face.end,
        at = { a = point(60 mm, 10 mm, 10 mm) },
        diameter = 4 mm,
        through_all
    )
}
"#;

#[test]
fn a_through_hole_is_a_round_watertight_cylinder() {
    let d = compile("/p/plate.part", PLATE_WITH_HOLE);
    assert_eq!(d.bodies.len(), 1);
    let b = &d.bodies[0];
    assert_watertight(b);
    let r = 2.0 * MM;
    let (w, dp, t) = (80.0 * MM, 50.0 * MM, 10.0 * MM);
    let v_exact = w * dp * t - PI * r * r * t;
    let v = volume(b);
    // Chords at 8° lose (1 − sin θ/θ) ≈ 0.33 % of the hole's area.
    assert!(
        (v - v_exact).abs() < 0.006 * PI * r * r * t,
        "volume {v} vs {v_exact}"
    );
    let a_exact = 2.0 * (w * dp - PI * r * r) + 2.0 * (w + dp) * t + 2.0 * PI * r * t;
    let a = area(b);
    assert!(
        (a - a_exact).abs() < 0.006 * 2.0 * PI * r * (t + r),
        "area {a} vs {a_exact}"
    );
    // The wall: every vertex on the cylinder, 36 to 72 segments around (5° to 10°), and
    // normals radial and smooth.
    let axis = [60.0 * MM, 10.0 * MM];
    let wall = b
        .faces
        .iter()
        .find(|f| {
            let p = pts(f);
            !p.is_empty()
                && p.iter()
                    .all(|q| ((q[0] - axis[0]).hypot(q[1] - axis[1]) - r).abs() < 1e-6)
        })
        .unwrap_or_else(|| {
            panic!(
                "no round hole wall: {:?}",
                b.faces.iter().map(names).collect::<Vec<_>>()
            )
        });
    let p = pts(wall);
    let top: Vec<&P> = p.iter().filter(|q| (q[2] - t).abs() < 1e-9).collect();
    assert!(
        (36..=72).contains(&top.len()),
        "{} segments around the hole",
        top.len()
    );
    let normals = vertex_normals(wall);
    for (q, n) in p.iter().zip(&normals) {
        let radial = unit([q[0] - axis[0], q[1] - axis[1], 0.0]);
        // The wall faces into the hole: its normal points at the axis.
        let ang = dot(*n, [-radial[0], -radial[1], 0.0])
            .clamp(-1.0, 1.0)
            .acos();
        assert!(
            ang < 3f64.to_radians(),
            "normal {n:?} at {q:?} is {ang} rad off"
        );
    }
    // The top face's loop around the hole uses the wall's own points (no cracks).
    for f in &b.faces {
        for q in pts(f) {
            let dr = (q[0] - axis[0]).hypot(q[1] - axis[1]);
            assert!(
                dr >= r - 1e-6,
                "face {} has a point inside the hole: {q:?}",
                names(f)
            );
        }
    }
}

const BENT_PLATE: &str = r#"runmat 0.7
std 1

part plate_sm {
    sketch s_plate on origin.xy {
        e1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        e2 = line(start = [80 mm, 0 mm], end = [80 mm, 40 mm])
        e3 = line(start = [80 mm, 40 mm], end = [0 mm, 40 mm])
        e4 = line(start = [0 mm, 40 mm], end = [0 mm, 0 mm])
    }
    feature base = sheet_metal.base_flange(
        profile = sketch.s_plate,
        rules = { thickness = 2 mm, bend_radius = 3 mm, rule = k_factor(k = 0.44) }
    )
    sketch s_line on origin.xy {
        bl = line(start = [50 mm, -5 mm], end = [50 mm, 45 mm])
    }
    feature bend = sheet_metal.sketched_bend(
        face = feature.base.face.sheet(plate, ordinal = 1),
        line = [s_line, bl],
        angle = 90 deg,
        fixed = positive
    )
}
"#;

#[test]
fn a_bend_is_smooth_round_and_watertight() {
    let d = compile("/p/plate_sm.part", BENT_PLATE);
    assert_eq!(d.bodies.len(), 1);
    let b = &d.bodies[0];
    assert_watertight(b);
    let (t, r, k, w) = (2.0 * MM, 3.0 * MM, 0.44, 40.0 * MM);
    let theta = PI / 2.0;
    let v_exact = 80.0 * MM * w * t + w * t * t * theta * (0.5 - k);
    let v = volume(b);
    for f in &b.faces {
        let rs: Vec<f64> = pts(f)
            .iter()
            .map(|q| (q[0] - 0.06).hypot(q[1] - 0.01))
            .collect();
        let zs: Vec<f64> = pts(f).iter().map(|q| q[2]).collect();
        eprintln!(
            "face {} tris {} pts {} r {:?} z {:?} a {}",
            names(f),
            f.triangles.len() / 3,
            f.positions.len() / 3,
            (
                rs.iter().cloned().fold(f64::INFINITY, f64::min),
                rs.iter().cloned().fold(0.0, f64::max)
            ),
            (
                zs.iter().cloned().fold(f64::INFINITY, f64::min),
                zs.iter().cloned().fold(0.0, f64::max)
            ),
            {
                let p = pts(f);
                tris(f)
                    .map(|t| {
                        let [a, bb, c] = t.map(|i| p[i]);
                        len(sub(bb, a)).max(len(sub(c, bb))).max(len(sub(a, c)))
                    })
                    .fold(0.0, f64::max)
            }
        );
        let p = pts(f);
        let mut vv = 0.0;
        for t in tris(f) {
            let [a, bb, c] = t.map(|i| p[i]);
            vv += dot(a, cross(bb, c)) / 6.0;
        }
        eprintln!("   vol contribution {}", vv * 1e9);
        if names(f).contains("wall") {
            for t in tris(f) {
                let [a, bb, c] = t.map(|i| p[i]);
                let n = unit(cross(sub(bb, a), sub(c, a)));
                let m = [
                    (a[0] + bb[0] + c[0]) / 3.0 - 0.06,
                    (a[1] + bb[1] + c[1]) / 3.0 - 0.01,
                    0.0,
                ];
                let rad = unit(m);
                let d = -dot(n, rad);
                if d < 0.99 {
                    eprintln!("   bad tri {:?} {:?} {:?} d {}", a, bb, c, d);
                }
            }
        }
    }
    // Chords at 8° on radii 3 and 5 mm.
    assert!(
        (v - v_exact).abs() < 0.004 * w * theta * ((r + t) * (r + t) - r * r) / 2.0,
        "volume {v} vs {v_exact}"
    );
    // Bend faces: the two cylinders of the bend, each 9 to 18 segments over 90°, normals
    // varying smoothly (neighbouring vertices across one segment differ by at most 10°).
    let bends: Vec<&FaceDisplay> = b
        .faces
        .iter()
        .filter(|f| names(f).contains("face.bend("))
        .collect();
    assert_eq!(
        bends.len(),
        2,
        "{:?}",
        b.faces.iter().map(names).collect::<Vec<_>>()
    );
    for f in bends {
        let p = pts(f);
        let n = vertex_normals(f);
        // Points on the bend's edge at y = 0: one per segment end.
        let edge = p.iter().filter(|q| q[1].abs() < 1e-9).count();
        assert!(
            (10..=19).contains(&edge),
            "{}: {} points along the 90° bend",
            names(f),
            edge
        );
        assert_smooth(&p, &n, f);
        // Shared vertices: one position per point on the face.
        let mut seen = std::collections::BTreeSet::new();
        for q in &p {
            assert!(
                seen.insert((q[0].to_bits(), q[1].to_bits(), q[2].to_bits())),
                "{}: duplicate vertex {q:?}",
                names(f)
            );
        }
    }
}

const FILLETED_PLATE: &str = r#"runmat 0.7
std 1

part plate {
    sketch s1 on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        l2 = line(start = [80 mm, 0 mm], end = [80 mm, 50 mm])
        l3 = line(start = [80 mm, 50 mm], end = [0 mm, 50 mm])
        l4 = line(start = [0 mm, 50 mm], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s1, distance = 10 mm)
    feature round = fillet(
        edges = [edges_between(feature.body.face.side(s1.l1), feature.body.face.end)],
        radius = 1.5 mm
    )
}
"#;

#[test]
fn a_fillet_is_a_round_watertight_quarter_cylinder() {
    let d = compile("/p/plate.part", FILLETED_PLATE);
    let b = &d.bodies[0];
    assert_watertight(b);
    let r = 1.5 * MM;
    let l = 80.0 * MM;
    let v_exact = 80.0 * MM * 50.0 * MM * 10.0 * MM - (1.0 - PI / 4.0) * r * r * l;
    let v = volume(b);
    assert!(
        (v - v_exact).abs() < 0.02 * (1.0 - PI / 4.0) * r * r * l,
        "volume {v} vs {v_exact}"
    );
    // The fillet: the face within the radius of the filleted edge (y = 0, z = 10 mm).
    let fillet = b
        .faces
        .iter()
        .find(|f| {
            let p = pts(f);
            p.len() > 4
                && p.iter()
                    .all(|q| q[1] <= r + 1e-9 && q[2] >= 10.0 * MM - r - 1e-9)
        })
        .expect("fillet face");
    for q in pts(fillet) {
        let d = (q[1] - r).hypot(q[2] - (10.0 * MM - r));
        assert!((d - r).abs() < 1e-7, "fillet point {q:?} off the radius");
    }
    let p = pts(fillet);
    let n = vertex_normals(fillet);
    let edge = p.iter().filter(|q| q[0].abs() < 1e-9).count();
    assert!(
        (10..=19).contains(&edge),
        "{edge} points along the fillet's end arc"
    );
    assert_smooth(&p, &n, fillet);
}

const BRACKET: &str = r#"runmat 0.7
std 1

part bracket {
    sketch s1 on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        l2 = line(start = [80 mm, 0 mm], end = [80 mm, 50 mm])
        l3 = line(start = [80 mm, 50 mm], end = [0 mm, 50 mm])
        l4 = line(start = [0 mm, 50 mm], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s1, distance = 10 mm)
    sketch s_cut on origin.xy {
        c1 = line(start = [20 mm, 20 mm], end = [40 mm, 20 mm])
        c2 = line(start = [40 mm, 20 mm], end = [40 mm, 30 mm])
        c3 = line(start = [40 mm, 30 mm], end = [20 mm, 30 mm])
        c4 = line(start = [20 mm, 30 mm], end = [20 mm, 20 mm])
    }
    feature cut = extrude(profile = sketch.s_cut, through_all, mode = cut)
    feature h = hole(
        face = feature.body.face.end,
        at = { a = point(60 mm, 10 mm, 10 mm) },
        diameter = 4 mm,
        through_all
    )
    feature round = fillet(
        edges = [edges_between(feature.body.face.side(s1.l1), feature.body.face.end)],
        radius = 1.5 mm
    )
}
"#;

#[test]
fn the_showcase_bracket_is_watertight_with_its_cut_hole_and_fillet() {
    let d = compile("/p/bracket.part", BRACKET);
    let b = &d.bodies[0];
    assert_watertight(b);
    let (r, rf) = (2.0 * MM, 1.5 * MM);
    let v_exact = 80.0 * MM * 50.0 * MM * 10.0 * MM
        - 20.0 * MM * 10.0 * MM * 10.0 * MM
        - PI * r * r * 10.0 * MM
        - (1.0 - PI / 4.0) * rf * rf * 80.0 * MM;
    let v = volume(b);
    assert!(
        (v - v_exact).abs() < 0.006 * PI * r * r * 10.0 * MM + 0.01 * rf * rf * 80.0 * MM,
        "volume {v} vs {v_exact}"
    );
}
