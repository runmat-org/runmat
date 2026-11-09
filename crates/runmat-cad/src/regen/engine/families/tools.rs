//! Rigid-motion helpers and the bend tools the sheet-metal lowerings share.

use super::*;

/// `R v` for a rigid motion's rotation.
pub(in crate::regen::engine) fn apply_rotation(t: &Transform, v: [f64; 3]) -> [f64; 3] {
    let r = &t.rotation;
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

/// The inverse of a rigid motion: `p ↦ Rᵀ (p − t)`.
pub(in crate::regen::engine) fn inverse(t: &Transform) -> Transform {
    let mut r = [[0.0; 3]; 3];
    for (i, row) in r.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = t.rotation[j][i];
        }
    }
    let rt = Transform {
        rotation: r,
        translation: [0.0; 3],
    };
    let tt = apply_rotation(&rt, t.translation);
    Transform {
        rotation: r,
        translation: [-tt[0], -tt[1], -tt[2]],
    }
}

/// `a ∘ b`: `b` first, then `a`.
pub(in crate::regen::engine) fn compose(a: &Transform, b: &Transform) -> Transform {
    let mut r = [[0.0; 3]; 3];
    for (i, row) in r.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a.rotation[i][k] * b.rotation[k][j]).sum();
        }
    }
    Transform {
        rotation: r,
        translation: add(apply_rotation(a, b.translation), a.translation),
    }
}

pub(super) fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    dot(sub(a, b), sub(a, b)).sqrt()
}

pub(in crate::regen::engine) fn plane_normal(g: &FaceGeom) -> Option<[f64; 3]> {
    match g.surface {
        SurfaceKind::Plane { normal } => Some(normalize(normal)),
        _ => None,
    }
}

/// The rectangle a face's box makes in a frame on it, inset.
pub(super) fn face_box(fr: &Frame, g: &FaceGeom, inset: f64) -> Vec<(String, [f64; 2])> {
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for k in 0..8 {
        let c = [
            if k & 1 == 0 {
                g.bbox_min[0]
            } else {
                g.bbox_max[0]
            },
            if k & 2 == 0 {
                g.bbox_min[1]
            } else {
                g.bbox_max[1]
            },
            if k & 4 == 0 {
                g.bbox_min[2]
            } else {
                g.bbox_max[2]
            },
        ];
        let d = sub(c, fr.origin);
        let q = [dot(d, fr.x), dot(d, fr.y)];
        for a in 0..2 {
            lo[a] = lo[a].min(q[a]);
            hi[a] = hi[a].max(q[a]);
        }
    }
    vec![
        ("bottom".to_string(), [lo[0] + inset, lo[1] + inset]),
        ("right".to_string(), [hi[0] - inset, lo[1] + inset]),
        ("top".to_string(), [hi[0] - inset, hi[1] - inset]),
        ("left".to_string(), [lo[0] + inset, hi[1] - inset]),
    ]
}

/// The annular sector of a bend swept along the sketch normal, labelled by the bend key.
/// A bend of a sheet section in `fr` (origin on the inner face where the bend starts, x
/// along the sheet, y the inner face's normal for an upward bend; z along the bend line):
/// `up` turns towards y about a centre `r` above the face, otherwise away from it about a
/// centre `r` below the outer face. Prism over `w` along z.
pub(in crate::regen::engine) fn sheet_bend_tool(
    fr: Frame,
    up: bool,
    r: f64,
    t: f64,
    angle: f64,
    key: &str,
    w: (f64, f64),
) -> LabeledTool {
    use std::f64::consts::FRAC_PI_2;
    let (c, from, to) = if up {
        ([0.0, r], -FRAC_PI_2, -FRAC_PI_2 + angle)
    } else {
        ([0.0, -t - r], FRAC_PI_2 - angle, FRAC_PI_2)
    };
    let (ri, ro) = (r, r + t);
    let at = |rad: f64, a: f64| [c[0] + rad * a.cos(), c[1] + rad * a.sin()];
    let sid = |s: &str| crate::ids::SketchEntityId::new(s);
    let profile = CanonicalProfile {
        frame: fr,
        loops: vec![vec![
            ProfileSegment {
                curve: sid("in"),
                kind: SegmentKind::Arc {
                    center: c,
                    clockwise: false,
                },
                start: at(ri, from),
                end: at(ri, to),
            },
            line_seg("e1", at(ri, to), at(ro, to)),
            ProfileSegment {
                curve: sid("out"),
                kind: SegmentKind::Arc {
                    center: c,
                    clockwise: true,
                },
                start: at(ro, to),
                end: at(ro, from),
            },
            line_seg("e0", at(ro, from), at(ri, from)),
        ]],
        merge_collinear: false,
    };
    let mut tool = LabeledTool::new(ToolShape::prism(profile, w.0, w.1));
    // Inner and outer as the bend sees them: an upward bend's inner face is on y's side.
    let (inner, outer) = if up { ("in", "out") } else { ("out", "in") };
    tool.sides.push((
        sid(inner),
        FaceLabel::key(Role::BendFace, key).with_ordinal(0),
    ));
    tool.sides.push((
        sid(outer),
        FaceLabel::key(Role::BendFace, key).with_ordinal(1),
    ));
    tool.sides.push((
        sid("e0"),
        FaceLabel::key(Role::SheetEdge, key).with_ordinal(4),
    ));
    tool.sides.push((
        sid("e1"),
        FaceLabel::key(Role::SheetEdge, key).with_ordinal(5),
    ));
    tool.start = Some(FaceLabel::key(Role::SheetEdge, key).with_ordinal(2));
    tool.end = Some(FaceLabel::key(Role::SheetEdge, key).with_ordinal(3));
    tool
}

pub(in crate::regen::engine) fn bend_tool(
    frame: Frame,
    b: &crate::model::sheet_metal::SectionBend,
    depth: f64,
    body: Option<FaceLabel>,
) -> LabeledTool {
    let (ri, ro) = (b.inner_radius, b.inner_radius + b.thickness);
    let c = b.center;
    let at = |r: f64, a: f64| [c[0] + r * a.cos(), c[1] + r * a.sin()];
    let sid = |s: &str| crate::ids::SketchEntityId::new(format!("{}.{}", b.key, s));
    let profile = CanonicalProfile {
        frame,
        loops: vec![vec![
            ProfileSegment {
                curve: sid("in"),
                kind: SegmentKind::Arc {
                    center: c,
                    clockwise: false,
                },
                start: at(ri, b.from),
                end: at(ri, b.to),
            },
            line_seg(&format!("{}.e1", b.key), at(ri, b.to), at(ro, b.to)),
            ProfileSegment {
                curve: sid("out"),
                kind: SegmentKind::Arc {
                    center: c,
                    clockwise: true,
                },
                start: at(ro, b.to),
                end: at(ro, b.from),
            },
            line_seg(&format!("{}.e0", b.key), at(ro, b.from), at(ri, b.from)),
        ]],
        merge_collinear: false,
    };
    let mut tool = LabeledTool::new(ToolShape::prism(profile, 0.0, depth));
    tool.sides.push((
        sid("in"),
        FaceLabel::key(Role::BendFace, b.key.clone()).with_ordinal(0),
    ));
    tool.sides.push((
        sid("out"),
        FaceLabel::key(Role::BendFace, b.key.clone()).with_ordinal(1),
    ));
    tool.sides.push((
        sid("e0"),
        FaceLabel::key(Role::SheetEdge, b.key.clone()).with_ordinal(4),
    ));
    tool.sides.push((
        sid("e1"),
        FaceLabel::key(Role::SheetEdge, b.key.clone()).with_ordinal(5),
    ));
    tool.start = Some(FaceLabel::key(Role::SheetEdge, b.key.clone()).with_ordinal(2));
    tool.end = Some(FaceLabel::key(Role::SheetEdge, b.key.clone()).with_ordinal(3));
    tool.body = body;
    tool
}
