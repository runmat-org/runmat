//! The lofted bend: a sheet formed between two open profiles on parallel planes (a transition
//! or a hopper).
//!
//! Each profile is drawn with lines and arcs. Each arc is formed in `facets` straight bends, so
//! the arc becomes `facets` lines between points on it. The profile with fewer points has its
//! longest pieces halved until both have as many, so every corner stays a corner. The second
//! profile is reversed when its far end lies nearer the first profile's start.
//!
//! Each profile is thickened in its own plane by the sheet's thickness, to the left of its
//! direction as seen along the first profile's plane normal. The two thin sections are lofted
//! with straight rulings: each pair of matching pieces makes one flat or twisted facet.

use super::*;

/// The profile's points in world coordinates, each arc split into `facets` pieces.
fn points(frame: &Frame, segs: &[ProfileSegment], facets: u32) -> Vec<[f64; 3]> {
    let world = |p: [f64; 2]| {
        add(
            frame.origin,
            add(scale(frame.x, p[0]), scale(frame.y, p[1])),
        )
    };
    let mut out = vec![world(segs[0].start)];
    for s in segs {
        if let SegmentKind::Arc { center, clockwise } = s.kind {
            let a0 = (s.start[1] - center[1]).atan2(s.start[0] - center[0]);
            let a1 = (s.end[1] - center[1]).atan2(s.end[0] - center[0]);
            let r = (s.start[0] - center[0]).hypot(s.start[1] - center[1]);
            let tau = std::f64::consts::TAU;
            let sweep = if clockwise {
                -(a0 - a1).rem_euclid(tau)
            } else {
                (a1 - a0).rem_euclid(tau)
            };
            for k in 1..facets {
                let a = a0 + sweep * k as f64 / facets as f64;
                out.push(world([center[0] + r * a.cos(), center[1] + r * a.sin()]));
            }
        }
        out.push(world(s.end));
    }
    out
}

/// Halves the longest piece of `pts` until it has `n` points.
fn refine(pts: &mut Vec<[f64; 3]>, n: usize) {
    while pts.len() < n {
        let k = (0..pts.len() - 1)
            .max_by(|&a, &b| {
                let la = norm(sub(pts[a + 1], pts[a]));
                let lb = norm(sub(pts[b + 1], pts[b]));
                la.total_cmp(&lb)
            })
            .expect("two points");
        let mid = scale(add(pts[k], pts[k + 1]), 0.5);
        pts.insert(k + 1, mid);
    }
}

fn norm(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

/// The closed thin section of polyline `pts` in `frame`'s plane, offset by `t` along
/// `cross(z, direction)`, in `frame`'s coordinates; `None` where the polyline doubles back.
fn thin(
    frame: &Frame,
    pts: &[[f64; 3]],
    z: [f64; 3],
    t: f64,
    key: &str,
) -> Option<CanonicalProfile> {
    let n = pts.len();
    let dirs: Vec<[f64; 3]> = (0..n - 1)
        .map(|k| normalize(sub(pts[k + 1], pts[k])))
        .collect();
    let mut off = Vec::with_capacity(n);
    for k in 0..n {
        let normal = |d: [f64; 3]| cross(z, d);
        let o = if k == 0 {
            scale(normal(dirs[0]), t)
        } else if k == n - 1 {
            scale(normal(dirs[n - 2]), t)
        } else {
            let (n1, n2) = (normal(dirs[k - 1]), normal(dirs[k]));
            let m = normalize(add(n1, n2));
            let c = dot(m, n1);
            if c < 0.2 {
                return None;
            }
            scale(m, t / c)
        };
        off.push(add(pts[k], o));
    }
    let local = |p: [f64; 3]| {
        let q = sub(p, frame.origin);
        [dot(q, frame.x), dot(q, frame.y)]
    };
    let ring: Vec<[f64; 2]> = pts
        .iter()
        .map(|p| local(*p))
        .chain(off.iter().rev().map(|p| local(*p)))
        .collect();
    let m = ring.len();
    Some(CanonicalProfile {
        frame: *frame,
        loops: vec![(0..m)
            .map(|k| {
                super::super::families::line_seg(&format!("{key}.{k}"), ring[k], ring[(k + 1) % m])
            })
            .collect()],
        merge_collinear: false,
    })
}

impl<'a, 'b> Run<'a, 'b> {
    pub(in crate::regen::engine) fn lofted_bend(
        &mut self,
        i: usize,
        id: &FeatureId,
        profiles: &[ProfileRef; 2],
        facets: u32,
        rules: &crate::model::sheet_metal::SheetMetalRules,
        naming: &NamingContext,
    ) -> Result<Compiled, Halt> {
        let invalid = |me: &mut Self, param: &str, message: &str| {
            me.error(
                id,
                Some(param),
                DiagCode::RegenInvalid,
                message.into(),
                Vec::new(),
            )
        };
        if facets == 0 {
            return Err(invalid(
                self,
                "facets",
                "a lofted bend forms each arc in at least one bend",
            ));
        }
        let rules = self.sheet_rules(id, rules)?;
        let t = rules.thickness;
        let (fa, sa) = self.chain(i, id, "profiles[0]", &profiles[0])?;
        let (fb, sb) = self.chain(i, id, "profiles[1]", &profiles[1])?;
        for (k, segs) in [&sa, &sb].iter().enumerate() {
            let param = format!("profiles[{k}]");
            if segs.is_empty() {
                return Err(invalid(
                    self,
                    &param,
                    "a lofted bend's profile has lines or arcs",
                ));
            }
            if segs
                .iter()
                .any(|s| !matches!(s.kind, SegmentKind::Line | SegmentKind::Arc { .. }))
            {
                return Err(invalid(
                    self,
                    &param,
                    "a lofted bend's profiles are drawn with lines and arcs",
                ));
            }
        }
        if dot(fa.z, fb.z).abs() < 1.0 - 1e-9 {
            return Err(invalid(
                self,
                "profiles",
                "a lofted bend's profiles lie on parallel planes",
            ));
        }
        if dot(sub(fb.origin, fa.origin), fa.z).abs() < 1e-9 {
            return Err(invalid(
                self,
                "profiles",
                "a lofted bend's profiles lie on two different planes",
            ));
        }
        let mut pa = points(&fa, &sa, facets);
        let mut pb = points(&fb, &sb, facets);
        let (a0, an) = (pa[0], pa[pa.len() - 1]);
        let (b0, bn) = (pb[0], pb[pb.len() - 1]);
        if norm(sub(a0, bn)) + norm(sub(an, b0)) < norm(sub(a0, b0)) + norm(sub(an, bn)) {
            pb.reverse();
        }
        let n = pa.len().max(pb.len());
        refine(&mut pa, n);
        refine(&mut pb, n);
        let key = |s: &str| format!("{}.{s}", id.0);
        // Both sections carry the same curve keys, so each facet's face takes its piece's key.
        let (Some(ta), Some(tb)) = (
            thin(&fa, &pa, fa.z, t, &id.0),
            thin(&fb, &pb, fa.z, t, &id.0),
        ) else {
            return Err(invalid(
                self,
                "profiles",
                "a lofted bend's profile turns back on itself too sharply for its thickness",
            ));
        };
        let curves: Vec<_> = ta.loops.iter().flatten().map(|s| s.curve.clone()).collect();
        let mut tool = LabeledTool::new(ToolShape::Loft {
            profiles: vec![ta, tb],
        });
        for c in curves {
            tool.sides
                .push((c.clone(), FaceLabel::key(Role::SheetFace, c.0.clone())));
        }
        tool.start = Some(FaceLabel::key(Role::SheetEdge, key("first")));
        tool.end = Some(FaceLabel::key(Role::SheetEdge, key("second")));
        Ok(op(
            labeled(vec![tool], Vec::new(), Combine::New, false),
            Vec::new(),
            naming.clone(),
        ))
    }
}
