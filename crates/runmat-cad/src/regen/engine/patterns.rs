//! Pattern layouts to instances: linear, grid, circular (equal or stepped), table,
//! curve-driven (translated or turned along a rotation-minimizing frame), sketch-driven, fill
//! (square, staggered and ring lattices inside a boundary) and variable (each instance with its
//! own values, the seeds rebuilt per instance).

use super::*;
use crate::model::datum::CurveRef;
use crate::model::detail::substitute_inputs;
use crate::model::sketch::{PointRole, SketchGeometry};
use crate::regen::records::FeatureRecord;

fn length(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

/// The rotation taking unit `u` onto unit `v` about `u × v` (the least rotation).
fn rotation_between(u: [f64; 3], v: [f64; 3]) -> Transform {
    let c = cross(u, v);
    let s = length(c);
    let d = dot(u, v);
    if s < 1e-12 {
        if d > 0.0 {
            return Transform::IDENTITY;
        }
        // Opposite: half a turn about any perpendicular.
        let seed = if u[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let axis = normalize(cross(u, seed));
        return Transform::rotation_about([0.0; 3], axis, std::f64::consts::PI);
    }
    Transform::rotation_about([0.0; 3], scale(c, 1.0 / s), s.atan2(d))
}

/// Point and transported frame rotation at arc length `at` along a polyline: discrete
/// parallel transport (the rotation-minimizing frame of a polyline) from its first segment.
fn along(points: &[[f64; 3]], at: f64) -> ([f64; 3], Transform) {
    let mut rot = Transform::IDENTITY;
    let mut left = at;
    let mut prev: Option<[f64; 3]> = None;
    for (k, w) in points.windows(2).enumerate() {
        let d = sub(w[1], w[0]);
        let l = length(d);
        if l < 1e-15 {
            continue;
        }
        let t = scale(d, 1.0 / l);
        if let Some(p) = prev {
            rot = rotation_between(p, t).compose(&rot);
        }
        prev = Some(t);
        let last = k + 2 == points.len();
        if left <= l + 1e-12 || last {
            return (add(w[0], scale(t, left.min(l))), rot);
        }
        left -= l;
    }
    (points[0], rot)
}

fn inside_polygon(poly: &[[f64; 2]], q: [f64; 2]) -> bool {
    let mut inside = false;
    let n = poly.len();
    for k in 0..n {
        let (a, b) = (poly[k], poly[(k + 1) % n]);
        if (a[1] > q[1]) != (b[1] > q[1]) {
            let x = a[0] + (q[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
            if q[0] < x {
                inside = !inside;
            }
        }
    }
    inside
}

fn polygon_distance(poly: &[[f64; 2]], q: [f64; 2]) -> f64 {
    let n = poly.len();
    let mut best = f64::INFINITY;
    for k in 0..n {
        let (a, b) = (poly[k], poly[(k + 1) % n]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let l2 = dx * dx + dy * dy;
        let t = if l2 > 0.0 {
            (((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / l2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (px, py) = (a[0] + t * dx - q[0], a[1] + t * dy - q[1]);
        best = best.min((px * px + py * py).sqrt());
    }
    best
}

impl<'a, 'b> Run<'a, 'b> {
    /// A curve as a polyline in part coordinates.
    pub(super) fn curve_points(
        &mut self,
        i: usize,
        id: &FeatureId,
        c: &CurveRef,
    ) -> Result<Vec<[f64; 3]>, Halt> {
        match c {
            CurveRef::Feature { feature } => {
                self.gate_feature(i, id, feature, "curve")?;
                match self.st.records.get(feature) {
                    Some(FeatureRecord::Curve(v)) => {
                        let mut pts = v.points.clone();
                        if v.closed && pts.len() > 2 {
                            pts.push(pts[0]);
                        }
                        Ok(pts)
                    }
                    _ => Err(self.error(
                        id,
                        Some("curve"),
                        DiagCode::RefKindMismatch,
                        format!("`{}` is not a curve", feature),
                        Vec::new(),
                    )),
                }
            }
            CurveRef::Edge { edge } => {
                let r = self.single(i, id, "curve", edge, Some(TopoKind::Edge))?;
                let h = *r.entities().iter().next().expect("one");
                let g = self.st.bodies[h.body as usize].topology.edges[h.index as usize].geom;
                Ok(vec![g.start, g.end])
            }
            CurveRef::Sketch { sketch } => {
                let (pts, _) = self.path(i, id, "curve", &ProfileRef::sketch(&sketch.0))?;
                Ok(pts)
            }
        }
    }

    fn rotation_instance(&self, pt: [f64; 3], dir: [f64; 3], a: f64, key: InstanceKey) -> Instance {
        let col = |v: [f64; 3]| rotate(v, dir, a);
        let (cx, cy, cz) = (
            col([1.0, 0.0, 0.0]),
            col([0.0, 1.0, 0.0]),
            col([0.0, 0.0, 1.0]),
        );
        let rotation = [
            [cx[0], cy[0], cz[0]],
            [cx[1], cy[1], cz[1]],
            [cx[2], cy[2], cz[2]],
        ];
        let t = sub(pt, rot_apply(&rotation, pt));
        Instance {
            key,
            transform: Transform {
                rotation,
                translation: t,
            },
        }
    }

    pub(super) fn pattern(
        &mut self,
        i: usize,
        id: &FeatureId,
        p: &Pattern,
    ) -> Result<Compiled, Halt> {
        let mut instances = Vec::new();
        let mut variable: Vec<&VariableInstance> = Vec::new();
        match &p.layout {
            PatternLayout::Linear {
                direction,
                spacing,
                count,
            } => {
                let (_, d) = self.axis(i, id, "direction", direction)?;
                let s = self.value(id, "spacing", spacing, Dimension::LENGTH)?;
                let n = self.count(id, "count", count)?;
                for k in 1..n {
                    instances.push(Instance {
                        key: InstanceKey::Linear { step: k },
                        transform: Transform::translation(scale(d, s * k as f64)),
                    });
                }
            }
            PatternLayout::Grid {
                first,
                first_spacing,
                first_count,
                second,
                second_spacing,
                second_count,
            } => {
                let (_, d1) = self.axis(i, id, "first", first)?;
                let s1 = self.value(id, "first_spacing", first_spacing, Dimension::LENGTH)?;
                let n1 = self.count(id, "first_count", first_count)?;
                let (_, d2) = self.axis(i, id, "second", second)?;
                let s2 = self.value(id, "second_spacing", second_spacing, Dimension::LENGTH)?;
                let n2 = self.count(id, "second_count", second_count)?;
                for a in 0..n1 {
                    for b in 0..n2 {
                        if a == 0 && b == 0 {
                            continue;
                        }
                        let t = add(scale(d1, s1 * a as f64), scale(d2, s2 * b as f64));
                        instances.push(Instance {
                            key: InstanceKey::Grid {
                                first: a,
                                second: b,
                            },
                            transform: Transform::translation(t),
                        });
                    }
                }
            }
            PatternLayout::Circular {
                axis,
                count,
                angle,
                equal_spacing,
            } => {
                let (pt, dir) = self.axis(i, id, "axis", axis)?;
                let n = self.count(id, "count", count)?;
                let total = self.value(id, "angle", angle, Dimension::ANGLE)?;
                let step = if *equal_spacing {
                    total / n as f64
                } else {
                    total
                };
                for k in 1..n {
                    instances.push(self.rotation_instance(
                        pt,
                        dir,
                        step * k as f64,
                        InstanceKey::Circular { step: k },
                    ));
                }
            }
            PatternLayout::Table { rows } => {
                for r in rows {
                    let t = [
                        self.value(id, "rows.dx", &r.dx, Dimension::LENGTH)?,
                        self.value(id, "rows.dy", &r.dy, Dimension::LENGTH)?,
                        self.value(id, "rows.dz", &r.dz, Dimension::LENGTH)?,
                    ];
                    instances.push(Instance {
                        key: InstanceKey::Row { key: r.key.clone() },
                        transform: Transform::translation(t),
                    });
                }
            }
            PatternLayout::Curve {
                curve,
                count,
                spacing,
                orientation,
            } => {
                let pts = self.curve_points(i, id, curve)?;
                let total: f64 = pts.windows(2).map(|w| length(sub(w[1], w[0]))).sum();
                let n = self.count(id, "count", count)?;
                let closed = pts.len() > 2 && length(sub(pts[0], pts[pts.len() - 1])) < 1e-9;
                let step = match spacing {
                    Some(s) => self.value(id, "spacing", s, Dimension::LENGTH)?,
                    None if closed => total / n as f64,
                    None => total / (n.max(2) - 1) as f64,
                };
                if total <= 0.0 || step * (n - 1) as f64 > total * (1.0 + 1e-9) {
                    return Err(self.error(
                        id,
                        Some("spacing"),
                        DiagCode::RegenInvalid,
                        format!(
                            "{} instances {} apart run past the curve ({} long)",
                            n, step, total
                        ),
                        Vec::new(),
                    ));
                }
                let (p0, _) = along(&pts, 0.0);
                for k in 1..n {
                    let (pk, rot) = along(&pts, step * k as f64);
                    let turn = match orientation {
                        CurveOrientation::Translate => Transform::IDENTITY,
                        CurveOrientation::AlignToCurve => rot,
                    };
                    let t = Transform::translation(pk)
                        .compose(&turn)
                        .compose(&Transform::translation(scale(p0, -1.0)));
                    instances.push(Instance {
                        key: InstanceKey::Curve { step: k },
                        transform: t,
                    });
                }
            }
            PatternLayout::Sketch {
                sketch,
                points,
                reference,
            } => {
                self.gate_feature(i, id, sketch, "sketch")?;
                let Some((solved, frame)) = self.st.sketches.get(sketch).cloned() else {
                    return Err(self.error(
                        id,
                        Some("sketch"),
                        DiagCode::RefKindMismatch,
                        format!("`{}` is not a sketch", sketch),
                        Vec::new(),
                    ));
                };
                let Some(FeatureKind::Sketch(sk)) = self.doc.feature(sketch).map(|f| &f.kind)
                else {
                    unreachable!("solved sketch")
                };
                let mut pts: Vec<(crate::ids::SketchEntityId, [f64; 3])> = Vec::new();
                for e in &sk.entities {
                    if !matches!(e.geometry, SketchGeometry::Point { .. }) {
                        continue;
                    }
                    if !points.is_empty() && !points.contains(&e.id) {
                        continue;
                    }
                    if let Some(q) = solved.point(&e.id, PointRole::At) {
                        pts.push((e.id.clone(), frame.to_world(q)));
                    }
                }
                let base = match reference {
                    Some(r) => solved
                        .point(r, PointRole::At)
                        .map(|q| (r.clone(), frame.to_world(q))),
                    None => pts.first().cloned(),
                };
                let Some((base_id, base_at)) = base else {
                    return Err(self.error(
                        id,
                        Some("sketch"),
                        DiagCode::RefEmpty,
                        format!("`{}` has no points to place instances on", sketch),
                        Vec::new(),
                    ));
                };
                for (pid, at) in pts {
                    if pid == base_id {
                        continue;
                    }
                    instances.push(Instance {
                        key: InstanceKey::Point {
                            point: pid.0.clone(),
                        },
                        transform: Transform::translation(sub(at, base_at)),
                    });
                }
            }
            PatternLayout::Fill {
                boundary,
                spacing,
                lattice,
                direction,
                margin,
                seed,
            } => {
                let prof = self.profile(i, id, "boundary", boundary)?;
                let frame = prof.frame;
                let s = self.value(id, "spacing", spacing, Dimension::LENGTH)?;
                if s <= 0.0 {
                    return Err(self.error(
                        id,
                        Some("spacing"),
                        DiagCode::ValueFailed,
                        "fill spacing must be positive".into(),
                        Vec::new(),
                    ));
                }
                let m = match margin {
                    Some(x) => self.value(id, "margin", x, Dimension::LENGTH)?,
                    None => 0.0,
                };
                let polys: Vec<Vec<[f64; 2]>> = prof
                    .loops
                    .iter()
                    .map(|l| {
                        let mut out = Vec::new();
                        for seg in l {
                            let smp = segment_samples(seg);
                            out.extend_from_slice(&smp[..smp.len() - 1]);
                        }
                        out
                    })
                    .collect();
                let inside = |q: [f64; 2]| {
                    // Inside the outer loop, outside every hole, `m` from every boundary.
                    let parity = polys.iter().filter(|p| inside_polygon(p, q)).count() % 2 == 1;
                    parity && polys.iter().all(|p| polygon_distance(p, q) >= m - 1e-12)
                };
                let local = |w: [f64; 3]| {
                    let d = sub(w, frame.origin);
                    [dot(d, frame.x), dot(d, frame.y)]
                };
                let c = match seed {
                    Some(sp) => local(self.point(i, id, "seed", sp)?),
                    None => {
                        let outer = &polys[0];
                        let n = outer.len().max(1) as f64;
                        let sx: f64 = outer.iter().map(|q| q[0]).sum();
                        let sy: f64 = outer.iter().map(|q| q[1]).sum();
                        [sx / n, sy / n]
                    }
                };
                let u = match direction {
                    Some(a) => {
                        let (_, d) = self.axis(i, id, "direction", a)?;
                        let q = [dot(d, frame.x), dot(d, frame.y)];
                        let l = (q[0] * q[0] + q[1] * q[1]).sqrt();
                        if l < 1e-9 {
                            return Err(self.error(
                                id,
                                Some("direction"),
                                DiagCode::RegenInvalid,
                                "the fill direction is normal to the boundary's plane".into(),
                                Vec::new(),
                            ));
                        }
                        [q[0] / l, q[1] / l]
                    }
                    None => [1.0, 0.0],
                };
                let v = [-u[1], u[0]];
                let mut lo = [f64::INFINITY; 2];
                let mut hi = [f64::NEG_INFINITY; 2];
                for q in polys.iter().flatten() {
                    for a in 0..2 {
                        lo[a] = lo[a].min(q[a]);
                        hi[a] = hi[a].max(q[a]);
                    }
                }
                let reach = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2)).sqrt()
                    + (c[0] - (lo[0] + hi[0]) / 2.0).abs()
                    + (c[1] - (lo[1] + hi[1]) / 2.0).abs();
                let to_world = |q: [f64; 2]| {
                    let off = [q[0] - c[0], q[1] - c[1]];
                    add(scale(frame.x, off[0]), scale(frame.y, off[1]))
                };
                let max_steps = (reach / s).ceil() as i64 + 1;
                if max_steps > 400 {
                    return Err(self.error(
                        id,
                        Some("spacing"),
                        DiagCode::RegenInvalid,
                        format!(
                            "a fill spacing of {} over this boundary makes too many instances",
                            s
                        ),
                        Vec::new(),
                    ));
                }
                match lattice {
                    FillLattice::Square | FillLattice::Staggered => {
                        let staggered = *lattice == FillLattice::Staggered;
                        let row = if staggered { s * 3f64.sqrt() / 2.0 } else { s };
                        for j in -max_steps..=max_steps {
                            let shift = if staggered && j.rem_euclid(2) == 1 {
                                s / 2.0
                            } else {
                                0.0
                            };
                            for k in -max_steps..=max_steps {
                                if j == 0 && k == 0 {
                                    continue;
                                }
                                let a = k as f64 * s + shift;
                                let b = j as f64 * row;
                                let q = [c[0] + a * u[0] + b * v[0], c[1] + a * u[1] + b * v[1]];
                                if inside(q) {
                                    instances.push(Instance {
                                        key: InstanceKey::Lattice {
                                            i: k as i32,
                                            j: j as i32,
                                        },
                                        transform: Transform::translation(to_world(q)),
                                    });
                                }
                            }
                        }
                    }
                    FillLattice::Circular => {
                        for ring in 1..=max_steps as u32 {
                            let r = ring as f64 * s;
                            let n = ((std::f64::consts::TAU * r / s).floor() as u32).max(1);
                            for step in 0..n {
                                let a = std::f64::consts::TAU * step as f64 / n as f64;
                                let (sa, ca) = a.sin_cos();
                                let q = [
                                    c[0] + r * (ca * u[0] + sa * v[0]),
                                    c[1] + r * (ca * u[1] + sa * v[1]),
                                ];
                                if inside(q) {
                                    instances.push(Instance {
                                        key: InstanceKey::Ring { ring, step },
                                        transform: Transform::translation(to_world(q)),
                                    });
                                }
                            }
                        }
                    }
                }
            }
            PatternLayout::Variable { instances: vs } => {
                if !matches!(p.seeds, PatternSeeds::Features { .. }) {
                    return Err(self.error(
                        id,
                        Some("layout"),
                        DiagCode::RegenInvalid,
                        "a variable pattern rebuilds features; pattern bodies with a table".into(),
                        Vec::new(),
                    ));
                }
                for x in vs {
                    let t = [
                        self.value(id, "instances.dx", &x.dx, Dimension::LENGTH)?,
                        self.value(id, "instances.dy", &x.dy, Dimension::LENGTH)?,
                        self.value(id, "instances.dz", &x.dz, Dimension::LENGTH)?,
                    ];
                    instances.push(Instance {
                        key: InstanceKey::Row { key: x.key.clone() },
                        transform: Transform::translation(t),
                    });
                    variable.push(x);
                }
            }
        }
        // Skipped instances leave the varied tools in step.
        let keep: Vec<bool> = instances.iter().map(|x| !p.skip.contains(&x.key)).collect();
        let mut varied = Vec::new();
        if !variable.is_empty() {
            let PatternSeeds::Features { features } = &p.seeds else {
                unreachable!("checked")
            };
            for (k, x) in variable.iter().enumerate() {
                if keep[k] {
                    varied.push(self.varied_tools(i, id, features, &x.values)?);
                }
            }
        }
        let mut k = 0;
        instances.retain(|_| {
            k += 1;
            keep[k - 1]
        });
        self.pattern_like(i, id, &p.seeds, instances, varied)
    }

    /// The seed features' tools rebuilt with these inputs replaced (their sketches too).
    fn varied_tools(
        &mut self,
        i: usize,
        id: &FeatureId,
        seeds: &[FeatureId],
        values: &BTreeMap<crate::ids::InputName, ValueExpr>,
    ) -> Result<Vec<ToolOp>, Halt> {
        let mut out = Vec::new();
        for (k, s) in seeds.iter().enumerate() {
            let param = format!("seeds[{}]", k);
            self.gate_feature(i, id, s, &param)?;
            let doc = self.doc;
            let Some(si) = doc.feature_index(s) else {
                unreachable!("gated")
            };
            let seed = substitute_inputs(&doc.features[si], values);
            // Sketches the seed reads, rebuilt with the same values for this instance.
            let mut saved = Vec::new();
            for dep in seed.kind.feature_dependencies() {
                let Some(di) = doc.feature_index(&dep) else {
                    continue;
                };
                if !matches!(doc.features[di].kind, FeatureKind::Sketch(_)) {
                    continue;
                }
                let sk = substitute_inputs(&doc.features[di], values);
                if sk == doc.features[di] {
                    continue;
                }
                if let Ok(Compiled::Sketch(solved, frame)) = self.compile(di, &sk) {
                    saved.push((dep.clone(), self.st.sketches.insert(dep, (solved, frame))));
                }
            }
            let compiled = self.compile(si, &seed);
            self.pending_record = None;
            for (dep, old) in saved {
                match old {
                    Some(v) => self.st.sketches.insert(dep, v),
                    None => self.st.sketches.remove(&dep),
                };
            }
            match compiled? {
                Compiled::Op {
                    tools: Some(ts), ..
                } => out.extend(ts),
                _ => {
                    let kind = doc.features[si].kind.name();
                    return Err(self.error(
                        id,
                        Some(&param),
                        DiagCode::KernelNotSupported,
                        format!("not supported yet: varying a {} feature in a pattern", kind),
                        Vec::new(),
                    ));
                }
            }
        }
        Ok(out)
    }
}
