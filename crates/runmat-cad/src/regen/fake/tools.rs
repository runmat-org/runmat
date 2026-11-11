//! The tools an operation cuts or adds: exact rectilinear prisms and the sampled shapes, with
//! the profile-to-prism lowering.

use super::*;

// ------------------------------------------------------------------------------- tools

#[derive(Clone, Debug)]
pub(super) struct Prism {
    pub(super) k: usize,
    pub(super) u: usize,
    pub(super) v: usize,
    pub(super) loops: Vec<Vec<[f64; 2]>>,
    pub(super) edge_labels: Vec<Vec<HistorySource>>,
    pub(super) lo: f64,
    pub(super) hi: f64,
    pub(super) lo_label: HistorySource,
    pub(super) hi_label: HistorySource,
}

impl Prism {
    pub(super) fn inside_uv(&self, q: [f64; 2]) -> bool {
        let mut inside = false;
        for l in &self.loops {
            let n = l.len();
            for i in 0..n {
                let (a, b) = (l[i], l[(i + 1) % n]);
                if (a[1] > q[1]) != (b[1] > q[1]) {
                    let x = a[0] + (q[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
                    if q[0] < x {
                        inside = !inside;
                    }
                }
            }
        }
        inside
    }

    pub(super) fn inside(&self, p: [f64; 3]) -> bool {
        p[self.k] > self.lo && p[self.k] < self.hi && self.inside_uv([p[self.u], p[self.v]])
    }

    pub(super) fn label(
        &self,
        axis: usize,
        offset: f64,
        center: [f64; 3],
    ) -> Option<HistorySource> {
        let uv = [center[self.u], center[self.v]];
        if axis == self.k {
            if (offset - self.lo).abs() < EPS && self.inside_uv(uv) {
                return Some(self.lo_label.clone());
            }
            if (offset - self.hi).abs() < EPS && self.inside_uv(uv) {
                return Some(self.hi_label.clone());
            }
            return None;
        }
        if center[self.k] < self.lo - EPS || center[self.k] > self.hi + EPS {
            return None;
        }
        let (on, other) = if axis == self.u { (0, 1) } else { (1, 0) };
        for (li, l) in self.loops.iter().enumerate() {
            let n = l.len();
            for i in 0..n {
                let (a, b) = (l[i], l[(i + 1) % n]);
                if (a[on] - offset).abs() < EPS && (b[on] - offset).abs() < EPS {
                    let (m0, m1) = (a[other].min(b[other]), a[other].max(b[other]));
                    if uv[other] >= m0 - EPS && uv[other] <= m1 + EPS {
                        return Some(self.edge_labels[li][i].clone());
                    }
                }
            }
        }
        None
    }

    pub(super) fn breaks(&self, out: &mut [Vec<f64>; 3]) {
        out[self.k].push(self.lo);
        out[self.k].push(self.hi);
        for l in &self.loops {
            for p in l {
                out[self.u].push(p[0]);
                out[self.v].push(p[1]);
            }
        }
    }

    /// Apply a signed axis permutation (right-angle rotations, axis reflections, translation)
    /// exactly; `None` for any other motion.
    pub(super) fn permuted(&self, t: &Transform) -> Option<Prism> {
        let perm = signed_permutation(t)?;
        let map = |a: usize, x: f64| -> (usize, f64) {
            let (b, sg) = perm[a];
            (b, sg * x + t.translation[b])
        };
        let mut p = self.clone();
        p.k = perm[self.k].0;
        p.u = perm[self.u].0;
        p.v = perm[self.v].0;
        for l in &mut p.loops {
            for q in l.iter_mut() {
                *q = [map(self.u, q[0]).1, map(self.v, q[1]).1];
            }
        }
        let (a, b) = (map(self.k, self.lo).1, map(self.k, self.hi).1);
        if a <= b {
            p.lo = a;
            p.hi = b;
        } else {
            p.lo = b;
            p.hi = a;
            std::mem::swap(&mut p.lo_label, &mut p.hi_label);
        }
        Some(p)
    }

    /// The same prism as a sampled tool, for motions that leave the grid axes.
    pub(super) fn to_sampled(&self) -> sampled::Sampled {
        let mut rot = [[0.0; 3]; 3];
        rot[self.u][0] = 1.0;
        rot[self.v][1] = 1.0;
        rot[self.k][2] = 1.0;
        let to_world = Transform {
            rotation: rot,
            translation: [0.0; 3],
        };
        let loops: Vec<Vec<(sampled::Curve2, HistorySource)>> = self
            .loops
            .iter()
            .zip(&self.edge_labels)
            .map(|(l, labs)| {
                (0..l.len())
                    .map(|i| {
                        (
                            sampled::Curve2::Line {
                                a: l[i],
                                b: l[(i + 1) % l.len()],
                            },
                            labs[i].clone(),
                        )
                    })
                    .collect()
            })
            .collect();
        sampled::prism(
            to_world,
            &loops,
            self.lo,
            self.hi,
            [self.lo_label.clone(), self.hi_label.clone()],
        )
    }

    pub(super) fn relabel(&mut self, seed: u32, instance: &InstanceKey) {
        let fix = |s: &mut HistorySource| {
            if let HistorySource::Generated {
                seed: sd,
                instance: inst,
                ..
            } = s
            {
                *sd = Some(seed);
                *inst = Some(instance.clone());
            }
        };
        fix(&mut self.lo_label);
        fix(&mut self.hi_label);
        for l in &mut self.edge_labels {
            for s in l {
                fix(s);
            }
        }
    }
}

/// For each source axis, the axis it maps to and the sign, when `t`'s rotation is a signed
/// permutation.
pub(super) fn signed_permutation(t: &Transform) -> Option<[(usize, f64); 3]> {
    let r = &t.rotation;
    let mut out = [(0usize, 0.0f64); 3];
    for a in 0..3 {
        let mut found = None;
        for b in 0..3 {
            let x = r[b][a];
            if (x.abs() - 1.0).abs() < EPS {
                if found.is_some() {
                    return None;
                }
                found = Some((b, x.signum()));
            } else if x.abs() > EPS {
                return None;
            }
        }
        out[a] = found?;
    }
    Some(out)
}

/// A tool shape on the grid: exact rectilinear prisms, or sampled shapes.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub(super) enum Tool {
    Exact(Prism),
    Sampled(Box<sampled::Sampled>),
    /// A closed set of labelled polygons (thickened or knitted sheets).
    Polys(Box<sheet::PolyTool>),
    /// A tool clipped to half-spaces `(p − origin)·normal ≤ 0`, the cuts labelled.
    Clipped(Box<Tool>, Vec<([f64; 3], [f64; 3], HistorySource)>),
}

pub(super) fn half(o: [f64; 3], n: [f64; 3], p: [f64; 3]) -> f64 {
    (p[0] - o[0]) * n[0] + (p[1] - o[1]) * n[1] + (p[2] - o[2]) * n[2]
}

impl Tool {
    pub(super) fn inside(&self, p: [f64; 3]) -> bool {
        match self {
            Tool::Exact(t) => t.inside(p),
            Tool::Sampled(t) => t.inside(p),
            Tool::Polys(t) => t.inside(p),
            Tool::Clipped(t, planes) => {
                t.inside(p) && planes.iter().all(|(o, n, _)| half(*o, *n, p) < 0.0)
            }
        }
    }

    pub(super) fn breaks(&self, out: &mut [Vec<f64>; 3]) {
        match self {
            Tool::Exact(t) => t.breaks(out),
            Tool::Sampled(t) => t.breaks(out),
            Tool::Polys(t) => t.breaks(out),
            Tool::Clipped(t, planes) => {
                t.breaks(out);
                for (o, n, _) in planes {
                    for a in 0..3 {
                        out[a].push(o[a]);
                    }
                    if axis_of(*n).is_none() {
                        let mut tmp: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
                        t.breaks(&mut tmp);
                        for a in 0..3 {
                            let lo = tmp[a].iter().cloned().fold(f64::INFINITY, f64::min);
                            let hi = tmp[a].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                            if lo.is_finite() && hi.is_finite() {
                                for k in 0..=sampled::RES {
                                    out[a].push(lo + (hi - lo) * k as f64 / sampled::RES as f64);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The label of a facet on `plane` at `cen` between cell centres `a` (the facet's cell)
    /// and `b` (across it), when this tool bounds it.
    pub(super) fn label(
        &self,
        plane: &PlaneKey,
        cen: [f64; 3],
        a: [f64; 3],
        b: [f64; 3],
    ) -> Option<(HistorySource, FacetKind, FacetExact)> {
        match self {
            Tool::Exact(t) => t
                .label(plane.axis, plane.offset, cen)
                .map(|l| (l, FacetKind::Planar, FacetExact::None)),
            Tool::Sampled(t) => t.crossing(a, b).map(|i| {
                let s = &t.surfaces[i];
                let kind = match s.plane {
                    Some((ax, off)) if ax == plane.axis && (off - plane.offset).abs() < 1e-7 => {
                        FacetKind::Planar
                    }
                    _ => FacetKind::Faceted(s.hint),
                };
                (s.label.clone(), kind, FacetExact::Surface(i))
            }),
            Tool::Polys(t) => t.label(plane, cen, a, b),
            Tool::Clipped(t, planes) => {
                for (o, n, l) in planes {
                    let (da, db) = (half(*o, *n, a), half(*o, *n, b));
                    if (da <= 0.0) != (db <= 0.0) {
                        let s = da / (da - db);
                        let x = [
                            a[0] + s * (b[0] - a[0]),
                            a[1] + s * (b[1] - a[1]),
                            a[2] + s * (b[2] - a[2]),
                        ];
                        let inside_other = t.inside(x) || t.inside(if da <= 0.0 { a } else { b });
                        if inside_other {
                            let kind = match axis_of(*n) {
                                Some((ax, _)) if ax == plane.axis => FacetKind::Planar,
                                _ => FacetKind::Faceted(SurfaceKind::Plane { normal: *n }),
                            };
                            let ex = FacetExact::Ready(Arc::new(exact_plane(*o, *n)));
                            return Some((l.clone(), kind, ex));
                        }
                    }
                }
                t.label(plane, cen, a, b)
            }
        }
    }

    /// The exact surface of the sampled surface `i` this tool labelled a facet with.
    pub(super) fn exact_surface(&self, i: usize) -> Option<exact::Exact> {
        match self {
            Tool::Sampled(t) => t.exact_of(i),
            Tool::Clipped(t, _) => t.exact_surface(i),
            _ => None,
        }
    }

    /// Replace labels by `f` (where it returns one).
    pub(super) fn map_labels(&mut self, f: &dyn Fn(&HistorySource) -> Option<HistorySource>) {
        let mut fix = |l: &mut HistorySource| {
            if let Some(n) = f(l) {
                *l = n;
            }
        };
        match self {
            Tool::Exact(t) => {
                fix(&mut t.lo_label);
                fix(&mut t.hi_label);
                for l in &mut t.edge_labels {
                    for x in l {
                        fix(x);
                    }
                }
            }
            Tool::Sampled(t) => t.map_labels(&mut fix),
            Tool::Polys(t) => {
                for (_, l) in &mut t.facets {
                    fix(l);
                }
            }
            Tool::Clipped(t, planes) => {
                t.map_labels(f);
                for (_, _, l) in planes {
                    fix(l);
                }
            }
        }
    }

    pub(super) fn relabel(&mut self, seed: u32, instance: &InstanceKey) {
        match self {
            Tool::Exact(t) => t.relabel(seed, instance),
            Tool::Sampled(t) => t.relabel(seed, instance),
            Tool::Clipped(t, _) => t.relabel(seed, instance),
            Tool::Polys(t) => {
                for (_, l) in &mut t.facets {
                    if let HistorySource::Generated {
                        seed: sd,
                        instance: inst,
                        ..
                    } = l
                    {
                        *sd = Some(seed);
                        *inst = Some(instance.clone());
                    }
                }
            }
        }
    }

    pub(super) fn transformed(&self, t: &Transform) -> Tool {
        match self {
            Tool::Exact(p) => match p.permuted(t) {
                Some(q) => Tool::Exact(q),
                None => Tool::Sampled(Box::new(p.to_sampled().transformed(t))),
            },
            Tool::Sampled(s) => Tool::Sampled(Box::new(s.transformed(t))),
            Tool::Polys(p) => {
                let facets = p
                    .facets
                    .iter()
                    .map(|(f, l)| {
                        let mut g = f.clone();
                        for lp in &mut g.loops {
                            for q in lp.iter_mut() {
                                *q = t.apply(*q);
                            }
                        }
                        g.normal = t.apply_vector(f.normal);
                        (g, l.clone())
                    })
                    .collect();
                Tool::Polys(Box::new(sheet::PolyTool::new(facets)))
            }
            Tool::Clipped(inner, planes) => Tool::Clipped(
                Box::new(inner.transformed(t)),
                planes
                    .iter()
                    .map(|(o, n, l)| (t.apply(*o), t.apply_vector(*n), l.clone()))
                    .collect(),
            ),
        }
    }
}

pub(super) fn gen(role: Role, tag: KernelTag, ordinal: Option<u32>) -> HistorySource {
    HistorySource::Generated {
        role,
        tag,
        ordinal,
        seed: None,
        instance: None,
    }
}

pub(super) fn axis_of(v: [f64; 3]) -> Option<(usize, f64)> {
    let mut found = None;
    for (a, x) in v.iter().enumerate() {
        if (x.abs() - 1.0).abs() < 1e-9 {
            if found.is_some() {
                return None;
            }
            found = Some((a, x.signum()));
        } else if x.abs() > 1e-9 {
            return None;
        }
    }
    found
}

pub(super) fn frame_axes(frame: &Frame) -> Result<(usize, f64, usize, usize), KernelError> {
    let unsupported = || KernelError::NotSupported {
        what: "fake kernel: frames not aligned to the part axes".into(),
    };
    let (k, ks) = axis_of(frame.z).ok_or_else(unsupported)?;
    let (u, _) = axis_of(frame.x).ok_or_else(unsupported)?;
    let (v, _) = axis_of(frame.y).ok_or_else(unsupported)?;
    Ok((k, ks, u, v))
}

pub(super) fn prism_from_profile(
    profile: &CanonicalProfile,
    start: f64,
    end: f64,
    through_all: Option<(f64, f64)>,
    profile_index: u32,
) -> Result<Prism, KernelError> {
    let (k, ks, u, v) = frame_axes(&profile.frame)?;
    let mut loops = Vec::new();
    let mut labels = Vec::new();
    for l in &profile.loops {
        let mut pts = Vec::new();
        let mut ls = Vec::new();
        for s in l {
            if !matches!(s.kind, SegmentKind::Line) {
                return Err(KernelError::NotSupported {
                    what: "fake kernel: arcs and circles in profiles".into(),
                });
            }
            let a = profile.frame.to_world(s.start);
            let b = profile.frame.to_world(s.end);
            if (a[u] - b[u]).abs() > EPS && (a[v] - b[v]).abs() > EPS {
                return Err(KernelError::NotSupported {
                    what: "fake kernel: profile segments not parallel to an axis".into(),
                });
            }
            pts.push([a[u], a[v]]);
            ls.push(gen(
                Role::Side,
                KernelTag::Curve {
                    profile: profile_index,
                    curve: s.curve.clone(),
                },
                None,
            ));
        }
        loops.push(pts);
        labels.push(ls);
    }
    let o = profile.frame.origin[k];
    let (mut s0, mut s1) = (o + ks * start, o + ks * end);
    let (mut l0, mut l1) = (
        gen(Role::StartCap, KernelTag::None, None),
        gen(Role::EndCap, KernelTag::None, None),
    );
    if let Some((blo, bhi)) = through_all {
        // Extend the end cap past every operand.
        s1 = if (end - start) * ks > 0.0 { bhi } else { blo };
    }
    if s0 > s1 {
        std::mem::swap(&mut s0, &mut s1);
        std::mem::swap(&mut l0, &mut l1);
    }
    Ok(Prism {
        k,
        u,
        v,
        loops,
        edge_labels: labels,
        lo: s0,
        hi: s1,
        lo_label: l0,
        hi_label: l1,
    })
}
