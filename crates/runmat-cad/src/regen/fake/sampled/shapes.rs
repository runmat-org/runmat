//! What a sampled tool is made of: labelled surfaces, shapes, extrusion bounds and the tool.

use super::*;

/// A labelled tool surface.
#[derive(Clone, Debug)]
pub(in crate::regen::fake) struct Surf {
    pub label: HistorySource,
    /// Informative geometry in world coordinates.
    pub hint: SurfaceKind,
    /// The exact grid plane the surface lies on, when it lies on one.
    pub plane: Option<(usize, f64)>,
}

#[derive(Clone, Debug)]
pub(in crate::regen::fake) enum Shape {
    Prism {
        loops: Vec<Vec<(Curve2, usize)>>,
        polys: Vec<Vec<[f64; 2]>>,
        z0: f64,
        z1: f64,
        caps: [usize; 2],
    },
    Revolve {
        loops: Vec<Vec<(Curve2, usize)>>,
        polys: Vec<Vec<[f64; 2]>>,
        ap: [f64; 2],
        ad: [f64; 2],
        e1: [f64; 2],
        /// Right-handed about `ad` when positive.
        sense: f64,
        angle: f64,
        caps: Option<[usize; 2]>,
    },
    Sweep {
        loops: Vec<Vec<Curve2>>,
        polys: Vec<Vec<[f64; 2]>>,
        /// Path vertex offsets from the first vertex, local.
        offsets: Vec<[f64; 3]>,
        /// Side surface per loop, curve and path segment.
        sides: Vec<Vec<Vec<usize>>>,
        caps: [usize; 2],
    },
    Loft {
        z: Vec<f64>,
        /// Per section, loop, curve: the curve's polyline (equal counts across sections).
        curves: Vec<Vec<Vec<Vec<[f64; 2]>>>>,
        /// Side surface per loop, curve and span.
        sides: Vec<Vec<Vec<usize>>>,
        caps: [usize; 2],
    },
    /// A whole body under a rigid motion or reflection; surface `i` is face `i`.
    Image { solid: Arc<FakeSolid> },
    /// A prism whose caps follow end conditions (per profile point), with tapered sides and
    /// an optional thin wall.
    Extrusion(Box<Extrusion>),
}

/// Where an extrusion cap lies above a profile point (local z).
#[derive(Clone, Debug)]
pub(in crate::regen::fake) enum ZBound {
    At(f64),
    /// The nearest hit, along local z from `from` (`sense` ±1; 0: either way), of a ray
    /// against world facet rects; then `offset` further along `sense`.
    Hits {
        rects: Arc<Vec<(usize, f64, Rect)>>,
        from: f64,
        sense: f64,
        offset: f64,
    },
    /// `distance` past another bound (a blind cap measured from a face start).
    Past {
        base: Box<ZBound>,
        distance: f64,
    },
}

impl ZBound {
    pub(super) fn eval(&self, q: [f64; 2], to_world: &Transform) -> Option<f64> {
        match self {
            ZBound::At(z) => Some(*z),
            ZBound::Hits {
                rects,
                from,
                sense,
                offset,
            } => {
                let o = to_world.apply([q[0], q[1], *from]);
                let d = to_world.apply_vector([0.0, 0.0, 1.0]);
                let mut best: Option<f64> = None;
                for (axis, off, r) in rects.iter() {
                    let da = d[*axis];
                    if da.abs() < 1e-12 {
                        continue;
                    }
                    let t = (off - o[*axis]) / da;
                    if *sense != 0.0 && t * sense <= 1e-9 {
                        continue;
                    }
                    let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
                    if !(0..3)
                        .all(|a| a == *axis || (p[a] >= r.lo[a] - 1e-9 && p[a] <= r.hi[a] + 1e-9))
                    {
                        continue;
                    }
                    if best.map(|b| t.abs() < b.abs()).unwrap_or(true) {
                        best = Some(t);
                    }
                }
                best.map(|t| {
                    let s = if *sense == 0.0 { 1.0 } else { *sense };
                    from + t + s * offset
                })
            }
            ZBound::Past { base, distance } => base.eval(q, to_world).map(|z| z + distance),
        }
    }
    pub(super) fn fixed(&self) -> Option<f64> {
        match self {
            ZBound::At(z) => Some(*z),
            _ => None,
        }
    }
}

/// A thin wall: inside, outside, and the cap thickness with its two inner cap surfaces
/// (surface indices, or the labels they are built from).
pub(in crate::regen::fake) type ThinBand<S> = (f64, f64, Option<(f64, [S; 2])>);

#[derive(Clone, Debug)]
pub(in crate::regen::fake) struct Extrusion {
    /// Per loop and curve: the curve, its outer side surface and (thin) its inner one.
    pub(super) loops: Vec<Vec<(Curve2, usize, Option<usize>)>>,
    pub(super) polys: Vec<Vec<[f64; 2]>>,
    pub(super) start: ZBound,
    pub(super) end: ZBound,
    pub(super) caps: [usize; 2],
    /// +1 when the end cap lies towards +z of the start.
    pub(super) end_sign: f64,
    /// Draft: hinge height, tangent on the end side, on the start side.
    pub(super) draft: Option<(f64, f64, f64)>,
    /// Thin wall: inside, outside, and cap thickness with the inner cap surfaces.
    pub(super) thin: Option<ThinBand<usize>>,
}

impl Extrusion {
    pub(super) fn taper(&self, z: f64) -> f64 {
        match self.draft {
            None => 0.0,
            Some((h, te, ts)) => {
                let t = if (z - h) * self.end_sign >= 0.0 {
                    te
                } else {
                    ts
                };
                (z - h).abs() * t
            }
        }
    }
    pub(super) fn signed(&self, q: [f64; 2]) -> (f64, bool) {
        let inside = inside_polys(&self.polys, q);
        let d = self
            .loops
            .iter()
            .flatten()
            .map(|(c, _, _)| c.distance(q))
            .fold(f64::INFINITY, f64::min);
        (if inside { d } else { -d }, inside)
    }
    pub(super) fn band(&self) -> (f64, f64) {
        match self.thin {
            Some((i, o, _)) => (o, i),
            None => (0.0, f64::INFINITY),
        }
    }
    pub(super) fn caps_at(&self, q: [f64; 2], to_world: &Transform) -> Option<(f64, f64)> {
        Some((self.start.eval(q, to_world)?, self.end.eval(q, to_world)?))
    }
    pub(super) fn inside(&self, p: [f64; 3], to_world: &Transform) -> bool {
        let q = [p[0], p[1]];
        let Some((zs, ze)) = self.caps_at(q, to_world) else {
            return false;
        };
        let (lo, hi) = (zs.min(ze), zs.max(ze));
        if p[2] <= lo || p[2] >= hi {
            return false;
        }
        let e = self.signed(q).0 - self.taper(p[2]);
        let (outside, inside) = self.band();
        if e > -outside && e < inside {
            return true;
        }
        if let Some((_, _, Some((cap, _)))) = self.thin {
            return e > -outside && (p[2] - lo < cap || hi - p[2] < cap);
        }
        false
    }
    pub(super) fn distances(&self, p: [f64; 3], to_world: &Transform, out: &mut [f64]) {
        let q = [p[0], p[1]];
        let Some((zs, ze)) = self.caps_at(q, to_world) else {
            return;
        };
        let (lo, hi) = (zs.min(ze), zs.max(ze));
        let zr = (lo - p[2]).max(p[2] - hi).max(0.0);
        let tp = self.taper(p[2]);
        let (_, inside_profile) = self.signed(q);
        let (outside, inside) = self.band();
        for (c, o, inn) in self.loops.iter().flatten() {
            let d = c.distance(q);
            let sd = if inside_profile { d } else { -d };
            out[*o] = out[*o].min((sd - tp + outside).abs() + zr);
            if let Some(k) = inn {
                out[*k] = out[*k].min((sd - tp - inside).abs() + zr);
            }
        }
        let e = self.signed(q).0 - tp;
        let off_band = (-outside - e)
            .max(if inside.is_finite() { e - inside } else { 0.0 })
            .max(0.0);
        let cap_band = (-outside - e).max(0.0);
        out[self.caps[0]] = out[self.caps[0]].min((p[2] - zs).abs() + cap_band.min(off_band));
        out[self.caps[1]] = out[self.caps[1]].min((p[2] - ze).abs() + cap_band.min(off_band));
        if let Some((_, _, Some((cap, [a, b])))) = self.thin {
            let s = if ze >= zs { 1.0 } else { -1.0 };
            let hollow = (inside - e).max(0.0);
            out[a] = out[a].min((p[2] - (zs + s * cap)).abs() + hollow);
            out[b] = out[b].min((p[2] - (ze - s * cap)).abs() + hollow);
        }
    }
}

#[derive(Clone, Debug)]
pub(in crate::regen::fake) struct Sampled {
    /// World to shape-local.
    pub to_local: Transform,
    pub shape: Shape,
    pub surfaces: Vec<Surf>,
    pub lo: [f64; 3],
    pub hi: [f64; 3],
    pub exact: Vec<(usize, f64)>,
}
