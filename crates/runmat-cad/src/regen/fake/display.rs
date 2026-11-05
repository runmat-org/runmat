//! Display tessellation for the test kernel's solids.
//!
//! Grid faces are unions of axis-aligned rectangles, so each rectangle becomes two triangles
//! on its plane, wound outwards. A faceted face (a curved or off-axis surface on the grid) draws
//! as its facets. Convex fillets and chamfers, which the grid keeps as synthetic blend faces over
//! a sharp edge, draw as a quarter cylinder (or a flat strip) between tangent lines, and the
//! planar faces they meet are trimmed back to those lines. Concave blends keep the sharp edge.

use super::*;
use crate::regen::kernel::{BodyMesh, FaceMesh, TessellationOptions, Tessellator};
use std::f64::consts::FRAC_PI_2;

/// A convex blend ready to draw: the decoration and its outward face normals.
struct Blend<'a> {
    dec: &'a Decoration,
    /// Outward normal of each of the two faces, as (axis, sign).
    normals: [(usize, f64); 2],
}

impl Blend<'_> {
    /// The point on the blend at angle `theta` (0 on face 0, a quarter turn on face 1), at
    /// `along` on the edge's axis.
    fn point(&self, theta: f64, along: f64) -> [f64; 3] {
        let size = self.dec.size;
        let mut p = self.dec.at;
        p[self.dec.along] = along;
        let [(a0, s0), (a1, s1)] = self.normals;
        // The centre sits `size` inside both faces.
        p[a0] -= s0 * size;
        p[a1] -= s1 * size;
        let (c, s) = if self.dec.role == Role::Chamfer {
            // A chamfer is the chord between the two tangent lines.
            let t = theta / FRAC_PI_2;
            (1.0 - t, t)
        } else {
            (theta.cos(), theta.sin())
        };
        p[a0] += s0 * size * c;
        p[a1] += s1 * size * s;
        p
    }
}

fn convex_blends(s: &FakeSolid) -> Vec<Blend<'_>> {
    s.decorations
        .iter()
        .filter(|d| d.size > EPS && !d.pieces.is_empty())
        .filter_map(|dec| {
            let normals = dec.planes.map(|p| (p.axis, p.dir as f64));
            let [(a0, s0), (a1, s1)] = normals;
            let mut mid = dec.at;
            mid[dec.along] = 0.5 * (dec.lo + dec.hi);
            let probe = |d0: f64, d1: f64| {
                let mut q = mid;
                let e = 1e-3 * dec.size.max(EPS);
                q[a0] += d0 * s0 * e;
                q[a1] += d1 * s1 * e;
                s.contains(q)
            };
            // A convex edge has the solid only in the quadrant behind both faces.
            let convex = probe(-1.0, -1.0) && !probe(-1.0, 1.0) && !probe(1.0, -1.0);
            convex.then_some(Blend { dec, normals })
        })
        .collect()
}

/// `r` with the strip within `size` of the blend's edge cut away, split along the edge where
/// the blend covers only part of it.
fn trim(r: Rect, plane: &PlaneKey, blend: &Blend<'_>) -> Vec<Rect> {
    let dec = blend.dec;
    let other = if plane.same(&dec.planes[0]) {
        dec.planes[1]
    } else if plane.same(&dec.planes[1]) {
        dec.planes[0]
    } else {
        return vec![r];
    };
    let (oa, edge) = (other.axis, dec.at[other.axis]);
    let along = dec.along;
    // Does the rectangle reach the edge line from inside the solid?
    let touches = if other.dir > 0 {
        (r.hi[oa] - edge).abs() < EPS
    } else {
        (r.lo[oa] - edge).abs() < EPS
    };
    let (lo, hi) = (dec.lo.max(r.lo[along]), dec.hi.min(r.hi[along]));
    if !touches || hi - lo <= EPS {
        return vec![r];
    }
    let mut out = Vec::new();
    for (a, b) in [(r.lo[along], lo), (hi, r.hi[along])] {
        if b - a > EPS {
            let mut piece = r;
            piece.lo[along] = a;
            piece.hi[along] = b;
            out.push(piece);
        }
    }
    let mut cut = r;
    cut.lo[along] = lo;
    cut.hi[along] = hi;
    if other.dir > 0 {
        cut.hi[oa] -= dec.size;
    } else {
        cut.lo[oa] += dec.size;
    }
    if cut.hi[oa] - cut.lo[oa] > EPS {
        out.push(cut);
    }
    out
}

/// Two triangles for a rectangle on `plane`, wound so their normal points along the plane's
/// outward direction.
fn push_rect(mesh: &mut FaceMesh, plane: &PlaneKey, r: &Rect) {
    let a = plane.axis;
    let (u, v) = ((a + 1) % 3, (a + 2) % 3);
    let corner = |cu: f64, cv: f64| {
        let mut p = [0.0; 3];
        p[a] = plane.offset;
        p[u] = cu;
        p[v] = cv;
        p
    };
    let base = mesh.positions.len() as u32;
    mesh.positions.extend([
        corner(r.lo[u], r.lo[v]),
        corner(r.hi[u], r.lo[v]),
        corner(r.hi[u], r.hi[v]),
        corner(r.lo[u], r.hi[v]),
    ]);
    // e_u × e_v = e_a, so counter-clockwise in (u, v) faces +a.
    if plane.dir > 0 {
        mesh.triangles
            .extend([[base, base + 1, base + 2], [base, base + 2, base + 3]]);
    } else {
        mesh.triangles
            .extend([[base, base + 2, base + 1], [base, base + 3, base + 2]]);
    }
}

fn push_blend(mesh: &mut FaceMesh, blend: &Blend<'_>, lo: f64, hi: f64, segments: usize) {
    let base = mesh.positions.len() as u32;
    for i in 0..=segments {
        let theta = FRAC_PI_2 * i as f64 / segments as f64;
        mesh.positions.push(blend.point(theta, lo));
        mesh.positions.push(blend.point(theta, hi));
    }
    // Orient the strip outwards: compare the first quad's normal with the outward bisector.
    let [p0, p1, p2] = [0, 1, 2].map(|i| mesh.positions[base as usize + i]);
    let (e1, e2) = (sub(p1, p0), sub(p2, p0));
    let n = cross(e1, e2);
    let [(a0, s0), (a1, s1)] = blend.normals;
    let mut out = [0.0; 3];
    out[a0] += s0;
    out[a1] += s1;
    let flip = dot(n, out) < 0.0;
    for i in 0..segments as u32 {
        let (a, b, c, d) = (
            base + 2 * i,
            base + 2 * i + 1,
            base + 2 * i + 2,
            base + 2 * i + 3,
        );
        if flip {
            mesh.triangles.extend([[a, c, b], [b, c, d]]);
        } else {
            mesh.triangles.extend([[a, b, c], [b, d, c]]);
        }
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The test kernel's display mesh of one solid.
pub(crate) fn tessellate_solid(s: &FakeSolid, options: &TessellationOptions) -> BodyMesh {
    if let Some(mesh) = super::mesh::tessellate(s, options) {
        return mesh;
    }
    facet_mesh(s, options)
}

/// The solid drawn on its grid facets, blends as strips over trimmed faces.
fn facet_mesh(s: &FakeSolid, options: &TessellationOptions) -> BodyMesh {
    let blends = convex_blends(s);
    let segments = ((FRAC_PI_2 / options.angle_tolerance.max(1e-3)).ceil() as usize).clamp(2, 64);
    let mut faces = vec![FaceMesh::default(); s.faces.len()];
    for (fi, f) in s.faces.iter().enumerate() {
        for (plane, rect) in f.rect_planes() {
            let mut rects = vec![rect];
            if f.plane.is_some() {
                for blend in &blends {
                    rects = rects
                        .into_iter()
                        .flat_map(|r| trim(r, &plane, blend))
                        .collect();
                }
            }
            for r in &rects {
                push_rect(&mut faces[fi], &plane, r);
            }
        }
    }
    // Which blend draws each blend face, for the faces and their tangent edges.
    let mut blend_of: BTreeMap<u32, (usize, f64, f64)> = BTreeMap::new();
    for (bi, blend) in blends.iter().enumerate() {
        for &(lo, hi, face) in &blend.dec.pieces {
            blend_of.insert(face, (bi, lo, hi));
            if let Some(mesh) = faces.get_mut(face as usize) {
                push_blend(mesh, blend, lo, hi, segments);
            }
        }
    }
    let edges = s
        .topology
        .edges
        .iter()
        .map(|e| {
            let geom = e.geom;
            if sub(geom.end, geom.start).iter().all(|x| x.abs() < EPS) {
                // A closed run around a faceted face: picked through its faces.
                return Vec::new();
            }
            // An edge of a drawn blend runs along its tangent line on the other face.
            for (k, face) in e.faces.iter().enumerate() {
                if let Some(&(bi, lo, hi)) = blend_of.get(face) {
                    let blend = &blends[bi];
                    let other = e.faces[1 - k] as usize;
                    let on_first = s
                        .faces
                        .get(other)
                        .and_then(|f| f.plane)
                        .is_some_and(|p| p.same(&blend.dec.planes[0]));
                    let theta = if on_first { 0.0 } else { FRAC_PI_2 };
                    return vec![blend.point(theta, lo), blend.point(theta, hi)];
                }
            }
            vec![geom.start, geom.end]
        })
        .collect();
    let vertices = s.topology.vertices.iter().map(|v| v.point).collect();
    BodyMesh {
        faces,
        edges,
        vertices,
    }
}

impl Tessellator for FakeKernel {
    fn tessellate(
        &self,
        body: &BodyHandle,
        options: &TessellationOptions,
    ) -> Result<BodyMesh, KernelError> {
        if body.kernel != Self::id_static() {
            return Err(KernelError::ForeignOperand { operand: 0 });
        }
        match body.downcast::<FakeSolid>() {
            Some(solid) => Ok(tessellate_solid(solid, options)),
            None => Err(KernelError::NotSupported {
                what: "display tessellation of a surface body on the test kernel".into(),
            }),
        }
    }
}
