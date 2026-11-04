//! Surface bodies (open sheets) on the fake kernel, and closed polygon sets as grid tools.
//!
//! A [`FakeSheet`] is a set of planar polygons ("facets"), each belonging to one face. A
//! curved surface is several facets with one face. Topology is derived from shared polygon
//! segments: a segment two faces share is an edge between them, a segment of one face only is
//! a boundary (laminar) edge, and the boundary segments of one face with one boundary label are
//! chained into one edge. Every boundary segment carries the label it was made with, so
//! boundary edges get history records ([`crate::refs::history::BoundaryRecord`]) from the
//! definition, never from output order.
//!
//! [`PolyTool`] turns a closed set of labelled polygons (a thickened sheet, a knitted shell)
//! into a grid tool: occupancy by ray parity, facet labels by the polygon the boundary lies on.

use super::sampled::Curve2;
use super::{FFace, FacetKind, FakeSolid, PlaneKey, Rect, EPS};
use crate::model::datum::Frame;
use crate::refs::history::*;
use crate::refs::name::{PlaneSide, Role};
use crate::regen::kernel::*;
use std::collections::{BTreeMap, BTreeSet};

type P3 = [f64; 3];
/// A polygon segment by its rounded end points (sorted).
type SegKey = ([i64; 3], [i64; 3]);

fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: P3, b: P3) -> P3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: P3, s: f64) -> P3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: P3, b: P3) -> P3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: P3) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: P3) -> P3 {
    let n = norm(a);
    if n > 0.0 {
        scale(a, 1.0 / n)
    } else {
        a
    }
}

/// A planar polygon with holes. `labels[l][i]` labels the boundary segment from point `i` of
/// loop `l` to the next, when it is (or becomes) a boundary edge.
#[derive(Clone, Debug)]
pub(super) struct Facet {
    pub loops: Vec<Vec<P3>>,
    pub normal: P3,
    pub labels: Vec<Vec<HistorySource>>,
}

impl Facet {
    fn area_centroid(&self) -> (f64, P3) {
        let mut area = 0.0;
        let mut c = [0.0; 3];
        for (li, l) in self.loops.iter().enumerate() {
            let o = l[0];
            for i in 1..l.len().saturating_sub(1) {
                let (a, b) = (sub(l[i], o), sub(l[i + 1], o));
                let t = 0.5 * dot(cross(a, b), self.normal);
                let s = if li == 0 { t.abs() } else { -t.abs() };
                area += s;
                let g = scale(add(add(o, l[i]), l[i + 1]), 1.0 / 3.0);
                c = add(c, scale(g, s));
            }
        }
        if area.abs() > 0.0 {
            c = scale(c, 1.0 / area);
        }
        (area.abs(), c)
    }

    fn segments(&self) -> impl Iterator<Item = (usize, usize, P3, P3)> + '_ {
        self.loops
            .iter()
            .enumerate()
            .flat_map(|(li, l)| (0..l.len()).map(move |i| (li, i, l[i], l[(i + 1) % l.len()])))
    }

    /// Whether `p` (on the facet's plane) is inside the polygon (even-odd over all loops).
    fn contains(&self, p: P3) -> bool {
        let n = self.normal;
        let drop = (0..3)
            .max_by(|&a, &b| n[a].abs().partial_cmp(&n[b].abs()).expect("finite"))
            .expect("three axes");
        let (u, v) = ((drop + 1) % 3, (drop + 2) % 3);
        let mut inside = false;
        for l in &self.loops {
            for i in 0..l.len() {
                let (a, b) = (l[i], l[(i + 1) % l.len()]);
                if (a[v] > p[v]) != (b[v] > p[v]) {
                    let x = a[u] + (p[v] - a[v]) * (b[u] - a[u]) / (b[v] - a[v]);
                    if p[u] < x {
                        inside = !inside;
                    }
                }
            }
        }
        inside
    }

    fn offset_of(&self) -> f64 {
        dot(self.normal, self.loops[0][0])
    }
}

/// The kernel's surface-body payload.
#[derive(Clone, Debug)]
pub struct FakeSheet {
    pub(super) facets: Vec<Facet>,
    /// Face of each facet.
    pub(super) face_of: Vec<u32>,
    pub(super) hints: Vec<SurfaceKind>,
    pub(super) topology: Topology,
    /// Boundary segments of each laminar edge: `(facet, loop, index)`.
    pub(super) boundary_segments: BTreeMap<u32, Vec<(usize, usize, usize)>>,
}

fn key(p: P3) -> [i64; 3] {
    [
        (p[0] * 1e9).round() as i64,
        (p[1] * 1e9).round() as i64,
        (p[2] * 1e9).round() as i64,
    ]
}

/// What a builder yields: facets with their face index, per-face sources and hints.
pub(super) struct SheetBuild {
    pub facets: Vec<Facet>,
    pub face_of: Vec<u32>,
    pub face_sources: Vec<Vec<HistorySource>>,
    pub hints: Vec<SurfaceKind>,
}

impl FakeSheet {
    /// Derive topology from facets; returns the sheet and, per laminar edge, its labels.
    pub(super) fn build(b: SheetBuild) -> (FakeSheet, BTreeMap<u32, Vec<HistorySource>>) {
        let nfaces = b.face_sources.len();
        // Segment occurrences.
        let mut occ: BTreeMap<SegKey, Vec<(usize, usize, usize)>> = BTreeMap::new();
        for (fi, f) in b.facets.iter().enumerate() {
            for (li, i, a, c) in f.segments() {
                let (ka, kc) = (key(a), key(c));
                if ka == kc {
                    continue;
                }
                let k = if ka <= kc { (ka, kc) } else { (kc, ka) };
                occ.entry(k).or_default().push((fi, li, i));
            }
        }
        // Group segments into edges: by face pair, or by face plus boundary label.
        #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
        enum EdgeKey {
            Between(u32, u32),
            Boundary(u32, Vec<HistorySource>),
        }
        let mut groups: BTreeMap<EdgeKey, Vec<(usize, usize, usize)>> = BTreeMap::new();
        for segs in occ.values() {
            let faces: BTreeSet<u32> = segs.iter().map(|s| b.face_of[s.0]).collect();
            if segs.len() == 1 {
                let (fi, li, i) = segs[0];
                let mut l = b.facets[fi].labels[li][i..=i].to_vec();
                l.sort();
                groups
                    .entry(EdgeKey::Boundary(b.face_of[fi], l))
                    .or_default()
                    .push(segs[0]);
            } else if faces.len() == 2 {
                let v: Vec<u32> = faces.into_iter().collect();
                groups
                    .entry(EdgeKey::Between(v[0], v[1]))
                    .or_default()
                    .push(segs[0]);
            }
            // Same face on both sides (inside a curved face), or more than two: no edge.
        }
        let mut topology = Topology::default();
        for (fi, src) in b.face_sources.iter().enumerate() {
            let _ = src;
            let mut area = 0.0;
            let mut c = [0.0; 3];
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for (k, f) in b.facets.iter().enumerate() {
                if b.face_of[k] as usize != fi {
                    continue;
                }
                let (a, g) = f.area_centroid();
                area += a;
                c = add(c, scale(g, a));
                for l in &f.loops {
                    for p in l {
                        for ax in 0..3 {
                            lo[ax] = lo[ax].min(p[ax]);
                            hi[ax] = hi[ax].max(p[ax]);
                        }
                    }
                }
            }
            if area > 0.0 {
                c = scale(c, 1.0 / area);
            }
            topology.faces.push(FaceTopo {
                geom: FaceGeom {
                    surface: b.hints[fi],
                    centroid: c,
                    area,
                    bbox_min: lo,
                    bbox_max: hi,
                },
            });
        }
        let mut boundary_segments = BTreeMap::new();
        let mut boundary_labels = BTreeMap::new();
        let mut ends: Vec<Vec<[i64; 3]>> = Vec::new();
        let mut points: BTreeMap<[i64; 3], P3> = BTreeMap::new();
        for (k, segs) in &groups {
            let ei = topology.edges.len() as u32;
            let mut length = 0.0;
            let mut deg: BTreeMap<[i64; 3], u32> = BTreeMap::new();
            let mut first = None;
            let mut last = None;
            for &(fi, li, i) in segs {
                let l = &b.facets[fi].loops[li];
                let (a, c) = (l[i], l[(i + 1) % l.len()]);
                length += norm(sub(c, a));
                *deg.entry(key(a)).or_default() += 1;
                *deg.entry(key(c)).or_default() += 1;
                points.insert(key(a), a);
                points.insert(key(c), c);
                first.get_or_insert(a);
                last = Some(c);
            }
            let (faces, laminar) = match k {
                EdgeKey::Between(x, y) => ([*x, *y], false),
                EdgeKey::Boundary(f, labels) => {
                    boundary_segments.insert(ei, segs.clone());
                    boundary_labels.insert(ei, labels.clone());
                    ([*f, *f], true)
                }
            };
            ends.push(
                deg.into_iter()
                    .filter(|(_, d)| d % 2 == 1)
                    .map(|(p, _)| p)
                    .collect(),
            );
            topology.edges.push(EdgeTopo {
                faces,
                geom: EdgeGeom {
                    start: first.unwrap_or([0.0; 3]),
                    end: last.unwrap_or([0.0; 3]),
                    length,
                },
                laminar,
            });
        }
        let _ = nfaces;
        let mut vmap: BTreeMap<[i64; 3], Vec<u32>> = BTreeMap::new();
        for (ei, e) in ends.iter().enumerate() {
            for p in e {
                vmap.entry(*p).or_default().push(ei as u32);
            }
        }
        for (p, edges) in vmap {
            topology.vertices.push(VertexTopo {
                edges,
                point: points[&p],
            });
        }
        (
            FakeSheet {
                facets: b.facets,
                face_of: b.face_of,
                hints: b.hints,
                topology,
                boundary_segments,
            },
            boundary_labels,
        )
    }

    /// The facets of one face.
    pub(super) fn face_facets(&self, face: u32) -> impl Iterator<Item = &Facet> + '_ {
        self.facets
            .iter()
            .zip(&self.face_of)
            .filter(move |(_, f)| **f == face)
            .map(|(x, _)| x)
    }

    /// Relabel every boundary segment with the boundary edge it is now part of, as an input
    /// source for the next operation.
    pub(super) fn as_input(&self, operand: u32) -> SheetBuild {
        let mut facets = self.facets.clone();
        for f in &mut facets {
            for l in &mut f.labels {
                for x in l.iter_mut() {
                    *x = HistorySource::Input {
                        from: OperandEntity {
                            operand,
                            entity: u32::MAX,
                        },
                    };
                }
            }
        }
        for (e, segs) in &self.boundary_segments {
            for &(fi, li, i) in segs {
                facets[fi].labels[li][i] = HistorySource::Input {
                    from: OperandEntity {
                        operand,
                        entity: *e,
                    },
                };
            }
        }
        let nf = self.topology.faces.len();
        SheetBuild {
            facets,
            face_of: self.face_of.clone(),
            face_sources: (0..nf)
                .map(|f| {
                    vec![HistorySource::Input {
                        from: OperandEntity {
                            operand,
                            entity: f as u32,
                        },
                    }]
                })
                .collect(),
            hints: self.hints.clone(),
        }
    }
}

/// Finish sheet builds into kernel output: one body per build.
pub(super) fn output(
    kernel: &KernelId,
    builds: Vec<(SheetBuild, Vec<HistorySource>)>,
    extra: Vec<(KernelBody, Vec<Vec<HistorySource>>, Vec<HistorySource>)>,
    deleted_bodies: Vec<u32>,
) -> Result<KernelOutput, KernelError> {
    let mut bodies = Vec::new();
    let mut history = OpHistory {
        deleted_bodies,
        ..Default::default()
    };
    for (b, body_sources) in builds {
        let face_sources = b.face_sources.clone();
        let (sheet, labels) = FakeSheet::build(b);
        let bi = bodies.len() as u32;
        for (f, src) in face_sources.into_iter().enumerate() {
            let evolution = evolution_of(&src);
            history.faces.push(FaceRecord {
                body: bi,
                face: f as u32,
                evolution,
                sources: src,
            });
        }
        for (e, l) in labels {
            let sources: Vec<HistorySource> = l
                .into_iter()
                .filter(|s| !matches!(s, HistorySource::Input { from } if from.entity == u32::MAX))
                .collect();
            if sources.is_empty() {
                return Err(KernelError::HistoryIncomplete {
                    message: "fake kernel: a sheet boundary has no source".into(),
                });
            }
            history.boundaries.push(BoundaryRecord {
                body: bi,
                edge: e,
                sources,
            });
        }
        history.bodies.push(BodyRecord {
            body: bi,
            sources: body_sources,
        });
        let topology = sheet.topology.clone();
        bodies.push(KernelBody {
            handle: BodyHandle::new(kernel.clone(), sheet),
            topology,
        });
    }
    for (kb, faces, body) in extra {
        let bi = bodies.len() as u32;
        for (f, src) in faces.into_iter().enumerate() {
            history.faces.push(FaceRecord {
                body: bi,
                face: f as u32,
                evolution: evolution_of(&src),
                sources: src,
            });
        }
        history.bodies.push(BodyRecord {
            body: bi,
            sources: body,
        });
        bodies.push(kb);
    }
    Ok(KernelOutput { bodies, history })
}

fn evolution_of(src: &[HistorySource]) -> Evolution {
    match src {
        [HistorySource::Input { .. }] => Evolution::Kept,
        [HistorySource::Generated { .. }] => Evolution::Generated,
        [HistorySource::Copy { .. }] => Evolution::Generated,
        _ => Evolution::Merged,
    }
}

fn gen_boundary(tag: KernelTag, ordinal: Option<u32>) -> HistorySource {
    super::gen(Role::Boundary, tag, ordinal)
}

fn label_of(l: &FaceLabel) -> HistorySource {
    super::gen(l.role, l.tag.clone(), l.ordinal)
}

/// Planar patches: one sheet per patch.
pub(super) fn planar(patches: &[SurfacePatch]) -> Vec<(SheetBuild, Vec<HistorySource>)> {
    patches
        .iter()
        .map(|p| {
            let fr = p.profile.frame;
            let mut loops = Vec::new();
            let mut labels = Vec::new();
            for l in &p.profile.loops {
                let mut pts = Vec::new();
                let mut ls = Vec::new();
                for seg in l {
                    let c = Curve2::from_segment(seg);
                    let poly = c.polyline(c.samples());
                    for q in &poly[..poly.len() - 1] {
                        pts.push(fr.to_world(*q));
                        ls.push(gen_boundary(
                            KernelTag::Curve {
                                profile: 0,
                                curve: seg.curve.clone(),
                            },
                            None,
                        ));
                    }
                }
                loops.push(pts);
                labels.push(ls);
            }
            let face = label_of(&p.label);
            (
                SheetBuild {
                    facets: vec![Facet {
                        loops,
                        normal: fr.z,
                        labels,
                    }],
                    face_of: vec![0],
                    face_sources: vec![vec![face.clone()]],
                    hints: vec![SurfaceKind::Plane { normal: fr.z }],
                },
                vec![super::gen(Role::Body, KernelTag::None, None)],
            )
        })
        .collect()
}

/// A swept sheet: each profile curve of open or closed chains makes one face; no caps.
pub(super) fn swept(tool: &LabeledTool) -> Result<SheetBuild, KernelError> {
    let label_for = |c: &crate::ids::SketchEntityId| -> HistorySource {
        tool.sides
            .iter()
            .find(|(k, _)| k == c)
            .map(|(_, l)| label_of(l))
            .unwrap_or_else(|| {
                super::gen(
                    Role::Side,
                    KernelTag::Curve {
                        profile: 0,
                        curve: c.clone(),
                    },
                    None,
                )
            })
    };
    // (profile, sweep: point -> (start-cap point, end-cap point) generator)
    let (profile, steps): (&CanonicalProfile, Vec<Transform>) = match &tool.shape {
        ToolShape::Prism {
            profile,
            start,
            end,
            start_bound,
            end_bound,
            draft,
            thin,
        } => {
            if !start_bound.is_blind() || !end_bound.is_blind() || draft.is_some() || thin.is_some()
            {
                return Err(KernelError::NotSupported {
                    what: "fake kernel: surface extrudes with end conditions, draft or thin".into(),
                });
            }
            let z = profile.frame.z;
            (
                profile,
                vec![
                    Transform::translation(scale(z, *start)),
                    Transform::translation(scale(z, *end)),
                ],
            )
        }
        ToolShape::Revolve {
            profile,
            axis_point,
            axis_direction,
            angle,
        } => {
            let n = ((angle.abs() / (std::f64::consts::PI / 24.0)).ceil() as usize).max(2);
            (
                profile,
                (0..=n)
                    .map(|k| {
                        Transform::rotation_about(
                            *axis_point,
                            *axis_direction,
                            angle * k as f64 / n as f64,
                        )
                    })
                    .collect(),
            )
        }
        _ => {
            return Err(KernelError::NotSupported {
                what: "fake kernel: swept or lofted surface bodies".into(),
            })
        }
    };
    let mut b = SheetBuild {
        facets: Vec::new(),
        face_of: Vec::new(),
        face_sources: Vec::new(),
        hints: Vec::new(),
    };
    let fr = profile.frame;
    for chain in &profile.loops {
        for seg in chain {
            let face = b.face_sources.len() as u32;
            b.face_sources.push(vec![label_for(&seg.curve)]);
            b.hints.push(SurfaceKind::Other);
            let c = Curve2::from_segment(seg);
            let poly: Vec<P3> = c
                .polyline(c.samples())
                .into_iter()
                .map(|q| fr.to_world(q))
                .collect();
            let curve_tag = || KernelTag::Curve {
                profile: 0,
                curve: seg.curve.clone(),
            };
            for s in 0..steps.len() - 1 {
                for i in 0..poly.len() - 1 {
                    let (t0, t1) = (&steps[s], &steps[s + 1]);
                    let q = [
                        t0.apply(poly[i]),
                        t0.apply(poly[i + 1]),
                        t1.apply(poly[i + 1]),
                        t1.apply(poly[i]),
                    ];
                    let n = unit(cross(sub(q[1], q[0]), sub(q[3], q[0])));
                    if norm(n) < 0.5 {
                        continue; // degenerate (a point on the revolve axis)
                    }
                    // Segment order: start-cap line, along the sweep at the curve's end
                    // point, end-cap line, along the sweep at its start point. Only boundary
                    // segments keep their label (shared ones become edges between faces).
                    let labels = vec![
                        gen_boundary(curve_tag(), Some(0)),
                        gen_boundary(curve_tag(), Some(3)),
                        gen_boundary(curve_tag(), Some(1)),
                        gen_boundary(curve_tag(), Some(2)),
                    ];
                    b.facets.push(Facet {
                        loops: vec![q.to_vec()],
                        normal: n,
                        labels: vec![labels],
                    });
                    b.face_of.push(face);
                }
            }
            let ns: Vec<P3> = b
                .facets
                .iter()
                .zip(&b.face_of)
                .filter(|(_, f)| **f == face)
                .map(|(x, _)| x.normal)
                .collect();
            if let Some(n0) = ns.first() {
                if ns.iter().all(|n| norm(sub(*n, *n0)) < 1e-9) {
                    b.hints[face as usize] = SurfaceKind::Plane { normal: *n0 };
                }
            }
        }
    }
    Ok(b)
}

/// Polygons of a solid's grid face (planar rects or faceted rects).
pub(super) fn solid_face_facets(
    s: &FakeSolid,
    face: u32,
    label: HistorySource,
) -> Result<Vec<Facet>, KernelError> {
    let f: &FFace = s.faces.get(face as usize).ok_or(KernelError::Failed {
        message: "face out of range".into(),
    })?;
    if !f.is_grid() {
        return Err(KernelError::NotSupported {
            what: "fake kernel: surfaces from blend faces".into(),
        });
    }
    Ok(f.rect_planes()
        .into_iter()
        .map(|(pk, r)| rect_facet(pk, r, label.clone()))
        .collect())
}

fn rect_facet(pk: PlaneKey, r: Rect, label: HistorySource) -> Facet {
    let a = pk.axis;
    let (u, v) = ((a + 1) % 3, (a + 2) % 3);
    let mk = |x: f64, y: f64| {
        let mut p = [0.0; 3];
        p[a] = pk.offset;
        p[u] = x;
        p[v] = y;
        p
    };
    let mut pts = vec![
        mk(r.lo[u], r.lo[v]),
        mk(r.hi[u], r.lo[v]),
        mk(r.hi[u], r.hi[v]),
        mk(r.lo[u], r.hi[v]),
    ];
    if pk.dir < 0 {
        pts.reverse();
    }
    let mut n = [0.0; 3];
    n[a] = pk.dir as f64;
    Facet {
        loops: vec![pts],
        normal: n,
        labels: vec![vec![label; 4]],
    }
}

/// For each boundary segment of a selection of a solid's faces, the face across it (by
/// geometry: the unselected face whose rect holds the segment's midpoint).
pub(super) fn label_by_neighbour(
    s: &FakeSolid,
    selected: &BTreeSet<u32>,
    facets: &mut [Facet],
    role: Role,
    operand: u32,
) {
    for f in facets.iter_mut() {
        for li in 0..f.loops.len() {
            let n = f.loops[li].len();
            for i in 0..n {
                let (a, b) = (f.loops[li][i], f.loops[li][(i + 1) % n]);
                let m = scale(add(a, b), 0.5);
                let across = s.faces.iter().enumerate().find(|(k, face)| {
                    !selected.contains(&(*k as u32))
                        && face.rects.iter().any(|r| r.contains(m))
                        && face.is_grid()
                });
                f.labels[li][i] = match across {
                    Some((k, _)) => super::gen(
                        role,
                        KernelTag::InputFace {
                            face: OperandEntity {
                                operand,
                                entity: k as u32,
                            },
                        },
                        None,
                    ),
                    None => super::gen(role, KernelTag::None, None),
                };
            }
        }
    }
}

/// Offset facets along per-vertex normals (planar faces move exactly; a faceted face stays
/// watertight).
pub(super) fn offset_facets(facets: &[Facet], distance: f64) -> Vec<Facet> {
    let mut vn: BTreeMap<[i64; 3], P3> = BTreeMap::new();
    for f in facets {
        for l in &f.loops {
            for p in l {
                let e = vn.entry(key(*p)).or_insert([0.0; 3]);
                *e = add(*e, f.normal);
            }
        }
    }
    facets
        .iter()
        .map(|f| {
            let planar_alone = facets.len() == 1;
            Facet {
                loops: f
                    .loops
                    .iter()
                    .map(|l| {
                        l.iter()
                            .map(|p| {
                                let n = if planar_alone {
                                    f.normal
                                } else {
                                    let v = unit(vn[&key(*p)]);
                                    // Keep the offset distance along the facet normal.
                                    let c = dot(v, f.normal).max(0.2);
                                    scale(v, 1.0 / c)
                                };
                                add(*p, scale(n, distance))
                            })
                            .collect()
                    })
                    .collect(),
                normal: f.normal,
                labels: f.labels.clone(),
            }
        })
        .collect()
}

/// Clip facets by a plane, keeping `keep`; segments on the plane get `cut` as their label.
pub(super) fn clip(
    facets: &[Facet],
    frame: &Frame,
    keep: PlaneSide,
    cut: HistorySource,
) -> Vec<Facet> {
    let sgn = if keep == PlaneSide::Positive {
        1.0
    } else {
        -1.0
    };
    let side = |p: P3| sgn * dot(sub(p, frame.origin), frame.z);
    let mut out = Vec::new();
    for f in facets {
        let mut loops = Vec::new();
        let mut labels = Vec::new();
        for (li, l) in f.loops.iter().enumerate() {
            let n = l.len();
            let mut pts = Vec::new();
            let mut ls = Vec::new();
            for i in 0..n {
                let (a, b) = (l[i], l[(i + 1) % n]);
                let (da, db) = (side(a), side(b));
                let lab = f.labels[li][i].clone();
                if da >= -EPS {
                    pts.push(a);
                    // The segment from a: original if b is kept too, else it ends on the plane.
                    ls.push(lab.clone());
                }
                if (da >= -EPS) != (db >= -EPS) {
                    let t = da / (da - db);
                    let x = add(a, scale(sub(b, a), t));
                    pts.push(x);
                    ls.push(if db >= -EPS { lab } else { cut.clone() });
                }
            }
            if pts.len() >= 3 {
                loops.push(pts);
                labels.push(ls);
            }
        }
        if !loops.is_empty() {
            out.push(Facet {
                loops,
                normal: f.normal,
                labels,
            });
        }
    }
    out
}

// ------------------------------------------------------------------ closed polygon tools

/// A closed set of labelled polygons used as a grid tool.
#[derive(Clone, Debug)]
pub(super) struct PolyTool {
    pub facets: Vec<(Facet, HistorySource)>,
    pub lo: P3,
    pub hi: P3,
}

impl PolyTool {
    pub fn new(facets: Vec<(Facet, HistorySource)>) -> PolyTool {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for (f, _) in &facets {
            for l in &f.loops {
                for p in l {
                    for a in 0..3 {
                        lo[a] = lo[a].min(p[a]);
                        hi[a] = hi[a].max(p[a]);
                    }
                }
            }
        }
        PolyTool { facets, lo, hi }
    }

    pub fn inside(&self, p: P3) -> bool {
        if (0..3).any(|a| p[a] < self.lo[a] - EPS || p[a] > self.hi[a] + EPS) {
            return false;
        }
        // Ray parity along a generic direction.
        let d = unit([1.0, 0.000_731, 0.001_379]);
        let mut n = 0;
        for (f, _) in &self.facets {
            let den = dot(f.normal, d);
            if den.abs() < 1e-12 {
                continue;
            }
            let t = (f.offset_of() - dot(f.normal, p)) / den;
            if t <= 0.0 {
                continue;
            }
            if f.contains(add(p, scale(d, t))) {
                n += 1;
            }
        }
        n % 2 == 1
    }

    pub fn breaks(&self, out: &mut [Vec<f64>; 3]) {
        let mut axis_aligned = true;
        for (f, _) in &self.facets {
            if super::axis_of(f.normal).is_none() {
                axis_aligned = false;
            }
            for l in &f.loops {
                for p in l {
                    for a in 0..3 {
                        out[a].push(p[a]);
                    }
                }
            }
        }
        if !axis_aligned {
            for a in 0..3 {
                for k in 0..=super::sampled::RES {
                    out[a].push(
                        self.lo[a]
                            + (self.hi[a] - self.lo[a]) * k as f64 / super::sampled::RES as f64,
                    );
                }
            }
        }
    }

    /// The label of a boundary facet between cell centres `a` and `b` on `plane` at `cen`.
    pub fn label(
        &self,
        plane: &PlaneKey,
        cen: P3,
        a: P3,
        b: P3,
    ) -> Option<(HistorySource, FacetKind)> {
        // Exact: a polygon on the facet's plane holding its centre.
        for (f, l) in &self.facets {
            if let Some((ax, s)) = super::axis_of(f.normal) {
                if ax == plane.axis
                    && (f.offset_of() * s - plane.offset).abs() < 1e-9
                    && f.contains(cen)
                {
                    return Some((l.clone(), FacetKind::Planar));
                }
            }
        }
        // Otherwise the polygon the segment between the two cell centres crosses.
        let d = sub(b, a);
        let mut best: Option<(f64, &Facet, &HistorySource)> = None;
        for (f, l) in &self.facets {
            let den = dot(f.normal, d);
            if den.abs() < 1e-12 {
                continue;
            }
            let t = (f.offset_of() - dot(f.normal, a)) / den;
            if !(-1e-9..=1.0 + 1e-9).contains(&t) {
                continue;
            }
            let x = add(a, scale(d, t));
            if f.contains(x) {
                let dist = (t - 0.5).abs();
                if best.map(|bb| dist < bb.0).unwrap_or(true) {
                    best = Some((dist, f, l));
                }
            }
        }
        best.map(|(_, f, l)| {
            (
                l.clone(),
                FacetKind::Faceted(SurfaceKind::Plane { normal: f.normal }),
            )
        })
    }
}
