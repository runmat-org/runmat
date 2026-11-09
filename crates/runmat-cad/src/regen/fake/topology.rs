//! Edges and vertices from a solid's grid faces, the decorations laid over them, and each
//! face's geometry summary.

use super::*;

/// Edges and vertices from a solid's grid faces, then decorations applied.
pub(super) fn build_topology(s: &mut FakeSolid) {
    let grid = &s.grid;
    let gi = |a: usize, x: f64| -> usize { grid[a].partition_point(|g| *g < x - EPS) };
    // Segment map: (along, start grid point) -> [(face, cell key)].
    let mut segs: BTreeMap<SegmentKey, Vec<(u32, [usize; 3])>> = BTreeMap::new();
    for (fi, f) in s.faces.iter().enumerate() {
        for (p, r) in f.rect_planes() {
            let r = &r;
            let mut c = [0usize; 3];
            for a in 0..3 {
                c[a] = gi(a, r.lo[a]);
            }
            let ia = gi(p.axis, p.offset);
            // The cell this facet bounds.
            let mut cell = c;
            cell[p.axis] = if p.dir > 0 { ia.saturating_sub(1) } else { ia };
            let others: Vec<usize> = (0..3).filter(|a| *a != p.axis).collect();
            for (along, fixed) in [(others[0], others[1]), (others[1], others[0])] {
                for off in 0..2 {
                    let mut pt = c;
                    pt[p.axis] = ia;
                    pt[fixed] = c[fixed] + off;
                    segs.entry((along, pt)).or_default().push((fi as u32, cell));
                }
            }
        }
    }
    // Pieces between two different faces.
    let mut pieces: BTreeMap<PieceKey, Vec<usize>> = BTreeMap::new();
    for ((along, pt), list) in &segs {
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        if list.len() == 2 {
            pairs.push((list[0].0, list[1].0));
        } else if list.len() == 4 {
            let mut by_cell: BTreeMap<[usize; 3], Vec<u32>> = BTreeMap::new();
            for (f, c) in list {
                by_cell.entry(*c).or_default().push(*f);
            }
            for fs in by_cell.values() {
                if fs.len() == 2 {
                    pairs.push((fs[0], fs[1]));
                }
            }
        }
        for (a, b) in pairs {
            if a == b {
                continue;
            }
            let key = (a.min(b), a.max(b));
            let mut line = *pt;
            line[*along] = 0;
            pieces
                .entry((key, *along, line))
                .or_default()
                .push(pt[*along]);
        }
    }
    // Runs.
    struct Run {
        faces: (u32, u32),
        along: usize,
        line: [usize; 3],
        lo: usize,
        hi: usize,
    }
    let mut runs: Vec<Run> = Vec::new();
    for ((faces, along, line), mut starts) in pieces {
        starts.sort();
        starts.dedup();
        let mut i = 0;
        while i < starts.len() {
            let lo = starts[i];
            let mut hi = lo + 1;
            while i + 1 < starts.len() && starts[i + 1] == hi {
                i += 1;
                hi += 1;
            }
            runs.push(Run {
                faces,
                along,
                line,
                lo,
                hi,
            });
            i += 1;
        }
    }
    // Split runs at endpoints of other runs lying inside them.
    let point = |r: &Run, t: usize| -> [usize; 3] {
        let mut p = r.line;
        p[r.along] = t;
        p
    };
    let ends: BTreeSet<[usize; 3]> = runs
        .iter()
        .flat_map(|r| [point(r, r.lo), point(r, r.hi)])
        .collect();
    let mut split_runs: Vec<Run> = Vec::new();
    for r in runs {
        let mut cuts: Vec<usize> = (r.lo + 1..r.hi)
            .filter(|t| ends.contains(&point(&r, *t)))
            .collect();
        cuts.push(r.hi);
        let mut lo = r.lo;
        for c in cuts {
            split_runs.push(Run {
                faces: r.faces,
                along: r.along,
                line: r.line,
                lo,
                hi: c,
            });
            lo = c;
        }
    }
    let world = |p: [usize; 3]| -> [f64; 3] { [grid[0][p[0]], grid[1][p[1]], grid[2][p[2]]] };

    // Faces geometry.
    let mut topo = Topology::default();
    for f in &s.faces {
        topo.faces.push(FaceTopo { geom: face_geom(f) });
    }
    // Runs touching a faceted face are chained into one edge per connected run between the
    // same two faces (a staircase approximating one curve is one edge). A chain continues
    // through a point only where exactly two runs meet.
    let faceted = |f: u32| {
        let face = &s.faces[f as usize];
        face.plane.is_none() && face.is_grid()
    };
    let mut degree: BTreeMap<[usize; 3], usize> = BTreeMap::new();
    for r in &split_runs {
        *degree.entry(point(r, r.lo)).or_default() += 1;
        *degree.entry(point(r, r.hi)).or_default() += 1;
    }
    let mut cparent: Vec<usize> = (0..split_runs.len()).collect();
    fn cfind(p: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while p[r] != r {
            r = p[r];
        }
        p[x] = r;
        r
    }
    {
        type ChainKey = ([usize; 3], (u32, u32));
        let mut at: BTreeMap<ChainKey, Vec<usize>> = BTreeMap::new();
        for (i, r) in split_runs.iter().enumerate() {
            if !(faceted(r.faces.0) || faceted(r.faces.1)) {
                continue;
            }
            for t in [r.lo, r.hi] {
                let pt = point(r, t);
                if degree.get(&pt).copied() == Some(2) {
                    at.entry((pt, r.faces)).or_default().push(i);
                }
            }
        }
        for list in at.values() {
            if list.len() == 2 {
                let (a, b) = (cfind(&mut cparent, list[0]), cfind(&mut cparent, list[1]));
                if a != b {
                    cparent[a.max(b)] = a.min(b);
                }
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..split_runs.len() {
        let r = cfind(&mut cparent, i);
        groups.entry(r).or_default().push(i);
    }
    let mut edges: Vec<EdgeRow> = Vec::new();
    for members in groups.values() {
        let r = &split_runs[members[0]];
        if members.len() == 1 {
            let a = point(r, r.lo);
            let b = point(r, r.hi);
            let (wa, wb) = (world(a), world(b));
            let pa = s.faces[r.faces.0 as usize].plane;
            let pb = s.faces[r.faces.1 as usize].plane;
            edges.push((
                EdgeTopo {
                    faces: [r.faces.0, r.faces.1],
                    geom: EdgeGeom {
                        start: wa,
                        end: wb,
                        length: (wb[r.along] - wa[r.along]).abs(),
                    },
                    laminar: false,
                },
                EdgeMeta {
                    along: r.along,
                    at: wa,
                    lo: wa[r.along],
                    hi: wb[r.along],
                    planes: match (pa, pb) {
                        (Some(x), Some(y)) => Some([x, y]),
                        _ => None,
                    },
                },
                vec![a, b],
            ));
            continue;
        }
        // A chain: its ends are the points used once within it; a closed chain has none.
        let mut uses: BTreeMap<[usize; 3], usize> = BTreeMap::new();
        let mut length = 0.0;
        for &m in members {
            let q = &split_runs[m];
            *uses.entry(point(q, q.lo)).or_default() += 1;
            *uses.entry(point(q, q.hi)).or_default() += 1;
            length += (grid[q.along][q.hi] - grid[q.along][q.lo]).abs();
        }
        let ends: Vec<[usize; 3]> = uses
            .iter()
            .filter(|(_, n)| **n == 1)
            .map(|(p, _)| *p)
            .collect();
        let (wa, wb) = match ends.as_slice() {
            [a, b] => (world(*a), world(*b)),
            _ => {
                let p = *uses.keys().next().expect("chain has points");
                (world(p), world(p))
            }
        };
        edges.push((
            EdgeTopo {
                faces: [r.faces.0, r.faces.1],
                geom: EdgeGeom {
                    start: wa,
                    end: wb,
                    length,
                },
                laminar: false,
            },
            EdgeMeta {
                along: r.along,
                at: wa,
                lo: 0.0,
                hi: length,
                planes: None,
            },
            ends,
        ));
    }

    // Decorations: replace matching edges with blend faces.
    let mut decorations = std::mem::take(&mut s.decorations);
    let mut final_edges: Vec<EdgeRow> = Vec::new();
    let mut new_faces: Vec<(FFace, FaceGeom)> = Vec::new();
    for dec in &mut decorations {
        dec.pieces.clear();
    }
    for (e, meta, ends) in edges {
        let mut hit = None;
        if let Some(planes) = meta.planes {
            for (di, dec) in decorations.iter().enumerate() {
                let same_planes = (planes[0].same(&dec.planes[0])
                    && planes[1].same(&dec.planes[1]))
                    || (planes[0].same(&dec.planes[1]) && planes[1].same(&dec.planes[0]));
                let same_line = meta.along == dec.along
                    && (0..3)
                        .filter(|a| *a != dec.along)
                        .all(|a| (meta.at[a] - dec.at[a]).abs() < EPS);
                let overlap = meta.lo < dec.hi - EPS && meta.hi > dec.lo + EPS;
                if same_planes && same_line && overlap {
                    hit = Some(di);
                    break;
                }
            }
        }
        match hit {
            None => final_edges.push((e, meta, ends)),
            Some(di) => {
                let dec = &mut decorations[di];
                let fidx = (s.faces.len() + new_faces.len()) as u32;
                dec.pieces.push((meta.lo, meta.hi, fidx));
                let mid = [
                    (e.geom.start[0] + e.geom.end[0]) / 2.0,
                    (e.geom.start[1] + e.geom.end[1]) / 2.0,
                    (e.geom.start[2] + e.geom.end[2]) / 2.0,
                ];
                let mut dir = [0.0; 3];
                dir[dec.along] = 1.0;
                let mut lo = e.geom.start;
                let mut hi = e.geom.end;
                for a in 0..3 {
                    let (x, y) = (lo[a].min(hi[a]), lo[a].max(hi[a]));
                    lo[a] = x;
                    hi[a] = y;
                }
                let area = e.geom.length
                    * dec.size
                    * if dec.role == Role::Blend {
                        std::f64::consts::FRAC_PI_2
                    } else {
                        std::f64::consts::SQRT_2
                    };
                let surface = if dec.role == Role::Blend && !dec.curved {
                    SurfaceKind::Cylinder {
                        axis_point: e.geom.start,
                        axis_direction: dir,
                        radius: dec.size,
                    }
                } else {
                    SurfaceKind::Other
                };
                new_faces.push((
                    FFace::blend(),
                    FaceGeom {
                        surface,
                        centroid: mid,
                        area,
                        bbox_min: lo,
                        bbox_max: hi,
                    },
                ));
                let meta2 = EdgeMeta {
                    planes: None,
                    ..meta
                };
                final_edges.push((
                    EdgeTopo {
                        faces: [e.faces[0], fidx],
                        geom: e.geom,
                        laminar: false,
                    },
                    meta2,
                    ends.clone(),
                ));
                final_edges.push((
                    EdgeTopo {
                        faces: [fidx, e.faces[1]],
                        geom: e.geom,
                        laminar: false,
                    },
                    meta2,
                    ends,
                ));
            }
        }
    }
    // Each blend piece meets the face across each of its ends along an arc (a chamfer: a
    // line) between its two tangent points, the edge a B-rep kernel builds there.
    let mut arcs: Vec<EdgeRow> = Vec::new();
    for dec in &decorations {
        for &(_, _, fidx) in &dec.pieces {
            let Some((sharp, ends)) = final_edges
                .iter()
                .find(|(e, _, _)| e.faces[1] == fidx)
                .map(|(e, _, ends)| (e, ends.clone()))
            else {
                continue;
            };
            let (f0, f1) = (sharp.faces[0], {
                let other = final_edges
                    .iter()
                    .find(|(e, _, _)| e.faces[0] == fidx)
                    .map(|(e, _, _)| e.faces[1]);
                match other {
                    Some(f) => f,
                    None => continue,
                }
            });
            for end in ends {
                // The face across the end: it shares an edge at this corner with both faces.
                let at_end = |f: u32| -> BTreeSet<u32> {
                    final_edges
                        .iter()
                        .filter(|(e, _, ends)| ends.contains(&end) && e.faces.contains(&f))
                        .flat_map(|(e, _, _)| e.faces)
                        .filter(|x| *x != f && *x != fidx)
                        .collect()
                };
                let across: Vec<u32> = at_end(f0)
                    .intersection(&at_end(f1))
                    .copied()
                    .filter(|x| *x != f0 && *x != f1)
                    .collect();
                let [x] = across.as_slice() else {
                    continue;
                };
                let corner = world(end);
                let [(a0, s0), (a1, s1)] = dec.planes.map(|p| (p.axis, p.dir as f64));
                let mut t0 = corner;
                t0[a1] -= s1 * dec.size;
                let mut t1 = corner;
                t1[a0] -= s0 * dec.size;
                let (start, finish) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
                let length = dec.size
                    * if dec.role == Role::Chamfer {
                        std::f64::consts::SQRT_2
                    } else {
                        std::f64::consts::FRAC_PI_2
                    };
                arcs.push((
                    EdgeTopo {
                        faces: [*x, fidx],
                        geom: EdgeGeom {
                            start,
                            end: finish,
                            length,
                        },
                        laminar: false,
                    },
                    EdgeMeta {
                        along: dec.along,
                        at: corner,
                        lo: corner[dec.along],
                        hi: corner[dec.along],
                        planes: None,
                    },
                    vec![end],
                ));
            }
        }
    }
    final_edges.extend(arcs);
    for (f, g) in new_faces {
        s.faces.push(f);
        topo.faces.push(FaceTopo { geom: g });
    }
    s.decorations = decorations
        .into_iter()
        .filter(|d| !d.pieces.is_empty())
        .collect();

    let mut vmap: BTreeMap<[usize; 3], Vec<u32>> = BTreeMap::new();
    for (ei, (_, _, ends)) in final_edges.iter().enumerate() {
        for p in ends {
            vmap.entry(*p).or_default().push(ei as u32);
        }
    }
    s.edge_meta = final_edges.iter().map(|(_, m, _)| *m).collect();
    topo.edges = final_edges.into_iter().map(|(e, _, _)| e).collect();
    topo.vertices = vmap
        .into_iter()
        .map(|(p, edges)| VertexTopo {
            edges,
            point: world(p),
        })
        .collect();
    s.topology = topo;
}

pub(super) fn face_geom(f: &FFace) -> FaceGeom {
    let mut area = 0.0;
    let mut c = [0.0; 3];
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for r in &f.rects {
        let mut a = 1.0;
        for ax in 0..3 {
            let w = r.hi[ax] - r.lo[ax];
            if w > EPS {
                a *= w;
            }
            lo[ax] = lo[ax].min(r.lo[ax]);
            hi[ax] = hi[ax].max(r.hi[ax]);
        }
        area += a;
        for ax in 0..3 {
            c[ax] += a * (r.lo[ax] + r.hi[ax]) / 2.0;
        }
    }
    if area > 0.0 {
        for x in &mut c {
            *x /= area;
        }
    }
    FaceGeom {
        surface: f.hint(),
        centroid: c,
        area,
        bbox_min: lo,
        bbox_max: hi,
    }
}
