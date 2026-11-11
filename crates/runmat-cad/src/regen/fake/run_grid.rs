//! One re-gridding: operands and tool on the union of their coordinates, cell occupancy, and
//! labelled boundary facets merged into faces.

use super::*;

pub(super) fn run_grid(job: GridJob<'_>) -> Result<GridResult, KernelError> {
    let mut breaks: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for o in &job.operands {
        for a in 0..3 {
            breaks[a].extend(o.grid[a].iter().copied());
        }
    }
    for t in job.tools.iter().chain(&job.carve) {
        t.breaks(&mut breaks);
    }
    if let Some((a, c, _)) = job.split {
        breaks[a].push(c);
    }
    match &job.modifier {
        Some(Modifier::Shell(sh)) => {
            if let Some(o) = job.operands.first() {
                for (fi, f) in o.faces.iter().enumerate() {
                    if sh.open.contains(&(fi as u32)) {
                        continue;
                    }
                    let sign = if sh.outward { -1.0 } else { 1.0 };
                    for (pk, _) in f.rect_planes() {
                        breaks[pk.axis].push(pk.offset - sign * (pk.dir as f64) * sh.of(fi as u32));
                    }
                }
            }
        }
        Some(Modifier::Fill(fs)) => {
            for a in 0..3 {
                breaks[a].push(fs.lo[a]);
                breaks[a].push(fs.hi[a]);
            }
        }
        Some(Modifier::Draft(ds)) => {
            for d in ds {
                // The moved plane of an offset face.
                breaks[d.plane.axis].push(d.plane.offset + (d.plane.dir as f64) * d.shift);
                for a in 0..3 {
                    let (lo, hi) = if a == d.plane.axis {
                        (d.plane.offset - d.reach, d.plane.offset + d.reach)
                    } else {
                        (d.lo[a], d.hi[a])
                    };
                    for k in 0..=sampled::RES {
                        breaks[a].push(lo + (hi - lo) * k as f64 / sampled::RES as f64);
                    }
                }
            }
        }
        None => {}
    }
    let grid: [Vec<f64>; 3] = [
        dedup_sorted(breaks[0].clone()),
        dedup_sorted(breaks[1].clone()),
        dedup_sorted(breaks[2].clone()),
    ];
    let d = [
        grid[0].len().saturating_sub(1),
        grid[1].len().saturating_sub(1),
        grid[2].len().saturating_sub(1),
    ];
    let n = d[0] * d[1] * d[2];
    if n == 0 {
        return Ok(GridResult {
            solids: Vec::new(),
            face_sources: Vec::new(),
            body_sources: Vec::new(),
        });
    }
    if n > 4_000_000 {
        return Err(KernelError::NotSupported {
            what: "fake kernel: grid too large".into(),
        });
    }
    let idx = |c: [usize; 3]| c[0] + d[0] * (c[1] + d[1] * c[2]);
    let center = |c: [usize; 3]| -> [f64; 3] {
        [
            (grid[0][c[0]] + grid[0][c[0] + 1]) / 2.0,
            (grid[1][c[1]] + grid[1][c[1] + 1]) / 2.0,
            (grid[2][c[2]] + grid[2][c[2] + 1]) / 2.0,
        ]
    };
    let mut owners = vec![0u64; n];
    let mut tools = vec![0u64; n];
    let mut occ = vec![false; n];
    for z in 0..d[2] {
        for y in 0..d[1] {
            for x in 0..d[0] {
                let c = [x, y, z];
                let p = center(c);
                let i = idx(c);
                let carved = job.carve.iter().any(|t| t.inside(p));
                for (k, o) in job.operands.iter().enumerate() {
                    if o.contains(p) && !carved {
                        owners[i] |= 1 << k;
                    }
                }
                for (k, t) in job.tools.iter().enumerate() {
                    if t.inside(p) {
                        tools[i] |= 1 << k.min(63);
                    }
                }
                let o = owners[i] != 0;
                let t = tools[i] != 0;
                occ[i] = match job.mode {
                    Mode::New => t,
                    Mode::Add => o || t,
                    Mode::Cut => o && !t,
                    Mode::Intersect => o && t,
                    Mode::BoolUnion | Mode::Split | Mode::Keep => o,
                    Mode::BoolCut => owners[i] & 1 != 0 && owners[i] & !1 == 0,
                    Mode::BoolIntersect => owners[i] & 1 != 0 && owners[i] & !1 != 0,
                };
                if matches!(job.mode, Mode::BoolCut | Mode::BoolIntersect) && occ[i] {
                    owners[i] = 1;
                }
                match &job.modifier {
                    Some(Modifier::Shell(sh)) if sh.outward => {
                        // The body becomes the cavity; walls grow around it.
                        let o = job.operands[0];
                        occ[i] = !occ[i] && sh.outward_wall(o, p);
                        owners[i] = if occ[i] { 1 } else { 0 };
                    }
                    Some(Modifier::Shell(sh)) if occ[i] => {
                        let o = job.operands[0];
                        let cavity = o.faces.iter().enumerate().all(|(fi, f)| {
                            sh.open.contains(&(fi as u32))
                                || !f.is_grid()
                                || face_distance(p, f) > sh.of(fi as u32) + EPS
                        });
                        if cavity {
                            occ[i] = false;
                        }
                    }
                    Some(Modifier::Fill(fs)) => {
                        if (0..3).all(|a| p[a] > fs.lo[a] - EPS && p[a] < fs.hi[a] + EPS) {
                            occ[i] = fs.cavity;
                            if fs.cavity {
                                owners[i] |= 1;
                            } else {
                                owners[i] = 0;
                            }
                        }
                    }
                    Some(Modifier::Draft(ds)) => {
                        // Near drafted faces the material is what every draft whose region
                        // holds the point leaves (corners where two drafts meet take both).
                        let mut decided = None;
                        for d in ds.iter().filter(|d| d.region(p)) {
                            decided = Some(decided.unwrap_or(true) && d.material(p));
                        }
                        if let Some(m) = decided {
                            occ[i] = m;
                            if m {
                                owners[i] |= 1;
                            }
                        } else if !occ[i] {
                            // Past the side edges of faces leaning out, the corner between
                            // two of them is material where both leave it (a mitre); one
                            // draft alone adds nothing past its face's edges.
                            let mut n = 0;
                            let mut all = true;
                            for d in ds.iter().filter(|d| d.corner(p)) {
                                n += 1;
                                all &= d.material(p);
                            }
                            if n >= 2 && all {
                                occ[i] = true;
                                owners[i] |= 1;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Components.
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while p[r] != r {
            r = p[r];
        }
        let mut y = x;
        while p[y] != r {
            let nx = p[y];
            p[y] = r;
            y = nx;
        }
        r
    }
    let connect = |a: usize, b: usize, axis: usize, boundary: usize| -> bool {
        if let Some((sa, c, _)) = job.split {
            if sa == axis && (grid[axis][boundary] - c).abs() < EPS {
                return false;
            }
        }
        match job.mode {
            Mode::New => !job.separate_tools || tools[a] & tools[b] != 0,
            Mode::Add => owners[a] & owners[b] != 0 || tools[a] != 0 || tools[b] != 0,
            Mode::BoolUnion => true,
            _ => owners[a] & owners[b] != 0,
        }
    };
    for z in 0..d[2] {
        for y in 0..d[1] {
            for x in 0..d[0] {
                let c = [x, y, z];
                let i = idx(c);
                if !occ[i] {
                    continue;
                }
                for a in 0..3 {
                    if c[a] + 1 < d[a] {
                        let mut nc = c;
                        nc[a] += 1;
                        let j = idx(nc);
                        if occ[j] && connect(i, j, a, c[a] + 1) {
                            let (ra, rb) = (find(&mut parent, i), find(&mut parent, j));
                            if ra != rb {
                                parent[ra.max(rb)] = ra.min(rb);
                            }
                        }
                    }
                }
            }
        }
    }
    let mut comp_of = vec![usize::MAX; n];
    let mut roots: Vec<usize> = Vec::new();
    for i in 0..n {
        if occ[i] {
            let r = find(&mut parent, i);
            let k = match roots.iter().position(|x| *x == r) {
                Some(k) => k,
                None => {
                    roots.push(r);
                    roots.len() - 1
                }
            };
            comp_of[i] = k;
        }
    }
    let ncomp = roots.len();

    // Facets.
    struct Facet {
        cell: usize,
        c: [usize; 3],
        axis: usize,
        dir: i8,
        comp: usize,
    }
    let mut facets: Vec<Facet> = Vec::new();
    let mut facet_at: BTreeMap<(usize, usize, i8), usize> = BTreeMap::new();
    for z in 0..d[2] {
        for y in 0..d[1] {
            for x in 0..d[0] {
                let c = [x, y, z];
                let i = idx(c);
                if !occ[i] {
                    continue;
                }
                for a in 0..3 {
                    for dir in [-1i8, 1] {
                        let other = if dir < 0 {
                            if c[a] == 0 {
                                None
                            } else {
                                let mut nc = c;
                                nc[a] -= 1;
                                Some(idx(nc))
                            }
                        } else if c[a] + 1 >= d[a] {
                            None
                        } else {
                            let mut nc = c;
                            nc[a] += 1;
                            Some(idx(nc))
                        };
                        let exposed = match other {
                            None => true,
                            Some(j) => !occ[j] || comp_of[j] != comp_of[i],
                        };
                        if exposed {
                            facet_at.insert((i, a, dir), facets.len());
                            facets.push(Facet {
                                cell: i,
                                c,
                                axis: a,
                                dir,
                                comp: comp_of[i],
                            });
                        }
                    }
                }
            }
        }
    }
    let facet_plane = |f: &Facet| -> PlaneKey {
        let b = if f.dir > 0 {
            f.c[f.axis] + 1
        } else {
            f.c[f.axis]
        };
        PlaneKey {
            axis: f.axis,
            dir: f.dir,
            offset: grid[f.axis][b],
        }
    };
    let facet_rect = |f: &Facet| -> Rect {
        let mut lo = [0.0; 3];
        let mut hi = [0.0; 3];
        for a in 0..3 {
            lo[a] = grid[a][f.c[a]];
            hi[a] = grid[a][f.c[a] + 1];
        }
        let p = facet_plane(f);
        lo[f.axis] = p.offset;
        hi[f.axis] = p.offset;
        Rect { lo, hi }
    };

    // Labels.
    let drafts: &[DraftSpec] = match &job.modifier {
        Some(Modifier::Draft(ds)) => ds,
        _ => &[],
    };
    let draft_reach = drafts.iter().map(|d| d.reach).fold(0.0, f64::max);
    let mut labels: Vec<Vec<HistorySource>> = Vec::with_capacity(facets.len());
    let mut kinds: Vec<FacetKind> = Vec::with_capacity(facets.len());
    let mut exacts: Vec<FacetExact> = Vec::with_capacity(facets.len());
    for f in &facets {
        let plane = facet_plane(f);
        let r = facet_rect(f);
        let mut cen = [0.0; 3];
        for a in 0..3 {
            cen[a] = (r.lo[a] + r.hi[a]) / 2.0;
        }
        let cin = center(f.c);
        // The cell across the facet (or the mirror point past the grid's edge).
        let cout = {
            let mut nc = f.c;
            let inside_grid = if f.dir > 0 {
                nc[f.axis] += 1;
                nc[f.axis] < d[f.axis]
            } else if nc[f.axis] > 0 {
                nc[f.axis] -= 1;
                true
            } else {
                false
            };
            if inside_grid {
                center(nc)
            } else {
                let mut m = cin;
                m[plane.axis] = 2.0 * plane.offset - cin[plane.axis];
                m
            }
        };
        let mut ls: Vec<HistorySource> = Vec::new();
        let mut kind = FacetKind::Planar;
        let mut ex = FacetExact::None;
        // A drafted face's tilted surface: the draft deciding either side of the facet whose
        // material changes across it. On the face's flank (see [`DraftSpec::flank`]) the
        // draft's half-space decides; elsewhere it leaves the facet to the operand's faces.
        for d in drafts
            .iter()
            .filter(|d| d.region(cin) || d.region(cout) || d.corner(cin) || d.corner(cout))
        {
            let side = |p: [f64; 3]| !d.flank(p) || d.material(p);
            if side(cin) != side(cout) {
                ls.push(HistorySource::Input {
                    from: OperandEntity {
                        operand: job.operand_ids[0],
                        entity: d.face,
                    },
                });
                kind = FacetKind::Faceted(SurfaceKind::Plane { normal: d.normal() });
                break;
            }
        }
        if ls.is_empty() {
            for (k, o) in job.operands.iter().enumerate() {
                let bool_mode = matches!(
                    job.mode,
                    Mode::BoolUnion | Mode::BoolCut | Mode::BoolIntersect
                );
                if owners[f.cell] & (1 << k) == 0 && !matches!(job.mode, Mode::Add) && !bool_mode {
                    continue;
                }
                for (fi, face) in o.faces.iter().enumerate() {
                    if let Some(Modifier::Fill(fs)) = &job.modifier {
                        if fs.faces.contains(&(fi as u32)) {
                            continue;
                        }
                    }
                    if face.lies_on(&plane, cen) {
                        ls.push(HistorySource::Input {
                            from: OperandEntity {
                                operand: job.operand_ids[k],
                                entity: fi as u32,
                            },
                        });
                        if let FacetKind::Faceted(h) = face.kind() {
                            kind = FacetKind::Faceted(h);
                            if let Some(e) = &face.exact {
                                ex = FacetExact::Ready(e.clone());
                            }
                        }
                    }
                }
            }
        }
        if ls.is_empty() && job.mode == Mode::BoolCut {
            // Cavity walls lie on the tool bodies' faces, facing the other way.
            let flipped = PlaneKey {
                dir: -plane.dir,
                ..plane
            };
            for (k, o) in job.operands.iter().enumerate().skip(1) {
                for (fi, face) in o.faces.iter().enumerate() {
                    if face.lies_on(&flipped, cen) {
                        ls.push(HistorySource::Input {
                            from: OperandEntity {
                                operand: job.operand_ids[k],
                                entity: fi as u32,
                            },
                        });
                        if let FacetKind::Faceted(h) = face.kind() {
                            kind = FacetKind::Faceted(h);
                            if let Some(e) = &face.exact {
                                ex = FacetExact::Ready(e.clone());
                            }
                        }
                    }
                }
            }
        }
        if ls.is_empty() && !drafts.is_empty() {
            // A neighbour of a drafted face, extended or trimmed on its own plane (out to a
            // mitred corner: within the reach along each axis).
            let o = job.operands[0];
            let mut best: Option<(f64, usize)> = None;
            for (fi, face) in o.faces.iter().enumerate() {
                if face.plane.map(|p| p.same(&plane)).unwrap_or(false) {
                    let d = face_distance(cen, face);
                    if d <= draft_reach * 3f64.sqrt() + EPS && best.map(|b| d < b.0).unwrap_or(true)
                    {
                        best = Some((d, fi));
                    }
                }
            }
            if let Some((_, fi)) = best {
                ls.push(HistorySource::Input {
                    from: OperandEntity {
                        operand: job.operand_ids[0],
                        entity: fi as u32,
                    },
                });
            }
        }
        if let (true, Some(Modifier::Fill(fs))) = (ls.is_empty(), &job.modifier) {
            let o = job.operands[0];
            let mut best: Option<(f64, usize)> = None;
            for (fi, face) in o.faces.iter().enumerate() {
                if fs.faces.contains(&(fi as u32)) {
                    continue;
                }
                if face.plane.map(|p| p.same(&plane)).unwrap_or(false) {
                    let d = face_distance(cen, face);
                    if best.map(|b| d < b.0).unwrap_or(true) {
                        best = Some((d, fi));
                    }
                }
            }
            if let Some((_, fi)) = best {
                ls.push(HistorySource::Input {
                    from: OperandEntity {
                        operand: job.operand_ids[0],
                        entity: fi as u32,
                    },
                });
            }
        }
        if let (true, Some(Modifier::Shell(sh))) = (ls.is_empty(), &job.modifier) {
            if sh.outward {
                let o = job.operands[0];
                let rim = o.faces.iter().enumerate().find(|(fi, f)| {
                    sh.open.contains(&(*fi as u32))
                        && f.plane.map(|p| p.same(&plane)).unwrap_or(false)
                });
                if o.contains(cout) {
                    // The cavity: the body's own faces, now facing into it.
                    let flipped = PlaneKey {
                        dir: -plane.dir,
                        ..plane
                    };
                    for (fi, face) in o.faces.iter().enumerate() {
                        if face.lies_on(&flipped, cen) {
                            ls.push(HistorySource::Input {
                                from: OperandEntity {
                                    operand: job.operand_ids[0],
                                    entity: fi as u32,
                                },
                            });
                        }
                    }
                } else if let Some((fi, _)) = rim {
                    let from = OperandEntity {
                        operand: job.operand_ids[0],
                        entity: fi as u32,
                    };
                    ls.push(gen(
                        Role::ShellRim,
                        KernelTag::InputFace { face: from },
                        None,
                    ));
                } else if let Some(fi) = sh.outer_of(o, cen, &plane) {
                    let from = OperandEntity {
                        operand: job.operand_ids[0],
                        entity: fi,
                    };
                    ls.push(gen(
                        Role::ShellOuter,
                        KernelTag::InputFace { face: from },
                        None,
                    ));
                }
            }
        }
        let inward_shell = job
            .modifier
            .as_ref()
            .filter(|m| !matches!(m, Modifier::Shell(sh) if sh.outward));
        if let Some(Modifier::Shell(sh)) = inward_shell {
            let o = job.operands[0];
            for l in ls.iter_mut() {
                if let HistorySource::Input { from } = l {
                    if sh.open.contains(&from.entity) {
                        *l = gen(Role::ShellRim, KernelTag::InputFace { face: *from }, None);
                    }
                }
            }
            if ls.is_empty() {
                let mut best: Option<(f64, usize)> = None;
                for (fi, face) in o.faces.iter().enumerate() {
                    if sh.open.contains(&(fi as u32)) || !face.is_grid() {
                        continue;
                    }
                    let d = (face_distance(cen, face) - sh.of(fi as u32)).abs();
                    if best.map(|b| d < b.0 - EPS).unwrap_or(true) {
                        best = Some((d, fi));
                    }
                }
                if let Some((_, fi)) = best {
                    let face = &o.faces[fi];
                    kind = match face.plane {
                        Some(p) if p.axis == plane.axis => FacetKind::Planar,
                        _ => FacetKind::Faceted(SurfaceKind::Other),
                    };
                    ls.push(gen(
                        Role::ShellInner,
                        KernelTag::InputFace {
                            face: OperandEntity {
                                operand: job.operand_ids[0],
                                entity: fi as u32,
                            },
                        },
                        None,
                    ));
                }
            }
        }
        if ls.is_empty() {
            for (ti, t) in job.tools.iter().chain(&job.carve).enumerate() {
                if let Some((l, k, e)) = t.label(&plane, cen, cin, cout) {
                    ls.push(l);
                    kind = k;
                    ex = match e {
                        FacetExact::Surface(i) => FacetExact::Tool(ti, i),
                        other => other,
                    };
                    break;
                }
            }
        }
        if ls.is_empty() {
            if let Some((sa, c, s)) = job.split {
                if sa == plane.axis && (plane.offset - c).abs() < EPS {
                    // The facet's solid is on the side opposite its outward direction.
                    let solid_positive = (plane.dir as f64) * s < 0.0;
                    let side = if solid_positive {
                        PlaneSide::Positive
                    } else {
                        PlaneSide::Negative
                    };
                    ls.push(gen(Role::SplitFace, KernelTag::Side { side }, None));
                }
            }
        }
        if ls.is_empty() {
            return Err(KernelError::HistoryIncomplete {
                message: format!("fake kernel: facet at {:?} has no source", cen),
            });
        }
        ls.sort();
        ls.dedup();
        labels.push(ls);
        kinds.push(kind);
        exacts.push(if kind == FacetKind::Planar {
            FacetExact::None
        } else {
            ex
        });
    }

    // Faces: planar facets join coplanar neighbours (so flush faces merge, see
    // [`coplanar_may_merge`]); a faceted facet
    // joins any facet sharing a facet edge and a label (one curved surface, one face).
    let mut fparent: Vec<usize> = (0..facets.len()).collect();
    for (fi, f) in facets.iter().enumerate() {
        if kinds[fi] != FacetKind::Planar {
            continue;
        }
        for b in 0..3 {
            if b == f.axis || f.c[b] + 1 >= d[b] {
                continue;
            }
            let mut nc = f.c;
            nc[b] += 1;
            if let Some(&gj) = facet_at.get(&(idx(nc), f.axis, f.dir)) {
                if facets[gj].comp == f.comp
                    && kinds[gj] == FacetKind::Planar
                    && facet_plane(&facets[gj]).same(&facet_plane(f))
                    && coplanar_may_merge(&labels[fi], &labels[gj], job.merge_collinear)
                {
                    let (ra, rb) = (find(&mut fparent, fi), find(&mut fparent, gj));
                    if ra != rb {
                        fparent[ra.max(rb)] = ra.min(rb);
                    }
                }
            }
        }
    }
    if kinds.iter().any(|k| *k != FacetKind::Planar) {
        let mut seg_facets: BTreeMap<SegmentKey, Vec<usize>> = BTreeMap::new();
        for (fi, f) in facets.iter().enumerate() {
            let ia = if f.dir > 0 {
                f.c[f.axis] + 1
            } else {
                f.c[f.axis]
            };
            let others: Vec<usize> = (0..3).filter(|a| *a != f.axis).collect();
            for (along, fixed) in [(others[0], others[1]), (others[1], others[0])] {
                for off in 0..2 {
                    let mut pt = f.c;
                    pt[f.axis] = ia;
                    pt[fixed] = f.c[fixed] + off;
                    seg_facets.entry((along, pt)).or_default().push(fi);
                }
            }
        }
        for list in seg_facets.values() {
            for (x, &i) in list.iter().enumerate() {
                for &j in &list[x + 1..] {
                    if facets[i].comp != facets[j].comp
                        || (kinds[i] == FacetKind::Planar && kinds[j] == FacetKind::Planar)
                    {
                        continue;
                    }
                    if labels[i].iter().any(|l| labels[j].contains(l)) {
                        let (ra, rb) = (find(&mut fparent, i), find(&mut fparent, j));
                        if ra != rb {
                            fparent[ra.max(rb)] = ra.min(rb);
                        }
                    }
                }
            }
        }
    }
    // Face order: by component, then first facet.
    let mut face_of_root: BTreeMap<usize, usize> = BTreeMap::new();
    let mut comp_faces: Vec<Vec<usize>> = vec![Vec::new(); ncomp]; // roots per comp in order
    let mut facet_face = vec![0usize; facets.len()];
    for fi in 0..facets.len() {
        let r = find(&mut fparent, fi);
        let comp = facets[fi].comp;
        let local = match face_of_root.get(&r) {
            Some(l) => *l,
            None => {
                let l = comp_faces[comp].len();
                comp_faces[comp].push(r);
                face_of_root.insert(r, l);
                l
            }
        };
        facet_face[fi] = local;
    }

    let mut solids = Vec::with_capacity(ncomp);
    let mut face_sources = Vec::with_capacity(ncomp);
    let mut body_sources = Vec::with_capacity(ncomp);
    for comp in 0..ncomp {
        let nf = comp_faces[comp].len();
        let mut faces: Vec<FFace> = vec![FFace::blend(); nf];
        let mut fsrc: Vec<BTreeSet<HistorySource>> = vec![BTreeSet::new(); nf];
        let mut planar = vec![true; nf];
        for (fi, f) in facets.iter().enumerate() {
            if f.comp != comp {
                continue;
            }
            let l = facet_face[fi];
            let pk = facet_plane(f);
            if let FacetKind::Faceted(h) = kinds[fi] {
                planar[l] = false;
                if faces[l].surface.is_none() {
                    faces[l].surface = Some(h);
                }
                if faces[l].exact.is_none() {
                    faces[l].exact = match &exacts[fi] {
                        FacetExact::Ready(e) => Some(e.clone()),
                        FacetExact::Tool(ti, i) => job
                            .tools
                            .iter()
                            .chain(&job.carve)
                            .nth(*ti)
                            .and_then(|t| t.exact_surface(*i))
                            .map(Arc::new),
                        _ => None,
                    };
                }
            }
            if let Some(p0) = faces[l]
                .plane
                .or_else(|| faces[l].facet_planes.first().copied())
            {
                if !p0.same(&pk) {
                    planar[l] = false;
                }
            }
            if faces[l].plane.is_none() && faces[l].facet_planes.is_empty() {
                faces[l].plane = Some(pk);
            }
            faces[l].facet_planes.push(pk);
            faces[l].rects.push(facet_rect(f));
            fsrc[l].extend(labels[fi].iter().cloned());
        }
        for (l, face) in faces.iter_mut().enumerate() {
            if planar[l] {
                face.facet_planes.clear();
                face.surface = None;
                face.exact = None;
            } else {
                face.plane = None;
            }
        }
        let mut cocc = vec![false; n];
        let mut bsrc: BTreeSet<HistorySource> = BTreeSet::new();
        let mut has_tool_only = false;
        for i in 0..n {
            if comp_of[i] == comp {
                cocc[i] = true;
                for k in 0..job.operands.len() {
                    if owners[i] & (1 << k) != 0 {
                        bsrc.insert(HistorySource::Input {
                            from: OperandEntity {
                                operand: job.operand_ids[k],
                                entity: 0,
                            },
                        });
                    }
                }
                if owners[i] == 0 {
                    has_tool_only = true;
                }
            }
        }
        let mut bvec: Vec<HistorySource> = bsrc.into_iter().collect();
        if bvec.is_empty() && has_tool_only {
            match &job.body_label {
                Some(per_tool) => {
                    let first = (0..n)
                        .find(|i| comp_of[*i] == comp)
                        .expect("component has cells");
                    let t = tools[first].trailing_zeros() as usize;
                    bvec.push(
                        per_tool
                            .get(t)
                            .cloned()
                            .unwrap_or_else(|| gen(Role::Body, KernelTag::None, None)),
                    );
                }
                None => bvec.push(gen(Role::Body, KernelTag::None, None)),
            }
        }
        // Split side tag for bodies of an operand cut in two.
        if let Some((a, c, s)) = job.split {
            let first = (0..n)
                .find(|i| comp_of[*i] == comp)
                .expect("component has cells");
            let ci = [first % d[0], (first / d[0]) % d[1], first / (d[0] * d[1])];
            let p = center(ci);
            let side = if (p[a] - c) * s > 0.0 {
                PlaneSide::Positive
            } else {
                PlaneSide::Negative
            };
            bvec.push(gen(Role::Body, KernelTag::Side { side }, None));
        }
        let solid = FakeSolid {
            grid: grid.clone(),
            occ: cocc,
            faces,
            edge_meta: Vec::new(),
            decorations: Vec::new(),
            topology: Topology::default(),
        };
        solids.push(solid);
        face_sources.push(fsrc.into_iter().map(|s| s.into_iter().collect()).collect());
        body_sources.push(bvec);
    }
    Ok(GridResult {
        solids,
        face_sources,
        body_sources,
    })
}
