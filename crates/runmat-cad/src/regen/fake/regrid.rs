//! What a re-gridding job carries: its mode, shell, draft and fill specifications, and the
//! distance helpers they use.

use super::*;

// ------------------------------------------------------------------------------- re-gridding

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mode {
    New,
    Add,
    Cut,
    Intersect,
    BoolUnion,
    BoolCut,
    BoolIntersect,
    Split,
    Keep,
}

pub(super) struct GridJob<'a> {
    pub(super) operands: Vec<&'a FakeSolid>,
    /// Operand indices into the operation's operand list (for labels).
    pub(super) operand_ids: Vec<u32>,
    pub(super) tools: Vec<Tool>,
    pub(super) mode: Mode,
    /// Split plane: axis, offset, normal sign.
    pub(super) split: Option<(usize, f64, f64)>,
    /// For imports: one tool part per body.
    pub(super) separate_tools: bool,
    pub(super) body_label: Option<Vec<HistorySource>>,
    /// A shell or draft applied to the (single) operand in `Mode::Keep`.
    pub(super) modifier: Option<Modifier>,
    /// Side faces of collinear profile curves merge into one face (the definition asked).
    pub(super) merge_collinear: bool,
    /// Tools removed from the operands before `tools` combine (a flange trimming its sheet).
    pub(super) carve: Vec<Tool>,
}

#[derive(Clone, Debug)]
pub(super) struct ShellSpec {
    pub(super) open: BTreeSet<u32>,
    pub(super) thickness: f64,
    /// Walls with their own thickness.
    pub(super) per_face: BTreeMap<u32, f64>,
    /// Walls grow outside the body, which becomes the cavity.
    pub(super) outward: bool,
}

impl ShellSpec {
    pub(super) fn of(&self, face: u32) -> f64 {
        self.per_face.get(&face).copied().unwrap_or(self.thickness)
    }

    /// For an outward shell, whether `p` (outside the body) is wall: within a face's thickness
    /// of it (offset faces meeting in sharp corners, so the distance is the largest per axis),
    /// and not in front of an open face.
    pub(super) fn outward_wall(&self, o: &FakeSolid, p: [f64; 3]) -> bool {
        let mut near = false;
        for (fi, f) in o.faces.iter().enumerate() {
            let t = self.of(fi as u32);
            if self.open.contains(&(fi as u32)) {
                if let Some(pk) = f.plane {
                    let ahead = (pk.dir as f64) * (p[pk.axis] - pk.offset) > EPS;
                    let across = f.rects.iter().any(|r| {
                        (0..3).all(|a| {
                            a == pk.axis || (p[a] > r.lo[a] - t - EPS && p[a] < r.hi[a] + t + EPS)
                        })
                    });
                    if ahead && across {
                        return false;
                    }
                }
                continue;
            }
            near |= face_cheb(p, f) <= t + EPS;
        }
        near
    }

    /// The face an outward wall facet at `p` on `plane` offsets: a face facing the same way
    /// at its thickness behind the facet, the nearest across it (corners go to the first).
    pub(super) fn outer_of(&self, o: &FakeSolid, p: [f64; 3], plane: &PlaneKey) -> Option<u32> {
        let mut best: Option<(f64, u32)> = None;
        for (fi, f) in o.faces.iter().enumerate() {
            if self.open.contains(&(fi as u32)) {
                continue;
            }
            let Some(fp) = f
                .plane
                .filter(|fp| fp.axis == plane.axis && fp.dir == plane.dir)
            else {
                continue;
            };
            let behind = (plane.dir as f64) * (plane.offset - fp.offset);
            if (behind - self.of(fi as u32)).abs() > EPS {
                continue;
            }
            let d = face_cheb(p, f);
            if best.map(|b| d < b.0 - EPS).unwrap_or(true) {
                best = Some((d, fi as u32));
            }
        }
        best.map(|b| b.1)
    }
}

#[derive(Clone, Debug)]
pub(super) struct DraftSpec {
    /// The drafted face's index in the operand.
    pub(super) face: u32,
    pub(super) plane: PlaneKey,
    pub(super) lo: [f64; 3],
    pub(super) hi: [f64; 3],
    pub(super) origin: [f64; 3],
    pub(super) pull: [f64; 3],
    pub(super) tan: f64,
    pub(super) reach: f64,
    /// Outward offset of the face plane (offset face, push-pull); zero for a draft.
    pub(super) shift: f64,
}

impl DraftSpec {
    pub(super) fn region(&self, p: [f64; 3]) -> bool {
        (0..3).all(|a| {
            if a == self.plane.axis {
                (p[a] - self.plane.offset).abs() <= self.reach + EPS
            } else {
                p[a] >= self.lo[a] - EPS && p[a] <= self.hi[a] + EPS
            }
        })
    }

    /// Within the face's extent along the pull, and across it out to the reach past its side
    /// edges, at any depth along the face's plane: where the draft's half-space decides the
    /// material either side of a facet (the far edge of an outward lean, mitred corners).
    pub(super) fn flank(&self, p: [f64; 3]) -> bool {
        (0..3).all(|a| {
            if a == self.plane.axis {
                true
            } else if self.pull[a].abs() > 1e-12 {
                p[a] >= self.lo[a] - EPS && p[a] <= self.hi[a] + EPS
            } else {
                p[a] >= self.lo[a] - self.reach - EPS && p[a] <= self.hi[a] + self.reach + EPS
            }
        })
    }

    /// Past the face's side edges (across the pull, not along it), within reach of its
    /// plane: where drafts of neighbouring faces that lean out meet in a mitred corner.
    pub(super) fn corner(&self, p: [f64; 3]) -> bool {
        !self.region(p)
            && (0..3).all(|a| {
                if a == self.plane.axis {
                    (p[a] - self.plane.offset).abs() <= self.reach + EPS
                } else if self.pull[a].abs() > 1e-12 {
                    p[a] >= self.lo[a] - EPS && p[a] <= self.hi[a] + EPS
                } else {
                    p[a] >= self.lo[a] - self.reach - EPS && p[a] <= self.hi[a] + self.reach + EPS
                }
            })
    }

    /// Whether `p` is material after the draft (near this face).
    pub(super) fn material(&self, p: [f64; 3]) -> bool {
        let h = (0..3)
            .map(|a| (p[a] - self.origin[a]) * self.pull[a])
            .sum::<f64>();
        (self.plane.dir as f64) * (p[self.plane.axis] - self.plane.offset) - self.shift
            < -self.tan * h
    }

    pub(super) fn normal(&self) -> [f64; 3] {
        let mut n = [0.0; 3];
        n[self.plane.axis] = self.plane.dir as f64;
        for a in 0..3 {
            n[a] += self.tan * self.pull[a];
        }
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        [n[0] / l, n[1] / l, n[2] / l]
    }
}

#[derive(Clone, Debug)]
pub(super) enum Modifier {
    Shell(ShellSpec),
    Draft(Vec<DraftSpec>),
    /// Delete faces and heal: the region the deleted faces bound is filled (a cavity) or
    /// cleared (a protrusion); neighbours extend over it on their own planes.
    Fill(FillSpec),
}

#[derive(Clone, Debug)]
pub(super) struct FillSpec {
    pub(super) faces: BTreeSet<u32>,
    pub(super) lo: [f64; 3],
    pub(super) hi: [f64; 3],
    pub(super) cavity: bool,
}

pub(super) fn rect_distance(p: [f64; 3], r: &Rect) -> f64 {
    let mut s = 0.0;
    for a in 0..3 {
        let d = (r.lo[a] - p[a]).max(p[a] - r.hi[a]).max(0.0);
        s += d * d;
    }
    s.sqrt()
}

/// The largest per-axis distance from `p` to a face's rects.
pub(super) fn face_cheb(p: [f64; 3], f: &FFace) -> f64 {
    f.rects
        .iter()
        .map(|r| {
            (0..3)
                .map(|a| (r.lo[a] - p[a]).max(p[a] - r.hi[a]).max(0.0))
                .fold(0.0, f64::max)
        })
        .fold(f64::INFINITY, f64::min)
}

pub(super) fn face_distance(p: [f64; 3], f: &FFace) -> f64 {
    f.rects
        .iter()
        .map(|r| rect_distance(p, r))
        .fold(f64::INFINITY, f64::min)
}

pub(super) struct GridResult {
    pub(super) solids: Vec<FakeSolid>,
    pub(super) face_sources: Vec<Vec<Vec<HistorySource>>>,
    pub(super) body_sources: Vec<Vec<HistorySource>>,
}

pub(super) fn dedup_sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.retain(|x| x.is_finite());
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let mut out: Vec<f64> = Vec::new();
    for x in v {
        if out.last().map(|l| (x - l).abs() > EPS).unwrap_or(true) {
            out.push(x);
        }
    }
    out
}

/// Whether two coplanar, adjacent planar facets belong to one face. Faces from different
/// sources merge where they are flush (a tool cap flush with an operand face, two operands
/// meeting in a union). Two faces that were already separate stay separate: distinct faces of
/// one operand, and side faces of distinct profile curves (collinear sketch lines), unless the
/// definition asks for collinear sides to merge.
pub(super) fn coplanar_may_merge(
    a: &[HistorySource],
    b: &[HistorySource],
    merge_collinear: bool,
) -> bool {
    if a.iter().any(|l| b.contains(l)) {
        return true;
    }
    let side_curve = |l: &HistorySource| {
        matches!(
            l,
            HistorySource::Generated {
                role: Role::Side,
                tag: KernelTag::Curve { .. },
                ..
            }
        )
    };
    if !a.is_empty() && !b.is_empty() && a.iter().all(side_curve) && b.iter().all(side_curve) {
        return merge_collinear;
    }
    let operand_of = |ls: &[HistorySource]| -> Option<u32> {
        let mut op = None;
        for l in ls {
            match l {
                HistorySource::Input { from } => match op {
                    None => op = Some(from.operand),
                    Some(o) if o == from.operand => {}
                    Some(_) => return None,
                },
                _ => return None,
            }
        }
        op
    };
    match (operand_of(a), operand_of(b)) {
        (Some(x), Some(y)) => x != y,
        _ => true,
    }
}
