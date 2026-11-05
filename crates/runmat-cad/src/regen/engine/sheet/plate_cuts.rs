//! The flat pattern of a plate that features besides its bends changed: cuts, holes and
//! notches in its flats develop onto the flat pattern.
//!
//! The developed plate is split into flat regions by its bends: the plate's unmoved part,
//! and the flat each sketched bend or jog moved (minus the flats bends of that flat moved in
//! turn). A region carries the rigid motion `M` from the developed plate to where its flat
//! lies in the part (the identity for the unmoved part). For each region the steps are:
//!
//! 1. a slab: the region's area of the developed plane (ending 1 µm short of each bend's
//!    tangent line, unbounded elsewhere), from `−t` to `2t` across the sheet;
//! 2. a copy of the part's body moved back by `M⁻¹`, so that region's flat lies on the
//!    developed plate;
//! 3. the slab minus that copy: what the features removed from the region, plus slab outside
//!    the sheet.
//!
//! The developed plate minus every region's result is the flat pattern. Each cut lands where
//! its material was on the developed plate. Outside the sheet's thickness the slabs cut
//! nothing, and the slab of one region holds no other region's material.
//!
//! Every face the other features made must lie on one flat, within the sheet's thickness:
//! material removed across a bend zone or a jog's middle flat, and material added off the
//! sheet (a boss), has no developed position, so the flat pattern refuses it and names it.

use super::super::families::inverse;
use super::*;

/// A flat region of a developed plate: `keep` half-planes `(p − o)·n ≥ d` (all must hold),
/// and the motion from the developed plate to where the flat lies.
struct Region {
    key: String,
    keep: Vec<([f64; 2], [f64; 2], f64)>,
    motion: Option<Transform>,
}

/// How far short of a bend's tangent line a region's slab stops.
const SHY: f64 = 1e-6;

impl Region {
    fn holds(&self, p: [f64; 2], tol: f64) -> bool {
        self.keep
            .iter()
            .all(|(o, n, d)| (p[0] - o[0]) * n[0] + (p[1] - o[1]) * n[1] >= d - tol)
    }
}

/// The convex polygon `poly` clipped to `(p − o)·n ≥ d`.
fn clip(poly: &[[f64; 2]], o: [f64; 2], n: [f64; 2], d: f64) -> Vec<[f64; 2]> {
    let side = |p: [f64; 2]| (p[0] - o[0]) * n[0] + (p[1] - o[1]) * n[1] - d;
    let mut out = Vec::new();
    for k in 0..poly.len() {
        let (p, q) = (poly[k], poly[(k + 1) % poly.len()]);
        let (sp, sq) = (side(p), side(q));
        if sp >= 0.0 {
            out.push(p);
        }
        if (sp >= 0.0) != (sq >= 0.0) {
            let f = sp / (sp - sq);
            out.push([p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1])]);
        }
    }
    out
}

/// Positions of a body list through a run of steps, by label, as `commit` moves them.
struct Positions {
    labels: Vec<u32>,
    next: u32,
}

impl Positions {
    fn new(n: usize) -> Self {
        Self {
            labels: (0..n as u32).collect(),
            next: n as u32,
        }
    }

    fn at(&self, label: u32) -> usize {
        self.labels
            .iter()
            .position(|l| *l == label)
            .expect("a live body")
    }

    /// Records a step on `operands` (labels) giving `outputs` bodies; returns their labels.
    fn step(&mut self, operands: &[u32], outputs: usize) -> Vec<u32> {
        let positions: Vec<usize> = operands.iter().map(|l| self.at(*l)).collect();
        let insert_at = positions.first().copied().unwrap_or(self.labels.len());
        let mut gone = positions;
        gone.sort_unstable();
        gone.dedup();
        for &p in gone.iter().rev() {
            self.labels.remove(p);
        }
        let insert_at = insert_at.min(self.labels.len());
        let new: Vec<u32> = (0..outputs as u32).map(|k| self.next + k).collect();
        self.next += outputs as u32;
        for (k, l) in new.iter().enumerate() {
            self.labels.insert(insert_at + k, *l);
        }
        new
    }
}

impl<'a, 'b> Run<'a, 'b> {
    /// The flat regions of a plate bent by `bends`.
    fn plate_regions(&self, bends: &[(FeatureId, Vec<FlatBendLine>)]) -> Vec<Region> {
        let mut out = vec![Region {
            key: "plate".into(),
            keep: Vec::new(),
            motion: None,
        }];
        let record = |f: &FeatureId| match self.st.records.get(f) {
            Some(FeatureRecord::PlateBend {
                lines,
                moved,
                motion,
                parent,
                ..
            }) => Some((lines.clone(), *moved, *motion, parent.clone())),
            _ => None,
        };
        for (b, _) in bends {
            let Some((lines, (o, n), motion, _)) = record(b) else {
                continue;
            };
            // The moved flat starts past the last bend line's allowance.
            let start = lines
                .iter()
                .map(|l| {
                    let mid = [(l.start[0] + l.end[0]) / 2.0, (l.start[1] + l.end[1]) / 2.0];
                    (mid[0] - o[0]) * n[0] + (mid[1] - o[1]) * n[1] + l.allowance / 2.0
                })
                .fold(0.0, f64::max);
            out.push(Region {
                key: b.0.clone(),
                keep: vec![(o, n, start + SHY)],
                motion: Some(motion),
            });
        }
        // Each bend's moved half-plane leaves its parent's region (the plate's, without one).
        for (b, _) in bends {
            let Some((_, (o, n), _, parent)) = record(b) else {
                continue;
            };
            let owner = parent.map(|p| p.0).unwrap_or_else(|| "plate".into());
            if let Some(r) = out.iter_mut().find(|r| r.key == owner) {
                r.keep.push((o, [-n[0], -n[1]], SHY));
            }
        }
        out
    }

    /// The steps that cut the developed plate (body `developed`, already made by the first
    /// step) by what features besides the bends removed from the flats of `body`. `foreign`
    /// are the faces of `body` those features made. Returns the steps after the first, and the
    /// developed plate's position after them.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn plate_cut_steps(
        &mut self,
        id: &FeatureId,
        body: usize,
        bends: &[(FeatureId, Vec<FlatBendLine>)],
        plate: &CanonicalProfile,
        t: f64,
        foreign: &[usize],
        naming: &NamingContext,
    ) -> Result<(Vec<Step>, usize), Halt> {
        let regions = self.plate_regions(bends);
        let frame = plate.frame;
        let local = |p: [f64; 3]| {
            let q = sub(p, frame.origin);
            [dot(q, frame.x), dot(q, frame.y), dot(q, frame.z)]
        };
        // Every face the other features made lies on one flat, within the sheet.
        let live = &self.st.bodies[body];
        let tol = 1e-7;
        let mut refused: Option<String> = None;
        for &f in foreign {
            let mut points = vec![live.topology.faces[f].geom.centroid];
            for e in &live.topology.edges {
                if e.faces.contains(&(f as u32)) {
                    points.push(e.geom.start);
                    points.push(e.geom.end);
                }
            }
            let on_a_flat = regions.iter().any(|r| {
                let back = r.motion.as_ref().map(inverse);
                points.iter().all(|p| {
                    let p = back.as_ref().map_or(*p, |m| m.apply(*p));
                    let q = local(p);
                    q[2] >= -tol && q[2] <= t + tol && r.holds([q[0], q[1]], tol + SHY)
                })
            });
            if !on_a_flat {
                let name = live.names.faces[f]
                    .iter()
                    .next()
                    .map(|n| {
                        format!(
                            "`{}`",
                            Shown(&Reference::face(n.clone()), Some(&self.st.names))
                        )
                    })
                    .unwrap_or_else(|| "a face".into());
                refused = Some(name);
                break;
            }
        }
        if let Some(name) = refused {
            return Err(self.not_lowered(
                id,
                &format!(
                    "flat patterns of material changed across a bend zone, on a jog's middle flat or off the sheet ({name} lies on no flat of the plate)"
                ),
            ));
        }
        // The slabs, in the developed plate's frame.
        let outline = plate
            .loops
            .first()
            .map(|l| l.iter().map(|s| s.start).collect::<Vec<_>>())
            .unwrap_or_default();
        let reach = outline
            .iter()
            .map(|p| p[0].abs().max(p[1].abs()))
            .fold(0.0, f64::max)
            * 4.0
            + 10.0 * t;
        let mut pos = Positions::new(self.st.bodies.len());
        let developed = pos.step(&[], 1)[0];
        let mut actual = body as u32;
        let mut steps = Vec::new();
        let mut holes = Vec::new();
        for r in &regions {
            let mut poly = vec![
                [-reach, -reach],
                [reach, -reach],
                [reach, reach],
                [-reach, reach],
            ];
            for (o, n, d) in &r.keep {
                poly = clip(&poly, *o, *n, *d);
            }
            if poly.len() < 3 {
                continue;
            }
            let corners: Vec<(String, [f64; 2])> = poly
                .iter()
                .enumerate()
                .map(|(k, p)| (format!("{}.s{k}", r.key), *p))
                .collect();
            let mut slab = LabeledTool::new(ToolShape::prism(
                super::super::families::polygon(frame, &corners),
                -t,
                2.0 * t,
            ));
            // Labelled like the sheet's edges, by region and side: the flat pattern keeps any
            // face of a slab only where a cut meets the plate's outline.
            let slab_key = format!("{}.slab", r.key);
            for (k, (c, _)) in corners.iter().enumerate() {
                slab.sides.push((
                    crate::ids::SketchEntityId::new(c.clone()),
                    FaceLabel::key(Role::SheetEdge, slab_key.clone()).with_ordinal(k as u32 + 2),
                ));
            }
            slab.start = Some(FaceLabel::key(Role::SheetEdge, slab_key.clone()).with_ordinal(0));
            slab.end = Some(FaceLabel::key(Role::SheetEdge, slab_key).with_ordinal(1));
            let s = pos.step(&[], 1)[0];
            steps.push(Step {
                op: labeled(vec![slab], Vec::new(), Combine::New, false),
                operands: Vec::new(),
                naming: naming.at_step(Some(format!("{}.slab", r.key)), Vec::new()),
            });
            let back = r
                .motion
                .as_ref()
                .map(inverse)
                .unwrap_or_else(|| Transform::translation([0.0; 3]));
            let at = pos.at(actual);
            let copied = pos.step(&[actual], 2);
            steps.push(Step {
                op: KernelOp::PatternBodies {
                    instances: vec![Instance {
                        key: InstanceKey::Row {
                            key: format!("{}.develop", r.key),
                        },
                        transform: back,
                    }],
                },
                operands: vec![at],
                naming: naming.at_step(Some(format!("{}.develop", r.key)), Vec::new()),
            });
            // The body itself stays where it was, under a new label.
            let copy = copied[1];
            actual = copied[0];
            let operands = vec![pos.at(s), pos.at(copy)];
            let hole = pos.step(&[s, copy], 1)[0];
            steps.push(Step {
                op: KernelOp::Boolean {
                    combine: Combine::Cut,
                    keep_tools: false,
                },
                operands,
                naming: naming.at_step(Some(format!("{}.removed", r.key)), Vec::new()),
            });
            holes.push(hole);
        }
        let mut labels = vec![developed];
        labels.extend(&holes);
        let operands: Vec<usize> = labels.iter().map(|l| pos.at(*l)).collect();
        let cut = pos.step(&labels, 1)[0];
        steps.push(Step {
            op: KernelOp::Boolean {
                combine: Combine::Cut,
                keep_tools: false,
            },
            operands,
            naming: naming.at_step(Some("cut".into()), Vec::new()),
        });
        Ok((steps, pos.at(cut)))
    }
}
