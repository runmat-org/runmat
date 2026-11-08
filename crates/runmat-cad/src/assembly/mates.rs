//! The mate solve for one scope: an assembly level plus every flexible sub-assembly under it.
//!
//! Bodies are the scope's member occurrences (rigid sub-assemblies are one body each). Fixed
//! members of the scope root are ground; fixed members of a floating flexible sub-assembly are
//! fastened to it (its anchor rides on its frame, and is not reported as a separate free body).
//! A flexible sub-assembly with no fixed member keeps its frame at its placement and only its
//! members solve. Connectors on occurrences deeper than a member body are carried to that body
//! by the placement chain. Mates become `runmat-constraint` mates; joint limits become range
//! targets, which the solver clamps when they bind.
//!
//! Pattern and mirror instances follow their seeds. Mates touching instances are solved in a
//! second pass after the instances are placed, with everything the first pass determined held
//! fixed, so a mate that would drag an instance away from its seed reports a conflict instead of
//! silently desynchronising the pattern.

use crate::assembly::model::*;
use crate::assembly::resolve::{
    InstancePlan, MateOutcome, MateStatus, NodeId, NodeOrigin, PoseStatus, Resolver,
};
use crate::diagnostics::{DiagCode, Repair, Severity, Site, Subject};
use crate::metering::{Stopwatch, UsageEvent};
use crate::regen::kernel::Transform;
use crate::value::ValueEnv;
use runmat_constraint as rc;
use std::collections::{BTreeMap, BTreeSet};

fn to_frame(t: &Transform) -> rc::Frame {
    rc::Frame {
        origin: t.translation,
        rotation: t.quaternion(),
    }
}

struct ActiveMate {
    owner: NodeId,
    def: MateDef,
    env: ValueEnv,
    /// (host node, frame in host coordinates) per side, or the error.
    a: Result<(NodeId, Transform), String>,
    b: Result<(NodeId, Transform), String>,
    /// The kind's own connectors ([`MateKind::extra_connectors`]), in order.
    extra: Vec<Result<(NodeId, Transform), String>>,
    outcome: usize,
}

impl ActiveMate {
    fn sides(&self) -> impl Iterator<Item = &Result<(NodeId, Transform), String>> {
        [&self.a, &self.b].into_iter().chain(self.extra.iter())
    }
}

/// What a mate needs besides its two connectors.
#[derive(Default)]
struct Aux {
    /// A cam's station radii.
    cam: Option<Result<Vec<f64>, String>>,
    /// A path mate's points in `a`'s frame.
    path: Option<Result<Vec<[f64; 3]>, String>>,
    /// A symmetric mate's plane.
    plane: Option<rc::EntityId>,
}

impl<'a, 'b> Resolver<'a, 'b> {
    /// Members of the scope rooted at `node`, top-down.
    fn scope_members(&self, node: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut queue: Vec<NodeId> = self.nodes[node as usize].children.clone();
        let mut i = 0;
        while i < queue.len() {
            let m = queue[i];
            out.push(m);
            let n = &self.nodes[m as usize];
            if n.flexible && n.expanded {
                queue.extend(n.children.iter().copied());
            }
            i += 1;
        }
        out
    }

    /// The member body carrying `host` (itself or its nearest ancestor that is a member), or
    /// `None` for the scope root.
    fn body_of(&self, host: NodeId, members: &BTreeSet<NodeId>, root: NodeId) -> Option<NodeId> {
        let mut cur = host;
        loop {
            if cur == root {
                return None;
            }
            if members.contains(&cur) {
                return Some(cur);
            }
            cur = self.nodes[cur as usize].parent?;
        }
    }

    pub(super) fn solve_scope(&mut self, node: NodeId, plans: Vec<InstancePlan>) {
        let members = self.scope_members(node);
        // Plans deferred by flexible members join this scope.
        let mut plans = plans;
        for m in &members {
            if let Some(p) = self.deferred.remove(m) {
                plans.extend(p);
            }
        }
        // Mates of this level and of flexible members.
        let mut owners = vec![node];
        owners.extend(members.iter().copied().filter(|m| {
            let n = &self.nodes[*m as usize];
            n.flexible && n.expanded
        }));
        let mut active: Vec<ActiveMate> = Vec::new();
        for owner in owners {
            let Some(info) = self.asm.get(&owner) else {
                continue;
            };
            let (doc, env) = (info.doc.clone(), info.env.clone());
            for m in &doc.mates {
                let site = Site::with_param(
                    Subject::Mate {
                        name: m.name.clone(),
                    },
                    doc.path.as_str(),
                );
                let on = !m.suppressed && self.holds(m.when.as_ref(), &doc.path, &site);
                let outcome = self.mates.len();
                self.mates.push(MateOutcome {
                    owner,
                    name: m.name.clone(),
                    kind: m.kind.name().into(),
                    status: MateStatus::Inactive,
                    active_limits: Vec::new(),
                });
                if !on {
                    continue;
                }
                let a = self.connector(owner, &m.a);
                let b = self.connector(owner, &m.b);
                let extra = m.kind.extra_connectors().into_iter().map(|c| self.connector(owner, c)).collect();
                active.push(ActiveMate {
                    owner,
                    def: m.clone(),
                    env: env.clone(),
                    a,
                    b,
                    extra,
                    outcome,
                });
            }
        }
        // Failed connectors: diagnosed, the mate dropped.
        let member_set: BTreeSet<NodeId> = members.iter().copied().collect();
        let mut usable = Vec::new();
        for m in active {
            let failed = m.sides().find_map(|s| s.as_ref().err()).cloned();
            match failed {
                None => usable.push(m),
                Some(e) => {
                    let file = self.asm[&m.owner].doc.path.clone();
                    self.diag(
                        DiagCode::AsmUnknownConnector,
                        Severity::Error,
                        Site::with_param(
                            Subject::Mate {
                                name: m.def.name.clone(),
                            },
                            file.as_str(),
                        ),
                        format!("mate `{}`: {}", m.def.name, e),
                        vec![Repair::SuppressMate {
                            assembly: file.clone(),
                            mate: m.def.name.clone(),
                        }],
                    );
                    self.mates[m.outcome].status = MateStatus::Failed { message: e };
                }
            }
        }
        if usable.is_empty() {
            self.apply_instances(&plans);
            if !self.deferred_fasteners.is_empty() {
                self.flush_member_fasteners(&members);
            }
            return;
        }
        self.stats.mate_scopes += 1;
        self.stats.mates += usable.len() as u64;

        let instances: BTreeSet<NodeId> = plans.iter().map(|p| p.instance).collect();
        let touches_instance = |r: &Self, m: &ActiveMate| {
            m.sides().any(|s| {
                let (host, _) = s.as_ref().unwrap();
                r.body_of(*host, &member_set, node)
                    .map(|b| {
                        // The instance itself, or anything inside an instance's subtree.
                        let mut cur = Some(b);
                        while let Some(c) = cur {
                            if instances.contains(&c) {
                                return true;
                            }
                            if c == node {
                                break;
                            }
                            cur = r.nodes[c as usize].parent;
                        }
                        false
                    })
                    .unwrap_or(false)
            })
        };
        let (first, second): (Vec<ActiveMate>, Vec<ActiveMate>) =
            usable.into_iter().partition(|m| !touches_instance(self, m));

        let pass1_members: Vec<NodeId> = members
            .iter()
            .copied()
            .filter(|m| !self.in_instance(*m, &instances, node))
            .collect();
        let determined = self.solve_pass(node, &pass1_members, &first, &BTreeSet::new());
        self.apply_instances(&plans);
        if !second.is_empty() {
            let mut hold: BTreeSet<NodeId> = determined;
            hold.extend(instances.iter().copied());
            self.solve_pass(node, &members, &second, &hold);
        }
        self.flush_member_fasteners(&members);
    }

    fn in_instance(&self, m: NodeId, instances: &BTreeSet<NodeId>, root: NodeId) -> bool {
        let mut cur = Some(m);
        while let Some(c) = cur {
            if c == root {
                return false;
            }
            if instances.contains(&c) {
                return true;
            }
            cur = self.nodes[c as usize].parent;
        }
        false
    }

    fn flush_member_fasteners(&mut self, members: &[NodeId]) {
        for m in members {
            if self.deferred_fasteners.remove(m) {
                if let Some(info) = self.asm.get(m) {
                    let (doc, env) = (info.doc.clone(), info.env.clone());
                    self.expand_fasteners(*m, &doc, &env);
                }
            }
        }
    }

    /// One solve. `hold` members are ground in this pass. Returns the members the pass
    /// touched (so a later pass can hold them).
    fn solve_pass(
        &mut self,
        root: NodeId,
        members: &[NodeId],
        mates: &[ActiveMate],
        hold: &BTreeSet<NodeId>,
    ) -> BTreeSet<NodeId> {
        let member_set: BTreeSet<NodeId> = members.iter().copied().collect();
        let mut sys = rc::System::new();
        let mut body: BTreeMap<NodeId, rc::EntityId> = BTreeMap::new();
        let mut scope_pose: BTreeMap<NodeId, Transform> = BTreeMap::new();
        let mut grounded: BTreeSet<NodeId> = BTreeSet::new();
        // Anchorless flexible sub-assembly frames held at their placement.
        let mut frames: BTreeSet<NodeId> = BTreeSet::new();
        let root_ground = sys.ground(rc::Frame::IDENTITY);
        let mut internal: BTreeSet<rc::ConstraintId> = BTreeSet::new();
        for &m in members {
            let pose = self.chain(root, m);
            scope_pose.insert(m, pose);
            let n = &self.nodes[m as usize];
            let parent = n.parent.unwrap_or(root);
            let parent_ground = parent == root || grounded.contains(&parent);
            // A flexible sub-assembly with no fixed member (no anchor) has nothing tying its
            // own frame to its members: the frame keeps its placement relative to its parent
            // (it is not a free body), and its floating members solve against the scope.
            let frame_only = n.flexible
                && n.expanded
                && !n
                    .children
                    .iter()
                    .any(|c| member_set.contains(c) && self.nodes[*c as usize].fixed);
            let anchored = n.fixed || frame_only;
            let ground = hold.contains(&m) || (anchored && parent_ground);
            if frame_only && ground {
                frames.insert(m);
            }
            let e = if ground {
                grounded.insert(m);
                sys.ground(to_frame(&pose))
            } else {
                sys.body(to_frame(&pose))
            };
            body.insert(m, e);
            if anchored && !ground {
                // Fixed in a floating flexible sub-assembly (or an anchorless sub-assembly's
                // frame inside one): fastened to it.
                if let Some(&pb) = body.get(&parent) {
                    let ca = sys
                        .connector(pb, to_frame(&n.local))
                        .expect("parent is a body");
                    let cb = sys
                        .connector(e, rc::Frame::IDENTITY)
                        .expect("member is a body");
                    if let Ok(c) =
                        sys.constrain(rc::Constraint::Mate(rc::Mate::Fasten { a: ca, b: cb }))
                    {
                        internal.insert(c);
                    }
                }
            }
        }
        let mut owner_of: BTreeMap<rc::ConstraintId, usize> = BTreeMap::new();
        let mut touched: BTreeSet<NodeId> = BTreeSet::new();
        for (mi, m) in mates.iter().enumerate() {
            // Each side's body and its frame on that body.
            let mut end = |r: &Self, side: &Result<(NodeId, Transform), String>| {
                let (host, f) = side.as_ref().unwrap();
                match r.body_of(*host, &member_set, root) {
                    Some(b) => {
                        touched.insert(b);
                        (body[&b], r.chain(b, *host).compose(f))
                    }
                    None => (root_ground, r.chain(root, *host).compose(f)),
                }
            };
            let mut sides = vec![end(self, &m.a), end(self, &m.b)];
            let extra: Vec<(rc::EntityId, Transform)> = m.extra.iter().map(|s| end(self, s)).collect();
            let mut aux = Aux::default();
            let mut problem = None;
            match &m.def.kind {
                // Each pair's mid-plane: half way between its two connectors, on the first's axes.
                MateKind::Width { .. } => {
                    for (i, (pair, what)) in [(extra[0], "groove"), (extra[1], "tab")].into_iter().enumerate() {
                        if pair.0 != sides[i].0 {
                            problem = Some(format!("the {what}'s two faces are on different components"));
                        }
                        let t = &mut sides[i].1.translation;
                        for k in 0..3 {
                            t[k] = (t[k] + pair.1.translation[k]) / 2.0;
                        }
                    }
                }
                MateKind::Symmetric { .. } => {
                    aux.plane = Some(sys.connector(extra[0].0, to_frame(&extra[0].1)).expect("a body"));
                }
                MateKind::Cam { profile } => {
                    let (host, f) = m.a.as_ref().unwrap();
                    aux.cam = Some(self.cam_stations(*host, f, &profile.profile));
                }
                MateKind::Path { path } => {
                    let (host, f) = m.a.as_ref().unwrap();
                    aux.path = Some(self.published_path(*host, f, path, "a path mate follows a published path"));
                }
                _ => {}
            }
            let ends: Vec<rc::EntityId> = sides.iter().map(|(e, local)| sys.connector(*e, to_frame(local)).expect("a body")).collect();
            let (a, b) = (ends[0], ends[1]);
            let built = match problem {
                Some(p) => Err(p),
                None => self.mate_constraints(&m.def.kind, a, b, &m.env, aux),
            };
            match built {
                Ok(list) => {
                    for c in list {
                        match sys.constrain(c) {
                            Ok(id) => {
                                owner_of.insert(id, mi);
                            }
                            Err(e) => {
                                let file = self.asm[&m.owner].doc.path.clone();
                                self.diag(
                                    DiagCode::AsmMateConflict,
                                    Severity::Error,
                                    Site::with_param(
                                        Subject::Mate {
                                            name: m.def.name.clone(),
                                        },
                                        file.as_str(),
                                    ),
                                    format!("mate `{}` cannot be built: {}", m.def.name, e),
                                    Vec::new(),
                                );
                                self.mates[m.outcome].status = MateStatus::Failed {
                                    message: e.to_string(),
                                };
                            }
                        }
                    }
                }
                Err(msg) => {
                    let file = self.asm[&m.owner].doc.path.clone();
                    self.diag(
                        DiagCode::ValueFailed,
                        Severity::Error,
                        Site::with_param(
                            Subject::Mate {
                                name: m.def.name.clone(),
                            },
                            file.as_str(),
                        ),
                        format!("mate `{}`: {}", m.def.name, msg),
                        Vec::new(),
                    );
                    self.mates[m.outcome].status = MateStatus::Failed { message: msg };
                }
            }
        }
        let watch = Stopwatch::start(self.meter);
        let report = rc::solve(&sys, &self.req.solve_options);
        self.meter.record(&UsageEvent::MateSolve {
            assembly: self
                .asm
                .get(&root)
                .map(|i| i.doc.path.clone())
                .unwrap_or_else(|| self.nodes[root as usize].definition.file.clone()),
            scope: self.display_path(root),
            pass: if hold.is_empty() { 1 } else { 2 },
            bodies: members.len() as u32,
            grounded: grounded.len() as u32,
            mates: mates.len() as u32,
            variables: report.values.len() as u32,
            iterations: report.iterations as u32,
            converged: report.converged,
            dof: report.dof as u32,
            duration_nanos: watch.elapsed(self.meter),
        });

        // Write poses back, parents first (members are top-down).
        let mut new_pose: BTreeMap<NodeId, Transform> = BTreeMap::new();
        for &m in members {
            let e = body[&m];
            let p = if grounded.contains(&m) {
                scope_pose[&m]
            } else {
                let vars = sys.entity_vars(e);
                let v: Vec<f64> = vars.iter().map(|v| report.value(*v)).collect();
                Transform::from_quaternion([v[3], v[4], v[5], v[6]], [v[0], v[1], v[2]])
            };
            new_pose.insert(m, p);
            let parent = self.nodes[m as usize].parent.unwrap_or(root);
            let parent_pose = if parent == root {
                Transform::IDENTITY
            } else {
                new_pose
                    .get(&parent)
                    .copied()
                    .unwrap_or_else(|| self.chain(root, parent))
            };
            let dof = report.entity_dof(e).map(|d| d as u32);
            let n = &mut self.nodes[m as usize];
            if !grounded.contains(&m) {
                n.local = parent_pose.inverse().compose(&p);
                n.dof = dof;
                n.pose = if !report.converged {
                    PoseStatus::Failed
                } else if dof == Some(0) {
                    PoseStatus::Solved
                } else {
                    PoseStatus::Floating
                };
            } else if !matches!(
                n.origin,
                NodeOrigin::Pattern { .. } | NodeOrigin::Mirror { .. }
            ) {
                n.dof = Some(0);
                if frames.contains(&m) {
                    n.pose = PoseStatus::Placed;
                }
            }
        }

        // Mate outcomes.
        let name = |r: &Self, c: &rc::ConstraintId| -> String {
            owner_of
                .get(c)
                .map(|i| mates[*i].def.name.clone())
                .unwrap_or_else(|| {
                    let _ = r;
                    if internal.contains(c) {
                        "(fixed in a flexible sub-assembly)".into()
                    } else {
                        "(internal)".into()
                    }
                })
        };
        let mut status: BTreeMap<usize, MateStatus> = BTreeMap::new();
        for d in &report.conflicting {
            if let Some(&mi) = owner_of.get(&d.constraint) {
                let with: Vec<String> = d.depends_on.iter().map(|c| name(self, c)).collect();
                status.insert(mi, MateStatus::Conflicting { with });
            }
        }
        for d in &report.redundant {
            if let Some(&mi) = owner_of.get(&d.constraint) {
                let with: Vec<String> = d.depends_on.iter().map(|c| name(self, c)).collect();
                status.entry(mi).or_insert(MateStatus::Redundant { with });
            }
        }
        for (c, bound) in &report.active_limits {
            if let Some(&mi) = owner_of.get(c) {
                self.mates[mates[mi].outcome].active_limits.push(*bound);
            }
        }
        for (mi, m) in mates.iter().enumerate() {
            if matches!(self.mates[m.outcome].status, MateStatus::Failed { .. }) {
                continue;
            }
            let file = self.asm[&m.owner].doc.path.clone();
            let site = Site::with_param(
                Subject::Mate {
                    name: m.def.name.clone(),
                },
                file.as_str(),
            );
            let st = match status.remove(&mi) {
                Some(s) => s,
                None if !report.converged => MateStatus::NotConverged,
                None => MateStatus::Satisfied,
            };
            match &st {
                MateStatus::Conflicting { with } => {
                    self.diag(
                        DiagCode::AsmMateConflict,
                        Severity::Error,
                        site,
                        format!("mate `{}` contradicts {}", m.def.name, with.join(", ")),
                        vec![Repair::SuppressMate {
                            assembly: file.clone(),
                            mate: m.def.name.clone(),
                        }],
                    );
                }
                MateStatus::Redundant { with } => {
                    self.diag(
                        DiagCode::AsmMateRedundant,
                        Severity::Warning,
                        site,
                        format!(
                            "mate `{}` is implied by {}",
                            m.def.name,
                            if with.is_empty() {
                                "the fixed occurrences".to_string()
                            } else {
                                with.join(", ")
                            }
                        ),
                        vec![Repair::SuppressMate {
                            assembly: file.clone(),
                            mate: m.def.name.clone(),
                        }],
                    );
                }
                MateStatus::NotConverged => {
                    self.diag(
                        DiagCode::AsmMateNotConverged,
                        Severity::Error,
                        site,
                        format!(
                            "the mates around `{}` did not converge (residual {:.3e})",
                            m.def.name, report.residual_norm
                        ),
                        Vec::new(),
                    );
                }
                _ => {}
            }
            if !self.mates[m.outcome].active_limits.is_empty() {
                let file = self.asm[&m.owner].doc.path.clone();
                self.diag(
                    DiagCode::AsmLimitActive,
                    Severity::Info,
                    Site::with_param(
                        Subject::Mate {
                            name: m.def.name.clone(),
                        },
                        file.as_str(),
                    ),
                    format!(
                        "mate `{}` rests on its limit ({:?})",
                        m.def.name, self.mates[m.outcome].active_limits
                    ),
                    Vec::new(),
                );
            }
            self.mates[m.outcome].status = st;
        }
        // A fixed member of a floating flexible sub-assembly rides on it (it is fastened to the
        // sub-assembly's frame), so only the sub-assembly is named.
        let rides = |r: &Self, m: NodeId| {
            let n = &r.nodes[m as usize];
            n.fixed
                && n.parent
                    .map(|p| p != root && member_set.contains(&p))
                    .unwrap_or(false)
        };
        let floating: Vec<String> = members
            .iter()
            .filter(|m| self.nodes[**m as usize].pose == PoseStatus::Floating && !rides(self, **m))
            .map(|m| {
                format!(
                    "{} ({} DOF)",
                    self.display_path(*m),
                    self.nodes[*m as usize].dof.unwrap_or(6)
                )
            })
            .collect();
        if !floating.is_empty() {
            let site = self.occ_site(root);
            let shown: Vec<String> = floating.iter().take(10).cloned().collect();
            self.diag(
                DiagCode::AsmUnderDefined,
                Severity::Info,
                site,
                format!(
                    "{} occurrence(s) still move: {}{}",
                    floating.len(),
                    shown.join(", "),
                    if floating.len() > 10 { ", …" } else { "" }
                ),
                Vec::new(),
            );
        }
        touched
    }

    /// `aux`: what a cam, path or symmetric mate needs besides its two connectors.
    fn mate_constraints(
        &self,
        kind: &MateKind,
        a: rc::EntityId,
        b: rc::EntityId,
        env: &ValueEnv,
        aux: Aux,
    ) -> Result<Vec<rc::Constraint>, String> {
        let target = |t: &TargetExpr, angle: bool| -> Result<rc::Target, String> {
            let v = |e| {
                if angle {
                    self.angle(e, env)
                } else {
                    self.length(e, env)
                }
            };
            Ok(match t {
                TargetExpr::Exact { value } => rc::Target::Exact(v(value)?),
                TargetExpr::Range { min, max } => {
                    let (lo, hi) = (v(min)?, v(max)?);
                    if lo > hi {
                        return Err(format!("the limit range {}..{} is empty", lo, hi));
                    }
                    rc::Target::Range { min: lo, max: hi }
                }
            })
        };
        let m = |x: rc::Mate| rc::Constraint::Mate(x);
        let angle = |t: &TargetExpr| -> Result<rc::Constraint, String> {
            Ok(m(rc::Mate::Angle {
                a,
                b,
                mode: rc::AngleMode::AboutZ,
                target: target(t, true)?,
            }))
        };
        let slide = |t: &TargetExpr| -> Result<rc::Constraint, String> {
            Ok(m(rc::Mate::Distance {
                a,
                b,
                mode: rc::DistanceMode::PlanePoint,
                target: target(t, false)?,
            }))
        };
        Ok(match kind {
            MateKind::Fasten => vec![m(rc::Mate::Fasten { a, b })],
            MateKind::Revolute { angle: t } => {
                let mut v = vec![m(rc::Mate::Revolute { a, b })];
                if let Some(t) = t {
                    v.push(angle(t)?);
                }
                v
            }
            MateKind::Slider { distance } => {
                let mut v = vec![m(rc::Mate::Slider { a, b })];
                if let Some(t) = distance {
                    v.push(slide(t)?);
                }
                v
            }
            MateKind::Cylindrical {
                angle: at,
                distance,
            } => {
                let mut v = vec![m(rc::Mate::Cylindrical { a, b })];
                if let Some(t) = at {
                    v.push(angle(t)?);
                }
                if let Some(t) = distance {
                    v.push(slide(t)?);
                }
                v
            }
            MateKind::Planar => vec![m(rc::Mate::Planar { a, b })],
            MateKind::Ball => vec![m(rc::Mate::Ball { a, b })],
            MateKind::Parallel => vec![m(rc::Mate::Parallel { a, b })],
            MateKind::Tangent { kind } => {
                let k = match kind {
                    TangentSpec::PlaneSphere { radius } => rc::TangentKind::PlaneSphere {
                        radius: self.length(radius, env)?,
                    },
                    TangentSpec::PlaneCylinder { radius } => rc::TangentKind::PlaneCylinder {
                        radius: self.length(radius, env)?,
                    },
                };
                vec![m(rc::Mate::Tangent { a, b, kind: k })]
            }
            MateKind::Distance { measure, target: t } => vec![m(rc::Mate::Distance {
                a,
                b,
                mode: match measure {
                    DistanceMeasure::PointPoint => rc::DistanceMode::PointPoint,
                    DistanceMeasure::PlanePoint => rc::DistanceMode::PlanePoint,
                },
                target: target(t, false)?,
            })],
            MateKind::PinSlot => vec![m(rc::Mate::PinSlot { a, b })],
            MateKind::Screw { lead } => vec![m(rc::Mate::Screw {
                a,
                b,
                lead: self.length(lead, env)?,
            })],
            MateKind::Gear { ratio } => vec![m(rc::Mate::Gear {
                a,
                b,
                ratio: self.ratio(ratio, env)?,
            })],
            MateKind::RackPinion { travel } => vec![m(rc::Mate::RackPinion {
                a,
                b,
                travel: self.length(travel, env)?,
            })],
            MateKind::Cam { profile } => {
                let radii = aux.cam.unwrap_or_else(|| Err(format!("the cam publishes no `{}`", profile.profile)))?;
                vec![m(rc::Mate::Cam { a, b, radii: radii.into() })]
            }
            // The mid-planes coincide: `a` and `b` are already moved half way to `a2` and `b2`.
            MateKind::Width { .. } => vec![m(rc::Mate::Planar { a, b })],
            MateKind::Symmetric { .. } => vec![m(rc::Mate::Symmetric {
                plane: aux.plane.expect("the plane's connector"),
                a,
                b,
            })],
            MateKind::Path { path } => {
                let points = aux.path.unwrap_or_else(|| Err(format!("no `{path}` path")))?;
                if points.len() < 2 {
                    return Err(format!("`{path}` has fewer than two points"));
                }
                vec![m(rc::Mate::Path { a, b, points: points.into() })]
            }
            MateKind::LinearCoupler { ratio } => vec![m(rc::Mate::LinearCoupler {
                a,
                b,
                ratio: self.ratio(ratio, env)?,
            })],
            MateKind::Hinge { angle: t } => {
                let mut v = vec![
                    m(rc::Mate::Cylindrical { a, b }),
                    m(rc::Mate::Distance {
                        a,
                        b,
                        mode: rc::DistanceMode::PlanePoint,
                        target: rc::Target::Exact(0.0),
                    }),
                ];
                if let Some(t) = t {
                    v.push(angle(t)?);
                }
                v
            }
            MateKind::UniversalJoint { joint_point } => vec![m(rc::Mate::UniversalJoint {
                a,
                b,
                joint_point: *joint_point,
            })],
            MateKind::Angle { measure, target: t } => vec![m(rc::Mate::Angle {
                a,
                b,
                mode: match measure {
                    AngleMeasure::AboutZ => rc::AngleMode::AboutZ,
                    AngleMeasure::BetweenZ => rc::AngleMode::BetweenZ,
                },
                target: target(t, true)?,
            })],
        })
    }
}
