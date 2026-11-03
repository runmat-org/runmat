//! Placement: anchors, connectors and frames; placements in dependency order; pattern and
//! mirror instances; fastener-rule expansion against the placed structure.

use crate::assembly::fasteners::{
    expand, load_table, ExpansionInputs, FastenerLayout, FastenerProblem, StackMember,
};
use crate::assembly::model::*;
use crate::assembly::resolve::{InstancePlan, NodeId, OwnedFasteners, PoseStatus, Resolver};
use crate::assembly::source::{PathChain, PublishedGeometry};
use crate::diagnostics::{DiagCode, Repair, Severity, Site, Subject};
use crate::refs::XrefContext;
use crate::regen::kernel::Transform;
use crate::value::ValueEnv;

pub(super) fn flip() -> Transform {
    Transform::rotation_about([0.0; 3], [1.0, 0.0, 0.0], std::f64::consts::PI)
}

impl<'a, 'b> Resolver<'a, 'b> {
    /// A frame in `host`'s definition coordinates. Explicit offsets are evaluated in `env` (the
    /// environment of the document that wrote them).
    pub(super) fn anchor_frame(
        &mut self,
        host: NodeId,
        anchor: &AnchorRef,
        env: &ValueEnv,
    ) -> Result<Transform, String> {
        match anchor {
            AnchorRef::Origin => Ok(Transform::IDENTITY),
            AnchorRef::Explicit { offset } => self.offset(offset, env),
            AnchorRef::Published { name } => {
                let n = self.nodes[host as usize].clone();
                if n.definition.kind == DefinitionKind::Assembly {
                    let Some(info) = self.asm.get(&host) else {
                        return Err(format!(
                            "`{}` is collapsed; its published `{}` is not available",
                            n.definition, name
                        ));
                    };
                    let doc = info.doc.clone();
                    let Some(p) = doc.publishes.iter().find(|p| &p.name == name) else {
                        return Err(format!("`{}` publishes no `{}`", n.definition, name));
                    };
                    return match &p.target {
                        AssemblyPublishTarget::Connector { connector } => {
                            let (inner, f) = self.connector(host, connector)?;
                            Ok(self.chain(host, inner).compose(&f))
                        }
                        AssemblyPublishTarget::Occurrence { path } => {
                            let inner = self.walk(host, path).map_err(|_| {
                                format!(
                                    "`{}` publishes `{}` at `{}`, which is absent",
                                    n.definition, name, path
                                )
                            })?;
                            Ok(self.chain(host, inner))
                        }
                        AssemblyPublishTarget::Envelope { .. } => Err(format!(
                            "`{}` in `{}` is an envelope, not a frame",
                            name, n.definition
                        )),
                    };
                }
                let s = self
                    .configured_summary(&n.definition, n.revision.as_ref(), &n.configuration)
                    .map_err(|e| e.to_string())?;
                match s.publishes.get(name).and_then(|g| g.frame()) {
                    Some(f) => Ok(Transform::from_frame(&f)),
                    None if s.publishes.contains_key(name) => Err(format!(
                        "`{}` in `{}` is not a frame, plane, axis or point",
                        name, n.definition
                    )),
                    None => Err(format!(
                        "`{}` (configuration {}) publishes no `{}`",
                        n.definition,
                        n.configuration.label(),
                        name
                    )),
                }
            }
        }
    }

    /// A mate connector declared by the assembly at `owner`: its host node and its frame in the
    /// host's coordinates.
    pub(super) fn connector(
        &mut self,
        owner: NodeId,
        name: &str,
    ) -> Result<(NodeId, Transform), String> {
        let info = self.asm.get(&owner).ok_or("the assembly is not expanded")?;
        let (doc, env) = (info.doc.clone(), info.env.clone());
        let c = doc
            .connector(name)
            .ok_or_else(|| format!("`{}` has no connector `{}`", doc.path, name))?
            .clone();
        let host = self.walk(owner, &c.occurrence).map_err(|i| {
            format!(
                "connector `{}` is on `{}`, which is not active in this configuration",
                name,
                RelPath(c.occurrence.0[..=i].to_vec())
            )
        })?;
        let mut f = if host == owner {
            match &c.anchor {
                AnchorRef::Origin => Transform::IDENTITY,
                AnchorRef::Explicit { offset } => self.offset(offset, &env)?,
                AnchorRef::Published { name: p } => {
                    // The assembly's own published frame (a sub-assembly connector of itself).
                    let Some(entry) = doc.publishes.iter().find(|e| &e.name == p) else {
                        return Err(format!("`{}` publishes no `{}`", doc.path, p));
                    };
                    match &entry.target {
                        AssemblyPublishTarget::Connector { connector } if connector != name => {
                            let (inner, f) = self.connector(owner, connector)?;
                            self.chain(owner, inner).compose(&f)
                        }
                        _ => return Err(format!("connector `{}` refers to itself", name)),
                    }
                }
            }
        } else {
            self.anchor_frame(host, &c.anchor, &env)?
        };
        if let Some(o) = &c.offset {
            f = f.compose(&self.offset(o, &env)?);
        }
        if c.flip {
            f = f.compose(&flip());
        }
        Ok((host, f))
    }

    /// A frame in `node`'s coordinates.
    pub(super) fn frame_ref(
        &mut self,
        node: NodeId,
        f: &FrameRef,
        env: &ValueEnv,
    ) -> Result<Transform, String> {
        match f {
            FrameRef::Origin => Ok(Transform::IDENTITY),
            FrameRef::Explicit { offset } => self.offset(offset, env),
            FrameRef::Connector { connector } => {
                let (host, t) = self.connector(node, connector)?;
                Ok(self.chain(node, host).compose(&t))
            }
            FrameRef::Interface {
                file,
                name,
                context,
            } => {
                let revision = match context {
                    XrefContext::Pinned { revision } => Some(revision.clone()),
                    XrefContext::Live => None,
                };
                let def = DefinitionRef::part(file.as_str(), None);
                let s = self
                    .configured_summary(&def, revision.as_ref(), &Default::default())
                    .map_err(|e| e.to_string())?;
                s.publishes
                    .get(name)
                    .and_then(PublishedGeometry::frame)
                    .map(|f| Transform::from_frame(&f))
                    .ok_or_else(|| format!("`{}` publishes no frame `{}`", file, name))
            }
        }
    }

    /// Place the authored children of `node` in dependency order.
    pub(super) fn place_children(&mut self, node: NodeId, doc: &AssemblyDocument, env: &ValueEnv) {
        let children: Vec<NodeId> = self.nodes[node as usize]
            .children
            .iter()
            .copied()
            .filter(|c| {
                matches!(
                    self.nodes[*c as usize].origin,
                    crate::assembly::resolve::NodeOrigin::Authored
                )
            })
            .collect();
        let mut placed = vec![false; children.len()];
        let index_of = |id: NodeId| children.iter().position(|c| *c == id);
        // The sibling an attachment depends on, by placement.
        let mut deps: Vec<Option<NodeId>> = Vec::with_capacity(children.len());
        let mut placements: Vec<Placement> = Vec::with_capacity(children.len());
        for &c in &children {
            let name = self.nodes[c as usize].segment.name.clone();
            let p = self
                .position_overrides
                .get(&c)
                .cloned()
                .or_else(|| self.active_placement(doc, &name, c))
                .unwrap_or_default();
            let dep = match &p {
                Placement::Attach {
                    to: FrameRef::Connector { connector },
                    ..
                } => doc
                    .connector(connector)
                    .and_then(|k| k.occurrence.0.first().cloned())
                    .and_then(|s| self.child(node, &s)),
                _ => None,
            };
            deps.push(dep);
            placements.push(p);
        }
        for _round in 0..=children.len() {
            let mut progress = false;
            for i in 0..children.len() {
                if placed[i] {
                    continue;
                }
                if let Some(d) = deps[i] {
                    match index_of(d) {
                        Some(j) if j == i => {}
                        Some(j) if !placed[j] => continue,
                        _ => {}
                    }
                }
                let c = children[i];
                let self_dep = deps[i] == Some(c);
                let result = if self_dep {
                    Err("it is attached to one of its own connectors".to_string())
                } else {
                    self.placement(node, c, &placements[i], env)
                };
                match result {
                    Ok(t) => {
                        let n = &mut self.nodes[c as usize];
                        n.local = t;
                        n.pose = if n.fixed {
                            PoseStatus::Placed
                        } else {
                            PoseStatus::Floating
                        };
                    }
                    Err(msg) => {
                        let site = self.occ_site(c);
                        let path = self.display_path(c);
                        self.diag(
                            DiagCode::AsmUnknownConnector,
                            Severity::Error,
                            site,
                            format!("`{}` cannot be placed: {}", path, msg),
                            Vec::new(),
                        );
                        self.nodes[c as usize].pose = PoseStatus::Failed;
                    }
                }
                placed[i] = true;
                progress = true;
            }
            if !progress {
                break;
            }
        }
        for (i, &c) in children.iter().enumerate() {
            if !placed[i] {
                let site = self.occ_site(c);
                let path = self.display_path(c);
                self.diag(
                    DiagCode::AsmCycle,
                    Severity::Error,
                    site,
                    format!("`{}` is attached through a cycle of connectors", path),
                    Vec::new(),
                );
                self.nodes[c as usize].pose = PoseStatus::Failed;
            }
        }
    }

    fn active_placement(
        &mut self,
        doc: &AssemblyDocument,
        name: &str,
        _c: NodeId,
    ) -> Option<Placement> {
        let file = doc.path.clone();
        let alts: Vec<Occurrence> = doc.alternatives(name).cloned().collect();
        alts.into_iter()
            .find(|o| {
                let site = Site::with_param(
                    Subject::Occurrence {
                        path: name.to_owned(),
                    },
                    file.as_str(),
                );
                self.holds(o.when.as_ref(), &file, &site)
            })
            .map(|o| o.placement)
    }

    fn placement(
        &mut self,
        node: NodeId,
        c: NodeId,
        p: &Placement,
        env: &ValueEnv,
    ) -> Result<Transform, String> {
        match p {
            Placement::Origin => Ok(Transform::IDENTITY),
            Placement::Offset { offset } => self.offset(offset, env),
            Placement::Attach {
                own,
                to,
                offset,
                flip: f,
            } => {
                let target = self.frame_ref(node, to, env)?;
                let own = self.anchor_frame(c, own, env)?;
                let mut t = target;
                if let Some(o) = offset {
                    t = t.compose(&self.offset(o, env)?);
                }
                if *f {
                    t = t.compose(&flip());
                }
                Ok(t.compose(&own.inverse()))
            }
        }
    }

    /// Place pattern and mirror instances from their seeds' current placements.
    pub(super) fn apply_instances(&mut self, plans: &[InstancePlan]) {
        for p in plans {
            let left = match p.host {
                Some(h) => {
                    let hc = self.chain(p.owner, h);
                    hc.compose(&p.left).compose(&hc.inverse())
                }
                None => p.left,
            };
            let seed = &self.nodes[p.seed as usize];
            let (sl, pose, dof) = (seed.local, seed.pose, seed.dof);
            let n = &mut self.nodes[p.instance as usize];
            n.local = left.compose(&sl).compose(&p.right);
            n.pose = pose;
            n.dof = dof;
            n.fixed = true;
        }
    }

    /// Expand the fastener rules of the assembly at `node`.
    pub(super) fn expand_fasteners(
        &mut self,
        node: NodeId,
        doc: &AssemblyDocument,
        env: &ValueEnv,
    ) {
        let file = doc.path.clone();
        for set in &doc.fasteners {
            let site = Site::with_param(
                Subject::FastenerSet {
                    name: set.name.clone(),
                },
                file.as_str(),
            );
            if !self.holds(set.when.as_ref(), &file, &site) {
                continue;
            }
            let mut fatal = None;
            let mut chains: Option<Vec<PathChain>> = None;
            let mut table = None;
            match &set.layout {
                FastenerLayout::Along(rule) => match self.walk(node, &rule.occurrence) {
                    Err(_) => {
                        fatal = Some((
                            DiagCode::FastenerPathMissing,
                            format!("`{}` is not active here", rule.occurrence),
                        ))
                    }
                    Ok(host) => {
                        let n = self.nodes[host as usize].clone();
                        match self.configured_summary(
                            &n.definition,
                            n.revision.as_ref(),
                            &n.configuration,
                        ) {
                            Err(e) => fatal = Some((DiagCode::FastenerPathMissing, e.to_string())),
                            Ok(s) => match s.publishes.get(&rule.path) {
                                Some(PublishedGeometry::Path { chains: cs }) => {
                                    let h = self.chain(node, host);
                                    chains = Some(
                                        cs.iter()
                                            .map(|c| PathChain {
                                                key: c.key.clone(),
                                                points: c
                                                    .points
                                                    .iter()
                                                    .map(|p| h.apply(*p))
                                                    .collect(),
                                                normals: c
                                                    .normals
                                                    .iter()
                                                    .map(|v| h.apply_vector(*v))
                                                    .collect(),
                                            })
                                            .collect(),
                                    );
                                }
                                _ => {
                                    fatal = Some((
                                        DiagCode::FastenerPathMissing,
                                        format!(
                                            "`{}` publishes no path `{}`",
                                            n.definition, rule.path
                                        ),
                                    ))
                                }
                            },
                        }
                    }
                },
                FastenerLayout::Table(rule) => match self.ctx.tables {
                    None => {
                        fatal = Some((
                            DiagCode::FastenerTableIntegrity,
                            "no table source is configured".into(),
                        ))
                    }
                    Some(ts) => match load_table(ts, &rule.table) {
                        Err(p) => fatal = Some((DiagCode::FastenerTableIntegrity, p.to_string())),
                        Ok(t) => {
                            let frame = if rule.frame.is_empty() {
                                Ok(Transform::IDENTITY)
                            } else {
                                self.walk(node, &rule.frame).map(|f| self.chain(node, f))
                            };
                            match frame {
                                Ok(f) => table = Some((t, f)),
                                Err(_) => {
                                    fatal = Some((
                                        DiagCode::FastenerTableSchema,
                                        format!("frame occurrence `{}` is absent", rule.frame),
                                    ))
                                }
                            }
                        }
                    },
                },
            }
            if let Some((code, msg)) = fatal {
                self.diag(
                    code,
                    Severity::Error,
                    site,
                    format!("fasteners `{}`: {}", set.name, msg),
                    Vec::new(),
                );
                continue;
            }
            let mut stack = Vec::new();
            for m in &set.stack {
                match self.walk(node, m) {
                    Ok(id) => {
                        let n = self.nodes[id as usize].clone();
                        let summary = self
                            .configured_summary(
                                &n.definition,
                                n.revision.as_ref(),
                                &n.configuration,
                            )
                            .ok();
                        stack.push(StackMember {
                            path: m.clone(),
                            definition: n.definition.clone(),
                            transform: self.chain(node, id),
                            summary,
                        });
                    }
                    Err(_) => {
                        self.diag(
                            DiagCode::AsmUnknownOccurrence,
                            Severity::Error,
                            site.clone(),
                            format!(
                                "fasteners `{}` clamp `{}`, which is not active here",
                                set.name, m
                            ),
                            Vec::new(),
                        );
                    }
                }
            }
            let ev = self.ev();
            let inputs = ExpansionInputs {
                ev: &ev,
                env,
                units: self.ctx.units,
                path: chains.as_deref(),
                table: table.as_ref().map(|(t, f)| (t.as_ref(), *f)),
                stack: &stack,
                catalog: self.ctx.catalog,
                gauge: self.ctx.gauge,
            };
            let (expansion, problems) = expand(set, &inputs);
            // One diagnostic per problem kind and set beyond the first few, so 100k
            // instances without a size do not flood the report.
            let mut per_kind: std::collections::BTreeMap<DiagCode, u32> = Default::default();
            for p in problems {
                let (code, repairs) = match &p {
                    FastenerProblem::PathMissing { .. } => (DiagCode::FastenerPathMissing, vec![]),
                    FastenerProblem::PathTooShort { .. } => {
                        (DiagCode::FastenerPathTooShort, vec![])
                    }
                    FastenerProblem::ExceptionUnmatched { index, .. } => (
                        DiagCode::FastenerExceptionUnmatched,
                        vec![Repair::RemoveException {
                            set: set.name.clone(),
                            index: *index,
                        }],
                    ),
                    FastenerProblem::ExceptionAmbiguous { .. } => {
                        (DiagCode::FastenerExceptionAmbiguous, vec![])
                    }
                    FastenerProblem::DuplicateKey { .. } => {
                        (DiagCode::FastenerDuplicateKey, vec![])
                    }
                    FastenerProblem::NoFit { .. } => (DiagCode::FastenerNoFit, vec![]),
                    FastenerProblem::TableIntegrity { .. } => {
                        (DiagCode::FastenerTableIntegrity, vec![])
                    }
                    FastenerProblem::TableSchema { .. } => (DiagCode::FastenerTableSchema, vec![]),
                    FastenerProblem::GaugeFailed { .. } => (DiagCode::FastenerGaugeFailed, vec![]),
                    FastenerProblem::Value { .. } => (DiagCode::ValueFailed, vec![]),
                };
                let n = per_kind.entry(code).or_insert(0);
                *n += 1;
                if *n <= 5 {
                    self.diag(
                        code,
                        Severity::Error,
                        site.clone(),
                        format!("fasteners `{}`: {}", set.name, p),
                        repairs,
                    );
                }
            }
            for (code, n) in per_kind {
                if n > 5 {
                    self.diag(
                        code,
                        Severity::Error,
                        site.clone(),
                        format!("fasteners `{}`: {} more like the above", set.name, n - 5),
                        Vec::new(),
                    );
                }
            }
            self.fasteners.push(OwnedFasteners {
                owner: node,
                expansion,
            });
        }
    }
}
