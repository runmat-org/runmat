//! The occurrences one exploded step moves: the occurrences it names, then whole patterns
//! (seeds and every instance laid out in this configuration), then whole mirrors (every image,
//! including one per instance of a mirrored pattern).

use super::{push, site, Eval, ExplodeStep};
use crate::assembly::model::{AssemblyDocument, DefinitionKind, PathSegment, RelPath};
use crate::assembly::resolve::{NodeId, NodeOrigin};
use crate::diagnostics::{DiagCode, Severity};
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Group {
    Pattern,
    Mirror,
}

impl Group {
    fn word(self) -> &'static str {
        match self {
            Group::Pattern => "pattern",
            Group::Mirror => "mirror",
        }
    }
}

impl Eval<'_, '_> {
    /// Every occurrence the step moves, in the order it lists them. `None` holds a named
    /// occurrence that is absent here, so translate spacing (`distance + i × spacing`) keeps
    /// the order as written.
    pub(super) fn step_members(&mut self, s: &ExplodeStep) -> Vec<Option<NodeId>> {
        let occ = format!("steps.{}.occurrences", s.id);
        let mut out: Vec<Option<NodeId>> =
            s.occurrences.iter().map(|o| self.node(o, &occ)).collect();
        let pat = format!("steps.{}.patterns", s.id);
        for g in &s.patterns {
            out.extend(self.group(g, Group::Pattern, &pat).into_iter().map(Some));
        }
        let mir = format!("steps.{}.mirrors", s.id);
        for g in &s.mirrors {
            out.extend(self.group(g, Group::Mirror, &mir).into_iter().map(Some));
        }
        out
    }

    /// The members of one pattern or mirror, named by the path of the assembly that declares
    /// it and then its own name.
    fn group(&mut self, path: &RelPath, kind: Group, param: &str) -> Vec<NodeId> {
        let view = self.view.clone();
        let Some((last, owner_path)) = path.0.split_last() else {
            return Vec::new();
        };
        if last.instance.is_some() {
            push(
                &mut self.rep,
                DiagCode::ExplodeUnknownOccurrence,
                Severity::Error,
                site(&view, param),
                format!(
                    "`{}` names an instance; a {} is named by the assembly that declares it and its own name",
                    path,
                    kind.word()
                ),
            );
            return Vec::new();
        }
        let owner = if owner_path.is_empty() {
            self.root
        } else {
            match self.node(&RelPath(owner_path.to_vec()), param) {
                Some(n) => n,
                None => return Vec::new(),
            }
        };
        let tree = self.tree;
        let owner_node = tree.node(owner);
        if owner_node.definition.kind != DefinitionKind::Assembly && owner != 0 {
            push(
                &mut self.rep,
                DiagCode::ExplodeUnknownOccurrence,
                Severity::Error,
                site(&view, param),
                format!(
                    "`{}` is a part; only assemblies declare patterns and mirrors",
                    tree.path(owner)
                ),
            );
            return Vec::new();
        }
        let doc: Option<Arc<AssemblyDocument>> = if owner == self.root {
            Some(Arc::new(self.doc.clone()))
        } else {
            self.ctx.source.and_then(|s| {
                s.assembly(&owner_node.definition.file, owner_node.revision.as_ref())
                    .ok()
            })
        };
        let name = last.name.as_str();
        // The pattern's seeds, from its declaration (an empty list when it is not declared).
        let mut seeds: Vec<String> = Vec::new();
        if let Some(d) = &doc {
            let declared = match kind {
                Group::Pattern => d.patterns.iter().find(|p| p.name == name).map(|p| {
                    seeds = p.seeds.clone();
                }),
                Group::Mirror => d.mirrors.iter().find(|m| m.name == name).map(|_| ()),
            };
            if declared.is_none() {
                let at = if owner_path.is_empty() {
                    d.path.0.clone()
                } else {
                    tree.path(owner)
                };
                push(
                    &mut self.rep,
                    DiagCode::ExplodeUnknownOccurrence,
                    Severity::Error,
                    site(&view, param),
                    format!(
                        "exploded view `{}` names {} `{}`, which `{}` does not declare",
                        view,
                        kind.word(),
                        name,
                        at
                    ),
                );
                return Vec::new();
            }
        }
        let made_by = |c: NodeId| {
            let n = tree.node(c);
            let ours = match (&n.origin, kind) {
                (NodeOrigin::Pattern { pattern }, Group::Pattern) => pattern == name,
                (NodeOrigin::Mirror { mirror }, Group::Mirror) => mirror == name,
                _ => false,
            };
            ours && n.segment.instance.as_ref().is_some_and(|i| i.by == name)
        };
        let made: Vec<NodeId> = owner_node
            .children
            .iter()
            .copied()
            .filter(|c| made_by(*c))
            .collect();
        if kind == Group::Pattern && doc.is_none() {
            // No declaration to read: the seeds are the occurrences the instances repeat.
            for &c in &made {
                let seed = &tree.node(c).segment.name;
                if !seeds.contains(seed) {
                    seeds.push(seed.clone());
                }
            }
        }
        let mut out: Vec<NodeId> = Vec::new();
        for seed in &seeds {
            let seg = PathSegment::named(seed);
            if let Some(c) = owner_node
                .children
                .iter()
                .copied()
                .find(|c| tree.node(*c).segment == seg)
            {
                out.push(c);
            }
        }
        out.extend(made);
        if out.is_empty() {
            push(
                &mut self.rep,
                DiagCode::ExplodeAbsent,
                Severity::Warning,
                site(&view, param),
                format!(
                    "{} `{}` lays out nothing in this configuration; the exploded view skips it",
                    kind.word(),
                    path
                ),
            );
        }
        out
    }
}
