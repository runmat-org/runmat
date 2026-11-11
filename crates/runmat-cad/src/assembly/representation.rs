//! Load levels R0–R3 and working sets.
//!
//! | Level | Loads |
//! |---|---|
//! | R0 | structure and summary (this crate's resolved tree, boxes, part numbers) |
//! | R1 | display LOD (tessellation artifacts per prototype) |
//! | R2 | exact B-rep (per prototype, at the occurrence's representation) |
//! | R3 | editable source text |
//!
//! Every operation declares the level it needs ([`Operation::level`]). A **working set** is a
//! saved query: rules that map occurrences to levels. [`plan`] evaluates it over a resolved tree
//! and returns what to load: the files to open as text, the artifacts to fetch (deduplicated by
//! prototype, so 400 identical ribs fetch one B-rep), and a level per occurrence.

use crate::assembly::model::{DefinitionKind, RelPath, Representation};
use crate::assembly::resolve::{NodeId, ResolvedAssembly};
use crate::assembly::source::{Aabb, DesignSource};
use crate::ids::{FilePath, ItemUuid, PartitionName};
use crate::model::items::PropertyValue;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum LoadLevel {
    #[default]
    R0,
    R1,
    R2,
    R3,
}

/// What a caller wants to do with occurrences.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Structure,
    Bom,
    WhereUsed,
    MassProperties,
    ConfigurationVerify,
    Display,
    Interference,
    Clearance,
    Drawing,
    Analysis,
    Edit,
}

impl Operation {
    pub fn level(self) -> LoadLevel {
        match self {
            Operation::Structure
            | Operation::Bom
            | Operation::WhereUsed
            | Operation::MassProperties
            | Operation::ConfigurationVerify => LoadLevel::R0,
            Operation::Display => LoadLevel::R1,
            Operation::Interference
            | Operation::Clearance
            | Operation::Drawing
            | Operation::Analysis => LoadLevel::R2,
            Operation::Edit => LoadLevel::R3,
        }
    }
}

/// A saved query over occurrences.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "query")]
pub enum Query {
    All,
    /// The occurrence at this path and everything under it.
    Under {
        path: RelPath,
    },
    /// A glob over display paths: `*` within a segment, `**` across segments.
    Glob {
        pattern: String,
    },
    /// Definitions whose file starts with this prefix (`wing/`).
    DefinitionPrefix {
        prefix: String,
    },
    Kind {
        kind: DefinitionKind,
    },
    Partition {
        partition: PartitionName,
    },
    /// An occurrence or definition property equal to a value (or present, when `None`).
    Property {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        equals: Option<PropertyValue>,
    },
    PartNumberPrefix {
        prefix: String,
    },
    /// World boxes overlapping this box.
    Near {
        aabb: Aabb,
    },
    MaxDepth {
        depth: u32,
    },
    Uuids {
        uuids: Vec<ItemUuid>,
    },
    And {
        of: Vec<Query>,
    },
    Or {
        of: Vec<Query>,
    },
    Not {
        of: Box<Query>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoadRule {
    pub query: Query,
    pub level: LoadLevel,
    /// The representation to load at R1/R2 (default: the occurrence's).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub representation: Option<Representation>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkingSet {
    pub name: String,
    pub rules: Vec<LoadRule>,
    #[serde(default)]
    pub default: LoadLevel,
}

/// An artifact to fetch: one prototype at one level and representation.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ArtifactRequest {
    pub prototype: u32,
    pub level: LoadLevel,
    pub representation: Representation,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LoadPlan {
    /// Level per node, by node id.
    pub levels: Vec<LoadLevel>,
    pub representations: Vec<Option<Representation>>,
    /// Source files to open as text (R3): definitions and the assemblies that author them.
    pub text: BTreeSet<FilePath>,
    pub artifacts: BTreeSet<ArtifactRequest>,
    pub counts: BTreeMap<LoadLevel, u64>,
}

impl LoadPlan {
    pub fn level(&self, n: NodeId) -> LoadLevel {
        self.levels.get(n as usize).copied().unwrap_or_default()
    }
    pub fn allows(&self, n: NodeId, op: Operation) -> bool {
        self.level(n) >= op.level()
    }
}

/// Glob over `/`-separated segments.
pub fn glob_match(pattern: &str, path: &str) -> bool {
    fn seg(p: &[u8], s: &[u8]) -> bool {
        match (p.first(), s.first()) {
            (None, None) => true,
            (Some(b'*'), _) => seg(&p[1..], s) || (!s.is_empty() && seg(p, &s[1..])),
            (Some(a), Some(b)) if a == b => seg(&p[1..], &s[1..]),
            _ => false,
        }
    }
    fn segs(p: &[&str], s: &[&str]) -> bool {
        match (p.first(), s.first()) {
            (None, None) => true,
            (Some(&"**"), _) => segs(&p[1..], s) || (!s.is_empty() && segs(p, &s[1..])),
            (Some(a), Some(b)) => seg(a.as_bytes(), b.as_bytes()) && segs(&p[1..], &s[1..]),
            _ => false,
        }
    }
    let p: Vec<&str> = pattern.split('/').filter(|x| !x.is_empty()).collect();
    let s: Vec<&str> = path.split('/').filter(|x| !x.is_empty()).collect();
    segs(&p, &s)
}

struct Eval<'a> {
    tree: &'a ResolvedAssembly,
    source: &'a dyn DesignSource,
    paths: Vec<Option<String>>,
    boxes: Vec<Option<Option<Aabb>>>,
}

impl Eval<'_> {
    fn path(&mut self, n: NodeId) -> &str {
        if self.paths[n as usize].is_none() {
            self.paths[n as usize] = Some(self.tree.path(n));
        }
        self.paths[n as usize].as_deref().unwrap()
    }

    fn world_box(&mut self, n: NodeId) -> Option<Aabb> {
        if let Some(b) = self.boxes[n as usize] {
            return b;
        }
        let node = self.tree.node(n);
        let b = if node.definition.kind == DefinitionKind::Assembly && node.expanded {
            let kids = node.children.clone();
            let mut acc = Aabb::EMPTY;
            for c in kids {
                if let Some(cb) = self.world_box(c) {
                    acc = acc.union(&cb);
                }
            }
            (!acc.is_empty()).then_some(acc)
        } else {
            self.source
                .configured(
                    &node.definition,
                    node.revision.as_ref(),
                    &node.configuration,
                )
                .ok()
                .and_then(|s| s.bbox)
                .map(|b| b.transformed(&node.world))
        };
        self.boxes[n as usize] = Some(b);
        b
    }

    fn matches(&mut self, q: &Query, n: NodeId) -> bool {
        let node = self.tree.node(n);
        match q {
            Query::All => true,
            Query::Under { path } => {
                let Some(target) = self.tree.find(path) else {
                    return false;
                };
                let mut cur = Some(n);
                while let Some(c) = cur {
                    if c == target {
                        return true;
                    }
                    cur = self.tree.node(c).parent;
                }
                false
            }
            Query::Glob { pattern } => {
                let p = pattern.clone();
                glob_match(&p, self.path(n))
            }
            Query::DefinitionPrefix { prefix } => {
                node.definition.file.0.starts_with(prefix.as_str())
            }
            Query::Kind { kind } => node.definition.kind == *kind,
            Query::Partition { partition } => {
                self.source.partition(&node.definition.file).as_ref() == Some(partition)
            }
            Query::Property { name, equals } => {
                let own = node
                    .properties
                    .iter()
                    .find(|p| p.name.0 == *name)
                    .map(|p| p.value.clone());
                let found = own.or_else(|| {
                    self.source
                        .definition(&node.definition, node.revision.as_ref())
                        .ok()
                        .and_then(|s| {
                            s.properties
                                .iter()
                                .find(|p| p.name.0 == *name)
                                .map(|p| p.value.clone())
                        })
                });
                match (found, equals) {
                    (Some(_), None) => true,
                    (Some(v), Some(e)) => &v == e,
                    _ => false,
                }
            }
            Query::PartNumberPrefix { prefix } => node
                .part_number
                .as_deref()
                .map(|p| p.starts_with(prefix.as_str()))
                .unwrap_or(false),
            Query::Near { aabb } => self
                .world_box(n)
                .map(|b| b.overlaps(aabb, 0.0))
                .unwrap_or(false),
            Query::MaxDepth { depth } => node.depth <= *depth,
            Query::Uuids { uuids } => uuids.contains(&node.uuid),
            Query::And { of } => of.iter().all(|x| self.matches(x, n)),
            Query::Or { of } => of.iter().any(|x| self.matches(x, n)),
            Query::Not { of } => !self.matches(of, n),
        }
    }
}

/// Evaluate a working set over a resolved tree.
pub fn plan(ws: &WorkingSet, tree: &ResolvedAssembly, source: &dyn DesignSource) -> LoadPlan {
    let n = tree.nodes.len();
    let mut ev = Eval {
        tree,
        source,
        paths: vec![None; n],
        boxes: vec![None; n],
    };
    let mut out = LoadPlan {
        levels: vec![ws.default; n],
        representations: vec![None; n],
        ..Default::default()
    };
    for i in 0..n as NodeId {
        for r in &ws.rules {
            if r.level <= out.levels[i as usize] && r.representation.is_none() {
                continue;
            }
            if ev.matches(&r.query, i) {
                if r.level > out.levels[i as usize] {
                    out.levels[i as usize] = r.level;
                }
                if r.representation.is_some() {
                    out.representations[i as usize] = r.representation;
                }
            }
        }
    }
    for i in 0..n {
        let node = &tree.nodes[i];
        let level = out.levels[i];
        *out.counts.entry(level).or_insert(0) += 1;
        if level >= LoadLevel::R1
            && node.definition.kind != DefinitionKind::Assembly
            && !node.represented_by_ancestor
        {
            let rep = out.representations[i].unwrap_or(if level == LoadLevel::R1 {
                Representation::Display
            } else {
                node.representation
            });
            out.artifacts.insert(ArtifactRequest {
                prototype: node.prototype,
                level: level.min(LoadLevel::R2),
                representation: rep,
            });
        }
        if level == LoadLevel::R3 {
            out.text.insert(node.definition.file.clone());
            if !node.authored_in.0.is_empty() {
                out.text.insert(node.authored_in.clone());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::glob_match;

    #[test]
    fn globs() {
        assert!(glob_match("wing/**/rib*", "wing/box/rib@ribs[3]"));
        assert!(glob_match("**", "a/b/c"));
        assert!(glob_match("a/*/c", "a/b/c"));
        assert!(!glob_match("a/*/c", "a/b/x/c"));
        assert!(glob_match("a/**/c", "a/c"));
    }
}
