//! The assembly JSON index (server index rows) and summary cards (a compact deterministic
//! digest per assembly, sized for an agent's context: about 0.5–2k tokens).
//!
//! Every list is in a stable order and every float is written as computed (no per-run state),
//! so identical inputs give identical bytes. Fastener instances are summarised per set (count,
//! omitted keys, selected parts, digest); the instance columns stay in the artifact cache.

use crate::assembly::bom::{flat_bom, Bom, BomOptions};
use crate::assembly::model::{DefinitionKind, DefinitionRef, RelPath, Representation};
use crate::assembly::resolve::{
    MateStatus, NodeId, NodeOrigin, PoseStatus, Prototype, ResolveStats, ResolvedAssembly,
};
use crate::assembly::source::DesignSource;
use crate::diagnostics::{Diagnostic, Severity};
use crate::ids::{FilePath, ItemUuid, MaterialRef, RevisionId};
use crate::refs::InstanceKey;
use crate::regen::kernel::Transform;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt::Write as _;

pub const ASSEMBLY_INDEX_FORMAT_VERSION: u32 = 2;

/// How an occurrence is positioned, as the tree prefixes it (SolidWorks' `(f)`, `(-)`, `(+)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DefinitionState {
    /// Positioned by its placement: `(f)`.
    Fixed,
    /// Mates position it fully.
    FullyDefined,
    /// Mates leave it degrees of freedom: `(-)`.
    UnderDefined,
    /// A mate on it contradicts the others: `(+)`.
    OverDefined,
    /// Its placement or mate solve failed; it keeps its last good placement.
    Failed,
}

/// The pattern, mirror or fastener set that made an occurrence, and its key in it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstanceRow {
    pub by: String,
    /// The key as occurrence paths write it (`2`, `@2`, `r1@3`).
    pub key: String,
    /// The key for people: the copy's number (`2`), or its row, point or grid position.
    pub label: String,
}

fn instance_label(key: &InstanceKey) -> String {
    match key {
        InstanceKey::Linear { step }
        | InstanceKey::Circular { step }
        | InstanceKey::Curve { step } => step.to_string(),
        InstanceKey::Grid { first, second } => format!("{first},{second}"),
        InstanceKey::Lattice { i, j } => format!("{i},{j}"),
        InstanceKey::Ring { ring, step } => format!("{ring}.{step}"),
        InstanceKey::Row { key } | InstanceKey::Point { point: key } => key.clone(),
        InstanceKey::Mirror => "mirror".into(),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OccurrenceRow {
    pub uuid: ItemUuid,
    pub path: String,
    /// The occurrence's own name, without its instance: `clip` for `clip@clips[2]`.
    pub name: String,
    /// Set for a copy made by a pattern, mirror or fastener set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance: Option<InstanceRow>,
    /// Absent for the root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<DefinitionState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<ItemUuid>,
    pub definition: DefinitionRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<RevisionId>,
    pub configuration: String,
    /// The configured input values, in name order, so a view can compare occurrences of one
    /// design (the Outline groups thirty ribs and shows what differs between them).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<InputValueRow>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part_number: Option<String>,
    pub origin: NodeOrigin,
    pub world: Transform,
    pub representation: Representation,
    pub pose: PoseStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dof: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<MaterialRef>,
    pub locked: bool,
    pub prototype: u32,
}

/// One configured input of an occurrence: its SI value, its quantity kind when known, and the
/// value as the label reads it (`88 mm`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputValueRow {
    pub name: String,
    pub si: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MateRow {
    pub assembly: String,
    pub name: String,
    pub kind: String,
    pub status: MateStatus,
    #[serde(default)]
    pub active_limits: Vec<f64>,
    /// The occurrences its two connectors sit on (the assembly's own frame is left out).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub occurrences: Vec<ItemUuid>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FastenerSetRow {
    pub assembly: String,
    pub set: String,
    pub count: u64,
    pub omitted: Vec<String>,
    /// Part number → count.
    pub parts: BTreeMap<String, u64>,
    pub without_size: u64,
    pub digest: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssemblyIndex {
    pub format_version: u32,
    pub top: FilePath,
    pub variant: BTreeMap<String, String>,
    pub occurrences: Vec<OccurrenceRow>,
    pub prototypes: Vec<Prototype>,
    pub mates: Vec<MateRow>,
    pub fasteners: Vec<FastenerSetRow>,
    pub suppressed: Vec<String>,
    pub bom: Bom,
    pub diagnostics: Vec<Diagnostic>,
    pub stats: ResolveStats,
    /// The top assembly's saved model views ([`crate::assembly::views`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub views: Vec<crate::assembly::views::ModelView>,
}

/// The child of `from` reached by walking `path` segment by segment.
fn walk(tree: &ResolvedAssembly, from: NodeId, path: &RelPath) -> Option<NodeId> {
    let mut at = from;
    for segment in &path.0 {
        at = *tree
            .node(at)
            .children
            .iter()
            .find(|c| &tree.node(**c).segment == segment)?;
    }
    Some(at)
}

/// The occurrences each mate's connectors sit on, in mate order.
fn mate_occurrences(
    tree: &ResolvedAssembly,
    source: Option<&dyn DesignSource>,
) -> Vec<Vec<NodeId>> {
    tree.mates
        .iter()
        .map(|m| {
            let owner = tree.node(m.owner);
            let Some(doc) = source.and_then(|s| {
                s.assembly(&owner.definition.file, owner.revision.as_ref())
                    .ok()
            }) else {
                return Vec::new();
            };
            let Some(def) = doc.mates.iter().find(|d| d.name == m.name) else {
                return Vec::new();
            };
            let mut hosts = Vec::new();
            for connector in [&def.a, &def.b] {
                let Some(c) = doc.connector(connector) else {
                    continue;
                };
                if c.occurrence.0.is_empty() {
                    continue;
                }
                if let Some(host) = walk(tree, m.owner, &c.occurrence) {
                    if !hosts.contains(&host) {
                        hosts.push(host);
                    }
                }
            }
            hosts
        })
        .collect()
}

pub fn build_assembly_index(
    tree: &ResolvedAssembly,
    source: Option<&dyn DesignSource>,
) -> AssemblyIndex {
    let hosts = mate_occurrences(tree, source);
    let over: std::collections::BTreeSet<NodeId> = tree
        .mates
        .iter()
        .zip(&hosts)
        .filter(|(m, _)| matches!(m.status, MateStatus::Conflicting { .. }))
        .flat_map(|(_, h)| h.iter().copied())
        .collect();
    let state = |i: NodeId| -> Option<DefinitionState> {
        let n = tree.node(i);
        n.parent?;
        Some(if over.contains(&i) {
            DefinitionState::OverDefined
        } else if n.pose == PoseStatus::Failed {
            DefinitionState::Failed
        } else if n.fixed && !n.attached && n.origin == NodeOrigin::Authored {
            // SolidWorks's `(f)`: fixed in space by its own placement. A pattern or mirror
            // instance is driven by its pattern, and an attached occurrence by its frame, so
            // both are fully defined.
            DefinitionState::Fixed
        } else if n.fixed {
            DefinitionState::FullyDefined
        } else if n.pose == PoseStatus::Floating || n.dof.is_some_and(|d| d > 0) {
            DefinitionState::UnderDefined
        } else {
            DefinitionState::FullyDefined
        })
    };
    let occurrences = tree
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| OccurrenceRow {
            uuid: n.uuid,
            path: tree.path(i as u32),
            name: n.segment.name.clone(),
            instance: n.segment.instance.as_ref().map(|r| InstanceRow {
                by: r.by.clone(),
                key: r.key.to_string(),
                label: instance_label(&r.key),
            }),
            state: state(i as NodeId),
            parent: n.parent.map(|p| tree.nodes[p as usize].uuid),
            definition: n.definition.clone(),
            revision: n.revision.clone(),
            configuration: n.configuration.label(),
            inputs: n
                .configuration
                .inputs
                .iter()
                .map(|(k, v)| {
                    let kind = n.configuration.kinds.get(k);
                    InputValueRow {
                        name: k.to_string(),
                        si: *v,
                        kind: kind.map(|q| q.0.clone()),
                        text: crate::assembly::source::display_quantity(kind, *v),
                    }
                })
                .collect(),
            part_number: n.part_number.clone(),
            origin: n.origin.clone(),
            world: n.world,
            representation: n.representation,
            pose: n.pose,
            dof: n.dof,
            material: n.material.clone(),
            locked: n.locked,
            prototype: n.prototype,
        })
        .collect();
    let mates = tree
        .mates
        .iter()
        .zip(&hosts)
        .map(|(m, h)| MateRow {
            assembly: tree.path(m.owner),
            name: m.name.clone(),
            kind: m.kind.clone(),
            status: m.status.clone(),
            active_limits: m.active_limits.clone(),
            occurrences: h.iter().map(|n| tree.node(*n).uuid).collect(),
        })
        .collect();
    let fasteners = tree
        .fasteners
        .iter()
        .map(|f| {
            let e = &f.expansion;
            let mut parts = BTreeMap::new();
            for (p, c) in e.counts() {
                parts.insert(p.part_number.clone(), c);
            }
            FastenerSetRow {
                assembly: tree.path(f.owner),
                set: e.set.clone(),
                count: e.len() as u64,
                omitted: e.omitted.iter().map(|k| k.to_string()).collect(),
                parts,
                without_size: e.selection.iter().filter(|s| **s == u32::MAX).count() as u64,
                digest: e.digest(),
            }
        })
        .collect();
    AssemblyIndex {
        format_version: ASSEMBLY_INDEX_FORMAT_VERSION,
        top: tree.top.clone(),
        variant: tree.variant.clone(),
        occurrences,
        prototypes: tree.prototypes.clone(),
        mates,
        fasteners,
        suppressed: tree.suppressed.clone(),
        bom: flat_bom(tree, source, &BomOptions::default()),
        diagnostics: tree.diagnostics.diagnostics.clone(),
        stats: tree.stats.clone(),
        views: source
            .and_then(|s| s.assembly(&tree.top, None).ok())
            .map(|doc| doc.views.clone())
            .unwrap_or_default(),
    }
}

pub fn assembly_index_json(index: &AssemblyIndex) -> String {
    serde_json::to_string_pretty(index).expect("the index serialises")
}

/// A compact, deterministic digest of one resolved assembly for agents.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SummaryCard {
    pub text: String,
    /// `sha256:` of the text.
    pub digest: String,
    /// About four characters per token.
    pub token_estimate: usize,
}

#[derive(Clone, Debug)]
pub struct CardOptions {
    pub max_children: usize,
    pub max_bom_rows: usize,
    pub max_diagnostics: usize,
}

impl Default for CardOptions {
    fn default() -> Self {
        Self {
            max_children: 24,
            max_bom_rows: 12,
            max_diagnostics: 6,
        }
    }
}

pub fn summary_card(
    tree: &ResolvedAssembly,
    source: Option<&dyn DesignSource>,
    opts: &CardOptions,
) -> SummaryCard {
    let root = tree.node(0);
    let mut t = String::new();
    let _ = writeln!(
        t,
        "assembly {} ({}){}",
        root.segment.name,
        tree.top,
        root.part_number
            .as_ref()
            .map(|p| format!(" pn {}", p))
            .unwrap_or_default()
    );
    if !tree.variant.is_empty() {
        let v: Vec<String> = tree
            .variant
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();
        let _ = writeln!(t, "configuration: {}", v.join(", "));
    }
    let (mut parts, mut asms, mut circuits, mut depth) = (0u64, 0u64, 0u64, 0u32);
    for n in tree.nodes.iter().skip(1) {
        match n.definition.kind {
            DefinitionKind::Part => parts += 1,
            DefinitionKind::Assembly => asms += 1,
            DefinitionKind::Circuit => circuits += 1,
        }
        depth = depth.max(n.depth);
    }
    let _ = writeln!(
        t,
        "structure: {} occurrences ({} parts, {} assemblies, {} circuits), depth {}, {} prototypes, {} suppressed",
        tree.nodes.len().saturating_sub(1),
        parts,
        asms,
        circuits,
        depth,
        tree.prototypes.len(),
        tree.suppressed.len()
    );
    // Direct children grouped by definition.
    let mut groups: BTreeMap<(String, String), (u64, Vec<String>)> = BTreeMap::new();
    for &c in &root.children {
        let n = tree.node(c);
        let e = groups
            .entry((n.definition.to_string(), n.configuration.label()))
            .or_insert((0, Vec::new()));
        e.0 += 1;
        if e.1.len() < 3 {
            e.1.push(n.segment.to_string());
        }
    }
    let _ = writeln!(t, "children ({} direct):", root.children.len());
    for (i, ((def, cfg), (q, names))) in groups.iter().enumerate() {
        if i >= opts.max_children {
            let _ = writeln!(
                t,
                "  … and {} more groups",
                groups.len() - opts.max_children
            );
            break;
        }
        let _ = writeln!(
            t,
            "  {} [{}] ×{}: {}{}",
            def,
            cfg,
            q,
            names.join(", "),
            if *q as usize > names.len() {
                ", …"
            } else {
                ""
            }
        );
    }
    let (mut ok, mut red, mut con, mut fail, mut inactive) = (0, 0, 0, 0, 0);
    for m in &tree.mates {
        match m.status {
            MateStatus::Satisfied => ok += 1,
            MateStatus::Redundant { .. } => red += 1,
            MateStatus::Conflicting { .. } => con += 1,
            MateStatus::NotConverged | MateStatus::Failed { .. } => fail += 1,
            MateStatus::Inactive => inactive += 1,
        }
    }
    let floating = tree
        .nodes
        .iter()
        .filter(|n| n.pose == PoseStatus::Floating)
        .count();
    if !tree.mates.is_empty() || floating > 0 {
        let _ = writeln!(
            t,
            "mates: {} ({} satisfied, {} redundant, {} conflicting, {} failed, {} inactive); {} floating",
            tree.mates.len(),
            ok,
            red,
            con,
            fail,
            inactive,
            floating
        );
    }
    if !tree.fasteners.is_empty() {
        let _ = writeln!(
            t,
            "fasteners: {} sets, {} instances",
            tree.fasteners.len(),
            tree.fastener_count()
        );
    }
    let bom = flat_bom(tree, source, &BomOptions::default());
    let mut rows: Vec<_> = bom.rows.iter().collect();
    rows.sort_by(|a, b| {
        b.quantity
            .cmp(&a.quantity)
            .then_with(|| a.item.cmp(&b.item))
    });
    let _ = writeln!(
        t,
        "bom ({} lines, {} parts, {} fasteners):",
        bom.rows.len(),
        bom.total_parts,
        bom.total_fasteners
    );
    for r in rows.iter().take(opts.max_bom_rows) {
        let _ = writeln!(
            t,
            "  {} ×{} {} [{}]",
            r.item.part_number.as_deref().unwrap_or("(no pn)"),
            r.quantity,
            r.item.definition,
            r.item.configuration
        );
    }
    if rows.len() > opts.max_bom_rows {
        let _ = writeln!(t, "  … and {} more lines", rows.len() - opts.max_bom_rows);
    }
    let errors: Vec<&Diagnostic> = tree.diagnostics.roots().collect();
    let warnings = tree
        .diagnostics
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Warning)
        .count();
    let _ = writeln!(
        t,
        "diagnostics: {} errors, {} warnings",
        errors.len(),
        warnings
    );
    for d in errors.iter().take(opts.max_diagnostics) {
        let _ = writeln!(t, "  {} {}", d.code, d.message);
    }
    let digest = format!("sha256:{}", hex::encode(Sha256::digest(t.as_bytes())));
    SummaryCard {
        token_estimate: t.len().div_ceil(4),
        text: t,
        digest,
    }
}
