//! Bills of materials and where-used.
//!
//! A BOM is computed from a resolved (100%) tree, so it is configuration-aware by
//! construction: resolve at a configuration, then group. Items group by definition, revision,
//! configuration, part number and material, so a dash number or a material override is its own
//! line. A part with a part number groups by that number: occurrences whose parameters differ
//! but keep the number (four arms set to their stations) are one line with their count, as a
//! parts list counts them. Fastener rules contribute counts per selected part, without occurrence nodes.
//!
//! Where-used runs two ways: within a resolved tree (occurrence paths of a definition), and
//! across the project's 150% structure ([`WhereUsedIndex`]: every authored use with its `when`
//! condition, plus fastener families and opposite-hand sources), transitively up to the tops.

use crate::assembly::fasteners::FastenerPart;
use crate::assembly::model::{AssemblyDocument, DefinitionKind, DefinitionRef, MirrorHand};
use crate::assembly::options::Condition;
use crate::assembly::resolve::{NodeId, ResolvedAssembly};
use crate::assembly::source::{DesignSource, OutputSummary};
use crate::ids::{FilePath, MaterialRef, PartName, RevisionId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BomKind {
    Part,
    Assembly,
    Circuit,
    Fastener,
}

/// What makes two BOM lines the same item.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BomItem {
    pub kind: BomKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part_number: Option<String>,
    pub definition: DefinitionRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<RevisionId>,
    /// The configuration label (`default`, `light`, `-08-12` for a dash number).
    pub configuration: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<MaterialRef>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BomRow {
    pub item: BomItem,
    pub quantity: u64,
    /// Up to `BomOptions::paths_per_row` occurrence paths, for where-used and balloons.
    #[serde(default)]
    pub paths: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndentedRow {
    pub level: u32,
    /// The parent assembly's path (empty for the top).
    pub parent: String,
    pub item: BomItem,
    pub quantity: u64,
    /// The first few occurrence paths of the line (as many as a flat row keeps), so a balloon
    /// on a sub-assembly finds its line.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub paths: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct BomOptions {
    pub include_assemblies: bool,
    pub include_fasteners: bool,
    pub paths_per_row: usize,
}

impl Default for BomOptions {
    fn default() -> Self {
        Self {
            include_assemblies: false,
            include_fasteners: true,
            paths_per_row: 8,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Bom {
    pub variant: BTreeMap<String, String>,
    pub rows: Vec<BomRow>,
    pub total_parts: u64,
    pub total_fasteners: u64,
}

fn kind(k: DefinitionKind) -> BomKind {
    match k {
        DefinitionKind::Part => BomKind::Part,
        DefinitionKind::Assembly => BomKind::Assembly,
        DefinitionKind::Circuit => BomKind::Circuit,
    }
}

/// What the BOM reads per prototype: its part number (a configuration's dash number when its
/// summary has one, else the definition's), and for a part file that makes several outputs,
/// those outputs with the definition's own material. One summary fetch per prototype.
struct PrototypeFacts {
    numbers: Vec<Option<String>>,
    outputs: Vec<(Vec<OutputSummary>, Option<MaterialRef>)>,
}

fn prototype_facts(tree: &ResolvedAssembly, source: Option<&dyn DesignSource>) -> PrototypeFacts {
    let mut facts = PrototypeFacts {
        numbers: Vec::new(),
        outputs: Vec::new(),
    };
    for p in &tree.prototypes {
        let part = p.definition.kind == DefinitionKind::Part;
        facts.numbers.push(source.and_then(|s| {
            if p.definition.kind == DefinitionKind::Assembly {
                return None;
            }
            s.configured(&p.definition, p.revision.as_ref(), &p.configuration)
                .ok()
                .and_then(|c| c.part_number.clone())
        }));
        let outputs = source
            .filter(|_| part && p.definition.part.is_none())
            .and_then(|s| s.definition(&p.definition, p.revision.as_ref()).ok())
            .filter(|d| d.outputs.len() > 1)
            .map(|d| (d.outputs.clone(), d.material.clone()))
            .unwrap_or_default();
        facts.outputs.push(outputs);
    }
    facts
}

fn item_of(tree: &ResolvedAssembly, n: NodeId, facts: &PrototypeFacts) -> BomItem {
    let node = tree.node(n);
    BomItem {
        kind: kind(node.definition.kind),
        part_number: facts
            .numbers
            .get(node.prototype as usize)
            .cloned()
            .flatten()
            .or_else(|| node.part_number.clone()),
        definition: node.definition.clone(),
        revision: node.revision.clone(),
        configuration: node.configuration.label(),
        material: node.material.clone(),
    }
}

/// The BOM items one occurrence counts as: its item, or for a part file that makes several
/// outputs one item per output (a display unit's screen and its case), each with its own part
/// number and material. The first output keeps a configuration's dash number, and a material
/// the occurrence sets applies to every output.
fn items_of(tree: &ResolvedAssembly, n: NodeId, facts: &PrototypeFacts) -> Vec<BomItem> {
    let item = item_of(tree, n, facts);
    let node = tree.node(n);
    let Some((outputs, own)) = facts
        .outputs
        .get(node.prototype as usize)
        .filter(|(o, _)| !o.is_empty())
    else {
        return vec![item];
    };
    let overridden = node.material.is_some() && node.material != *own;
    outputs
        .iter()
        .enumerate()
        .map(|(i, o)| BomItem {
            part_number: if i == 0 {
                item.part_number.clone()
            } else {
                o.part_number.clone()
            },
            definition: DefinitionRef {
                part: Some(PartName::new(o.name.as_str())),
                ..item.definition.clone()
            },
            material: if overridden {
                item.material.clone()
            } else {
                o.material.clone()
            },
            ..item.clone()
        })
        .collect()
}

/// The key occurrences of one BOM line share: the item itself, or for a part that carries a
/// part number the item without its configuration label (the first occurrence's label stays on
/// the line).
fn group_key(item: &BomItem) -> BomItem {
    if item.kind != BomKind::Assembly && item.part_number.as_deref().is_some_and(|n| !n.is_empty())
    {
        BomItem {
            configuration: String::new(),
            ..item.clone()
        }
    } else {
        item.clone()
    }
}

fn fastener_item(p: &FastenerPart) -> BomItem {
    BomItem {
        kind: BomKind::Fastener,
        part_number: Some(p.part_number.clone()),
        definition: p.family.clone(),
        revision: None,
        configuration: p.configuration.0.clone(),
        material: None,
    }
}

/// The flat (summarised) BOM: every leaf, optionally every assembly, plus fasteners.
pub fn flat_bom(
    tree: &ResolvedAssembly,
    source: Option<&dyn DesignSource>,
    opts: &BomOptions,
) -> Bom {
    let facts = prototype_facts(tree, source);
    let mut rows: BTreeMap<BomItem, (BomItem, u64, Vec<String>)> = BTreeMap::new();
    let mut parts = 0;
    for (i, n) in tree.nodes.iter().enumerate().skip(1) {
        let is_asm = n.definition.kind == DefinitionKind::Assembly;
        // A collapsed or failed sub-assembly stands for itself.
        let leaf = !is_asm || !n.expanded;
        if !(leaf || opts.include_assemblies) {
            continue;
        }
        for item in items_of(tree, i as NodeId, &facts) {
            let e = rows
                .entry(group_key(&item))
                .or_insert_with(|| (item, 0, Vec::new()));
            e.1 += 1;
            if e.2.len() < opts.paths_per_row {
                e.2.push(tree.path(i as NodeId));
            }
            if !is_asm {
                parts += 1;
            }
        }
    }
    let mut fasteners = 0;
    if opts.include_fasteners {
        for f in &tree.fasteners {
            for (p, c) in f.expansion.counts() {
                let item = fastener_item(p);
                let e = rows
                    .entry(group_key(&item))
                    .or_insert_with(|| (item, 0, Vec::new()));
                e.1 += c;
                fasteners += c;
                if e.2.len() < opts.paths_per_row {
                    let owner = tree.path(f.owner);
                    e.2.push(if owner.is_empty() {
                        format!("fasteners.{}", f.expansion.set)
                    } else {
                        format!("{}/fasteners.{}", owner, f.expansion.set)
                    });
                }
            }
        }
    }
    Bom {
        variant: tree.variant.clone(),
        rows: rows
            .into_iter()
            .map(|(_, (item, quantity, paths))| BomRow {
                item,
                quantity,
                paths,
            })
            .collect(),
        total_parts: parts,
        total_fasteners: fasteners,
    }
}

/// The indented BOM: per assembly occurrence, its direct children grouped by item.
pub fn indented_bom(
    tree: &ResolvedAssembly,
    source: Option<&dyn DesignSource>,
) -> Vec<IndentedRow> {
    let facts = prototype_facts(tree, source);
    let mut out = Vec::new();
    fn walk(
        tree: &ResolvedAssembly,
        n: NodeId,
        facts: &PrototypeFacts,
        out: &mut Vec<IndentedRow>,
    ) {
        let node = tree.node(n);
        let mut groups: BTreeMap<BomItem, (BomItem, u64, Vec<NodeId>)> = BTreeMap::new();
        for &c in &node.children {
            for item in items_of(tree, c, facts) {
                let e = groups
                    .entry(group_key(&item))
                    .or_insert_with(|| (item, 0, Vec::new()));
                e.1 += 1;
                e.2.push(c);
            }
        }
        let mut fasteners: BTreeMap<BomItem, u64> = BTreeMap::new();
        for f in tree.fasteners.iter().filter(|f| f.owner == n) {
            for (p, c) in f.expansion.counts() {
                *fasteners.entry(fastener_item(p)).or_insert(0) += c;
            }
        }
        let parent = tree.path(n);
        for (_, (item, q, members)) in groups {
            let expand = item.kind == BomKind::Assembly;
            let paths = members
                .iter()
                .take(BomOptions::default().paths_per_row)
                .map(|&m| tree.path(m))
                .collect();
            out.push(IndentedRow {
                level: node.depth + 1,
                parent: parent.clone(),
                item,
                quantity: q,
                paths,
            });
            // Every member of the group has the same structure; list the first's children.
            if expand {
                if let Some(&first) = members.first() {
                    if tree.node(first).expanded {
                        walk(tree, first, facts, out);
                    }
                }
            }
        }
        for (item, q) in fasteners {
            out.push(IndentedRow {
                level: node.depth + 1,
                parent: parent.clone(),
                item,
                quantity: q,
                paths: Vec::new(),
            });
        }
    }
    walk(tree, 0, &facts, &mut out);
    out
}

/// Occurrences of a definition in a resolved tree.
pub fn where_used_in(tree: &ResolvedAssembly, definition: &DefinitionRef) -> Vec<NodeId> {
    tree.nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| &n.definition == definition)
        .map(|(i, _)| i as NodeId)
        .collect()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "use")]
pub enum UseKind {
    Occurrence,
    OppositeHand { mirror: String },
    FastenerFamily { set: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WhereUsedEntry {
    pub assembly: FilePath,
    pub name: String,
    pub kind: UseKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<Condition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<RevisionId>,
}

/// Reverse use index over a project's assembly files (150%: every alternative).
#[derive(Clone, Debug, Default)]
pub struct WhereUsedIndex {
    pub uses: BTreeMap<DefinitionRef, Vec<WhereUsedEntry>>,
    /// By file (any part output of it).
    pub by_file: BTreeMap<FilePath, BTreeSet<DefinitionRef>>,
}

impl WhereUsedIndex {
    pub fn build<'a>(docs: impl IntoIterator<Item = &'a Arc<AssemblyDocument>>) -> Self {
        let mut idx = WhereUsedIndex::default();
        let mut add = |d: &DefinitionRef, e: WhereUsedEntry| {
            idx.by_file
                .entry(d.file.clone())
                .or_default()
                .insert(d.clone());
            idx.uses.entry(d.clone()).or_default().push(e);
        };
        for doc in docs {
            for o in &doc.occurrences {
                add(
                    &o.source.definition,
                    WhereUsedEntry {
                        assembly: doc.path.clone(),
                        name: o.name.clone(),
                        kind: UseKind::Occurrence,
                        when: o.when.clone(),
                        revision: match &o.source.context {
                            crate::refs::XrefContext::Pinned { revision } => Some(revision.clone()),
                            _ => None,
                        },
                    },
                );
            }
            for m in &doc.mirrors {
                if let MirrorHand::Opposite { source, .. } = &m.hand {
                    add(
                        &source.definition,
                        WhereUsedEntry {
                            assembly: doc.path.clone(),
                            name: m.name.clone(),
                            kind: UseKind::OppositeHand {
                                mirror: m.name.clone(),
                            },
                            when: m.when.clone(),
                            revision: None,
                        },
                    );
                }
            }
            for f in &doc.fasteners {
                add(
                    &f.fastener.family,
                    WhereUsedEntry {
                        assembly: doc.path.clone(),
                        name: f.name.clone(),
                        kind: UseKind::FastenerFamily {
                            set: f.name.clone(),
                        },
                        when: f.when.clone(),
                        revision: None,
                    },
                );
            }
        }
        for v in idx.uses.values_mut() {
            v.sort_by(|a, b| (&a.assembly, &a.name).cmp(&(&b.assembly, &b.name)));
        }
        idx
    }

    /// Direct uses of a definition.
    pub fn direct(&self, d: &DefinitionRef) -> &[WhereUsedEntry] {
        self.uses.get(d).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Every assembly file that uses `file` directly or through others.
    pub fn transitive(&self, file: &FilePath) -> BTreeSet<FilePath> {
        let mut out = BTreeSet::new();
        let mut queue = vec![file.clone()];
        while let Some(f) = queue.pop() {
            for d in self.by_file.get(&f).into_iter().flatten() {
                for e in self.direct(d) {
                    if out.insert(e.assembly.clone()) {
                        queue.push(e.assembly.clone());
                    }
                }
            }
        }
        out
    }
}
