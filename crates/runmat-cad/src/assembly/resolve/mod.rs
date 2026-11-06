//! Composition: a top assembly plus a configuration → the resolved (100%) occurrence tree.
//!
//! Steps, per assembly level (post-order, so a rigid sub-assembly's layout is known before its
//! parent uses it):
//! 1. **Select.** For each child name, the alternative whose `when` holds at the configuration
//!    point (two at once is `cad.asm.duplicate_name`; the validator proves it cannot happen).
//! 2. **Override.** Typed overrides travel down as pending paths. At each level the pending
//!    entries whose first segment names no child are diagnosed (`override_target_missing`).
//!    When two ancestors set the same slot, the outer one wins and the inner is reported.
//! 3. **Configure.** The configuration selection is evaluated in the parent's value
//!    environment and checked against the definition's typed inputs and configurations.
//! 4. **Place.** Placements in dependency order (attachments to sibling connectors).
//! 5. **Solve.** Mates of this level, plus those of flexible sub-assemblies, through
//!    `runmat-constraint` (two passes when mates touch pattern instances).
//! 6. **Repeat.** Patterns and mirrors place their instances; fastener rules expand.
//!
//! Identity: an authored occurrence's local id is its lock UUID (`<file>#occurrence.<name>`),
//! kept through renames by the lock's aliases. An instance's local id chains its maker's lock
//! id, the seed's and the semantic key. A node's id chains its parent's id and its local id, so
//! ids are deterministic, stable under renames and reorders, and need no lock row per instance.

mod tree;
mod values;
mod compose;
mod instances;

use crate::assembly::fasteners::{FastenerCatalog, FastenerExpansion, StackGauge, TableSource};
use crate::assembly::model::*;
use crate::assembly::options::{
    AxisValue, Condition, OptionProblem, OptionSpace, OptionsBlock, Point,
};
use crate::assembly::source::{
    ConfigKey, ConfiguredSummary, DefinitionSummary, DesignSource, PublishedGeometry, SourceError,
};
use crate::diagnostics::{Cause, DiagCode, DiagnosticReport, Repair, Severity, Site, Subject};
use crate::ids::{ConfigurationName, FilePath, InputName, ItemUuid, MaterialRef, RevisionId};
use crate::metering::{NoUsageMeter, UsageMeter};
use crate::model::items::Property;
use crate::refs::lock::{ItemPath, LockSection};
use crate::refs::{InstanceKey, XrefContext};
use crate::regen::kernel::Transform;
use crate::units::{Dimension, UnitAuthority};
use crate::value::{DynamicValueEvaluator, Evaluator, ValueEnv, ValueExpr};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

pub type NodeId = u32;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "origin")]
pub enum NodeOrigin {
    Root,
    Authored,
    Pattern { pattern: String },
    Mirror { mirror: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoseStatus {
    Root,
    /// Positioned by its placement.
    Placed,
    /// Positioned by mates.
    Solved,
    /// Not fixed and not fully determined by mates.
    Floating,
    /// Its placement or solve failed; it keeps its last good placement.
    Failed,
}

/// One occurrence in the resolved tree.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OccNode {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<NodeId>,
    pub segment: PathSegment,
    pub depth: u32,
    pub uuid: ItemUuid,
    /// The lock held an entry for this occurrence (or its maker).
    pub locked: bool,
    pub definition: DefinitionRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<RevisionId>,
    pub configuration: ConfigKey,
    /// The assembly file that authored this occurrence (empty for the root).
    pub authored_in: FilePath,
    pub origin: NodeOrigin,
    pub local: Transform,
    pub world: Transform,
    /// Positioned by its placement (true) or by mates (false).
    pub fixed: bool,
    /// Its placement attaches it to a frame (a skeleton's interface, another occurrence's
    /// connector): positioned by that reference, as a mate would, rather than fixed in space.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub attached: bool,
    pub flexible: bool,
    pub pose: PoseStatus,
    /// Remaining degrees of freedom after the mate solve (`None`: not in a solve).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dof: Option<u32>,
    pub representation: Representation,
    /// An ancestor's lightweight representation stands in for this occurrence.
    pub represented_by_ancestor: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<MaterialRef>,
    #[serde(default)]
    pub properties: Vec<Property>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part_number: Option<String>,
    pub prototype: u32,
    pub children: Vec<NodeId>,
    /// A sub-assembly whose children were loaded (false: collapsed or failed).
    pub expanded: bool,
    /// The definition failed to load; the node stands as a placeholder.
    pub failed: bool,
}

/// One shared definition at one configuration and representation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Prototype {
    pub definition: DefinitionRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<RevisionId>,
    pub configuration: ConfigKey,
    pub representation: Representation,
    pub instances: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum MateStatus {
    Satisfied,
    Redundant { with: Vec<String> },
    Conflicting { with: Vec<String> },
    NotConverged,
    Inactive,
    Failed { message: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MateOutcome {
    /// The assembly node whose document declares the mate.
    pub owner: NodeId,
    pub name: String,
    pub kind: String,
    pub status: MateStatus,
    /// Limits that ended active, with the bound used.
    #[serde(default)]
    pub active_limits: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OwnedFasteners {
    pub owner: NodeId,
    pub expansion: FastenerExpansion,
}

/// A declared space allocation, placed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResolvedEnvelope {
    /// The assembly node that declares it.
    pub declared_by: NodeId,
    pub name: String,
    /// The occurrence allowed inside (its owner), if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub occupant: Option<NodeId>,
    pub keep_out: bool,
    /// In the declaring assembly's coordinates.
    pub local: crate::assembly::source::Aabb,
    /// In top coordinates.
    pub world: crate::assembly::source::Aabb,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ResolveStats {
    pub nodes: u64,
    pub assemblies: u64,
    pub collapsed: u64,
    pub prototypes: u64,
    pub mate_scopes: u64,
    pub mates: u64,
    pub fastener_sets: u64,
    pub fasteners: u64,
    pub unlocked: u64,
    pub suppressed: u64,
}

/// Which configuration of the product to resolve.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "variant")]
pub enum VariantSelection {
    /// Axis defaults (an error if an axis has none).
    #[default]
    Defaults,
    Named {
        name: String,
    },
    Values {
        values: BTreeMap<String, AxisValue>,
    },
}

#[derive(Clone, Debug)]
pub struct ResolveRequest {
    pub variant: VariantSelection,
    /// A configuration of the top assembly.
    pub configuration: Option<ConfigurationName>,
    pub revision: Option<RevisionId>,
    /// Stop expanding below this depth (the root is depth 0).
    pub max_depth: Option<u32>,
    /// Assemblies to keep collapsed (their summary stands in).
    pub collapse: BTreeSet<FilePath>,
    pub solve_mates: bool,
    pub expand_fasteners: bool,
    pub solve_options: runmat_constraint::SolveOptions,
}

impl Default for ResolveRequest {
    fn default() -> Self {
        Self {
            variant: VariantSelection::Defaults,
            configuration: None,
            revision: None,
            max_depth: None,
            collapse: BTreeSet::new(),
            solve_mates: true,
            expand_fasteners: true,
            solve_options: runmat_constraint::SolveOptions::default(),
        }
    }
}

pub struct ResolveContext<'a> {
    pub source: &'a dyn DesignSource,
    pub units: &'a dyn UnitAuthority,
    pub dynamic: &'a dyn DynamicValueEvaluator,
    pub lock: Option<&'a LockSection>,
    pub catalog: Option<&'a dyn FastenerCatalog>,
    pub tables: Option<&'a dyn TableSource>,
    pub gauge: Option<&'a dyn StackGauge>,
}

#[derive(Clone, Debug)]
pub struct ResolvedAssembly {
    pub top: FilePath,
    /// The configuration point as `axis = value`.
    pub variant: BTreeMap<String, String>,
    pub nodes: Vec<OccNode>,
    pub fasteners: Vec<OwnedFasteners>,
    pub mates: Vec<MateOutcome>,
    pub prototypes: Vec<Prototype>,
    pub envelopes: Vec<ResolvedEnvelope>,
    /// Display paths of occurrences suppressed in this configuration.
    pub suppressed: Vec<String>,
    pub diagnostics: DiagnosticReport,
    pub stats: ResolveStats,
}

impl ResolvedAssembly {
    pub fn root(&self) -> NodeId {
        0
    }

    pub fn node(&self, id: NodeId) -> &OccNode {
        &self.nodes[id as usize]
    }

    /// The display path: `wing.left/box/rib@ribs[3]`.
    pub fn path(&self, id: NodeId) -> String {
        let mut segs = Vec::new();
        let mut cur = Some(id);
        while let Some(c) = cur {
            let n = &self.nodes[c as usize];
            if n.parent.is_some() {
                segs.push(n.segment.to_string());
            }
            cur = n.parent;
        }
        segs.reverse();
        segs.join("/")
    }

    pub fn rel_path(&self, id: NodeId) -> RelPath {
        let mut segs = Vec::new();
        let mut cur = Some(id);
        while let Some(c) = cur {
            let n = &self.nodes[c as usize];
            if n.parent.is_some() {
                segs.push(n.segment.clone());
            }
            cur = n.parent;
        }
        segs.reverse();
        RelPath(segs)
    }

    /// Find a node by path from the root.
    pub fn find(&self, path: &RelPath) -> Option<NodeId> {
        let mut cur = 0;
        for s in &path.0 {
            cur = *self.nodes[cur as usize]
                .children
                .iter()
                .find(|c| &self.nodes[**c as usize].segment == s)?;
        }
        Some(cur)
    }

    /// Find by display path (`a/b@p[3]/c`).
    pub fn find_display(&self, path: &str) -> Option<NodeId> {
        if path.is_empty() {
            return Some(0);
        }
        let mut cur = 0;
        for s in path.split('/') {
            cur = *self.nodes[cur as usize]
                .children
                .iter()
                .find(|c| self.nodes[**c as usize].segment.to_string() == s)?;
        }
        Some(cur)
    }

    pub fn by_uuid(&self, uuid: ItemUuid) -> Option<NodeId> {
        self.nodes
            .iter()
            .position(|n| n.uuid == uuid)
            .map(|i| i as NodeId)
    }

    pub fn fastener_count(&self) -> u64 {
        self.fasteners
            .iter()
            .map(|f| f.expansion.len() as u64)
            .sum()
    }
}

// ------------------------------------------------------------------------------ identity

fn hash16(parts: &[&[u8]]) -> ItemUuid {
    let mut h = Sha256::new();
    for p in parts {
        h.update((p.len() as u64).to_be_bytes());
        h.update(p);
    }
    let d: [u8; 32] = h.finalize().into();
    let mut b = [0u8; 16];
    b.copy_from_slice(&d[..16]);
    ItemUuid(u128::from_be_bytes(b))
}

/// The lock path of an authored item of an assembly (`occurrence`, `pattern`, `mirror`,
/// `fasteners`, `mate`, `connector`).
pub fn item_path(file: &FilePath, kind: &str, name: &str) -> ItemPath {
    ItemPath {
        file: file.clone(),
        item: vec![kind.to_owned(), name.to_owned()],
    }
}

/// Mint lock entries for every authored item of an assembly document. Instances need none.
pub fn lock_assembly(
    doc: &AssemblyDocument,
    lock: &mut LockSection,
    minter: &mut dyn crate::refs::lock::UuidMinter,
) -> usize {
    let before = lock.len();
    lock.ensure(&item_path(&doc.path, "assembly", &doc.name), minter);
    let mut names: BTreeSet<(&str, &str)> = BTreeSet::new();
    for o in &doc.occurrences {
        names.insert(("occurrence", &o.name));
    }
    for p in &doc.patterns {
        names.insert(("pattern", &p.name));
    }
    for m in &doc.mirrors {
        names.insert(("mirror", &m.name));
    }
    for f in &doc.fasteners {
        names.insert(("fasteners", &f.name));
    }
    for m in &doc.mates {
        names.insert(("mate", &m.name));
    }
    for c in &doc.connectors {
        names.insert(("connector", &c.name));
    }
    for e in &doc.exploded {
        names.insert(("exploded", &e.name));
    }
    for (k, n) in names {
        lock.ensure(&item_path(&doc.path, k, n), minter);
    }
    lock.len() - before
}

// ------------------------------------------------------------------------------ the resolver

#[derive(Clone, Debug)]
pub(super) struct Pending {
    pub remaining: Vec<PathSegment>,
    /// The override's whole target as written, and the component that carries it.
    pub target: RelPath,
    pub occurrence: String,
    pub kind: OverrideKind,
    pub file: FilePath,
    pub index: usize,
    /// Depth of the declaring assembly node: lower is outer, and outer wins.
    pub depth: u32,
}

pub(super) struct AsmInfo {
    pub doc: Arc<AssemblyDocument>,
    pub env: ValueEnv,
}

pub(super) struct InstancePlan {
    pub owner: NodeId,
    pub instance: NodeId,
    pub seed: NodeId,
    /// `local(instance) = outer ∘ left ∘ local(seed) ∘ right`, with `left` conjugated by the
    /// host's chain (`H ∘ left ∘ H⁻¹`) when `host` is set. `outer` is a mirror's reflection
    /// when the mirror repeats a pattern instance, and the identity otherwise.
    pub outer: Transform,
    pub left: Transform,
    pub right: Transform,
    pub host: Option<NodeId>,
}

pub(super) type DefinitionMemo = Result<Arc<DefinitionSummary>, SourceError>;
pub(super) type ConfiguredMemo = Result<Arc<ConfiguredSummary>, SourceError>;

pub(super) struct Resolver<'a, 'b> {
    pub ctx: &'a ResolveContext<'b>,
    pub meter: &'a dyn UsageMeter,
    pub req: &'a ResolveRequest,
    pub space: OptionSpace,
    pub point: Option<Point>,
    pub nodes: Vec<OccNode>,
    pub fasteners: Vec<OwnedFasteners>,
    pub mates: Vec<MateOutcome>,
    pub prototypes: Vec<Prototype>,
    pub envelopes: Vec<ResolvedEnvelope>,
    pub proto_index: BTreeMap<String, u32>,
    pub suppressed: Vec<String>,
    pub diags: DiagnosticReport,
    pub stats: ResolveStats,
    pub asm: BTreeMap<NodeId, AsmInfo>,
    pub defs: BTreeMap<(DefinitionRef, Option<RevisionId>), DefinitionMemo>,
    pub configured: BTreeMap<(DefinitionRef, Option<RevisionId>, String), ConfiguredMemo>,
    /// Condition problems already reported, by (file, condition text).
    pub reported: BTreeSet<(String, String)>,
    pub unlocked_files: BTreeMap<FilePath, u64>,
    /// Position overrides by node, applied by `place_children`.
    pub position_overrides: BTreeMap<NodeId, Placement>,
    /// Instance plans of flexible members, applied by the enclosing scope's solve.
    pub deferred: BTreeMap<NodeId, Vec<InstancePlan>>,
    /// Flexible members whose fastener rules expand after the enclosing scope's solve.
    pub deferred_fasteners: BTreeSet<NodeId>,
}

pub fn resolve(top: &FilePath, req: &ResolveRequest, ctx: &ResolveContext<'_>) -> ResolvedAssembly {
    resolve_metered(top, req, ctx, &NoUsageMeter)
}

/// [`resolve`], reporting every mate-solve pass to `meter`.
pub fn resolve_metered(
    top: &FilePath,
    req: &ResolveRequest,
    ctx: &ResolveContext<'_>,
    meter: &dyn UsageMeter,
) -> ResolvedAssembly {
    let mut r = Resolver {
        ctx,
        meter,
        req,
        space: OptionSpace::new(&OptionsBlock::default()),
        point: Some(Vec::new()),
        nodes: Vec::new(),
        fasteners: Vec::new(),
        mates: Vec::new(),
        prototypes: Vec::new(),
        envelopes: Vec::new(),
        proto_index: BTreeMap::new(),
        suppressed: Vec::new(),
        diags: DiagnosticReport::default(),
        stats: ResolveStats::default(),
        asm: BTreeMap::new(),
        defs: BTreeMap::new(),
        configured: BTreeMap::new(),
        reported: BTreeSet::new(),
        unlocked_files: BTreeMap::new(),
        position_overrides: BTreeMap::new(),
        deferred: BTreeMap::new(),
        deferred_fasteners: BTreeSet::new(),
    };
    let mut variant = BTreeMap::new();
    let doc = match ctx.source.assembly(top, req.revision.as_ref()) {
        Ok(d) => d,
        Err(e) => {
            r.diag(
                DiagCode::AsmUnknownDefinition,
                Severity::Error,
                Site::of(Subject::Document),
                e.to_string(),
                Vec::new(),
            );
            return r.finish(top, variant);
        }
    };
    if let Some(options) = &doc.options {
        r.space = OptionSpace::new(options);
        for (rule, p) in r.space.rule_problems.clone() {
            r.diag(
                option_code(&p),
                Severity::Error,
                Site::with_param(Subject::Document, &format!("options.rules.{}", rule)),
                p.to_string(),
                Vec::new(),
            );
        }
    }
    r.point = r.select_point(&doc);
    if let Some(p) = &r.point {
        variant = r.space.describe_point(p);
    }

    // The root node.
    let root_def = DefinitionRef::assembly(top.as_str());
    let (root_uuid, locked) = r.local_uuid(top, "assembly", &doc.name);
    r.nodes.push(OccNode {
        parent: None,
        segment: PathSegment::named(&doc.name),
        depth: 0,
        uuid: hash16(&[b"runmat-cad/occurrence/v1/root", &root_uuid.0.to_be_bytes()]),
        locked,
        definition: root_def.clone(),
        revision: req.revision.clone(),
        configuration: req
            .configuration
            .as_ref()
            .map(|c| ConfigKey::named(c.as_str()))
            .unwrap_or_default(),
        authored_in: FilePath::new(""),
        origin: NodeOrigin::Root,
        local: Transform::IDENTITY,
        world: Transform::IDENTITY,
        fixed: true,
        attached: false,
        flexible: false,
        pose: PoseStatus::Root,
        dof: None,
        representation: Representation::Full,
        represented_by_ancestor: false,
        material: None,
        properties: doc.properties.clone(),
        part_number: doc.part_number.clone(),
        prototype: 0,
        children: Vec::new(),
        expanded: true,
        failed: false,
    });
    let proto = r.prototype(
        &root_def,
        req.revision.as_ref(),
        &r.nodes[0].configuration.clone(),
        Representation::Full,
    );
    r.nodes[0].prototype = proto;
    let mut overrides = BTreeMap::new();
    let mut suppress = BTreeSet::new();
    if let Some(c) = &req.configuration {
        match doc.configuration(c) {
            Some(c) => {
                overrides = c.overrides.clone();
                suppress = c.suppress.clone();
            }
            None => {
                r.diag(
                    DiagCode::AsmUnknownConfiguration,
                    Severity::Error,
                    Site::of(Subject::Configuration { name: c.0.clone() }),
                    format!("`{}` has no configuration `{}`", top, c),
                    Vec::new(),
                );
            }
        }
    }
    let env = r.env_for(&doc, &overrides);
    r.compose_assembly(0, doc, env, Vec::new(), vec![top.clone()], false, &suppress);
    r.finish(top, variant)
}

pub(super) fn option_code(p: &OptionProblem) -> DiagCode {
    match p {
        OptionProblem::UnknownAxis { .. } | OptionProblem::UnknownGroup { .. } => {
            DiagCode::AsmUnknownAxis
        }
        OptionProblem::BadValue { .. } | OptionProblem::GroupCycle { .. } => {
            DiagCode::AsmBadAxisValue
        }
        OptionProblem::TooComplex { .. } => DiagCode::AsmConditionTooComplex,
    }
}

impl<'a, 'b> Resolver<'a, 'b> {
    pub(super) fn diag(
        &mut self,
        code: DiagCode,
        severity: Severity,
        site: Site,
        message: String,
        repairs: Vec<Repair>,
    ) -> u32 {
        self.diags
            .push(code, severity, site, message, Cause::Root, repairs)
    }

    fn finish(mut self, top: &FilePath, variant: BTreeMap<String, String>) -> ResolvedAssembly {
        for (file, n) in std::mem::take(&mut self.unlocked_files) {
            self.diag(
                DiagCode::AsmUnlocked,
                Severity::Info,
                Site::with_param(Subject::Document, file.as_str()),
                format!(
                    "{} item(s) of `{}` have no lock entry; their ids are derived from their paths and change if they are renamed (run `lock.update`)",
                    n, file
                ),
                Vec::new(),
            );
        }
        // World transforms, parents first (creation order guarantees it).
        for i in 0..self.nodes.len() {
            let w = match self.nodes[i].parent {
                None => self.nodes[i].local,
                Some(p) => self.nodes[p as usize].world.compose(&self.nodes[i].local),
            };
            self.nodes[i].world = w;
        }
        for e in &mut self.envelopes {
            e.world = e
                .local
                .transformed(&self.nodes[e.declared_by as usize].world);
        }
        self.stats.nodes = self.nodes.len() as u64;
        self.stats.prototypes = self.prototypes.len() as u64;
        self.stats.fastener_sets = self.fasteners.len() as u64;
        self.stats.fasteners = self
            .fasteners
            .iter()
            .map(|f| f.expansion.len() as u64)
            .sum();
        self.stats.suppressed = self.suppressed.len() as u64;
        ResolvedAssembly {
            top: top.clone(),
            variant,
            nodes: self.nodes,
            fasteners: self.fasteners,
            mates: self.mates,
            prototypes: self.prototypes,
            envelopes: self.envelopes,
            suppressed: self.suppressed,
            diagnostics: self.diags,
            stats: self.stats,
        }
    }

    fn select_point(&mut self, doc: &AssemblyDocument) -> Option<Point> {
        let values = match &self.req.variant {
            VariantSelection::Defaults => BTreeMap::new(),
            VariantSelection::Values { values } => values.clone(),
            VariantSelection::Named { name } => {
                let found = doc
                    .options
                    .as_ref()
                    .and_then(|o| o.configurations.iter().find(|c| &c.name == name));
                match found {
                    Some(c) => c.values.clone(),
                    None => {
                        self.diag(
                            DiagCode::AsmUnknownConfiguration,
                            Severity::Error,
                            Site::of(Subject::Configuration { name: name.clone() }),
                            format!("the options block has no configuration `{}`", name),
                            Vec::new(),
                        );
                        return None;
                    }
                }
            }
        };
        match self.space.point(&values) {
            Ok(p) => {
                if !self.space.is_legal(&p) {
                    let broken: Vec<String> = doc
                        .options
                        .as_ref()
                        .map(|o| {
                            o.rules
                                .iter()
                                .filter(|r| !self.space.eval(&r.condition, &p).unwrap_or(true))
                                .map(|r| format!("{} ({})", r.name, r.condition))
                                .collect()
                        })
                        .unwrap_or_default();
                    self.diag(
                        DiagCode::AsmIllegalConfiguration,
                        Severity::Error,
                        Site::of(Subject::Document),
                        format!(
                            "the configuration breaks the option rules: {}",
                            broken.join("; ")
                        ),
                        Vec::new(),
                    );
                }
                Some(p)
            }
            Err(problems) => {
                for p in problems {
                    self.diag(
                        option_code(&p),
                        Severity::Error,
                        Site::of(Subject::Document),
                        p.to_string(),
                        Vec::new(),
                    );
                }
                None
            }
        }
    }
}

pub(super) fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if n > 0.0 {
        [a[0] / n, a[1] / n, a[2] / n]
    } else {
        a
    }
}
