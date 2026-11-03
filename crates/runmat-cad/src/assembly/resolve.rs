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
    for (k, n) in names {
        lock.ensure(&item_path(&doc.path, k, n), minter);
    }
    lock.len() - before
}

// ------------------------------------------------------------------------------ the resolver

#[derive(Clone, Debug)]
pub(super) struct Pending {
    pub remaining: Vec<PathSegment>,
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
    /// `local(instance) = left ∘ local(seed) ∘ right`, with `left` conjugated by the host's
    /// chain (`H ∘ left ∘ H⁻¹`) when `host` is set.
    pub left: Transform,
    pub right: Transform,
    pub host: Option<NodeId>,
}

pub(super) type DefinitionMemo = Result<Arc<DefinitionSummary>, SourceError>;
pub(super) type ConfiguredMemo = Result<Arc<ConfiguredSummary>, SourceError>;

pub(super) struct Resolver<'a, 'b> {
    pub ctx: &'a ResolveContext<'b>,
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
    let mut r = Resolver {
        ctx,
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

    /// Evaluate a `when` at the configuration point. Problems are reported once per text.
    pub(super) fn holds(&mut self, c: Option<&Condition>, file: &FilePath, site: &Site) -> bool {
        let Some(c) = c else { return true };
        let Some(point) = &self.point else {
            return false;
        };
        match self.space.eval(c, point) {
            Ok(b) => b,
            Err(p) => {
                if self.reported.insert((file.0.clone(), c.to_string())) {
                    self.diag(
                        option_code(&p),
                        Severity::Error,
                        site.clone(),
                        format!("`when {}`: {}", c, p),
                        Vec::new(),
                    );
                }
                false
            }
        }
    }

    pub(super) fn ev(&self) -> Evaluator<'a> {
        Evaluator {
            units: self.ctx.units,
            dynamic: self.ctx.dynamic,
        }
    }

    pub(super) fn env_for(
        &mut self,
        doc: &AssemblyDocument,
        overrides: &BTreeMap<InputName, ValueExpr>,
    ) -> ValueEnv {
        let env = self.ev().evaluate_env(&doc.inputs, &doc.lets, overrides);
        for (name, e) in &env.failed {
            self.diag(
                DiagCode::ValueFailed,
                Severity::Error,
                Site::with_param(Subject::Value { name: name.clone() }, doc.path.as_str()),
                format!("`{}`: {}", doc.path, e),
                Vec::new(),
            );
        }
        env
    }

    fn local_uuid(&mut self, file: &FilePath, kind: &str, name: &str) -> (ItemUuid, bool) {
        let p = item_path(file, kind, name);
        if let Some(l) = self.ctx.lock.and_then(|l| l.lookup(&p)) {
            return (l.uuid, true);
        }
        *self.unlocked_files.entry(file.clone()).or_insert(0) += 1;
        self.stats.unlocked += 1;
        (
            hash16(&[
                b"runmat-cad/occurrence/v1/unlocked",
                p.to_string().as_bytes(),
            ]),
            false,
        )
    }

    pub(super) fn definition(
        &mut self,
        d: &DefinitionRef,
        rev: Option<&RevisionId>,
    ) -> Result<Arc<DefinitionSummary>, SourceError> {
        let key = (d.clone(), rev.cloned());
        if let Some(v) = self.defs.get(&key) {
            return v.clone();
        }
        let v = self.ctx.source.definition(d, rev);
        self.defs.insert(key, v.clone());
        v
    }

    pub(super) fn configured_summary(
        &mut self,
        d: &DefinitionRef,
        rev: Option<&RevisionId>,
        c: &ConfigKey,
    ) -> Result<Arc<ConfiguredSummary>, SourceError> {
        let key = (d.clone(), rev.cloned(), c.digest());
        if let Some(v) = self.configured.get(&key) {
            return v.clone();
        }
        let v = self.ctx.source.configured(d, rev, c);
        self.configured.insert(key, v.clone());
        v
    }

    fn prototype(
        &mut self,
        d: &DefinitionRef,
        rev: Option<&RevisionId>,
        c: &ConfigKey,
        rep: Representation,
    ) -> u32 {
        let key = format!(
            "{}|{}|{}|{:?}",
            d,
            rev.map(|r| r.0.as_str()).unwrap_or(""),
            c.digest(),
            rep
        );
        if let Some(&i) = self.proto_index.get(&key) {
            self.prototypes[i as usize].instances += 1;
            return i;
        }
        let i = self.prototypes.len() as u32;
        self.prototypes.push(Prototype {
            definition: d.clone(),
            revision: rev.cloned(),
            configuration: c.clone(),
            representation: rep,
            instances: 1,
        });
        self.proto_index.insert(key, i);
        i
    }

    pub(super) fn display_path(&self, id: NodeId) -> String {
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

    pub(super) fn occ_site(&self, id: NodeId) -> Site {
        Site::of(Subject::Occurrence {
            path: self.display_path(id),
        })
    }

    pub(super) fn child(&self, parent: NodeId, seg: &PathSegment) -> Option<NodeId> {
        self.nodes[parent as usize]
            .children
            .iter()
            .copied()
            .find(|c| &self.nodes[*c as usize].segment == seg)
    }

    /// Walk a relative path; `Err(i)` names the first segment that is missing.
    pub(super) fn walk(&self, from: NodeId, path: &RelPath) -> Result<NodeId, usize> {
        let mut cur = from;
        for (i, s) in path.0.iter().enumerate() {
            cur = self.child(cur, s).ok_or(i)?;
        }
        Ok(cur)
    }

    /// `from`-coordinates of the descendant `to`'s frame.
    pub(super) fn chain(&self, from: NodeId, to: NodeId) -> Transform {
        let mut stack = Vec::new();
        let mut cur = to;
        while cur != from {
            stack.push(cur);
            match self.nodes[cur as usize].parent {
                Some(p) => cur = p,
                None => break,
            }
        }
        let mut t = Transform::IDENTITY;
        for n in stack.iter().rev() {
            t = t.compose(&self.nodes[*n as usize].local);
        }
        t
    }

    // ------------------------------------------------------------------ values to transforms

    pub(super) fn length(&self, e: &ValueExpr, env: &ValueEnv) -> Result<f64, String> {
        self.ev()
            .eval_as(e, env, Dimension::LENGTH)
            .map_err(|e| e.to_string())
    }

    pub(super) fn angle(&self, e: &ValueExpr, env: &ValueEnv) -> Result<f64, String> {
        self.ev()
            .eval_as(e, env, Dimension::ANGLE)
            .map_err(|e| e.to_string())
    }

    pub(super) fn count(&self, e: &ValueExpr, env: &ValueEnv) -> Result<u32, String> {
        let q = self.ev().eval(e, env).map_err(|e| e.to_string())?;
        if !q.dimension.is_dimensionless() || q.si < 1.0 || q.si.fract().abs() > 1e-9 {
            return Err(format!(
                "a count must be a whole number of at least 1, found {}",
                q.si
            ));
        }
        Ok(q.si.round() as u32)
    }

    pub(super) fn vec3(&self, v: &[ValueExpr; 3], env: &ValueEnv) -> Result<[f64; 3], String> {
        Ok([
            self.length(&v[0], env)?,
            self.length(&v[1], env)?,
            self.length(&v[2], env)?,
        ])
    }

    pub(super) fn offset(&self, o: &OffsetExpr, env: &ValueEnv) -> Result<Transform, String> {
        let t = self.vec3(&o.translation, env)?;
        let r = match &o.rotation {
            RotationExpr::Identity => Transform::IDENTITY,
            RotationExpr::AxisAngle { axis, angle } => {
                Transform::rotation_about([0.0; 3], *axis, self.angle(angle, env)?)
            }
            RotationExpr::Euler { yaw, pitch, roll } => {
                let z = Transform::rotation_about([0.0; 3], [0.0, 0.0, 1.0], self.angle(yaw, env)?);
                let y =
                    Transform::rotation_about([0.0; 3], [0.0, 1.0, 0.0], self.angle(pitch, env)?);
                let x =
                    Transform::rotation_about([0.0; 3], [1.0, 0.0, 0.0], self.angle(roll, env)?);
                z.compose(&y).compose(&x)
            }
            RotationExpr::Basis { x, z } => {
                let n = |a: [f64; 3]| {
                    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
                    [a[0] / l, a[1] / l, a[2] / l]
                };
                let cr = |a: [f64; 3], b: [f64; 3]| {
                    [
                        a[1] * b[2] - a[2] * b[1],
                        a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0],
                    ]
                };
                let zz = n(*z);
                let yy = n(cr(zz, *x));
                if !yy.iter().all(|v| v.is_finite()) {
                    return Err("the basis X and Z are parallel".into());
                }
                let xx = cr(yy, zz);
                Transform::from_frame(&crate::model::datum::Frame {
                    origin: [0.0; 3],
                    x: xx,
                    y: yy,
                    z: zz,
                })
            }
        };
        Ok(Transform::translation(t).compose(&r))
    }

    // ------------------------------------------------------------------ configurations

    fn eval_selection(
        &mut self,
        sel: &ConfigurationSelection,
        env: &ValueEnv,
        summary: &DefinitionSummary,
        file: &FilePath,
        site: &Site,
    ) -> Result<ConfigKey, (DiagCode, String)> {
        match sel {
            ConfigurationSelection::Default => Ok(ConfigKey::default()),
            ConfigurationSelection::Named { name } => {
                self.check_config_name(name, summary)?;
                Ok(ConfigKey {
                    name: Some(name.clone()),
                    inputs: BTreeMap::new(),
                })
            }
            ConfigurationSelection::Inputs { base, values } => {
                if let Some(b) = base {
                    self.check_config_name(b, summary)?;
                }
                let mut inputs = BTreeMap::new();
                for (name, expr) in values {
                    let si = self.typed_input(name, expr, env, summary)?;
                    inputs.insert(name.clone(), si);
                }
                Ok(ConfigKey {
                    name: base.clone(),
                    inputs,
                })
            }
            ConfigurationSelection::Switch { cases, otherwise } => {
                for c in cases {
                    if self.holds(Some(&c.when), file, site) {
                        return self.eval_selection(&c.select, env, summary, file, site);
                    }
                }
                self.eval_selection(otherwise, env, summary, file, site)
            }
        }
    }

    fn check_config_name(
        &self,
        name: &ConfigurationName,
        s: &DefinitionSummary,
    ) -> Result<(), (DiagCode, String)> {
        if s.configurations.contains(name) {
            Ok(())
        } else {
            Err((
                DiagCode::AsmUnknownConfiguration,
                format!(
                    "`{}` has no configuration `{}` (it has: {})",
                    s.definition,
                    name,
                    s.configurations
                        .iter()
                        .map(|c| c.0.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ))
        }
    }

    fn typed_input(
        &self,
        name: &InputName,
        expr: &ValueExpr,
        env: &ValueEnv,
        s: &DefinitionSummary,
    ) -> Result<f64, (DiagCode, String)> {
        let Some((_, kind)) = s.inputs.iter().find(|(n, _)| n == name) else {
            return Err((
                DiagCode::AsmInputType,
                format!("`{}` has no input `{}`", s.definition, name),
            ));
        };
        let Some(info) = self.ctx.units.kind(kind) else {
            return Err((
                DiagCode::AsmInputType,
                format!("input `{}` has an unknown kind `{}`", name, kind),
            ));
        };
        let q = self
            .ev()
            .eval(expr, env)
            .map_err(|e| (DiagCode::AsmInputType, format!("input `{}`: {}", name, e)))?;
        if q.dimension != info.dimension {
            return Err((
                DiagCode::AsmInputType,
                format!(
                    "input `{}` of `{}` is a {} ({}), found {}",
                    name, s.definition, info.id, info.si_symbol, q.dimension
                ),
            ));
        }
        Ok(q.si)
    }

    /// The overrides of a sub-assembly's inputs from its configuration key.
    fn assembly_overrides(
        &self,
        doc: &AssemblyDocument,
        key: &ConfigKey,
    ) -> (BTreeMap<InputName, ValueExpr>, BTreeSet<String>) {
        let mut overrides = BTreeMap::new();
        let mut suppress = BTreeSet::new();
        if let Some(n) = &key.name {
            if let Some(c) = doc.configuration(n) {
                overrides = c.overrides.clone();
                suppress = c.suppress.clone();
            }
        }
        for (name, si) in &key.inputs {
            if let Some(decl) = doc.inputs.iter().find(|i| &i.name == name) {
                if let Some(info) = self.ctx.units.kind(&decl.kind) {
                    overrides.insert(name.clone(), ValueExpr::qty(*si, &info.si_symbol));
                }
            }
        }
        (overrides, suppress)
    }

    // ------------------------------------------------------------------ composition

    #[allow(clippy::too_many_arguments)]
    pub(super) fn compose_assembly(
        &mut self,
        node: NodeId,
        doc: Arc<AssemblyDocument>,
        env: ValueEnv,
        pending: Vec<Pending>,
        stack: Vec<FilePath>,
        flexible_member: bool,
        suppress: &BTreeSet<String>,
    ) {
        self.stats.assemblies += 1;
        let file = doc.path.clone();
        let depth = self.nodes[node as usize].depth;
        let names = doc.child_names();
        if names.len() > 500 {
            self.diag(
                DiagCode::AsmTooManyChildren,
                Severity::Warning,
                Site::with_param(Subject::Document, file.as_str()),
                format!(
                    "`{}` has {} direct children; split it by zone or partition (about 500 at most)",
                    file,
                    names.len()
                ),
                Vec::new(),
            );
        }
        self.asm.insert(
            node,
            AsmInfo {
                doc: doc.clone(),
                env: env.clone(),
            },
        );

        // Segments that exist in this configuration, suppressed or not (override targets).
        let mut present: BTreeSet<PathSegment> = BTreeSet::new();
        // 1–3: authored children.
        for name in names {
            let mut active = Vec::new();
            for (i, o) in doc.occurrences.iter().enumerate() {
                if o.name != name {
                    continue;
                }
                let site = Site::with_param(
                    Subject::Occurrence {
                        path: name.to_owned(),
                    },
                    file.as_str(),
                );
                if self.holds(o.when.as_ref(), &file, &site) {
                    active.push(i);
                }
            }
            if active.len() > 1 {
                let witness = self
                    .point
                    .as_ref()
                    .map(|p| self.space.describe_point(p))
                    .unwrap_or_default();
                self.diag(
                    DiagCode::AsmDuplicateName,
                    Severity::Error,
                    Site::with_param(
                        Subject::Occurrence {
                            path: name.to_owned(),
                        },
                        file.as_str(),
                    ),
                    format!(
                        "{} alternatives of `{}` in `{}` are active at once; the first is used",
                        active.len(),
                        name,
                        file
                    ),
                    vec![Repair::ExcludeConfiguration {
                        assembly: file.clone(),
                        occurrence: name.to_owned(),
                        witness,
                    }],
                );
            }
            let Some(&first) = active.first() else {
                if doc.alternatives(name).any(|o| o.required) {
                    self.diag(
                        DiagCode::AsmIncompleteVariant,
                        Severity::Error,
                        Site::with_param(
                            Subject::Occurrence {
                                path: name.to_owned(),
                            },
                            file.as_str(),
                        ),
                        format!(
                            "no alternative of the required occurrence `{}` is active",
                            name
                        ),
                        Vec::new(),
                    );
                }
                continue;
            };
            let o = &doc.occurrences[first];
            present.insert(PathSegment::named(&o.name));
            if o.suppressed || suppress.contains(&o.name) {
                self.stats.suppressed += 1;
                let p = self.display_path(node);
                self.suppressed.push(if p.is_empty() {
                    o.name.clone()
                } else {
                    format!("{}/{}", p, o.name)
                });
                continue;
            }
            self.compose_occurrence(
                node,
                &doc,
                &env,
                o,
                PathSegment::named(&o.name),
                &pending,
                NodeOrigin::Authored,
                &stack,
                flexible_member,
            );
        }

        // 6 (structure): pattern and mirror instances.
        let plans = self.compose_instances(
            node,
            &doc,
            &env,
            &pending,
            &stack,
            flexible_member,
            &mut present,
        );

        // Pending overrides whose first segment names no child here.
        for p in &pending {
            let Some(first) = p.remaining.first() else {
                continue;
            };
            if self.child(node, first).is_none() && !present.contains(first) {
                let candidates: Vec<String> = self.nodes[node as usize]
                    .children
                    .iter()
                    .map(|c| self.nodes[*c as usize].segment.to_string())
                    .filter(|s| s.contains(&first.name) || first.name.contains(s.as_str()))
                    .collect();
                let path = self.display_path(node);
                self.diag(
                    DiagCode::AsmOverrideTargetMissing,
                    Severity::Error,
                    Site::with_param(Subject::Override { index: p.index }, p.file.as_str()),
                    format!(
                        "override {} in `{}` ({}) targets `{}`, which `{}` does not contain in this configuration",
                        p.index,
                        p.file,
                        p.kind.slot(),
                        RelPath(p.remaining.clone()),
                        if path.is_empty() { doc.name.clone() } else { path }
                    ),
                    if candidates.is_empty() {
                        vec![Repair::RemoveOverride {
                            assembly: p.file.clone(),
                            index: p.index,
                        }]
                    } else {
                        vec![
                            Repair::RetargetOverride {
                                assembly: p.file.clone(),
                                index: p.index,
                                candidates,
                            },
                            Repair::RemoveOverride {
                                assembly: p.file.clone(),
                                index: p.index,
                            },
                        ]
                    },
                );
            }
        }
        let _ = depth;

        // 4: placements of authored children.
        self.place_children(node, &doc, &env);
        // 5: mates (a flexible member's mates, instances and fasteners join its parent's solve).
        if !self.req.solve_mates {
            self.apply_instances(&plans);
        } else if flexible_member {
            self.deferred.insert(node, plans);
            if self.req.expand_fasteners && !doc.fasteners.is_empty() {
                self.deferred_fasteners.insert(node);
            }
            self.declare_envelopes(node, &doc, &env);
            return;
        } else {
            self.solve_scope(node, plans);
        }
        // 6: fasteners.
        if self.req.expand_fasteners {
            self.expand_fasteners(node, &doc, &env);
        }
        self.declare_envelopes(node, &doc, &env);
    }

    #[allow(clippy::too_many_arguments)]
    fn compose_occurrence(
        &mut self,
        parent: NodeId,
        doc: &Arc<AssemblyDocument>,
        env: &ValueEnv,
        o: &Occurrence,
        segment: PathSegment,
        pending: &[Pending],
        origin: NodeOrigin,
        stack: &[FilePath],
        parent_flexible: bool,
    ) -> Option<NodeId> {
        let file = doc.path.clone();
        let pdepth = self.nodes[parent as usize].depth;
        // Overrides that land here: outer (lower depth) wins per slot.
        let mut here: BTreeMap<String, &Pending> = BTreeMap::new();
        let mut shadowed: Vec<(&Pending, &Pending)> = Vec::new();
        for p in pending {
            if p.remaining.len() == 1 && p.remaining[0] == segment {
                let slot = p.kind.slot();
                match here.get(&slot) {
                    Some(q) if q.depth <= p.depth => shadowed.push((p, q)),
                    Some(q) => {
                        shadowed.push((q, p));
                        here.insert(slot, p);
                    }
                    None => {
                        here.insert(slot, p);
                    }
                }
            }
        }
        let child_path = {
            let p = self.display_path(parent);
            if p.is_empty() {
                segment.to_string()
            } else {
                format!("{}/{}", p, segment)
            }
        };
        for (lost, won) in shadowed {
            self.diag(
                DiagCode::AsmOverrideShadowed,
                Severity::Info,
                Site::with_param(Subject::Override { index: lost.index }, lost.file.as_str()),
                format!(
                    "override {} in `{}` sets `{}` of `{}`, but `{}` (outer) sets it too and wins",
                    lost.index,
                    lost.file,
                    lost.kind.slot(),
                    child_path,
                    won.file
                ),
                Vec::new(),
            );
        }
        if let Some(p) = here.get("suppress") {
            if let OverrideKind::Suppress { suppressed: true } = p.kind {
                self.stats.suppressed += 1;
                self.suppressed.push(child_path);
                return None;
            }
        }

        // Effective source and configuration.
        let mut selection = o.configuration.clone();
        if let Some(p) = here.get("configuration") {
            if let OverrideKind::Configuration { select } = &p.kind {
                selection = select.clone();
            }
        }
        let mut extra_inputs: BTreeMap<InputName, ValueExpr> = BTreeMap::new();
        for (slot, p) in &here {
            if let (true, OverrideKind::Input { name, value }) =
                (slot.starts_with("input."), &p.kind)
            {
                extra_inputs.insert(name.clone(), value.clone());
            }
        }
        if !extra_inputs.is_empty() {
            selection = match selection {
                ConfigurationSelection::Default => ConfigurationSelection::Inputs {
                    base: None,
                    values: extra_inputs,
                },
                ConfigurationSelection::Named { name } => ConfigurationSelection::Inputs {
                    base: Some(name),
                    values: extra_inputs,
                },
                ConfigurationSelection::Inputs { base, mut values } => {
                    values.extend(extra_inputs);
                    ConfigurationSelection::Inputs { base, values }
                }
                other => other,
            };
        }
        let revision = match &o.source.context {
            XrefContext::Pinned { revision } => Some(revision.clone()),
            XrefContext::Live => None,
        };
        let def = o.source.definition.clone();
        let site = Site::with_param(
            Subject::Occurrence {
                path: child_path.clone(),
            },
            file.as_str(),
        );

        // Live across partitions is not allowed (decision 11).
        if revision.is_none() {
            let mine = self.ctx.source.partition(&file);
            let theirs = self.ctx.source.partition(&def.file);
            if mine.is_some() && theirs.is_some() && mine != theirs {
                let rel = self.ctx.source.latest_release(&def.file);
                self.diag(
                    DiagCode::AsmLiveAcrossPartition,
                    Severity::Error,
                    site.clone(),
                    format!(
                        "`{}` uses `{}` live, but it is in another partition; pin it to a released revision",
                        child_path, def.file
                    ),
                    rel.map(|r| {
                        vec![Repair::PinOccurrence {
                            assembly: file.clone(),
                            occurrence: o.name.clone(),
                            revision: r,
                        }]
                    })
                    .unwrap_or_default(),
                );
            }
        }

        let summary = self.definition(&def, revision.as_ref());
        let (config, part_number, failed, def_material) = match &summary {
            Ok(s) => {
                let s = s.clone();
                let key = match self.eval_selection(&selection, env, &s, &file, &site) {
                    Ok(k) => k,
                    Err((code, msg)) => {
                        self.diag(code, Severity::Error, site.clone(), msg, Vec::new());
                        ConfigKey::default()
                    }
                };
                (key, s.part_number.clone(), false, s.material.clone())
            }
            Err(e) => {
                self.diag(
                    DiagCode::AsmUnknownDefinition,
                    Severity::Error,
                    site.clone(),
                    format!("`{}`: {}", child_path, e),
                    Vec::new(),
                );
                (ConfigKey::default(), None, true, None)
            }
        };

        // Representation: override, own, or the parent's.
        let parent_rep = self.nodes[parent as usize].representation;
        let parent_by_ancestor = self.nodes[parent as usize].represented_by_ancestor
            || matches!(
                parent_rep,
                Representation::Envelope | Representation::Interface
            ) && self.nodes[parent as usize].parent.is_some();
        let mut rep = o.representation.unwrap_or(parent_rep);
        if let Some(p) = here.get("representation") {
            if let OverrideKind::Representation { representation } = p.kind {
                rep = representation;
            }
        }
        if let Ok(s) = &summary {
            if !s.provides(rep) {
                self.diag(
                    DiagCode::AsmRepresentationUnavailable,
                    Severity::Warning,
                    site.clone(),
                    format!(
                        "`{}` declares no `{:?}` representation; `full` is used",
                        def, rep
                    ),
                    Vec::new(),
                );
                rep = Representation::Full;
            }
        }

        let mut placement_fixed = o.fixed;
        let mut flexible = o.flexible;
        if let Some(p) = here.get("flexible") {
            if let OverrideKind::Flexible { flexible: f } = p.kind {
                flexible = f;
            }
        }
        if let Some(p) = here.get("position") {
            if !(o.fixed || parent_flexible) {
                self.diag(
                    DiagCode::AsmOverrideNotAllowed,
                    Severity::Error,
                    Site::with_param(Subject::Override { index: p.index }, p.file.as_str()),
                    format!(
                        "`{}` is positioned by mates in a rigid sub-assembly; a position override cannot move it (make the sub-assembly flexible in this context)",
                        child_path
                    ),
                    Vec::new(),
                );
            } else {
                placement_fixed = true;
            }
        }
        let mut material = o.material.clone().or(def_material);
        if let Some(p) = here.get("material") {
            if let OverrideKind::Material { material: m } = &p.kind {
                material = Some(m.clone());
            }
        }
        let mut properties = o.properties.clone();
        for (slot, p) in &here {
            if let (true, OverrideKind::Property { property }) =
                (slot.starts_with("property."), &p.kind)
            {
                properties.retain(|q| q.name != property.name);
                properties.push(property.clone());
            }
        }

        // Identity.
        let (local_id, locked) = match &origin {
            NodeOrigin::Authored | NodeOrigin::Root => {
                self.local_uuid(&file, "occurrence", &o.name)
            }
            NodeOrigin::Pattern { pattern } => {
                let (pid, pl) = self.local_uuid(&file, "pattern", pattern);
                let (sid, sl) = self.local_uuid(&file, "occurrence", &o.name);
                let key = segment
                    .instance
                    .as_ref()
                    .map(|i| i.key.to_string())
                    .unwrap_or_default();
                (
                    hash16(&[
                        b"runmat-cad/occurrence/v1/instance",
                        &pid.0.to_be_bytes(),
                        &sid.0.to_be_bytes(),
                        key.as_bytes(),
                    ]),
                    pl && sl,
                )
            }
            NodeOrigin::Mirror { mirror } => {
                let (mid, ml) = self.local_uuid(&file, "mirror", mirror);
                let (sid, sl) = self.local_uuid(&file, "occurrence", &o.name);
                (
                    hash16(&[
                        b"runmat-cad/occurrence/v1/mirror",
                        &mid.0.to_be_bytes(),
                        &sid.0.to_be_bytes(),
                    ]),
                    ml && sl,
                )
            }
        };
        let parent_uuid = self.nodes[parent as usize].uuid;
        let uuid = hash16(&[
            b"runmat-cad/occurrence/v1/node",
            &parent_uuid.0.to_be_bytes(),
            &local_id.0.to_be_bytes(),
        ]);
        let id = self.nodes.len() as NodeId;
        let proto = self.prototype(&def, revision.as_ref(), &config, rep);
        self.nodes.push(OccNode {
            parent: Some(parent),
            segment: segment.clone(),
            depth: pdepth + 1,
            uuid,
            locked,
            definition: def.clone(),
            revision: revision.clone(),
            configuration: config.clone(),
            authored_in: file.clone(),
            origin,
            local: Transform::IDENTITY,
            world: Transform::IDENTITY,
            fixed: placement_fixed,
            flexible: flexible && def.kind == DefinitionKind::Assembly,
            pose: PoseStatus::Placed,
            dof: None,
            representation: rep,
            represented_by_ancestor: parent_by_ancestor,
            material,
            properties,
            part_number,
            prototype: proto,
            children: Vec::new(),
            expanded: false,
            failed,
        });
        self.nodes[parent as usize].children.push(id);
        // Position override: replaces the placement (applied in place_children through this map).
        if let Some(p) = here.get("position") {
            if let (OverrideKind::Position { placement }, true) = (&p.kind, placement_fixed) {
                self.position_overrides.insert(id, placement.clone());
            }
        }

        // Descend into sub-assemblies.
        if def.kind == DefinitionKind::Assembly && !failed {
            let depth_ok = self.req.max_depth.map(|m| pdepth + 1 < m).unwrap_or(true);
            if !depth_ok || self.req.collapse.contains(&def.file) {
                self.stats.collapsed += 1;
                return Some(id);
            }
            if stack.contains(&def.file) {
                let mut cycle: Vec<String> = stack.iter().map(|f| f.0.clone()).collect();
                cycle.push(def.file.0.clone());
                self.diag(
                    DiagCode::AsmCycle,
                    Severity::Error,
                    site,
                    format!("assemblies use each other: {}", cycle.join(" → ")),
                    Vec::new(),
                );
                self.nodes[id as usize].failed = true;
                return Some(id);
            }
            match self.ctx.source.assembly(&def.file, revision.as_ref()) {
                Ok(sub) => {
                    let mut child_pending: Vec<Pending> = pending
                        .iter()
                        .filter(|p| p.remaining.len() > 1 && p.remaining[0] == segment)
                        .map(|p| Pending {
                            remaining: p.remaining[1..].to_vec(),
                            ..p.clone()
                        })
                        .collect();
                    for (i, ov) in o.overrides.iter().enumerate() {
                        let osite = Site::with_param(Subject::Override { index: i }, file.as_str());
                        if !self.holds(ov.when.as_ref(), &file, &osite) {
                            continue;
                        }
                        if ov.target.is_empty() {
                            continue;
                        }
                        child_pending.push(Pending {
                            remaining: ov.target.0.clone(),
                            kind: ov.set.clone(),
                            file: file.clone(),
                            index: i,
                            depth: pdepth,
                        });
                    }
                    let (overrides, suppress) = self.assembly_overrides(&sub, &config);
                    let env = self.env_for(&sub, &overrides);
                    let mut st = stack.to_vec();
                    st.push(def.file.clone());
                    self.nodes[id as usize].expanded = true;
                    let flex = self.nodes[id as usize].flexible;
                    self.compose_assembly(id, sub, env, child_pending, st, flex, &suppress);
                }
                Err(e) => {
                    self.diag(
                        DiagCode::AsmUnknownDefinition,
                        Severity::Error,
                        site,
                        format!("`{}`: {}", child_path, e),
                        Vec::new(),
                    );
                    self.nodes[id as usize].failed = true;
                }
            }
        } else if !o.overrides.is_empty() {
            for (i, ov) in o.overrides.iter().enumerate() {
                if ov.target.is_empty() {
                    continue;
                }
                self.diag(
                    DiagCode::AsmOverrideTargetMissing,
                    Severity::Error,
                    Site::with_param(Subject::Override { index: i }, file.as_str()),
                    format!(
                        "override {} of `{}` targets `{}`, but a {:?} has no occurrences",
                        i, o.name, ov.target, def.kind
                    ),
                    vec![Repair::RemoveOverride {
                        assembly: file.clone(),
                        index: i,
                    }],
                );
            }
        }
        Some(id)
    }

    #[allow(clippy::too_many_arguments)]
    fn compose_instances(
        &mut self,
        node: NodeId,
        doc: &Arc<AssemblyDocument>,
        env: &ValueEnv,
        pending: &[Pending],
        stack: &[FilePath],
        flexible_member: bool,
        present: &mut BTreeSet<PathSegment>,
    ) -> Vec<InstancePlan> {
        let file = doc.path.clone();
        let mut plans = Vec::new();
        for p in &doc.patterns {
            let site = Site::with_param(
                Subject::Pattern {
                    name: p.name.clone(),
                },
                file.as_str(),
            );
            if !self.holds(p.when.as_ref(), &file, &site) {
                continue;
            }
            let (transforms, host) = match self.pattern_transforms(node, &p.layout, env) {
                Ok(t) => t,
                Err(msg) => {
                    self.diag(
                        DiagCode::AsmBadPattern,
                        Severity::Error,
                        site,
                        format!("pattern `{}`: {}", p.name, msg),
                        Vec::new(),
                    );
                    continue;
                }
            };
            for seed_name in &p.seeds {
                let Some(seed) = self.child(node, &PathSegment::named(seed_name)) else {
                    self.diag(
                        DiagCode::AsmUnknownOccurrence,
                        Severity::Error,
                        site.clone(),
                        format!(
                            "pattern `{}` repeats `{}`, which is not active here",
                            p.name, seed_name
                        ),
                        Vec::new(),
                    );
                    continue;
                };
                let Some(occ) = self.active_occurrence(doc, seed_name) else {
                    continue;
                };
                for (key, t) in &transforms {
                    if p.skip.contains(key) {
                        continue;
                    }
                    let seg = PathSegment::instance(seed_name, &p.name, key.clone());
                    present.insert(seg.clone());
                    if let Some(inst) = self.compose_occurrence(
                        node,
                        doc,
                        env,
                        &occ,
                        seg,
                        pending,
                        NodeOrigin::Pattern {
                            pattern: p.name.clone(),
                        },
                        stack,
                        flexible_member,
                    ) {
                        plans.push(InstancePlan {
                            owner: node,
                            instance: inst,
                            seed,
                            left: *t,
                            right: Transform::IDENTITY,
                            host,
                        });
                    }
                }
            }
        }
        for m in &doc.mirrors {
            let site = Site::with_param(
                Subject::Pattern {
                    name: m.name.clone(),
                },
                file.as_str(),
            );
            if !self.holds(m.when.as_ref(), &file, &site) {
                continue;
            }
            let world = match self.plane(node, &m.plane, env) {
                Ok((pt, n)) => Transform::reflection(pt, n),
                Err(msg) => {
                    self.diag(
                        DiagCode::AsmBadPattern,
                        Severity::Error,
                        site,
                        format!("mirror `{}`: {}", m.name, msg),
                        Vec::new(),
                    );
                    continue;
                }
            };
            for seed_name in &m.seeds {
                let Some(seed) = self.child(node, &PathSegment::named(seed_name)) else {
                    self.diag(
                        DiagCode::AsmUnknownOccurrence,
                        Severity::Error,
                        site.clone(),
                        format!(
                            "mirror `{}` reflects `{}`, which is not active here",
                            m.name, seed_name
                        ),
                        Vec::new(),
                    );
                    continue;
                };
                let Some(mut occ) = self.active_occurrence(doc, seed_name) else {
                    continue;
                };
                let (pt, n) = match &m.hand {
                    MirrorHand::Symmetric { point, normal } => (point, normal),
                    MirrorHand::Opposite {
                        source,
                        configuration,
                        point,
                        normal,
                    } => {
                        occ.source = source.clone();
                        occ.configuration = configuration.clone();
                        (point, normal)
                    }
                };
                let local = match self.vec3(pt, env) {
                    Ok(p) => Transform::reflection(p, normalize(*n)),
                    Err(msg) => {
                        self.diag(
                            DiagCode::AsmBadPattern,
                            Severity::Error,
                            site.clone(),
                            msg,
                            Vec::new(),
                        );
                        continue;
                    }
                };
                let seg = PathSegment::instance(seed_name, &m.name, InstanceKey::Mirror);
                present.insert(seg.clone());
                if let Some(inst) = self.compose_occurrence(
                    node,
                    doc,
                    env,
                    &occ,
                    seg,
                    pending,
                    NodeOrigin::Mirror {
                        mirror: m.name.clone(),
                    },
                    stack,
                    flexible_member,
                ) {
                    plans.push(InstancePlan {
                        owner: node,
                        instance: inst,
                        seed,
                        left: world,
                        right: local,
                        host: None,
                    });
                }
            }
        }
        plans
    }

    fn declare_envelopes(&mut self, node: NodeId, doc: &AssemblyDocument, env: &ValueEnv) {
        for e in &doc.envelopes {
            let site = Site::with_param(
                Subject::Document,
                &format!("{}#envelope.{}", doc.path, e.name),
            );
            let (min, max) = match (self.vec3(&e.min, env), self.vec3(&e.max, env)) {
                (Ok(a), Ok(b)) => (a, b),
                (Err(m), _) | (_, Err(m)) => {
                    self.diag(
                        DiagCode::ValueFailed,
                        Severity::Error,
                        site,
                        format!("envelope `{}`: {}", e.name, m),
                        Vec::new(),
                    );
                    continue;
                }
            };
            let occupant = match &e.owner {
                None => None,
                Some(o) => {
                    let path = RelPath(o.split('/').map(PathSegment::named).collect());
                    match self.walk(node, &path) {
                        Ok(n) => Some(n),
                        Err(_) => {
                            self.diag(
                                DiagCode::AsmUnknownOccurrence,
                                Severity::Warning,
                                site,
                                format!(
                                    "envelope `{}` is owned by `{}`, which is not active here",
                                    e.name, o
                                ),
                                Vec::new(),
                            );
                            None
                        }
                    }
                }
            };
            let local = crate::assembly::source::Aabb::new(min, max);
            self.envelopes.push(ResolvedEnvelope {
                declared_by: node,
                name: e.name.clone(),
                occupant,
                keep_out: e.keep_out,
                local,
                world: local,
            });
        }
    }

    fn active_occurrence(&mut self, doc: &AssemblyDocument, name: &str) -> Option<Occurrence> {
        let file = doc.path.clone();
        let alts: Vec<Occurrence> = doc.alternatives(name).cloned().collect();
        alts.into_iter().find(|o| {
            let site = Site::with_param(
                Subject::Occurrence {
                    path: name.to_owned(),
                },
                file.as_str(),
            );
            self.holds(o.when.as_ref(), &file, &site)
        })
    }

    fn plane(
        &mut self,
        node: NodeId,
        p: &PlaneSpec,
        env: &ValueEnv,
    ) -> Result<([f64; 3], [f64; 3]), String> {
        match p {
            PlaneSpec::Explicit { point, normal } => {
                Ok((self.vec3(point, env)?, normalize(*normal)))
            }
            PlaneSpec::Frame { frame } => {
                let t = self.frame_ref(node, frame, env)?;
                let f = t.to_frame();
                Ok((f.origin, f.z))
            }
        }
    }

    /// Instance transforms in this assembly's coordinates, or, for a followed pattern, in the
    /// host occurrence's definition coordinates (conjugated by the host's placement when the
    /// plan is applied, after placement and the solve).
    #[allow(clippy::type_complexity)]
    fn pattern_transforms(
        &mut self,
        node: NodeId,
        layout: &OccurrenceLayout,
        env: &ValueEnv,
    ) -> Result<(Vec<(InstanceKey, Transform)>, Option<NodeId>), String> {
        let mut out = Vec::new();
        match layout {
            OccurrenceLayout::Linear {
                direction,
                spacing,
                count,
            } => {
                let d = normalize(*direction);
                let s = self.length(spacing, env)?;
                let n = self.count(count, env)?;
                for k in 1..n {
                    let t = k as f64 * s;
                    out.push((
                        InstanceKey::Linear { step: k },
                        Transform::translation([d[0] * t, d[1] * t, d[2] * t]),
                    ));
                }
            }
            OccurrenceLayout::Grid {
                first,
                first_spacing,
                first_count,
                second,
                second_spacing,
                second_count,
            } => {
                let (d1, d2) = (normalize(*first), normalize(*second));
                let (s1, s2) = (
                    self.length(first_spacing, env)?,
                    self.length(second_spacing, env)?,
                );
                let (n1, n2) = (
                    self.count(first_count, env)?,
                    self.count(second_count, env)?,
                );
                for i in 0..n1 {
                    for j in 0..n2 {
                        if i == 0 && j == 0 {
                            continue;
                        }
                        let (a, b) = (i as f64 * s1, j as f64 * s2);
                        out.push((
                            InstanceKey::Grid {
                                first: i,
                                second: j,
                            },
                            Transform::translation([
                                d1[0] * a + d2[0] * b,
                                d1[1] * a + d2[1] * b,
                                d1[2] * a + d2[2] * b,
                            ]),
                        ));
                    }
                }
            }
            OccurrenceLayout::Circular {
                axis_point,
                axis,
                count,
                angle,
            } => {
                let p = self.vec3(axis_point, env)?;
                let n = self.count(count, env)?;
                let a = self.angle(angle, env)?;
                let full = (a.abs() - std::f64::consts::TAU).abs() < 1e-9;
                let step = if full || n == 1 {
                    a / n as f64
                } else {
                    a / (n - 1) as f64
                };
                for k in 1..n {
                    out.push((
                        InstanceKey::Circular { step: k },
                        Transform::rotation_about(p, *axis, k as f64 * step),
                    ));
                }
            }
            OccurrenceLayout::Table { rows } => {
                let mut seen = BTreeSet::new();
                for r in rows {
                    if !seen.insert(r.key.clone()) {
                        return Err(format!("row key `{}` appears twice", r.key));
                    }
                    out.push((
                        InstanceKey::Row { key: r.key.clone() },
                        self.offset(&r.offset, env)?,
                    ));
                }
            }
            OccurrenceLayout::FollowPattern {
                occurrence,
                publish,
            } => {
                let host = self.walk(node, occurrence).map_err(|i| {
                    format!("no occurrence `{}`", RelPath(occurrence.0[..=i].to_vec()))
                })?;
                let n = self.nodes[host as usize].clone();
                let s = self
                    .configured_summary(&n.definition, n.revision.as_ref(), &n.configuration)
                    .map_err(|e| e.to_string())?;
                let Some(PublishedGeometry::Pattern { frames }) = s.publishes.get(publish) else {
                    return Err(format!(
                        "`{}` publishes no pattern `{}`",
                        n.definition, publish
                    ));
                };
                let mut frames = frames.clone();
                frames.sort_by(|a, b| a.0.cmp(&b.0));
                let Some((_, f0)) = frames.first() else {
                    return Ok((out, None));
                };
                // The host's frame is not final until placement; the transform is applied in
                // host coordinates, so it is conjugated when the plan is applied. Here the host
                // chain is taken as placed (patterns follow after placement and the solve).
                let f0 = Transform::from_frame(f0);
                for (k, f) in frames.iter().skip(1) {
                    out.push((k.clone(), Transform::from_frame(f).compose(&f0.inverse())));
                }
                return Ok((out, Some(host)));
            }
        }
        Ok((out, None))
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
