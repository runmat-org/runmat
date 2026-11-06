//! Regeneration: evaluate values, run the feature list through the kernel, carry names,
//! resolve references, and continue past failures.
//!
//! A failed feature is a root diagnostic; the document passes through it unchanged. A later
//! feature whose references were made by the failed one is a cascade of that root. Features
//! that do not depend on it still regenerate, so the model stays viewable and analysable.

mod references;
mod inputs;
mod compile;
mod apply;
mod sketch_status;
mod profiles;
use profiles::*;

use crate::diagnostics::{
    describe_ref_problem_in, describe_wait_in, Cause, DiagCode, DiagnosticReport, Repair, Severity,
    Site, Subject,
};
use crate::ids::{ConfigurationName, FeatureId, PartName, RegionName};
use crate::metering::{FeatureWork, NoUsageMeter, SolveOutcome, Stopwatch, UsageEvent, UsageMeter};
use crate::model::datum::{add, cross, dot, normalize, scale, sub};
use crate::model::*;
use crate::pmi::{resolve_pmi_with, ResolvedPmi};
use crate::refs::history::{OperandEntity, SurfaceKind, Topology};
use crate::refs::lock::{ItemPath, LockSection};
use crate::refs::naming::{
    cad_names_for_import, edge_name, name_outputs, BodyNames, NamingContext, NamingOutput,
    OperandView, SlotTie,
};
use crate::refs::publish::{
    ForeignEntity, ForeignTarget, PublishTable, PublishedCatalog, XrefProblem,
};
use crate::refs::resolve::{
    BodyView, EntityHandle, FeatureEntry, FeatureTable, MakerStatus, RefProblem, Resolution,
    ResolutionPolicy, ResolutionStats, ResolveNote, ResolveOutcome, Resolved, Resolver, WaitCause,
    WaitReason,
};
use crate::refs::values::{
    published_sites, value_error_diagnostic, value_read_diagnostics, PublishedValues,
};
use crate::refs::{EntityName, InstanceKey, NameTable, Reference, Shown, TopoKind};
use crate::regen::deps::DependencyGraph;
use crate::regen::hash::{chain, CacheKey, KeyInputs};
use crate::regen::kernel::*;
use crate::sketch_solver::SketchSolver;
use crate::units::{Dimension, UnitAuthority};
use crate::value::{DynamicValueEvaluator, Evaluator, ValueEnv, ValueExpr};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod families;
mod features;
mod patterns;
mod sheet;

pub struct RegenContext<'a> {
    pub kernel: &'a dyn CadKernel,
    pub solver: &'a dyn SketchSolver,
    pub units: &'a dyn UnitAuthority,
    pub dynamic: &'a dyn DynamicValueEvaluator,
    pub lock: Option<&'a LockSection>,
    pub catalog: Option<&'a dyn PublishedCatalog>,
    pub policy: ResolutionPolicy,
    /// Body slots stored after the previous regeneration ([`crate::regen::slots`]). When set,
    /// each item naming one body is compared with its stored lineage signature.
    pub slots: Option<&'a crate::regen::slots::SlotTable>,
}

#[derive(Clone, Debug, Default)]
pub struct RegenRequest {
    pub configuration: Option<ConfigurationName>,
}

/// Kernel results by chained key. Shared across regenerations of one document.
#[derive(Clone, Debug, Default)]
pub struct RegenCache {
    pub entries: BTreeMap<CacheKey, KernelOutput>,
    pub hits: u32,
    pub misses: u32,
}

#[derive(Clone, Debug)]
pub struct LiveBody {
    pub handle: BodyHandle,
    pub topology: Topology,
    pub names: BodyNames,
}

#[derive(Clone, Debug, Default)]
pub struct PartState {
    pub bodies: Vec<LiveBody>,
    pub datums: BTreeMap<FeatureId, DatumValue>,
    pub sketches: BTreeMap<FeatureId, (SolvedSketch, Frame)>,
    /// Sketches whose constraints conflict, kept for display only (written coordinates, the
    /// conflict set marked); no feature reads them.
    pub over_defined_sketches: BTreeMap<FeatureId, (SolvedSketch, Frame)>,
    pub burials: BTreeMap<(TopoKind, EntityName), FeatureId>,
    /// Every one-level name this regeneration produced, by token.
    pub names: NameTable,
    /// Tool operations each tool feature applied, for patterns and mirrors of features.
    pub tools: BTreeMap<FeatureId, Vec<ToolOp>>,
    pub table: FeatureTable,
    /// Typed records of geometry-free features and derived data (reference curves, threads,
    /// sheet metal rules and developments, mold analyses), by feature.
    pub records: BTreeMap<FeatureId, crate::regen::records::FeatureRecord>,
    /// The kernel operation of each sketched bend and jog, which a fold of the plate replays
    /// as one of its steps.
    pub replays: BTreeMap<FeatureId, KernelOp>,
}

impl PartState {
    pub fn views(&self) -> Vec<BodyView<'_>> {
        self.bodies
            .iter()
            .map(|b| BodyView {
                names: &b.names,
                topology: &b.topology,
            })
            .collect()
    }

    /// Names of an entity, for keys, the index and foreign targets.
    pub fn entity_names(&self, h: &EntityHandle) -> Vec<EntityName> {
        let b = &self.bodies[h.body as usize];
        match h.kind {
            TopoKind::Face => b.names.faces[h.index as usize].iter().cloned().collect(),
            TopoKind::Body => b.names.body.iter().cloned().collect(),
            TopoKind::Edge => edge_name(&b.names, &b.topology, h.index)
                .map(|p| p.to_vec())
                .unwrap_or_default(),
            TopoKind::Vertex => {
                let mut v: Vec<EntityName> = b
                    .topology
                    .vertex_faces(h.index)
                    .into_iter()
                    .filter_map(|f| b.names.primary(f).cloned())
                    .collect();
                v.sort();
                v
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum FeatureStatus {
    Regenerated,
    CacheHit,
    Frozen,
    Suppressed,
    SuppressedInConfiguration,
    RolledBack,
    Waiting { reason: WaitReason },
    Failed { diagnostic: u32 },
    Blocked { root: FeatureId, diagnostic: u32 },
}

impl FeatureStatus {
    pub fn ran(&self) -> bool {
        matches!(
            self,
            FeatureStatus::Regenerated | FeatureStatus::CacheHit | FeatureStatus::Frozen
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReferenceRecord {
    pub param: String,
    pub reference: Reference,
    pub outcome: ResolveOutcome,
    /// The resolved entities by structured name (empty unless resolved). Merge re-checks
    /// compare these.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entities: Vec<Vec<EntityName>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureOutcome {
    pub id: FeatureId,
    pub kind: String,
    pub status: FeatureStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_key: Option<CacheKey>,
    pub references: Vec<ReferenceRecord>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ItemOutcome {
    pub references: Vec<ReferenceRecord>,
    /// The resolved entities by name.
    pub entities: Vec<Vec<EntityName>>,
    pub ok: bool,
}

#[derive(Clone, Debug)]
pub struct RegenResult {
    pub configuration: Option<ConfigurationName>,
    pub values: ValueEnv,
    pub deps: DependencyGraph,
    pub features: Vec<FeatureOutcome>,
    pub state: PartState,
    pub parts: BTreeMap<PartName, ItemOutcome>,
    pub regions: BTreeMap<RegionName, ItemOutcome>,
    pub materials: Vec<ItemOutcome>,
    /// The resolved publish table, for the project catalog.
    pub publishes: PublishTable,
    /// The part's PMI resolved against this state (empty when the part declares none). Its
    /// diagnostics are also in [`RegenResult::diagnostics`].
    pub pmi: ResolvedPmi,
    pub diagnostics: DiagnosticReport,
    pub stats: ResolutionStats,
    pub final_key: CacheKey,
    /// Stored body slots compared with this regeneration (empty unless
    /// [`RegenContext::slots`] was set).
    pub slots: Vec<crate::regen::slots::SlotRow>,
}

impl RegenResult {
    pub fn feature(&self, id: &str) -> Option<&FeatureOutcome> {
        self.features.iter().find(|f| f.id.0 == id)
    }
    pub fn status(&self, id: &str) -> Option<&FeatureStatus> {
        self.feature(id).map(|f| &f.status)
    }
}

enum Halt {
    Wait(WaitReason),
    Fail(u32),
    Block(FeatureId, u32),
}

// Built once per feature per regeneration; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
enum Compiled {
    /// Geometry-free records (reference curves, cosmetic threads, mold analyses, sheet
    /// metal developments): stored on the state as typed records.
    Record(crate::regen::records::FeatureRecord),
    Datum(DatumValue),
    Sketch(SolvedSketch, Frame),
    Op {
        op: KernelOp,
        operands: Vec<usize>,
        naming: NamingContext,
        import: bool,
        instances: Option<BTreeSet<InstanceKey>>,
        tools: Option<Vec<ToolOp>>,
    },
    Delete(Vec<usize>),
    /// A compound feature: kernel operations run in order, each on the bodies the earlier
    /// steps left. The feature owns every name its steps make (each step names its outputs
    /// with the feature and its own [`NamingContext::step`]), and names carry from step to
    /// step through each step's history exactly as from feature to feature, so the feature's
    /// history is the composition of its steps' histories. Only the last step's bodies are
    /// committed; names an intermediate step made and a later step removed are never seen.
    Steps(Vec<Step>),
}

/// One kernel operation of a compound feature.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct Step {
    pub op: KernelOp,
    /// The bodies the step reads, by position in the body list as the earlier steps left it.
    /// The step's outputs take the place of its first operand.
    pub operands: Vec<usize>,
    #[serde(skip)]
    pub naming: NamingContext,
}

struct Run<'a, 'b> {
    doc: &'a PartDocument,
    ctx: &'a RegenContext<'b>,
    ev: Evaluator<'a>,
    values: &'a ValueEnv,
    st: PartState,
    diags: DiagnosticReport,
    stats: ResolutionStats,
    /// Root diagnostic per failed feature (and per failed value, keyed `input.x` / `let.x`).
    failure_diag: BTreeMap<String, u32>,
    /// A record the feature being compiled makes alongside its operation (a hole feature's
    /// holes), stored when the operation succeeds.
    pending_record: Option<crate::regen::records::FeatureRecord>,
    records: Vec<ReferenceRecord>,
    ref_ids: Vec<(String, Vec<Vec<EntityName>>)>,
    meter: &'a dyn UsageMeter,
}

pub fn regenerate(
    doc: &PartDocument,
    ctx: &RegenContext<'_>,
    req: &RegenRequest,
    cache: &mut RegenCache,
) -> RegenResult {
    regenerate_metered(doc, ctx, req, cache, &NoUsageMeter)
}

/// [`regenerate`], reporting sketch solves, feature regenerations and the part regeneration
/// to `meter`.
pub fn regenerate_metered(
    doc: &PartDocument,
    ctx: &RegenContext<'_>,
    req: &RegenRequest,
    cache: &mut RegenCache,
    meter: &dyn UsageMeter,
) -> RegenResult {
    let mut result = regenerate_reported(doc, ctx, req, cache, meter);
    result.diagnostics.offer_suppress();
    result
}

fn regenerate_reported(
    doc: &PartDocument,
    ctx: &RegenContext<'_>,
    req: &RegenRequest,
    cache: &mut RegenCache,
    meter: &dyn UsageMeter,
) -> RegenResult {
    if doc
        .features
        .iter()
        .any(|f| matches!(f.kind, FeatureKind::Library(_)))
    {
        let (expanded, errors) = doc.expand_library();
        if expanded.features == doc.features && errors.is_empty() {
            return regenerate_unexpanded(doc, ctx, req, cache, meter);
        }
        let mut r = regenerate_unexpanded(&expanded, ctx, req, cache, meter);
        for (f, e) in errors {
            r.diagnostics.push(
                DiagCode::RegenInvalid,
                Severity::Error,
                Site::feature(&f, None),
                e.to_string(),
                Cause::Root,
                Vec::new(),
            );
        }
        return r;
    }
    regenerate_unexpanded(doc, ctx, req, cache, meter)
}

fn regenerate_unexpanded(
    doc: &PartDocument,
    ctx: &RegenContext<'_>,
    req: &RegenRequest,
    cache: &mut RegenCache,
    meter: &dyn UsageMeter,
) -> RegenResult {
    let whole = Stopwatch::start(meter);
    // Published values from other files are read through the catalog, memoised per read.
    let published = PublishedValues::new(ctx.dynamic, ctx.catalog, ctx.lock, &doc.path);
    let ev = Evaluator {
        units: ctx.units,
        dynamic: &published,
    };
    let mut diags = DiagnosticReport::default();
    let config = match &req.configuration {
        Some(name) => {
            let c = doc.configuration(name);
            if c.is_none() {
                diags.push(
                    DiagCode::RegenInvalid,
                    Severity::Error,
                    Site {
                        subject: Subject::Document,
                        param: None,
                    },
                    format!("unknown configuration `{}`", name),
                    Cause::Root,
                    Vec::new(),
                );
            }
            c
        }
        None => None,
    };
    // The configuration's feature parameters, written into a copy of the history: everything
    // below reads one form. A feature it cannot set keeps its own definition, with an error.
    let patched;
    let doc = match config.as_ref().filter(|c| !c.features.is_empty()) {
        Some(c) => {
            let (features, problems) = c.apply_features(&doc.features);
            for problem in problems {
                diags.push(
                    DiagCode::RegenInvalid,
                    Severity::Error,
                    Site {
                        subject: Subject::Configuration {
                            name: c.name.0.clone(),
                        },
                        param: Some(format!("features.{}", problem.feature)),
                    },
                    problem.message,
                    Cause::Root,
                    Vec::new(),
                );
            }
            patched = PartDocument {
                features,
                ..doc.clone()
            };
            &patched
        }
        None => doc,
    };
    if let Some(c) = &config {
        for id in c.suppress.iter().filter(|id| doc.feature(id).is_none()) {
            diags.push(
                DiagCode::RegenInvalid,
                Severity::Error,
                Site {
                    subject: Subject::Configuration {
                        name: c.name.0.clone(),
                    },
                    param: Some("suppress".into()),
                },
                format!(
                    "configuration `{}` suppresses `{id}`, which the part does not have",
                    c.name
                ),
                Cause::Root,
                Vec::new(),
            );
        }
    }
    let overrides = config
        .as_ref()
        .map(|c| c.overrides.clone())
        .unwrap_or_default();
    let values = ev.evaluate_env(&doc.inputs, &doc.lets, &overrides);
    let mut failure_diag = BTreeMap::new();
    for (name, err) in &values.failed {
        let site = Site {
            subject: Subject::Value { name: name.clone() },
            param: None,
        };
        let (code, message, repairs) = value_error_diagnostic(&site, err);
        let id = diags.push(code, Severity::Error, site, message, Cause::Root, repairs);
        failure_diag.insert(name.clone(), id);
    }
    let deps = DependencyGraph::analyse(doc);
    let mut st = PartState::default();
    for (i, f) in doc.features.iter().enumerate() {
        st.table.entries.insert(
            f.id.clone(),
            FeatureEntry {
                index: i,
                status: None,
            },
        );
    }
    let mut run = Run {
        doc,
        ctx,
        ev: Evaluator {
            units: ctx.units,
            dynamic: &published,
        },
        values: &values,
        st,
        diags,
        stats: ResolutionStats::default(),
        failure_diag,
        pending_record: None,
        records: Vec::new(),
        ref_ids: Vec::new(),
        meter,
    };
    let rollback = doc
        .rollback_after
        .as_ref()
        .and_then(|r| doc.feature_index(r));
    let kernel_id = ctx.kernel.id();
    let kernel_str = format!("{}/{}", kernel_id.name, kernel_id.version);
    let env_str = format!("{}|{}", ctx.solver.id(), ctx.units.fingerprint());
    let mut prev = CacheKey::root(
        doc.path.as_str(),
        config.as_ref().map(|c| c.name.0.as_str()),
    );
    let mut outcomes = Vec::new();

    for (i, f) in doc.features.iter().enumerate() {
        run.records.clear();
        run.ref_ids.clear();
        let watch = Stopwatch::start(meter);
        let pre = if rollback.map(|r| i > r).unwrap_or(false) {
            Some((FeatureStatus::RolledBack, MakerStatus::RolledBack))
        } else if f.suppressed {
            Some((FeatureStatus::Suppressed, MakerStatus::Suppressed))
        } else if config
            .as_ref()
            .map(|c| c.suppress.contains(&f.id))
            .unwrap_or(false)
        {
            Some((
                FeatureStatus::SuppressedInConfiguration,
                MakerStatus::SuppressedInConfiguration,
            ))
        } else {
            None
        };
        let (status, key) = if let Some((fs, ms)) = pre {
            run.set_status(&f.id, ms);
            (fs, None)
        } else {
            match run.compile(i, f) {
                Err(h) => (run.halt(&f.id, h), None),
                Ok(compiled) => {
                    let key = {
                        let content: serde_json::Value = match &compiled {
                            Compiled::Datum(d) => {
                                serde_json::to_value(d).expect("datum serializes")
                            }
                            Compiled::Sketch(s, fr) => {
                                serde_json::to_value((&s.cache, fr)).expect("sketch serializes")
                            }
                            Compiled::Op { op, .. } => {
                                serde_json::to_value(op).expect("op serializes")
                            }
                            Compiled::Steps(steps) => {
                                serde_json::to_value(steps).expect("steps serialize")
                            }
                            Compiled::Delete(_) => serde_json::Value::String("delete".into()),
                            Compiled::Record(r) => {
                                serde_json::to_value(r).expect("record serializes")
                            }
                        };
                        chain(KeyInputs {
                            previous: &prev,
                            feature: f.id.as_str(),
                            content: &content,
                            references: &run.ref_ids,
                            kernel: &kernel_str,
                            environment: &env_str,
                        })
                    };
                    match run.apply(i, f, compiled, &key, cache) {
                        Ok(s) => {
                            run.set_status(&f.id, MakerStatus::Ran);
                            prev = key.clone();
                            (s, Some(key))
                        }
                        Err(h) => (run.halt(&f.id, h), Some(key)),
                    }
                }
            }
        };
        meter.record(&UsageEvent::FeatureRegeneration {
            file: doc.path.clone(),
            feature: f.id.clone(),
            kind: f.kind.name().to_owned(),
            work: feature_work(&status),
            duration_nanos: watch.elapsed(meter),
        });
        outcomes.push(FeatureOutcome {
            id: f.id.clone(),
            kind: f.kind.name().to_owned(),
            status,
            cache_key: key,
            references: std::mem::take(&mut run.records),
        });
    }

    let (parts, regions, materials, publishes) = run.items();
    value_read_diagnostics(&published_sites(doc), &published, &mut run.diags);
    let count = |w: FeatureWork| {
        outcomes
            .iter()
            .filter(|o| feature_work(&o.status) == w)
            .count() as u32
    };
    meter.record(&UsageEvent::PartRegeneration {
        file: doc.path.clone(),
        configuration: req.configuration.as_ref().map(|c| c.0.clone()),
        features: outcomes.len() as u32,
        computed: count(FeatureWork::Computed),
        cache_hits: count(FeatureWork::CacheHit),
        failed: count(FeatureWork::Failed),
        duration_nanos: whole.elapsed(meter),
    });
    let mut result = RegenResult {
        configuration: req.configuration.clone(),
        values: values.clone(),
        deps,
        features: outcomes,
        state: run.st,
        parts,
        regions,
        materials,
        publishes,
        pmi: ResolvedPmi::default(),
        diagnostics: run.diags,
        stats: run.stats,
        final_key: prev,
        slots: Vec::new(),
    };
    if !doc.pmi.is_empty() {
        let pmi = resolve_pmi_with(
            doc,
            &doc.pmi,
            &result,
            ctx.units,
            &published,
            ctx.lock,
            ctx.catalog,
            ctx.policy,
        );
        result.diagnostics.append(&pmi.diagnostics);
        result.pmi = pmi;
    }
    if let Some(stored) = ctx.slots {
        let (rows, diags) = crate::regen::slots::compare_slots(stored, doc, &result);
        result.diagnostics.append(&diags);
        result.slots = rows;
    }
    result
}

fn feature_work(s: &FeatureStatus) -> FeatureWork {
    match s {
        FeatureStatus::Regenerated => FeatureWork::Computed,
        FeatureStatus::CacheHit => FeatureWork::CacheHit,
        FeatureStatus::Frozen => FeatureWork::Frozen,
        FeatureStatus::Failed { .. } | FeatureStatus::Blocked { .. } => FeatureWork::Failed,
        FeatureStatus::Suppressed
        | FeatureStatus::SuppressedInConfiguration
        | FeatureStatus::RolledBack
        | FeatureStatus::Waiting { .. } => FeatureWork::Skipped,
    }
}

impl<'a, 'b> Run<'a, 'b> {}
