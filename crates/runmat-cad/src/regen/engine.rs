//! Regeneration: evaluate values, run the feature list through the kernel, carry names,
//! resolve references, and continue past failures.
//!
//! A failed feature is a root diagnostic; the document passes through it unchanged. A later
//! feature whose references were made by the failed one is a cascade of that root. Features
//! that do not depend on it still regenerate, so the model stays viewable and analysable.

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
    cad_names_for_import, edge_name, name_outputs, BodyNames, NamingContext, OperandView, SlotTie,
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
    pub burials: BTreeMap<(TopoKind, EntityName), FeatureId>,
    /// Every one-level name this regeneration produced, by token.
    pub names: NameTable,
    /// Tool operations each tool feature applied, for patterns and mirrors of features.
    pub tools: BTreeMap<FeatureId, Vec<ToolOp>>,
    pub table: FeatureTable,
    /// Typed records of geometry-free features and derived data (reference curves, threads,
    /// sheet metal rules and developments, mold analyses), by feature.
    pub records: BTreeMap<FeatureId, crate::regen::records::FeatureRecord>,
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

impl<'a, 'b> Run<'a, 'b> {
    fn set_status(&mut self, f: &FeatureId, s: MakerStatus) {
        if let Some(e) = self.st.table.entries.get_mut(f) {
            e.status = Some(s);
        }
    }

    fn halt(&mut self, f: &FeatureId, h: Halt) -> FeatureStatus {
        match h {
            Halt::Wait(reason) => {
                let root = match &reason.cause {
                    WaitCause::MakerWaiting { root } => root.clone(),
                    _ => reason.maker.clone(),
                };
                self.set_status(f, MakerStatus::Waiting { root });
                FeatureStatus::Waiting { reason }
            }
            Halt::Fail(d) => {
                self.failure_diag.insert(format!("feature.{}", f), d);
                self.set_status(f, MakerStatus::Failed);
                FeatureStatus::Failed { diagnostic: d }
            }
            Halt::Block(root, d) => {
                self.failure_diag.insert(format!("feature.{}", f), d);
                self.set_status(f, MakerStatus::Blocked { root: root.clone() });
                FeatureStatus::Blocked {
                    root,
                    diagnostic: d,
                }
            }
        }
    }

    fn root_of(&self, root: &FeatureId) -> u32 {
        // A cascade points at the root diagnostic, following cascades back.
        let id = self
            .failure_diag
            .get(&format!("feature.{}", root))
            .copied()
            .unwrap_or(0);
        match self.diags.diagnostics.get(id as usize).map(|d| d.cause) {
            Some(Cause::Cascade { root }) => root,
            _ => id,
        }
    }

    fn error(
        &mut self,
        f: &FeatureId,
        param: Option<&str>,
        code: DiagCode,
        message: String,
        repairs: Vec<Repair>,
    ) -> Halt {
        Halt::Fail(self.diags.push(
            code,
            Severity::Error,
            Site::feature(f, param),
            message,
            Cause::Root,
            repairs,
        ))
    }

    fn resolver(&self, consumer: Option<usize>) -> Resolver<'_> {
        Resolver {
            file: &self.doc.path,
            bodies: self.st.views(),
            features: &self.st.table,
            consumer,
            burials: &self.st.burials,
            publishes: &self.doc.publishes,
            lock: self.ctx.lock,
            catalog: self.ctx.catalog,
            policy: self.ctx.policy,
            names: Some(&self.st.names),
        }
    }

    /// Record an outcome, emit its diagnostics, and turn it into a resolution or a halt.
    fn take(
        &mut self,
        site: Site,
        reference: &Reference,
        outcome: ResolveOutcome,
    ) -> Result<Resolution, Halt> {
        self.stats.record(&outcome);
        self.records.push(ReferenceRecord {
            param: site.param.clone().unwrap_or_default(),
            reference: reference.clone(),
            outcome: outcome.clone(),
            entities: Vec::new(),
        });
        match outcome {
            ResolveOutcome::Resolved(r) => {
                for n in &r.notes {
                    self.note(&site, reference, n);
                }
                let names: Vec<Vec<EntityName>> = r
                    .entities()
                    .iter()
                    .map(|h| self.st.entity_names(h))
                    .collect();
                let names = match &r.target {
                    Resolved::Foreign { target } => {
                        target.entities.iter().map(|e| e.names.clone()).collect()
                    }
                    _ => names,
                };
                if let Some(rec) = self.records.last_mut() {
                    rec.entities = names.clone();
                }
                self.ref_ids
                    .push((site.param.clone().unwrap_or_default(), names));
                Ok(r)
            }
            ResolveOutcome::Waiting(w) => {
                let (msg, repairs) = describe_wait_in(reference, &w, Some(&self.st.names));
                self.diags.push(
                    DiagCode::RefWaiting,
                    Severity::Info,
                    site,
                    msg,
                    Cause::Root,
                    repairs,
                );
                Err(Halt::Wait(w))
            }
            ResolveOutcome::Failed(RefProblem::Blocked { maker, root }) => {
                let root_diag = self.root_of(&root);
                let (code, msg, repairs) = describe_ref_problem_in(
                    &site,
                    reference,
                    &RefProblem::Blocked {
                        maker,
                        root: root.clone(),
                    },
                    Some(&self.st.names),
                );
                let d = self.diags.push(
                    code,
                    Severity::Error,
                    site,
                    msg,
                    Cause::Cascade { root: root_diag },
                    repairs,
                );
                Err(Halt::Block(root, d))
            }
            ResolveOutcome::Failed(p) => {
                let (code, msg, repairs) =
                    describe_ref_problem_in(&site, reference, &p, Some(&self.st.names));
                Err(Halt::Fail(self.diags.push(
                    code,
                    Severity::Error,
                    site,
                    msg,
                    Cause::Root,
                    repairs,
                )))
            }
        }
    }

    fn note(&mut self, site: &Site, reference: &Reference, n: &ResolveNote) {
        match n {
            ResolveNote::ViaAlias {
                from,
                to,
                rewritten,
            } => {
                self.diags.push(
                    DiagCode::RefViaAlias,
                    Severity::Info,
                    site.clone(),
                    format!(
                        "`{}` was renamed to `{}`; resolved through the lock alias",
                        from, to
                    ),
                    Cause::Root,
                    vec![Repair::RewriteReference {
                        site: site.clone(),
                        reference: rewritten.clone(),
                    }],
                );
            }
            ResolveNote::FallbackUsed { maker } => {
                self.diags.push(
                    DiagCode::RefFallbackUsed,
                    Severity::Warning,
                    site.clone(),
                    format!(
                        "`{}` resolved by geometry among `{}`'s entities, not by name; check it",
                        reference, maker
                    ),
                    Cause::Root,
                    vec![Repair::RePick {
                        site: site.clone(),
                        candidates: Vec::new(),
                    }],
                );
            }
            ResolveNote::PinBehind {
                file,
                pinned,
                latest,
            } => {
                self.diags.push(
                    DiagCode::XrefPinBehind,
                    Severity::Info,
                    site.clone(),
                    format!("`{}` is pinned to {}; {} is released", file, pinned, latest),
                    Cause::Root,
                    vec![Repair::UpdatePin {
                        site: site.clone(),
                        file: file.clone(),
                        from: pinned.clone(),
                        to: latest.clone(),
                    }],
                );
            }
            ResolveNote::Dormant { waiting } => {
                let makers: Vec<String> = waiting.iter().map(|w| w.maker.0.clone()).collect();
                self.diags.push(
                    DiagCode::RefDormant,
                    Severity::Info,
                    site.clone(),
                    format!(
                        "{} member(s) of `{}` are waiting on {}; the rest apply",
                        waiting.len(),
                        site.key(),
                        makers.join(", ")
                    ),
                    Cause::Root,
                    makers
                        .into_iter()
                        .map(|m| Repair::Unsuppress {
                            feature: FeatureId::new(m),
                        })
                        .collect(),
                );
            }
            ResolveNote::XrefRenamed {
                from_file,
                to_file,
                from_name,
                to_name,
                rewritten,
            } => {
                let what = match (from_file == to_file, from_name == to_name) {
                    (false, false) => format!(
                        "`{}` → `{}` and `{}` → `{}`",
                        from_file, to_file, from_name, to_name
                    ),
                    (false, true) => format!("`{}` → `{}`", from_file, to_file),
                    _ => format!("`{}` → `{}` in `{}`", from_name, to_name, to_file),
                };
                self.diags.push(
                    DiagCode::XrefViaAlias,
                    Severity::Info,
                    site.clone(),
                    format!("renamed ({}); resolved through the lock alias", what),
                    Cause::Root,
                    vec![Repair::RewriteReference {
                        site: site.clone(),
                        reference: rewritten.clone(),
                    }],
                );
            }
            ResolveNote::Merged { .. } | ResolveNote::Pieces { .. } => {}
        }
    }

    fn single(
        &mut self,
        i: usize,
        f: &FeatureId,
        param: &str,
        r: &Reference,
        kind: Option<TopoKind>,
    ) -> Result<Resolution, Halt> {
        let o = self
            .resolver(Some(i))
            .resolve(r, kind, Cardinality::ExactlyOne);
        self.take(Site::feature(f, Some(param)), r, o)
    }

    fn list(
        &mut self,
        i: usize,
        f: &FeatureId,
        param: &str,
        refs: &[Reference],
        kind: Option<TopoKind>,
    ) -> Result<BTreeSet<EntityHandle>, Halt> {
        let (each, combined) = self.resolver(Some(i)).resolve_list(refs, kind);
        // Record and diagnose per member (waiting members are dormant, not halts).
        let mut set = BTreeSet::new();
        let mut first_halt = None;
        let mut waiting = Vec::new();
        for (k, (r, o)) in refs.iter().zip(each).enumerate() {
            let site = Site::feature(f, Some(&format!("{}[{}]", param, k)));
            match o {
                ResolveOutcome::Waiting(w) => {
                    self.stats.record(&ResolveOutcome::Waiting(w.clone()));
                    self.records.push(ReferenceRecord {
                        param: site.param.clone().unwrap_or_default(),
                        reference: r.clone(),
                        outcome: ResolveOutcome::Waiting(w.clone()),
                        entities: Vec::new(),
                    });
                    waiting.push(w);
                }
                other => match self.take(site, r, other) {
                    Ok(res) => set.extend(res.entities()),
                    Err(h) => {
                        if first_halt.is_none() {
                            first_halt = Some(h);
                        }
                    }
                },
            }
        }
        if let Some(h) = first_halt {
            return Err(h);
        }
        if let ResolveOutcome::Waiting(w) = combined {
            let (msg, repairs) = describe_wait_in(&refs[0], &w, Some(&self.st.names));
            self.diags.push(
                DiagCode::RefWaiting,
                Severity::Info,
                Site::feature(f, Some(param)),
                msg,
                Cause::Root,
                repairs,
            );
            return Err(Halt::Wait(w));
        }
        if !waiting.is_empty() {
            self.note(
                &Site::feature(f, Some(param)),
                &refs[0],
                &ResolveNote::Dormant { waiting },
            );
        }
        Ok(set)
    }

    /// Gate a feature named directly (a profile sketch, a datum, a seed).
    fn gate_feature(
        &mut self,
        i: usize,
        f: &FeatureId,
        dep: &FeatureId,
        param: &str,
    ) -> Result<(), Halt> {
        let entry = self.st.table.entries.get(dep).cloned();
        let site = Site::feature(f, Some(param));
        let Some(entry) = entry else {
            return Err(self.error(
                f,
                Some(param),
                DiagCode::RefMakerMissing,
                format!("`{}` is not in the file", dep),
                Vec::new(),
            ));
        };
        if entry.index >= i {
            return Err(self.error(
                f,
                Some(param),
                DiagCode::RefForward,
                format!("`{}` comes later in the history", dep),
                vec![Repair::Reorder {
                    feature: f.clone(),
                    after: dep.clone(),
                }],
            ));
        }
        let wait = |cause| WaitReason {
            maker: dep.clone(),
            cause,
        };
        let w = match entry.status {
            Some(MakerStatus::Ran) => return Ok(()),
            Some(MakerStatus::Suppressed) => wait(WaitCause::Suppressed),
            Some(MakerStatus::SuppressedInConfiguration) => {
                wait(WaitCause::SuppressedInConfiguration)
            }
            Some(MakerStatus::RolledBack) => wait(WaitCause::RolledBack),
            Some(MakerStatus::Waiting { root }) => wait(WaitCause::MakerWaiting { root }),
            Some(MakerStatus::Failed) | Some(MakerStatus::Blocked { .. }) | None => {
                let root = match entry.status {
                    Some(MakerStatus::Blocked { root }) => root,
                    _ => dep.clone(),
                };
                let root_diag = self.root_of(&root);
                let d = self.diags.push(
                    DiagCode::RegenCascade,
                    Severity::Error,
                    site,
                    format!("`{}` needs `{}`, which did not regenerate", f, dep),
                    Cause::Cascade { root: root_diag },
                    Vec::new(),
                );
                return Err(Halt::Block(root, d));
            }
        };
        let r = Reference::feature_body(dep.as_str());
        let (msg, repairs) = describe_wait_in(&r, &w, Some(&self.st.names));
        self.diags.push(
            DiagCode::RefWaiting,
            Severity::Info,
            site,
            msg,
            Cause::Root,
            repairs,
        );
        Err(Halt::Wait(w))
    }

    fn value(
        &mut self,
        f: &FeatureId,
        param: &str,
        v: &ValueExpr,
        dim: Dimension,
    ) -> Result<f64, Halt> {
        let mut ins = BTreeSet::new();
        let mut ls = BTreeSet::new();
        v.collect_names(&mut ins, &mut ls);
        let failed: Option<String> = ins
            .iter()
            .map(|n| format!("input.{}", n))
            .chain(ls.iter().map(|n| format!("let.{}", n)))
            .find(|k| self.values.failed.contains_key(k));
        if let Some(k) = failed {
            let root = self.failure_diag.get(&k).copied().unwrap_or(0);
            let d = self.diags.push(
                DiagCode::RegenCascade,
                Severity::Error,
                Site::feature(f, Some(param)),
                format!("`{}` reads `{}`, which failed to evaluate", f, k),
                Cause::Cascade { root },
                Vec::new(),
            );
            return Err(Halt::Block(f.clone(), d));
        }
        self.ev.eval_as(v, self.values, dim).map_err(|e| {
            let (code, message, repairs) =
                value_error_diagnostic(&Site::feature(f, Some(param)), &e);
            self.error(f, Some(param), code, message, repairs)
        })
    }

    fn count(&mut self, f: &FeatureId, param: &str, v: &ValueExpr) -> Result<u32, Halt> {
        let n = self.value(f, param, v, Dimension::DIMENSIONLESS)?;
        if n < 1.0 || n.fract() != 0.0 || n > 10_000.0 {
            return Err(self.error(
                f,
                Some(param),
                DiagCode::ValueFailed,
                format!("count must be a whole number from 1, got {}", n),
                Vec::new(),
            ));
        }
        Ok(n as u32)
    }

    // ------------------------------------------------------------- geometry from references

    fn face_frame(&self, h: &EntityHandle) -> Option<Frame> {
        let g = &self.st.bodies[h.body as usize].topology.faces[h.index as usize].geom;
        let SurfaceKind::Plane { normal } = g.surface else {
            return None;
        };
        let z = normalize(normal);
        let origin = scale(z, dot(z, g.centroid));
        let seed = if dot(z, [1.0, 0.0, 0.0]).abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let x = normalize(sub(seed, scale(z, dot(seed, z))));
        let y = cross(z, x);
        Some(Frame { origin, x, y, z })
    }

    fn datum_frame(
        &mut self,
        i: usize,
        f: &FeatureId,
        param: &str,
        d: &DatumRef,
    ) -> Result<Frame, Halt> {
        match d {
            DatumRef::OriginPlane { plane } => Ok(Frame::origin_plane(*plane)),
            DatumRef::OriginFrame => Ok(Frame::WORLD),
            DatumRef::Feature { feature } => {
                self.gate_feature(i, f, feature, param)?;
                match self.st.datums.get(feature) {
                    Some(DatumValue::Plane { frame }) | Some(DatumValue::Frame { frame }) => {
                        Ok(*frame)
                    }
                    _ => Err(self.error(
                        f,
                        Some(param),
                        DiagCode::RefKindMismatch,
                        format!("`{}` is not a plane", feature),
                        Vec::new(),
                    )),
                }
            }
            other => Err(self.error(
                f,
                Some(param),
                DiagCode::RefKindMismatch,
                format!("{} is not a plane", other),
                Vec::new(),
            )),
        }
    }

    fn plane(&mut self, i: usize, f: &FeatureId, param: &str, p: &PlaneRef) -> Result<Frame, Halt> {
        match p {
            PlaneRef::Datum { datum } => self.datum_frame(i, f, param, datum),
            PlaneRef::Face { face } => {
                let r = self.single(i, f, param, face, Some(TopoKind::Face))?;
                let h = *r.entities().iter().next().ok_or_else(|| {
                    self.error(
                        f,
                        Some(param),
                        DiagCode::RefEmpty,
                        "no face".into(),
                        Vec::new(),
                    )
                })?;
                self.face_frame(&h).ok_or_else(|| {
                    self.error(
                        f,
                        Some(param),
                        DiagCode::RefKindMismatch,
                        format!("`{}` is not planar", face),
                        Vec::new(),
                    )
                })
            }
        }
    }

    fn axis(
        &mut self,
        i: usize,
        f: &FeatureId,
        param: &str,
        a: &AxisRef,
    ) -> Result<([f64; 3], [f64; 3]), Halt> {
        match a {
            AxisRef::Datum { datum } => match datum {
                DatumRef::OriginAxis { axis } => Ok((
                    [0.0; 3],
                    match axis {
                        OriginAxis::X => [1.0, 0.0, 0.0],
                        OriginAxis::Y => [0.0, 1.0, 0.0],
                        OriginAxis::Z => [0.0, 0.0, 1.0],
                    },
                )),
                DatumRef::Feature { feature } => {
                    self.gate_feature(i, f, feature, param)?;
                    match self.st.datums.get(feature) {
                        Some(DatumValue::Axis { point, direction }) => Ok((*point, *direction)),
                        Some(DatumValue::Frame { frame }) => Ok((frame.origin, frame.z)),
                        _ => Err(self.error(
                            f,
                            Some(param),
                            DiagCode::RefKindMismatch,
                            format!("`{}` is not an axis", feature),
                            Vec::new(),
                        )),
                    }
                }
                other => Err(self.error(
                    f,
                    Some(param),
                    DiagCode::RefKindMismatch,
                    format!("{} is not an axis", other),
                    Vec::new(),
                )),
            },
            AxisRef::Edge { edge } => {
                let r = self.single(i, f, param, edge, Some(TopoKind::Edge))?;
                let h = *r.entities().iter().next().expect("exactly one");
                let g = self.st.bodies[h.body as usize].topology.edges[h.index as usize].geom;
                Ok((g.start, normalize(sub(g.end, g.start))))
            }
        }
    }

    fn point(
        &mut self,
        i: usize,
        f: &FeatureId,
        param: &str,
        p: &PointRef,
    ) -> Result<[f64; 3], Halt> {
        match p {
            PointRef::Datum { datum } => match datum {
                DatumRef::OriginPoint | DatumRef::OriginFrame => Ok([0.0; 3]),
                DatumRef::Feature { feature } => {
                    self.gate_feature(i, f, feature, param)?;
                    match self.st.datums.get(feature) {
                        Some(DatumValue::Point { at }) => Ok(*at),
                        Some(DatumValue::Frame { frame }) | Some(DatumValue::Plane { frame }) => {
                            Ok(frame.origin)
                        }
                        Some(DatumValue::Axis { point, .. }) => Ok(*point),
                        None => Err(self.error(
                            f,
                            Some(param),
                            DiagCode::RefKindMismatch,
                            format!("`{}` is not a point", feature),
                            Vec::new(),
                        )),
                    }
                }
                other => Err(self.error(
                    f,
                    Some(param),
                    DiagCode::RefKindMismatch,
                    format!("{} is not a point", other),
                    Vec::new(),
                )),
            },
            PointRef::Coords { x, y, z } => Ok([
                self.value(f, param, x, Dimension::LENGTH)?,
                self.value(f, param, y, Dimension::LENGTH)?,
                self.value(f, param, z, Dimension::LENGTH)?,
            ]),
            PointRef::Vertex { vertex } => {
                let r = self.single(i, f, param, vertex, Some(TopoKind::Vertex))?;
                let h = *r.entities().iter().next().expect("exactly one");
                Ok(self.st.bodies[h.body as usize].topology.vertices[h.index as usize].point)
            }
            PointRef::Sketch {
                sketch,
                entity,
                role,
            } => {
                self.gate_feature(i, f, sketch, param)?;
                let Some((solved, frame)) = self.st.sketches.get(sketch) else {
                    return Err(self.error(
                        f,
                        Some(param),
                        DiagCode::RefKindMismatch,
                        format!("`{}` is not a sketch", sketch),
                        Vec::new(),
                    ));
                };
                match solved.point(entity, *role) {
                    Some(p) => Ok(frame.to_world(p)),
                    None => Err(self.error(
                        f,
                        Some(param),
                        DiagCode::RefNotGenerated,
                        format!("sketch `{}` has no point `{}.{:?}`", sketch, entity, role),
                        Vec::new(),
                    )),
                }
            }
        }
    }

    fn profile(
        &mut self,
        i: usize,
        f: &FeatureId,
        param: &str,
        p: &ProfileRef,
    ) -> Result<CanonicalProfile, Halt> {
        self.gate_feature(i, f, &p.sketch, param)?;
        let Some((solved, frame)) = self.st.sketches.get(&p.sketch).cloned() else {
            return Err(self.error(
                f,
                Some(param),
                DiagCode::RefKindMismatch,
                format!("`{}` is not a sketch", p.sketch),
                Vec::new(),
            ));
        };
        let Some(sketch_feature) = self.doc.feature(&p.sketch) else {
            unreachable!("gated")
        };
        let FeatureKind::Sketch(sketch) = &sketch_feature.kind else {
            unreachable!("sketch state")
        };
        match build_loops(sketch, &solved, &p.loops) {
            Ok(loops) => Ok(CanonicalProfile {
                frame,
                loops,
                merge_collinear: p.merge_collinear,
            }),
            Err(msg) => {
                Err(self.error(f, Some(param), DiagCode::SketchOpenProfile, msg, Vec::new()))
            }
        }
    }

    /// A sweep path: the sketch's curves chained end to end (open or closed), as a polyline in
    /// part coordinates plus the path curve each segment samples.
    fn path(
        &mut self,
        i: usize,
        f: &FeatureId,
        param: &str,
        p: &ProfileRef,
    ) -> Result<(Vec<[f64; 3]>, Vec<u32>), Halt> {
        self.gate_feature(i, f, &p.sketch, param)?;
        let Some((solved, frame)) = self.st.sketches.get(&p.sketch).cloned() else {
            return Err(self.error(
                f,
                Some(param),
                DiagCode::RefKindMismatch,
                format!("`{}` is not a sketch", p.sketch),
                Vec::new(),
            ));
        };
        let Some(FeatureKind::Sketch(sketch)) = self.doc.feature(&p.sketch).map(|x| &x.kind) else {
            unreachable!("gated")
        };
        match build_path(sketch, &solved) {
            Ok(segs) => {
                let mut pts = Vec::new();
                let mut curves = Vec::new();
                for (ci, seg) in segs.iter().enumerate() {
                    let samples = segment_samples(seg);
                    if pts.is_empty() {
                        pts.push(frame.to_world(samples[0]));
                    }
                    for q in &samples[1..] {
                        pts.push(frame.to_world(*q));
                        curves.push(ci as u32);
                    }
                }
                Ok((pts, curves))
            }
            Err(msg) => {
                Err(self.error(f, Some(param), DiagCode::SketchOpenProfile, msg, Vec::new()))
            }
        }
    }

    fn bodies_of(set: &BTreeSet<EntityHandle>) -> Vec<usize> {
        let mut v: Vec<usize> = set.iter().map(|h| h.body as usize).collect();
        v.sort();
        v.dedup();
        v
    }

    fn operand_entities(set: &BTreeSet<EntityHandle>, operands: &[usize]) -> Vec<OperandEntity> {
        set.iter()
            .map(|h| OperandEntity {
                operand: operands
                    .iter()
                    .position(|b| *b == h.body as usize)
                    .expect("operand listed") as u32,
                entity: h.index,
            })
            .collect()
    }

    fn mode_operands(
        &mut self,
        i: usize,
        f: &FeatureId,
        mode: &BodyMode,
    ) -> Result<(Combine, Vec<usize>), Halt> {
        let combine = match mode {
            BodyMode::New => return Ok((Combine::New, Vec::new())),
            BodyMode::Add { .. } => Combine::Add,
            BodyMode::Cut { .. } => Combine::Cut,
            BodyMode::Intersect { .. } => Combine::Intersect,
        };
        let targets = mode.targets();
        if targets.is_empty() {
            return self.implicit_bodies(i, f).map(|b| (combine, b));
        }
        let set = self.list(i, f, "targets", targets, Some(TopoKind::Body))?;
        Ok((combine, Self::bodies_of(&set)))
    }

    /// Every body, for a feature with implicit targets. With none present, the feature waits on
    /// (or cascades from) the earlier body feature that did not run; with none at all it fails.
    fn implicit_bodies(&mut self, i: usize, f: &FeatureId) -> Result<Vec<usize>, Halt> {
        if !self.st.bodies.is_empty() {
            return Ok((0..self.st.bodies.len()).collect());
        }
        let earlier: Vec<FeatureId> = self.doc.features[..i]
            .iter()
            .filter(|x| x.kind.is_body_feature())
            .filter(|x| {
                !matches!(
                    self.st
                        .table
                        .entries
                        .get(&x.id)
                        .and_then(|e| e.status.clone()),
                    Some(MakerStatus::Ran)
                )
            })
            .map(|x| x.id.clone())
            .collect();
        if let Some(dep) = earlier.first() {
            self.gate_feature(i, f, dep, "targets")?;
        }
        Err(self.error(
            f,
            Some("targets"),
            DiagCode::RefEmpty,
            format!("`{}` has no body to act on", f),
            Vec::new(),
        ))
    }

    fn compile(&mut self, i: usize, f: &Feature) -> Result<Compiled, Halt> {
        let id = &f.id;
        // Any parameter reading a failed value makes the whole feature a cascade of it (sketch
        // coordinates are evaluated by the solver, so check here rather than per value).
        for v in f.kind.values() {
            let mut ins = std::collections::BTreeSet::new();
            let mut ls = std::collections::BTreeSet::new();
            v.collect_names(&mut ins, &mut ls);
            if ins
                .iter()
                .map(|n| format!("input.{}", n))
                .chain(ls.iter().map(|n| format!("let.{}", n)))
                .any(|k| self.values.failed.contains_key(&k))
            {
                self.value(id, "values", v, Dimension::DIMENSIONLESS)?;
            }
        }
        if let Some(c) = self.compile_family(i, f)? {
            return Ok(c);
        }
        let simple = |profiles: Vec<FeatureId>| NamingContext::simple(id, profiles);
        match &f.kind {
            FeatureKind::Sketch(s) => {
                let frame = self.plane(i, id, "on", &s.plane)?;
                let watch = Stopwatch::start(self.meter);
                let solved = self.ctx.solver.solve(s, &self.ev, self.values);
                self.meter.record(&UsageEvent::SketchSolve {
                    file: self.doc.path.clone(),
                    feature: id.clone(),
                    entities: s.entities.len() as u32,
                    constraints: s.constraints.len() as u32,
                    dof: solved.as_ref().ok().and_then(|x| x.dof),
                    outcome: match &solved {
                        Err(_) => SolveOutcome::Failed,
                        Ok(x) => match x.status {
                            SketchSolveStatus::WellDefined => SolveOutcome::WellDefined,
                            SketchSolveStatus::UnderDefined => SolveOutcome::UnderDefined,
                            SketchSolveStatus::OverDefined => SolveOutcome::OverDefined,
                            SketchSolveStatus::Explicit => SolveOutcome::Explicit,
                        },
                    },
                    duration_nanos: watch.elapsed(self.meter),
                });
                match solved {
                    Ok(solved) => {
                        if !solved.redundant.is_empty() {
                            self.diags.push(
                                DiagCode::SketchRedundant,
                                Severity::Warning,
                                Site::feature(id, Some("constraints")),
                                format!("constraints {:?} are implied by others", solved.redundant),
                                Cause::Root,
                                Vec::new(),
                            );
                        }
                        Ok(Compiled::Sketch(solved, frame))
                    }
                    Err(e) => Err(self.error(
                        id,
                        Some("constraints"),
                        DiagCode::SketchSolveFailed,
                        e.message,
                        Vec::new(),
                    )),
                }
            }
            FeatureKind::Plane(p) => {
                let frame = match p {
                    PlaneDef::Offset { base, distance } => {
                        let b = self.plane(i, id, "base", base)?;
                        let d = self.value(id, "distance", distance, Dimension::LENGTH)?;
                        b.offset(d)
                    }
                    PlaneDef::Angle { base, axis, angle } => {
                        let b = self.plane(i, id, "base", base)?;
                        let (pt, dir) = self.axis(i, id, "axis", axis)?;
                        let a = self.value(id, "angle", angle, Dimension::ANGLE)?;
                        let rot = |v: [f64; 3]| rotate(v, dir, a);
                        Frame {
                            origin: add(pt, rot(sub(b.origin, pt))),
                            x: rot(b.x),
                            y: rot(b.y),
                            z: rot(b.z),
                        }
                    }
                    _ => unreachable!("handled by compile_family"),
                };
                Ok(Compiled::Datum(DatumValue::Plane { frame }))
            }
            FeatureKind::Axis(a) => {
                let (point, direction) = match a {
                    AxisDef::Through { a, b } => {
                        let pa = self.point(i, id, "a", a)?;
                        let pb = self.point(i, id, "b", b)?;
                        (pa, normalize(sub(pb, pa)))
                    }
                    AxisDef::OfEdge { edge } => {
                        self.axis(i, id, "edge", &AxisRef::Edge { edge: edge.clone() })?
                    }
                    AxisDef::OfFace { face } => {
                        let r = self.single(i, id, "face", face, Some(TopoKind::Face))?;
                        let h = *r.entities().iter().next().expect("one");
                        match self.st.bodies[h.body as usize].topology.faces[h.index as usize]
                            .geom
                            .surface
                        {
                            SurfaceKind::Cylinder {
                                axis_point,
                                axis_direction,
                                ..
                            } => (axis_point, axis_direction),
                            _ => {
                                return Err(self.error(
                                    id,
                                    Some("face"),
                                    DiagCode::RefKindMismatch,
                                    format!("`{}` is not cylindrical", face),
                                    Vec::new(),
                                ))
                            }
                        }
                    }
                    _ => unreachable!("handled by compile_family"),
                };
                Ok(Compiled::Datum(DatumValue::Axis { point, direction }))
            }
            FeatureKind::Point(p) => {
                let at = match p {
                    PointDef::At { at } => self.point(i, id, "at", at)?,
                    PointDef::Offset { base, dx, dy, dz } => {
                        let b = self.point(i, id, "base", base)?;
                        add(
                            b,
                            [
                                self.value(id, "dx", dx, Dimension::LENGTH)?,
                                self.value(id, "dy", dy, Dimension::LENGTH)?,
                                self.value(id, "dz", dz, Dimension::LENGTH)?,
                            ],
                        )
                    }
                    _ => unreachable!("handled by compile_family"),
                };
                Ok(Compiled::Datum(DatumValue::Point { at }))
            }
            FeatureKind::Frame(fr) => {
                let frame = match fr {
                    FrameDef::At { origin, z, x } => {
                        let o = self.point(i, id, "origin", origin)?;
                        let (_, zd) = self.axis(i, id, "z", z)?;
                        let xd = match x {
                            Some(x) => self.axis(i, id, "x", x)?.1,
                            None => {
                                if dot(zd, [1.0, 0.0, 0.0]).abs() < 0.9 {
                                    [1.0, 0.0, 0.0]
                                } else {
                                    [0.0, 1.0, 0.0]
                                }
                            }
                        };
                        let xd = normalize(sub(xd, scale(zd, dot(xd, zd))));
                        Frame {
                            origin: o,
                            x: xd,
                            y: cross(zd, xd),
                            z: zd,
                        }
                    }
                    FrameDef::OnFace { face } => {
                        self.plane(i, id, "face", &PlaneRef::Face { face: face.clone() })?
                    }
                };
                Ok(Compiled::Datum(DatumValue::Frame { frame }))
            }
            FeatureKind::Extrude(e) => {
                let profile = self.profile(i, id, "profile", &e.profile)?;
                let sign = if e.reverse { -1.0 } else { 1.0 };
                let (start, end, through_all) = match &e.extent {
                    Extent::Blind { distance } => {
                        let d = self.value(id, "distance", distance, Dimension::LENGTH)?;
                        (0.0, sign * d, false)
                    }
                    Extent::Symmetric { distance } => {
                        let d = self.value(id, "distance", distance, Dimension::LENGTH)?;
                        (-d / 2.0, d / 2.0, false)
                    }
                    Extent::ThroughAll => (0.0, sign, true),
                    Extent::UpTo { face } => {
                        let r = self.single(i, id, "up_to", face, Some(TopoKind::Face))?;
                        let h = *r.entities().iter().next().expect("one");
                        let c = self.st.bodies[h.body as usize].topology.faces[h.index as usize]
                            .geom
                            .centroid;
                        let d = dot(sub(c, profile.frame.origin), profile.frame.z);
                        (0.0, d, false)
                    }
                };
                let (combine, operands) = self.mode_operands(i, id, &e.mode)?;
                let tool = ToolOp {
                    shape: ToolShape::Prism {
                        profile,
                        start,
                        end,
                        through_all,
                    },
                    combine,
                };
                Ok(Compiled::Op {
                    op: KernelOp::Tool(tool.clone()),
                    operands,
                    naming: simple(vec![e.profile.sketch.clone()]),
                    import: false,
                    instances: None,
                    tools: Some(vec![tool]),
                })
            }
            FeatureKind::Revolve(r) => {
                let profile = self.profile(i, id, "profile", &r.profile)?;
                let (axis_point, axis_direction) = self.axis(i, id, "axis", &r.axis)?;
                let angle = self.value(id, "angle", &r.angle, Dimension::ANGLE)?;
                let (combine, operands) = self.mode_operands(i, id, &r.mode)?;
                let tool = ToolOp {
                    shape: ToolShape::Revolve {
                        profile,
                        axis_point,
                        axis_direction,
                        angle,
                    },
                    combine,
                };
                Ok(Compiled::Op {
                    op: KernelOp::Tool(tool.clone()),
                    operands,
                    naming: simple(vec![r.profile.sketch.clone()]),
                    import: false,
                    instances: None,
                    tools: Some(vec![tool]),
                })
            }
            FeatureKind::Sweep(s) => {
                let profile = self.profile(i, id, "profile", &s.profile)?;
                let (path, path_curves) = self.path(i, id, "path", &s.path)?;
                let (combine, operands) = self.mode_operands(i, id, &s.mode)?;
                let tool = ToolOp {
                    shape: ToolShape::Sweep {
                        profile,
                        path,
                        path_curves,
                    },
                    combine,
                };
                Ok(Compiled::Op {
                    op: KernelOp::Tool(tool),
                    operands,
                    naming: simple(vec![s.profile.sketch.clone()]),
                    import: false,
                    instances: None,
                    tools: None,
                })
            }
            FeatureKind::Loft(l) => {
                let mut profiles = Vec::new();
                for (k, p) in l.profiles.iter().enumerate() {
                    profiles.push(self.profile(i, id, &format!("profiles[{}]", k), p)?);
                }
                self.list(i, id, "guides", &l.guides, Some(TopoKind::Edge))
                    .map(|_| ())
                    .or_else(|h| if l.guides.is_empty() { Ok(()) } else { Err(h) })?;
                let (combine, operands) = self.mode_operands(i, id, &l.mode)?;
                let tool = ToolOp {
                    shape: ToolShape::Loft { profiles },
                    combine,
                };
                Ok(Compiled::Op {
                    op: KernelOp::Tool(tool),
                    operands,
                    naming: simple(l.profiles.iter().map(|p| p.sketch.clone()).collect()),
                    import: false,
                    instances: None,
                    tools: None,
                })
            }
            FeatureKind::Hole(h) => {
                let r = self.single(i, id, "face", &h.face, Some(TopoKind::Face))?;
                let face = *r.entities().iter().next().expect("one");
                let frame = self.face_frame(&face).ok_or_else(|| {
                    self.error(
                        id,
                        Some("face"),
                        DiagCode::RefKindMismatch,
                        "hole face is not planar".into(),
                        Vec::new(),
                    )
                })?;
                let mut positions = Vec::new();
                for (k, p) in h.positions.iter().enumerate() {
                    let w = self.point(i, id, &format!("positions[{}]", k), &p.at)?;
                    let d = sub(w, frame.origin);
                    positions.push((p.key.clone(), [dot(d, frame.x), dot(d, frame.y)]));
                }
                let diameter = self.value(id, "diameter", &h.diameter, Dimension::LENGTH)?;
                let depth = match &h.depth {
                    Some(d) => Some(self.value(id, "depth", d, Dimension::LENGTH)?),
                    None => None,
                };
                let (counterbore, countersink) = match &h.form {
                    HoleForm::Simple => (None, None),
                    HoleForm::Counterbore { diameter, depth } => (
                        Some((
                            self.value(id, "form.diameter", diameter, Dimension::LENGTH)?,
                            self.value(id, "form.depth", depth, Dimension::LENGTH)?,
                        )),
                        None,
                    ),
                    HoleForm::Countersink { diameter, angle } => (
                        None,
                        Some((
                            self.value(id, "form.diameter", diameter, Dimension::LENGTH)?,
                            self.value(id, "form.angle", angle, Dimension::ANGLE)?,
                        )),
                    ),
                };
                let operands = vec![face.body as usize];
                let tool = ToolOp {
                    shape: ToolShape::Holes {
                        frame,
                        positions,
                        diameter,
                        depth,
                        counterbore,
                        countersink,
                    },
                    combine: Combine::Cut,
                };
                Ok(Compiled::Op {
                    op: KernelOp::Tool(tool.clone()),
                    operands,
                    naming: simple(Vec::new()),
                    import: false,
                    instances: None,
                    tools: Some(vec![tool]),
                })
            }
            FeatureKind::Fillet(fl) => {
                let set = self.list(i, id, "edges", &fl.edges, Some(TopoKind::Edge))?;
                let radius = self.value(id, "radius", &fl.radius, Dimension::LENGTH)?;
                let operands = Self::bodies_of(&set);
                let edges = Self::operand_entities(&set, &operands);
                Ok(Compiled::Op {
                    op: KernelOp::Fillet { edges, radius },
                    operands,
                    naming: simple(Vec::new()),
                    import: false,
                    instances: None,
                    tools: None,
                })
            }
            FeatureKind::Chamfer(c) => {
                let set = self.list(i, id, "edges", &c.edges, Some(TopoKind::Edge))?;
                let distance = self.value(id, "distance", &c.distance, Dimension::LENGTH)?;
                let distance2 = match &c.distance2 {
                    Some(d) => Some(self.value(id, "distance2", d, Dimension::LENGTH)?),
                    None => None,
                };
                let operands = Self::bodies_of(&set);
                let edges = Self::operand_entities(&set, &operands);
                Ok(Compiled::Op {
                    op: KernelOp::Chamfer {
                        edges,
                        distance,
                        distance2,
                    },
                    operands,
                    naming: simple(Vec::new()),
                    import: false,
                    instances: None,
                    tools: None,
                })
            }
            FeatureKind::Shell(s) => {
                let b = self.single(i, id, "body", &s.body, Some(TopoKind::Body))?;
                let body = b.entities().iter().next().expect("one").body as usize;
                let faces = if s.open_faces.is_empty() {
                    BTreeSet::new()
                } else {
                    self.list(i, id, "open_faces", &s.open_faces, Some(TopoKind::Face))?
                };
                let thickness = self.value(id, "thickness", &s.thickness, Dimension::LENGTH)?;
                let operands = vec![body];
                if faces.iter().any(|h| h.body as usize != body) {
                    return Err(self.error(
                        id,
                        Some("open_faces"),
                        DiagCode::RefKindMismatch,
                        "open faces must be on the shelled body".into(),
                        Vec::new(),
                    ));
                }
                let open_faces = Self::operand_entities(&faces, &operands);
                Ok(Compiled::Op {
                    op: KernelOp::Shell {
                        open_faces,
                        thickness,
                    },
                    operands,
                    naming: simple(Vec::new()),
                    import: false,
                    instances: None,
                    tools: None,
                })
            }
            FeatureKind::Draft(d) => {
                let set = self.list(i, id, "faces", &d.faces, Some(TopoKind::Face))?;
                let neutral = self.plane(i, id, "neutral", &d.neutral)?;
                let angle = self.value(id, "angle", &d.angle, Dimension::ANGLE)?;
                let operands = Self::bodies_of(&set);
                let faces = Self::operand_entities(&set, &operands);
                Ok(Compiled::Op {
                    op: KernelOp::Draft {
                        faces,
                        neutral,
                        angle,
                    },
                    operands,
                    naming: simple(Vec::new()),
                    import: false,
                    instances: None,
                    tools: None,
                })
            }
            FeatureKind::Pattern(p) => {
                let mut instances = Vec::new();
                match &p.layout {
                    PatternLayout::Linear {
                        direction,
                        spacing,
                        count,
                    } => {
                        let (_, d) = self.axis(i, id, "direction", direction)?;
                        let s = self.value(id, "spacing", spacing, Dimension::LENGTH)?;
                        let n = self.count(id, "count", count)?;
                        for k in 1..n {
                            instances.push(Instance {
                                key: InstanceKey::Linear { step: k },
                                transform: Transform::translation(scale(d, s * k as f64)),
                            });
                        }
                    }
                    PatternLayout::Grid {
                        first,
                        first_spacing,
                        first_count,
                        second,
                        second_spacing,
                        second_count,
                    } => {
                        let (_, d1) = self.axis(i, id, "first", first)?;
                        let s1 =
                            self.value(id, "first_spacing", first_spacing, Dimension::LENGTH)?;
                        let n1 = self.count(id, "first_count", first_count)?;
                        let (_, d2) = self.axis(i, id, "second", second)?;
                        let s2 =
                            self.value(id, "second_spacing", second_spacing, Dimension::LENGTH)?;
                        let n2 = self.count(id, "second_count", second_count)?;
                        for a in 0..n1 {
                            for b in 0..n2 {
                                if a == 0 && b == 0 {
                                    continue;
                                }
                                let t = add(scale(d1, s1 * a as f64), scale(d2, s2 * b as f64));
                                instances.push(Instance {
                                    key: InstanceKey::Grid {
                                        first: a,
                                        second: b,
                                    },
                                    transform: Transform::translation(t),
                                });
                            }
                        }
                    }
                    PatternLayout::Circular { axis, count, angle } => {
                        let (pt, dir) = self.axis(i, id, "axis", axis)?;
                        let n = self.count(id, "count", count)?;
                        let total = self.value(id, "angle", angle, Dimension::ANGLE)?;
                        for k in 1..n {
                            let a = total * k as f64 / n as f64;
                            let col = |v: [f64; 3]| rotate(v, dir, a);
                            let (cx, cy, cz) = (
                                col([1.0, 0.0, 0.0]),
                                col([0.0, 1.0, 0.0]),
                                col([0.0, 0.0, 1.0]),
                            );
                            let rotation = [
                                [cx[0], cy[0], cz[0]],
                                [cx[1], cy[1], cz[1]],
                                [cx[2], cy[2], cz[2]],
                            ];
                            let t = sub(pt, rot_apply(&rotation, pt));
                            instances.push(Instance {
                                key: InstanceKey::Circular { step: k },
                                transform: Transform {
                                    rotation,
                                    translation: t,
                                },
                            });
                        }
                    }
                    PatternLayout::Table { rows } => {
                        for r in rows {
                            let t = [
                                self.value(id, "rows.dx", &r.dx, Dimension::LENGTH)?,
                                self.value(id, "rows.dy", &r.dy, Dimension::LENGTH)?,
                                self.value(id, "rows.dz", &r.dz, Dimension::LENGTH)?,
                            ];
                            instances.push(Instance {
                                key: InstanceKey::Row { key: r.key.clone() },
                                transform: Transform::translation(t),
                            });
                        }
                    }
                }
                instances.retain(|inst| !p.skip.contains(&inst.key));
                self.pattern_like(i, id, &p.seeds, instances)
            }
            FeatureKind::Mirror(m) => {
                let frame = self.plane(i, id, "plane", &m.plane)?;
                let instances = vec![Instance {
                    key: InstanceKey::Mirror,
                    transform: Transform::reflection(frame.origin, frame.z),
                }];
                self.pattern_like(i, id, &m.seeds, instances)
            }
            FeatureKind::Boolean(b) => {
                let t = self.single(i, id, "target", &b.target, Some(TopoKind::Body))?;
                let target = t.entities().iter().next().expect("one").body as usize;
                let tools = self.list(i, id, "tools", &b.tools, Some(TopoKind::Body))?;
                let mut operands = vec![target];
                for tb in Self::bodies_of(&tools) {
                    if tb != target {
                        operands.push(tb);
                    }
                }
                let combine = match b.kind {
                    BooleanKind::Union => Combine::Add,
                    BooleanKind::Subtract => Combine::Cut,
                    BooleanKind::Intersect => Combine::Intersect,
                };
                Ok(Compiled::Op {
                    op: KernelOp::Boolean {
                        combine,
                        keep_tools: b.keep_tools,
                    },
                    operands,
                    naming: simple(Vec::new()),
                    import: false,
                    instances: None,
                    tools: None,
                })
            }
            FeatureKind::Split(s) => {
                let b = self.single(i, id, "body", &s.body, Some(TopoKind::Body))?;
                let body = b.entities().iter().next().expect("one").body as usize;
                let frame = self.plane(i, id, "tool", &s.tool)?;
                Ok(Compiled::Op {
                    op: KernelOp::Split { frame },
                    operands: vec![body],
                    naming: simple(Vec::new()),
                    import: false,
                    instances: None,
                    tools: None,
                })
            }
            FeatureKind::BodyOp(op) => match op {
                BodyOp::Delete { bodies } => {
                    let set = self.list(i, id, "bodies", bodies, Some(TopoKind::Body))?;
                    Ok(Compiled::Delete(Self::bodies_of(&set)))
                }
                BodyOp::Translate { bodies, dx, dy, dz } => {
                    let set = self.list(i, id, "bodies", bodies, Some(TopoKind::Body))?;
                    let t = [
                        self.value(id, "dx", dx, Dimension::LENGTH)?,
                        self.value(id, "dy", dy, Dimension::LENGTH)?,
                        self.value(id, "dz", dz, Dimension::LENGTH)?,
                    ];
                    Ok(Compiled::Op {
                        op: KernelOp::Transform {
                            transform: Transform::translation(t),
                        },
                        operands: Self::bodies_of(&set),
                        naming: simple(Vec::new()),
                        import: false,
                        instances: None,
                        tools: None,
                    })
                }
            },
            FeatureKind::Direct(_)
            | FeatureKind::Surface(_)
            | FeatureKind::SheetMetal(_)
            | FeatureKind::Weldment(_)
            | FeatureKind::Mold(_)
            | FeatureKind::Thread(_)
            | FeatureKind::Emboss(_)
            | FeatureKind::Rib(_)
            | FeatureKind::Library(_)
            | FeatureKind::Curve(_)
            | FeatureKind::MateConnector(_) => unreachable!("handled by compile_family"),
            FeatureKind::Import(imp) => Ok(Compiled::Op {
                op: KernelOp::Import {
                    source: imp.source.clone(),
                    digest: imp.digest.clone(),
                },
                operands: Vec::new(),
                naming: simple(Vec::new()),
                import: true,
                instances: None,
                tools: None,
            }),
        }
    }

    fn pattern_like(
        &mut self,
        i: usize,
        id: &FeatureId,
        seeds: &PatternSeeds,
        instances: Vec<Instance>,
    ) -> Result<Compiled, Halt> {
        let keys: BTreeSet<InstanceKey> = instances.iter().map(|x| x.key.clone()).collect();
        match seeds {
            PatternSeeds::Features { features } => {
                let mut tools = Vec::new();
                let mut seed_ids = Vec::new();
                let mut profiles = Vec::new();
                for (k, s) in features.iter().enumerate() {
                    self.gate_feature(i, id, s, &format!("seeds[{}]", k))?;
                    let Some(t) = self.st.tools.get(s).cloned() else {
                        let kind = self
                            .doc
                            .feature(s)
                            .map(|x| x.kind.name())
                            .unwrap_or("feature");
                        return Err(self.error(
                            id,
                            Some(&format!("seeds[{}]", k)),
                            DiagCode::KernelNotSupported,
                            format!("not supported yet: patterning a {} feature", kind),
                            Vec::new(),
                        ));
                    };
                    let sketch = match self.doc.feature(s).map(|x| &x.kind) {
                        Some(FeatureKind::Extrude(e)) => vec![e.profile.sketch.clone()],
                        Some(FeatureKind::Revolve(r)) => vec![r.profile.sketch.clone()],
                        _ => Vec::new(),
                    };
                    for tool in t {
                        tools.push(tool);
                        seed_ids.push(s.clone());
                        profiles.push(sketch.clone());
                    }
                }
                let naming = NamingContext {
                    feature: id.clone(),
                    profiles,
                    seeds: seed_ids,
                };
                let operands = self.implicit_bodies(i, id)?;
                Ok(Compiled::Op {
                    op: KernelOp::PatternTools {
                        seeds: tools,
                        instances,
                    },
                    operands,
                    naming,
                    import: false,
                    instances: Some(keys),
                    tools: None,
                })
            }
            PatternSeeds::Bodies { bodies } => {
                let set = self.list(i, id, "bodies", bodies, Some(TopoKind::Body))?;
                Ok(Compiled::Op {
                    op: KernelOp::PatternBodies { instances },
                    operands: Self::bodies_of(&set),
                    naming: NamingContext::simple(id, Vec::new()),
                    import: false,
                    instances: Some(keys),
                    tools: None,
                })
            }
        }
    }

    fn apply(
        &mut self,
        _i: usize,
        f: &Feature,
        compiled: Compiled,
        key: &CacheKey,
        cache: &mut RegenCache,
    ) -> Result<FeatureStatus, Halt> {
        let id = &f.id;
        match compiled {
            Compiled::Record(r) => {
                self.st.records.insert(id.clone(), r);
                Ok(FeatureStatus::Regenerated)
            }
            Compiled::Datum(d) => {
                self.st.datums.insert(id.clone(), d);
                Ok(FeatureStatus::Regenerated)
            }
            Compiled::Sketch(s, frame) => {
                self.st.sketches.insert(id.clone(), (s, frame));
                Ok(FeatureStatus::Regenerated)
            }
            Compiled::Delete(bodies) => {
                for &b in bodies.iter().rev() {
                    let body = self.st.bodies.remove(b);
                    for n in body.names.body {
                        self.st
                            .burials
                            .entry((TopoKind::Body, n))
                            .or_insert_with(|| id.clone());
                    }
                    for n in body.names.faces.into_iter().flatten() {
                        self.st
                            .burials
                            .entry((TopoKind::Face, n))
                            .or_insert_with(|| id.clone());
                    }
                }
                Ok(FeatureStatus::Regenerated)
            }
            Compiled::Op {
                op,
                operands,
                naming,
                import,
                instances,
                tools,
            } => {
                let mut status = FeatureStatus::Regenerated;
                let mut output = None;
                if let Some(frozen) = &f.frozen {
                    match cache.entries.get(frozen) {
                        Some(o) => {
                            if frozen != key {
                                self.diags.push(
                                    DiagCode::RegenFrozenStale,
                                    Severity::Info,
                                    Site::feature(id, None),
                                    format!("`{}` is frozen; upstream changes since the freeze are not applied to it", id),
                                    Cause::Root,
                                    Vec::new(),
                                );
                            }
                            output = Some(o.clone());
                            status = FeatureStatus::Frozen;
                        }
                        None => {
                            self.diags.push(
                                DiagCode::RegenFrozenMissing,
                                Severity::Warning,
                                Site::feature(id, None),
                                format!("`{}` is frozen but its cached result is gone; regenerated instead", id),
                                Cause::Root,
                                Vec::new(),
                            );
                        }
                    }
                }
                if output.is_none() {
                    if let Some(o) = cache.entries.get(key) {
                        cache.hits += 1;
                        status = FeatureStatus::CacheHit;
                        output = Some(o.clone());
                    }
                }
                let output = match output {
                    Some(o) => o,
                    None => {
                        let handles: Vec<&BodyHandle> = operands
                            .iter()
                            .map(|b| &self.st.bodies[*b].handle)
                            .collect();
                        match self.ctx.kernel.execute(&op, &handles) {
                            Ok(o) => {
                                cache.misses += 1;
                                cache.entries.insert(key.clone(), o.clone());
                                o
                            }
                            Err(e) => {
                                let code = match &e {
                                    KernelError::NotSupported { .. } => {
                                        DiagCode::KernelNotSupported
                                    }
                                    KernelError::HistoryIncomplete { .. } => {
                                        DiagCode::KernelHistoryIncomplete
                                    }
                                    _ => DiagCode::KernelFailed,
                                };
                                return Err(self.error(id, None, code, e.to_string(), Vec::new()));
                            }
                        }
                    }
                };
                let named = {
                    let views: Vec<OperandView<'_>> = operands
                        .iter()
                        .map(|b| OperandView {
                            names: &self.st.bodies[*b].names,
                            topology: &self.st.bodies[*b].topology,
                        })
                        .collect();
                    let outs: Vec<&Topology> = output.bodies.iter().map(|b| &b.topology).collect();
                    name_outputs(&naming, &views, &outs, &output.history)
                };
                let mut named = match named {
                    Ok(n) => n,
                    Err(e) => {
                        return Err(self.error(
                            id,
                            None,
                            DiagCode::KernelHistoryIncomplete,
                            e.to_string(),
                            Vec::new(),
                        ))
                    }
                };
                if import {
                    cad_names_for_import(id, &mut named);
                }
                if let Err(c) = named.intern_into(&mut self.st.names) {
                    return Err(self.error(
                        id,
                        None,
                        DiagCode::RefTokenCollision,
                        c.to_string(),
                        Vec::new(),
                    ));
                }
                for tie in &named.slot_ties {
                    let Some(first) = tie.names.first() else {
                        continue;
                    };
                    let body_ref = Reference::body(first.clone());
                    let candidates = slot_tie_candidates(&body_ref, tie, &named.bodies);
                    self.diags.push(
                        DiagCode::RefBodySlotTie,
                        Severity::Warning,
                        Site::feature(id, None),
                        format!(
                            "`{}` split `{}` into {} pieces equally similar to it by lineage; \
                             references to it now select all of them",
                            id,
                            Shown(&body_ref, Some(&self.st.names)),
                            tie.bodies.len()
                        ),
                        Cause::Root,
                        vec![Repair::Disambiguate {
                            site: Site::feature(id, None),
                            candidates,
                        }],
                    );
                }
                let insert_at = operands.first().copied().unwrap_or(self.st.bodies.len());
                for &b in operands.iter().rev() {
                    self.st.bodies.remove(b);
                }
                let insert_at = insert_at.min(self.st.bodies.len());
                for (k, (kb, names)) in output.bodies.into_iter().zip(named.bodies).enumerate() {
                    self.st.bodies.insert(
                        insert_at + k,
                        LiveBody {
                            handle: kb.handle,
                            topology: kb.topology,
                            names,
                        },
                    );
                }
                for (kind, n) in named.buried {
                    self.st
                        .burials
                        .entry((kind, n))
                        .or_insert_with(|| id.clone());
                }
                if let Some(keys) = instances {
                    self.st.table.instances.insert(id.clone(), keys);
                }
                if let Some(t) = tools {
                    self.st.tools.insert(id.clone(), t);
                }
                Ok(status)
            }
        }
    }

    #[allow(clippy::type_complexity)]
    fn items(
        &mut self,
    ) -> (
        BTreeMap<PartName, ItemOutcome>,
        BTreeMap<RegionName, ItemOutcome>,
        Vec<ItemOutcome>,
        PublishTable,
    ) {
        let doc = self.doc;
        let mut parts = BTreeMap::new();
        for p in &doc.parts {
            self.records.clear();
            let o =
                self.resolver(None)
                    .resolve(&p.body, Some(TopoKind::Body), Cardinality::ExactlyOne);
            let site = Site {
                subject: Subject::Part {
                    part: p.name.clone(),
                },
                param: Some("body".into()),
            };
            let r = self.take(site, &p.body, o);
            let entities = match &r {
                Ok(res) => res
                    .entities()
                    .iter()
                    .map(|h| self.st.entity_names(h))
                    .collect(),
                Err(_) => Vec::new(),
            };
            parts.insert(
                p.name.clone(),
                ItemOutcome {
                    references: std::mem::take(&mut self.records),
                    entities,
                    ok: r.is_ok(),
                },
            );
        }
        let mut regions = BTreeMap::new();
        for rg in &doc.regions {
            self.records.clear();
            let (each, _) = self.resolver(None).resolve_list(&rg.select, Some(rg.kind));
            let mut set = BTreeSet::new();
            let mut foreign: Vec<Vec<EntityName>> = Vec::new();
            let mut ok = true;
            for (k, (r, o)) in rg.select.iter().zip(each).enumerate() {
                let site = Site {
                    subject: Subject::Region {
                        region: rg.name.clone(),
                    },
                    param: Some(format!("select[{}]", k)),
                };
                match self.take(site, r, o) {
                    Ok(res) => {
                        if let Resolved::Foreign { target } = &res.target {
                            foreign.extend(target.entities.iter().map(|e| e.names.clone()));
                        }
                        set.extend(res.entities());
                    }
                    Err(_) => ok = false,
                }
            }
            if ok {
                let n = set.len() + foreign.len();
                let bad = match rg.expect {
                    Cardinality::ExactlyOne => n != 1,
                    Cardinality::AtLeastOne => n == 0,
                    Cardinality::Any => false,
                };
                if bad {
                    ok = false;
                    let candidates = if n > 1 && rg.select.len() == 1 {
                        self.resolver(None).disambiguate(&rg.select[0], &set)
                    } else {
                        Vec::new()
                    };
                    let site = Site {
                        subject: Subject::Region {
                            region: rg.name.clone(),
                        },
                        param: None,
                    };
                    self.diags.push(
                        DiagCode::RegionExpectation,
                        Severity::Error,
                        site.clone(),
                        format!(
                            "region `{}` expects {:?} but selects {}",
                            rg.name, rg.expect, n
                        ),
                        Cause::Root,
                        vec![Repair::Disambiguate { site, candidates }],
                    );
                }
            }
            let mut entities: Vec<Vec<EntityName>> =
                set.iter().map(|h| self.st.entity_names(h)).collect();
            entities.extend(foreign);
            regions.insert(
                rg.name.clone(),
                ItemOutcome {
                    references: std::mem::take(&mut self.records),
                    entities,
                    ok,
                },
            );
        }
        let mut materials = Vec::new();
        for (k, m) in doc.materials.iter().enumerate() {
            self.records.clear();
            let o =
                self.resolver(None)
                    .resolve(&m.body, Some(TopoKind::Body), Cardinality::AtLeastOne);
            let site = Site {
                subject: Subject::Material { index: k },
                param: Some("body".into()),
            };
            let r = self.take(site, &m.body, o);
            let entities = match &r {
                Ok(res) => res
                    .entities()
                    .iter()
                    .map(|h| self.st.entity_names(h))
                    .collect(),
                Err(_) => Vec::new(),
            };
            materials.push(ItemOutcome {
                references: std::mem::take(&mut self.records),
                entities,
                ok: r.is_ok(),
            });
        }
        let mut table = PublishTable::new();
        for p in &doc.publishes {
            self.records.clear();
            let site = Site {
                subject: Subject::Publish {
                    publish: p.name.clone(),
                },
                param: None,
            };
            let target = ForeignTarget {
                file: doc.path.clone(),
                revision: None,
                name: p.name.clone(),
                entities: Vec::new(),
                datum: None,
                value: None,
            };
            let result = match &p.target {
                PublishTarget::Entities { kind, references } => {
                    let mut entities = Vec::new();
                    let mut ok = true;
                    for r in references {
                        let o =
                            self.resolver(None)
                                .resolve(r, Some(*kind), Cardinality::AtLeastOne);
                        match self.take(site.clone(), r, o) {
                            Ok(res) => {
                                for h in res.entities() {
                                    entities.push(ForeignEntity {
                                        kind: h.kind,
                                        names: self.st.entity_names(&h),
                                    });
                                }
                            }
                            Err(_) => ok = false,
                        }
                    }
                    if ok {
                        Ok(ForeignTarget { entities, ..target })
                    } else {
                        Err(())
                    }
                }
                PublishTarget::Datum { datum } => match datum {
                    DatumRef::Feature { feature } => match self.st.datums.get(feature) {
                        Some(d) => Ok(ForeignTarget {
                            datum: Some(*d),
                            ..target
                        }),
                        None => Err(()),
                    },
                    DatumRef::OriginPlane { plane } => Ok(ForeignTarget {
                        datum: Some(DatumValue::Plane {
                            frame: Frame::origin_plane(*plane),
                        }),
                        ..target
                    }),
                    _ => Ok(ForeignTarget {
                        datum: Some(DatumValue::Frame {
                            frame: Frame::WORLD,
                        }),
                        ..target
                    }),
                },
                PublishTarget::Region { region } => match regions.get(region) {
                    Some(o) if o.ok => Ok(ForeignTarget {
                        entities: o
                            .entities
                            .iter()
                            .map(|n| ForeignEntity {
                                kind: doc
                                    .regions
                                    .iter()
                                    .find(|r| &r.name == region)
                                    .map(|r| r.kind)
                                    .unwrap_or(TopoKind::Face),
                                names: n.clone(),
                            })
                            .collect(),
                        ..target
                    }),
                    _ => Err(()),
                },
                PublishTarget::Part { part } => match parts.get(part) {
                    Some(o) if o.ok => Ok(ForeignTarget {
                        entities: o
                            .entities
                            .iter()
                            .map(|n| ForeignEntity {
                                kind: TopoKind::Body,
                                names: n.clone(),
                            })
                            .collect(),
                        ..target
                    }),
                    _ => Err(()),
                },
                PublishTarget::Value { input } => match self.values.inputs.get(input) {
                    Some(q) => Ok(ForeignTarget {
                        value: Some(*q),
                        ..target
                    }),
                    None => Err(()),
                },
                PublishTarget::Let { name } => match self.values.lets.get(name) {
                    Some(q) => Ok(ForeignTarget {
                        value: Some(*q),
                        ..target
                    }),
                    None => Err(()),
                },
            };
            table.insert(
                p.name.clone(),
                result.map_err(|_| XrefProblem::PublisherBroken {
                    file: doc.path.clone(),
                    name: p.name.clone(),
                }),
            );
        }
        // Lock ids are looked up, never minted, during regeneration.
        if let Some(lock) = self.ctx.lock {
            for p in &doc.parts {
                let _ = lock.lookup(&ItemPath::part(&doc.path, &p.name));
            }
        }
        (parts, regions, materials, table)
    }
}

/// Rotate `v` about unit axis `k` by `angle` (Rodrigues).
fn rotate(v: [f64; 3], k: [f64; 3], angle: f64) -> [f64; 3] {
    let k = normalize(k);
    let (s, c) = angle.sin_cos();
    add(
        add(scale(v, c), scale(cross(k, v), s)),
        scale(k, dot(k, v) * (1.0 - c)),
    )
}

fn rot_apply(r: &[[f64; 3]; 3], p: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2],
        r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2],
        r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2],
    ]
}

/// Closed loops from a solved sketch: explicit loops as listed, or every closed chain of
/// non-construction lines and arcs (circles are loops alone). The outer loop (largest area)
/// comes first.
fn build_loops(
    sketch: &Sketch,
    solved: &SolvedSketch,
    explicit: &[Vec<crate::ids::SketchEntityId>],
) -> Result<Vec<Vec<ProfileSegment>>, String> {
    let seg_of = |id: &crate::ids::SketchEntityId| -> Option<ProfileSegment> {
        let e = sketch.entities.iter().find(|e| &e.id == id)?;
        match &e.geometry {
            SketchGeometry::Line { .. } => Some(ProfileSegment {
                curve: id.clone(),
                kind: SegmentKind::Line,
                start: solved.point(id, PointRole::Start)?,
                end: solved.point(id, PointRole::End)?,
            }),
            SketchGeometry::Arc { .. } => Some(ProfileSegment {
                curve: id.clone(),
                kind: SegmentKind::Arc {
                    center: solved.point(id, PointRole::Center)?,
                    clockwise: false,
                },
                start: solved.point(id, PointRole::Start)?,
                end: solved.point(id, PointRole::End)?,
            }),
            SketchGeometry::Circle { .. } => {
                let c = solved.point(id, PointRole::Center)?;
                let r = *solved.radii.get(id)?;
                Some(ProfileSegment {
                    curve: id.clone(),
                    kind: SegmentKind::Circle {
                        center: c,
                        radius: r,
                    },
                    start: [c[0] + r, c[1]],
                    end: [c[0] + r, c[1]],
                })
            }
            SketchGeometry::Point { .. } => None,
        }
    };
    let close = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9;
    let mut loops: Vec<Vec<ProfileSegment>> = Vec::new();
    if !explicit.is_empty() {
        for l in explicit {
            let mut segs = Vec::new();
            for id in l {
                segs.push(seg_of(id).ok_or_else(|| format!("`{}` is not a profile curve", id))?);
            }
            loops.push(segs);
        }
    } else {
        let mut pool: Vec<ProfileSegment> = sketch
            .entities
            .iter()
            .filter(|e| !e.construction)
            .filter_map(|e| seg_of(&e.id))
            .collect();
        while !pool.is_empty() {
            let first = pool.remove(0);
            if let SegmentKind::Circle { .. } = first.kind {
                loops.push(vec![first]);
                continue;
            }
            let start = first.start;
            let mut chain = vec![first];
            loop {
                let end = chain.last().expect("non-empty").end;
                if close(end, start) {
                    break;
                }
                let Some(k) = pool.iter().position(|s| {
                    !matches!(s.kind, SegmentKind::Circle { .. })
                        && (close(s.start, end) || close(s.end, end))
                }) else {
                    return Err(format!(
                        "the profile is open at ({:.6}, {:.6}) after `{}`",
                        end[0],
                        end[1],
                        chain.last().expect("non-empty").curve
                    ));
                };
                let mut s = pool.remove(k);
                if !close(s.start, end) {
                    reverse_segment(&mut s);
                }
                chain.push(s);
            }
            loops.push(chain);
        }
    }
    if loops.is_empty() {
        return Err("the sketch has no closed profile".into());
    }
    let area = |l: &Vec<ProfileSegment>| -> f64 {
        l.iter()
            .map(|s| match s.kind {
                SegmentKind::Circle { radius, .. } => std::f64::consts::PI * radius * radius,
                _ => (s.start[0] * s.end[1] - s.end[0] * s.start[1]) / 2.0,
            })
            .sum::<f64>()
            .abs()
    };
    let outer = (0..loops.len())
        .max_by(|a, b| {
            area(&loops[*a])
                .partial_cmp(&area(&loops[*b]))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.cmp(a))
        })
        .expect("non-empty");
    let o = loops.remove(outer);
    loops.insert(0, o);
    Ok(loops)
}

fn reverse_segment(s: &mut ProfileSegment) {
    std::mem::swap(&mut s.start, &mut s.end);
    if let SegmentKind::Arc { clockwise, .. } = &mut s.kind {
        *clockwise = !*clockwise;
    }
}

/// Points along a segment from start to end: the ends for a line, samples for an arc.
fn segment_samples(s: &ProfileSegment) -> Vec<[f64; 2]> {
    match s.kind {
        SegmentKind::Line => vec![s.start, s.end],
        SegmentKind::Arc { center, clockwise } => {
            let r = ((s.start[0] - center[0]).powi(2) + (s.start[1] - center[1]).powi(2)).sqrt();
            let a0 = (s.start[1] - center[1]).atan2(s.start[0] - center[0]);
            let a1 = (s.end[1] - center[1]).atan2(s.end[0] - center[0]);
            let tau = std::f64::consts::TAU;
            let mut sweep = a1 - a0;
            if clockwise {
                while sweep >= -1e-12 {
                    sweep -= tau;
                }
            } else {
                while sweep <= 1e-12 {
                    sweep += tau;
                }
            }
            let n = ((sweep.abs() / (std::f64::consts::PI / 12.0)).ceil() as usize).max(2);
            (0..=n)
                .map(|k| {
                    let a = a0 + sweep * k as f64 / n as f64;
                    [center[0] + r * a.cos(), center[1] + r * a.sin()]
                })
                .collect()
        }
        SegmentKind::Circle { center, radius } => (0..=24)
            .map(|k| {
                let a = std::f64::consts::TAU * k as f64 / 24.0;
                [center[0] + radius * a.cos(), center[1] + radius * a.sin()]
            })
            .collect(),
    }
}

/// A sketch's non-construction curves chained end to end into one path, starting from a free
/// end when the path is open.
fn build_path(sketch: &Sketch, solved: &SolvedSketch) -> Result<Vec<ProfileSegment>, String> {
    let close = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9;
    let mut pool: Vec<ProfileSegment> = Vec::new();
    for e in sketch.entities.iter().filter(|e| !e.construction) {
        let seg = match &e.geometry {
            SketchGeometry::Line { .. } => ProfileSegment {
                curve: e.id.clone(),
                kind: SegmentKind::Line,
                start: solved
                    .point(&e.id, PointRole::Start)
                    .ok_or("path line not solved")?,
                end: solved
                    .point(&e.id, PointRole::End)
                    .ok_or("path line not solved")?,
            },
            SketchGeometry::Arc { .. } => ProfileSegment {
                curve: e.id.clone(),
                kind: SegmentKind::Arc {
                    center: solved
                        .point(&e.id, PointRole::Center)
                        .ok_or("path arc not solved")?,
                    clockwise: false,
                },
                start: solved
                    .point(&e.id, PointRole::Start)
                    .ok_or("path arc not solved")?,
                end: solved
                    .point(&e.id, PointRole::End)
                    .ok_or("path arc not solved")?,
            },
            _ => continue,
        };
        pool.push(seg);
    }
    if pool.is_empty() {
        return Err("the path sketch has no curves".into());
    }
    // Start at a free end: an endpoint no other curve shares.
    let shared = |p: [f64; 2], me: usize, pool: &[ProfileSegment]| {
        pool.iter()
            .enumerate()
            .any(|(j, s)| j != me && (close(s.start, p) || close(s.end, p)))
    };
    let mut first = 0;
    let mut flip = false;
    'find: for (i, s) in pool.iter().enumerate() {
        if !shared(s.start, i, &pool) {
            first = i;
            break 'find;
        }
        if !shared(s.end, i, &pool) {
            first = i;
            flip = true;
            break 'find;
        }
    }
    let mut seg = pool.remove(first);
    if flip {
        reverse_segment(&mut seg);
    }
    let mut chain = vec![seg];
    while !pool.is_empty() {
        let end = chain.last().expect("non-empty").end;
        let Some(k) = pool
            .iter()
            .position(|s| close(s.start, end) || close(s.end, end))
        else {
            return Err(format!(
                "the path is broken at ({:.6}, {:.6}) after `{}`",
                end[0],
                end[1],
                chain.last().expect("non-empty").curve
            ));
        };
        let mut s = pool.remove(k);
        if !close(s.start, end) {
            reverse_segment(&mut s);
        }
        chain.push(s);
    }
    Ok(chain)
}

/// For each body of a slot tie, a reference that selects it alone: the body carrying a face
/// name no other tied piece has.
fn slot_tie_candidates(
    body_ref: &Reference,
    tie: &SlotTie,
    bodies: &[BodyNames],
) -> Vec<Reference> {
    let sets: Vec<BTreeSet<&EntityName>> = tie
        .bodies
        .iter()
        .map(|b| {
            bodies
                .get(*b as usize)
                .map(|n| n.faces.iter().flatten().collect())
                .unwrap_or_default()
        })
        .collect();
    let mut out = Vec::new();
    for (i, mine) in sets.iter().enumerate() {
        let unique = mine.iter().find(|n| {
            sets.iter()
                .enumerate()
                .all(|(j, o)| j == i || !o.contains(*n))
        });
        if let Some(n) = unique {
            out.push(Reference::piece(
                body_ref.clone(),
                vec![Reference::face((*n).clone())],
            ));
        }
    }
    out
}
