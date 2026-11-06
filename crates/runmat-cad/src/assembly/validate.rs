//! The variant validator: proves that every legal configuration resolves uniquely.
//!
//! Over the product reachable from a top assembly, with the top's option space:
//! - every named configuration is a legal point;
//! - within each assembly, the alternatives of one child name never overlap anywhere in the
//!   configurations where that assembly is used (its **context region**: the union, over its
//!   uses, of the conditions along the way), and `required` names are always covered;
//! - mates, overrides, patterns, mirrors and fastener rules are active only where the
//!   occurrences they name exist (a mate active where its occurrence is absent is an error);
//! - configuration switches are checked for overlapping cases (first match applies).
//!
//! Each check is exact region arithmetic ([`crate::assembly::options`]); a failure names a
//! witness configuration. Files are visited once, in topological order of the use graph, so
//! the cost is linear in the authored structure, not in the number of configurations.

use crate::assembly::fasteners::FastenerLayout;
use crate::assembly::model::*;
use crate::assembly::options::{Condition, OptionProblem, OptionSpace, OptionsBlock, Region};
use crate::assembly::resolve::option_code;
use crate::assembly::source::DesignSource;
use crate::diagnostics::{Cause, DiagCode, DiagnosticReport, Repair, Severity, Site, Subject};
use crate::ids::{FilePath, RevisionId};
use crate::refs::{InstanceKey, XrefContext};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ValidationStats {
    pub files: u64,
    pub slots: u64,
    pub pair_checks: u64,
    pub reference_checks: u64,
}

#[derive(Clone, Debug, Default)]
pub struct ValidationReport {
    pub diagnostics: DiagnosticReport,
    pub stats: ValidationStats,
}

impl ValidationReport {
    pub fn ok(&self) -> bool {
        self.diagnostics.errors().next().is_none()
    }
}

type FileKey = (FilePath, Option<RevisionId>);

struct V<'a> {
    source: &'a dyn DesignSource,
    space: OptionSpace,
    out: ValidationReport,
    docs: BTreeMap<FileKey, Option<Arc<AssemblyDocument>>>,
}

fn rev_of(c: &XrefContext) -> Option<RevisionId> {
    match c {
        XrefContext::Pinned { revision } => Some(revision.clone()),
        XrefContext::Live => None,
    }
}

impl V<'_> {
    fn diag(
        &mut self,
        code: DiagCode,
        severity: Severity,
        site: Site,
        message: String,
        repairs: Vec<Repair>,
    ) {
        self.out
            .diagnostics
            .push(code, severity, site, message, Cause::Root, repairs);
    }

    fn doc(&mut self, key: &FileKey) -> Option<Arc<AssemblyDocument>> {
        if let Some(d) = self.docs.get(key) {
            return d.clone();
        }
        let d = self.source.assembly(&key.0, key.1.as_ref()).ok();
        self.docs.insert(key.clone(), d.clone());
        d
    }

    fn region(&mut self, c: Option<&Condition>, file: &FilePath, site: &Site) -> Region {
        match c {
            None => self.space.full_region(),
            Some(c) => match self.space.compile(c) {
                Ok(r) => r,
                Err(p) => {
                    self.problem(&p, file, site, c);
                    Region::default()
                }
            },
        }
    }

    fn problem(&mut self, p: &OptionProblem, file: &FilePath, site: &Site, c: &Condition) {
        self.diag(
            option_code(p),
            Severity::Error,
            site.clone(),
            format!("`{}`: `when {}`: {}", file, c, p),
            Vec::new(),
        );
    }

    fn and(&mut self, a: &Region, b: &Region, site: &Site) -> Region {
        match self.space.intersect(a, b) {
            Ok(r) => r,
            Err(p) => {
                self.diag(
                    option_code(&p),
                    Severity::Error,
                    site.clone(),
                    p.to_string(),
                    Vec::new(),
                );
                Region::default()
            }
        }
    }

    fn minus(&mut self, a: &Region, b: &Region, site: &Site) -> Region {
        match self.space.subtract(a, b) {
            Ok(r) => r,
            Err(p) => {
                self.diag(
                    option_code(&p),
                    Severity::Error,
                    site.clone(),
                    p.to_string(),
                    Vec::new(),
                );
                Region::default()
            }
        }
    }

    fn or(&mut self, a: &Region, b: &Region, site: &Site) -> Region {
        let rest = self.minus(b, a, site);
        let mut out = a.clone();
        out.boxes.extend(rest.boxes);
        out
    }

    /// Where a direct child name exists in `doc` (no context applied).
    fn existence(&mut self, doc: &AssemblyDocument, seg: &PathSegment) -> Region {
        let site = Site::with_param(
            Subject::Occurrence {
                path: seg.to_string(),
            },
            doc.path.as_str(),
        );
        let mut acc = Region::default();
        for o in doc.alternatives(&seg.name) {
            if o.suppressed {
                continue;
            }
            let r = self.region(o.when.as_ref(), &doc.path, &site);
            acc = self.or(&acc, &r, &site);
        }
        if let Some(inst) = &seg.instance {
            // An instance exists where its maker is active (and the seed exists).
            let maker = doc
                .patterns
                .iter()
                .find(|p| p.name == inst.by)
                .map(|p| p.when.clone())
                .or_else(|| {
                    doc.mirrors
                        .iter()
                        .find(|m| m.name == inst.by)
                        .map(|m| m.when.clone())
                });
            match maker {
                Some(w) => {
                    let r = self.region(w.as_ref(), &doc.path, &site);
                    acc = self.and(&acc, &r, &site);
                }
                None => acc = Region::default(),
            }
            // A mirror's image of a pattern instance also needs that pattern active.
            if inst.key != InstanceKey::Mirror {
                if let Some(m) = doc.mirrors.iter().find(|m| m.name == inst.by) {
                    let pattern = doc.patterns.iter().find(|p| {
                        m.seeds.contains(&p.name) && p.seeds.contains(&seg.name)
                    });
                    if let Some(p) = pattern {
                        let r = self.region(p.when.as_ref(), &doc.path, &site);
                        acc = self.and(&acc, &r, &site);
                    }
                }
            }
        }
        acc
    }

    /// Where a path below `doc` exists, descending through sub-assembly files. A path through
    /// a name whose alternatives are different files is the union over those alternatives.
    fn path_existence(&mut self, doc: &Arc<AssemblyDocument>, path: &RelPath) -> Region {
        let Some(seg) = path.0.first() else {
            return self.space.full_region();
        };
        let site = Site::of(Subject::Occurrence {
            path: seg.to_string(),
        });
        let here = self.existence(doc, seg);
        if path.0.len() == 1 || here.is_empty() {
            return here;
        }
        let rest = RelPath(path.0[1..].to_vec());
        let alts: Vec<Occurrence> = doc.alternatives(&seg.name).cloned().collect();
        let mut below = Region::default();
        for o in alts {
            if o.source.definition.kind != DefinitionKind::Assembly {
                continue;
            }
            let key = (o.source.definition.file.clone(), rev_of(&o.source.context));
            let Some(sub) = self.doc(&key) else { continue };
            let r = self.region(o.when.as_ref(), &doc.path, &site);
            let inner = self.path_existence(&sub, &rest);
            let both = self.and(&r, &inner, &site);
            below = self.or(&below, &both, &site);
        }
        self.and(&here, &below, &site)
    }

    fn require_exists(
        &mut self,
        ctx: &Region,
        active: &Region,
        doc: &Arc<AssemblyDocument>,
        path: &RelPath,
        what: &str,
        site: Site,
    ) {
        if path.is_empty() {
            return;
        }
        self.out.stats.reference_checks += 1;
        let scope = self.and(ctx, active, &site);
        if scope.is_empty() {
            return;
        }
        let e = self.path_existence(doc, path);
        let missing = self.minus(&scope, &e, &site);
        if let Some(w) = self.space.witness(&missing) {
            let witness = self.space.describe_point(&w);
            self.diag(
                DiagCode::AsmAbsentInConfiguration,
                Severity::Error,
                site,
                format!(
                    "{} in `{}` refers to `{}`, which is absent in configurations it is active in (e.g. {})",
                    what,
                    doc.path,
                    path,
                    describe(&witness)
                ),
                Vec::new(),
            );
        }
    }

    fn check_file(&mut self, doc: &Arc<AssemblyDocument>, ctx: &Region) -> Vec<(FileKey, Region)> {
        self.out.stats.files += 1;
        let file = doc.path.clone();
        let mut uses = Vec::new();
        for name in doc.child_names() {
            self.out.stats.slots += 1;
            let site = Site::with_param(
                Subject::Occurrence {
                    path: name.to_owned(),
                },
                file.as_str(),
            );
            let alts: Vec<(usize, Occurrence)> = doc
                .occurrences
                .iter()
                .enumerate()
                .filter(|(_, o)| o.name == name && !o.suppressed)
                .map(|(i, o)| (i, o.clone()))
                .collect();
            let mut regions = Vec::with_capacity(alts.len());
            for (_, o) in &alts {
                let r = self.region(o.when.as_ref(), &file, &site);
                let r = self.and(&r, ctx, &site);
                regions.push(r);
            }
            for i in 0..alts.len() {
                for j in i + 1..alts.len() {
                    self.out.stats.pair_checks += 1;
                    let both = self.and(&regions[i], &regions[j], &site);
                    if let Some(w) = self.space.witness(&both) {
                        let witness = self.space.describe_point(&w);
                        self.diag(
                            DiagCode::AsmAmbiguousVariant,
                            Severity::Error,
                            site.clone(),
                            format!(
                                "alternatives {} and {} of `{}` in `{}` are both active at {}",
                                alts[i].0,
                                alts[j].0,
                                name,
                                file,
                                describe(&witness)
                            ),
                            vec![Repair::ExcludeConfiguration {
                                assembly: file.clone(),
                                occurrence: name.to_owned(),
                                witness,
                            }],
                        );
                    }
                }
            }
            if alts.iter().any(|(_, o)| o.required) {
                let mut cover = Region::default();
                for r in &regions {
                    cover = self.or(&cover, r, &site);
                }
                let gap = self.minus(ctx, &cover, &site);
                if let Some(w) = self.space.witness(&gap) {
                    let witness = self.space.describe_point(&w);
                    self.diag(
                        DiagCode::AsmIncompleteVariant,
                        Severity::Error,
                        site.clone(),
                        format!(
                            "required `{}` in `{}` has no active alternative at {}",
                            name,
                            file,
                            describe(&witness)
                        ),
                        Vec::new(),
                    );
                }
            }
            for ((_, o), r) in alts.iter().zip(&regions) {
                // Configuration switches.
                self.check_switch(&o.configuration, r, &file, &site);
                // Overrides target paths inside the child's definition.
                if o.source.definition.kind == DefinitionKind::Assembly {
                    let ckey = (o.source.definition.file.clone(), rev_of(&o.source.context));
                    if let Some(sub) = self.doc(&ckey) {
                        for (k, ov) in o.overrides.iter().enumerate() {
                            let osite =
                                Site::with_param(Subject::Override { index: k }, file.as_str());
                            let ow = self.region(ov.when.as_ref(), &file, &osite);
                            self.require_exists(
                                r,
                                &ow,
                                &sub,
                                &ov.target,
                                &format!("override {} of `{}`", k, name),
                                osite,
                            );
                        }
                    }
                    uses.push((ckey, r.clone()));
                }
            }
        }
        // Mates and their connectors.
        for m in &doc.mates {
            if m.suppressed {
                continue;
            }
            let site = Site::with_param(
                Subject::Mate {
                    name: m.name.clone(),
                },
                file.as_str(),
            );
            let mr = self.region(m.when.as_ref(), &file, &site);
            for c in [&m.a, &m.b] {
                match doc.connector(c) {
                    None => self.diag(
                        DiagCode::AsmUnknownConnector,
                        Severity::Error,
                        site.clone(),
                        format!(
                            "mate `{}` in `{}` uses an undeclared connector `{}`",
                            m.name, file, c
                        ),
                        Vec::new(),
                    ),
                    Some(conn) => {
                        let path = conn.occurrence.clone();
                        self.require_exists(
                            ctx,
                            &mr,
                            doc,
                            &path,
                            &format!("mate `{}`", m.name),
                            site.clone(),
                        );
                    }
                }
            }
        }
        for p in &doc.patterns {
            let site = Site::with_param(
                Subject::Pattern {
                    name: p.name.clone(),
                },
                file.as_str(),
            );
            let pr = self.region(p.when.as_ref(), &file, &site);
            for s in &p.seeds {
                self.require_exists(
                    ctx,
                    &pr,
                    doc,
                    &RelPath::new(&[s]),
                    &format!("pattern `{}`", p.name),
                    site.clone(),
                );
            }
            if let OccurrenceLayout::FollowPattern { occurrence, .. } = &p.layout {
                self.require_exists(
                    ctx,
                    &pr,
                    doc,
                    occurrence,
                    &format!("pattern `{}`", p.name),
                    site.clone(),
                );
            }
        }
        for m in &doc.mirrors {
            let site = Site::with_param(
                Subject::Pattern {
                    name: m.name.clone(),
                },
                file.as_str(),
            );
            let mr = self.region(m.when.as_ref(), &file, &site);
            for s in &m.seeds {
                self.require_exists(
                    ctx,
                    &mr,
                    doc,
                    &RelPath::new(&[s]),
                    &format!("mirror `{}`", m.name),
                    site.clone(),
                );
            }
            if let MirrorHand::Opposite { source, .. } = &m.hand {
                if source.definition.kind == DefinitionKind::Assembly {
                    uses.push((
                        (source.definition.file.clone(), rev_of(&source.context)),
                        mr.clone(),
                    ));
                }
            }
        }
        for f in &doc.fasteners {
            let site = Site::with_param(
                Subject::FastenerSet {
                    name: f.name.clone(),
                },
                file.as_str(),
            );
            let fr = self.region(f.when.as_ref(), &file, &site);
            for s in &f.stack {
                self.require_exists(
                    ctx,
                    &fr,
                    doc,
                    s,
                    &format!("fasteners `{}`", f.name),
                    site.clone(),
                );
            }
            match &f.layout {
                FastenerLayout::Along(a) => self.require_exists(
                    ctx,
                    &fr,
                    doc,
                    &a.occurrence,
                    &format!("fasteners `{}`", f.name),
                    site.clone(),
                ),
                FastenerLayout::Table(t) => self.require_exists(
                    ctx,
                    &fr,
                    doc,
                    &t.frame,
                    &format!("fasteners `{}`", f.name),
                    site.clone(),
                ),
            }
        }
        uses
    }

    fn check_switch(
        &mut self,
        sel: &ConfigurationSelection,
        active: &Region,
        file: &FilePath,
        site: &Site,
    ) {
        if let ConfigurationSelection::Switch { cases, otherwise } = sel {
            let mut regions = Vec::new();
            for c in cases {
                let r = self.region(Some(&c.when), file, site);
                let r = self.and(&r, active, site);
                regions.push(r);
            }
            for i in 0..regions.len() {
                for j in i + 1..regions.len() {
                    let both = self.and(&regions[i], &regions[j], site);
                    if let Some(w) = self.space.witness(&both) {
                        let witness = self.space.describe_point(&w);
                        self.diag(
                            DiagCode::AsmAmbiguousVariant,
                            Severity::Warning,
                            site.clone(),
                            format!(
                                "configuration cases {} and {} in `{}` overlap at {}; the first applies",
                                i,
                                j,
                                file,
                                describe(&witness)
                            ),
                            Vec::new(),
                        );
                    }
                }
            }
            for c in cases {
                self.check_switch(&c.select, active, file, site);
            }
            self.check_switch(otherwise, active, file, site);
        }
    }
}

fn describe(w: &BTreeMap<String, String>) -> String {
    if w.is_empty() {
        return "every configuration".into();
    }
    w.iter()
        .map(|(k, v)| format!("{} = {}", k, v))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Validate the product under `top`.
pub fn validate_product(
    top: &FilePath,
    revision: Option<&RevisionId>,
    source: &dyn DesignSource,
) -> ValidationReport {
    let mut v = V {
        source,
        space: OptionSpace::new(&OptionsBlock::default()),
        out: ValidationReport::default(),
        docs: BTreeMap::new(),
    };
    let key = (top.clone(), revision.cloned());
    let Some(doc) = v.doc(&key) else {
        v.diag(
            DiagCode::AsmUnknownDefinition,
            Severity::Error,
            Site::of(Subject::Document),
            format!("`{}` is not in the project", top),
            Vec::new(),
        );
        return v.out;
    };
    if let Some(o) = &doc.options {
        v.space = OptionSpace::new(o);
        for (rule, p) in v.space.rule_problems.clone() {
            v.diag(
                option_code(&p),
                Severity::Error,
                Site::with_param(Subject::Document, &format!("options.rules.{}", rule)),
                p.to_string(),
                Vec::new(),
            );
        }
        for c in &o.configurations {
            let site = Site::of(Subject::Configuration {
                name: c.name.clone(),
            });
            match v.space.point(&c.values) {
                Ok(p) if !v.space.is_legal(&p) => {
                    let broken: Vec<String> = o
                        .rules
                        .iter()
                        .filter(|r| !v.space.eval(&r.condition, &p).unwrap_or(true))
                        .map(|r| r.name.clone())
                        .collect();
                    v.diag(
                        DiagCode::AsmIllegalConfiguration,
                        Severity::Error,
                        site,
                        format!("configuration `{}` breaks {}", c.name, broken.join(", ")),
                        Vec::new(),
                    );
                }
                Ok(_) => {}
                Err(ps) => {
                    for p in ps {
                        v.diag(
                            option_code(&p),
                            Severity::Error,
                            site.clone(),
                            format!("configuration `{}`: {}", c.name, p),
                            Vec::new(),
                        );
                    }
                }
            }
        }
        if v.space.legal().is_empty() {
            v.diag(
                DiagCode::AsmIllegalConfiguration,
                Severity::Error,
                Site::of(Subject::Document),
                "the option rules admit no configuration at all".into(),
                Vec::new(),
            );
        }
    }
    // Reachable files and their use edges.
    let mut order: Vec<FileKey> = Vec::new();
    let mut indegree: BTreeMap<FileKey, usize> = BTreeMap::new();
    let mut edges: BTreeMap<FileKey, Vec<FileKey>> = BTreeMap::new();
    let mut seen: BTreeSet<FileKey> = BTreeSet::new();
    let mut queue = vec![key.clone()];
    indegree.insert(key.clone(), 0);
    while let Some(k) = queue.pop() {
        if !seen.insert(k.clone()) {
            continue;
        }
        let Some(d) = v.doc(&k) else { continue };
        let mut kids: BTreeSet<FileKey> = BTreeSet::new();
        for o in &d.occurrences {
            if o.source.definition.kind == DefinitionKind::Assembly {
                kids.insert((o.source.definition.file.clone(), rev_of(&o.source.context)));
            }
        }
        for m in &d.mirrors {
            if let MirrorHand::Opposite { source, .. } = &m.hand {
                if source.definition.kind == DefinitionKind::Assembly {
                    kids.insert((source.definition.file.clone(), rev_of(&source.context)));
                }
            }
        }
        for c in &kids {
            *indegree.entry(c.clone()).or_insert(0) += 1;
            queue.push(c.clone());
        }
        edges.insert(k.clone(), kids.into_iter().collect());
    }
    // Kahn's order (sorted ready set for determinism).
    let mut ready: BTreeSet<FileKey> = indegree
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(k, _)| k.clone())
        .collect();
    let mut indeg = indegree.clone();
    while let Some(k) = ready.iter().next().cloned() {
        ready.remove(&k);
        order.push(k.clone());
        for c in edges.get(&k).cloned().unwrap_or_default() {
            let e = indeg.get_mut(&c).unwrap();
            *e -= 1;
            if *e == 0 {
                ready.insert(c);
            }
        }
    }
    for (k, d) in &indeg {
        if *d > 0 {
            v.diag(
                DiagCode::AsmCycle,
                Severity::Error,
                Site::with_param(Subject::Document, k.0.as_str()),
                format!("`{}` is part of an assembly cycle", k.0),
                Vec::new(),
            );
        }
    }
    let mut ctx: BTreeMap<FileKey, Region> = BTreeMap::new();
    ctx.insert(key.clone(), v.space.legal().clone());
    for k in order {
        let Some(c) = ctx.remove(&k) else { continue };
        let Some(d) = v.doc(&k) else { continue };
        let uses = v.check_file(&d, &c);
        for (child, r) in uses {
            let site = Site::with_param(Subject::Document, child.0.as_str());
            let cur = ctx.remove(&child).unwrap_or_default();
            let merged = v.or(&cur, &r, &site);
            ctx.insert(child, merged);
        }
    }
    v.out
}
