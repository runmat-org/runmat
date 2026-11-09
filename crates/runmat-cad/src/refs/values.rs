//! Published plain values read across files.
//!
//! A file publishes a parameter (`publish station_pitch = input.pitch`, or a derived `let`);
//! another file reads it as [`ValueExpr::Published`]. The read goes through the project
//! catalog by the rules every cross-file reference follows ([`read_published`]): live within
//! a partition, pinned to a released revision across partitions, file and publish renames
//! through the lock's aliases. The value keeps its dimension, so a consumer input of the wrong
//! kind fails with a unit mismatch rather than silently converting.
//!
//! [`PublishedValues`] wraps the configured dynamic-value evaluator: everything else passes
//! through, and published reads are answered from the catalog and memoised per (file, name,
//! context) for one evaluation pass. [`value_read_diagnostics`] turns the reads a document
//! made into diagnostics at the sites that wrote them (via-alias with a rewrite, pin behind
//! with an update), and [`value_error_diagnostic`] gives failed reads their `cad.xref.*` code
//! and repairs.

use crate::diagnostics::{
    describe_ref_problem, Cause, DiagCode, DiagnosticReport, Repair, Severity, Site, Subject,
};
use crate::ids::{FilePath, PublishName, RevisionId};
use crate::model::PartDocument;
use crate::refs::lock::LockSection;
use crate::refs::publish::{read_published, PublishedCatalog, XrefProblem};
use crate::refs::reference::{Reference, XrefContext};
use crate::refs::RefProblem;
use crate::value::{
    DynamicCall, DynamicContract, DynamicOutput, DynamicValueEvaluator, Quantity, ValueError,
    ValueExpr, ValueXref,
};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::BTreeMap;

/// One published value a document read.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ValueRead {
    /// The read as written.
    pub xref: ValueXref,
    /// Where the read landed (after aliases), when it resolved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_file: Option<FilePath>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_name: Option<PublishName>,
    /// The revision read (`None`: the live state).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<RevisionId>,
    pub renamed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub newer_release: Option<RevisionId>,
    pub value: Result<Quantity, ValueError>,
}

/// A [`DynamicValueEvaluator`] that answers published-value reads from the project catalog
/// and passes everything else to `inner`.
pub struct PublishedValues<'a> {
    pub inner: &'a dyn DynamicValueEvaluator,
    pub catalog: Option<&'a dyn PublishedCatalog>,
    pub lock: Option<&'a LockSection>,
    /// The reading file (its partition decides live vs pinned).
    pub consumer: &'a FilePath,
    reads: RefCell<BTreeMap<ValueXref, ValueRead>>,
}

impl<'a> PublishedValues<'a> {
    pub fn new(
        inner: &'a dyn DynamicValueEvaluator,
        catalog: Option<&'a dyn PublishedCatalog>,
        lock: Option<&'a LockSection>,
        consumer: &'a FilePath,
    ) -> Self {
        Self {
            inner,
            catalog,
            lock,
            consumer,
            reads: RefCell::new(BTreeMap::new()),
        }
    }

    /// Every read made so far, in key order.
    pub fn reads(&self) -> Vec<ValueRead> {
        self.reads.borrow().values().cloned().collect()
    }

    pub fn read(&self, xref: &ValueXref) -> Option<ValueRead> {
        self.reads.borrow().get(xref).cloned()
    }

    fn fetch(&self, x: &ValueXref) -> ValueRead {
        let xref = |problem: XrefProblem| ValueError::Xref {
            xref: Box::new(x.clone()),
            problem: Box::new(problem),
        };
        let mut read = ValueRead {
            xref: x.clone(),
            resolved_file: None,
            resolved_name: None,
            revision: None,
            renamed: false,
            newer_release: None,
            value: Err(xref(XrefProblem::NoCatalog)),
        };
        let Some(catalog) = self.catalog else {
            return read;
        };
        match read_published(
            catalog,
            self.lock,
            self.consumer,
            &x.file,
            &x.name,
            &x.context,
        ) {
            Err(p) => read.value = Err(xref(p)),
            Ok(r) => {
                read.resolved_file = Some(r.file.clone());
                read.resolved_name = Some(r.name.clone());
                read.revision = r.target.revision.clone();
                read.renamed = r.renamed;
                read.newer_release = r.newer_release.clone();
                read.value = match r.target.value {
                    Some(q) => Ok(q),
                    None => Err(ValueError::NotAValue {
                        file: r.file.clone(),
                        name: r.name.clone(),
                        found: if r.target.datum.is_some() {
                            "a datum".into()
                        } else {
                            "geometry".into()
                        },
                    }),
                };
            }
        }
        read
    }
}

impl DynamicValueEvaluator for PublishedValues<'_> {
    fn evaluate(
        &self,
        call: &DynamicCall,
        arguments: &[(String, Quantity)],
    ) -> Result<Quantity, String> {
        self.inner.evaluate(call, arguments)
    }
    fn digest(&self, call: &DynamicCall) -> String {
        self.inner.digest(call)
    }
    fn contract(&self, call: &DynamicCall) -> DynamicContract {
        self.inner.contract(call)
    }
    fn evaluate_output(
        &self,
        call: &DynamicCall,
        arguments: &[(String, Quantity)],
    ) -> Result<DynamicOutput, String> {
        self.inner.evaluate_output(call, arguments)
    }
    fn snapshot(&self, key: &str) -> Option<Quantity> {
        self.inner.snapshot(key)
    }
    fn reference(&self, path: &[String]) -> Option<Result<Quantity, String>> {
        self.inner.reference(path)
    }
    fn published(&self, xref: &ValueXref) -> Option<Result<Quantity, ValueError>> {
        if let Some(r) = self.reads.borrow().get(xref) {
            return Some(r.value.clone());
        }
        let read = self.fetch(xref);
        let value = read.value.clone();
        self.reads.borrow_mut().insert(xref.clone(), read);
        Some(value)
    }
}

/// Every published value a part document writes, with the site that writes it: inputs
/// (`input.x`, defaults and configuration overrides), lets (`let.y`) and feature parameters
/// (`feature.f#values`).
pub fn published_sites(doc: &PartDocument) -> Vec<(Site, &ValueExpr)> {
    let mut out: Vec<(Site, &ValueExpr)> = Vec::new();
    let value_site = |name: String| Site::of(Subject::Value { name });
    for i in &doc.inputs {
        out.push((value_site(format!("input.{}", i.name)), &i.default));
    }
    for l in &doc.lets {
        out.push((value_site(format!("let.{}", l.name)), &l.value));
    }
    for c in &doc.configurations {
        for (n, e) in &c.overrides {
            out.push((
                Site::with_param(
                    Subject::Configuration {
                        name: c.name.0.clone(),
                    },
                    &format!("input.{}", n),
                ),
                e,
            ));
        }
    }
    for f in &doc.features {
        for v in f.kind.values() {
            out.push((Site::feature(&f.id, Some("values")), v));
        }
    }
    out.retain(|(_, e)| {
        let mut p = Vec::new();
        e.collect_published(&mut p);
        !p.is_empty()
    });
    out
}

/// Notes for reads that resolved but should be updated: through an alias (rewrite the
/// written name) and pinned behind the latest release (update the pin).
pub fn value_read_diagnostics(
    sites: &[(Site, &ValueExpr)],
    values: &PublishedValues<'_>,
    report: &mut DiagnosticReport,
) {
    for (site, expr) in sites {
        let mut reads = Vec::new();
        expr.collect_published(&mut reads);
        for x in reads {
            let (file, name, context) = (&x.file, &x.name, &x.context);
            let Some(r) = values.read(x) else {
                continue;
            };
            if r.value.is_err() {
                continue;
            }
            let (Some(to_file), Some(to_name)) = (&r.resolved_file, &r.resolved_name) else {
                continue;
            };
            if r.renamed {
                report.push(
                    DiagCode::XrefViaAlias,
                    Severity::Warning,
                    site.clone(),
                    format!(
                        "published value `{}` from `{}` was renamed to `{}` in `{}`; update the reference",
                        name, file, to_name, to_file
                    ),
                    Cause::Root,
                    vec![Repair::RewriteReference {
                        site: site.clone(),
                        reference: Reference::Published {
                            file: Some(to_file.clone()),
                            name: to_name.clone(),
                            context: context.clone(),
                        },
                    }],
                );
            }
            if let (XrefContext::Pinned { revision }, Some(latest)) = (context, &r.newer_release) {
                report.push(
                    DiagCode::XrefPinBehind,
                    Severity::Info,
                    site.clone(),
                    format!(
                        "published value `{}` is pinned to `{}` revision {}; {} is released",
                        name, to_file, revision, latest
                    ),
                    Cause::Root,
                    vec![Repair::UpdatePin {
                        site: site.clone(),
                        file: to_file.clone(),
                        from: revision.clone(),
                        to: latest.clone(),
                    }],
                );
            }
        }
    }
}

/// The code, message and repairs for a value error: cross-file failures get their
/// `cad.xref.*` code and repairs; everything else is `cad.value.failed`.
pub fn value_error_diagnostic(site: &Site, e: &ValueError) -> (DiagCode, String, Vec<Repair>) {
    match e {
        ValueError::Xref { xref, problem } => {
            let reference = Reference::Published {
                file: Some(xref.file.clone()),
                name: xref.name.clone(),
                context: xref.context.clone(),
            };
            let (code, msg, repairs) = describe_ref_problem(
                site,
                &reference,
                &RefProblem::Xref {
                    problem: (**problem).clone(),
                },
            );
            (
                code,
                format!("published value `{}`: {}", xref.name, msg),
                repairs,
            )
        }
        ValueError::NotAValue { .. } => (DiagCode::XrefNotAValue, e.to_string(), Vec::new()),
        _ => (DiagCode::ValueFailed, e.to_string(), Vec::new()),
    }
}
