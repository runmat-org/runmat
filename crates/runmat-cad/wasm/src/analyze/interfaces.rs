//! `assembly.interface-check`: every interface the open assembly's mates, attachments and
//! publishes stand on, checked against what the definitions publish now.
//!
//! - Each mate is listed with its solve status, and each of its connectors with what it stands
//!   on: a name a part or sub-assembly publishes (which must still be there and still be a
//!   frame, plane, axis or point), the assembly's own sketch, or a frame written in the assembly
//!   (no interface to break; listed for information).
//! - A pinned occurrence's interface is compared with the working copy of its file: a frame that
//!   moved more than `values.tolerance` (mm) or turned is a warning to update the pin.
//! - Occurrences attached to another file's published frame (a skeleton's interface) are checked
//!   the same way.
//! - Each `publish` of this assembly must still name a connector, envelope or occurrence it has.

use super::{Analysis, Row};
use crate::runtime::Files;
use runmat_cad::assembly::model::{
    AnchorRef, AssemblyDocument, AssemblyPublishTarget, DefinitionKind, FrameRef, Placement,
    RelPath,
};
use runmat_cad::assembly::resolve::{MateStatus, NodeId, ResolvedAssembly};
use runmat_cad::assembly::source::{DesignSource, PublishedGeometry};
use runmat_cad::ids::PublishName;
use runmat_cad::model::datum::Frame;
use serde_json::{json, Map, Value};

const ERROR: Option<&str> = Some("error");
const WARNING: Option<&str> = Some("warning");
const INFO: Option<&str> = Some("info");

fn kind_of(g: &PublishedGeometry) -> &'static str {
    match g {
        PublishedGeometry::Frame { .. } => "frame",
        PublishedGeometry::Plane { .. } => "plane",
        PublishedGeometry::Axis { .. } => "axis",
        PublishedGeometry::Point { .. } => "point",
        PublishedGeometry::Pattern { .. } => "pattern",
        PublishedGeometry::Path { .. } => "edges",
        PublishedGeometry::Envelope { .. } => "envelope",
        PublishedGeometry::Value { .. } => "value",
        PublishedGeometry::Entities { .. } => "topology",
    }
}

/// How far `b` is from `a`: origin distance (m) and the largest axis turn (radians).
fn drift(a: &Frame, b: &Frame) -> (f64, f64) {
    let d = (0..3)
        .map(|i| (a.origin[i] - b.origin[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    let turn = [(a.x, b.x), (a.z, b.z)]
        .iter()
        .map(|(u, v)| {
            let c = (0..3).map(|i| u[i] * v[i]).sum::<f64>().clamp(-1.0, 1.0);
            c.acos()
        })
        .fold(0.0, f64::max);
    (d, turn)
}

/// What a named interface of `node`'s definition is now: `Ok((kind, frame))`, or the finding
/// that it's broken.
fn published(
    resolved: &ResolvedAssembly,
    source: &dyn DesignSource,
    node: NodeId,
    name: &PublishName,
    tolerance: f64,
) -> (String, String, Option<&'static str>) {
    let n = resolved.node(node);
    let def = n.definition.clone();
    if n.failed {
        return (
            format!("{def} · {name}"),
            format!("`{def}` doesn't build, so `{name}` can't be checked"),
            ERROR,
        );
    }
    if def.kind == DefinitionKind::Assembly {
        let found = source
            .assembly(&def.file, n.revision.as_ref())
            .ok()
            .and_then(|doc| doc.publishes.iter().find(|p| &p.name == name).cloned());
        return match found {
            Some(p) => {
                let what = match p.target {
                    AssemblyPublishTarget::Connector { connector } => {
                        format!("connector {connector}")
                    }
                    AssemblyPublishTarget::Envelope { envelope } => {
                        return (
                            format!("{def} · {name}"),
                            format!("`{name}` is the envelope `{envelope}`, not a frame"),
                            ERROR,
                        )
                    }
                    AssemblyPublishTarget::Occurrence { path } => format!("occurrence {path}"),
                };
                (format!("{def} · {name}"), format!("Matches ({what})"), None)
            }
            None => (
                format!("{def} · {name}"),
                format!("`{def}` no longer publishes `{name}`"),
                ERROR,
            ),
        };
    }
    let now = source.configured(&def, n.revision.as_ref(), &n.configuration);
    let Ok(now) = now else {
        return (
            format!("{def} · {name}"),
            format!("`{def}` isn't available, so `{name}` can't be checked"),
            ERROR,
        );
    };
    let Some(g) = now.publishes.get(name) else {
        return (
            format!("{def} · {name}"),
            format!("`{def}` no longer publishes `{name}`"),
            ERROR,
        );
    };
    let label = format!("{def} · {name} ({})", kind_of(g));
    let Some(frame) = g.frame() else {
        return (
            label,
            format!(
                "`{name}` is now a {}, not a frame, plane, axis or point",
                kind_of(g)
            ),
            ERROR,
        );
    };
    // A pinned occurrence: the working copy's interface against the pinned one.
    if let Some(revision) = &n.revision {
        if let Some(live) = source
            .configured(&def, None, &n.configuration)
            .ok()
            .and_then(|s| s.publishes.get(name).and_then(PublishedGeometry::frame))
        {
            let (d, turn) = drift(&frame, &live);
            if d > tolerance || turn > 1e-6 {
                return (
                    label,
                    format!(
                        "Pinned at {revision}; the working copy moved it {:.3} mm and turned it {:.2}°: update the pin",
                        d * 1000.0,
                        turn.to_degrees()
                    ),
                    WARNING,
                );
            }
        }
        return (label, format!("Matches (pinned at {revision})"), None);
    }
    (label, "Matches".into(), None)
}

fn status_text(status: Option<&MateStatus>) -> (String, Option<&'static str>) {
    match status {
        None => ("Not solved".into(), INFO),
        Some(MateStatus::Satisfied) => ("Solved".into(), None),
        Some(MateStatus::Inactive) => ("Inactive in this configuration".into(), INFO),
        Some(MateStatus::Redundant { with }) => {
            (format!("Redundant with {}", with.join(", ")), WARNING)
        }
        Some(MateStatus::Conflicting { with }) => {
            (format!("Conflicts with {}", with.join(", ")), ERROR)
        }
        Some(MateStatus::NotConverged) => ("Doesn't solve".into(), ERROR),
        Some(MateStatus::Failed { message }) => (message.clone(), ERROR),
    }
}

fn occurrence_item(resolved: &ResolvedAssembly, node: NodeId) -> Option<String> {
    (node != resolved.root()).then(|| format!("occurrence:{}", resolved.node(node).uuid))
}

/// One row per checked interface.
struct Rows {
    rows: Vec<Row>,
    checked: usize,
}

impl Rows {
    fn push(&mut self, cells: [String; 4], items: Vec<String>, severity: Option<&'static str>) {
        self.rows.push(Row {
            cells: cells.to_vec(),
            items,
            severity,
            legend: None,
            reference: None,
        });
    }
}

/// What connector `name` of `doc` stands on: the row's interface and finding.
fn connector_row(
    doc: &AssemblyDocument,
    resolved: &ResolvedAssembly,
    source: &dyn DesignSource,
    name: &str,
    tolerance: f64,
) -> (String, String, Option<&'static str>, Option<String>) {
    let Some(c) = doc.connector(name) else {
        return (
            "—".into(),
            format!("The connector `{name}` is gone"),
            ERROR,
            None,
        );
    };
    let node = resolved.find(&c.occurrence);
    let item = node.and_then(|n| occurrence_item(resolved, n));
    let at = if c.occurrence.is_empty() {
        "the assembly".to_string()
    } else {
        c.occurrence.to_string()
    };
    match (&c.anchor, node) {
        (_, None) => (
            at.clone(),
            format!("`{at}` isn't in this configuration"),
            ERROR,
            None,
        ),
        (AnchorRef::Published { name }, Some(n)) => {
            let (what, finding, severity) = published(resolved, source, n, name, tolerance);
            (format!("{at}: {what}"), finding, severity, item)
        }
        (AnchorRef::Sketch { sketch, entity, .. }, Some(_)) => (
            format!("sketch {sketch}.{entity}"),
            "Matches (the assembly's own sketch)".into(),
            None,
            item,
        ),
        (AnchorRef::Origin, Some(_)) => (
            format!("{at}: origin"),
            "Not a published interface (the origin)".into(),
            INFO,
            item,
        ),
        (AnchorRef::Explicit { .. }, Some(_)) => (
            format!("{at}: frame written in the assembly"),
            "Not a published interface".into(),
            INFO,
            item,
        ),
    }
}

/// Runs the interface check on the assembly `path`.
pub fn interface_check(
    path: &str,
    text: &str,
    files: &Files,
    configuration: Option<&str>,
    values: &Map<String, Value>,
) -> Result<Analysis, String> {
    // The catalog field is in millimetres.
    let tolerance = values
        .get("tolerance")
        .and_then(Value::as_f64)
        .unwrap_or(0.01)
        / 1000.0;
    let resolution = crate::runtime::resolve_assembly_in(path, text, files, configuration)
        .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
    let resolved = &resolution.resolved;
    let source: &dyn DesignSource = &resolution.source;
    let doc = source
        .assembly(&resolved.top, None)
        .map_err(|e| e.to_string())?;
    let root = resolved.root();
    let mut out = Rows {
        rows: Vec::new(),
        checked: 0,
    };
    for mate in doc.mates.iter().filter(|m| !m.suppressed) {
        let outcome = resolved
            .mates
            .iter()
            .find(|o| o.owner == root && o.name == mate.name);
        let (status, severity) = status_text(outcome.map(|o| &o.status));
        let mut findings = Vec::new();
        for c in mate.connectors() {
            let (what, finding, sev, item) = connector_row(&doc, resolved, source, c, tolerance);
            findings.push((c.to_string(), what, finding, sev, item));
        }
        out.checked += 1;
        let mate_item = format!("mate:{}", mate.name);
        out.push(
            [
                mate.name.clone(),
                mate.kind.name().into(),
                "—".into(),
                status,
            ],
            std::iter::once(mate_item.clone())
                .chain(findings.iter().filter_map(|f| f.4.clone()))
                .collect(),
            severity,
        );
        for (c, what, finding, sev, item) in findings {
            out.push(
                [mate.name.clone(), format!("connector {c}"), what, finding],
                std::iter::once(mate_item.clone()).chain(item).collect(),
                sev,
            );
        }
    }
    // Occurrences attached to another file's published frame.
    for occ in doc.occurrences.iter().filter(|o| !o.suppressed) {
        let Placement::Attach {
            to:
                FrameRef::Interface {
                    file,
                    name,
                    context,
                },
            ..
        } = &occ.placement
        else {
            continue;
        };
        out.checked += 1;
        let item = resolved
            .find(&RelPath::new(&[occ.name.as_str()]))
            .and_then(|n| occurrence_item(resolved, n));
        let def = runmat_cad::assembly::model::DefinitionRef::part(&file.0, None);
        let revision = match context {
            runmat_cad::refs::XrefContext::Pinned { revision } => Some(revision.clone()),
            runmat_cad::refs::XrefContext::Live => None,
        };
        let key = Default::default();
        let (finding, severity) = match source.configured(&def, revision.as_ref(), &key) {
            Err(e) => (e.to_string(), ERROR),
            Ok(s) => match s.publishes.get(name) {
                None => (format!("`{file}` no longer publishes `{name}`"), ERROR),
                Some(g) if g.frame().is_none() => (
                    format!(
                        "`{name}` is now a {}, not a frame, plane, axis or point",
                        kind_of(g)
                    ),
                    ERROR,
                ),
                Some(g) => match (&revision, g.frame()) {
                    (Some(rev), Some(pinned)) => {
                        let live = source
                            .configured(&def, None, &key)
                            .ok()
                            .and_then(|s| s.publishes.get(name).and_then(PublishedGeometry::frame));
                        match live.map(|l| drift(&pinned, &l)) {
                            Some((d, turn)) if d > tolerance || turn > 1e-6 => (
                                format!(
                                    "Pinned at {rev}; the working copy moved it {:.3} mm and turned it {:.2}°: update the pin",
                                    d * 1000.0,
                                    turn.to_degrees()
                                ),
                                WARNING,
                            ),
                            _ => (format!("Matches (pinned at {rev})"), None),
                        }
                    }
                    _ => ("Matches".into(), None),
                },
            },
        };
        out.push(
            [
                occ.name.clone(),
                "attached".into(),
                format!("{file} · {name}"),
                finding,
            ],
            item.into_iter().collect(),
            severity,
        );
    }
    // What this assembly publishes.
    for p in &doc.publishes {
        out.checked += 1;
        let (target, finding, severity, item) = match &p.target {
            AssemblyPublishTarget::Connector { connector } => {
                let (what, finding, sev, item) =
                    connector_row(&doc, resolved, source, connector, tolerance);
                (
                    format!("connector {connector} · {what}"),
                    finding,
                    sev,
                    item,
                )
            }
            AssemblyPublishTarget::Envelope { envelope } => {
                if doc.envelopes.iter().any(|e| &e.name == envelope) {
                    (format!("envelope {envelope}"), "Matches".into(), None, None)
                } else {
                    (
                        format!("envelope {envelope}"),
                        format!("The envelope `{envelope}` is gone"),
                        ERROR,
                        None,
                    )
                }
            }
            AssemblyPublishTarget::Occurrence { path } => match resolved.find(path) {
                Some(n) => (
                    format!("occurrence {path}"),
                    "Matches".into(),
                    None,
                    occurrence_item(resolved, n),
                ),
                None => (
                    format!("occurrence {path}"),
                    format!("`{path}` isn't in this configuration"),
                    ERROR,
                    None,
                ),
            },
        };
        out.push(
            [p.name.to_string(), "published".into(), target, finding],
            item.into_iter().collect(),
            severity,
        );
    }
    let count = |s: &str| out.rows.iter().filter(|r| r.severity == Some(s)).count();
    let (errors, warnings) = (count("error"), count("warning"));
    let plural = |n: usize| if n == 1 { "" } else { "s" };
    let summary = if out.checked == 0 {
        "No mates, attachments or publishes to check".to_string()
    } else {
        format!(
            "{} interface{} checked · {} broken · {} to review",
            out.checked,
            plural(out.checked),
            errors,
            warnings
        )
    };
    let data = json!({
        "checked": out.checked,
        "errors": errors,
        "warnings": warnings,
    });
    Ok(Analysis {
        title: "Interface check".into(),
        summary,
        columns: ["Item", "Kind", "Interface", "Finding"]
            .map(String::from)
            .to_vec(),
        rows: out.rows,
        problems: Vec::new(),
        data,
        settings: vec![format!("Tolerance: {:.3} mm", tolerance * 1000.0)],
        ..Default::default()
    })
}
