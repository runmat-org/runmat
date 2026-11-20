//! `assembly.copy-with-mates`: components copied together with their connectors and mates.
//!
//! Each copy takes the next unused name of its original (`block` → `block2`) and starts where
//! the original sits. The connectors on a copied component are copied onto the copy, and so is
//! every mate that touches one:
//!
//! - a mate between two copied components joins the two copies;
//! - a mate to anything outside the copy (another component, the assembly's ground or one of its
//!   sketches) takes the next picked new reference, in the order the assembly lists its mates.
//!   A pick is a face, edge, vertex, datum, connector or sketch point, made into a connector the
//!   same way `assembly.mate` makes one. When the picks run out, the mate keeps its original
//!   reference.
//!
//! A copy with mates floats, so the copied mates place it.

use super::anchors::Publisher;
use super::mate::{connector, resolve};
use super::{model, Compiled};
use crate::features::picks_of;
use runmat_cad::assembly::model::{AssemblyDocument, FrameRef, MateDef, Placement};
use runmat_cad::syntax::edit::EditFailure;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

type Values = Map<String, Value>;

/// The stem (its trailing number dropped) numbered from `from` up: `block` → `block2`.
fn next_name(taken: impl Fn(&str) -> bool, stem: &str, from: usize) -> String {
    let base = stem.trim_end_matches(|c: char| c.is_ascii_digit());
    let base = if base.is_empty() { stem } else { base };
    (from..)
        .map(|n| format!("{base}{n}"))
        .find(|n| !taken(n))
        .expect("an unused name")
}

pub(super) fn copy_with_mates(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    publisher: &mut Publisher,
    names: &[String],
    values: &Values,
) -> Result<String, EditFailure> {
    // Copies of the components, by original name.
    let mut copies: BTreeMap<String, String> = BTreeMap::new();
    for name in names {
        if !document.occurrences.iter().any(|o| &o.name == name) {
            return Err(model(format!(
                "`{name}` is inside a sub-assembly; copy it in that sub-assembly"
            )));
        }
        let copy_name = next_name(
            |n| document.occurrences.iter().any(|o| o.name == n) || copies.values().any(|c| c == n),
            name,
            2,
        );
        copies.insert(name.clone(), copy_name);
    }
    // Connectors on a copied component, copied onto its copy.
    let mut connectors: BTreeMap<String, String> = BTreeMap::new();
    let on_copied: Vec<_> = document
        .connectors
        .iter()
        .filter(|c| {
            c.occurrence
                .0
                .first()
                .is_some_and(|s| s.instance.is_none() && copies.contains_key(&s.name))
        })
        .cloned()
        .collect();
    for mut c in on_copied {
        let new_name = next_name(
            |n| {
                document.connectors.iter().any(|x| x.name == n)
                    || connectors.values().any(|x| x == n)
            },
            &format!("{}_c", copies[&c.occurrence.0[0].name]),
            1,
        );
        connectors.insert(c.name.clone(), new_name.clone());
        c.occurrence.0[0].name = copies[&c.occurrence.0[0].name].clone();
        c.name = new_name;
        document.connectors.push(c);
    }
    // The occurrences, after their connectors (an attached placement may name one).
    for name in names {
        let mut copy = document
            .occurrences
            .iter()
            .find(|o| &o.name == name)
            .cloned()
            .expect("checked above");
        copy.name = copies[name].clone();
        if let Placement::Attach {
            to: FrameRef::Connector { connector },
            ..
        } = &mut copy.placement
        {
            if let Some(mapped) = connectors.get(connector) {
                *connector = mapped.clone();
            }
        }
        let at = document
            .occurrences
            .iter()
            .position(|o| &o.name == name)
            .map_or(document.occurrences.len(), |i| i + 1);
        document.occurrences.insert(at, copy);
    }
    // The mates touching a copied connector.
    let mut references = picks_of(values, "references").into_iter();
    let mut copied_mates = 0;
    let mut retargeted = 0;
    let mut kept = 0;
    let touching: Vec<MateDef> = document
        .mates
        .iter()
        .filter(|m| m.connectors().iter().any(|c| connectors.contains_key(*c)))
        .cloned()
        .collect();
    let mut floating = BTreeMap::new();
    for original in touching {
        let mut mate = original.clone();
        let mut sides: Vec<&mut String> = vec![&mut mate.a, &mut mate.b];
        let mut outside = Vec::new();
        for (i, side) in sides.iter_mut().enumerate() {
            match connectors.get(side.as_str()) {
                Some(mapped) => **side = mapped.clone(),
                None => outside.push(i),
            }
        }
        for i in outside {
            let Some(pick) = references.next() else {
                kept += 1;
                continue;
            };
            let resolved = resolve(document, compiled, &pick, "the new reference")?;
            if copies.values().any(|c| c == &resolved.occurrence) {
                return Err(model("pick a new reference outside the copied components"));
            }
            let side = if i == 0 { &original.a } else { &original.b };
            let flip = document.connector(side).is_some_and(|c| c.flip);
            let made = connector(document, publisher, &resolved, flip)?;
            if i == 0 {
                mate.a = made;
            } else {
                mate.b = made;
            }
            retargeted += 1;
        }
        // The kind's own connectors (a symmetric mate's plane, a width mate's other faces).
        remap_kind(&mut mate.kind, &connectors);
        mate.name = next_name(
            |n| document.mates.iter().any(|m| m.name == n),
            &original.name,
            2,
        );
        for c in [&mate.a, &mate.b] {
            if let Some(o) = document.connector(c).and_then(|c| c.occurrence.0.first()) {
                if copies.values().any(|v| v == &o.name) {
                    floating.insert(o.name.clone(), ());
                }
            }
        }
        document.mates.push(mate);
        copied_mates += 1;
    }
    for o in document.occurrences.iter_mut() {
        if floating.contains_key(&o.name) {
            o.fixed = false;
        }
    }
    if references.next().is_some() {
        return Err(model(format!(
            "{} new references picked for {} outside mate ends; pick one per mate to replace",
            picks_of(values, "references").len(),
            retargeted + kept
        )));
    }
    let list = copies.values().cloned().collect::<Vec<_>>().join(", ");
    let mut summary = format!(
        "Copied {list} with {copied_mates} mate{}",
        if copied_mates == 1 { "" } else { "s" }
    );
    if retargeted > 0 {
        summary.push_str(&format!(", {retargeted} on new references"));
    }
    if kept > 0 {
        summary.push_str(&format!(", {kept} on the original references"));
    }
    Ok(summary)
}

fn remap_kind(
    kind: &mut runmat_cad::assembly::model::MateKind,
    connectors: &BTreeMap<String, String>,
) {
    use runmat_cad::assembly::model::MateKind;
    let map = |c: &mut String| {
        if let Some(m) = connectors.get(c.as_str()) {
            *c = m.clone();
        }
    };
    match kind {
        MateKind::Width { a2, b2 } => {
            map(a2);
            map(b2);
        }
        MateKind::Symmetric { plane } => map(plane),
        _ => {}
    }
}
