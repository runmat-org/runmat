//! The plain-text ECO report.

use super::*;
use std::collections::BTreeMap;

/* ------------------------------------------------------------------------------------------------
 * Report
 * --------------------------------------------------------------------------------------------- */

/// A plain-text ECO report (Altium's Report Changes): every line grouped by type, with status.
pub fn report(plan: &EcoPlan, checks: &BTreeMap<String, Check>, design: &str) -> String {
    let mut out = String::new();
    let title = match plan.direction {
        Direction::Forward => "Update board",
        Direction::Back => "Update schematic",
    };
    out.push_str(&format!(
        "{title} (ECO) for {design}: {} changes\n",
        plan.changes.len()
    ));
    let mut group = None;
    for c in &plan.changes {
        if group != Some(c.kind.group()) {
            group = Some(c.kind.group());
            let n = plan
                .changes
                .iter()
                .filter(|x| x.kind.group() == c.kind.group())
                .count();
            out.push_str(&format!("\n{} ({n})\n", c.kind.group().label()));
        }
        let status = checks
            .get(&c.id)
            .map(|k| k.status.as_str())
            .unwrap_or("not selected");
        let ft = match (&c.from, &c.to) {
            (Some(f), Some(t)) => format!("{f} → {t}"),
            (None, Some(t)) => t.clone(),
            (Some(f), None) => format!("{f} → removed"),
            (None, None) => String::new(),
        };
        out.push_str(&format!(
            "  [{status}] {} {}: {ft}  ({}",
            c.kind.label(),
            c.item,
            c.cause.sheet
        ));
        if let Some(src) = &c.cause.source {
            out.push_str(&format!(", {src}"));
        }
        out.push_str(")\n");
        if let Some(r) = checks.get(&c.id).and_then(|k| k.reason()) {
            out.push_str(&format!("      {r}\n"));
        }
    }
    out
}
