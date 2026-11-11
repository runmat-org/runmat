//! A board's pour connect rules (`RULEKIND=PolygonConnect`) as layout rules: how pads and
//! vias join a pour (solid, thermal relief or not at all), the relief's spoke width and gap.
//!
//! - A rule gives one style for every object (`CONNECTSTYLE`, `RELIEFCONDUCTORWIDTH`,
//!   `AIRGAPWIDTH`), or one per object kind under the `THPAD.`, `SMDPAD.` and `VIA.` prefixes:
//!   each becomes its own rule on `pad.tht`, `pad.smd` or `via`.
//! - Selectors are wrapped in `:where(…)` (zero specificity), so the file's priorities alone
//!   order the rules, as the format applies them.
//! - The first scope may be `All`, `InNet('…')` or `InNetClass('…')`; the second must be
//!   `All`. Other scopes are reported as not carried.
//! - A lower `PRIORITY` number wins in the file; layout rules rank higher numbers first, so
//!   the order is turned around.

use super::Record;
use crate::board_rules::{props, RuleSpec, RuleValue};
use crate::import::Report;

fn first_scope(expr: &str) -> Option<String> {
    let e = expr.trim();
    if e.eq_ignore_ascii_case("All") {
        return Some(String::new());
    }
    let arg = |prefix: &str| -> Option<String> {
        let inner = e.strip_prefix(prefix)?.strip_suffix(')')?.trim();
        let name = inner
            .strip_prefix('\'')
            .and_then(|s| s.strip_suffix('\''))
            .unwrap_or(inner);
        (!name.contains(['\'', '"', '(', ')'])).then(|| name.to_owned())
    };
    if let Some(n) = arg("InNetClass(") {
        return Some(format!(":net-class(\"{n}\")"));
    }
    arg("InNet(").map(|n| format!(":net(\"{n}\")"))
}

fn style(s: &str) -> Option<&'static str> {
    match s.trim().to_ascii_lowercase().as_str() {
        "direct" => Some("solid"),
        "relief" => Some("thermal"),
        "noconnect" => Some("none"),
        _ => None,
    }
}

/// The layout rules of one `PolygonConnect` record, or `None` when its scope or style isn't
/// read (reported).
pub(super) fn rules(r: &Record, report: &mut Report) -> Option<Vec<RuleSpec>> {
    if r.str("ENABLED").trim().eq_ignore_ascii_case("FALSE") {
        return Some(Vec::new());
    }
    let name = r.str("NAME").trim().to_owned();
    let (s1, s2) = (r.str("SCOPE1EXPRESSION"), r.str("SCOPE2EXPRESSION"));
    let scope = first_scope(s1).filter(|_| s2.trim().eq_ignore_ascii_case("All"));
    let Some(scope) = scope else {
        report.not_carried(format!(
            "pour connect rules scoped to {} and {}",
            s1.trim(),
            s2.trim()
        ));
        return None;
    };
    let priority = 100 - r.str("PRIORITY").trim().parse::<i32>().unwrap_or(1);
    let kinds: &[(&str, &str)] = if r.str("CONNECTSTYLE").trim().is_empty() {
        &[
            ("THPAD.", "pad.tht"),
            ("SMDPAD.", "pad.smd"),
            ("VIA.", "via"),
        ]
    } else {
        &[("", "*")]
    };
    let mut out = Vec::new();
    for (prefix, kind) in kinds {
        let key = |k: &str| format!("{prefix}{k}");
        let Some(st) = style(r.str(&key("CONNECTSTYLE"))) else {
            report.not_carried(format!(
                "pour connect style {:?}",
                r.str(&key("CONNECTSTYLE"))
            ));
            continue;
        };
        // Zero specificity, as the file has none: its priorities alone decide between rules.
        let inner = if scope.is_empty() {
            kind.to_string()
        } else {
            format!("{kind}{scope}")
        };
        let selector = if inner == "*" {
            inner
        } else {
            format!("*:where({inner})")
        };
        let id = if prefix.is_empty() {
            name.clone()
        } else {
            format!(
                "{name} {}",
                prefix.trim_end_matches('.').to_ascii_lowercase()
            )
        };
        let mut spec = RuleSpec::single(&id, &selector)
            .priority(priority)
            .set(props::ZONE_CONNECT, RuleValue::Text(st.into()));
        if st == "thermal" {
            for (field, prop) in [
                ("RELIEFCONDUCTORWIDTH", props::SPOKE_WIDTH),
                ("AIRGAPWIDTH", props::THERMAL_GAP),
            ] {
                let v = r.len(&key(field));
                if v > 0 {
                    spec = spec.set(prop, RuleValue::Len(v));
                }
            }
            report.approximated("thermal reliefs take RunMat's spoke angle and count");
        }
        out.push(spec);
    }
    Some(out)
}
