//! Reading the layout's values that aren't items: flex bends, fabrication limits, via kinds
//! and design rules, and printing a rule's value back.

use super::*;
use crate::board_rules::{RuleSpec, RuleValue};
use crate::geom::Coord;

pub(super) fn read_bend(v: &Val) -> Read<Bend> {
    let mut f = Fields::of(v, "bend")?;
    let b = Bend {
        a: point(f.need("from")?)?,
        b: point(f.need("to")?)?,
        radius: len(f.need("radius")?)?,
        angle: angle(f.need("angle")?)?,
        keepout: f.or("keepout", 0, len)?,
    };
    f.finish()?;
    Ok(b)
}

pub(super) const LIMIT_KEYS: [&str; 9] = [
    "clearance",
    "track",
    "via_diameter",
    "drill",
    "annular",
    "hole_to_hole",
    "copper_to_edge",
    "silk_to_pad",
    "microvia_drill",
];

pub(super) fn limit_fields(l: &mut FabLimits) -> [&mut Coord; 9] {
    [
        &mut l.min_clearance,
        &mut l.min_track,
        &mut l.min_via_diameter,
        &mut l.min_drill,
        &mut l.min_annular,
        &mut l.min_hole_to_hole,
        &mut l.min_copper_to_edge,
        &mut l.min_silk_to_pad,
        &mut l.min_microvia_drill,
    ]
}

pub(super) fn read_limits(v: &Val) -> Read<FabLimits> {
    let mut f = Fields::of(v, "limits")?;
    let mut l = FabLimits::default();
    for (key, field) in LIMIT_KEYS.iter().zip(limit_fields(&mut l)) {
        if let Some(v) = f.opt(key, len)? {
            *field = v;
        }
    }
    f.finish()?;
    Ok(l)
}

pub(super) const VIA_KINDS: [(ViaKind, &str); 4] = [
    (ViaKind::Through, "through"),
    (ViaKind::Blind, "blind"),
    (ViaKind::Buried, "buried"),
    (ViaKind::Micro, "micro"),
];

/// A rule value: `0.2 mm` is a length, `2` a whole number, `2.5` a number, `true` a flag,
/// `[through, micro]` via kinds, anything else text.
pub(super) fn rule_value(v: &Val) -> Read<RuleValue> {
    Ok(match &v.v {
        V::Qty(..) => RuleValue::Len(len(v)?),
        V::Num(t) if t.contains(['.', 'e', 'E']) => RuleValue::Num(num(v)?),
        V::Num(_) => RuleValue::Int(whole(v)?),
        V::Word(w) if w == "true" || w == "false" => RuleValue::Bool(boolean(v)?),
        V::Word(w) | V::Str(w) => RuleValue::Text(w.clone()),
        V::List(items) => RuleValue::ViaKinds(
            items
                .iter()
                .map(|i| {
                    let t = text(i)?;
                    VIA_KINDS
                        .iter()
                        .find(|(_, n)| *n == t)
                        .map(|(k, _)| *k)
                        .ok_or_else(|| {
                            diagnostic(
                                i.span,
                                format!("`{t}` is not a via kind; expected through, blind, buried or micro"),
                            )
                        })
                })
                .collect::<Read<_>>()?,
        ),
        V::Rec(_) => return Err(diagnostic(v.span, "a rule value is a length, number, flag or text")),
    })
}

pub(super) fn rule_value_text(v: &RuleValue) -> String {
    match v {
        RuleValue::Len(l) => format!("{} mm", mm_text(*l)),
        RuleValue::Num(n) => {
            let t = f64_text(*n);
            if t.contains(['.', 'e', 'E', 'N', 'i']) {
                t
            } else {
                format!("{t}.0")
            }
        }
        RuleValue::Int(i) => i.to_string(),
        RuleValue::Bool(b) => b.to_string(),
        RuleValue::Text(t) => name_text(t),
        RuleValue::ViaKinds(kinds) => {
            let names: Vec<&str> = kinds
                .iter()
                .map(|k| VIA_KINDS.iter().find(|(x, _)| x == k).unwrap().1)
                .collect();
            format!("[{}]", names.join(", "))
        }
    }
}

pub(super) fn read_rule(id: &str, v: &Val) -> Read<RuleSpec> {
    let mut f = Fields::of(v, "rule")?;
    let mut spec = match (f.get("select"), f.get("between")) {
        (Some(s), None) => RuleSpec::single(id, &text(s)?),
        (None, Some(b)) => match list(b)? {
            [a, c] => RuleSpec::pair(id, &text(a)?, &text(c)?),
            _ => return Err(diagnostic(b.span, "`between` takes two selectors")),
        },
        _ => {
            return Err(diagnostic(
                v.span,
                "a rule has `select = \"<selector>\"` or `between = [\"<a>\", \"<b>\"]`",
            ))
        }
    };
    spec.priority = f.or("priority", 0, whole)? as i32;
    for (k, v) in f.rest() {
        spec.decls.push((k.to_owned(), rule_value(v)?));
    }
    Ok(spec)
}
