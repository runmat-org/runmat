//! Reading the layout's values that aren't items: flex bends, fabrication limits, via kinds
//! and design rules, and printing a rule's value back.

use super::*;
use crate::board_rules::{RuleSpec, RuleValue};
use crate::geom::Coord;
use runmat_design_syntax::json::quote;

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

/// `fabrication = { preset = fab_house, outputs = [gerber, drill], layers = ["F.Cu"], variants = [""], folder = "fab/x", zip = true, through_hole = false }`.
pub(super) fn read_fabrication(v: &Val) -> Read<crate::fab::settings::FabSettings> {
    use crate::fab::assembly::{BomColumn, PlaceSides};
    use crate::fab::settings::{FabOutput, FabPreset, FabSettings};
    let mut f = Fields::of(v, "fabrication settings")?;
    let mut out = FabSettings::default();
    if let Some(p) = f.get("preset") {
        let t = text(p)?;
        out.preset = FabPreset::from_word(&t).ok_or_else(|| diagnostic(p.span, format!("`{t}` is not a preset; expected fab_house, assembly, full or custom")))?;
        out.outputs = out.preset.outputs();
    }
    if let Some(o) = f.get("outputs") {
        out.outputs = list(o)?
            .iter()
            .map(|i| {
                let t = text(i)?;
                FabOutput::from_word(&t).ok_or_else(|| diagnostic(i.span, format!("`{t}` is not a fabrication output")))
            })
            .collect::<Read<_>>()?;
    }
    if let Some(l) = f.get("layers") {
        out.layers = Some(list(l)?.iter().map(text).collect::<Read<_>>()?);
    }
    if let Some(l) = f.get("variants") {
        out.variants = list(l)?.iter().map(text).collect::<Read<_>>()?;
    }
    out.folder = f.opt("folder", text)?;
    out.revision = f.opt("revision", text)?;
    out.zip = f.or("zip", false, boolean)?;
    out.include_through_hole = f.or("through_hole", false, boolean)?;
    if let Some(g) = f.get("gerber") {
        let mut gf = Fields::of(g, "Gerber format")?;
        if let Some(u) = gf.get("units") {
            out.gerber.units = unit_of(u)?;
        }
        if let Some(d) = gf.get("decimals") {
            let n = whole(d)?;
            if n != 5 && n != 6 {
                return Err(diagnostic(d.span, "Gerber decimals are 5 or 6"));
            }
            out.gerber.decimals = n as u8;
        }
        gf.finish()?;
    }
    if let Some(p) = f.get("pick_place") {
        let mut pf = Fields::of(p, "pick-and-place format")?;
        if let Some(s) = pf.get("sides") {
            let t = text(s)?;
            out.pick_place.sides = PlaceSides::from_word(&t).ok_or_else(|| diagnostic(s.span, format!("`{t}` is not a side choice; expected both, separate, top or bottom")))?;
        }
        if let Some(u) = pf.get("units") {
            out.pick_place.units = unit_of(u)?;
        }
        pf.finish()?;
    }
    if let Some(b) = f.get("bom") {
        let mut bf = Fields::of(b, "bill of materials format")?;
        if let Some(c) = bf.get("columns") {
            out.bom.columns = list(c)?
                .iter()
                .map(|i| {
                    let t = text(i)?;
                    BomColumn::from_word(&t).ok_or_else(|| diagnostic(i.span, format!("`{t}` is not a BOM column; expected designators, quantity, value, package, mpn or manufacturer")))
                })
                .collect::<Read<_>>()?;
        }
        out.bom.group_by_value = bf.or("group_by_value", true, boolean)?;
        bf.finish()?;
    }
    f.finish()?;
    Ok(out)
}

fn unit_of(v: &Val) -> Read<crate::fab::LengthUnit> {
    let t = text(v)?;
    crate::fab::LengthUnit::from_word(&t).ok_or_else(|| diagnostic(v.span, format!("`{t}` is not a unit; expected mm or inch")))
}

/// The settings as the layout writes them; [`read_fabrication`] reads them back.
pub(super) fn fabrication_text(s: &crate::fab::settings::FabSettings) -> String {
    use crate::fab::settings::FabPreset;
    let mut r = Rec::new();
    r.put("preset", s.preset.word());
    if s.preset == FabPreset::Custom || s.outputs != s.preset.outputs() {
        let words: Vec<&str> = s.outputs.iter().map(|o| o.word()).collect();
        r.put("outputs", format!("[{}]", words.join(", ")));
    }
    let names = |items: &mut dyn Iterator<Item = &String>| format!("[{}]", items.map(|t| quote(t)).collect::<Vec<_>>().join(", "));
    if let Some(l) = &s.layers {
        r.put("layers", names(&mut l.iter()));
    }
    if !s.variants.is_empty() {
        r.put("variants", names(&mut s.variants.iter()));
    }
    if let Some(rev) = &s.revision {
        r.put("revision", quote(rev));
    }
    if let Some(folder) = &s.folder {
        r.put("folder", quote(folder));
    }
    r.flag("zip", s.zip);
    r.flag("through_hole", s.include_through_hole);
    // Formats are written only when they differ from the defaults.
    if s.gerber != crate::fab::GerberFormat::default() {
        r.put("gerber", format!("{{ units = {}, decimals = {} }}", s.gerber.units.word(), s.gerber.decimals));
    }
    if s.pick_place != Default::default() {
        r.put("pick_place", format!("{{ sides = {}, units = {} }}", s.pick_place.sides.word(), s.pick_place.units.word()));
    }
    if s.bom != Default::default() {
        let cols: Vec<&str> = s.bom.columns.iter().map(|c| c.word()).collect();
        r.put("bom", format!("{{ columns = [{}], group_by_value = {} }}", cols.join(", "), s.bom.group_by_value));
    }
    r.text()
}

/// `source` (a `.circuit.layout`) with its `fabrication = { … }` statement set to `settings`:
/// the statement is replaced in place, or added after `limits` (else at the block's start).
/// The rest of the file is left as written.
pub fn with_fabrication_settings(source: &str, settings: &crate::fab::settings::FabSettings) -> Result<String, String> {
    let parsed = super::parse_board("layout", source);
    let line = format!("{} = {}", words::FABRICATION, fabrication_text(settings));
    if let Some(span) = parsed.spans.get(words::FABRICATION) {
        return Ok(format!("{}{line}{}", &source[..span.start], &source[span.end..]));
    }
    let at = match (parsed.spans.get(words::LIMITS), parsed.block) {
        (Some(limits), _) => limits.end,
        (None, Some(block)) => block.start + 1,
        (None, None) => return Err("the layout has no `layout { … }` block".into()),
    };
    let indent = source[..at].rsplit('\n').next().map(|l| l.len() - l.trim_start().len()).filter(|n| *n > 0).unwrap_or(4);
    Ok(format!("{}\n{}{line}{}", &source[..at], " ".repeat(indent), &source[at..]))
}
