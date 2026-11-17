//! Hovers for the words of a design file that are not declared names: keywords, operations and
//! their arguments, allowed values, units, reference path words, materials, and circuit parts
//! and packages.
//!
//! Every sentence comes from a registry the compiler or the docs also read:
//! - [`runmat_cad::syntax::arguments`] for operations, arguments and allowed values;
//! - `runmat_cad::syntax::forms::words::SUMMARIES` and `refs::path_word_summary` for CAD words;
//! - [`runmat_design_syntax::dialect::word_summary`] for words every dialect shares;
//! - `runmat_circuit::syntax::summaries` for circuit and layout words;
//! - `runmat_types::physical` for units and materials;
//! - the circuit compile's part pool for parts and packages.

use super::calls::{self, Callee};
use runmat_cad::syntax::arguments::Argument;
use runmat_design_syntax::Dialect;

/// Markdown for the word at `start..end`, or `None` when no registry knows it.
pub(super) fn hover(
    dialect: Dialect,
    source: &str,
    start: usize,
    end: usize,
    pool: &dyn Fn() -> Option<runmat_circuit::library::Pool>,
) -> Option<String> {
    let word = &source[start..end];
    let before = source[..start].trim_end();
    let after = source[end..].trim_start();
    if let Some(text) = string_hover(dialect, source, start, end, pool) {
        return Some(text);
    }
    if before.ends_with(|c: char| c.is_ascii_digit()) {
        let unit_end = source[start..]
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '/' | '^' | '*')))
            .map_or(source.len(), |i| start + i);
        if let Some(text) = unit_hover(&source[start..unit_end]) {
            return Some(text);
        }
    }
    let cad = matches!(dialect, Dialect::Part | Dialect::Assembly | Dialect::Drawing);
    if cad {
        if let Some(text) = cad_hover(dialect, source, start, end, word, before, after) {
            return Some(text);
        }
    } else if let Some(summary) = runmat_circuit::syntax::summaries::summary(word) {
        return Some(format!("**`{word}`**\n\n{summary}"));
    }
    runmat_design_syntax::dialect::word_summary(word).map(|s| format!("**`{word}`**\n\n{s}"))
}

fn cad_hover(
    dialect: Dialect,
    source: &str,
    start: usize,
    end: usize,
    word: &str,
    before: &str,
    after: &str,
) -> Option<String> {
    // A path word: `feature.body.face.side(…)`, `origin.yz`.
    if before.ends_with('.') {
        if let Some(s) = runmat_cad::syntax::refs::path_word_summary(word) {
            return Some(format!("**`{word}`** (reference path)\n\n{s}"));
        }
    }
    // An operation or a choice written as a call: `extrude(`, `counterbore(`, `line(`.
    if after.starts_with('(') && !before.ends_with('.') {
        let head_start = source[..start]
            .char_indices()
            .rev()
            .find(|(_, c)| !(c.is_ascii_alphanumeric() || *c == '_' || *c == '.'))
            .map_or(0, |(i, c)| i + c.len_utf8());
        let callee_word = &source[head_start..end];
        if let Some(callee) = calls::callee(source, head_start, callee_word) {
            return Some(callee_hover(&callee, super::fence(dialect)));
        }
    }
    if let Some(call) = calls::call_at(source, start) {
        // An argument name: `distance = …`, or a flag written alone (`through_all`).
        let written_alone = after.starts_with([',', ')']) && (before.ends_with([',', '(']));
        if after.starts_with('=') && !after.starts_with("==") || written_alone {
            if let Some(argument) = call.callee.argument(word) {
                return Some(argument_hover(&call.callee, argument));
            }
        }
        // An allowed value: `mode = cut`, `form = counterbore`.
        if call.in_value && before.ends_with('=') {
            if let Some(argument) = call.argument.as_deref().and_then(|a| call.callee.argument(a)) {
                if let Some(choice) = argument.choice(word) {
                    let mut text = format!(
                        "**`{}`** · value of `{}` in `{}`\n\n{}",
                        choice.word,
                        argument.name,
                        call.callee.word(),
                        choice.summary
                    );
                    if !choice.fields.is_empty() {
                        text.push_str("\n\n");
                        text.push_str(&argument_list(choice.fields));
                    }
                    return Some(text);
                }
            }
        }
    }
    let _ = end;
    runmat_cad::syntax::forms::words::SUMMARIES
        .iter()
        .find(|(w, _)| *w == word)
        .map(|(w, s)| format!("**`{w}`**\n\n{s}"))
}

fn callee_hover(callee: &Callee, fence: &str) -> String {
    let (signature, arguments) = callee.signature();
    let mut text = format!("```{fence}\n{signature}\n```\n\n{}", callee.summary());
    if let Callee::Choice(op, argument, _) = callee {
        text.push_str(&format!("\n\nA value of `{}` in `{}`.", argument.name, op.word));
    }
    if !arguments.is_empty() {
        text.push_str("\n\n");
        let owned: Vec<Argument> = arguments.into_iter().copied().collect();
        text.push_str(&argument_list(&owned));
    }
    text
}

/// `- `distance` length (mm), required: How far …` per argument.
fn argument_list(arguments: &[Argument]) -> String {
    arguments
        .iter()
        .filter(|a| !a.is_nested())
        .map(|a| format!("- `{}` {}: {}", a.name, kind_text(a), a.summary))
        .collect::<Vec<_>>()
        .join("\n")
}

fn kind_text(a: &Argument) -> String {
    let mut kind = match a.quantity.unit() {
        Some(unit) => format!("{} ({unit})", a.quantity.label()),
        None => a.quantity.label().to_string(),
    };
    match (a.default, a.required) {
        (Some(d), _) => kind.push_str(&format!(", default `{d}`")),
        (None, true) => kind.push_str(", required"),
        _ => {}
    }
    kind
}

fn argument_hover(callee: &Callee, argument: &Argument) -> String {
    let mut text = format!(
        "**`{}`** · {} · argument of `{}`\n\n{}",
        argument.name,
        kind_text(argument),
        callee.word(),
        argument.summary
    );
    if !argument.choices.is_empty() {
        let choices: Vec<String> = argument
            .choices
            .iter()
            .map(|c| format!("- `{}`: {}", c.word, c.summary))
            .collect();
        text.push_str(&format!("\n\n**Allowed values**\n{}", choices.join("\n")));
    }
    text
}

/// `mm`: millimeter, length; 1 mm = 0.001 m.
pub(super) fn unit_hover(unit: &str) -> Option<String> {
    let facts = runmat_types::physical::unit_facts(unit)?;
    let mut text = format!("**`{unit}`**");
    if let Some(name) = &facts.name {
        text.push_str(&format!(" {name}"));
    }
    if let Some(kind) = facts.kind {
        text.push_str(&format!(" · {kind}"));
    }
    if let Some(si) = facts.si.filter(|si| *si != unit && *si != "1") {
        let scale = format_number(facts.scale);
        if facts.offset == 0.0 {
            text.push_str(&format!("\n\n1 {unit} = {scale} {si}"));
        } else {
            text.push_str(&format!(
                "\n\nx {unit} = ({scale} x + {}) {si}",
                format_number(facts.offset)
            ));
        }
    }
    Some(text)
}

/// Hovers inside a string: a material name, a circuit part number or a package.
fn string_hover(
    dialect: Dialect,
    source: &str,
    start: usize,
    end: usize,
    pool: &dyn Fn() -> Option<runmat_circuit::library::Pool>,
) -> Option<String> {
    let line_start = source[..start].rfind('\n').map_or(0, |i| i + 1);
    let quotes = source[line_start..start].matches('"').count();
    if quotes.is_multiple_of(2) {
        // Not in a string; a bare material id (`material = aluminum_6061_t6`) still reads.
        let before = source[..start].trim_end();
        if before.ends_with('=') && before[..before.len() - 1].trim_end().ends_with("material") {
            return runmat_types::physical::materials::material(&source[start..end])
                .map(material_hover);
        }
        return None;
    }
    let open = line_start + source[line_start..start].rfind('"')?;
    let close = start + source[start..].find('"')?;
    let text = &source[open + 1..close];
    let head = source[line_start..open].trim_end().strip_suffix('=')?.trim_end();
    let field = head
        .rsplit(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .next()
        .unwrap_or("");
    let _ = end;
    match (dialect, field) {
        (Dialect::Part | Dialect::Assembly, "material") => {
            Some(match runmat_types::physical::materials::material_named(text) {
                Some(m) => material_hover(m),
                None => format!(
                    "**{text}** is not a library material. Its density comes from the part's `density` property or the material's name."
                ),
            })
        }
        (Dialect::Circuit | Dialect::CircuitLayout, "part") => part_hover(&pool()?, text),
        (Dialect::CircuitLayout, "package") => package_hover(&pool()?, text),
        _ => None,
    }
}

pub(super) fn material_hover(m: &runmat_types::physical::materials::Material) -> String {
    let mut rows = vec![
        format!("- Density: {} kg/m³", format_number(m.mass_density)),
        format!("- Young's modulus: {} GPa", format_number(m.youngs_modulus / 1e9)),
    ];
    if let Some(y) = m.yield_strength {
        rows.push(format!("- Yield strength: {} MPa", format_number(y / 1e6)));
    }
    rows.push(format!("- Poisson's ratio: {}", format_number(m.poisson_ratio)));
    rows.push(format!(
        "- Thermal conductivity: {} W/(m·K)",
        format_number(m.conductivity)
    ));
    format!(
        "**{}** · material `{}`\n\n{}\n\nSource: {}",
        m.label,
        m.id,
        rows.join("\n"),
        m.source
    )
}

fn part_hover(pool: &runmat_circuit::library::Pool, mpn: &str) -> Option<String> {
    let part = pool.parts.values().find(|p| p.mpn == mpn)?;
    let mut text = format!("**{}** · {}\n\n{}", part.mpn, part.manufacturer, part.description);
    if let Some(package) = pool.packages.get(&part.package) {
        text.push_str(&format!("\n\n- Package: `{}`", package.name));
    }
    text.push_str(&format!("\n- Lifecycle: {:?}", part.lifecycle).to_lowercase());
    for (name, param) in part.params.iter().take(6) {
        let value = param.written.clone().unwrap_or_else(|| format!("{:?}", param.value));
        text.push_str(&format!("\n- {name}: {value}"));
    }
    if let Some(datasheet) = &part.datasheet {
        text.push_str(&format!("\n\n[Datasheet]({datasheet})"));
    }
    Some(text)
}

fn package_hover(pool: &runmat_circuit::library::Pool, name: &str) -> Option<String> {
    let package = pool.packages.values().find(|p| p.name == name)?;
    Some(format!(
        "**`{}`** · package\n\n{}\n\n- Pads: {}\n- Mounting: {}",
        package.name,
        package.description,
        package.pads.len(),
        if package.smd { "surface mount" } else { "through hole" }
    ))
}

/// A number with up to 6 significant digits and no trailing zeros.
pub(super) fn format_number(value: f64) -> String {
    if value == 0.0 {
        return "0".into();
    }
    let magnitude = value.abs().log10().floor() as i32;
    if !(-4..=6).contains(&magnitude) {
        return format!("{value:e}");
    }
    let decimals = (5 - magnitude).max(0) as usize;
    let text = format!("{value:.decimals$}");
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

/// The facts a reference resolves to, in millimeters: `1 face: planar, 2400 mm²`.
pub(super) fn entity_lines(count: u64, facts: &[serde_json::Value]) -> String {
    let mm = |v: &serde_json::Value| -> String {
        let coords: Vec<String> = v
            .as_array()
            .into_iter()
            .flatten()
            .map(|x| format_number(x.as_f64().unwrap_or(0.0) * 1e3))
            .collect();
        format!("({})", coords.join(", "))
    };
    let dir = |v: &serde_json::Value| -> String {
        let coords: Vec<String> = v
            .as_array()
            .into_iter()
            .flatten()
            .map(|x| format_number((x.as_f64().unwrap_or(0.0) * 1e4).round() / 1e4))
            .collect();
        format!("({})", coords.join(", "))
    };
    let mut lines = Vec::new();
    for f in facts {
        let line = match f["entity"].as_str() {
            Some("face") => {
                let mut l = format!(
                    "- {} face, {} mm²",
                    f["surface"].as_str().unwrap_or("other"),
                    format_number(f["area"].as_f64().unwrap_or(0.0) * 1e6)
                );
                if f["normal"].is_array() {
                    l.push_str(&format!(", normal {}", dir(&f["normal"])));
                }
                if let Some(r) = f["radius"].as_f64() {
                    l.push_str(&format!(", radius {} mm", format_number(r * 1e3)));
                }
                l.push_str(&format!(", center {} mm", mm(&f["centroid"])));
                l
            }
            Some("edge") => {
                let mut l = format!(
                    "- {} edge, {} mm long",
                    f["curve"].as_str().unwrap_or("other"),
                    format_number(f["length"].as_f64().unwrap_or(0.0) * 1e3)
                );
                if let Some(r) = f["radius"].as_f64() {
                    l.push_str(&format!(", radius {} mm", format_number(r * 1e3)));
                }
                l.push_str(&format!(", from {} to {} mm", mm(&f["start"]), mm(&f["end"])));
                l
            }
            Some("vertex") => format!("- vertex at {} mm", mm(&f["point"])),
            Some("body") => format!(
                "- body, {} faces, {} edges",
                f["faces"].as_u64().unwrap_or(0),
                f["edges"].as_u64().unwrap_or(0)
            ),
            _ => continue,
        };
        lines.push(line);
    }
    let shown = lines.len() as u64;
    let noun = if count == 1 { "entity" } else { "entities" };
    let mut text = format!("**Resolves to** {count} {noun} in the regenerated part");
    if !lines.is_empty() {
        text.push('\n');
        text.push_str(&lines.join("\n"));
    }
    if count > shown && shown > 0 {
        text.push_str(&format!("\n- … and {} more", count - shown));
    }
    text
}
