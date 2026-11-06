//! Assembly occurrences as `use` statements.
//!
//! ```text
//! use part bracket from "parts/bracket.part" as left {
//!     configuration = named(name = light)
//!     fixed = false
//! }
//! use assembly wing from "wing.assembly" @ "R3" as port {
//!     override "rib@ribs[3]/clip" {
//!         configuration = named(name = heavy)
//!         input.thickness = 2 mm
//!     }
//!     override spar {
//!         replace = part("parts/spar_b.part", revision = "B")
//!     }
//!     override "rib@ribs[7]" {
//!         suppressed = true
//!     }
//! }
//! ```
//!
//! The kind word is the definition's kind and the name after it is the part the file
//! defines; for a file that defines one part it is the file's stem, each character a name
//! cannot hold written `_` (`use part engine_a from "propulsion/engine-a.part" as left`). `as` names the occurrence;
//! `@ "<revision>"` pins it to a released revision. The block holds every other occurrence
//! field that is not at its default, as `<field> = <value>`.
//!
//! `override <target> { … }` sets typed overrides on a descendant. The target is a name, or
//! a quoted path (`"box/rib@ribs[3]/clip"`: `/` between levels, `@<pattern>[<key>]` for a
//! pattern instance, the key written `3`, `1,2`, `@3` (circular), `"row"`, `mirror`,
//! `point:p1`, `~3`, `lattice:-1,2` or `ring:1@4`).
//! Each line sets one override, in order:
//!
//! | line | override |
//! |---|---|
//! | `configuration = <selection>` | the configuration |
//! | `input.<name> = <value>` | one input value on top of the configuration |
//! | `suppressed = true` | suppression |
//! | `replace = <kind>("<file>"[, part = <name>][, revision = "<rev>"])` | another definition |
//! | `placement = …`, `representation = …`, `material = …`, `property = …`, `flexible = …` | the same fields as on an occurrence |
//! | `when = <condition>` | the condition every override of the block takes |

use crate::refs::reference::XrefContext;
use runmat_design_syntax::ast::{self, Expr, Statement, UseSource};
use runmat_design_syntax::document::BlockEntry;
use runmat_design_syntax::json::{
    is_identifier, lower_field, print_field, quote, Json, LowerError, ValueSyntax, INDENT,
};
use runmat_types::Span;

const KINDS: &[&str] = &["part", "assembly", "circuit"];

/// `(line key, override tag, the override's value field)` for the one-value overrides.
const PLAIN_OVERRIDES: &[(&str, &str, &str)] = &[
    ("configuration", "configuration", "select"),
    ("suppressed", "suppress", "suppressed"),
    ("placement", "position", "placement"),
    ("representation", "representation", "representation"),
    ("material", "material", "material"),
    ("property", "property", "property"),
    ("flexible", "flexible", "flexible"),
];

fn stem(file: &str) -> &str {
    let name = file.rsplit(['/', '\\']).next().unwrap_or(file);
    name.split('.').next().unwrap_or(name)
}

/// The name that stands for a file's single part: its stem, with each character a name
/// cannot hold written `_` (`engine-a.part` → `engine_a`).
fn stem_name(file: &str) -> String {
    let mut name: String = stem(file)
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if name.chars().next().is_none_or(|c| c.is_ascii_digit()) {
        name.insert(0, '_');
    }
    name
}

fn error(span: Span, message: impl Into<String>) -> LowerError {
    LowerError::new(span, message)
}

/// The occurrence as a `use` statement, or `None` when it has no text (a name or file that
/// cannot be written).
pub fn print(name: &str, fields: &[(String, Json)], hook: &dyn ValueSyntax) -> Option<String> {
    let source = fields.iter().find(|(k, _)| k == "source")?.1.clone();
    let definition = source.get("definition")?;
    let kind = definition.get("kind")?.as_str()?;
    if !KINDS.contains(&kind) || !is_identifier(name) {
        return None;
    }
    let pin = match source.get("context") {
        None => String::new(),
        Some(context) => match context.decode::<XrefContext>().ok()? {
            XrefContext::Live => String::new(),
            XrefContext::Pinned { revision } => format!(" @ {}", quote(&revision.0)),
        },
    };
    // In a known document, a design prints by symbol (`crate::modules::unbound`).
    let definition = &hook
        .document_path()
        .and_then(|doc| crate::modules::unbound("definition", definition, doc))
        .unwrap_or_else(|| definition.clone());
    let head = match definition.get("file") {
        Some(file) => file_form(kind, file.as_str()?, definition, &pin, name)?,
        // The module form (`crate::modules::unbound` wrote the symbol).
        None => {
            let symbol = definition.get("name")?.as_str()?;
            if !is_identifier(symbol) {
                return None;
            }
            let from = match definition.get("from").map(|f| f.as_str()) {
                None => String::new(),
                Some(Some(f)) if f.starts_with('.') || f.starts_with('/') => {
                    format!(" from {}", quote(f))
                }
                Some(Some(package)) => format!(" from {package}"),
                Some(None) => return None,
            };
            format!("use {kind} {symbol}{from}{pin} as {name}")
        }
    };
    match hook.document_path() {
        // Interface frames in the occurrence's fields print by symbol too.
        Some(doc) => {
            let fields: Vec<(String, Json)> = fields
                .iter()
                // `replace = part("<file>", …)` overrides keep the file form (`print_replace`).
                .map(|(k, v)| match k.as_str() {
                    "overrides" => (k.clone(), v.clone()),
                    _ => (k.clone(), crate::modules::unbound_tree(k, v, doc)),
                })
                .collect();
            print_rest(head, &fields, hook)
        }
        None => print_rest(head, fields, hook),
    }
}

fn file_form(kind: &str, file: &str, definition: &Json, pin: &str, name: &str) -> Option<String> {
    let symbol = match definition.get("part") {
        Some(Json::String(part)) if *part != stem_name(file) && is_identifier(part) => part.clone(),
        Some(_) => return None,
        None if is_identifier(&stem_name(file)) => stem_name(file),
        None => return None,
    };
    if let Json::Object(entries) = definition {
        if entries
            .iter()
            .any(|(k, _)| !["kind", "file", "part"].contains(&k.as_str()))
        {
            return None;
        }
    }
    Some(format!("use {kind} {symbol} from {}{pin} as {name}", quote(file)))
}

fn print_rest(mut out: String, fields: &[(String, Json)], hook: &dyn ValueSyntax) -> Option<String> {
    let rest: Vec<&(String, Json)> = fields.iter().filter(|(k, _)| k != "source").collect();
    if !rest.is_empty() {
        out.push_str(" {\n");
        for (key, value) in rest {
            if !is_identifier(key) {
                return None;
            }
            if key == "overrides" {
                // Overrides have no other form: a model whose overrides cannot be written
                // as blocks has no text.
                let blocks = print_overrides(value, hook)?;
                for line in blocks.lines() {
                    out.push_str(INDENT);
                    out.push_str(line);
                    out.push('\n');
                }
                continue;
            }
            let text = print_field(key, value, hook, None, 1);
            out.push_str(&format!("{INDENT}{key} = {text}\n"));
        }
        out.push('}');
    }
    Some(out)
}

// ------------------------------------------------------------------------ override blocks

/// An instance key as written in a target path: `3` (linear), `1,2` (grid), `@3`
/// (circular), `"row"` (table row), `mirror`, `point:p1` (sketch point), `~3` (curve
/// step), `lattice:-1,2` (fill) and `ring:1@4` (circular fill).
fn key_text(key: &Json) -> Option<String> {
    let number = |field: &str| match key.get(field) {
        Some(Json::Number(n)) => n.as_u64(),
        _ => None,
    };
    let signed = |field: &str| match key.get(field) {
        Some(Json::Number(n)) => n.as_i64(),
        _ => None,
    };
    let word = |text: &str| {
        !text.is_empty()
            && text
                .chars()
                .all(|c| c.is_alphanumeric() || "_-.".contains(c))
    };
    Some(match key.get("instance")?.as_str()? {
        "linear" => format!("{}", number("step")?),
        "grid" => format!("{},{}", number("first")?, number("second")?),
        "circular" => format!("@{}", number("step")?),
        "row" => {
            let row = key.get("key")?.as_str()?;
            if !word(row) {
                return None;
            }
            format!("\"{row}\"")
        }
        "mirror" => "mirror".into(),
        "point" => {
            let point = key.get("point")?.as_str()?;
            if !word(point) {
                return None;
            }
            format!("point:{point}")
        }
        "curve" => format!("~{}", number("step")?),
        "lattice" => format!("lattice:{},{}", signed("i")?, signed("j")?),
        "ring" => format!("ring:{}@{}", number("ring")?, number("step")?),
        _ => return None,
    })
}

fn key_json(text: &str) -> Option<Json> {
    let tag = |tag: &str, rest: Vec<(&str, Json)>| {
        let mut entries = vec![("instance".to_string(), Json::String(tag.into()))];
        entries.extend(rest.into_iter().map(|(k, v)| (k.to_string(), v)));
        Json::Object(entries)
    };
    let number = |text: &str| -> Option<Json> {
        text.parse::<u32>()
            .ok()
            .map(|n| Json::Number(serde_json::Number::from(n)))
    };
    let signed = |text: &str| -> Option<Json> {
        text.parse::<i32>()
            .ok()
            .map(|n| Json::Number(serde_json::Number::from(n)))
    };
    if text == "mirror" {
        return Some(tag("mirror", vec![]));
    }
    if let Some(row) = text.strip_prefix('"').and_then(|t| t.strip_suffix('"')) {
        return (!row.is_empty()).then(|| tag("row", vec![("key", Json::String(row.into()))]));
    }
    if let Some(point) = text.strip_prefix("point:") {
        return (!point.is_empty())
            .then(|| tag("point", vec![("point", Json::String(point.into()))]));
    }
    if let Some(rest) = text.strip_prefix("lattice:") {
        let (i, j) = rest.split_once(',')?;
        return Some(tag("lattice", vec![("i", signed(i)?), ("j", signed(j)?)]));
    }
    if let Some(rest) = text.strip_prefix("ring:") {
        let (ring, step) = rest.split_once('@')?;
        return Some(tag(
            "ring",
            vec![("ring", number(ring)?), ("step", number(step)?)],
        ));
    }
    if let Some(step) = text.strip_prefix('~') {
        return Some(tag("curve", vec![("step", number(step)?)]));
    }
    if let Some(step) = text.strip_prefix('@') {
        return Some(tag("circular", vec![("step", number(step)?)]));
    }
    if let Some((first, second)) = text.split_once(',') {
        return Some(tag(
            "grid",
            vec![("first", number(first)?), ("second", number(second)?)],
        ));
    }
    Some(tag("linear", vec![("step", number(text)?)]))
}

/// A target path (`box/rib@ribs[3]/clip`) as text, or `None`.
fn path_text(path: &Json) -> Option<String> {
    let Json::Array(segments) = path else {
        return None;
    };
    if segments.is_empty() {
        return None;
    }
    let mut out = Vec::new();
    for segment in segments {
        let name = segment.get("name")?.as_str()?;
        if !is_identifier(name) {
            return None;
        }
        match segment.get("instance") {
            None => out.push(name.to_string()),
            Some(instance) => {
                let by = instance.get("by")?.as_str()?;
                if !is_identifier(by) {
                    return None;
                }
                out.push(format!("{name}@{by}[{}]", key_text(instance.get("key")?)?));
            }
        }
    }
    let text = out.join("/");
    // The text must read back to the same path.
    (path_json(&text).as_ref() == Some(path)).then_some(text)
}

/// Reads a target path.
fn path_json(text: &str) -> Option<Json> {
    let mut segments = Vec::new();
    for segment in text.split('/') {
        let entry = match segment.split_once('@') {
            None => {
                if !is_identifier(segment) {
                    return None;
                }
                vec![("name".to_string(), Json::String(segment.into()))]
            }
            Some((name, rest)) => {
                let (by, key) = rest.split_once('[')?;
                let key = key.strip_suffix(']')?;
                if !is_identifier(name) || !is_identifier(by) {
                    return None;
                }
                vec![
                    ("name".to_string(), Json::String(name.into())),
                    (
                        "instance".to_string(),
                        Json::Object(vec![
                            ("by".to_string(), Json::String(by.into())),
                            ("key".to_string(), key_json(key)?),
                        ]),
                    ),
                ]
            }
        };
        segments.push(Json::Object(entry));
    }
    Some(Json::Array(segments))
}

fn target_head(path: &str) -> String {
    if is_identifier(path) {
        path.to_string()
    } else {
        quote(path)
    }
}

/// `replace = part("<file>", part = <name>, revision = "<rev>")`.
fn print_replace(source: &Json) -> Option<String> {
    let definition = source.get("definition")?;
    let kind = definition.get("kind")?.as_str()?;
    let file = definition.get("file")?.as_str()?;
    if !KINDS.contains(&kind) {
        return None;
    }
    let mut arguments = vec![quote(file)];
    if let Json::Object(entries) = definition {
        for (key, value) in entries {
            match (key.as_str(), value) {
                ("kind" | "file", _) => {}
                ("part", Json::String(part)) if is_identifier(part) => {
                    arguments.push(format!("part = {part}"))
                }
                _ => return None,
            }
        }
    }
    if let Json::Object(entries) = source {
        if entries
            .iter()
            .any(|(k, _)| !["definition", "context"].contains(&k.as_str()))
        {
            return None;
        }
    }
    match source.get("context") {
        None => {}
        Some(context) => match context.decode::<XrefContext>().ok()? {
            XrefContext::Live => {}
            XrefContext::Pinned { revision } => {
                arguments.push(format!("revision = {}", quote(&revision.0)))
            }
        },
    }
    Some(format!("replace = {kind}({})", arguments.join(", ")))
}

/// One override's line inside its block, or `None` when it has no line form.
fn override_line(set: &Json, hook: &dyn ValueSyntax) -> Option<String> {
    let tag = set.get("set")?.as_str()?;
    let Json::Object(entries) = set else {
        return None;
    };
    let fields: Vec<&(String, Json)> = entries.iter().filter(|(k, _)| k != "set").collect();
    if tag == "input" {
        let name = set.get("name")?.as_str()?;
        let value = set.get("value")?;
        if fields.len() != 2 || !is_identifier(name) {
            return None;
        }
        return Some(format!(
            "input.{name} = {}",
            print_field("value", value, hook, None, 2)
        ));
    }
    if tag == "replace" {
        if fields.len() != 1 {
            return None;
        }
        return print_replace(set.get("source")?);
    }
    let (key, _, field) = PLAIN_OVERRIDES.iter().find(|(_, t, _)| *t == tag)?;
    let [(name, value)] = fields.as_slice() else {
        return None;
    };
    (name == field).then(|| format!("{key} = {}", print_field(key, value, hook, None, 2)))
}

/// The occurrence's overrides as `override <target> { … }` blocks, one per run of overrides
/// with the same target and condition; `None` when one has no block form.
fn print_overrides(overrides: &Json, hook: &dyn ValueSyntax) -> Option<String> {
    let Json::Array(items) = overrides else {
        return None;
    };
    if items.is_empty() {
        return None;
    }
    let mut blocks: Vec<(String, Option<&Json>, Vec<String>)> = Vec::new();
    for item in items {
        let Json::Object(entries) = item else {
            return None;
        };
        if entries
            .iter()
            .any(|(k, _)| !["target", "when", "set"].contains(&k.as_str()))
        {
            return None;
        }
        let target = path_text(item.get("target")?)?;
        let when = item.get("when");
        let line = override_line(item.get("set")?, hook)?;
        match blocks.last_mut() {
            Some((t, w, lines)) if *t == target && *w == when => lines.push(line),
            _ => blocks.push((target, when, vec![line])),
        }
    }
    let mut out = String::new();
    for (target, when, lines) in blocks {
        out.push_str(&format!("override {} {{\n", target_head(&target)));
        if let Some(when) = when {
            out.push_str(&format!(
                "{INDENT}when = {}\n",
                print_field("when", when, hook, None, 2)
            ));
        }
        for line in lines {
            out.push_str(INDENT);
            out.push_str(&line.replace('\n', &format!("\n{INDENT}")));
            out.push('\n');
        }
        out.push_str("}\n");
    }
    Some(out)
}

fn lower_replace(expr: &Expr) -> Result<Json, LowerError> {
    let span = expr.span();
    let usage = "`replace = <kind>(\"<file>\"[, part = <name>][, revision = \"<rev>\"])`";
    let Expr::Call(call) = expr else {
        return Err(error(span, format!("expected {usage}")));
    };
    let kind = match call.callee() {
        Some(Expr::Reference(reference)) => match reference.segments().as_slice() {
            [kind] if KINDS.contains(&kind.as_str()) => kind.clone(),
            _ => return Err(error(span, format!("expected {usage}"))),
        },
        _ => return Err(error(span, format!("expected {usage}"))),
    };
    let mut file = None;
    let mut part = None;
    let mut revision = None;
    for argument in call.arguments() {
        match argument {
            ast::Argument::Positional(Expr::String(text)) if file.is_none() => {
                file = Some(text.value())
            }
            ast::Argument::Named(named) => {
                let key = named.name().map(|n| n.text()).unwrap_or_default();
                match (key.as_str(), named.value()) {
                    ("part", Some(Expr::Reference(r))) if r.segments().len() == 1 => {
                        part = Some(r.segments().remove(0))
                    }
                    ("revision", Some(Expr::String(text))) => revision = Some(text.value()),
                    _ => {
                        return Err(error(
                            named.span(),
                            format!("`{key}` is not an argument of {usage}"),
                        ))
                    }
                }
            }
            other => {
                let at = match &other {
                    ast::Argument::Positional(e) => e.span(),
                    ast::Argument::Named(n) => n.span(),
                };
                return Err(error(at, format!("expected {usage}")));
            }
        }
    }
    let file = file.ok_or_else(|| error(span, format!("expected {usage}")))?;
    let mut definition = vec![
        ("kind".to_string(), Json::String(kind)),
        ("file".to_string(), Json::String(file)),
    ];
    if let Some(part) = part {
        definition.push(("part".into(), Json::String(part)));
    }
    let mut source = vec![("definition".to_string(), Json::Object(definition))];
    if let Some(revision) = revision {
        source.push((
            "context".into(),
            Json::Object(vec![
                ("context".into(), Json::String("pinned".into())),
                ("revision".into(), Json::String(revision)),
            ]),
        ));
    }
    Ok(Json::Object(source))
}

/// Reads one `override <target> { … }` block into its overrides.
fn lower_override(
    block: &ast::BlockStatement,
    hook: &dyn ValueSyntax,
    spans: &mut Vec<(String, Span)>,
    first: usize,
) -> Result<Vec<Json>, LowerError> {
    use runmat_design_syntax::ast::Headed;
    let span = block.span();
    let usage = "an override is `override <target> { <line> … }`";
    let target_text = block
        .name()
        .ok_or_else(|| error(block.keyword_span(), usage))?
        .text();
    if !block.clauses().is_empty() {
        return Err(error(span, usage));
    }
    let target = path_json(&target_text).ok_or_else(|| {
        error(
            span,
            format!("`{target_text}` is not a path such as \"box/rib@ribs[3]/clip\""),
        )
    })?;
    let body = block.block().ok_or_else(|| error(span, usage))?;
    let mut when = None;
    let mut sets: Vec<(Json, Span)> = Vec::new();
    for statement in body.statements() {
        let at = statement.span();
        let Statement::Assignment(assignment) = &statement else {
            return Err(error(
                at,
                "an override block holds `<what> = <value>` lines",
            ));
        };
        let Some(value) = assignment.value() else {
            return Err(error(at, "expected `<what> = <value>`"));
        };
        let target = assignment.target();
        let set = match target.as_slice() {
            [word] if word == "when" => {
                if when.is_some() {
                    return Err(error(at, "`when` is given twice"));
                }
                when = Some(lower_field(
                    "when",
                    &value,
                    hook,
                    None,
                    "when",
                    &mut Vec::new(),
                )?);
                continue;
            }
            [word] if word == "replace" => Json::Object(vec![
                ("set".into(), Json::String("replace".into())),
                ("source".into(), lower_replace(&value)?),
            ]),
            [input, name] if input == "input" => Json::Object(vec![
                ("set".into(), Json::String("input".into())),
                ("name".into(), Json::String(name.clone())),
                (
                    "value".into(),
                    lower_field("value", &value, hook, None, "value", &mut Vec::new())?,
                ),
            ]),
            [word] => {
                let Some((key, tag, field)) = PLAIN_OVERRIDES.iter().find(|(k, _, _)| k == word)
                else {
                    return Err(error(
                        assignment.target_span(),
                        format!("`{word}` is not an override; see `override` in the CAD docs"),
                    ));
                };
                Json::Object(vec![
                    ("set".into(), Json::String((*tag).into())),
                    (
                        (*field).into(),
                        lower_field(key, &value, hook, None, key, &mut Vec::new())?,
                    ),
                ])
            }
            _ => {
                return Err(error(
                    assignment.target_span(),
                    "expected `<what> = <value>` or `input.<name> = <value>`",
                ))
            }
        };
        sets.push((set, at));
    }
    if sets.is_empty() {
        return Err(error(span, "an override block sets at least one override"));
    }
    let mut out = Vec::new();
    for (index, (set, at)) in sets.into_iter().enumerate() {
        spans.push((format!("overrides[{}]", first + index), at));
        let mut entry = vec![("target".to_string(), target.clone())];
        if let Some(when) = &when {
            entry.push(("when".into(), when.clone()));
        }
        entry.push(("set".into(), set));
        out.push(Json::Object(entry));
    }
    Ok(out)
}

/// Reads `use <kind> <name> from "<file>" [@ "<pin>"] as <occurrence> [{ … }]`.
pub fn lower(
    declaration: &ast::UseDeclaration,
    hook: &dyn ValueSyntax,
) -> Result<BlockEntry, LowerError> {
    let span = declaration.span();
    let form = "an occurrence is `use <kind> <name> as <name>` (this folder), \
                `use <kind> <name> from \"<folder>\" as <name>`, or `use <alias>.<kind>.<name> as <name>`";
    // (kind, name, the module it is in), or the file form's file.
    let (kind, symbol, module, file) = match (declaration.symbol(), declaration.source()) {
        (Some((kind, symbol)), Some(UseSource::Path(path)))
            if crate::modules::is_file_path(&path) =>
        {
            (kind.text(), symbol.text(), None, Some(path))
        }
        (Some((kind, symbol)), Some(UseSource::Path(folder))) => {
            let folder = if folder.starts_with('.') || folder.starts_with('/') {
                folder
            } else {
                format!("./{folder}")
            };
            (kind.text(), symbol.text(), Some(folder), None)
        }
        (Some((kind, symbol)), Some(UseSource::Package(segments))) => {
            (kind.text(), symbol.text(), Some(segments.join(".")), None)
        }
        (Some((kind, symbol)), None) => (kind.text(), symbol.text(), None, None),
        (None, Some(UseSource::Package(segments))) if segments.len() == 3 => (
            segments[1].clone(),
            segments[2].clone(),
            Some(segments[0].clone()),
            None,
        ),
        _ => return Err(error(span, form)),
    };
    if !KINDS.contains(&kind.as_str()) {
        return Err(error(
            span,
            format!("`{kind}` is not part, assembly or circuit"),
        ));
    }
    let alias = declaration
        .alias()
        .ok_or_else(|| error(span, "an occurrence needs `as <name>`"))?
        .text();
    let mut definition = vec![("kind".to_string(), Json::String(kind))];
    match file {
        Some(file) => {
            definition.push(("file".into(), Json::String(file.clone())));
            if symbol != stem_name(&file) {
                definition.push(("part".into(), Json::String(symbol)));
            }
        }
        None => {
            // A symbol: `crate::modules::bind` resolves it to its file through the folder scope.
            definition.push(("name".into(), Json::String(symbol)));
            if let Some(module) = module {
                definition.push(("from".into(), Json::String(module)));
            }
        }
    }
    let mut source = vec![("definition".to_string(), Json::Object(definition))];
    if let Some(revision) = declaration.pin() {
        source.push((
            "context".into(),
            Json::Object(vec![
                ("context".into(), Json::String("pinned".into())),
                ("revision".into(), Json::String(revision)),
            ]),
        ));
    }
    let source = Json::Object(source);
    let mut fields = vec![("source".to_string(), source.clone())];
    let mut written = vec![("source".to_string(), source, span)];
    let mut spans = Vec::new();
    let head_end = declaration
        .overrides()
        .map_or(span.end, |block| block.span().start);
    let value_span = declaration.overrides().map(|block| block.span());
    if let Some(block) = declaration.overrides() {
        let mut overrides: Vec<Json> = Vec::new();
        let mut override_span: Option<Span> = None;
        let mut override_blocks = 0usize;
        for statement in block.statements() {
            let at = statement.span();
            if let Statement::Block(inner) = &statement {
                use runmat_design_syntax::ast::Headed;
                if inner.keyword() == "override" {
                    let read = lower_override(inner, hook, &mut spans, overrides.len())?;
                    override_blocks += 1;
                    written.push((
                        format!("override[{override_blocks}]"),
                        Json::Array(read.clone()),
                        at,
                    ));
                    overrides.extend(read);
                    override_span = Some(match override_span {
                        Some(s) => Span {
                            start: s.start.min(at.start),
                            end: s.end.max(at.end),
                        },
                        None => at,
                    });
                    continue;
                }
            }
            let Statement::Assignment(assignment) = &statement else {
                return Err(error(
                    at,
                    "an occurrence's block holds `<field> = <value>` lines and `override` blocks",
                ));
            };
            let target = assignment.target();
            let (Some(value), [key]) = (assignment.value(), target.as_slice()) else {
                return Err(error(at, "expected `<field> = <value>`"));
            };
            if key == "overrides" {
                return Err(error(
                    assignment.target_span(),
                    "overrides are `override <target> { … }` blocks",
                ));
            }
            if key == "source" || key == "name" || fields.iter().any(|(k, _)| k == key) {
                return Err(error(at, format!("`{key}` cannot be set here")));
            }
            let item = lower_field(key, &value, hook, None, key, &mut spans)?;
            written.push((key.clone(), item.clone(), value.span()));
            fields.push((key.clone(), item));
        }
        if let Some(at) = override_span {
            spans.push(("overrides".into(), at));
            fields.push(("overrides".into(), Json::Array(overrides)));
        }
    }
    Ok(BlockEntry {
        word: "occurrence".into(),
        name: Some(alias),
        fields,
        spans,
        written,
        head_end,
        value_span,
    })
}
