//! A part rename carried into the project files that name the part's items:
//!
//! - [`rename_in_drawing`]: a feature rename reaches a drawing whose views show the part; their
//!   section planes and annotations name the part's features in the part's own forms.
//! - [`rename_input_in_assembly`]: an input rename reaches an assembly that sets that input on an
//!   occurrence of the part: an occurrence's `configuration = { select = inputs, values = { … } }`
//!   (in a switch's cases too), an opposite-hand mirror's configuration, and `override` blocks
//!   whose target resolves, through the sub-assemblies among the files, to an occurrence of the
//!   part (`input.<name> = …` lines and `configuration = …` selections).
//! - [`rename_region_in_study`]: a region rename reaches a `.fea` study whose geometry reads
//!   the part (`file = "…/<part>.part"`): `select name "<region>"` in that geometry names the
//!   part's region and follows. A study's own region names (`region mount : face { … }`) are the
//!   study's and stay. Geometry read from another file (a STEP export, another part) is never
//!   touched, even when its labels spell the same name.

use super::{failure, rename_feature_json};
use crate::assembly::{
    AssemblyDocument, ConfigurationSelection, DefinitionKind, DefinitionRef, MirrorHand,
    Occurrence, OverrideKind, PathSegment,
};
use crate::ids::{FilePath, InputName};
use crate::modules::{folder_of, normalize, ModuleScope};
use crate::syntax::edit::{diff_source_in, EditFailure, SourceChange};
use crate::syntax::{parse_assembly_in, parse_drawing_in, ASSEMBLY, DRAWING};
use runmat_design_syntax::edit::{apply_edits, TextEdit};
use runmat_design_syntax::lexical::{tokenize_fea, FeaTokenKind};
use runmat_design_syntax::{
    parse_fea, FeaDeclarationKind, FeaExpressionKind, FeaModuleItem, FeaStatementKind,
};
use serde_json::Value;
use std::collections::BTreeMap;

/// Carries the rename of feature `from` of the part `part` into the drawing `path`: its views
/// of that part and their annotations follow. `None` when the drawing doesn't name it.
pub fn rename_in_drawing(
    source: &str,
    path: &str,
    scope: &ModuleScope,
    part: &str,
    from: &str,
    to: &str,
) -> Result<Option<SourceChange>, EditFailure> {
    let parsed = parse_drawing_in(path, source, scope);
    let (Some(document), true) = (parsed.document.clone(), parsed.diagnostics.is_empty()) else {
        // A drawing with errors isn't touched; it shows its own problems.
        return Ok(None);
    };
    let part = FilePath::new(part);
    let views: Vec<String> = document
        .views
        .iter()
        .filter(|v| v.model.file == part)
        .map(|v| v.name.to_string())
        .collect();
    if views.is_empty() {
        return Ok(None);
    }
    let mut json = serde_json::to_value(&document).map_err(|e| failure(e.to_string()))?;
    let shows = |entry: &Value, key: &str| {
        entry
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|v| views.iter().any(|n| n == v))
    };
    for (key, field) in [("views", "name"), ("annotations", "view")] {
        for entry in json
            .get_mut(key)
            .and_then(Value::as_array_mut)
            .into_iter()
            .flatten()
        {
            if shows(entry, field) {
                rename_feature_json(entry, from, to, false);
            }
        }
    }
    let new: crate::drawing::DrawingDocument =
        serde_json::from_value(json).map_err(|e| failure(format!("the renamed drawing: {e}")))?;
    if new == document {
        return Ok(None);
    }
    diff_source_in(&DRAWING, path, source, &new, &parsed.uses, &[], scope).map(Some)
}

/// The assemblies among a project's files, read in `scope`, by path. One with errors is left
/// out: it shows its own problems and isn't edited.
pub fn assemblies_in<'a>(
    files: impl IntoIterator<Item = (&'a str, &'a str)>,
    scope: &ModuleScope,
) -> BTreeMap<String, AssemblyDocument> {
    files
        .into_iter()
        .filter(|(p, _)| p.ends_with(".assembly"))
        .filter_map(|(p, t)| {
            let parsed = parse_assembly_in(p, t, scope);
            let ok = parsed.diagnostics.is_empty();
            parsed.document.filter(|_| ok).map(|d| (p.to_owned(), d))
        })
        .collect()
}

fn is_part(definition: &DefinitionRef, part: &FilePath) -> bool {
    definition.kind == DefinitionKind::Part && &definition.file == part
}

/// The definitions the path `target` reaches from `start`, through the assemblies known. A
/// segment that names a mirror instance of an opposite hand reaches the mirror's definition.
fn reached<'a>(
    start: &'a DefinitionRef,
    target: &[PathSegment],
    assemblies: &'a BTreeMap<String, AssemblyDocument>,
) -> Vec<&'a DefinitionRef> {
    let mut at = vec![start];
    for segment in target {
        let mut next = Vec::new();
        for definition in at {
            if definition.kind != DefinitionKind::Assembly {
                continue;
            }
            let Some(assembly) = assemblies.get(definition.file.as_str()) else {
                continue;
            };
            let mirror = segment.instance.as_ref().and_then(|i| {
                assembly.mirrors.iter().find(|m| m.name == i.by).and_then(|m| match &m.hand {
                    MirrorHand::Opposite { source, .. } => Some(&source.definition),
                    MirrorHand::Symmetric { .. } => None,
                })
            });
            if let Some(definition) = mirror {
                next.push(definition);
                continue;
            }
            // Alternatives share a name (the 150% structure): each is reached.
            next.extend(
                assembly
                    .occurrences
                    .iter()
                    .filter(|o| o.name == segment.name)
                    .map(|o| &o.source.definition),
            );
        }
        at = next;
    }
    at
}

fn rename_selection(selection: &mut ConfigurationSelection, from: &str, to: &str) {
    match selection {
        ConfigurationSelection::Inputs { values, .. } => {
            if let Some(value) = values.remove(&InputName::new(from)) {
                values.insert(InputName::new(to), value);
            }
        }
        ConfigurationSelection::Switch { cases, otherwise } => {
            for case in cases {
                rename_selection(&mut case.select, from, to);
            }
            rename_selection(otherwise, from, to);
        }
        ConfigurationSelection::Default | ConfigurationSelection::Named { .. } => {}
    }
}

/// Renames input `from` of `part` on one occurrence: its own selection when it is an
/// occurrence of the part, and every override whose target reaches the part.
fn rename_on_occurrence(
    occurrence: &mut Occurrence,
    part: &FilePath,
    assemblies: &BTreeMap<String, AssemblyDocument>,
    from: &str,
    to: &str,
) {
    if is_part(&occurrence.source.definition, part) {
        rename_selection(&mut occurrence.configuration, from, to);
    }
    let start = occurrence.source.definition.clone();
    // A target replaced by the part takes the part's inputs.
    let replaced: Vec<_> = occurrence
        .overrides
        .iter()
        .filter_map(|o| match &o.set {
            OverrideKind::Replace { source } if is_part(&source.definition, part) => {
                Some(o.target.clone())
            }
            _ => None,
        })
        .collect();
    for item in &mut occurrence.overrides {
        let ours = replaced.contains(&item.target)
            || reached(&start, &item.target.0, assemblies)
                .into_iter()
                .any(|d| is_part(d, part));
        if !ours {
            continue;
        }
        match &mut item.set {
            OverrideKind::Input { name, .. } if name.as_str() == from => {
                *name = InputName::new(to);
            }
            OverrideKind::Configuration { select } => rename_selection(select, from, to),
            _ => {}
        }
    }
}

/// Carries the rename of input `from` of the part `part` into the assembly `path`: the values
/// it sets on occurrences of the part, directly or by override, follow. `assemblies` holds the
/// project's assemblies by path ([`assemblies_in`]), so an override's target is resolved
/// through the sub-assemblies. `None` when the assembly sets no such input.
pub fn rename_input_in_assembly(
    source: &str,
    path: &str,
    scope: &ModuleScope,
    assemblies: &BTreeMap<String, AssemblyDocument>,
    part: &str,
    from: &str,
    to: &str,
) -> Result<Option<SourceChange>, EditFailure> {
    let parsed = parse_assembly_in(path, source, scope);
    let (Some(document), true) = (parsed.document.clone(), parsed.diagnostics.is_empty()) else {
        return Ok(None);
    };
    let part = FilePath::new(part);
    let mut new = document.clone();
    for occurrence in &mut new.occurrences {
        rename_on_occurrence(occurrence, &part, assemblies, from, to);
    }
    for mirror in &mut new.mirrors {
        if let MirrorHand::Opposite {
            source,
            configuration,
            ..
        } = &mut mirror.hand
        {
            if is_part(&source.definition, &part) {
                rename_selection(configuration, from, to);
            }
        }
    }
    if new == document {
        return Ok(None);
    }
    diff_source_in(&ASSEMBLY, path, source, &new, &parsed.uses, &[], scope).map(Some)
}

/// The text of a string token, without its quotes.
fn unquoted(text: &str) -> &str {
    text.strip_prefix('"')
        .and_then(|t| t.strip_suffix('"'))
        .unwrap_or(text)
}

/// Carries the rename of region `from` of the part `part` into the study `path`: in each
/// `geometry` whose `file` is the part, `select name "<from>"` becomes `select name "<to>"`.
/// `None` when no geometry of the study reads the part's region.
pub fn rename_region_in_study(
    source: &str,
    path: &str,
    part: &str,
    from: &str,
    to: &str,
) -> Result<Option<SourceChange>, EditFailure> {
    let module = parse_fea(source);
    if !module.diagnostics.is_empty() {
        return Ok(None);
    }
    let part = normalize(part);
    let folder = folder_of(path);
    let reads_part = |relative: &str| {
        let resolved = if relative.starts_with('/') || folder.is_empty() {
            normalize(relative)
        } else {
            normalize(&format!("{folder}/{relative}"))
        };
        resolved == part
    };
    let mut spans = Vec::new();
    for item in &module.items {
        let FeaModuleItem::Declaration(decl) = item else {
            continue;
        };
        if decl.kind != FeaDeclarationKind::Geometry {
            continue;
        }
        let reads = decl.body.statements.iter().any(|s| match &s.kind {
            FeaStatementKind::Assignment { target, value, .. } => {
                target.segments.len() == 1
                    && target.segments[0].text == "file"
                    && matches!(&value.kind, FeaExpressionKind::String(f) if reads_part(f))
            }
            _ => false,
        });
        if reads {
            spans.push(decl.body.span);
        }
    }
    if spans.is_empty() {
        return Ok(None);
    }
    let stream = tokenize_fea(source);
    let tokens: Vec<_> = stream
        .tokens
        .iter()
        .filter(|t| !matches!(t.kind, FeaTokenKind::Trivia(_)))
        .collect();
    let text = |i: usize| &source[tokens[i].span.start..tokens[i].span.end];
    let mut edits = Vec::new();
    for i in 2..tokens.len() {
        let token = tokens[i];
        let inside = spans
            .iter()
            .any(|s| s.start <= token.span.start && token.span.end <= s.end);
        if inside
            && token.kind == FeaTokenKind::String
            && unquoted(text(i)) == from
            && text(i - 1) == "name"
            && text(i - 2) == "select"
        {
            edits.push(TextEdit::replace(token.span, format!("\"{to}\"")));
        }
    }
    if edits.is_empty() {
        return Ok(None);
    }
    let renamed = apply_edits(source, &edits).map_err(EditFailure::Apply)?;
    Ok(Some(SourceChange {
        edits,
        source: renamed,
        reprinted: false,
    }))
}
