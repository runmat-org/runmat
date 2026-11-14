//! Hover, completion and signature help from the argument registry and the other word
//! registries ([`super::words`]), and the regenerated entities a reference resolves to.

use lsp_types::{
    CompletionItem, CompletionItemKind, Documentation, MarkupContent, MarkupKind, ParameterInformation,
    ParameterLabel, Position, SignatureHelp, SignatureInformation,
};
use runmat_cad::syntax::arguments::Quantity;
use runmat_design_syntax::Dialect;
use serde_json::Value;

use super::calls::{self, Call};
use super::words;
use super::DesignDocumentAnalysis;
use crate::core::position::position_to_offset;

fn markdown(value: String) -> Documentation {
    Documentation::MarkupContent(MarkupContent {
        kind: MarkupKind::Markdown,
        value,
    })
}

impl DesignDocumentAnalysis {
    fn is_cad(&self) -> bool {
        matches!(self.dialect, Dialect::Part | Dialect::Assembly | Dialect::Drawing)
    }

    /// The circuit compile's part pool: the standard library, the parts the design declares
    /// and the libraries its `use` lines bring in.
    pub(super) fn pool(&self) -> Option<runmat_circuit::library::Pool> {
        let options = runmat_circuit::runtime::CompileOptions::default();
        runmat_circuit::runtime::design_and_pool(self.uri.path(), self.source(), &self.files, &options)
            .ok()
            .map(|(_, pool)| pool)
    }

    /// Facts the Desktop's CAD host forwards for this file (`<feature>/<param>` →
    /// `[count, [facts…]]`). They replace this server's own: the browser server regenerates on the
    /// test kernel, which doesn't match the viewport.
    pub fn with_entities(mut self, entities: Option<&Value>) -> Self {
        if let (Some(entities), Some(projection)) = (entities, self.projection.as_mut()) {
            projection["facts"] = entities.clone();
        }
        self
    }

    /// Markdown for a word no declared name covers.
    pub(super) fn word_hover(&self, offset: usize) -> Option<(String, (usize, usize))> {
        let (start, end) = calls::word_at(self.source(), offset)?;
        let pool = || self.pool();
        let text = words::hover(self.dialect, self.source(), start, end, &pool)?;
        Some((text, (start, end)))
    }

    /// What the reference argument around `offset` resolves to in the regenerated part.
    pub(super) fn reference_facts(&self, offset: usize) -> Option<String> {
        if self.dialect != Dialect::Part {
            return None;
        }
        let call = calls::call_at(self.source(), offset)?;
        let name = call.argument.as_deref()?;
        let argument = call.callee.argument(name)?;
        if !matches!(
            argument.quantity,
            Quantity::Reference | Quantity::References | Quantity::Plane | Quantity::Axis | Quantity::Point
        ) {
            return None;
        }
        let feature = feature_name(self.source(), call.open)?;
        let facts = self.projection.as_ref()?.get("facts")?.as_object()?;
        let prefix = format!("{feature}/{name}");
        let mut count = 0;
        let mut rows = Vec::new();
        for (key, value) in facts {
            let rest = key.strip_prefix(&prefix);
            if !rest.is_some_and(|r| r.is_empty() || r.starts_with('[') || r.starts_with('.')) {
                continue;
            }
            count += value[0].as_u64().unwrap_or(0);
            rows.extend(value[1].as_array().into_iter().flatten().cloned());
        }
        (!facts.is_empty() && facts.keys().any(|k| k.starts_with(&prefix)))
            .then(|| words::entity_lines(count, &rows))
    }

    /// Argument names and allowed values for the call around the cursor, and material ids.
    pub(super) fn call_completions(&self, offset: usize, before: &str) -> Option<Vec<CompletionItem>> {
        if !self.is_cad() {
            return None;
        }
        let source = self.source();
        if let Some(items) = material_completions(before) {
            return Some(items);
        }
        let call = calls::call_at(source, offset)?;
        let typed = before.trim_end_matches(|c: char| c.is_ascii_alphanumeric() || c == '_');
        if call.in_value {
            if !typed.trim_end().ends_with('=') {
                return None;
            }
            let argument = call.argument.as_deref().and_then(|a| call.callee.argument(a))?;
            let mut items: Vec<CompletionItem> = argument
                .choices
                .iter()
                .map(|c| CompletionItem {
                    label: c.word.to_string(),
                    kind: Some(CompletionItemKind::ENUM_MEMBER),
                    detail: Some(format!("{} of {}", argument.name, call.callee.word())),
                    documentation: Some(markdown(c.summary.to_string())),
                    ..CompletionItem::default()
                })
                .collect();
            if argument.quantity == Quantity::Flag {
                for word in ["true", "false"] {
                    items.push(CompletionItem {
                        label: word.into(),
                        kind: Some(CompletionItemKind::KEYWORD),
                        ..CompletionItem::default()
                    });
                }
            }
            return (!items.is_empty()).then_some(items);
        }
        if !typed.trim_end().ends_with(['(', ',']) {
            return None;
        }
        let items: Vec<CompletionItem> = call
            .callee
            .arguments()
            .iter()
            .filter(|a| !a.is_nested() && !call.written.iter().any(|w| w == a.name))
            .enumerate()
            .map(|(i, a)| {
                let flag = a.quantity == Quantity::Flag;
                CompletionItem {
                    label: a.name.to_string(),
                    kind: Some(CompletionItemKind::FIELD),
                    detail: Some(a.quantity.label().to_string()),
                    documentation: Some(markdown(a.summary.to_string())),
                    insert_text: Some(if flag { a.name.to_string() } else { format!("{} = ", a.name) }),
                    sort_text: Some(format!("{i:03}")),
                    ..CompletionItem::default()
                }
            })
            .collect();
        (!items.is_empty()).then_some(items)
    }

    pub fn signature_help(&self, position: &Position) -> Option<SignatureHelp> {
        if !self.is_cad() {
            return None;
        }
        let offset = position_to_offset(self.source(), position);
        let call: Call = calls::call_at(self.source(), offset)?;
        let (label, arguments) = call.callee.signature();
        let open = label.find('(').map_or(0, |i| i + 1);
        let mut at = open;
        let parameters: Vec<ParameterInformation> = arguments
            .iter()
            .map(|a| {
                let start = label[at..].find(a.name).map_or(at, |i| at + i);
                at = start + a.name.len();
                ParameterInformation {
                    label: ParameterLabel::LabelOffsets([start as u32, at as u32]),
                    documentation: Some(markdown(format!("{}: {}", a.quantity.label(), a.summary))),
                }
            })
            .collect();
        let active = match call.argument.as_deref() {
            Some(name) => arguments.iter().position(|a| a.name == name),
            // Written by position (sketch constraints): the argument at that place.
            None => Some(call.position).filter(|p| *p < arguments.len()),
        };
        Some(SignatureHelp {
            signatures: vec![SignatureInformation {
                label,
                documentation: Some(markdown(call.callee.summary().to_string())),
                parameters: Some(parameters),
                active_parameter: active.map(|a| a as u32),
            }],
            active_signature: Some(0),
            active_parameter: active.map(|a| a as u32),
        })
    }
}

/// The feature a call at `open` belongs to: the nearest `feature <name> =` before it.
fn feature_name(source: &str, open: usize) -> Option<String> {
    let head = &source[..open];
    let at = head.rfind("feature ")?;
    let rest = head[at + "feature ".len()..].trim_start();
    let name: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    rest[name.len()..]
        .trim_start()
        .starts_with('=')
        .then_some(name)
        .filter(|n| !n.is_empty())
}

/// Library materials after `material = ` or `material = "`.
fn material_completions(before: &str) -> Option<Vec<CompletionItem>> {
    let trimmed = before
        .trim_end_matches(|c: char| c.is_ascii_alphanumeric() || matches!(c, '_' | ' ' | '-'))
        .trim_end_matches('"')
        .trim_end();
    let quoted = before.matches('"').count() % 2 == 1;
    let head = trimmed.strip_suffix('=')?.trim_end();
    if !head.ends_with("material") {
        return None;
    }
    Some(
        runmat_types::physical::materials::MATERIALS
            .iter()
            .map(|m| CompletionItem {
                label: if quoted { m.label.to_string() } else { m.id.to_string() },
                kind: Some(CompletionItemKind::CONSTANT),
                detail: Some(if quoted { m.id.to_string() } else { m.label.to_string() }),
                documentation: Some(markdown(words::material_hover(m))),
                ..CompletionItem::default()
            })
            .collect(),
    )
}
