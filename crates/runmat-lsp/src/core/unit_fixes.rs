//! Quick fixes for the one wrong-unit diagnostic, the same in every language.
//!
//! `.fea`, `.part`, `.assembly`, `.drawing`, `.circuit` and `.circuit.layout` all report a wrong
//! unit with `runmat_types::physical::unit_mismatch_message`, underlining the value. For each
//! such diagnostic in the requested range this offers the value's own number in each unit of the
//! expected kind (`10 kg` → `10 mm`, `10 cm`, `10 m`, `10 in`). It never changes the number.

use lsp_types::{
    CodeAction, CodeActionKind, CodeActionOrCommand, Diagnostic, Range, TextEdit, Url,
    WorkspaceEdit,
};
use runmat_types::physical::{expected_si_example, unit_fixes};
use std::collections::HashMap;

use super::position::position_to_offset;

/// The unit quick fixes for `diagnostics` (as published for `source`) that touch `range`.
pub fn unit_code_actions(
    uri: &Url,
    source: &str,
    diagnostics: &[Diagnostic],
    range: Range,
) -> Vec<CodeActionOrCommand> {
    let mut actions = Vec::new();
    for diagnostic in diagnostics {
        if diagnostic.range.end < range.start || range.end < diagnostic.range.start {
            continue;
        }
        let Some(example) = expected_si_example(&diagnostic.message) else {
            continue;
        };
        let start = position_to_offset(source, &diagnostic.range.start);
        let end = position_to_offset(source, &diagnostic.range.end);
        let Some(value) = source.get(start..end) else {
            continue;
        };
        for replacement in unit_fixes(value, example) {
            let edit = TextEdit {
                range: diagnostic.range,
                new_text: replacement.clone(),
            };
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: format!("Write `{replacement}`"),
                kind: Some(CodeActionKind::QUICKFIX),
                diagnostics: Some(vec![diagnostic.clone()]),
                edit: Some(WorkspaceEdit {
                    changes: Some(HashMap::from([(uri.clone(), vec![edit])])),
                    ..WorkspaceEdit::default()
                }),
                ..CodeAction::default()
            }));
        }
    }
    actions
}
