//! Shared static form admission for `who` and `whos`. Literal option spelling
//! is decoded only here; runtime admission has its own typed source authority.

use crate::catalog::inference::argument_error;
use runmat_types::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum QueryForm {
    Workspace,
    File,
    Either,
}

pub(super) struct QueryAnalysis {
    pub form: QueryForm,
    pub diagnostics: Vec<InferenceDiagnostic>,
}

pub(super) fn analyze(request: &CallRequest, label: &'static str) -> QueryAnalysis {
    let mut diagnostics = Vec::new();
    let mut possible_file = false;
    let mut definite_file = false;
    let mut index = 0;
    while index < request.arguments.len() {
        let value = &request.arguments[index];
        if invalid_selector(value) {
            diagnostics.push(argument_error(
                "RM-WORKSPACE-QUERY-SELECTOR",
                "workspace selector must be text or a cell of text",
                index,
            ));
        }
        // Runtime option admission is ASCII case-insensitive; static form
        // selection must make the same decision for known literal controls.
        let literal = request
            .literals
            .literal_string_at(index)
            .map(|text| text.to_ascii_lowercase());
        match literal.as_deref() {
            Some("-file") => {
                if definite_file {
                    diagnostics.push(argument_error(
                        "RM-WORKSPACE-QUERY-FILE-DUPLICATE",
                        "-file may only occur once",
                        index,
                    ));
                }
                definite_file = true;
                possible_file = true;
                index += 1;
                if index == request.arguments.len() {
                    diagnostics.push(argument_error(
                        "RM-WORKSPACE-QUERY-FILENAME",
                        "-file requires a filename",
                        index - 1,
                    ));
                    break;
                }
                if invalid_filename(&request.arguments[index]) {
                    diagnostics.push(argument_error(
                        "RM-WORKSPACE-QUERY-FILENAME-TYPE",
                        "filename must be a scalar string or character row",
                        index,
                    ));
                }
            }
            Some("-regexp") => {
                if index + 1 == request.arguments.len()
                    || request
                        .literals
                        .literal_string_at(index + 1)
                        .as_deref()
                        .is_some_and(|next| next.starts_with('-'))
                {
                    diagnostics.push(argument_error(
                        "RM-WORKSPACE-QUERY-REGEXP",
                        "-regexp requires at least one pattern",
                        index,
                    ));
                }
            }
            Some(text) if text.starts_with('-') => {
                diagnostics.push(argument_error(
                    "RM-WORKSPACE-QUERY-OPTION",
                    "unsupported workspace-query option",
                    index,
                ));
            }
            None if may_be_option(value) => possible_file = true,
            _ => {}
        }
        index += 1;
    }
    let form = if definite_file {
        QueryForm::File
    } else if possible_file {
        QueryForm::Either
    } else {
        QueryForm::Workspace
    };
    // Keep the owning identity in the diagnostic text while the static code
    // remains a shared workspace-query admission category.
    for diagnostic in &mut diagnostics {
        diagnostic.message = format!("{label}: {}", diagnostic.message);
    }
    QueryAnalysis { form, diagnostics }
}

fn may_be_option(value: &ValueFact) -> bool {
    matches!(
        &value.kind,
        ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
    )
}

fn invalid_selector(value: &ValueFact) -> bool {
    if matches!(
        &value.residency,
        ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
    ) {
        return true;
    }
    match &value.kind {
        ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character => false,
        ValueKindFact::Cell(cell) => {
            cell.elements.iter().any(invalid_cell_member)
                || (!cell.elements_complete && invalid_cell_member(&cell.element))
        }
        _ => true,
    }
}

fn invalid_cell_member(value: &ValueFact) -> bool {
    if matches!(
        &value.residency,
        ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
    ) {
        return true;
    }
    match &value.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::String => value.shape.is_definitely_nonscalar(),
        ValueKindFact::Character => definitely_not_char_row(&value.shape),
        _ => true,
    }
}

fn invalid_filename(value: &ValueFact) -> bool {
    if matches!(
        &value.residency,
        ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
    ) {
        return true;
    }
    match &value.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::String => value.shape.is_definitely_nonscalar(),
        ValueKindFact::Character => definitely_not_char_row(&value.shape),
        _ => true,
    }
}

fn definitely_not_char_row(shape: &ShapeFact) -> bool {
    shape.known_dims().is_some_and(|dims| {
        dims.first()
            .is_some_and(|rows| rows.is_some_and(|rows| rows != 1))
            || dims
                .iter()
                .skip(2)
                .any(|dimension| dimension.is_some_and(|extent| extent != 1))
    })
}

pub(super) fn apply_form_effects(
    result: &mut CallInference,
    form: QueryForm,
    workspace_may_suspend: bool,
) {
    result.effects.0.insert(EffectKind::MayThrow);
    result
        .capabilities
        .0
        .insert(CapabilityRequirement::HostRuntime);
    if form != QueryForm::File {
        result.effects.0.insert(EffectKind::WorkspaceRead);
    }
    if form != QueryForm::Workspace {
        result.effects.0.insert(EffectKind::FilesystemRead);
        result.effects.0.insert(EffectKind::EnvironmentRead);
        result
            .capabilities
            .0
            .insert(CapabilityRequirement::Filesystem);
    }
    if form != QueryForm::Workspace || workspace_may_suspend {
        result.effects.0.insert(EffectKind::MaySuspend);
    }
}
