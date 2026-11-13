use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

fn text(value: &ValueFact) -> bool {
    super::super::repl_fs::text_input::scalar(value)
}

fn check_text(request: &CallRequest, index: usize, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if request
        .arguments
        .get(index)
        .is_some_and(|value| !text(value))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-MAT-TEXT",
            "MAT filename, option, and selectors must be scalar text",
            index,
        ));
    }
}

fn check_selector_text(
    request: &CallRequest,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if request.arguments.get(index).is_some_and(|value| {
        !matches!(
            value.kind,
            ValueKindFact::Unknown | ValueKindFact::Character | ValueKindFact::String
        )
    }) {
        diagnostics.push(argument_error(
            "RM-CATALOG-MAT-SELECTOR",
            "MAT selectors must be character rows or string values",
            index,
        ));
    }
}

fn loaded_struct() -> ValueFact {
    let mut result = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        BTreeMap::new(),
        false,
    )));
    result.residency = ResidencyFact::Host;
    result
}

fn matfile_object() -> ValueFact {
    let mut result = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(super::MATFILE_CLASS.owned()),
        properties: BTreeMap::new(),
        properties_complete: false,
        handle_semantics: Some(true),
    }));
    result.residency = ResidencyFact::Host;
    result
}

fn check_receiver(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if let Some(receiver) = request.arguments.first() {
        match &receiver.kind {
            ValueKindFact::Unknown => {}
            ValueKindFact::Object(object)
                if object
                    .runtime_class
                    .as_ref()
                    .is_none_or(|class| class.is(super::MATFILE_CLASS)) => {}
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-MATFILE-RECEIVER",
                "receiver must be a matlab.io.MatFile object",
                0,
            )),
        }
    }
}

pub(super) fn load(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !request.arguments.is_empty() {
        check_text(request, 0, &mut diagnostics);
    }
    for index in 1..request.arguments.len() {
        check_selector_text(request, index, &mut diagnostics);
    }
    for index in 0..request.arguments.len() {
        let Some(token) = request.literals.literal_string_at(index) else {
            continue;
        };
        if !token.starts_with('-') {
            continue;
        }
        match super::MatLoadOption::parse(&token) {
            Some(super::MatLoadOption::Regexp)
                if index + 1 == request.arguments.len()
                    || request
                        .literals
                        .literal_string_at(index + 1)
                        .is_some_and(|next| next.starts_with('-')) =>
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-LOAD-REGEXP",
                    "load: '-regexp' requires at least one pattern",
                    index,
                ));
            }
            None => diagnostics.push(argument_error(
                "RM-CATALOG-LOAD-OPTION",
                "unsupported load option",
                index,
            )),
            _ => {}
        }
    }
    if request
        .literals
        .literal_string_at(0)
        .is_some_and(|name| name.trim().is_empty())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOAD-FILENAME",
            "MAT filename must not be empty",
            0,
        ));
    }
    let mut result = finish_fixed(entry, request, loaded_struct(), diagnostics);
    if request.outputs.requested.known_count() == Some(1) {
        result.effects.0.remove(&EffectKind::WorkspaceWrite);
    }
    result
}

pub(super) fn save(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !request.arguments.is_empty() {
        check_text(request, 0, &mut diagnostics);
    }
    for index in 1..request.arguments.len() {
        check_selector_text(request, index, &mut diagnostics);
    }
    for index in 0..request.arguments.len() {
        let Some(token) = request.literals.literal_string_at(index) else {
            continue;
        };
        if !token.starts_with('-') {
            continue;
        }
        match super::MatSaveOption::parse(&token) {
            Some(super::MatSaveOption::Struct | super::MatSaveOption::Regexp)
                if index + 1 == request.arguments.len() =>
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SAVE-OPTION-VALUE",
                    "save option requires a following value",
                    index,
                ));
            }
            Some(super::MatSaveOption::Regexp)
                if request
                    .literals
                    .literal_string_at(index + 1)
                    .is_some_and(|next| next.starts_with('-')) =>
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SAVE-REGEXP",
                    "save: '-regexp' requires at least one pattern",
                    index,
                ));
            }
            None => diagnostics.push(argument_error(
                "RM-CATALOG-SAVE-OPTION",
                "unsupported save option",
                index,
            )),
            _ => {}
        }
    }
    if request
        .literals
        .literal_string_at(0)
        .is_some_and(|name| name.trim().is_empty())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-SAVE-FILENAME",
            "MAT filename must not be empty",
            0,
        ));
    }
    finish_fixed(
        entry,
        request,
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        diagnostics,
    )
}

pub(super) fn matfile(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() || request.arguments.len().is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-MATFILE-ARITY",
            "matfile requires a filename and zero or more complete Writable name-value pairs",
            0,
        ));
    }
    check_text(request, 0, &mut diagnostics);
    if request
        .literals
        .literal_string_at(0)
        .is_some_and(|name| name.trim().is_empty())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-MATFILE-FILENAME",
            "MAT filename must not be empty",
            0,
        ));
    }
    for name_index in (1..request.arguments.len()).step_by(2) {
        check_text(request, name_index, &mut diagnostics);
        if request
            .literals
            .literal_string_at(name_index)
            .is_some_and(|name| super::MatFileOptionName::parse(&name).is_none())
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-MATFILE-OPTION",
                "only the Writable option is supported",
                name_index,
            ));
        }
        let flag_index = name_index + 1;
        if let Some(flag) = request.arguments.get(flag_index) {
            if !matches!(
                flag.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Logical
                    | ValueKindFact::Numeric(NumericFact {
                        domain: NumericDomain::Real,
                        ..
                    })
            ) || flag.shape.is_definitely_nonscalar()
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-MATFILE-WRITABLE",
                    "Writable must be a scalar logical or numeric value",
                    flag_index,
                ));
            }
        }
    }
    finish_fixed(entry, request, matfile_object(), diagnostics)
}

pub(super) fn subsref(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-MATFILE-SUBSREF-ARITY",
            "subsref requires an object and standard subscript",
            0,
        ));
    }
    check_receiver(request, &mut diagnostics);
    let mut output = ValueFact::unknown(DynamicReason::ExternalData);
    output.residency = ResidencyFact::Host;
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn subsasgn(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-MATFILE-SUBSASGN-ARITY",
            "subsasgn requires an object, standard subscript, and rhs value",
            0,
        ));
    }
    check_receiver(request, &mut diagnostics);
    finish_fixed(entry, request, matfile_object(), diagnostics)
}
