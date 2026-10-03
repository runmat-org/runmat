//! Shared, typed static facts for the FEA object-constructor family.

use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{CallInference, CallRequest, ResidencyFact, StaticClassIdentity, ValueKindFact};

mod facts;
mod kind;

pub(crate) use facts::host_text;
use facts::{object_result, possible_host_study};
pub use kind::FeaConstructorKind;

pub fn infer_constructor(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    kind: FeaConstructorKind,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let path_study = kind == FeaConstructorKind::Study && request.arguments.len() == 1;
    let positional = if path_study { 1 } else { kind.positional() };
    let arity_valid = if kind.is_document_load() || path_study {
        request.arguments.len() == 1
    } else if kind == FeaConstructorKind::Step {
        request.arguments.len() == 2
    } else {
        request.arguments.len() >= positional
            && (request.arguments.len() - positional).is_multiple_of(2)
    };
    if !arity_valid {
        diagnostics.push(argument_error(
            "RM-CATALOG-FEA-ARITY",
            "FEA constructor requires its positional values and complete name/value pairs",
            request.arguments.len(),
        ));
    }
    for index in 0..request.arguments.len().min(positional) {
        if matches!(
            kind,
            FeaConstructorKind::Model | FeaConstructorKind::Study | FeaConstructorKind::AuthorStudy
        ) && index == 1
            && !path_study
        {
            let input = &request.arguments[index];
            let valid = match &input.kind {
                ValueKindFact::Unknown => true,
                ValueKindFact::Object(object) => object
                    .runtime_class
                    .as_ref()
                    .is_none_or(|class| class.is(StaticClassIdentity::new("geometry.Asset"))),
                _ => false,
            } && !matches!(input.residency, ResidencyFact::Device { .. })
                && !matches!(input.shape.element_count(), Some(count) if count != 1);
            if !valid {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-GEOMETRY",
                    "FEA model/study authoring geometry must be a host geometry.Asset",
                    index,
                ));
            }
        } else if kind == FeaConstructorKind::AuthorStudy && index == 2 {
            let summary = &request.arguments[index];
            if matches!(summary.residency, ResidencyFact::Device { .. })
                || !matches!(
                    &summary.kind,
                    ValueKindFact::Unknown
                        | ValueKindFact::Struct(_)
                        | ValueKindFact::Object(_)
                        | ValueKindFact::String
                        | ValueKindFact::Character
                )
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-AUTHORING-SUMMARY",
                    "fea.authorStudy requires a host mesh-authoring summary object or JSON text",
                    index,
                ));
            }
        } else if kind == FeaConstructorKind::Sweep && index == 1 {
            let input = &request.arguments[index];
            let valid = match &input.kind {
                ValueKindFact::Cell(cell) => {
                    possible_host_study(&cell.element)
                        && cell.elements.iter().all(possible_host_study)
                        && !matches!(input.residency, ResidencyFact::Device { .. })
                }
                _ => possible_host_study(input),
            };
            if !valid {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-STUDIES",
                    "fea.sweep requires a host fea.Study or cell of studies",
                    index,
                ));
            }
        } else if !host_text(&request.arguments[index]) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEA-TEXT",
                "FEA structural identifiers, kinds, and paths must be host text scalars",
                index,
            ));
        }
    }
    if !kind.is_document_load() && !path_study && kind != FeaConstructorKind::Step {
        let mut all_option_names_known = true;
        let mut has_profile = false;
        for index in (positional..request.arguments.len()).step_by(2) {
            match request.literals.literal_string_at(index) {
                Some(name) => {
                    if name.eq_ignore_ascii_case("profile") {
                        has_profile = true;
                    }
                }
                None => all_option_names_known = false,
            }
            if !host_text(&request.arguments[index]) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-OPTION",
                    "FEA option names must be host text scalars",
                    index,
                ));
            }
            if let Some(value) = request.arguments.get(index + 1) {
                if matches!(value.residency, ResidencyFact::Device { .. }) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-RESIDENT",
                        "FEA constructor option values do not implicitly gather resident data",
                        index + 1,
                    ));
                }
                if kind == FeaConstructorKind::Sweep
                    && request.literals.literal_string_at(index).as_deref() == Some("failfast")
                    && (!matches!(value.kind, ValueKindFact::Logical | ValueKindFact::Unknown)
                        || matches!(value.shape.element_count(), Some(count) if count != 1))
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-FAILFAST",
                        "fea.sweep FailFast must be a host logical scalar",
                        index + 1,
                    ));
                }
            }
        }
        if matches!(
            kind,
            FeaConstructorKind::Model | FeaConstructorKind::Study | FeaConstructorKind::AuthorStudy
        ) && all_option_names_known
            && !has_profile
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEA-PROFILE",
                "FEA model and study authoring constructors require a Profile option",
                positional,
            ));
        }
    }
    // Literal facts here do not certify nested typed component payloads or geometry.
    finish_fixed(
        entry,
        request,
        object_result(kind, kind.is_document_load() || path_study),
        diagnostics,
    )
}

#[cfg(test)]
mod tests;
