//! Shared, typed static facts for the FEA object-constructor family.

use super::options::{FeaMaterialAssignmentOption, FeaModelOption, FeaStudyOption, FeaSweepOption};
use crate::{
    catalog::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, ResidencyFact, StaticClassIdentity, ValueFact, ValueKindFact,
};

mod facts;
mod kind;

pub(crate) use facts::host_text;
use facts::{object_result, possible_host_study};
pub use kind::FeaConstructorKind;

fn known_steps_nonempty(value: &ValueFact) -> Option<bool> {
    match &value.kind {
        ValueKindFact::Cell(_) => value.shape.element_count().map(|count| count != 0),
        ValueKindFact::Object(_) if value.shape.element_count() == Some(1) => Some(true),
        _ => None,
    }
}

pub fn infer_constructor(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    kind: FeaConstructorKind,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let positional = kind.positional();
    let arity_valid = request.arguments.len() >= positional
        && (request.arguments.len() - positional).is_multiple_of(2);
    if !arity_valid {
        diagnostics.push(argument_error(
            "RM-CATALOG-FEA-ARITY",
            "FEA constructor requires its positional values and complete name/value pairs",
            request.arguments.len(),
        ));
    }
    for index in 0..request.arguments.len().min(positional) {
        if matches!(kind, FeaConstructorKind::Model | FeaConstructorKind::Study) && index == 1 {
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
        } else if (kind == FeaConstructorKind::Study && index == 2)
            || kind == FeaConstructorKind::MaterialAssignment
        {
            let class = if kind == FeaConstructorKind::Study {
                StaticClassIdentity::new("fea.Model")
            } else if index == 0 {
                StaticClassIdentity::new("fea.PropertyRegionTarget")
            } else {
                StaticClassIdentity::new("fea.Material")
            };
            let input = &request.arguments[index];
            let valid = !matches!(input.residency, ResidencyFact::Device { .. })
                && !matches!(input.shape.element_count(), Some(count) if count != 1)
                && match &input.kind {
                    ValueKindFact::Unknown => true,
                    ValueKindFact::Object(object) => object
                        .runtime_class
                        .as_ref()
                        .is_none_or(|identity| identity.is(class)),
                    _ => false,
                };
            if !valid {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-COMPONENT",
                    "FEA constructor requires its typed host model, property-region target, or material",
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
    {
        let mut all_option_names_known = true;
        // Runtime replaces Steps on every occurrence; an unknown later name
        // could replace them too. Presence alone does not prove nonemptiness.
        let mut effective_steps_nonempty = Some(false);
        let mut has_sequence = false;
        for index in (positional..request.arguments.len()).step_by(2) {
            match request.literals.literal_string_at(index) {
                Some(name) => {
                    let supported = match kind {
                        FeaConstructorKind::Study => FeaStudyOption::decode(&name).is_some(),
                        FeaConstructorKind::Model => match FeaModelOption::decode(&name) {
                            Some(FeaModelOption::Steps) => {
                                effective_steps_nonempty = request
                                    .arguments
                                    .get(index + 1)
                                    .and_then(known_steps_nonempty);
                                true
                            }
                            Some(FeaModelOption::Sequence) => {
                                has_sequence = true;
                                true
                            }
                            Some(_) => true,
                            None => false,
                        },
                        FeaConstructorKind::MaterialAssignment => {
                            FeaMaterialAssignmentOption::decode(&name).is_some()
                        }
                        FeaConstructorKind::Sweep => FeaSweepOption::decode(&name).is_some(),
                        _ => true,
                    };
                    if !supported {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-FEA-OPTION",
                            "unsupported option for this typed FEA constructor",
                            index,
                        ));
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
                    && request
                        .literals
                        .literal_string_at(index)
                        .as_deref()
                        .and_then(FeaSweepOption::decode)
                        == Some(FeaSweepOption::FailFast)
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
        if kind == FeaConstructorKind::Model
            && all_option_names_known
            && ((effective_steps_nonempty == Some(true) && has_sequence)
                || (effective_steps_nonempty == Some(false) && !has_sequence))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEA-SEQUENCE",
                "fea.model requires exactly one of Sequence or Steps",
                positional,
            ));
        }
    }
    // Literal facts here do not certify nested typed component payloads or geometry.
    finish_fixed(entry, request, object_result(kind), diagnostics)
}

#[cfg(test)]
mod tests;
