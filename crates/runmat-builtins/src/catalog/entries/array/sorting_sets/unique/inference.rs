use super::super::{
    facts::*,
    set_facts::{output_shape, ElementShape, RecordMode},
};
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    count(request, 1, &mut diagnostics);
    let input = argument(request, 0);
    accepted_data(&input, true, false, &mut diagnostics, 0);
    let mut rows = RecordMode::Elements;
    let mut order = None;
    let mut occurrence = None;
    let mut index = 1;
    while index < request.arguments.len() {
        match text_option(request, index, &mut diagnostics).as_deref() {
            Some("rows") => rows = RecordMode::Rows,
            Some(option @ ("sorted" | "stable")) => {
                if order
                    .as_ref()
                    .is_some_and(|previous: &String| previous != option)
                {
                    diagnostics.push(argument_error(
                        "RM-UNIQUE-ORDER",
                        "conflicting unique order flags",
                        index,
                    ));
                }
                order = request.literals.literal_string_at(index);
            }
            Some(option @ ("first" | "last")) => {
                if occurrence
                    .as_ref()
                    .is_some_and(|previous: &String| previous != option)
                {
                    diagnostics.push(argument_error(
                        "RM-UNIQUE-OCCURRENCE",
                        "conflicting first/last flags",
                        index,
                    ));
                }
                occurrence = Some(option.to_owned());
            }
            Some("treatmissingasdistinct") => {
                index += 1;
                if index == request.arguments.len() {
                    diagnostics.push(argument_error(
                        "RM-UNIQUE-OPTION",
                        "TreatMissingAsDistinct requires a scalar logical value",
                        index - 1,
                    ));
                    break;
                }
                let value = &request.arguments[index];
                if value.shape.is_definitely_nonscalar()
                    || !matches!(
                        value.kind,
                        ValueKindFact::Unknown
                            | ValueKindFact::Logical
                            | ValueKindFact::Numeric(NumericFact {
                                domain: NumericDomain::Real,
                                ..
                            })
                    )
                    || request
                        .literals
                        .numeric_at(index)
                        .is_some_and(|v| v != 0.0 && v != 1.0)
                {
                    diagnostics.push(argument_error(
                        "RM-UNIQUE-OPTION",
                        "TreatMissingAsDistinct requires logical or numeric zero/one",
                        index,
                    ));
                }
            }
            Some(_) => diagnostics.push(argument_error(
                "RM-UNIQUE-OPTION",
                "unsupported unique option",
                index,
            )),
            None if rows != RecordMode::Rows => rows = RecordMode::Unknown,
            None => {}
        }
        index += 1;
    }
    if rows == RecordMode::Rows {
        matrix(&input, 0, &mut diagnostics);
    }
    let mut values = input.clone();
    values.shape = output_shape(&input, None, rows, ElementShape::PreserveVector);
    let mapping_length = match rows {
        RecordMode::Rows => dimension(&input.shape, 0),
        RecordMode::Elements => input.shape.element_count(),
        RecordMode::Unknown => None,
    };
    finish_fixed_outputs(
        entry,
        request,
        vec![
            values,
            index_fact(&input),
            super::super::facts::index(column(mapping_length), input.residency.clone()),
        ],
        diagnostics,
    )
}
fn index_fact(input: &ValueFact) -> ValueFact {
    index(column(None), input.residency.clone())
}
