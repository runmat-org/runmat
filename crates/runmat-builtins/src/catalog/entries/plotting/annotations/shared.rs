use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;

#[derive(Clone, Copy)]
pub(super) enum AnnotationForm {
    AxesText,
    AxesNumericText,
    FigureLabel,
    PositionedText,
    ReferenceLine,
}

fn text_candidate(value: &ValueFact, numeric_allowed: bool) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Unknown
            | ValueKindFact::String
            | ValueKindFact::Character
            | ValueKindFact::Cell(_)
    ) || (numeric_allowed
        && matches!(
            value.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
        )
        && value.shape.element_count().is_none_or(|n| n == 1))
}

fn real_numeric_candidate(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
    ) && value.storage != StorageFact::Sparse
}

fn scalar_numeric_candidate(value: &ValueFact) -> bool {
    real_numeric_candidate(value) && value.shape.element_count().is_none_or(|count| count == 1)
}

fn positioned_form_possible(args: &[ValueFact], target: bool, three_dimensional: bool) -> bool {
    let offset = usize::from(target);
    let coordinate_count = if three_dimensional { 3 } else { 2 };
    let label_index = offset + coordinate_count;
    args.len() > label_index
        && (!target || scalar_numeric_candidate(&args[0]))
        && args[offset..label_index]
            .iter()
            .all(scalar_numeric_candidate)
        && text_candidate(&args[label_index], false)
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    form: AnnotationForm,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let args = &request.arguments;
    let first = args.first();
    let minimum = match form {
        AnnotationForm::AxesText
        | AnnotationForm::AxesNumericText
        | AnnotationForm::FigureLabel
        | AnnotationForm::ReferenceLine => 1,
        AnnotationForm::PositionedText => 3,
    };
    if args.len() < minimum {
        diagnostics.push(argument_error(
            "RM-CATALOG-ANNOTATION-ARITY",
            "annotation arguments are incomplete",
            args.len(),
        ));
    }
    if !matches!(form, AnnotationForm::PositionedText) {
        for (index, value) in args.iter().enumerate() {
            if matches!(
                value.residency,
                ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
            ) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ANNOTATION-RESIDENCY",
                    "graphics annotation controls must be host values",
                    index,
                ));
            }
        }
    }
    match form {
        AnnotationForm::AxesText
        | AnnotationForm::AxesNumericText
        | AnnotationForm::FigureLabel => {
            if let Some(value) = first {
                let numeric_allowed = !matches!(form, AnnotationForm::AxesText);
                let ordinary = text_candidate(value, numeric_allowed);
                let targeted = args.len() >= 2
                    && scalar_numeric_candidate(value)
                    && text_candidate(&args[1], numeric_allowed);
                if !ordinary && !targeted {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-ANNOTATION-TEXT",
                        "annotation text must be a supported text or scalar numeric value",
                        0,
                    ));
                }
            }
        }
        AnnotationForm::PositionedText => {
            let possible = [false, true].into_iter().any(|target| {
                [false, true].into_iter().any(|three_dimensional| {
                    positioned_form_possible(args, target, three_dimensional)
                })
            });
            if args.len() >= 3 && !possible {
                diagnostics.push(argument_error(
                    "RM-CATALOG-TEXT-FORM",
                    "text requires real scalar coordinates and a text label",
                    0,
                ));
            }
        }
        AnnotationForm::ReferenceLine => {
            if let Some(value) = first {
                if !real_numeric_candidate(value) && !matches!(value.kind, ValueKindFact::Logical) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-REFERENCE-LINE-COORDINATE",
                        "reference line coordinates must be real numeric values",
                        0,
                    ));
                }
            }
        }
    }
    let mut output = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    if matches!(form, AnnotationForm::ReferenceLine) {
        if let Some(count) = first.and_then(|value| value.shape.element_count()) {
            if count != 1 {
                output.shape = ShapeFact::from(vec![Some(1), Some(count)]);
                output.storage = StorageFact::Dense;
            }
        } else {
            output.shape = ShapeFact::from(vec![Some(1), None]);
            output.storage = StorageFact::Dense;
        }
    }
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}
