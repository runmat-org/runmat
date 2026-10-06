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
                let target = args.len() >= 2
                    && real_numeric_candidate(value)
                    && value.shape.element_count().is_none_or(|n| n == 1);
                let text_index = usize::from(target);
                if let Some(text) = args.get(text_index) {
                    if !text_candidate(text, !matches!(form, AnnotationForm::AxesText)) {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-ANNOTATION-TEXT",
                            "annotation text must be a supported text or scalar numeric value",
                            text_index,
                        ));
                    }
                }
            }
        }
        AnnotationForm::PositionedText => {
            for (index, value) in args.iter().take(2).enumerate() {
                if !real_numeric_candidate(value)
                    || value.shape.element_count().is_some_and(|n| n != 1)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-TEXT-POSITION",
                        "text coordinates must be real numeric scalars",
                        index,
                    ));
                }
            }
            if args.len() == 3 && !text_candidate(&args[2], false) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-TEXT-LABEL",
                    "text requires a label after two coordinates",
                    2,
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
