use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = vec![];
    if request.arguments.len() > 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-DASPECT-ARITY",
            "daspect accepts a ratio or mode and an optional leading axes handle",
            2,
        ));
    }
    let ratio = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let mut output = ratio;
    let mut setter = false;
    if request.arguments.len() == 2 {
        let target = &request.arguments[0];
        if !matches!(
            target.kind,
            ValueKindFact::Unknown | ValueKindFact::Numeric(_)
        ) || target.shape.element_count().is_some_and(|n| n != 1)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-DASPECT-TARGET",
                "explicit target must be a scalar numeric axes alias",
                0,
            ));
        }
    }
    if let Some((index, control)) = request.arguments.iter().enumerate().next_back() {
        match &control.kind {
            ValueKindFact::String | ValueKindFact::Character => {
                output = ValueFact::scalar(ValueKindFact::String);
                if let Some(text) = request.literals.literal_string_at(index) {
                    match text.trim() {
                        "mode" => {}
                        "auto" | "manual" => setter = true,
                        _ => diagnostics.push(argument_error(
                            "RM-CATALOG-DASPECT-MODE",
                            "mode must be auto, manual, or mode",
                            index,
                        )),
                    }
                }
            }
            ValueKindFact::Numeric(numeric) => {
                if numeric.domain == NumericDomain::Complex
                    || control.storage == StorageFact::Sparse
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-DASPECT-NUMERIC",
                        "ratio must be real dense host numeric data",
                        index,
                    ));
                }
                // A lone scalar might name existing axes; no content is inferred
                // from its shape. With a leading target the last value is a ratio.
                let may_be_target =
                    request.arguments.len() == 1 && control.shape.element_count() != Some(3);
                if control.shape.element_count().is_some_and(|n| n != 1) || !may_be_target {
                    setter = true;
                    if let Some(dims) = control.shape.known_dims() {
                        let compatible = match dims.as_slice() {
                            [n] => n.is_none_or(|n| n == 3),
                            [rows, cols] => {
                                (rows.is_none_or(|n| n == 1) && cols.is_none_or(|n| n == 3))
                                    || (rows.is_none_or(|n| n == 3) && cols.is_none_or(|n| n == 1))
                            }
                            _ => false,
                        };
                        if !compatible {
                            diagnostics.push(argument_error(
                                "RM-CATALOG-DASPECT-SHAPE",
                                "ratio must be a three-element row or column vector",
                                index,
                            ));
                        }
                    }
                }
                if let Some(values) = request.literals.literal_vector_at(index) {
                    if values
                        .iter()
                        .filter_map(LiteralContext::numeric_from_literal)
                        .any(|n| !n.is_finite() || n <= 0.)
                    {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-DASPECT-RATIO",
                            "ratio entries must be positive and finite",
                            index,
                        ));
                    }
                }
            }
            ValueKindFact::Unknown => output = ValueFact::unknown(DynamicReason::RuntimeValue),
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-DASPECT-INPUT",
                "expected a host ratio, scalar axes alias or mode text",
                index,
            )),
        }
    }
    if setter
        && request
            .outputs
            .requested
            .known_count()
            .is_some_and(|n| n > 0)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-DASPECT-SETTER-OUTPUT",
            "daspect setters return no value",
            0,
        ));
    }
    finish_fixed(entry, request, output, diagnostics)
}
