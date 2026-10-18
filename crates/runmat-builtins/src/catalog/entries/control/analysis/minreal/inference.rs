use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    standard, CallInference, CallRequest, LiteralValue, NumericClass, NumericDomain, ObjectFact,
    ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-MINREAL-ARITY",
            "minreal expects a tf model and optional tolerance",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if let Some(model) = request.arguments.first() {
        let accepted = match &model.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => match object.runtime_class.as_ref() {
                Some(class) => class == &standard::TRANSFER_FUNCTION.owned(),
                None => object.class.is_none(),
            },
            _ => false,
        };
        if !accepted
            || model.shape.element_count().is_some_and(|count| count != 1)
            || matches!(model.residency, ResidencyFact::Remote { .. })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-MINREAL-MODEL",
                "sys must be a scalar SISO tf object",
                0,
            ));
        }
    }
    if let Some(tolerance) = request.arguments.get(1) {
        let numeric = match &tolerance.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Numeric(value) => {
                value.domain == NumericDomain::Real
                    && matches!(value.class, NumericClass::Single | NumericClass::Double)
            }
            _ => false,
        };
        if !numeric
            || tolerance
                .shape
                .element_count()
                .is_some_and(|count| count != 1)
            || !matches!(
                tolerance.residency,
                ResidencyFact::Host | ResidencyFact::Unknown
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-MINREAL-TOLERANCE",
                "tol must be a host real single/double scalar",
                1,
            ));
        }
        if let Some(literal) = request.literals.literal_args.get(1) {
            let valid = match literal {
                LiteralValue::Number(value) => Some(value.is_finite() && *value > 0.0),
                LiteralValue::Real {
                    text,
                    class: NumericClass::Single | NumericClass::Double,
                } => text
                    .parse::<f64>()
                    .ok()
                    .map(|value| value.is_finite() && value > 0.0),
                LiteralValue::Integer { .. }
                | LiteralValue::Bool(_)
                | LiteralValue::Complex { .. } => Some(false),
                _ => None,
            };
            if valid == Some(false) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-MINREAL-TOLERANCE",
                    "tol must be finite and positive",
                    1,
                ));
            }
        }
    }
    // A successful reduction returns a scalar host tf, but its coefficient
    // properties need not have the input model's shapes or values.
    let output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::TRANSFER_FUNCTION.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    finish_fixed(entry, request, output, diagnostics)
}
