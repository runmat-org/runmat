use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    standard, CallInference, CallRequest, DynamicReason, LiteralValue, NumericClass, NumericDomain,
    NumericFact, ObjectFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let args = &request.arguments;
    if !(1..=3).contains(&args.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FEEDBACK-ARITY",
            "feedback expects one to three arguments",
            args.len().saturating_sub(1),
        ));
    }
    for (index, system) in args.iter().take(2).enumerate() {
        let valid = match &system.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => match object.runtime_class.as_ref() {
                Some(class) => class == &standard::TRANSFER_FUNCTION.owned(),
                None => object.class.is_none(),
            },
            ValueKindFact::Numeric(_) | ValueKindFact::Logical => true,
            _ => false,
        };
        if !valid
            || system.shape.element_count().is_some_and(|count| count != 1)
            || matches!(system.residency, ResidencyFact::Remote { .. })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEEDBACK-SYSTEM",
                "system must be a scalar SISO tf object or scalar gain",
                index,
            ));
        }
    }
    if let Some(sign) = args.get(2) {
        let real_scalar = matches!(sign.kind, ValueKindFact::Unknown | ValueKindFact::Logical)
            || sign
                .numeric()
                .is_some_and(|n| n.domain == NumericDomain::Real);
        if !real_scalar || sign.shape.element_count().is_some_and(|count| count != 1) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEEDBACK-SIGN",
                "sign must be the real scalar -1 or +1",
                2,
            ));
        }
        if let Some(literal) = request.literals.literal_args.get(2) {
            let known_value = match literal {
                LiteralValue::Number(value) => Some(*value),
                LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => {
                    text.parse::<f64>().ok()
                }
                LiteralValue::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
                _ => None,
            };
            if known_value.is_some_and(|value| value != -1.0 && value != 1.0) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEEDBACK-SIGN",
                    "sign must be -1 or +1",
                    2,
                ));
            }
        }
    }
    let mut properties = BTreeMap::new();
    for name in ["Numerator", "Denominator"] {
        let mut coefficients = ValueFact::unknown(DynamicReason::RuntimeValue);
        coefficients.shape = ShapeFact::from(vec![Some(1), None]);
        coefficients.storage = StorageFact::Dense;
        coefficients.residency = ResidencyFact::Host;
        properties.insert(name.to_string(), coefficients);
    }
    properties.insert(
        "Variable".into(),
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), None]),
            StorageFact::Dense,
        ),
    );
    let scalar_double = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    for name in ["Ts", "InputDelay", "OutputDelay"] {
        properties.insert(name.to_string(), scalar_double.clone());
    }
    let output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::TRANSFER_FUNCTION.owned()),
            properties,
            properties_complete: true,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    finish_fixed(entry, request, output, diagnostics)
}
