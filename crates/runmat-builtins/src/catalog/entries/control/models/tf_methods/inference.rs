use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    standard, CallInference, CallRequest, DynamicReason, InferenceDiagnostic, LiteralValue,
    NumericClass, NumericDomain, NumericFact, ObjectFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
enum Form {
    Binary,
    Unary,
    Power,
}

pub(super) fn binary(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, Form::Binary)
}

pub(super) fn unary(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, Form::Unary)
}

pub(super) fn power(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, Form::Power)
}

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry, form: Form) -> CallInference {
    let mut diagnostics = Vec::new();
    let expected = if matches!(form, Form::Unary) { 1 } else { 2 };
    if request.arguments.len() != expected {
        diagnostics.push(argument_error(
            "RM-CATALOG-TF-METHOD-ARITY",
            "incorrect number of tf operator operands",
            request.arguments.len(),
        ));
    }
    for (index, operand) in request.arguments.iter().take(expected).enumerate() {
        if index == 1 && matches!(form, Form::Power) {
            validate_exponent(
                operand,
                request.literals.literal_args.get(index),
                &mut diagnostics,
            );
        } else {
            validate_model_or_gain(
                operand,
                index,
                matches!(form, Form::Unary | Form::Power),
                &mut diagnostics,
            );
        }
    }
    if matches!(form, Form::Binary)
        && request.arguments.len() == 2
        && request.arguments.iter().all(|operand| {
            matches!(
                &operand.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical
            )
        })
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-TF-METHOD-MODEL",
            "at least one operand must be a tf model",
            0,
        ));
    }
    let known_real = request
        .arguments
        .iter()
        .enumerate()
        .take(expected)
        .all(|(index, operand)| {
            index == 1 && matches!(form, Form::Power) || is_known_real(operand)
        });
    finish_fixed(entry, request, result_fact(known_real), diagnostics)
}

fn validate_model_or_gain(
    value: &ValueFact,
    index: usize,
    require_model: bool,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    match &value.kind {
        ValueKindFact::Unknown => {}
        ValueKindFact::Object(object) => {
            if object
                .runtime_class
                .as_ref()
                .is_some_and(|class| class != &standard::TRANSFER_FUNCTION.owned())
                || object.runtime_class.is_none() && object.class.is_some()
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-TF-METHOD-MODEL",
                    "operator requires a SISO tf model",
                    index,
                ));
            }
        }
        ValueKindFact::Numeric(_) | ValueKindFact::Logical if !require_model => {
            if value.storage == StorageFact::Sparse
                || value.shape.element_count().is_some_and(|count| count != 1)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-TF-METHOD-SCALAR",
                    "gain operand must be a dense numeric scalar",
                    index,
                ));
            }
        }
        _ => diagnostics.push(argument_error(
            "RM-CATALOG-TF-METHOD-MODEL",
            "operator requires a SISO tf model or scalar gain",
            index,
        )),
    }
}

fn validate_exponent(
    value: &ValueFact,
    literal: Option<&LiteralValue>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if !matches!(value.kind, ValueKindFact::Unknown)
        && (!matches!(value.kind, ValueKindFact::Logical)
            && !value
                .numeric()
                .is_some_and(|numeric| numeric.domain == NumericDomain::Real)
            || value.storage == StorageFact::Sparse
            || value.shape.element_count().is_some_and(|count| count != 1))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-TF-METHOD-EXPONENT",
            "exponent must be a real integer-valued scalar",
            1,
        ));
    }
    let invalid = match literal {
        Some(LiteralValue::Integer { text, .. }) => text
            .parse::<i128>()
            .is_ok_and(|value| value < i64::MIN as i128 || value > i64::MAX as i128),
        Some(LiteralValue::Number(value)) => invalid_float_exponent(*value),
        Some(LiteralValue::Real { text, .. }) => {
            text.parse::<f64>().is_ok_and(invalid_float_exponent)
        }
        _ => false,
    };
    if invalid {
        diagnostics.push(argument_error(
            "RM-CATALOG-TF-METHOD-EXPONENT",
            "exponent must be an integer within the signed 64-bit range",
            1,
        ));
    }
}

fn invalid_float_exponent(value: f64) -> bool {
    !value.is_finite()
        || value.fract() != 0.0
        || value < i64::MIN as f64
        || value >= i64::MAX as f64
}

fn is_known_real(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Numeric(numeric) => numeric.domain == NumericDomain::Real,
        ValueKindFact::Logical => true,
        ValueKindFact::Object(object)
            if object.runtime_class.as_ref() == Some(&standard::TRANSFER_FUNCTION.owned()) =>
        {
            ["Numerator", "Denominator"].iter().all(|name| {
                object.properties.get(*name).is_some_and(|property| {
                    property
                        .numeric()
                        .is_some_and(|numeric| numeric.domain == NumericDomain::Real)
                })
            })
        }
        _ => false,
    }
}

fn result_fact(known_real: bool) -> ValueFact {
    let mut coefficient = ValueFact::unknown(DynamicReason::RuntimeValue);
    coefficient.shape = ShapeFact::from(vec![Some(1), None]);
    coefficient.storage = StorageFact::Dense;
    coefficient.residency = runmat_types::ResidencyFact::Host;
    if known_real {
        coefficient.kind = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
    }
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut properties = BTreeMap::new();
    properties.insert("Numerator".into(), coefficient.clone());
    properties.insert("Denominator".into(), coefficient);
    properties.insert("Ts".into(), scalar.clone());
    properties.insert("InputDelay".into(), scalar.clone());
    properties.insert("OutputDelay".into(), scalar);
    properties.insert(
        "Variable".into(),
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), None]),
            StorageFact::Dense,
        ),
    );
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::TRANSFER_FUNCTION.owned()),
            properties,
            properties_complete: true,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}
