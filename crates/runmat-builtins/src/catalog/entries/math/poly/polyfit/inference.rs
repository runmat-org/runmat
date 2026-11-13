use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    LiteralValue, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, StructFact,
    ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

use super::super::PolyfitStatField;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(3..=4).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLYFIT-ARITY",
            "polyfit requires X, Y, degree, and optionally weights",
            request.arguments.len().saturating_sub(1),
        ));
    }
    validate_vector(request.arguments.first(), 0, false, &mut diagnostics);
    validate_vector(request.arguments.get(1), 1, true, &mut diagnostics);
    validate_degree(request, &mut diagnostics);
    validate_vector(request.arguments.get(3), 3, false, &mut diagnostics);
    let x_len = request
        .arguments
        .first()
        .and_then(|value| value.shape.element_count());
    let y_len = request
        .arguments
        .get(1)
        .and_then(|value| value.shape.element_count());
    if x_len == Some(0) || y_len == Some(0) {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLYFIT-EMPTY",
            "X and Y must not be empty",
            0,
        ));
    }
    if x_len.zip(y_len).is_some_and(|(x, y)| x != y) {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLYFIT-LENGTH",
            "X and Y must contain the same number of samples",
            1,
        ));
    }
    if request
        .arguments
        .get(3)
        .and_then(|value| value.shape.element_count())
        .zip(x_len)
        .is_some_and(|(weights, x)| weights != x)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLYFIT-WEIGHTS-LENGTH",
            "weights must contain one value per sample",
            3,
        ));
    }
    let width = degree_width(request);
    let coefficients = coefficient_fact(request.arguments.get(1), width);
    let outputs = match request.outputs.requested.known_count() {
        Some(0) => Vec::new(),
        Some(count) => (0..count)
            .map(|index| match index {
                0 => coefficients.clone(),
                1 => statistics_fact(width),
                2 => double_array(ShapeFact::from(vec![Some(1), Some(2)])),
                _ => double_scalar(),
            })
            .collect(),
        None => vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
    };
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn validate_vector(
    value: Option<&ValueFact>,
    index: usize,
    allow_complex: bool,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let Some(value) = value else { return };
    let valid = match value.kind {
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        })
        | ValueKindFact::Logical
        | ValueKindFact::Unknown => true,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Complex,
            ..
        }) => allow_complex,
        _ => false,
    };
    if !valid {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLYFIT-CLASS",
            "sample and weight values must be numeric; X and weights must be real",
            index,
        ));
    }
    if let Some(dims) = value.shape.known_dims() {
        if dims.iter().filter(|dim| dim.is_some_and(|n| n > 1)).count() > 1 {
            diagnostics.push(argument_error(
                "RM-CATALOG-POLYFIT-VECTOR",
                "samples and weights must form vectors",
                index,
            ));
        }
    }
}

fn validate_degree(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let Some(value) = request.arguments.get(2) else {
        return;
    };
    if !matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    ) || value.shape.element_count().is_some_and(|count| count != 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLYFIT-DEGREE",
            "degree must be a nonnegative integer scalar",
            2,
        ));
    }
    if let Some(LiteralValue::Integer { text, .. }) = request.literals.literal_args.get(2) {
        if text.parse::<usize>().is_err() {
            diagnostics.push(argument_error(
                "RM-CATALOG-POLYFIT-DEGREE",
                "degree must be a nonnegative finite integer",
                2,
            ));
        }
    } else if let Some(degree) = request.literals.numeric_at(2) {
        if !degree.is_finite() || degree < 0.0 || degree.fract() != 0.0 {
            diagnostics.push(argument_error(
                "RM-CATALOG-POLYFIT-DEGREE",
                "degree must be a nonnegative finite integer",
                2,
            ));
        }
    }
}

fn degree_width(request: &CallRequest) -> Option<usize> {
    if let Some(LiteralValue::Integer { text, .. }) = request.literals.literal_args.get(2) {
        return text.parse::<usize>().ok()?.checked_add(1);
    }
    let value = request.literals.numeric_at(2)?;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value >= usize::MAX as f64 {
        return None;
    }
    (value as usize).checked_add(1)
}

fn coefficient_fact(y: Option<&ValueFact>, width: Option<usize>) -> ValueFact {
    let mut result = ValueFact::unknown(DynamicReason::RuntimeValue);
    result.shape = ShapeFact::from(vec![Some(1), width]);
    result.storage = StorageFact::Dense;
    result.residency = runmat_types::ResidencyFact::Host;
    if y.is_some_and(|value| {
        matches!(
            value.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Logical
        )
    }) {
        result.kind = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
    }
    result
}

fn statistics_fact(width: Option<usize>) -> ValueFact {
    let mut fields = BTreeMap::new();
    fields.insert(
        PolyfitStatField::R.public_name().into(),
        double_array(ShapeFact::from(vec![width, width])),
    );
    fields.insert(
        PolyfitStatField::DegreesOfFreedom.public_name().into(),
        double_scalar(),
    );
    fields.insert(
        PolyfitStatField::ResidualNorm.public_name().into(),
        double_scalar(),
    );
    ValueFact::proven(
        ValueKindFact::Struct(StructFact::scalar(fields, true)),
        ShapeFact::Scalar,
        StorageFact::Scalar,
    )
}

fn double_array(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn double_scalar() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}
