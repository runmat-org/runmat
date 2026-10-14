use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-UNWRAP-ARITY",
            "unwrap expects P and optional tolerance and dimension",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if let Some(phase) = request.arguments.first() {
        if matches!(phase.storage, StorageFact::Sparse | StorageFact::Opaque)
            || !matches!(
                phase.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Logical
                    | ValueKindFact::Unknown
            )
            || request
                .literals
                .literal_args
                .first()
                .is_some_and(inexact_integer_literal)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-UNWRAP-PHASE",
                "unwrap requires dense real numeric or logical phase data",
                0,
            ));
        }
    }
    if let Some(tolerance) = request.arguments.get(1) {
        if !is_empty(request, 1)
            && (tolerance.shape.is_definitely_nonscalar()
                || !real_scalar_kind(&tolerance.kind)
                || matches!(tolerance.storage, StorageFact::Sparse | StorageFact::Opaque)
                || request
                    .literals
                    .literal_args
                    .get(1)
                    .is_some_and(invalid_tolerance))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-UNWRAP-TOL",
                "tolerance must be a finite nonnegative scalar or []",
                1,
            ));
        }
    }
    if let Some(dimension) = request.arguments.get(2) {
        if !is_empty(request, 2)
            && (dimension.shape.is_definitely_nonscalar()
                || !real_scalar_kind(&dimension.kind)
                || matches!(dimension.storage, StorageFact::Sparse | StorageFact::Opaque)
                || request
                    .literals
                    .literal_args
                    .get(2)
                    .is_some_and(invalid_dimension))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-UNWRAP-DIM",
                "dimension must be a positive integer scalar or []",
                2,
            ));
        }
    }
    let output = match request.arguments.first() {
        Some(phase) if diagnostics.is_empty() => output_fact(phase),
        _ => ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn inexact_integer_literal(literal: &LiteralValue) -> bool {
    if literal.integer_is_exact_binary64() == Some(false) {
        return true;
    }
    match literal {
        LiteralValue::Vector(values) => values.iter().any(inexact_integer_literal),
        LiteralValue::Matrix(rows) => rows.iter().flatten().any(inexact_integer_literal),
        _ => false,
    }
}

fn output_fact(phase: &ValueFact) -> ValueFact {
    let class = match phase.kind {
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            ..
        }) => NumericClass::Single,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical => NumericClass::Double,
        _ => return ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        phase.shape.clone(),
        StorageFact::Unknown,
    );
    output.residency = ResidencyFact::Host;
    output.layout = LayoutFact::Unknown;
    output.contiguity = ContiguityFact::Unknown;
    output.alias = AliasFact::Unknown;
    output.view = ViewFact::Unknown;
    output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    output
}

fn real_scalar_kind(kind: &ValueKindFact) -> bool {
    matches!(
        kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    )
}

fn is_empty(request: &CallRequest, index: usize) -> bool {
    matches!(
        request.literals.literal_args.get(index),
        Some(LiteralValue::Empty)
    ) || request
        .arguments
        .get(index)
        .is_some_and(|arg| arg.shape.element_count() == Some(0))
}

fn numeric_literal(literal: &LiteralValue) -> Option<f64> {
    match literal {
        LiteralValue::Number(value) => Some(*value),
        LiteralValue::Integer { .. } => literal.integer_value_i128().map(|v| v as f64),
        LiteralValue::Bool(value) => Some(f64::from(*value as u8)),
        LiteralValue::Real { .. } => LiteralContext::numeric_from_literal(literal),
        _ => None,
    }
}

fn invalid_tolerance(literal: &LiteralValue) -> bool {
    if let LiteralValue::Integer { .. } = literal {
        return literal.integer_value_i128().is_some_and(|value| value < 0)
            || literal.integer_is_exact_binary64() == Some(false);
    }
    match numeric_literal(literal) {
        Some(value) => !value.is_finite() || value < 0.0,
        None => !matches!(literal, LiteralValue::Empty | LiteralValue::Unknown),
    }
}

fn invalid_dimension(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Integer { .. } => literal
            .integer_value_i128()
            .is_some_and(|v| v <= 0 || usize::try_from(v).is_err()),
        _ => match numeric_literal(literal) {
            Some(value) => {
                let rounded = value.round();
                !value.is_finite()
                    || rounded < 1.0
                    || (rounded - value).abs() > 1e-6
                    || rounded > usize::MAX as f64
                    || (usize::BITS == 64 && rounded == usize::MAX as f64)
            }
            None => !matches!(literal, LiteralValue::Empty | LiteralValue::Unknown),
        },
    }
}
