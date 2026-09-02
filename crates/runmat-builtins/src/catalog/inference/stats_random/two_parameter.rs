use super::super::{argument_error, finish_fixed, numeric_kind, preserved_binary_residency};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    AliasFact, CallInference, CallRequest, ContiguityFact, DynamicReason, InferenceDiagnostic,
    LayoutFact, LiteralValue, MutationFact, NumericClass, NumericDomain, ShapeFact, StorageFact,
    ValueFact, ValueKindFact, ViewFact,
};

#[derive(Clone, Copy)]
enum ParameterPolicy {
    Numeric,
    NumericOrLogical,
}

pub(in crate::catalog::inference) fn infer_gamrnd(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_two_parameter_random(
        request,
        entry,
        "gamrnd",
        "shape and scale",
        ParameterPolicy::Numeric,
        validate_gamma_literals,
    )
}

pub(in crate::catalog::inference) fn infer_binornd(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_two_parameter_random(
        request,
        entry,
        "binornd",
        "trial-count and probability",
        ParameterPolicy::NumericOrLogical,
        validate_binomial_literals,
    )
}

fn infer_two_parameter_random(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    name: &'static str,
    parameter_names: &'static str,
    parameter_policy: ParameterPolicy,
    validate_literals: fn(&CallRequest, &mut Vec<InferenceDiagnostic>),
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-RANDOM-ARITY",
            format!("{name} requires {parameter_names} parameters"),
            request.arguments.len(),
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    }

    validate_parameter_facts(request, name, parameter_policy, &mut diagnostics);
    validate_size_facts(request, name, &mut diagnostics);
    validate_literals(request, &mut diagnostics);

    let parameters = &request.arguments[..2];
    let output_class = if parameters.iter().any(|argument| {
        matches!(
            argument.kind,
            ValueKindFact::Numeric(runmat_types::NumericFact {
                class: NumericClass::Single,
                ..
            })
        )
    }) {
        Some(NumericClass::Single)
    } else if parameters.iter().all(|argument| {
        matches!(
            argument.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        )
    }) {
        Some(NumericClass::Double)
    } else {
        None
    };

    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    if let Some(output_class) = output_class {
        output.kind = numeric_kind(output_class, NumericDomain::Real);
    }
    output.shape = random_output_shape(request, name, &mut diagnostics);
    output.storage = if output.is_scalar() {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    output.layout = LayoutFact::ColumnMajor;
    output.contiguity = ContiguityFact::Contiguous;
    output.view = ViewFact::Materialized;
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
    output.residency = preserved_binary_residency(
        &request.arguments[0].residency,
        &request.arguments[1].residency,
    );
    finish_fixed(entry, request, output, diagnostics)
}

fn validate_parameter_facts(
    request: &CallRequest,
    name: &str,
    policy: ParameterPolicy,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    for (index, argument) in request.arguments.iter().take(2).enumerate() {
        let kind_supported = matches!(
            argument.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Unknown
        ) || matches!(policy, ParameterPolicy::NumericOrLogical)
            && matches!(argument.kind, ValueKindFact::Logical);
        if matches!(argument.storage, StorageFact::Sparse) || !kind_supported {
            diagnostics.push(argument_error(
                "RM-CATALOG-RANDOM-PARAMETER",
                format!("{name} parameters must be dense real numeric values"),
                index,
            ));
        } else if matches!(
            argument.kind,
            ValueKindFact::Numeric(runmat_types::NumericFact {
                domain: NumericDomain::Complex,
                ..
            })
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-RANDOM-PARAMETER",
                format!("{name} parameters must be real"),
                index,
            ));
        }
    }
}

fn validate_size_facts(
    request: &CallRequest,
    name: &str,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    for (index, argument) in request.arguments.iter().enumerate().skip(2) {
        if matches!(argument.storage, StorageFact::Sparse)
            || !matches!(
                argument.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Unknown
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-RANDOM-SIZE",
                format!("{name} size controls must be dense real numeric values"),
                index,
            ));
        }
    }
}

fn random_output_shape(
    request: &CallRequest,
    name: &str,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ShapeFact {
    if request.arguments.len() == 2 {
        return scalar_expansion_shape(
            &request.arguments[0].shape,
            &request.arguments[1].shape,
            name,
            diagnostics,
        );
    }

    let explicit = explicit_size_shape(request);
    if !matches!(explicit, ShapeFact::Unknown | ShapeFact::Ranked { .. }) {
        for (index, parameter) in request.arguments.iter().take(2).enumerate() {
            if parameter
                .shape
                .element_count()
                .is_some_and(|count| count != 1)
                && !parameter.shape.is_proven_equivalent(&explicit)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-RANDOM-SIZE-MISMATCH",
                    format!("{name} explicit size must match each nonscalar parameter"),
                    index,
                ));
            }
        }
    }
    explicit
}

fn scalar_expansion_shape(
    left: &ShapeFact,
    right: &ShapeFact,
    name: &str,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ShapeFact {
    if left.element_count() == Some(1) {
        return right.clone();
    }
    if right.element_count() == Some(1) {
        return left.clone();
    }
    if left == right || left.is_proven_equivalent(right) {
        return left.clone();
    }
    if left.known_dims().is_some() && right.known_dims().is_some() {
        diagnostics.push(InferenceDiagnostic::error(
            "RM-CATALOG-RANDOM-SHAPE",
            format!("{name} parameters must be scalar or have matching shapes"),
        ));
    }
    ShapeFact::Unknown
}

fn explicit_size_shape(request: &CallRequest) -> ShapeFact {
    let mut dimensions = request.literals.numeric_dims_from(2);
    if request.arguments.len() == 3 {
        if let Some(vector) = request.literals.numeric_vector_at(2) {
            dimensions = vector;
        } else if dimensions.first().is_some_and(Option::is_some) {
            dimensions.push(dimensions[0]);
        } else {
            return ShapeFact::Ranked { rank: 2 };
        }
    }
    if dimensions.is_empty() {
        return ShapeFact::Unknown;
    }
    while dimensions.len() > 2 && dimensions.last() == Some(&Some(1)) {
        dimensions.pop();
    }
    ShapeFact::from(dimensions)
}

fn validate_gamma_literals(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    validate_numeric_literal(
        request,
        0,
        diagnostics,
        "RM-CATALOG-GAMRND-SHAPE",
        |value| value >= 0.0,
    );
    validate_numeric_literal(
        request,
        1,
        diagnostics,
        "RM-CATALOG-GAMRND-SCALE",
        |value| value > 0.0,
    );
}

fn validate_binomial_literals(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    validate_numeric_literal(
        request,
        0,
        diagnostics,
        "RM-CATALOG-BINORND-TRIALS",
        |value| value > 0.0 && value.fract() == 0.0,
    );
    validate_numeric_literal(
        request,
        1,
        diagnostics,
        "RM-CATALOG-BINORND-PROBABILITY",
        |value| (0.0..=1.0).contains(&value),
    );
}

fn validate_numeric_literal(
    request: &CallRequest,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
    code: &'static str,
    valid: impl Fn(f64) -> bool + Copy,
) {
    let Some(literal) = request.literals.literal_args.get(index) else {
        return;
    };
    if numeric_literal_values(literal).is_some_and(|values| {
        values
            .into_iter()
            .any(|value| !value.is_finite() || !valid(value))
    }) {
        diagnostics.push(argument_error(
            code,
            "random-distribution parameter is outside its supported domain",
            index,
        ));
    }
}

fn numeric_literal_values(literal: &LiteralValue) -> Option<Vec<f64>> {
    match literal {
        LiteralValue::Vector(values) => values.iter().try_fold(Vec::new(), |mut output, value| {
            output.extend(numeric_literal_values(value)?);
            Some(output)
        }),
        LiteralValue::Matrix(rows) => {
            rows.iter()
                .flatten()
                .try_fold(Vec::new(), |mut output, value| {
                    output.extend(numeric_literal_values(value)?);
                    Some(output)
                })
        }
        LiteralValue::Bool(value) => Some(vec![if *value { 1.0 } else { 0.0 }]),
        value => runmat_types::LiteralContext::numeric_from_literal(value).map(|value| vec![value]),
    }
}
