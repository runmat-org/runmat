use super::{argument_error, finish_fixed, numeric_kind};
use crate::{
    BuiltinCatalogEntry, DegreeTrigonometricFunction, HyperbolicFunction,
    InverseHyperbolicFunction, InverseTrigonometricFunction, LogarithmBase,
    PiScaledTrigonometricFunction, RootKind, RoundingFunction, TrigonometricFunction,
};

pub(super) fn infer_round(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-ROUND-ARITY",
            "round requires one to three inputs",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ROUND-ARITY",
            "round accepts at most three inputs",
            3,
        ));
    }

    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ROUND-SPARSE",
            "round does not currently accept sparse input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    if let Some(digits) = request.arguments.get(1) {
        let scalar_control = digits.is_scalar()
            && matches!(
                digits.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
            );
        if !scalar_control {
            diagnostics.push(argument_error(
                "RM-CATALOG-ROUND-DIGITS",
                "round requires N to be an integer-valued numeric scalar",
                1,
            ));
        }
        if let Some(LiteralValue::Number(value)) = request.literals.literal_args.get(1) {
            if !value.is_finite()
                || value.fract() != 0.0
                || *value < f64::from(i32::MIN)
                || *value > f64::from(i32::MAX)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ROUND-DIGITS",
                    "round requires N to be a finite integer in the supported signed range",
                    1,
                ));
            }
        }
    }

    if let Some(mode) = request.arguments.get(2) {
        if !matches!(
            mode.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-ROUND-MODE",
                "round requires mode to be \"decimals\" or \"significant\"",
                2,
            ));
        }
        let mode_literal = request
            .literals
            .literal_args
            .get(2)
            .and_then(|literal| match literal {
                LiteralValue::String(value)
                | LiteralValue::Character(value)
                | LiteralValue::Keyword(value) => Some(value.as_str()),
                _ => None,
            });
        if let Some(mode) = mode_literal {
            if !matches!(
                mode.to_ascii_lowercase().as_str(),
                "decimal" | "decimals" | "significant"
            ) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ROUND-MODE",
                    "round requires mode to be \"decimals\" or \"significant\"",
                    2,
                ));
            }
            if mode.eq_ignore_ascii_case("significant")
                && matches!(
                    request.literals.literal_args.get(1),
                    Some(LiteralValue::Number(value)) if *value <= 0.0
                )
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ROUND-DIGITS",
                    "round requires positive N for significant-digit rounding",
                    1,
                ));
            }
        }
    }

    let mut output = input.clone();
    match &mut output.kind {
        ValueKindFact::Numeric(numeric)
            if numeric.domain == NumericDomain::Complex
                && !matches!(numeric.class, NumericClass::Double | NumericClass::Single) =>
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ROUND-COMPLEX-INTEGER",
                "round does not accept complex fixed-width integer input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric)
            if !matches!(numeric.class, NumericClass::Double | NumericClass::Single)
                && request.arguments.len() > 1 =>
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ROUND-INTEGER-FORM",
                "typed integer X supports only round(X)",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric)
            if matches!(numeric.class, NumericClass::Double | NumericClass::Single) =>
        {
            materialize_output(&mut output);
            output.residency = input.residency.clone();
        }
        ValueKindFact::Numeric(_) => {}
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize_output(&mut output);
            output.residency = input.residency.clone();
        }
        ValueKindFact::Unknown => preserve_shape_on_dynamic_input(&mut output),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-ROUND-INPUT",
                "round requires numeric, logical, or character input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn infer_rounding(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    function: RoundingFunction,
) -> CallInference {
    let name = match function {
        RoundingFunction::Ceil => "ceil",
        RoundingFunction::Fix => "fix",
        RoundingFunction::Floor => "floor",
    };
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-ROUNDING-ARITY",
            format!("{name} requires exactly one input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ROUNDING-ARITY",
            format!("{name} accepts exactly one input"),
            1,
        ));
    }

    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ROUNDING-SPARSE",
            format!("{name} does not currently accept sparse input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let mut output = input.clone();
    match &mut output.kind {
        ValueKindFact::Numeric(numeric)
            if numeric.domain == NumericDomain::Complex
                && !matches!(numeric.class, NumericClass::Double | NumericClass::Single) =>
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ROUNDING-COMPLEX-INTEGER",
                format!("{name} does not accept complex fixed-width integer input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric)
            if matches!(numeric.class, NumericClass::Double | NumericClass::Single) =>
        {
            materialize_output(&mut output);
            output.residency = input.residency.clone();
        }
        ValueKindFact::Numeric(_) => {
            // Every real fixed-width integer is already integral. The runtime
            // returns the exact storage (and resident handle) unchanged.
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize_output(&mut output);
            output.residency = input.residency.clone();
        }
        ValueKindFact::Object(object)
            if object.runtime_class.as_ref().is_some_and(|class| {
                class.is(runmat_types::standard::TABLE)
                    || class.is(runmat_types::standard::TIMETABLE)
            }) =>
        {
            object.properties.clear();
            object.properties_complete = false;
            output.alias = AliasFact::Unique;
            output.mutation = MutationFact::ValueSemantics;
        }
        ValueKindFact::Object(_) | ValueKindFact::Unknown => {
            preserve_shape_on_dynamic_input(&mut output);
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-ROUNDING-INPUT",
                format!("{name} requires numeric, logical, character, or supported tabular input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn infer_hyperbolic(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    function: HyperbolicFunction,
) -> CallInference {
    let name = match function {
        HyperbolicFunction::Sine => "sinh",
        HyperbolicFunction::Cosine => "cosh",
        HyperbolicFunction::Tangent => "tanh",
    };
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-HYPERBOLIC-ARITY",
            format!("{name} requires exactly one input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-HYPERBOLIC-ARITY",
            format!("{name} accepts exactly one input"),
            1,
        ));
    }

    let mut output = input.clone();
    match &mut output.kind {
        ValueKindFact::Numeric(numeric)
            if numeric.domain == NumericDomain::Complex
                && !matches!(numeric.class, NumericClass::Double | NumericClass::Single) =>
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-HYPERBOLIC-COMPLEX-INTEGER",
                format!("{name} does not accept complex fixed-width integer input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric) => {
            if !matches!(numeric.class, NumericClass::Double | NumericClass::Single) {
                numeric.class = NumericClass::Double;
            }
            if matches!(output.storage, StorageFact::Sparse) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-HYPERBOLIC-SPARSE",
                    format!("{name} does not currently accept sparse input"),
                    0,
                ));
                output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
            } else {
                materialize_output(&mut output);
            }
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize_output(&mut output);
        }
        ValueKindFact::Unknown => preserve_shape_on_dynamic_input(&mut output),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-HYPERBOLIC-INPUT",
                format!("{name} requires numeric, logical, or character input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn infer_degree_trigonometric(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    function: DegreeTrigonometricFunction,
) -> CallInference {
    let (name, accepts_character, preserves_residency) = match function {
        DegreeTrigonometricFunction::Sin => ("sind", false, false),
        DegreeTrigonometricFunction::Cos => ("cosd", true, true),
        DegreeTrigonometricFunction::Tan => ("tand", true, true),
    };
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-DEGREE-TRIGONOMETRIC-ARITY",
            format!("{name} requires exactly one input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-DEGREE-TRIGONOMETRIC-ARITY",
            format!("{name} accepts exactly one input"),
            1,
        ));
    }
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-DEGREE-TRIGONOMETRIC-SPARSE",
            format!("{name} does not currently accept sparse input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let residency = if preserves_residency {
        input.residency.clone()
    } else {
        ResidencyFact::Host
    };
    let mut output = input.clone();
    match &mut output.kind {
        ValueKindFact::Numeric(numeric)
            if numeric.domain == NumericDomain::Complex
                && !matches!(numeric.class, NumericClass::Double | NumericClass::Single) =>
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-DEGREE-TRIGONOMETRIC-COMPLEX-INTEGER",
                format!("{name} does not accept complex fixed-width integer input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric) => {
            if !matches!(numeric.class, NumericClass::Double | NumericClass::Single) {
                numeric.class = NumericClass::Double;
            }
            materialize_output(&mut output);
            output.residency = residency.clone();
        }
        ValueKindFact::Logical => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize_output(&mut output);
            output.residency = residency.clone();
        }
        ValueKindFact::Character if accepts_character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize_output(&mut output);
            output.residency = residency.clone();
        }
        ValueKindFact::Unknown => {
            preserve_shape_on_dynamic_input(&mut output);
            output.residency = residency;
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-DEGREE-TRIGONOMETRIC-INPUT",
                format!("{name} requires a supported numeric input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn infer_inverse_trigonometric(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    function: InverseTrigonometricFunction,
) -> CallInference {
    let name = match function {
        InverseTrigonometricFunction::Sine => "asin",
        InverseTrigonometricFunction::Cosine => "acos",
        InverseTrigonometricFunction::Tangent => "atan",
    };
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-INVERSE-TRIGONOMETRIC-ARITY",
            format!("{name} requires exactly one input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    let accepts_like = function == InverseTrigonometricFunction::Tangent;
    let arity_is_valid = if accepts_like {
        matches!(request.arguments.len(), 1 | 3)
    } else {
        request.arguments.len() == 1
    };
    if !arity_is_valid {
        diagnostics.push(argument_error(
            "RM-CATALOG-INVERSE-TRIGONOMETRIC-ARITY",
            if accepts_like {
                format!("{name} accepts one input or an input followed by \"like\" and a prototype")
            } else {
                format!("{name} accepts exactly one input")
            },
            request.arguments.len().saturating_sub(1),
        ));
    }
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-INVERSE-TRIGONOMETRIC-SPARSE",
            format!("{name} does not currently accept sparse input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let literal_domain = request
        .literals
        .literal_args
        .first()
        .and_then(|literal| inverse_trigonometric_literal_domain(literal, function));
    let mut output = input.clone();
    match &input.kind {
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) if !matches!(class, NumericClass::Double | NumericClass::Single) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-INVERSE-TRIGONOMETRIC-COMPLEX-INTEGER",
                format!("{name} does not accept complex fixed-width integer input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) => {
            output.kind = numeric_kind(*class, NumericDomain::Complex);
            materialize_output(&mut output);
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }) => {
            let output_class = if matches!(class, NumericClass::Double | NumericClass::Single) {
                *class
            } else {
                NumericClass::Double
            };
            if let Some(domain) = literal_domain.or_else(|| {
                (function == InverseTrigonometricFunction::Tangent).then_some(NumericDomain::Real)
            }) {
                output.kind = numeric_kind(output_class, domain);
                materialize_output(&mut output);
            } else {
                preserve_shape_on_dynamic_input(&mut output);
                output.residency = input.residency.clone();
            }
        }
        ValueKindFact::Logical => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize_output(&mut output);
        }
        ValueKindFact::Character => {
            if let Some(domain) = literal_domain.or_else(|| {
                (function == InverseTrigonometricFunction::Tangent).then_some(NumericDomain::Real)
            }) {
                output.kind = numeric_kind(NumericClass::Double, domain);
                materialize_output(&mut output);
            } else {
                preserve_shape_on_dynamic_input(&mut output);
                output.residency = input.residency.clone();
            }
        }
        ValueKindFact::Unknown => {
            preserve_shape_on_dynamic_input(&mut output);
            output.residency = input.residency.clone();
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-INVERSE-TRIGONOMETRIC-INPUT",
                format!("{name} requires numeric, logical, or character input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    if accepts_like {
        apply_inverse_trigonometric_like(request, &mut output, name, &mut diagnostics);
    }
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn infer_inverse_hyperbolic(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    function: InverseHyperbolicFunction,
) -> CallInference {
    let name = match function {
        InverseHyperbolicFunction::Cosine => "acosh",
        InverseHyperbolicFunction::Sine => "asinh",
        InverseHyperbolicFunction::Tangent => "atanh",
    };
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-INVERSE-HYPERBOLIC-ARITY",
            format!("{name} requires exactly one input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-INVERSE-HYPERBOLIC-ARITY",
            format!("{name} accepts exactly one input"),
            request.arguments.len().saturating_sub(1),
        ));
    }
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-INVERSE-HYPERBOLIC-SPARSE",
            format!("{name} does not currently accept sparse input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let literal_domain = request
        .literals
        .literal_args
        .first()
        .and_then(|literal| inverse_hyperbolic_literal_domain(literal, function));
    let always_real = function == InverseHyperbolicFunction::Sine;
    let logical_always_real = matches!(
        function,
        InverseHyperbolicFunction::Sine | InverseHyperbolicFunction::Tangent
    );
    let mut output = input.clone();
    match &input.kind {
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) if !matches!(class, NumericClass::Double | NumericClass::Single) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-INVERSE-HYPERBOLIC-COMPLEX-INTEGER",
                format!("{name} does not accept complex fixed-width integer input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) => {
            output.kind = numeric_kind(*class, NumericDomain::Complex);
            materialize_output(&mut output);
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }) => {
            let output_class = if matches!(class, NumericClass::Double | NumericClass::Single) {
                *class
            } else {
                NumericClass::Double
            };
            if let Some(domain) =
                literal_domain.or_else(|| always_real.then_some(NumericDomain::Real))
            {
                output.kind = numeric_kind(output_class, domain);
                materialize_output(&mut output);
            } else {
                preserve_shape_on_dynamic_input(&mut output);
                output.residency = input.residency.clone();
            }
        }
        ValueKindFact::Logical => {
            if let Some(domain) =
                literal_domain.or_else(|| logical_always_real.then_some(NumericDomain::Real))
            {
                output.kind = numeric_kind(NumericClass::Double, domain);
                materialize_output(&mut output);
            } else {
                preserve_shape_on_dynamic_input(&mut output);
                output.residency = input.residency.clone();
            }
        }
        ValueKindFact::Character => {
            if let Some(domain) =
                literal_domain.or_else(|| always_real.then_some(NumericDomain::Real))
            {
                output.kind = numeric_kind(NumericClass::Double, domain);
                materialize_output(&mut output);
            } else {
                preserve_shape_on_dynamic_input(&mut output);
                output.residency = input.residency.clone();
            }
        }
        ValueKindFact::Unknown => {
            preserve_shape_on_dynamic_input(&mut output);
            output.residency = input.residency.clone();
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-INVERSE-HYPERBOLIC-INPUT",
                format!("{name} requires numeric, logical, or character input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn inverse_hyperbolic_literal_domain(
    literal: &LiteralValue,
    function: InverseHyperbolicFunction,
) -> Option<NumericDomain> {
    match literal {
        LiteralValue::Number(value) => Some(inverse_hyperbolic_real_domain(*value, function)),
        LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => text
            .parse::<f64>()
            .ok()
            .map(|value| inverse_hyperbolic_real_domain(value, function)),
        LiteralValue::Bool(value) => Some(inverse_hyperbolic_real_domain(
            if *value { 1.0 } else { 0.0 },
            function,
        )),
        LiteralValue::Character(value) => combine_numeric_domains(value.chars().map(|character| {
            inverse_hyperbolic_real_domain(f64::from(u32::from(character)), function)
        })),
        LiteralValue::Complex { .. } => Some(NumericDomain::Complex),
        LiteralValue::Vector(values) => combine_optional_numeric_domains(
            values
                .iter()
                .map(|value| inverse_hyperbolic_literal_domain(value, function)),
        ),
        LiteralValue::Matrix(rows) => combine_optional_numeric_domains(
            rows.iter()
                .flatten()
                .map(|value| inverse_hyperbolic_literal_domain(value, function)),
        ),
        LiteralValue::Empty => Some(NumericDomain::Real),
        LiteralValue::String(_)
        | LiteralValue::Keyword(_)
        | LiteralValue::Symbolic(_)
        | LiteralValue::Unknown => None,
    }
}

fn inverse_hyperbolic_real_domain(
    value: f64,
    function: InverseHyperbolicFunction,
) -> NumericDomain {
    let is_real = match function {
        InverseHyperbolicFunction::Cosine => value.is_nan() || value >= 1.0,
        InverseHyperbolicFunction::Sine => true,
        InverseHyperbolicFunction::Tangent => value.is_nan() || (-1.0..=1.0).contains(&value),
    };
    if is_real {
        NumericDomain::Real
    } else {
        NumericDomain::Complex
    }
}

fn combine_optional_numeric_domains(
    domains: impl IntoIterator<Item = Option<NumericDomain>>,
) -> Option<NumericDomain> {
    let mut combined = NumericDomain::Real;
    for domain in domains {
        if domain? == NumericDomain::Complex {
            combined = NumericDomain::Complex;
        }
    }
    Some(combined)
}

fn combine_numeric_domains(
    domains: impl IntoIterator<Item = NumericDomain>,
) -> Option<NumericDomain> {
    combine_optional_numeric_domains(domains.into_iter().map(Some))
}

fn apply_inverse_trigonometric_like(
    request: &CallRequest,
    output: &mut ValueFact,
    name: &str,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    let [_, keyword, prototype] = request.arguments.as_slice() else {
        return;
    };
    let keyword_literal = request
        .literals
        .literal_args
        .get(1)
        .and_then(|literal| match literal {
            LiteralValue::String(value)
            | LiteralValue::Character(value)
            | LiteralValue::Keyword(value) => Some(value.as_str()),
            _ => None,
        });
    match keyword_literal {
        Some(value) if value.eq_ignore_ascii_case("like") => {}
        Some(_) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-INVERSE-TRIGONOMETRIC-LIKE",
                format!("{name} accepts only the \"like\" option"),
                1,
            ));
            output.residency = ResidencyFact::Unknown;
            return;
        }
        None if matches!(
            keyword.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) =>
        {
            output.residency = ResidencyFact::Unknown;
            return;
        }
        None => {
            diagnostics.push(argument_error(
                "RM-CATALOG-INVERSE-TRIGONOMETRIC-LIKE",
                format!("{name} requires \"like\" as its second input"),
                1,
            ));
            output.residency = ResidencyFact::Unknown;
            return;
        }
    }

    let output_is_complex = matches!(
        output.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Complex,
            ..
        })
    );
    match &prototype.kind {
        ValueKindFact::Numeric(prototype_numeric) => {
            if prototype_numeric.domain == NumericDomain::Complex {
                if let ValueKindFact::Numeric(output_numeric) = &mut output.kind {
                    output_numeric.domain = NumericDomain::Complex;
                }
            } else if output_is_complex {
                diagnostics.push(argument_error(
                    "RM-CATALOG-INVERSE-TRIGONOMETRIC-PROTOTYPE",
                    format!("{name} cannot place a complex result like a real prototype"),
                    2,
                ));
            }
            output.residency = prototype.residency.clone();
        }
        ValueKindFact::Logical => {
            if output_is_complex {
                diagnostics.push(argument_error(
                    "RM-CATALOG-INVERSE-TRIGONOMETRIC-PROTOTYPE",
                    format!("{name} cannot place a complex result like a real prototype"),
                    2,
                ));
            }
            output.residency = prototype.residency.clone();
        }
        ValueKindFact::Unknown => {
            output.residency = prototype.residency.clone();
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-INVERSE-TRIGONOMETRIC-PROTOTYPE",
                format!("{name} requires a numeric or logical prototype"),
                2,
            ));
            output.residency = ResidencyFact::Unknown;
        }
    }
}

fn inverse_trigonometric_literal_domain(
    literal: &LiteralValue,
    function: InverseTrigonometricFunction,
) -> Option<NumericDomain> {
    if function == InverseTrigonometricFunction::Tangent {
        return literal_is_real(literal).then_some(NumericDomain::Real);
    }
    match literal {
        LiteralValue::Number(value) => Some(unit_interval_domain(*value)),
        LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => {
            text.parse::<f64>().ok().map(unit_interval_domain)
        }
        LiteralValue::Bool(_) => Some(NumericDomain::Real),
        LiteralValue::Character(value) => Some(if value.chars().all(|ch| u32::from(ch) <= 1) {
            NumericDomain::Real
        } else {
            NumericDomain::Complex
        }),
        LiteralValue::Complex { .. } => Some(NumericDomain::Complex),
        LiteralValue::Vector(values) => combine_inverse_literal_domains(values, function),
        LiteralValue::Matrix(rows) => {
            let values = rows.iter().flatten().cloned().collect::<Vec<_>>();
            combine_inverse_literal_domains(&values, function)
        }
        LiteralValue::Empty => Some(NumericDomain::Real),
        LiteralValue::String(_)
        | LiteralValue::Keyword(_)
        | LiteralValue::Symbolic(_)
        | LiteralValue::Unknown => None,
    }
}

fn literal_is_real(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Number(_)
        | LiteralValue::Real { .. }
        | LiteralValue::Integer { .. }
        | LiteralValue::Bool(_)
        | LiteralValue::Character(_)
        | LiteralValue::Empty => true,
        LiteralValue::Vector(values) => values.iter().all(literal_is_real),
        LiteralValue::Matrix(rows) => rows.iter().flatten().all(literal_is_real),
        LiteralValue::Complex { .. }
        | LiteralValue::String(_)
        | LiteralValue::Keyword(_)
        | LiteralValue::Symbolic(_)
        | LiteralValue::Unknown => false,
    }
}

fn unit_interval_domain(value: f64) -> NumericDomain {
    if value.is_nan() || (-1.0..=1.0).contains(&value) {
        NumericDomain::Real
    } else {
        NumericDomain::Complex
    }
}

fn combine_inverse_literal_domains(
    values: &[LiteralValue],
    function: InverseTrigonometricFunction,
) -> Option<NumericDomain> {
    let mut domain = NumericDomain::Real;
    for value in values {
        let value_domain = inverse_trigonometric_literal_domain(value, function)?;
        if value_domain == NumericDomain::Complex {
            domain = NumericDomain::Complex;
        }
    }
    Some(domain)
}
use runmat_types::{
    infer_call, AliasFact, CallContract, CallInference, CallRequest, ContiguityFact, DynamicReason,
    LayoutFact, LiteralValue, MutationFact, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, StorageFact, ValueFact, ValueKindFact, ViewFact,
};

pub(super) fn infer_exp(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_exponential(request, entry, ExponentialKind::Exp)
}

pub(super) fn infer_trigonometric(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    function: TrigonometricFunction,
) -> CallInference {
    let name = match function {
        TrigonometricFunction::Sin => "sin",
        TrigonometricFunction::Cos => "cos",
        TrigonometricFunction::Tan => "tan",
    };
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-TRIGONOMETRIC-ARITY",
            format!("{name} requires an input value"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };

    let mut output = input.clone();
    match &mut output.kind {
        ValueKindFact::Numeric(numeric)
            if numeric.domain == NumericDomain::Complex
                && !matches!(numeric.class, NumericClass::Double | NumericClass::Single) =>
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-TRIGONOMETRIC-COMPLEX-INTEGER",
                format!("{name} does not accept complex fixed-width integer input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric) => {
            if !matches!(numeric.class, NumericClass::Double | NumericClass::Single) {
                numeric.class = NumericClass::Double;
            }
            if matches!(output.storage, StorageFact::Sparse) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-TRIGONOMETRIC-SPARSE",
                    format!("{name} does not currently accept sparse input"),
                    0,
                ));
                output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
            } else {
                materialize_output(&mut output);
            }
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize_output(&mut output);
        }
        ValueKindFact::Symbolic => {}
        ValueKindFact::Unknown => preserve_shape_on_dynamic_input(&mut output),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-TRIGONOMETRIC-INPUT",
                format!("{name} requires numeric, logical, character, or symbolic input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }

    match request.arguments.as_slice() {
        [_] => {
            if matches!(output.residency, ResidencyFact::Device { .. }) {
                output.residency = ResidencyFact::Unknown;
            }
        }
        [_, keyword, prototype] => {
            let keyword_literal =
                request
                    .literals
                    .literal_args
                    .get(1)
                    .and_then(|literal| match literal {
                        LiteralValue::String(value)
                        | LiteralValue::Character(value)
                        | LiteralValue::Keyword(value) => Some(value.as_str()),
                        _ => None,
                    });
            match keyword_literal {
                Some(value) if value.eq_ignore_ascii_case("like") => {
                    apply_trigonometric_prototype(&mut output, prototype, name, &mut diagnostics);
                }
                Some(_) => diagnostics.push(argument_error(
                    "RM-CATALOG-TRIGONOMETRIC-LIKE",
                    format!("{name} accepts only the \"like\" option"),
                    1,
                )),
                None if matches!(
                    keyword.kind,
                    ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
                ) =>
                {
                    output.residency = ResidencyFact::Unknown;
                }
                None => diagnostics.push(argument_error(
                    "RM-CATALOG-TRIGONOMETRIC-LIKE",
                    format!("{name} requires \"like\" as its second input"),
                    1,
                )),
            }
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-TRIGONOMETRIC-ARITY",
                format!(
                    "{name} accepts one input or an input followed by \"like\" and a prototype"
                ),
                request.arguments.len().saturating_sub(1),
            ));
            output.residency = ResidencyFact::Unknown;
        }
    }

    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn infer_pi_scaled_trigonometric(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    function: PiScaledTrigonometricFunction,
) -> CallInference {
    let name = match function {
        PiScaledTrigonometricFunction::Sin => "sinpi",
        PiScaledTrigonometricFunction::Cos => "cospi",
    };
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-PI-TRIGONOMETRIC-ARITY",
            format!("{name} requires exactly one input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-PI-TRIGONOMETRIC-ARITY",
            format!("{name} accepts exactly one input"),
            1,
        ));
    }

    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-PI-TRIGONOMETRIC-SPARSE",
            format!("{name} does not currently accept sparse input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let output_residency = match function {
        PiScaledTrigonometricFunction::Sin => ResidencyFact::Host,
        PiScaledTrigonometricFunction::Cos => input.residency.clone(),
    };

    let mut output = input.clone();
    match &mut output.kind {
        ValueKindFact::Numeric(numeric)
            if numeric.domain == NumericDomain::Complex
                && !matches!(numeric.class, NumericClass::Double | NumericClass::Single) =>
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-PI-TRIGONOMETRIC-COMPLEX-INTEGER",
                format!("{name} does not accept complex fixed-width integer input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric) => {
            if !matches!(numeric.class, NumericClass::Double | NumericClass::Single) {
                numeric.class = NumericClass::Double;
            }
            materialize_output(&mut output);
            output.residency = output_residency.clone();
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize_output(&mut output);
            output.residency = output_residency.clone();
        }
        ValueKindFact::Unknown => {
            preserve_shape_on_dynamic_input(&mut output);
            output.residency = output_residency;
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-PI-TRIGONOMETRIC-INPUT",
                format!("{name} requires numeric, logical, or character input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }

    finish_fixed(entry, request, output, diagnostics)
}

fn apply_trigonometric_prototype(
    output: &mut ValueFact,
    prototype: &ValueFact,
    name: &str,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    match &prototype.kind {
        ValueKindFact::Numeric(prototype_numeric) => {
            if let ValueKindFact::Numeric(output_numeric) = &mut output.kind {
                if prototype_numeric.domain == NumericDomain::Complex {
                    output_numeric.domain = NumericDomain::Complex;
                }
            }
            output.residency = prototype.residency.clone();
        }
        ValueKindFact::Logical | ValueKindFact::Unknown => {
            output.residency = prototype.residency.clone();
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-TRIGONOMETRIC-PROTOTYPE",
                format!("{name} requires a numeric or logical prototype"),
                2,
            ));
            output.residency = ResidencyFact::Unknown;
        }
    }
}

pub(super) fn infer_expm1(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_exponential(request, entry, ExponentialKind::Expm1)
}

pub(super) fn infer_log1p(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOG1P-ARITY",
            "log1p requires exactly one input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOG1P-ARITY",
            "log1p accepts exactly one input",
            1,
        ));
    }
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOG1P-SPARSE",
            "log1p does not currently accept sparse input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let mut output = input.clone();
    let literal_domain = request
        .literals
        .literal_args
        .first()
        .and_then(log1p_literal_domain);
    let mut changes_class = false;
    match &input.kind {
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) if !matches!(class, NumericClass::Double | NumericClass::Single) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-LOG1P-COMPLEX-INTEGER",
                "log1p does not accept complex fixed-width integer input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) => {
            output.kind = numeric_kind(*class, NumericDomain::Complex);
            materialize_output(&mut output);
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }) => {
            let output_class = if matches!(class, NumericClass::Double | NumericClass::Single) {
                *class
            } else {
                changes_class = true;
                NumericClass::Double
            };
            let domain = literal_domain.or_else(|| {
                matches!(
                    class,
                    NumericClass::UInt8
                        | NumericClass::UInt16
                        | NumericClass::UInt32
                        | NumericClass::UInt64
                )
                .then_some(NumericDomain::Real)
            });
            if let Some(domain) = domain {
                output.kind = numeric_kind(output_class, domain);
                materialize_output(&mut output);
            } else {
                preserve_shape_on_dynamic_input(&mut output);
            }
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            changes_class = true;
            materialize_output(&mut output);
        }
        ValueKindFact::Object(_) | ValueKindFact::Unknown => {
            preserve_shape_on_dynamic_input(&mut output);
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-LOG1P-INPUT",
                "log1p requires numeric, logical, or character input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    if changes_class && matches!(output.residency, ResidencyFact::Device { .. }) {
        output.residency = ResidencyFact::Unknown;
    }
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn infer_logarithm(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    base: LogarithmBase,
) -> CallInference {
    let name = match base {
        LogarithmBase::Natural => "log",
        LogarithmBase::Common => "log10",
    };
    let (output, diagnostics) =
        infer_logarithm_value(request, name, matches!(base, LogarithmBase::Natural));
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn infer_root(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    kind: RootKind,
) -> CallInference {
    let name = match kind {
        RootKind::Principal => "sqrt",
        RootKind::RealOnly => "realsqrt",
    };
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-ROOT-ARITY",
            format!("{name} requires exactly one input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ROOT-ARITY",
            format!("{name} accepts exactly one input"),
            1,
        ));
    }

    if kind == RootKind::Principal && matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SQRT-SPARSE",
            "sqrt does not currently accept sparse input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let literal = request
        .literals
        .literal_args
        .first()
        .and_then(root_literal_domain);
    let mut output = input.clone();
    match kind {
        RootKind::Principal => infer_principal_root(input, literal, &mut output, &mut diagnostics),
        RootKind::RealOnly => infer_real_root(input, literal, &mut output, &mut diagnostics),
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn infer_principal_root(
    input: &ValueFact,
    literal: Option<RootLiteralDomain>,
    output: &mut ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    match &input.kind {
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) if !matches!(class, NumericClass::Double | NumericClass::Single) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-SQRT-COMPLEX-INTEGER",
                "sqrt does not accept complex fixed-width integer input",
                0,
            ));
            *output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) => {
            output.kind = numeric_kind(*class, NumericDomain::Complex);
            materialize_output(output);
            dynamic_device_residency(output);
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }) => {
            let output_class = if matches!(class, NumericClass::Double | NumericClass::Single) {
                *class
            } else {
                NumericClass::Double
            };
            let domain = literal
                .map(RootLiteralDomain::principal_output)
                .or_else(|| {
                    matches!(
                        class,
                        NumericClass::UInt8
                            | NumericClass::UInt16
                            | NumericClass::UInt32
                            | NumericClass::UInt64
                    )
                    .then_some(NumericDomain::Real)
                });
            if let Some(domain) = domain {
                output.kind = numeric_kind(output_class, domain);
                materialize_output(output);
                dynamic_device_residency(output);
            } else {
                preserve_shape_on_dynamic_input(output);
            }
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize_output(output);
            dynamic_device_residency(output);
        }
        ValueKindFact::Symbolic => {}
        ValueKindFact::Unknown => preserve_shape_on_dynamic_input(output),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-SQRT-INPUT",
                "sqrt requires numeric, logical, character, or symbolic input",
                0,
            ));
            *output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
}

fn infer_real_root(
    input: &ValueFact,
    literal: Option<RootLiteralDomain>,
    output: &mut ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    match &input.kind {
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double | NumericClass::Single,
            domain: NumericDomain::Real,
        }) => {
            if matches!(
                literal,
                Some(RootLiteralDomain::Negative | RootLiteralDomain::Complex)
            ) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-REALSQRT-DOMAIN",
                    "realsqrt input must be real and nonnegative",
                    0,
                ));
            }
            materialize_output_preserving_storage(output);
            if matches!(output.storage, StorageFact::Sparse) {
                output.residency = ResidencyFact::Host;
            } else {
                dynamic_device_residency(output);
            }
        }
        ValueKindFact::Unknown => preserve_shape_on_dynamic_input(output),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-REALSQRT-INPUT",
                "realsqrt requires real single or double input",
                0,
            ));
            *output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
}

fn dynamic_device_residency(output: &mut ValueFact) {
    if matches!(output.residency, ResidencyFact::Device { .. }) {
        output.residency = ResidencyFact::Unknown;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootLiteralDomain {
    Nonnegative,
    Negative,
    Complex,
}

impl RootLiteralDomain {
    const fn principal_output(self) -> NumericDomain {
        match self {
            Self::Nonnegative => NumericDomain::Real,
            Self::Negative | Self::Complex => NumericDomain::Complex,
        }
    }
}

fn root_literal_domain(literal: &LiteralValue) -> Option<RootLiteralDomain> {
    match literal {
        LiteralValue::Number(value) => Some(root_real_domain(*value)),
        LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => {
            text.parse().ok().map(root_real_domain)
        }
        LiteralValue::Complex { .. } => Some(RootLiteralDomain::Complex),
        LiteralValue::Bool(_) | LiteralValue::Character(_) | LiteralValue::Empty => {
            Some(RootLiteralDomain::Nonnegative)
        }
        LiteralValue::Vector(values) => root_literal_sequence_domain(values.iter()),
        LiteralValue::Matrix(rows) => root_literal_sequence_domain(rows.iter().flatten()),
        LiteralValue::String(_)
        | LiteralValue::Keyword(_)
        | LiteralValue::Symbolic(_)
        | LiteralValue::Unknown => None,
    }
}

fn root_literal_sequence_domain<'a>(
    values: impl Iterator<Item = &'a LiteralValue>,
) -> Option<RootLiteralDomain> {
    let mut domain = RootLiteralDomain::Nonnegative;
    for value in values {
        match root_literal_domain(value)? {
            RootLiteralDomain::Complex => return Some(RootLiteralDomain::Complex),
            RootLiteralDomain::Negative => domain = RootLiteralDomain::Negative,
            RootLiteralDomain::Nonnegative => {}
        }
    }
    Some(domain)
}

fn root_real_domain(value: f64) -> RootLiteralDomain {
    if value < 0.0 {
        RootLiteralDomain::Negative
    } else {
        RootLiteralDomain::Nonnegative
    }
}

pub(super) fn infer_log2(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let requested = request.outputs.requested.known_count();
    if requested == Some(2) {
        return infer_log2_dissection(request, entry);
    }

    let (value_output, mut diagnostics) = infer_logarithm_value(request, "log2", false);
    let exponent_output = log2_dissection_output(
        request.arguments.first(),
        &mut diagnostics,
        requested.is_none(),
    );
    let mut contract = CallContract::fixed(vec![value_output, exponent_output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

fn infer_log2_dissection(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = log2_dissection_output(request.arguments.first(), &mut diagnostics, true);
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOG2-ARITY",
            "log2 accepts exactly one input",
            1,
        ));
    }
    let mut contract = CallContract::fixed(vec![output.clone(), output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

fn log2_dissection_output(
    input: Option<&ValueFact>,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
    diagnose_invalid: bool,
) -> ValueFact {
    let Some(input) = input else {
        if diagnose_invalid {
            diagnostics.push(argument_error(
                "RM-CATALOG-LOG2-ARITY",
                "log2 requires exactly one input",
                0,
            ));
        }
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    if matches!(input.residency, ResidencyFact::Device { .. }) {
        if diagnose_invalid {
            diagnostics.push(argument_error(
                "RM-CATALOG-LOG2-GPU-DISSECTION",
                "two-output log2 does not support GPU-resident input",
                0,
            ));
        }
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    if matches!(input.storage, StorageFact::Sparse) {
        if diagnose_invalid {
            diagnostics.push(argument_error(
                "RM-CATALOG-LOG2-SPARSE",
                "log2 does not currently accept sparse input",
                0,
            ));
        }
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }

    let mut output = input.clone();
    match &mut output.kind {
        ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Complex => {
            if diagnose_invalid {
                diagnostics.push(argument_error(
                    "RM-CATALOG-LOG2-COMPLEX-DISSECTION",
                    "two-output log2 requires real input under the current compatibility pin",
                    0,
                ));
            }
            return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric) => {
            if !matches!(numeric.class, NumericClass::Double | NumericClass::Single) {
                numeric.class = NumericClass::Double;
            }
            numeric.domain = NumericDomain::Real;
            materialize_output(&mut output);
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize_output(&mut output);
        }
        ValueKindFact::Object(object)
            if object.runtime_class.as_ref().is_some_and(|class| {
                class.is(runmat_types::standard::TABLE)
                    || class.is(runmat_types::standard::TIMETABLE)
            }) =>
        {
            object.properties.clear();
            object.properties_complete = false;
            output.alias = AliasFact::Unique;
            output.mutation = MutationFact::ValueSemantics;
        }
        ValueKindFact::Object(_) | ValueKindFact::Unknown if !diagnose_invalid => {
            preserve_shape_on_dynamic_input(&mut output);
        }
        _ => {
            if diagnose_invalid {
                diagnostics.push(argument_error(
                    "RM-CATALOG-LOG2-DISSECTION-INPUT",
                    "two-output log2 requires real single, double, or supported tabular input",
                    0,
                ));
            }
            output = ValueFact::unknown(if diagnose_invalid {
                DynamicReason::UnsupportedRepresentation
            } else {
                DynamicReason::RuntimeValue
            });
        }
    }
    output
}

fn infer_logarithm_value(
    request: &CallRequest,
    name: &str,
    accepts_symbolic: bool,
) -> (ValueFact, Vec<runmat_types::InferenceDiagnostic>) {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOGARITHM-ARITY",
            format!("{name} requires exactly one input"),
            0,
        ));
        return (ValueFact::unknown(DynamicReason::RuntimeValue), diagnostics);
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOGARITHM-ARITY",
            format!("{name} accepts exactly one input"),
            1,
        ));
    }
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOGARITHM-SPARSE",
            format!("{name} does not currently accept sparse input"),
            0,
        ));
        return (
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let mut output = input.clone();
    let literal_domain = request
        .literals
        .literal_args
        .first()
        .and_then(logarithm_literal_domain);
    let mut changes_class = false;
    match &input.kind {
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) if !matches!(class, NumericClass::Double | NumericClass::Single) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-LOGARITHM-COMPLEX-INTEGER",
                format!("{name} does not accept complex fixed-width integer input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) => {
            output.kind = numeric_kind(*class, NumericDomain::Complex);
            materialize_output(&mut output);
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }) => {
            let output_class = if matches!(class, NumericClass::Double | NumericClass::Single) {
                *class
            } else {
                changes_class = true;
                NumericClass::Double
            };
            let domain = literal_domain.or_else(|| {
                matches!(
                    class,
                    NumericClass::UInt8
                        | NumericClass::UInt16
                        | NumericClass::UInt32
                        | NumericClass::UInt64
                )
                .then_some(NumericDomain::Real)
            });
            if let Some(domain) = domain {
                output.kind = numeric_kind(output_class, domain);
                materialize_output(&mut output);
            } else {
                preserve_shape_on_dynamic_input(&mut output);
            }
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            changes_class = true;
            materialize_output(&mut output);
        }
        ValueKindFact::Symbolic if accepts_symbolic => {
            materialize_output_preserving_storage(&mut output);
        }
        ValueKindFact::Object(object)
            if object.runtime_class.as_ref().is_some_and(|class| {
                class.is(runmat_types::standard::TABLE)
                    || class.is(runmat_types::standard::TIMETABLE)
            }) =>
        {
            let ValueKindFact::Object(object) = &mut output.kind else {
                unreachable!("tabular object branch preserves object fact")
            };
            object.properties.clear();
            object.properties_complete = false;
            output.alias = AliasFact::Unique;
            output.mutation = MutationFact::ValueSemantics;
        }
        ValueKindFact::Object(_) | ValueKindFact::Unknown => {
            preserve_shape_on_dynamic_input(&mut output);
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-LOGARITHM-INPUT",
                format!(
                    "{name} requires numeric, logical, character, or supported tabular input{}",
                    if accepts_symbolic {
                        ", or a symbolic expression"
                    } else {
                        ""
                    }
                ),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    if changes_class && matches!(output.residency, ResidencyFact::Device { .. }) {
        output.residency = ResidencyFact::Unknown;
    }
    (output, diagnostics)
}

fn logarithm_literal_domain(literal: &LiteralValue) -> Option<NumericDomain> {
    match literal {
        LiteralValue::Number(value) => Some(logarithm_real_domain(*value)),
        LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => {
            text.parse().ok().map(logarithm_real_domain)
        }
        LiteralValue::Complex { .. } => Some(NumericDomain::Complex),
        LiteralValue::Bool(_) | LiteralValue::Character(_) | LiteralValue::Empty => {
            Some(NumericDomain::Real)
        }
        LiteralValue::Vector(values) => logarithm_literal_sequence_domain(values.iter()),
        LiteralValue::Matrix(rows) => logarithm_literal_sequence_domain(rows.iter().flatten()),
        LiteralValue::String(_)
        | LiteralValue::Keyword(_)
        | LiteralValue::Symbolic(_)
        | LiteralValue::Unknown => None,
    }
}

fn logarithm_literal_sequence_domain<'a>(
    values: impl Iterator<Item = &'a LiteralValue>,
) -> Option<NumericDomain> {
    let mut domain = NumericDomain::Real;
    for value in values {
        match logarithm_literal_domain(value)? {
            NumericDomain::Complex => domain = NumericDomain::Complex,
            NumericDomain::Real => {}
        }
    }
    Some(domain)
}

fn logarithm_real_domain(value: f64) -> NumericDomain {
    if value < 0.0 {
        NumericDomain::Complex
    } else {
        NumericDomain::Real
    }
}

fn log1p_literal_domain(literal: &LiteralValue) -> Option<NumericDomain> {
    match literal {
        LiteralValue::Number(value) => Some(log1p_real_domain(*value)),
        LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => {
            text.parse().ok().map(log1p_real_domain)
        }
        LiteralValue::Complex { .. } => Some(NumericDomain::Complex),
        LiteralValue::Bool(_) | LiteralValue::Character(_) | LiteralValue::Empty => {
            Some(NumericDomain::Real)
        }
        LiteralValue::Vector(values) => log1p_literal_sequence_domain(values.iter()),
        LiteralValue::Matrix(rows) => log1p_literal_sequence_domain(rows.iter().flatten()),
        LiteralValue::String(_)
        | LiteralValue::Keyword(_)
        | LiteralValue::Symbolic(_)
        | LiteralValue::Unknown => None,
    }
}

fn log1p_literal_sequence_domain<'a>(
    values: impl Iterator<Item = &'a LiteralValue>,
) -> Option<NumericDomain> {
    let mut domain = NumericDomain::Real;
    for value in values {
        match log1p_literal_domain(value)? {
            NumericDomain::Complex => domain = NumericDomain::Complex,
            NumericDomain::Real => {}
        }
    }
    Some(domain)
}

fn log1p_real_domain(value: f64) -> NumericDomain {
    if value < -1.0 {
        NumericDomain::Complex
    } else {
        NumericDomain::Real
    }
}

#[derive(Debug, Clone, Copy)]
enum ExponentialKind {
    Exp,
    Expm1,
}

impl ExponentialKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Exp => "exp",
            Self::Expm1 => "expm1",
        }
    }

    const fn accepts_symbolic(self) -> bool {
        matches!(self, Self::Exp)
    }

    const fn preserves_sparse_zeros(self) -> bool {
        matches!(self, Self::Expm1)
    }
}

fn infer_exponential(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    operation: ExponentialKind,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-EXPONENTIAL-ARITY",
            format!("{} requires exactly one input", operation.name()),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-EXPONENTIAL-ARITY",
            format!("{} accepts exactly one input", operation.name()),
            1,
        ));
    }

    let mut output = input.clone();
    let mut changes_numeric_class = false;
    match &mut output.kind {
        ValueKindFact::Numeric(numeric) => {
            if numeric.domain == NumericDomain::Complex
                && !matches!(numeric.class, NumericClass::Double | NumericClass::Single)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-EXPONENTIAL-COMPLEX-INTEGER",
                    format!(
                        "{} does not accept complex fixed-width integer input",
                        operation.name()
                    ),
                    0,
                ));
                output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
                return finish_fixed(entry, request, output, diagnostics);
            }
            if !matches!(numeric.class, NumericClass::Double | NumericClass::Single) {
                numeric.class = NumericClass::Double;
                changes_numeric_class = true;
            }
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            changes_numeric_class = true;
        }
        ValueKindFact::Symbolic if operation.accepts_symbolic() => {
            return finish_fixed(entry, request, output, diagnostics);
        }
        ValueKindFact::Object(object) => {
            if object.runtime_class.as_ref().is_some_and(|class| {
                class.is(runmat_types::standard::TABLE)
                    || class.is(runmat_types::standard::TIMETABLE)
            }) {
                object.properties.clear();
                object.properties_complete = false;
                output.alias = AliasFact::Unique;
                output.mutation = MutationFact::ValueSemantics;
            } else {
                preserve_shape_on_dynamic_input(&mut output);
            }
            return finish_fixed(entry, request, output, diagnostics);
        }
        ValueKindFact::Unknown => {
            preserve_shape_on_dynamic_input(&mut output);
            return finish_fixed(entry, request, output, diagnostics);
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-EXPONENTIAL-INPUT",
                format!(
                    "{} requires numeric, logical, character, or supported tabular input{}",
                    operation.name(),
                    if operation.accepts_symbolic() {
                        ", or a symbolic expression"
                    } else {
                        ""
                    }
                ),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
            return finish_fixed(entry, request, output, diagnostics);
        }
    }

    if matches!(output.storage, StorageFact::Sparse) {
        output.residency = ResidencyFact::Host;
        if !operation.preserves_sparse_zeros() {
            output.storage = StorageFact::Dense;
        }
    } else if changes_numeric_class && matches!(output.residency, ResidencyFact::Device { .. }) {
        output.residency = ResidencyFact::Unknown;
    }
    materialize_output_preserving_storage(&mut output);
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn infer_phase_angle(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-PHASE-ANGLE-ARITY",
            "angle requires exactly one input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-PHASE-ANGLE-ARITY",
            "angle accepts exactly one input",
            1,
        ));
    }
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-PHASE-ANGLE-SPARSE",
            "angle does not currently accept sparse input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let mut output = input.clone();
    match &mut output.kind {
        ValueKindFact::Numeric(numeric)
            if matches!(numeric.class, NumericClass::Double | NumericClass::Single) =>
        {
            numeric.domain = NumericDomain::Real;
            materialize_output(&mut output);
        }
        ValueKindFact::Unknown => preserve_shape_on_dynamic_input(&mut output),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-PHASE-ANGLE-INPUT",
                "angle requires real or complex single- or double-precision input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn infer_signum(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-SIGNUM-ARITY",
            "sign requires exactly one input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SIGNUM-ARITY",
            "sign accepts exactly one input",
            1,
        ));
    }
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SIGNUM-SPARSE",
            "sign does not currently accept sparse input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let mut output = input.clone();
    let mut changes_class = false;
    match &output.kind {
        ValueKindFact::Numeric(NumericFact { class, domain })
            if domain == &NumericDomain::Complex
                && !matches!(class, NumericClass::Double | NumericClass::Single) =>
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-SIGNUM-COMPLEX-INTEGER",
                "sign does not accept complex fixed-width integer input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(_) => materialize_output(&mut output),
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            changes_class = true;
            materialize_output(&mut output);
        }
        ValueKindFact::Unknown => preserve_shape_on_dynamic_input(&mut output),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-SIGNUM-INPUT",
                "sign requires numeric, logical, or character input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    if changes_class && matches!(output.residency, ResidencyFact::Device { .. }) {
        output.residency = ResidencyFact::Unknown;
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn materialize_output(output: &mut ValueFact) {
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
}

fn materialize_output_preserving_storage(output: &mut ValueFact) {
    if matches!(output.storage, StorageFact::Sparse) {
        output.view = ViewFact::Materialized;
        output.alias = AliasFact::Unique;
        output.mutation = MutationFact::ValueSemantics;
    } else {
        materialize_output(output);
    }
}

fn preserve_shape_on_dynamic_input(output: &mut ValueFact) {
    let shape = output.shape.clone();
    *output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.shape = shape;
}
