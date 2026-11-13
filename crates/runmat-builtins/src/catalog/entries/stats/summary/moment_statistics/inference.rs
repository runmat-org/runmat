//! Static facts for the flagged deviation and moment reductions.
use super::policy::{moment_flag_from_f64, MomentControl, MomentStatisticKind};
use crate::{BuiltinCatalogEntry, ReductionNaN as MissingMode};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

#[derive(Clone, Debug)]
enum Selector {
    Default,
    All,
    Dimensions(Vec<usize>),
    Dynamic,
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    kind: MomentStatisticKind,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let input = request.arguments.first();
    if input.is_none() {
        diagnostics.push(error("ARITY", "input data is required"));
    }
    let class = input.and_then(|fact| match fact.kind {
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }) => Some(match class {
            NumericClass::Single => NumericClass::Single,
            NumericClass::Double
            | NumericClass::Int8
            | NumericClass::UInt8
            | NumericClass::Int16
            | NumericClass::UInt16
            | NumericClass::Int32
            | NumericClass::UInt32
            | NumericClass::Int64
            | NumericClass::UInt64 => NumericClass::Double,
        }),
        ValueKindFact::Numeric(_) => {
            diagnostics.push(error("COMPLEX", "real numeric input is required"));
            None
        }
        ValueKindFact::Logical => Some(NumericClass::Double),
        ValueKindFact::Unknown => None,
        _ => {
            diagnostics.push(error("INPUT", "numeric input is required"));
            None
        }
    });
    let (selector, _flag, _missing) = parse_options(request, kind, &mut diagnostics);
    let shape = input
        .map(|fact| reduced_shape(&fact.shape, &selector))
        .unwrap_or(ShapeFact::Unknown);
    let mut output = match class {
        Some(class) => ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }),
            shape,
            StorageFact::Unknown,
        ),
        None => {
            let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
            output.shape = shape;
            output
        }
    };
    output.storage = match output.shape.element_count() {
        Some(1) if class == Some(NumericClass::Single) => StorageFact::Dense,
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn parse_options(
    request: &CallRequest,
    kind: MomentStatisticKind,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> (Selector, u8, MissingMode) {
    let mut flag = kind.default_flag();
    let mut selector = Selector::Default;
    let mut missing = MissingMode::Omit;
    let mut index = 1;
    let mut ambiguous_first = false;
    if let Some(first) = request.literals.literal_args.get(1) {
        match first_flag_literal(first) {
            LiteralValue::Empty => index += 1,
            LiteralValue::Bool(value) => {
                flag = u8::from(*value);
                index += 1;
            }
            LiteralValue::Number(_) | LiteralValue::Real { .. } | LiteralValue::Integer { .. } => {
                match literal_flag(first_flag_literal(first)) {
                    Some(value) => flag = value,
                    _ => diagnostics.push(error("FLAG", "flag must be 0 or 1")),
                }
                index += 1;
            }
            LiteralValue::Unknown => {
                // The first runtime value may be a flag or a dimension.
                // Later source-known options cannot resolve that ambiguity.
                ambiguous_first = true;
                index += 1;
            }
            _ => {}
        }
    }
    while index < request.arguments.len() {
        let literal = request
            .literals
            .literal_args
            .get(index)
            .unwrap_or(&LiteralValue::Unknown);
        match literal {
            LiteralValue::Character(text)
            | LiteralValue::String(text)
            | LiteralValue::Keyword(text) => match MomentControl::from_public_option(text) {
                Some(MomentControl::All) => {
                    selector = Selector::All;
                }
                Some(MomentControl::Missing(policy)) => {
                    missing = policy;
                }
                None => diagnostics.push(error("OPTIONS", "unsupported reduction option")),
            },
            LiteralValue::Number(_)
            | LiteralValue::Real { .. }
            | LiteralValue::Integer { .. }
            | LiteralValue::Vector(_)
            | LiteralValue::Matrix(_) => {
                selector = parse_selector(literal, diagnostics);
            }
            LiteralValue::Unknown => selector = Selector::Dynamic,
            _ => diagnostics.push(error("OPTIONS", "unsupported reduction option")),
        }
        index += 1;
    }
    if ambiguous_first || !diagnostics.is_empty() {
        selector = Selector::Dynamic;
    }
    (selector, flag, missing)
}

fn first_flag_literal(literal: &LiteralValue) -> &LiteralValue {
    match literal {
        LiteralValue::Vector(values) if values.is_empty() => &LiteralValue::Empty,
        LiteralValue::Vector(values)
            if values.len() == 1 && is_numeric_flag_literal(&values[0]) =>
        {
            &values[0]
        }
        LiteralValue::Matrix(rows) if rows.is_empty() || rows.iter().all(Vec::is_empty) => {
            &LiteralValue::Empty
        }
        LiteralValue::Matrix(rows)
            if rows.len() == 1 && rows[0].len() == 1 && is_numeric_flag_literal(&rows[0][0]) =>
        {
            &rows[0][0]
        }
        other => other,
    }
}

fn is_numeric_flag_literal(literal: &LiteralValue) -> bool {
    matches!(
        literal,
        LiteralValue::Number(_)
            | LiteralValue::Real { .. }
            | LiteralValue::Integer { .. }
            | LiteralValue::Unknown
    )
}

fn literal_flag(literal: &LiteralValue) -> Option<u8> {
    match literal {
        LiteralValue::Integer { text, .. } => text.parse::<u8>().ok().filter(|flag| *flag <= 1),
        other => LiteralContext::numeric_from_literal(other).and_then(moment_flag_from_f64),
    }
}

fn parse_selector(literal: &LiteralValue, diagnostics: &mut Vec<InferenceDiagnostic>) -> Selector {
    let values: Vec<_> = match literal {
        LiteralValue::Vector(values) => values.iter().collect(),
        LiteralValue::Matrix(rows) if rows.len() == 1 || rows.iter().all(|row| row.len() == 1) => {
            rows.iter().flatten().collect()
        }
        LiteralValue::Matrix(_) => {
            diagnostics.push(error("DIMENSION", "dimension array must be a vector"));
            return Selector::Dynamic;
        }
        other => vec![other],
    };
    let mut dims = Vec::new();
    for value in values {
        if matches!(value, LiteralValue::Unknown) {
            return Selector::Dynamic;
        }
        match literal_dimension(value) {
            Some(dim) if dim > 0 => dims.push(dim),
            _ => diagnostics.push(error("DIMENSION", "dimensions must be positive integers")),
        }
    }
    if dims.is_empty() {
        diagnostics.push(error("DIMENSION", "at least one dimension is required"));
    }
    Selector::Dimensions(dims)
}

fn literal_dimension(literal: &LiteralValue) -> Option<usize> {
    match literal {
        LiteralValue::Integer { text, .. } => text.parse::<usize>().ok(),
        LiteralValue::Bool(value) => Some(usize::from(*value)),
        other => LiteralContext::numeric_from_literal(other)
            .and_then(|value| runmat_types::decode_shape_dimension(value).ok()),
    }
}

fn reduced_shape(input: &ShapeFact, selector: &Selector) -> ShapeFact {
    if matches!(selector, Selector::All) {
        return ShapeFact::Scalar;
    }
    if matches!(selector, Selector::Dynamic) {
        return ShapeFact::Unknown;
    }
    let ShapeFact::Shaped { dims } = input else {
        return match input {
            ShapeFact::Scalar => ShapeFact::Scalar,
            ShapeFact::Ranked { rank } => ShapeFact::Ranked { rank: *rank },
            ShapeFact::Unknown | ShapeFact::Shaped { .. } => ShapeFact::Unknown,
        };
    };
    let mut out = dims.clone();
    let axes = match selector {
        Selector::Default => {
            let mut first = 0;
            for (index, extent) in dims.iter().enumerate() {
                match extent {
                    runmat_types::DimensionFact::Known(1) => {}
                    runmat_types::DimensionFact::Known(_) => {
                        first = index;
                        break;
                    }
                    runmat_types::DimensionFact::Unknown
                    | runmat_types::DimensionFact::Symbolic(_) => {
                        return ShapeFact::Ranked { rank: dims.len() };
                    }
                }
            }
            vec![first]
        }
        Selector::Dimensions(values) => {
            values.iter().filter_map(|dim| dim.checked_sub(1)).collect()
        }
        Selector::All | Selector::Dynamic => unreachable!("handled above"),
    };
    for axis in axes {
        if let Some(dim) = out.get_mut(axis) {
            *dim = runmat_types::DimensionFact::Known(1);
        }
    }
    ShapeFact::Shaped { dims: out }
}

fn error(code: &'static str, message: impl Into<String>) -> InferenceDiagnostic {
    InferenceDiagnostic::error(code, message)
}

#[cfg(test)]
mod tests;
