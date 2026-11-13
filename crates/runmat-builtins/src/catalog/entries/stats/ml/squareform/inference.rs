use super::super::distance_geometry::{condensed_length, matrix_size_for_condensed_length};
use super::SquareformDirection;
use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SQUAREFORM-ARITY",
            "squareform expects a distance input and at most one direction",
            0,
        ));
    }
    let direction = match request.arguments.get(1) {
        None => Some(SquareformDirection::Auto),
        Some(_) => match request.literals.literal_args.get(1) {
            Some(
                LiteralValue::Keyword(text)
                | LiteralValue::String(text)
                | LiteralValue::Character(text),
            ) => {
                let parsed = SquareformDirection::from_keyword(text);
                if parsed.is_none() {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-SQUAREFORM-DIRECTION",
                        "squareform direction must be 'tomatrix' or 'tovector'",
                        1,
                    ));
                }
                parsed
            }
            Some(LiteralValue::Unknown) | None => None,
            Some(_) => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SQUAREFORM-DIRECTION",
                    "squareform direction must be text",
                    1,
                ));
                None
            }
        },
    };
    let output = match request.arguments.first() {
        Some(input) => {
            let kind = match &input.kind {
                ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real => {
                    ValueKindFact::Numeric(*numeric)
                }
                ValueKindFact::Logical => ValueKindFact::Logical,
                ValueKindFact::Unknown => ValueKindFact::Unknown,
                _ => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-SQUAREFORM-CLASS",
                        "squareform requires a real numeric or logical distance array",
                        0,
                    ));
                    ValueKindFact::Unknown
                }
            };
            let shape = output_shape(&input.shape, direction, &mut diagnostics);
            let mut result = ValueFact::proven(kind, shape, StorageFact::Dense);
            if matches!(result.kind, ValueKindFact::Unknown)
                || matches!(result.shape, ShapeFact::Unknown | ShapeFact::Ranked { .. })
            {
                result.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
            }
            result
        }
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-SQUAREFORM-OUTPUTS",
            "squareform returns one output",
            0,
        ));
    }
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn output_shape(
    input: &ShapeFact,
    direction: Option<SquareformDirection>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ShapeFact {
    let Some(direction) = direction else {
        return ShapeFact::Unknown;
    };
    let dims = match input {
        ShapeFact::Scalar => [Some(1), Some(1)],
        ShapeFact::Shaped { dims } if dims.len() == 2 => [
            match dims[0] {
                DimensionFact::Known(n) => Some(n),
                _ => None,
            },
            match dims[1] {
                DimensionFact::Known(n) => Some(n),
                _ => None,
            },
        ],
        ShapeFact::Shaped { .. } => {
            diagnostics.push(argument_error(
                "RM-CATALOG-SQUAREFORM-RANK",
                "squareform requires a vector or two-dimensional matrix",
                0,
            ));
            return ShapeFact::Unknown;
        }
        ShapeFact::Ranked { rank: 2 } => [None, None],
        ShapeFact::Ranked { .. } | ShapeFact::Unknown => return ShapeFact::Unknown,
    };
    let vector = dims[0] == Some(1) || dims[1] == Some(1);
    let conversion = match direction {
        SquareformDirection::Auto if vector => SquareformDirection::ToMatrix,
        SquareformDirection::Auto if dims[0].is_some() && dims[1].is_some() => {
            SquareformDirection::ToVector
        }
        SquareformDirection::Auto => return ShapeFact::Unknown,
        forced => forced,
    };
    match conversion {
        SquareformDirection::ToMatrix => {
            if dims[0].is_some() && dims[1].is_some() && !vector {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SQUAREFORM-VECTOR",
                    "vector input is required for 'tomatrix'",
                    0,
                ));
                return ShapeFact::Unknown;
            }
            let Some(length) = dims[0].zip(dims[1]).and_then(|(a, b)| a.checked_mul(b)) else {
                return ShapeFact::from(vec![None, None]);
            };
            match matrix_size_for_condensed_length(length) {
                Some(size) => ShapeFact::from(vec![Some(size), Some(size)]),
                None => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-SQUAREFORM-LENGTH",
                        "condensed vector length must be triangular",
                        0,
                    ));
                    ShapeFact::Unknown
                }
            }
        }
        SquareformDirection::ToVector => {
            if dims[0].is_some() && dims[1].is_some() && dims[0] != dims[1] {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SQUAREFORM-SQUARE",
                    "matrix input must be square for 'tovector'",
                    0,
                ));
                return ShapeFact::Unknown;
            }
            let length = dims[0].and_then(condensed_length);
            ShapeFact::from(vec![Some(1), length])
        }
        SquareformDirection::Auto => unreachable!(),
    }
}
