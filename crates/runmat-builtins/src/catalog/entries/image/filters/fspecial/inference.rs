use super::FspecialFilterKind as Kind;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CertaintyFact, DynamicReason,
    LiteralValue, NumericDomain, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.arguments.len();
    if !(1..=3).contains(&count) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FSPECIAL-ARITY",
            "fspecial expects a filter type and at most two parameters",
            0,
        ));
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-FSPECIAL-OUTPUTS",
            "fspecial returns one kernel",
            1,
        ));
    }
    if request.arguments.first().is_some_and(|fact| {
        !matches!(
            fact.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        )
    }) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FSPECIAL-TYPE",
            "filter type must be a string scalar or character row",
            0,
        ));
    }
    let spelling = request.literals.literal_string_at(0);
    let kind = spelling.as_deref().and_then(Kind::from_name);
    if spelling.is_some() {
        let maximum = match kind {
            Some(kind) => Some(kind.maximum_arguments()),
            None => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FSPECIAL-KIND",
                    "unsupported fspecial filter type",
                    0,
                ));
                None
            }
        };
        if maximum.is_some_and(|maximum| count > maximum) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FSPECIAL-FORM",
                "too many parameters for selected fspecial filter",
                count.saturating_sub(1),
            ));
        }
    }
    for index in 1..count {
        let value = &request.arguments[index];
        if matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque)
            || !matches!(
                value.kind,
                ValueKindFact::Numeric(runmat_types::NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Logical
                    | ValueKindFact::Unknown
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FSPECIAL-PARAM",
                "filter parameters must be real numeric or logical values",
                index,
            ));
        }
        if request.literals.integer_is_exact_binary64_at(index) == Some(false)
            && !(index == 1
                && matches!(
                    kind,
                    Some(Kind::Average | Kind::Gaussian | Kind::Log | Kind::Motion)
                ))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FSPECIAL-WIDE",
                "computational integer parameter must be exactly representable in binary64",
                index,
            ));
        }
    }
    let dims = shape_for(kind, request);
    // The ambient resident-output extension can publish single or double;
    // the call request does not prove that environment/provider decision.
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.shape = dims.map_or(ShapeFact::Ranked { rank: 2 }, |[rows, cols]| {
        ShapeFact::from(vec![rows, cols])
    });
    output.storage = StorageFact::Dense;
    output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn dimension(value: f64) -> Option<usize> {
    (value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value < usize::MAX as f64)
        .then_some(value as usize)
}

fn size_control(request: &CallRequest) -> Option<[Option<usize>; 2]> {
    if request.arguments.len() < 2 {
        return None;
    }
    if let Some(value) = scalar_structural(request, 1) {
        return Some([Some(value), Some(value)]);
    }
    let vector = request.literals.numeric_vector_at(1)?;
    match vector.as_slice() {
        [one] => Some([*one, *one]),
        [rows, cols] => Some([*rows, *cols]),
        _ => None,
    }
}

fn scalar_structural(request: &CallRequest, index: usize) -> Option<usize> {
    match request.literals.literal_args.get(index) {
        Some(LiteralValue::Integer { text, .. }) => text.parse().ok(),
        _ => request.literals.numeric_at(index).and_then(dimension),
    }
}

fn shape_for(kind: Option<Kind>, request: &CallRequest) -> Option<[Option<usize>; 2]> {
    let fixed = |n| Some([Some(n), Some(n)]);
    match kind? {
        Kind::Average | Kind::Gaussian => {
            if request.arguments.len() == 1 {
                fixed(3)
            } else {
                size_control(request)
            }
        }
        Kind::Log => {
            if request.arguments.len() == 1 {
                fixed(5)
            } else {
                size_control(request)
            }
        }
        Kind::Laplacian | Kind::Prewitt | Kind::Sobel | Kind::Unsharp => fixed(3),
        Kind::Disk => {
            let radius = if request.arguments.len() == 1 {
                Some(5.0)
            } else {
                request.literals.numeric_at(1)
            }?;
            let size = dimension(radius.ceil())?.checked_mul(2)?.checked_add(1)?;
            fixed(size)
        }
        Kind::Motion => {
            let length = if request.arguments.len() == 1 {
                Some(9.0)
            } else {
                request.literals.numeric_at(1)
            }?;
            if !length.is_finite() || length <= 0.0 {
                return None;
            }
            let length = if matches!(
                request.literals.literal_args.get(1),
                Some(LiteralValue::Integer { .. })
            ) {
                scalar_structural(request, 1)?
            } else {
                dimension(length.round())?
            };
            let size = if length % 2 == 0 {
                length.checked_add(1)?
            } else {
                length
            };
            fixed(size)
        }
    }
}
