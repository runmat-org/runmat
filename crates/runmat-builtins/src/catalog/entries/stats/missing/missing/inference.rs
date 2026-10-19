use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, InferenceDiagnostic, LiteralValue,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn dimension(literal: &LiteralValue) -> Result<Option<usize>, ()> {
    match literal {
        LiteralValue::Unknown => Ok(None),
        LiteralValue::Integer { text, .. } => text.parse::<usize>().map(Some).map_err(|_| ()),
        LiteralValue::Number(value) => {
            if !value.is_finite()
                || *value < 0.0
                || value.fract() != 0.0
                || *value >= usize::MAX as f64
            {
                Err(())
            } else {
                Ok(Some(*value as usize))
            }
        }
        LiteralValue::Real { text, class } => {
            let value = if *class == runmat_types::NumericClass::Single {
                text.parse::<f32>().ok().map(f64::from)
            } else {
                text.parse::<f64>().ok()
            };
            value.map_or(Err(()), |value| dimension(&LiteralValue::Number(value)))
        }
        _ => Err(()),
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut dimensions = Vec::new();
    let mut unknown = false;
    let mut diagnostics = Vec::new();
    let mut vector = false;
    let mut index = 0;
    while index < request.arguments.len() {
        match request.literals.literal_args.get(index) {
            Some(LiteralValue::Unknown) => unknown = true,
            Some(LiteralValue::Vector(values)) if index == 0 && request.arguments.len() == 1 => {
                vector = true;
                for value in values {
                    match dimension(value) {
                        Ok(size) => dimensions.push(size),
                        Err(()) => diagnostics.push(InferenceDiagnostic::error(
                            "RM.MISSING.INVALID_ARGUMENT",
                            "missing sizes must be nonnegative finite integers",
                        )),
                    }
                }
            }
            Some(LiteralValue::Empty) if index == 0 && request.arguments.len() == 1 => {
                vector = true;
                dimensions.extend([Some(0), Some(0)]);
            }
            Some(LiteralValue::String(text) | LiteralValue::Character(text))
                if text.eq_ignore_ascii_case("like") =>
            {
                if index + 1 >= request.arguments.len() {
                    diagnostics.push(InferenceDiagnostic::error(
                        "RM.MISSING.INVALID_ARGUMENT",
                        "missing 'like' requires a prototype",
                    ));
                }
                index += 1;
            }
            Some(literal) => match dimension(literal) {
                Ok(size) => dimensions.push(size),
                Err(()) => diagnostics.push(InferenceDiagnostic::error(
                    "RM.MISSING.INVALID_ARGUMENT",
                    "missing sizes must be nonnegative finite integers",
                )),
            },
            None => unknown = true,
        }
        index += 1;
    }
    let shape = if unknown {
        ShapeFact::Unknown
    } else if dimensions.is_empty() {
        if vector {
            ShapeFact::from(vec![Some(0), Some(0)])
        } else {
            ShapeFact::Scalar
        }
    } else {
        if dimensions.len() == 1 {
            dimensions.push(dimensions[0]);
        }
        ShapeFact::from(dimensions)
    };
    let mut contract = CallContract::fixed(vec![ValueFact::proven(
        ValueKindFact::String,
        shape,
        StorageFact::Dense,
    )]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
