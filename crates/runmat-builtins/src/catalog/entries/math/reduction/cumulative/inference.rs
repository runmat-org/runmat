use super::{CumulativeControl, CumulativeKind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, AliasFact, CallContract, CallInference, CallRequest, DynamicReason,
    InferenceDiagnostic, LiteralValue, NumericClass, NumericDomain, NumericFact, ResidencyFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(
    kind: CumulativeKind,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() || request.arguments.len() > 4 {
        diagnostics.push(InferenceDiagnostic::error(
            entry.descriptor.errors[0].code,
            "cumulative scans require one input and at most three controls",
        ));
    }
    controls(request, entry, &mut diagnostics);
    let output = match request.arguments.first() {
        Some(input) if input.storage != StorageFact::Sparse => {
            let numeric = match input.kind {
                ValueKindFact::Numeric(value)
                    if !(value.class.integer_class().is_some()
                        && value.domain == NumericDomain::Complex) =>
                {
                    Some(value)
                }
                ValueKindFact::Logical => Some(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                ValueKindFact::Unknown => None,
                _ => {
                    diagnostics.push(InferenceDiagnostic::error(entry.descriptor.errors[1].code,"cumulative input must be a dense numeric or logical array, not complex-integer storage"));
                    None
                }
            };
            match numeric {
                Some(numeric) => {
                    let mut result = input.clone();
                    result.kind = ValueKindFact::Numeric(numeric);
                    crate::catalog::inference::materialize(&mut result);
                    result.alias = AliasFact::Unknown;
                    if !matches!(input.residency, ResidencyFact::Host) {
                        result.residency = ResidencyFact::Unknown;
                    }
                    result
                }
                None => {
                    let mut result = ValueFact::unknown(DynamicReason::RuntimeValue);
                    result.shape = input.shape.clone();
                    result
                }
            }
        }
        Some(_) => {
            diagnostics.push(InferenceDiagnostic::error(
                entry.descriptor.errors[1].code,
                "sparse cumulative inputs are unsupported",
            ));
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let outputs = if kind.has_indices() {
        let mut indices = output.clone();
        indices.kind = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
        vec![output, indices]
    } else {
        vec![output]
    };
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(diagnostics);
    inferred
}

fn controls(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let (mut dimension, mut direction, mut missing) = (false, false, false);
    for index in 1..request.arguments.len() {
        let Some(literal) = request.literals.literal_args.get(index) else {
            continue;
        };
        let valid = match CumulativeControl::from_literal(literal) {
            Some(CumulativeControl::Direction(_)) => {
                let valid = !direction;
                direction = true;
                valid
            }
            Some(CumulativeControl::Missing(_)) => {
                let valid = !missing;
                missing = true;
                valid
            }
            None => match literal {
                LiteralValue::Empty => true,
                LiteralValue::Vector(values) if values.is_empty() => true,
                LiteralValue::Unknown | LiteralValue::Symbolic(_) => true,
                LiteralValue::Integer { .. } => {
                    let valid = !dimension
                        && literal
                            .integer_value_i128()
                            .is_some_and(|n| n > 0 && usize::try_from(n).is_ok());
                    dimension = true;
                    valid
                }
                LiteralValue::Number(_) | LiteralValue::Real { .. } => {
                    let valid = !dimension
                        && crate::ResolveContext::numeric_from_literal(literal)
                            .and_then(|v| runmat_types::decode_shape_dimension(v).ok())
                            .is_some_and(|n| n > 0);
                    dimension = true;
                    valid
                }
                _ => false,
            },
        };
        if !valid {
            diagnostics.push(InferenceDiagnostic::error(
                entry.descriptor.errors[0].code,
                "invalid or repeated cumulative dimension, direction, or missing-value control",
            ));
        }
    }
}
