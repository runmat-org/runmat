use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let arity = request.arguments.len();
    if !(2..=3).contains(&arity) {
        diagnostics.push(argument_error(
            "RM-CATALOG-POL2CART-ARITY",
            "pol2cart expects two or three inputs",
            arity.saturating_sub(1),
        ));
    }
    let count = request.outputs.requested.known_count().unwrap_or(1);
    if count > arity.min(3) {
        diagnostics.push(argument_error(
            "RM-CATALOG-POL2CART-OUTPUTS",
            "requested outputs exceed the selected pol2cart form",
            0,
        ));
    }
    let mut shape = ShapeFact::from(vec![Some(1), Some(1)]);
    for (index, input) in request.arguments.iter().enumerate() {
        if input.storage == StorageFact::Sparse
            || !matches!(
                input.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Unknown
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-POL2CART-INPUT",
                "pol2cart requires real dense numeric inputs",
                index,
            ));
        }
        shape = match broadcast_shape(&shape, &input.shape) {
            Ok(shape) => shape,
            Err(error) => {
                diagnostics.push(error);
                ShapeFact::Unknown
            }
        };
    }
    let pair = match (
        request.arguments.first().and_then(ValueFact::numeric),
        request.arguments.get(1).and_then(ValueFact::numeric),
    ) {
        (Some(a), Some(b)) => Some(
            if a.class == NumericClass::Single || b.class == NumericClass::Single {
                NumericClass::Single
            } else {
                NumericClass::Double
            },
        ),
        _ => None,
    };
    let all_host = request
        .arguments
        .iter()
        .all(|input| input.residency == ResidencyFact::Host);
    let may_prepare = [NumericClass::Double, NumericClass::Single].iter().any(|class| {
        request.arguments.iter().all(|input| {
            matches!(input.kind, ValueKindFact::Unknown)
                || matches!(input.kind, ValueKindFact::Numeric(numeric) if numeric.class == *class && numeric.domain == NumericDomain::Real)
        })
    });
    let outputs = (0..count)
        .map(|index| {
            let class = if index < 2 {
                pair
            } else if index == 2 {
                request
                    .arguments
                    .get(2)
                    .and_then(ValueFact::numeric)
                    .map(|n| {
                        if n.class == NumericClass::Single {
                            NumericClass::Single
                        } else {
                            NumericClass::Double
                        }
                    })
            } else {
                None
            };
            let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
            if let Some(class) = class {
                fact.kind = ValueKindFact::Numeric(NumericFact {
                    class,
                    domain: NumericDomain::Real,
                });
            }
            fact.shape = shape.clone();
            crate::catalog::inference::materialize(&mut fact);
            fact.residency = if all_host && !may_prepare {
                ResidencyFact::Host
            } else {
                ResidencyFact::Unknown
            };
            fact
        })
        .collect();
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}
