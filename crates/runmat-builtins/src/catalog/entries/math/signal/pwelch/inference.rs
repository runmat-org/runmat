use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let invalid_arity = request.arguments.is_empty() || request.arguments.len() > 8;
    if invalid_arity {
        diagnostics.push(argument_error(
            "RM-CATALOG-PWELCH-ARITY",
            "pwelch expects a signal and at most seven controls",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let signal = request.arguments.first();
    let class = match signal.map(|fact| &fact.kind) {
        Some(ValueKindFact::Numeric(n)) if n.class == NumericClass::Single => {
            Some(NumericClass::Single)
        }
        Some(ValueKindFact::Numeric(n))
            if n.class == NumericClass::Double || n.domain == NumericDomain::Real =>
        {
            Some(NumericClass::Double)
        }
        Some(ValueKindFact::Logical) => Some(NumericClass::Double),
        Some(ValueKindFact::Unknown) | None => None,
        _ => {
            diagnostics.push(argument_error("RM-CATALOG-PWELCH-SIGNAL", "pwelch expects a real or complex floating signal, or an admitted real integer/logical signal", 0));
            None
        }
    };
    if let Some(fact) = signal {
        if let Some(dims) = fact.shape.known_dims() {
            if dims.len() > 2 || dims.iter().any(|dim| *dim == Some(0)) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-PWELCH-SHAPE",
                    "pwelch expects a nonempty vector or 2-D matrix",
                    0,
                ));
            }
        }
    }
    for (index, control) in request.arguments.iter().enumerate().skip(1) {
        let invalid_kind = matches!(
            control.kind,
            ValueKindFact::Cell(_)
                | ValueKindFact::Struct(_)
                | ValueKindFact::Object(_)
                | ValueKindFact::Callable(_)
                | ValueKindFact::ClassReference(_)
                | ValueKindFact::Foreign(_)
        );
        let negative_scalar = request
            .literals
            .literal_args
            .get(index)
            .is_some_and(|literal| {
                LiteralContext::numeric_from_literal(literal)
                    .is_some_and(|value| !value.is_finite() || value < 0.0)
            });
        if invalid_kind || negative_scalar {
            diagnostics.push(argument_error(
                "RM-CATALOG-PWELCH-CONTROL",
                "pwelch control is not an admitted numeric value or option",
                index,
            ));
        }
    }
    let count = request.outputs.requested.known_count().unwrap_or(1);
    let outputs = (0..count)
        .map(|slot| {
            if slot >= 2 {
                return ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }));
            }
            let Some(class) = class.filter(|_| diagnostics.is_empty()) else {
                return ValueFact::unknown(DynamicReason::RuntimeValue);
            };
            let columns =
                signal
                    .and_then(|fact| fact.shape.known_dims())
                    .and_then(|dims| match dims.as_slice() {
                        [Some(1), _] | [_, Some(1)] => Some(1),
                        [_, Some(cols)] => Some(*cols),
                        _ => None,
                    });
            let shape = ShapeFact::from(vec![None, if slot == 0 { columns } else { Some(1) }]);
            let mut output = ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class,
                    domain: NumericDomain::Real,
                }),
                shape,
                StorageFact::Dense,
            );
            if !signal.is_some_and(|input| input.residency == ResidencyFact::Host) {
                output.residency = ResidencyFact::Unknown;
                output.layout = LayoutFact::Unknown;
                output.contiguity = ContiguityFact::Unknown;
                output.view = ViewFact::Unknown;
                output.alias = AliasFact::Unknown;
            }
            output
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
