use super::{PwelchControlCursor, PwelchControlRole, PwelchKeyword};
use crate::catalog::inference::{argument_error, literal_text};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

fn known_keyword(literal: &LiteralValue) -> Option<PwelchKeyword> {
    literal_text(literal).map(|text| PwelchKeyword::decode(&text))
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let invalid_arity = request.arguments.is_empty() || request.arguments.len() > 10;
    if invalid_arity {
        diagnostics.push(argument_error(
            "RM-CATALOG-PWELCH-ARITY",
            "pwelch expects a signal, at most seven positional controls, and an optional Parent pair",
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
            if dims.len() > 2 || dims.contains(&Some(0)) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-PWELCH-SHAPE",
                    "pwelch expects a nonempty vector or 2-D matrix",
                    0,
                ));
            }
        }
    }
    let complex_signal = signal.is_some_and(|fact| {
        matches!(
            &fact.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Complex,
                ..
            })
        )
    });
    if complex_signal {
        if let Some((index, _)) = request
            .literals
            .literal_args
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, literal)| known_keyword(literal) == Some(PwelchKeyword::Onesided))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-PWELCH-RANGE",
                "one-sided Welch spectra require a real signal",
                index,
            ));
        }
    }
    let mut parent_seen = false;
    for (index, literal) in request.literals.literal_args.iter().enumerate().skip(1) {
        if known_keyword(literal) != Some(PwelchKeyword::Parent) {
            continue;
        }
        if parent_seen || index + 1 >= request.arguments.len() {
            diagnostics.push(argument_error(
                "RM-CATALOG-PWELCH-PARENT",
                "Parent requires one existing axes handle",
                index,
            ));
            continue;
        }
        parent_seen = true;
        let target = &request.arguments[index + 1];
        let definitely_invalid = match &target.kind {
            ValueKindFact::Numeric(_) => match &target.shape {
                ShapeFact::Shaped { dims } => dims
                    .iter()
                    .any(|dim| matches!(dim, DimensionFact::Known(size) if *size != 1)),
                ShapeFact::Unknown | ShapeFact::Scalar | ShapeFact::Ranked { .. } => false,
            },
            ValueKindFact::Unknown => false,
            _ => true,
        };
        let definitely_root = request
            .literals
            .literal_args
            .get(index + 1)
            .and_then(LiteralContext::numeric_from_literal)
            .is_some_and(|handle| handle == 0.0);
        if definitely_invalid || definitely_root {
            diagnostics.push(argument_error(
                "RM-CATALOG-PWELCH-PARENT",
                "Parent must be an axes handle scalar",
                index + 1,
            ));
        }
    }
    let mut positional_count = 0usize;
    let mut unknown_parent_name = false;
    let mut cursor = PwelchControlCursor::default();
    let mut index = 1usize;
    while index < request.arguments.len() {
        let literal = request
            .literals
            .literal_args
            .get(index)
            .and_then(known_keyword);
        if literal == Some(PwelchKeyword::Parent) {
            index += if index + 1 < request.arguments.len() {
                2
            } else {
                1
            };
            continue;
        }
        if literal.is_none()
            && matches!(
                request.arguments[index].kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            )
        {
            unknown_parent_name = true;
        }
        let role = cursor.next(literal.as_ref());
        if !unknown_parent_name
            && role == PwelchControlRole::Window
            && matches!(literal.as_ref(), Some(PwelchKeyword::Unknown(_)))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-PWELCH-WINDOW",
                "pwelch window position does not accept this option",
                index,
            ));
        }
        if !unknown_parent_name
            && role == PwelchControlRole::Option
            && (matches!(literal.as_ref(), Some(PwelchKeyword::Unknown(_)))
                || (literal.is_none()
                    && matches!(
                        request.arguments[index].kind,
                        ValueKindFact::Numeric(_) | ValueKindFact::Logical
                    )))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-PWELCH-RANGE",
                "pwelch selector tail requires a range, scale, or trace keyword",
                index,
            ));
        }
        positional_count += 1;
        index += 1;
    }
    if positional_count > 7 && !unknown_parent_name {
        diagnostics.push(argument_error(
            "RM-CATALOG-PWELCH-ARITY",
            "pwelch accepts no more than seven positional controls",
            8,
        ));
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
    let outputs = (0..2)
        .map(|slot| {
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
    // Surplus runtime padding is not a computed spectral result or a
    // confidence interval. Do not assign it numerical result facts.
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(ValueFact::unknown(
        DynamicReason::UnsupportedRepresentation,
    )));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let parent_possible = request
        .arguments
        .iter()
        .enumerate()
        .skip(1)
        .any(|(index, fact)| match &fact.kind {
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown => request
                .literals
                .literal_args
                .get(index)
                .and_then(known_keyword)
                .is_none_or(|name| name == PwelchKeyword::Parent),
            _ => false,
        });
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 0)
        && !parent_possible
    {
        contract
            .effects
            .0
            .remove(&runmat_types::EffectKind::UserInterface);
        contract
            .effects
            .0
            .remove(&runmat_types::EffectKind::HostCallback);
        contract
            .capabilities
            .0
            .remove(&runmat_types::CapabilityRequirement::UserInterface);
    }
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}
