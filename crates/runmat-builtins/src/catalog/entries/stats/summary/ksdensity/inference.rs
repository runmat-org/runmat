use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let double = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    });
    let second = request.arguments.get(1);
    let explicit = second.filter(|value| {
        matches!(value.kind, ValueKindFact::Numeric(_))
            && value.shape.element_count().is_some_and(|count| count > 0)
    });
    let start = if second.is_some_and(|value| matches!(value.kind, ValueKindFact::Numeric(_))) {
        2
    } else {
        1
    };
    let known_no_points = second.is_none()
        || second.is_some_and(|value| {
            matches!(value.kind, ValueKindFact::Character | ValueKindFact::String)
                || value.shape.element_count() == Some(0)
        });
    let mut point_count = Some(100usize);
    if known_no_points
        && request.arguments.len() >= start
        && (request.arguments.len() - start).is_multiple_of(2)
    {
        for index in (start..request.arguments.len()).step_by(2) {
            let name = request
                .literals
                .literal_string_at(index)
                .and_then(|value| super::KsdensityOptionName::from_public(&value));
            match name {
                Some(super::KsdensityOptionName::NumPoints) => {
                    point_count = request
                        .literals
                        .literal_args
                        .get(index + 1)
                        .and_then(LiteralContext::numeric_from_literal)
                        .filter(|value| {
                            value.is_finite()
                                && *value >= 1.0
                                && *value < 2f64.powi(usize::BITS as i32)
                                && value.fract() == 0.0
                        })
                        .map(|value| value as usize)
                }
                None => point_count = None,
                _ => {}
            }
        }
    } else if !known_no_points {
        point_count = None;
    }
    let shape = if let Some(points) = explicit {
        points.shape.clone()
    } else if known_no_points {
        ShapeFact::from(vec![Some(1), point_count])
    } else {
        ShapeFact::Unknown
    };
    let xi_kind = if let Some(points) = explicit {
        points.kind.clone()
    } else if known_no_points
        || second.is_some_and(|value| {
            matches!(
                value.kind,
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real
                })
            )
        })
    {
        double.clone()
    } else {
        ValueKindFact::Unknown
    };
    let f = ValueFact::proven(double.clone(), shape.clone(), StorageFact::Dense);
    let xi = ValueFact::proven(xi_kind, shape, StorageFact::Dense);
    let bw = ValueFact::scalar(double);
    let mut contract = CallContract::fixed(vec![f, xi, bw]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 0)
    {
        result.effects.0.remove(&EffectKind::UserInterface);
        result.effects.0.remove(&EffectKind::HostCallback);
        result
            .capabilities
            .0
            .remove(&CapabilityRequirement::UserInterface);
    }
    if request.arguments.is_empty() {
        result.diagnostics.push(InferenceDiagnostic::error(
            "ARITY",
            "ksdensity requires univariate sample data",
        ));
    }
    result
}
