use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, standard, CallContract, CallInference, CallRequest, CapabilityRequirement,
    EffectKind, InferenceDiagnostic, NumericClass, NumericDomain, NumericFact, ResidencyFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[derive(Clone, Copy)]
enum ResponseKind {
    Step,
    Impulse,
}

impl ResponseKind {
    fn name(self) -> &'static str {
        match self {
            Self::Step => "step",
            Self::Impulse => "impulse",
        }
    }

    fn default_count(self) -> usize {
        match self {
            Self::Step => 101,
            Self::Impulse => 100,
        }
    }
}

pub(super) fn infer_step(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, ResponseKind::Step)
}

pub(super) fn infer_impulse(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, ResponseKind::Impulse)
}

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry, kind: ResponseKind) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.outputs.requested.known_count();
    let plot = count == Some(0);
    let name = kind.name();
    if request.arguments.is_empty()
        || (!plot && request.arguments.len() > 2)
        || (matches!(kind, ResponseKind::Impulse) && request.arguments.len() > 2)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-RESPONSE-ARITY",
            format!("{name} requires a model and accepts at most one time input in output forms"),
            request.arguments.len().saturating_sub(1),
        ));
    }
    if count.is_some_and(|count| count > 2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-RESPONSE-OUTPUTS",
            format!("{name} supports at most two outputs"),
            0,
        ));
    }
    let model_indices = if plot && matches!(kind, ResponseKind::Step) {
        let mut indices = vec![0];
        let mut style_seen = false;
        let mut time_seen = false;
        for (index, arg) in request.arguments.iter().enumerate().skip(1) {
            match &arg.kind {
                ValueKindFact::String | ValueKindFact::Character if !time_seen => {
                    if style_seen {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-STEP-STYLE",
                            "one style per model is supported",
                            index,
                        ));
                    }
                    style_seen = true;
                }
                ValueKindFact::Object(_) if !time_seen => {
                    indices.push(index);
                    style_seen = false;
                }
                ValueKindFact::Unknown if !time_seen => {
                    // The runtime value may be another model, a style or the
                    // final time. No later argument order is proven invalid.
                    style_seen = false;
                }
                ValueKindFact::Numeric(_) | ValueKindFact::Logical if !time_seen => {
                    check_time(arg, index, kind, &mut diagnostics);
                    time_seen = true;
                }
                _ => diagnostics.push(argument_error(
                    "RM-CATALOG-STEP-PLOT-ARG",
                    "time must follow every model and style",
                    index,
                )),
            }
        }
        indices
    } else {
        if let Some(time) = request.arguments.get(1) {
            check_time(time, 1, kind, &mut diagnostics);
        }
        vec![0]
    };
    for index in model_indices {
        if let Some(model) = request.arguments.get(index) {
            let supported = match &model.kind {
                ValueKindFact::Unknown => true,
                ValueKindFact::Object(object) => {
                    let identity = match object.runtime_class.as_ref() {
                        Some(class) => class == &standard::TRANSFER_FUNCTION.owned(),
                        None => object.class.is_none(),
                    };
                    identity
                        && (!object.properties_complete
                            || [
                                "Numerator",
                                "Denominator",
                                "Ts",
                                "InputDelay",
                                "OutputDelay",
                            ]
                            .iter()
                            .all(|key| object.properties.contains_key(*key)))
                }
                _ => false,
            };
            if !supported
                || model.shape.element_count().is_some_and(|n| n != 1)
                || matches!(model.residency, ResidencyFact::Remote { .. })
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-RESPONSE-MODEL",
                    format!("{name} requires a scalar tf model with required properties"),
                    index,
                ));
            }
        }
    }
    let sample_count = if request.arguments.len() == 1 {
        Some(kind.default_count())
    } else {
        request.arguments.get(1).and_then(|arg| {
            let n = arg.shape.element_count()?;
            (n > 1).then_some(n)
        })
    };
    let output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![sample_count, Some(1)]),
        StorageFact::Dense,
    );
    let mut contract = CallContract::fixed(if plot {
        vec![]
    } else {
        vec![output.clone(), output]
    });
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    if count.is_some_and(|n| n > 0) {
        contract.effects.0.remove(&EffectKind::UserInterface);
        contract
            .capabilities
            .0
            .remove(&CapabilityRequirement::UserInterface);
    }
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn check_time(
    time: &ValueFact,
    index: usize,
    kind: ResponseKind,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let supported = matches!(
        time.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Unknown
    ) || (matches!(kind, ResponseKind::Impulse)
        && matches!(time.kind, ValueKindFact::Logical));
    if !supported
        || time
            .shape
            .known_dims()
            .is_some_and(|dims| dims.iter().filter(|dim| dim.is_some_and(|n| n > 1)).count() > 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-RESPONSE-TIME",
            "time must be a real numeric scalar or vector",
            index,
        ));
    }
}
