use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, standard, AliasFact, CallContract, CallInference, CallRequest,
    CapabilityRequirement, CertaintyFact, ContiguityFact, DynamicReason, EffectKind, LayoutFact,
    MutationFact, NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact, ViewFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let plot = request.outputs.requested.known_count() == Some(0);
    let data_form = request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 0);
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() || (data_form && request.arguments.len() != 1) {
        diagnostics.push(argument_error(
            "RM-CATALOG-PZMAP-ARITY",
            "output forms require one model; plot form accepts models and styles",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|n| n > 2)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-PZMAP-OUTPUTS",
            "pzmap supports at most two outputs",
            0,
        ));
    }
    let mut poles = None;
    let mut zeros = None;
    if let Some(model) = request.arguments.first() {
        match &model.kind {
            ValueKindFact::Unknown => {}
            ValueKindFact::Object(object) => match object.runtime_class.as_ref() {
                Some(class) if class == &standard::TRANSFER_FUNCTION.owned() => {
                    if object.properties_complete
                        && [
                            "Numerator",
                            "Denominator",
                            "Ts",
                            "InputDelay",
                            "OutputDelay",
                        ]
                        .iter()
                        .any(|name| !object.properties.contains_key(*name))
                    {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-PZMAP-MODEL",
                            "tf model is missing required properties",
                            0,
                        ));
                    }
                    poles = object
                        .properties
                        .get("Denominator")
                        .and_then(|v| (v.shape.element_count() == Some(1)).then_some(0));
                    zeros = object
                        .properties
                        .get("Numerator")
                        .and_then(|v| (v.shape.element_count() == Some(1)).then_some(0));
                }
                Some(class) if class == &standard::STATE_SPACE.owned() => {
                    if object.properties_complete
                        && ["A", "B", "C", "D", "Ts", "InputDelay", "OutputDelay"]
                            .iter()
                            .any(|name| !object.properties.contains_key(*name))
                    {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-PZMAP-MODEL",
                            "ss model is missing required properties",
                            0,
                        ));
                    }
                    poles = object
                        .properties
                        .get("A")
                        .and_then(|a| a.shape.known_dims())
                        .and_then(|dims| match dims.as_slice() {
                            [Some(rows), Some(cols)] if rows == cols => Some(*rows),
                            _ => None,
                        });
                }
                None if object.class.is_some() => diagnostics.push(argument_error(
                    "RM-CATALOG-PZMAP-MODEL",
                    "sys must be a tf or ss object",
                    0,
                )),
                None => {}
                Some(_) => diagnostics.push(argument_error(
                    "RM-CATALOG-PZMAP-MODEL",
                    "sys must be a tf or ss object",
                    0,
                )),
            },
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-PZMAP-MODEL",
                "sys must be a tf or ss object",
                0,
            )),
        }
        if model.shape.element_count().is_some_and(|n| n != 1)
            || matches!(model.residency, ResidencyFact::Remote { .. })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-PZMAP-MODEL",
                "sys must be a scalar local model",
                0,
            ));
        }
    }
    if plot {
        let mut style_seen = false;
        for (index, arg) in request.arguments.iter().enumerate().skip(1) {
            match &arg.kind {
                ValueKindFact::String | ValueKindFact::Character => {
                    if style_seen {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-PZMAP-PLOT-STYLE",
                            "only one line style is supported per model",
                            index,
                        ));
                    }
                    style_seen = true;
                }
                ValueKindFact::Object(object) => {
                    if object.runtime_class.as_ref().is_some_and(|class| {
                        class != &standard::TRANSFER_FUNCTION.owned()
                            && class != &standard::STATE_SPACE.owned()
                    }) || (object.runtime_class.is_none() && object.class.is_some())
                    {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-PZMAP-PLOT-ARG",
                            "plot model must be tf or ss",
                            index,
                        ));
                    }
                    style_seen = false;
                }
                ValueKindFact::Unknown => style_seen = false,
                _ => diagnostics.push(argument_error(
                    "RM-CATALOG-PZMAP-PLOT-ARG",
                    "plot arguments must be models or line styles",
                    index,
                )),
            }
        }
    }
    let mut contract = CallContract::fixed(if plot {
        vec![]
    } else {
        vec![root_column(poles), root_column(zeros)]
    });
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|n| n > 0)
    {
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

fn root_column(count: Option<usize>) -> ValueFact {
    let mut result = ValueFact::unknown(DynamicReason::RuntimeValue);
    result.shape = ShapeFact::from(vec![count, Some(1)]);
    result.storage = StorageFact::Dense;
    result.layout = LayoutFact::ColumnMajor;
    result.contiguity = ContiguityFact::Contiguous;
    result.view = ViewFact::Materialized;
    result.residency = ResidencyFact::Host;
    result.alias = AliasFact::Unique;
    result.mutation = MutationFact::ValueSemantics;
    if count == Some(0) {
        result.kind = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
        result.certainty = CertaintyFact::Proven;
    }
    result
}
