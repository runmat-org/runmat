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
    let args = &request.arguments;
    let mut diagnostics = Vec::new();
    if args.is_empty() || (data_form && !(1..=2).contains(&args.len())) {
        diagnostics.push(argument_error(
            "RM-CATALOG-RLOCUS-ARITY",
            "output forms take a model and optional gain vector",
            args.len().saturating_sub(1),
        ));
    }
    let mut branches = None;
    if let Some(model) = args.first() {
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
                            "RM-CATALOG-RLOCUS-MODEL",
                            "tf model is missing required properties",
                            0,
                        ));
                    }
                    let den = object
                        .properties
                        .get("Denominator")
                        .and_then(|v| v.shape.element_count());
                    let num = object
                        .properties
                        .get("Numerator")
                        .and_then(|v| v.shape.element_count());
                    if den == Some(1) && num == Some(1) {
                        branches = Some(0);
                    }
                }
                Some(class) if class == &standard::STATE_SPACE.owned() => {
                    if object.properties_complete
                        && ["A", "B", "C", "D", "Ts", "InputDelay", "OutputDelay"]
                            .iter()
                            .any(|name| !object.properties.contains_key(*name))
                    {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-RLOCUS-MODEL",
                            "ss model is missing required properties",
                            0,
                        ));
                    }
                    // Polynomial degree may drop at runtime; A shape does not prove branch count.
                }
                None if object.class.is_some() => diagnostics.push(argument_error(
                    "RM-CATALOG-RLOCUS-MODEL",
                    "sys must be a tf or ss object",
                    0,
                )),
                None => {}
                Some(_) => diagnostics.push(argument_error(
                    "RM-CATALOG-RLOCUS-MODEL",
                    "sys must be a tf or ss object",
                    0,
                )),
            },
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-RLOCUS-MODEL",
                "sys must be a tf or ss object",
                0,
            )),
        }
        if model.shape.element_count().is_some_and(|n| n != 1)
            || matches!(model.residency, ResidencyFact::Remote { .. })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-RLOCUS-MODEL",
                "sys must be a scalar local model",
                0,
            ));
        }
    }
    let mut gains = None;
    if let Some(k) = args.get(1) {
        if !plot {
            let numeric = matches!(
                k.kind,
                ValueKindFact::Unknown | ValueKindFact::Numeric(_) | ValueKindFact::Logical
            );
            if !numeric {
                diagnostics.push(argument_error(
                    "RM-CATALOG-RLOCUS-GAIN",
                    "k must be a numeric vector",
                    1,
                ));
            }
            if let Some(dims) = k.shape.known_dims() {
                if dims.iter().filter(|dim| dim.is_some_and(|n| n > 1)).count() > 1 {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-RLOCUS-GAIN",
                        "k must be a vector",
                        1,
                    ));
                } else {
                    gains = k.shape.element_count();
                }
            }
            if gains == Some(0) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-RLOCUS-GAIN",
                    "k must not be empty",
                    1,
                ));
            }
        }
    }
    if plot {
        let mut style_seen = false;
        let mut gain_seen = false;
        for (index, arg) in args.iter().enumerate().skip(1) {
            match &arg.kind {
                ValueKindFact::String | ValueKindFact::Character => {
                    if style_seen {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-RLOCUS-PLOT-STYLE",
                            "only one style is supported per model",
                            index,
                        ));
                    }
                    style_seen = true;
                }
                ValueKindFact::Object(object) => {
                    if gain_seen
                        || object.runtime_class.as_ref().is_some_and(|class| {
                            class != &standard::TRANSFER_FUNCTION.owned()
                                && class != &standard::STATE_SPACE.owned()
                        })
                        || (object.runtime_class.is_none() && object.class.is_some())
                    {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-RLOCUS-PLOT-MODEL",
                            "tf or ss models must precede the gain vector",
                            index,
                        ));
                    }
                    style_seen = false;
                }
                ValueKindFact::Numeric(_) | ValueKindFact::Logical => {
                    if gain_seen
                        || arg.shape.known_dims().is_some_and(|dims| {
                            dims.iter().filter(|dim| dim.is_some_and(|n| n > 1)).count() > 1
                        })
                        || arg.shape.element_count() == Some(0)
                    {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-RLOCUS-PLOT-GAIN",
                            "only one nonempty gain vector is supported after models",
                            index,
                        ));
                    }
                    gain_seen = true;
                }
                ValueKindFact::Unknown => {}
                _ => diagnostics.push(argument_error(
                    "RM-CATALOG-RLOCUS-PLOT-ARG",
                    "unsupported root-locus plot argument",
                    index,
                )),
            }
        }
    }
    let output = roots_matrix(branches, gains);
    let gain_output = real_double(ShapeFact::from(vec![Some(1), gains]));
    let mut contract = CallContract::fixed(if plot {
        vec![]
    } else {
        vec![output, gain_output]
    });
    if !plot {
        // The runtime pads extra requested slots with numeric zero.
        contract.maximum_outputs = None;
        contract.variadic_output = Some(Box::new(real_double(ShapeFact::Scalar)));
    }
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

fn roots_matrix(branches: Option<usize>, gains: Option<usize>) -> ValueFact {
    let mut result = ValueFact::unknown(DynamicReason::RuntimeValue);
    result.shape = ShapeFact::from(vec![branches, gains]);
    result.storage = StorageFact::Dense;
    result.layout = LayoutFact::ColumnMajor;
    result.contiguity = ContiguityFact::Contiguous;
    result.view = ViewFact::Materialized;
    result.residency = ResidencyFact::Host;
    result.alias = AliasFact::Unique;
    result.mutation = MutationFact::ValueSemantics;
    if branches == Some(0) {
        result.kind = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
        result.certainty = CertaintyFact::Proven;
    }
    result
}

fn real_double(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}
