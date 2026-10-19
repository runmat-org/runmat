use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, standard, AliasFact, CallContract, CallInference, CallRequest, CertaintyFact,
    ContiguityFact, DynamicReason, LayoutFact, MutationFact, NumericClass, NumericDomain,
    NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact, ViewFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-DAMP-ARITY",
            "damp expects exactly one dynamic-system model",
            request.arguments.len().saturating_sub(1),
        ));
    }

    let mut mode_count = None;
    if let Some(model) = request.arguments.first() {
        let supported = match &model.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => match object.runtime_class.as_ref() {
                Some(class) if class == &standard::STATE_SPACE.owned() => {
                    mode_count = state_space_mode_count(object);
                    true
                }
                Some(class) if class == &standard::TRANSFER_FUNCTION.owned() => {
                    mode_count = static_gain_mode_count(object);
                    true
                }
                None => object.class.is_none(),
                Some(_) => false,
            },
            _ => false,
        };
        if !supported
            || model.shape.element_count().is_some_and(|count| count != 1)
            || matches!(model.residency, ResidencyFact::Remote { .. })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-DAMP-MODEL",
                "sys must be a scalar tf or ss model object",
                0,
            ));
        }
    }

    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 3)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-DAMP-OUTPUTS",
            "damp supports at most three outputs",
            0,
        ));
    }

    let shape = ShapeFact::from(vec![mode_count, Some(1)]);
    let mut poles = ValueFact::unknown(DynamicReason::RuntimeValue);
    poles.shape = shape.clone();
    poles.storage = StorageFact::Dense;
    poles.layout = LayoutFact::ColumnMajor;
    poles.contiguity = ContiguityFact::Contiguous;
    poles.view = ViewFact::Materialized;
    poles.residency = ResidencyFact::Host;
    poles.alias = AliasFact::Unique;
    poles.mutation = MutationFact::ValueSemantics;
    if mode_count == Some(0) {
        // With no poles the runtime materializes an empty real double column.
        poles.kind = real_double();
        poles.certainty = CertaintyFact::Proven;
    }
    let outputs = vec![
        ValueFact::proven(real_double(), shape.clone(), StorageFact::Dense),
        ValueFact::proven(real_double(), shape, StorageFact::Dense),
        poles,
    ];
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn real_double() -> ValueKindFact {
    ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    })
}

fn state_space_mode_count(object: &runmat_types::ObjectFact) -> Option<usize> {
    let dimensions = object.properties.get("A")?.shape.known_dims()?;
    match dimensions.as_slice() {
        [Some(rows), Some(columns)] if rows == columns => Some(*rows),
        _ => None,
    }
}

fn static_gain_mode_count(object: &runmat_types::ObjectFact) -> Option<usize> {
    // Arbitrary denominator shapes do not prove polynomial degree: leading
    // zeros can be trimmed when a mutable tf object is analyzed.
    (object.properties.get("Denominator")?.shape.element_count() == Some(1)).then_some(0)
}
