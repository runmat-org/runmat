use runmat_types::{
    CallRequest, CapabilitySet, DynamicReason, EffectSet, InferenceDiagnostic, LiteralContext,
    OutputSelection, RequestedOutputCount, StructFact, ValueFact, ValueKindFact,
};

pub(super) type FieldOutputs = Vec<(String, Vec<ValueFact>)>;

pub(super) struct InferredFields {
    pub fields: FieldOutputs,
    pub effects: EffectSet,
    pub capabilities: CapabilitySet,
    pub diagnostics: Vec<InferenceDiagnostic>,
}

pub(super) fn infer(request: &CallRequest, count: usize) -> InferredFields {
    let mut result = InferredFields {
        fields: fields(request),
        effects: EffectSet::default(),
        capabilities: CapabilitySet::default(),
        diagnostics: Vec::new(),
    };
    let Some(callable) = request
        .arguments
        .first()
        .and_then(|value| match &value.kind {
            ValueKindFact::Callable(callable) => Some(callable),
            _ => None,
        })
    else {
        result.fields = unresolved_outputs(result.fields, count);
        return result;
    };

    for (_, outputs) in &mut result.fields {
        let field = outputs
            .pop()
            .expect("field discovery produces one input fact per field");
        let mut inferred = crate::infer_callable_call(
            callable,
            &CallRequest {
                arguments: vec![field],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(requested_outputs(count)),
            },
        );
        result.effects.0.extend(inferred.effects.0);
        result.capabilities.0.extend(inferred.capabilities.0);
        for diagnostic in &mut inferred.diagnostics {
            diagnostic.argument = Some(1);
        }
        result.diagnostics.append(&mut inferred.diagnostics);
        *outputs = inferred.outputs;
    }
    result
}

fn fields(request: &CallRequest) -> FieldOutputs {
    match request.arguments.get(1).map(|value| &value.kind) {
        Some(ValueKindFact::Struct(StructFact {
            fields,
            fields_complete,
            ..
        })) => {
            let mut discovered = fields
                .iter()
                .map(|(name, value)| (name.clone(), vec![value.clone()]))
                .collect::<FieldOutputs>();
            if !fields_complete {
                discovered.push((
                    String::new(),
                    vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
                ));
            }
            discovered
        }
        _ => vec![(
            String::new(),
            vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        )],
    }
}

fn unresolved_outputs(fields: FieldOutputs, count: usize) -> FieldOutputs {
    fields
        .into_iter()
        .map(|(name, _)| {
            (
                name,
                vec![ValueFact::unknown(DynamicReason::UnresolvedCallable); count],
            )
        })
        .collect()
}

fn requested_outputs(count: usize) -> RequestedOutputCount {
    if count == 1 {
        RequestedOutputCount::One
    } else {
        RequestedOutputCount::Exactly(count)
    }
}
