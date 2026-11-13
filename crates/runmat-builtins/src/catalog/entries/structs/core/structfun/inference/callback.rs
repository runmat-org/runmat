use crate::catalog::inference::gathered_value_fact;
use runmat_types::FactJoin;
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
    let callable = request
        .arguments
        .first()
        .and_then(|value| match &value.kind {
            ValueKindFact::Callable(callable) => Some(callable),
            _ => None,
        });

    for (_, outputs) in &mut result.fields {
        let field = outputs
            .pop()
            .expect("field discovery produces one input fact per field");
        let field = gathered_value_fact(&field);
        if let Some(callable) = callable {
            let mut inferred = crate::infer_callable_call(
                callable,
                &CallRequest {
                    arguments: vec![field.clone()],
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
        } else {
            *outputs = vec![ValueFact::unknown(DynamicReason::UnresolvedCallable); count];
        }
        for index in (2..request.arguments.len()).step_by(2) {
            let handler = request
                .literals
                .literal_args
                .get(index)
                .and_then(crate::catalog::inference::literal_text)
                .is_some_and(|name| name.trim().eq_ignore_ascii_case("ErrorHandler"));
            if !handler {
                continue;
            }
            let Some(ValueKindFact::Callable(handler)) =
                request.arguments.get(index + 1).map(|v| &v.kind)
            else {
                *outputs = vec![ValueFact::unknown(DynamicReason::UnresolvedCallable); count];
                continue;
            };
            let recovered = crate::infer_callable_call(
                handler,
                &CallRequest {
                    arguments: vec![
                        ValueFact::unknown(DynamicReason::RuntimeValue),
                        field.clone(),
                    ],
                    literals: LiteralContext::default(),
                    outputs: OutputSelection::new(requested_outputs(count)),
                },
            );
            result.effects.0.extend(recovered.effects.0);
            result.capabilities.0.extend(recovered.capabilities.0);
            result.diagnostics.extend(recovered.diagnostics);
            for (output, recovery) in outputs.iter_mut().zip(recovered.outputs) {
                *output = output.join(&recovery);
            }
        }
        for output in outputs {
            *output = gathered_value_fact(output);
        }
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

fn requested_outputs(count: usize) -> RequestedOutputCount {
    if count == 1 {
        RequestedOutputCount::One
    } else {
        RequestedOutputCount::Exactly(count)
    }
}
