use crate::{
    DynamicReason, FactInference, FactJoin, InferenceDiagnostic, MemberName, OutputListFact,
    ValueFact, ValueKindFact,
};

pub fn infer_member_read(
    base: &ValueFact,
    member: &MemberName,
    sequence_use: crate::SequenceUse,
) -> FactInference {
    match &base.kind {
        ValueKindFact::Struct(structure) => {
            let field = structure.fields.get(&member.0).cloned();
            if field.is_none() && structure.fields_complete {
                return missing_member(member);
            }
            if base.is_scalar() {
                return resolve_sequence_fact(
                    vec![field.unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue))],
                    false,
                    sequence_use,
                );
            }
            let (outputs, variadic) = if structure.elements_complete {
                let outputs = structure
                    .elements
                    .iter()
                    .map(|element| {
                        element
                            .get(&member.0)
                            .cloned()
                            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue))
                    })
                    .collect();
                (outputs, false)
            } else {
                (field.into_iter().collect(), true)
            };
            resolve_sequence_fact(outputs, variadic, sequence_use)
        }
        ValueKindFact::Object(object) => object.properties.get(&member.0).cloned().map_or_else(
            || {
                if object.properties_complete {
                    missing_member(member)
                } else {
                    FactInference::exact(ValueFact::unknown(DynamicReason::DynamicDispatch))
                }
            },
            FactInference::exact,
        ),
        ValueKindFact::Unknown => {
            FactInference::exact(ValueFact::unknown(DynamicReason::DynamicDispatch))
        }
        _ => FactInference {
            fact: ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics: vec![InferenceDiagnostic::error(
                "RM-TYPE-MEMBER-READ",
                "member access requires a struct or object value",
            )],
        },
    }
}

fn resolve_sequence_fact(
    mut outputs: Vec<ValueFact>,
    variadic: bool,
    sequence_use: crate::SequenceUse,
) -> FactInference {
    match sequence_use {
        crate::SequenceUse::Discard => FactInference::exact(ValueFact::scalar(
            ValueKindFact::OutputList(OutputListFact {
                outputs: Vec::new(),
                variadic: false,
            }),
        )),
        crate::SequenceUse::RequireSingle => {
            if !variadic && outputs.len() != 1 {
                return sequence_arity_error(outputs.len(), 1, true);
            }
            if outputs.len() == 1 {
                FactInference::exact(outputs.remove(0))
            } else {
                FactInference::exact(ValueFact::unknown(DynamicReason::RuntimeValue))
            }
        }
        crate::SequenceUse::SelectPrefix { count } => {
            if !variadic && outputs.len() < count {
                return sequence_arity_error(outputs.len(), count, false);
            }
            outputs.truncate(count);
            FactInference::exact(ValueFact::scalar(ValueKindFact::OutputList(
                OutputListFact {
                    outputs,
                    variadic: variadic && count > 0,
                },
            )))
        }
        crate::SequenceUse::ExpandAll
        | crate::SequenceUse::SelectCurrentFunctionOutputs
        | crate::SequenceUse::SelectDestinationCardinality => {
            FactInference::exact(ValueFact::scalar(ValueKindFact::OutputList(
                OutputListFact { outputs, variadic },
            )))
        }
    }
}

fn sequence_arity_error(actual: usize, requested: usize, exact: bool) -> FactInference {
    let requirement = if exact { "exactly" } else { "at least" };
    FactInference {
        fact: ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
        diagnostics: vec![InferenceDiagnostic::error(
            "RM-TYPE-COMMA-LIST-ARITY",
            format!(
                "value sequence contains {actual} values; this context requires {requirement} {requested}"
            ),
        )],
    }
}

pub fn infer_member_write(
    base: &ValueFact,
    member: &MemberName,
    assigned: &ValueFact,
    allow_creation: bool,
) -> FactInference {
    let mut output = base.clone();
    match &mut output.kind {
        ValueKindFact::Struct(structure) => {
            if structure.fields_complete
                && !allow_creation
                && !structure.fields.contains_key(&member.0)
            {
                return missing_member(member);
            }
            if base.is_scalar() {
                structure.fields.insert(member.0.clone(), assigned.clone());
                if structure.elements_complete {
                    for element in &mut structure.elements {
                        element.insert(member.0.clone(), assigned.clone());
                    }
                }
            } else if let ValueKindFact::OutputList(list) = &assigned.kind {
                let conservative = list
                    .outputs
                    .iter()
                    .cloned()
                    .reduce(|left, right| left.join(&right))
                    .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
                structure.fields.insert(member.0.clone(), conservative);
                if structure.elements_complete
                    && !list.variadic
                    && list.outputs.len() == structure.elements.len()
                {
                    for (element, value) in structure.elements.iter_mut().zip(&list.outputs) {
                        element.insert(member.0.clone(), value.clone());
                    }
                } else {
                    structure.elements.clear();
                    structure.elements_complete = false;
                }
            } else {
                structure.fields.insert(
                    member.0.clone(),
                    ValueFact::unknown(DynamicReason::RuntimeValue),
                );
                structure.elements.clear();
                structure.elements_complete = false;
            }
        }
        ValueKindFact::Object(object) => {
            if object.properties_complete
                && !allow_creation
                && !object.properties.contains_key(&member.0)
            {
                return missing_member(member);
            }
            object.properties.insert(member.0.clone(), assigned.clone());
        }
        _ => {
            return FactInference {
                fact: ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
                diagnostics: vec![InferenceDiagnostic::error(
                    "RM-TYPE-MEMBER-WRITE",
                    "member assignment requires a struct or object value",
                )],
            }
        }
    }
    FactInference::exact(output)
}

fn missing_member(member: &MemberName) -> FactInference {
    FactInference {
        fact: ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
        diagnostics: vec![InferenceDiagnostic::error(
            "RM-TYPE-MEMBER-MISSING",
            format!(
                "member '{}' is not present in the complete value fact",
                member.0
            ),
        )],
    }
}
