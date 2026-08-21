use crate::{
    DynamicReason, FactInference, InferenceDiagnostic, MemberName, ValueFact, ValueKindFact,
};

pub fn infer_member_read(base: &ValueFact, member: &MemberName) -> FactInference {
    match &base.kind {
        ValueKindFact::Struct(structure) => structure.fields.get(&member.0).cloned().map_or_else(
            || {
                if structure.fields_complete {
                    missing_member(member)
                } else {
                    FactInference::exact(ValueFact::unknown(DynamicReason::RuntimeValue))
                }
            },
            FactInference::exact,
        ),
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
            structure.fields.insert(member.0.clone(), assigned.clone());
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
