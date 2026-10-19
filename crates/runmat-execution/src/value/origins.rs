use std::collections::HashSet;

use super::{
    CallableOwner, DurableObjectArray, DurablePlainObject, InlineValue, RegisteredData,
    StructField, ValuePayload,
};
use crate::FunctionBindingOrigin;

pub(super) fn binding_origins(values: &[ValuePayload]) -> HashSet<FunctionBindingOrigin> {
    let mut origins = HashSet::new();
    for value in values {
        visit_payload(value, &mut origins);
    }
    origins
}

fn visit_payload(value: &ValuePayload, origins: &mut HashSet<FunctionBindingOrigin>) {
    let ValuePayload::Inline(value) = value else {
        return;
    };
    match value.as_ref() {
        InlineValue::Callable(callable) => {
            if let CallableOwner::Bound(origin) = callable.owner {
                origins.insert(origin);
            }
            visit_payloads(&callable.captures, origins);
        }
        InlineValue::Cell { values, .. } => visit_payloads(values, origins),
        InlineValue::Struct(fields) | InlineValue::DurableOrderedStruct(fields) => {
            visit_fields(fields, origins);
        }
        InlineValue::Symbolic(value) | InlineValue::ImmutableValueClass(value) => {
            visit_registered(value, origins);
        }
        InlineValue::DurablePlainObject(value) => visit_plain_object(value, origins),
        InlineValue::DurableStructArray(value) => {
            for field in &value.fields {
                visit_payloads(&field.values, origins);
            }
        }
        InlineValue::DurableObjectArray(value) => visit_object_array(value, origins),
        InlineValue::Null
        | InlineValue::Logical(_)
        | InlineValue::F64Bits(_)
        | InlineValue::I8(_)
        | InlineValue::I16(_)
        | InlineValue::I32(_)
        | InlineValue::I64(_)
        | InlineValue::U8(_)
        | InlineValue::U16(_)
        | InlineValue::U32(_)
        | InlineValue::U64(_)
        | InlineValue::ComplexF64Bits { .. }
        | InlineValue::String(_)
        | InlineValue::Char { .. }
        | InlineValue::StringArray { .. }
        | InlineValue::Dense(_)
        | InlineValue::Sparse(_)
        | InlineValue::Exception(_) => {}
    }
}

fn visit_payloads(values: &[ValuePayload], origins: &mut HashSet<FunctionBindingOrigin>) {
    for value in values {
        visit_payload(value, origins);
    }
}

fn visit_fields(fields: &[StructField], origins: &mut HashSet<FunctionBindingOrigin>) {
    for field in fields {
        visit_payload(&field.value, origins);
    }
}

fn visit_registered(value: &RegisteredData, origins: &mut HashSet<FunctionBindingOrigin>) {
    for field in &value.fields {
        visit_payload(&field.value, origins);
    }
}

fn visit_plain_object(value: &DurablePlainObject, origins: &mut HashSet<FunctionBindingOrigin>) {
    visit_fields(&value.properties, origins);
}

fn visit_object_array(value: &DurableObjectArray, origins: &mut HashSet<FunctionBindingOrigin>) {
    for element in &value.elements {
        visit_plain_object(element, origins);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{CallableForm, CallableValue, RegisteredField, ValueLimits};
    use crate::Digest;
    use runmat_types::{FunctionSourceRole, ProgramFunctionId};

    fn bound_callable(origin: FunctionBindingOrigin, captures: Vec<ValuePayload>) -> ValuePayload {
        let owner = CallableOwner::Bound(origin);
        let qualified_name = "nested".to_string();
        let form = if captures.is_empty() {
            CallableForm::Handle
        } else {
            CallableForm::Closure
        };
        let source_role = FunctionSourceRole::Local;
        ValuePayload::Inline(Box::new(InlineValue::Callable(CallableValue {
            callable_digest: CallableValue::identity_digest(
                owner,
                &qualified_name,
                form,
                source_role,
                None,
            ),
            owner,
            qualified_name,
            form,
            source_role,
            source_file: None,
            captures,
        })))
    }

    #[test]
    fn nested_callable_origins_are_deduplicated() {
        let first = FunctionBindingOrigin {
            program: Digest::sha256("first"),
            function: ProgramFunctionId(1),
        };
        let second = FunctionBindingOrigin {
            program: Digest::sha256("second"),
            function: ProgramFunctionId(2),
        };
        let nested = ValuePayload::Inline(Box::new(InlineValue::Cell {
            shape: vec![1, 2],
            values: vec![
                bound_callable(first, Vec::new()),
                bound_callable(second, vec![bound_callable(first, Vec::new())]),
            ],
        }));

        let aggregate =
            ValuePayload::Inline(Box::new(InlineValue::ImmutableValueClass(RegisteredData {
                type_identity: "example.Value".into(),
                schema_version: 1,
                fields: vec![RegisteredField {
                    name: "callback".into(),
                    value: nested,
                }],
            })));
        let origins = ValuePayload::binding_origins(&[aggregate], ValueLimits::default())
            .expect("valid aggregate");
        assert_eq!(origins.len(), 2);
        assert!(origins.contains(&first));
        assert!(origins.contains(&second));
    }

    #[test]
    fn origin_walk_rejects_payloads_over_the_selected_depth_limit() {
        let nested = ValuePayload::Inline(Box::new(InlineValue::Cell {
            shape: vec![1, 1],
            values: vec![ValuePayload::Inline(Box::new(InlineValue::Cell {
                shape: vec![1, 1],
                values: vec![ValuePayload::Inline(Box::new(InlineValue::Logical(true)))],
            }))],
        }));

        assert!(ValuePayload::binding_origins(
            &[nested],
            ValueLimits {
                max_depth: 1,
                ..ValueLimits::default()
            }
        )
        .is_err());
    }
}
