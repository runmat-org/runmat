use super::*;

/// A pure structural fact, not the result of a getter or `subsref` probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MemberPresence {
    Present,
    Absent,
    Unknown,
}

fn declared_or_stored(
    obj: &runmat_value::ObjectInstance,
    field: &runmat_types::MemberName,
) -> Result<MemberPresence, RuntimeError> {
    let selected = crate::class_registry::checked_receiver_projection(obj)?;
    if obj.properties().contains_key(&field.0)
        || obj.has_dynamic_property(&field.0)
        || selected.as_ref().map_or_else(
            || crate::class_registry::lookup_property(obj.class_name(), field).is_some(),
            |projection| {
                projection
                    .receiver_property(obj.class_name(), field)
                    .is_some()
            },
        )
    {
        Ok(MemberPresence::Present)
    } else {
        Ok(MemberPresence::Absent)
    }
}

/// Distinguish absent members from unknown storage without invoking user code.
/// Unknown foreign receivers retain their existing member-resolution path.
pub(crate) fn has_stored_member(
    base: &Value,
    field: &runmat_types::MemberName,
) -> Result<MemberPresence, RuntimeError> {
    match base {
        Value::Object(obj) => declared_or_stored(obj, field),
        Value::HandleObject(handle) => runmat_gc::gc_with_value(&handle.target, |value| {
            let Value::Object(obj) = value else {
                return Err(invalid_handle());
            };
            if !handle.valid
                || obj.class_name() != &handle.class_name
                || !crate::object_handle_flag_valid(obj)
            {
                return Err(invalid_handle());
            }
            declared_or_stored(obj, field)
        })
        .map_err(|_| invalid_handle())?,
        Value::ClassRef(class) => Ok(
            if crate::class_registry::lookup_property(class, field).is_some()
                || crate::class_registry::class_has_enumeration_member(class, &field.0)
            {
                MemberPresence::Present
            } else {
                MemberPresence::Absent
            },
        ),
        Value::Struct(value) => Ok(if value.fields.contains_key(&field.0) {
            MemberPresence::Present
        } else {
            MemberPresence::Absent
        }),
        Value::StructArray(array) => Ok(if array.field_values(&field.0).is_some() {
            MemberPresence::Present
        } else {
            MemberPresence::Absent
        }),
        Value::ObjectArray(array) => {
            if array.is_empty()
                && crate::class_registry::lookup_property(array.class_name(), field).is_some()
            {
                return Ok(MemberPresence::Present);
            }
            for element in array.data() {
                match has_stored_member(element, field)? {
                    MemberPresence::Present => return Ok(MemberPresence::Present),
                    MemberPresence::Unknown => return Ok(MemberPresence::Unknown),
                    MemberPresence::Absent => {}
                }
            }
            Ok(MemberPresence::Absent)
        }
        Value::Foreign(_) => Ok(MemberPresence::Unknown),
        _ => Ok(MemberPresence::Absent),
    }
}

fn invalid_handle() -> RuntimeError {
    mex(
        "RunMat:getfield:InvalidHandle",
        "Invalid or deleted handle object.",
    )
}
