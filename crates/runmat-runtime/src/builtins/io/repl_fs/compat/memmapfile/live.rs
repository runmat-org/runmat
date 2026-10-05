//! Class protocol for a sealed, opened memmap receiver. `Data` is projected
//! from the backing on access; no public cached Data property authorizes I/O.

use std::collections::HashMap;

use runmat_builtins::MEMMAPFILE_CLASS;
use runmat_macros::runtime_builtin;
use runmat_types::MemberAccess;
use runmat_value::{MemmapReceiverAuthority, Value};

use super::{format_error, MemmapFormat};
use crate::object::indexing::{ObjectIndexKind, ObjectIndexSelector, ObjectSubscript};
use crate::{BuiltinResult, OBJECT_SUBSASGN_METHOD, OBJECT_SUBSREF_METHOD};

mod data;
use data::{read_data, write_data};
mod ranges;

#[cfg(test)]
#[path = "live/tests.rs"]
mod tests;

const SUBSREF: &str = "memmapfile.subsref";
const SUBSASGN: &str = "memmapfile.subsasgn";
static CLASS: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(MEMMAPFILE_CLASS);

pub(in crate::builtins::io::repl_fs::compat) fn ensure_class_registered() {
    CLASS.ensure(|| {
        let mut properties = HashMap::new();
        for name in ["Filename", "Writable", "Offset", "Format", "Repeat", "Data"] {
            properties.insert(
                name.into(),
                crate::class_registry::RuntimeProperty {
                    name: name.into(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: true,
                    get_access: MemberAccess::Public,
                    set_access: if name == "Filename" {
                        MemberAccess::Private
                    } else {
                        MemberAccess::Public
                    },
                    default_value: None,
                },
            );
        }
        let mut methods = HashMap::new();
        for (name, function_name) in [
            (OBJECT_SUBSREF_METHOD, SUBSREF),
            (OBJECT_SUBSASGN_METHOD, SUBSASGN),
        ] {
            methods.insert(
                name.into(),
                crate::class_registry::RuntimeMethod {
                    name: name.into(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: function_name.into(),
                    implicit_class_argument: None,
                },
            );
        }
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: MEMMAPFILE_CLASS.into(),
            parent: None,
            properties,
            methods,
        });
    });
}

#[runtime_builtin(
    name = "memmapfile.subsref",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::repl_fs::compat::memmapfile::live"
)]
pub async fn subsref(
    runtime: Option<crate::context::RuntimeContext>,
    receiver: Value,
    subscript: Value,
) -> BuiltinResult<Value> {
    ensure_class_registered();
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    if path.steps().len() > 1 && member(&path.steps()[0])? == Property::Data {
        return ranges::read(runtime.as_ref(), receiver, path).await;
    }
    crate::object::protocol::execute_owned_subsref(
        receiver,
        path,
        move |receiver, step| async move { read_first(runtime.as_ref(), receiver, step).await },
        None,
    )
    .await
}

#[runtime_builtin(
    name = "memmapfile.subsasgn",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::repl_fs::compat::memmapfile::live"
)]
pub async fn subsasgn(
    runtime: Option<crate::context::RuntimeContext>,
    receiver: Value,
    subscript: Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    ensure_class_registered();
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    if path.steps().len() > 1 && member(&path.steps()[0])? == Property::Data {
        return ranges::write(runtime.as_ref(), receiver, path, rhs).await;
    }
    let read_runtime = runtime.clone();
    crate::object::protocol::execute_owned_subsasgn(
        receiver,
        path,
        vec![rhs],
        move |receiver, step| async move { read_first(read_runtime.as_ref(), receiver, step).await },
        move |receiver, step, values| async move {
            let [value] = <[Value; 1]>::try_from(values).map_err(|_| format_error("memmapfile: assignment requires one value"))?;
            write_first(runtime.as_ref(), receiver, step, value).await
        },
        None,
    )
    .await
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Property {
    Filename,
    Writable,
    Offset,
    Format,
    Repeat,
    Data,
}

fn member(step: &ObjectSubscript) -> BuiltinResult<Property> {
    if step.kind() != ObjectIndexKind::Member {
        return Err(format_error(
            "memmapfile: receiver requires a property access",
        ));
    }
    let ObjectIndexSelector::Member(name) = step.selector() else {
        return Err(format_error("memmapfile: invalid property selector"));
    };
    match name.0.as_str() {
        "Filename" => Ok(Property::Filename),
        "Writable" => Ok(Property::Writable),
        "Offset" => Ok(Property::Offset),
        "Format" => Ok(Property::Format),
        "Repeat" => Ok(Property::Repeat),
        "Data" => Ok(Property::Data),
        _ => Err(format_error("memmapfile: unknown property")),
    }
}

async fn read_first(
    runtime: Option<&crate::context::RuntimeContext>,
    receiver: Value,
    step: ObjectSubscript,
) -> BuiltinResult<Value> {
    let property = member(&step)?;
    let Value::Object(object) = receiver else {
        return Err(format_error("memmapfile: receiver is not an object"));
    };
    if !object.is_class(MEMMAPFILE_CLASS) {
        return Err(format_error("memmapfile: receiver has the wrong class"));
    }
    let state = object
        .memmap_receiver()
        .ok_or_else(|| format_error("memmapfile: receiver has no sealed state"))?;
    let backing = match &state.authority {
        MemmapReceiverAuthority::Opened(_) => Some(super::authority::resolve(runtime, state)?),
        MemmapReceiverAuthority::Detached(_) => None,
    };
    match property {
        Property::Filename => object
            .properties
            .get("Filename")
            .cloned()
            .ok_or_else(|| format_error("memmapfile: Filename presentation is missing")),
        Property::Writable => Ok(Value::Bool(state.writable)),
        Property::Offset => super::memmap_count_value(state.offset),
        Property::Repeat => state
            .repeat
            .map(super::memmap_count_value)
            .transpose()
            .map(|value| value.unwrap_or(Value::Num(f64::INFINITY))),
        Property::Format => MemmapFormat::from_receiver(state)?.to_value(),
        Property::Data => match (&state.authority, backing) {
            (MemmapReceiverAuthority::Opened(_), Some(backing)) => {
                read_data(runtime, state, &backing).await
            }
            (MemmapReceiverAuthority::Detached(snapshot), None) => Ok((**snapshot).clone()),
            _ => unreachable!("typed mapped authority and backing agree"),
        },
    }
}

async fn write_first(
    runtime: Option<&crate::context::RuntimeContext>,
    receiver: Value,
    step: ObjectSubscript,
    rhs: Value,
) -> BuiltinResult<Value> {
    let property = member(&step)?;
    let Value::Object(mut object) = receiver else {
        return Err(format_error("memmapfile: receiver is not an object"));
    };
    if !object.is_class(MEMMAPFILE_CLASS) {
        return Err(format_error("memmapfile: receiver has the wrong class"));
    }
    let state = object
        .memmap_receiver()
        .cloned()
        .ok_or_else(|| format_error("memmapfile: receiver has no sealed state"))?;
    if !matches!(
        property,
        Property::Data
            | Property::Writable
            | Property::Offset
            | Property::Repeat
            | Property::Format
    ) {
        return Err(format_error("memmapfile: property is not assignable"));
    }
    if matches!(&state.authority, MemmapReceiverAuthority::Detached(_)) {
        return Err(format_error("memmapfile: detached snapshot is read-only"));
    }
    let backing = super::authority::resolve(runtime, &state)?;
    let rhs = if let Some(runtime) = runtime {
        runtime
            .cpu_cooperation()
            .await_cancellable(crate::gather_if_needed_async(&rhs))
            .await
            .map_err(|source| source.into_runtime_error())??
    } else {
        crate::gather_if_needed_async(&rhs).await?
    };
    let opened = if let Some(runtime) = runtime {
        runtime
            .cpu_cooperation()
            .await_cancellable(backing.lock())
            .await
            .map_err(|source| source.into_runtime_error())?
    } else {
        backing.lock().await
    };
    opened.check_current().map_err(|source| {
        crate::build_runtime_error(format!("memmapfile: mapped authority changed: {source}"))
            .with_builtin("memmapfile")
            .with_source(source)
            .build()
    })?;
    let opened_writable = opened.writable();
    let opened_extent = opened.opened_extent();
    drop(opened);
    match property {
        Property::Data => write_data(runtime, &state, &backing, &rhs).await?,
        Property::Writable => {
            let Value::Bool(writable) = rhs else {
                return Err(format_error("memmapfile: Writable requires logical scalar"));
            };
            if writable && !opened_writable {
                return Err(format_error("memmapfile: opened backing is read-only"));
            }
            object
                .memmap_receiver_mut()
                .expect("resolved state")
                .writable = writable;
        }
        Property::Offset => {
            let offset = super::numeric_usize(&rhs, "Offset")?;
            if u64::try_from(offset)
                .ok()
                .is_none_or(|offset| offset > opened_extent)
            {
                return Err(format_error("memmapfile: Offset exceeds opened extent"));
            }
            object.memmap_receiver_mut().expect("resolved state").offset = offset;
        }
        Property::Repeat => {
            let repeat = super::repeat_count(&rhs)?;
            MemmapFormat::from_receiver(&state)?.validate_layout(repeat)?;
            object.memmap_receiver_mut().expect("resolved state").repeat = repeat;
        }
        Property::Format => {
            let format = MemmapFormat::from_value(&rhs)?;
            format.validate_layout(state.repeat)?;
            let receiver = object.memmap_receiver_mut().expect("resolved state");
            receiver.fields = format.receiver_fields();
            receiver.format_presentation = format.receiver_presentation();
        }
        Property::Filename => return Err(format_error("memmapfile: property is not assignable")),
    }
    Ok(Value::Object(object))
}
