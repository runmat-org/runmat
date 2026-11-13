//! Class protocol for a sealed, opened memmap receiver. `Data` is projected
//! from the backing on access; no public cached Data property authorizes I/O.

use std::collections::HashMap;

use runmat_builtins::{
    MEMMAPFILE_CLASS, MEMMAPFILE_ERROR_TOO_MANY_OUTPUTS, MEMMAPFILE_EXPLICIT_GPU_EXTENSION,
    MEMMAPFILE_SUBSASGN_DESCRIPTOR, MEMMAPFILE_SUBSREF_DESCRIPTOR,
};
use runmat_macros::runtime_builtin;
use runmat_types::MemberAccess;
use runmat_value::{ExactObjectBuilder, MemmapReceiverAuthority, MemmapReceiverState, Value};

use super::{format_error, MemmapFormat};
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::execution::{RootedValue, RootedValueLease};
use crate::object::indexing::{ObjectIndexKind, ObjectIndexSelector, ObjectSubscript};
use crate::{BuiltinResult, OBJECT_SUBSASGN_METHOD, OBJECT_SUBSREF_METHOD};

mod data;
use data::{read_data, write_data};
mod ranges;

fn replace_receiver_state(
    runtime: Option<&crate::context::RuntimeContext>,
    object: runmat_value::ObjectInstance,
    state: MemmapReceiverState,
) -> BuiltinResult<runmat_value::ObjectInstance> {
    let policy = HostAllocationReceipt::policy(runtime).map_err(|source| {
        host_allocation_runtime_error(
            runtime,
            "RunMat:memmapfile:Allocation",
            "memmapfile receiver authority is no longer current",
            source,
        )
    })?;
    if policy == crate::context::host_allocation::HostAllocationPolicy::Unbounded {
        let mut builder = object.into_unbounded_builder().map_err(|_| {
            format_error("memmapfile: admitted receiver cannot enter an unbounded edit")
        })?;
        builder.replace_memmap_receiver(state);
        return Ok(builder.finish());
    }
    let runtime = runtime.expect("Managed allocation policy has a selected runtime");
    let allocation = |message: &'static str, source: HostAllocationError| {
        host_allocation_runtime_error(
            Some(runtime),
            "RunMat:memmapfile:Allocation",
            message,
            source,
        )
    };
    let replacement = ExactObjectBuilder::replacement_quote(
        object.property_count(),
        object.dynamic_property_count(),
    )
    .map_err(|source| {
        host_allocation_runtime_error(
            Some(runtime),
            "RunMat:memmapfile:Allocation",
            "memmapfile replacement geometry is invalid",
            HostAllocationError::Numeric(source),
        )
    })?;
    let metadata = object.unclaimed_metadata_quote().map_err(|source| {
        host_allocation_runtime_error(
            Some(runtime),
            "RunMat:memmapfile:Allocation",
            "memmapfile metadata geometry is invalid",
            HostAllocationError::Numeric(source),
        )
    })?;
    let native =
        ExactObjectBuilder::memmap_receiver_native_requested_bytes(&state).map_err(|source| {
            host_allocation_runtime_error(
                Some(runtime),
                "RunMat:memmapfile:Allocation",
                "memmapfile receiver geometry is invalid",
                HostAllocationError::Numeric(source),
            )
        })?;
    let physical = replacement
        .requested_bytes
        .checked_add(metadata.requested_bytes)
        .and_then(|bytes| bytes.checked_add(native))
        .ok_or_else(|| {
            host_allocation_runtime_error(
                Some(runtime),
                "RunMat:memmapfile:Allocation",
                "memmapfile replacement geometry overflow",
                HostAllocationError::Overflow,
            )
        })?;
    let claims = replacement
        .retained_claims
        .checked_add(metadata.retained_claims)
        .and_then(|claims| {
            claims.checked_add(ExactObjectBuilder::memmap_receiver_native_retained_claim_count())
        })
        .ok_or_else(|| {
            host_allocation_runtime_error(
                Some(runtime),
                "RunMat:memmapfile:Allocation",
                "memmapfile replacement claim count overflow",
                HostAllocationError::Overflow,
            )
        })?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(physical).map_err(|_| {
            host_allocation_runtime_error(
                Some(runtime),
                "RunMat:memmapfile:Allocation",
                "memmapfile replacement geometry overflow",
                HostAllocationError::Overflow,
            )
        })?,
        claims,
    )
    .map_err(|source| allocation("memmapfile replacement retained quote failed", source))?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, retained)
            .map_err(|source| allocation("memmapfile replacement quote failed", source))?,
    )
    .map_err(|source| allocation("memmapfile replacement admission failed", source))?
    .ok_or_else(|| {
        host_allocation_runtime_error(
            Some(runtime),
            "RunMat:memmapfile:Allocation",
            "memmapfile replacement has no managed owner",
            HostAllocationError::MissingManagedOwner,
        )
    })?;
    let permit = receipt
        .output_permit(runtime)
        .map_err(|source| allocation("memmapfile replacement permit is stale", source))?;
    object.try_admit_metadata_exact(&permit).map_err(|source| {
        host_allocation_runtime_error(
            Some(runtime),
            "RunMat:memmapfile:Allocation",
            "memmapfile metadata admission failed",
            HostAllocationError::Numeric(source),
        )
    })?;
    let builder = ExactObjectBuilder::prepare_copy(
        &object,
        object.property_count(),
        object.dynamic_property_count(),
        &permit,
    );
    receipt
        .checkpoint(runtime)
        .map_err(|source| allocation("memmapfile replacement was cancelled", source))?;
    let mut builder = builder
        .map_err(|source| format_error(format!("memmapfile: replacement failed: {source}")))?;
    let native = builder.replace_memmap_receiver(state, &permit);
    receipt
        .checkpoint(runtime)
        .map_err(|source| allocation("memmapfile replacement was cancelled", source))?;
    native.map_err(|source| {
        host_allocation_runtime_error(
            Some(runtime),
            "RunMat:memmapfile:Allocation",
            "memmapfile receiver allocation failed",
            HostAllocationError::Numeric(source),
        )
    })?;
    let replacement = builder.finish();
    receipt
        .checkpoint(runtime)
        .map_err(|source| allocation("memmapfile replacement was cancelled", source))?;
    let replacement = replacement.map_err(|source| {
        host_allocation_runtime_error(
            Some(runtime),
            "RunMat:memmapfile:Allocation",
            "memmapfile replacement publication failed",
            HostAllocationError::Numeric(source),
        )
    })?;
    Ok(replacement)
}

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
                    role: runmat_types::ClassMethodRole::Ordinary,
                },
            );
        }
        crate::class_registry::register_builtin_class(
            MEMMAPFILE_CLASS,
            crate::class_registry::RuntimeClass {
                events: Default::default(),
                name: MEMMAPFILE_CLASS.into(),
                parent: None,
                properties,
                methods,
            },
        );
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
) -> BuiltinResult<RootedValue> {
    crate::output_count::validate_fixed_outputs(
        &MEMMAPFILE_SUBSREF_DESCRIPTOR,
        &MEMMAPFILE_ERROR_TOO_MANY_OUTPUTS,
        SUBSREF,
    )?;
    if crate::builtins::common::validation::value_contains_explicit_gpu(&subscript) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &MEMMAPFILE_EXPLICIT_GPU_EXTENSION,
            SUBSREF,
        )?;
    }
    ensure_class_registered();
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    if path.steps().len() > 1 && member(&path.steps()[0])? == Property::Data {
        return ranges::read(runtime.as_ref(), receiver, path).await;
    }
    let read_runtime = runtime.clone();
    crate::object::protocol::execute_owned_subsref_with_context(
        runtime.as_ref(),
        receiver,
        path,
        move |receiver, step| async move {
            read_first(read_runtime.as_ref(), receiver, step).await
        },
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
) -> BuiltinResult<RootedValue> {
    crate::output_count::validate_fixed_outputs(
        &MEMMAPFILE_SUBSASGN_DESCRIPTOR,
        &MEMMAPFILE_ERROR_TOO_MANY_OUTPUTS,
        SUBSASGN,
    )?;
    if crate::builtins::common::validation::value_contains_explicit_gpu(&subscript)
        || crate::builtins::common::validation::value_contains_explicit_gpu(&rhs)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &MEMMAPFILE_EXPLICIT_GPU_EXTENSION,
            SUBSASGN,
        )?;
    }
    ensure_class_registered();
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    if path.steps().len() > 1 && member(&path.steps()[0])? == Property::Data {
        return ranges::write(runtime.as_ref(), receiver, path, rhs).await;
    }
    let read_runtime = runtime.clone();
    let write_runtime = runtime.clone();
    crate::object::protocol::execute_owned_subsasgn_with_context(
        runtime.as_ref(),
        receiver,
        path,
        vec![rhs],
        move |receiver, step| async move { read_first(read_runtime.as_ref(), receiver, step).await },
        move |receiver, step, values| async move {
            let [value] = <[Value; 1]>::try_from(values).map_err(|_| format_error("memmapfile: assignment requires one value"))?;
            write_first(write_runtime.as_ref(), receiver, step, value).await
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
    match name.as_str() {
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
) -> BuiltinResult<RootedValue> {
    let _receiver_root = RootedValueLease::for_values(std::iter::once(&receiver))
        .map_err(|error| error.into_runtime_error())?;
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
    let value = match property {
        Property::Filename => object
            .properties()
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
    }?;
    RootedValue::new(value).map_err(|error| error.into_runtime_error())
}

async fn write_first(
    runtime: Option<&crate::context::RuntimeContext>,
    receiver: Value,
    step: ObjectSubscript,
    rhs: Value,
) -> BuiltinResult<RootedValue> {
    let _input_roots = RootedValueLease::for_values([&receiver, &rhs].into_iter())
        .map_err(|error| error.into_runtime_error())?;
    let property = member(&step)?;
    let Value::Object(object) = receiver else {
        return Err(format_error("memmapfile: receiver is not an object"));
    };
    if !object.is_class(MEMMAPFILE_CLASS) {
        return Err(format_error("memmapfile: receiver has the wrong class"));
    }
    let mut state = object
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
    let _gathered_root = RootedValueLease::for_values(std::iter::once(&rhs))
        .map_err(|error| error.into_runtime_error())?;
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
            state.writable = writable;
        }
        Property::Offset => {
            let offset = super::numeric_usize(&rhs, "Offset")?;
            if u64::try_from(offset)
                .ok()
                .is_none_or(|offset| offset > opened_extent)
            {
                return Err(format_error("memmapfile: Offset exceeds opened extent"));
            }
            state.offset = offset;
        }
        Property::Repeat => {
            let repeat = super::repeat_count(&rhs)?;
            MemmapFormat::from_receiver(&state)?.validate_layout(repeat)?;
            state.repeat = repeat;
        }
        Property::Format => {
            let format = MemmapFormat::from_value(&rhs)?;
            format.validate_layout(state.repeat)?;
            state.fields = format.receiver_fields();
            state.format_presentation = format.receiver_presentation();
        }
        Property::Filename => return Err(format_error("memmapfile: property is not assignable")),
    }
    let object = if property == Property::Data {
        object
    } else {
        replace_receiver_state(runtime, object, state)?
    };
    match runtime {
        Some(runtime) => RootedValue::in_context(Value::Object(object), runtime),
        None => RootedValue::new(Value::Object(object)).map_err(|error| error.into_runtime_error()),
    }
}
