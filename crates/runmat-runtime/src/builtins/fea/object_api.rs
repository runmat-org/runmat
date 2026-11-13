//! Legacy FEA adapter ownership boundaries. These helpers do not infer an
//! ambient context: contextless adapters may construct only under ordinary
//! policy. Existing generations are edited by the canonical replacement path.
use runmat_value::{ObjectInstance, UnboundedObjectBuilder, Value};

use super::{builtin_error, builtin_error_with_source, BuiltinResult, ERROR_INTERNAL};

pub(super) fn ordinary_builder(
    builtin: &'static str,
    class_name: &str,
) -> BuiltinResult<UnboundedObjectBuilder> {
    use crate::context::host_allocation::{HostAllocationPolicy, HostAllocationReceipt};
    let policy = HostAllocationReceipt::policy(None).map_err(|source| {
        builtin_error_with_source(builtin, &ERROR_INTERNAL, "FEA object authority", source)
    })?;
    if !matches!(policy, HostAllocationPolicy::Unbounded) {
        return Err(builtin_error(
            builtin,
            &ERROR_INTERNAL,
            "contextless FEA object construction requires ordinary allocation authority",
        ));
    }
    Ok(UnboundedObjectBuilder::new(class_name.to_owned()))
}

pub(super) fn replace(
    builtin: &'static str,
    object: &mut ObjectInstance,
    name: &str,
    value: Value,
) -> BuiltinResult<Option<Value>> {
    crate::execution::object::replacement::replace_property(
        None,
        object,
        name,
        value,
        |message, source| builtin_error_with_source(builtin, &ERROR_INTERNAL, message, source),
        |message| builtin_error(builtin, &ERROR_INTERNAL, message),
    )
}

pub(super) fn remove(
    builtin: &'static str,
    object: &mut ObjectInstance,
    name: &str,
) -> BuiltinResult<Option<Value>> {
    if object.property(name).is_none() {
        return Ok(None);
    }
    crate::execution::object::replacement::remove_property(
        None,
        object,
        name,
        |message, source| builtin_error_with_source(builtin, &ERROR_INTERNAL, message, source),
        |message| builtin_error(builtin, &ERROR_INTERNAL, message),
    )
    .map(Some)
}
