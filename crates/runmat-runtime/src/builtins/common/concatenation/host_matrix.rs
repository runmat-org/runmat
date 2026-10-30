//! Synchronous host-only matrix literal route through the canonical cat tail.

use super::{concat_error, validate_matrix_row_widths};
use crate::BuiltinResult;
use runmat_value::{Tensor, Value};

/// A literal may bypass builtin suspension only when its input classes and
/// exact selected registrations are the canonical host implementations.
/// Managed-policy output accounting remains on the ordinary async route.
pub fn host_matrix_can_run_inline(
    context: &crate::context::RuntimeContext,
    rows: &[Vec<Value>],
) -> BuiltinResult<bool> {
    use crate::context::host_allocation::{HostAllocationPolicy, HostAllocationReceipt};

    let policy = HostAllocationReceipt::policy(Some(context)).map_err(|error| {
        crate::builtin::builtin_control_error(Some(&context.cancellation_context()), "cat", error)
    })?;
    if policy != HostAllocationPolicy::Unbounded || context.service_ports().builtin().is_some() {
        return Ok(false);
    }
    Ok(platform_can_run_inline(rows))
}

// Browser registration can change between a read and an allocating host
// assembly, so it retains the ordinary selected async route.
#[cfg(target_arch = "wasm32")]
fn platform_can_run_inline(_rows: &[Vec<Value>]) -> bool {
    false
}

#[cfg(not(target_arch = "wasm32"))]
fn platform_can_run_inline(rows: &[Vec<Value>]) -> bool {
    use crate::builtins::array::shape::{cat, horzcat, vertcat};
    static CANONICAL_NATIVE_BINDINGS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let canonical = *CANONICAL_NATIVE_BINDINGS.get_or_init(|| {
        for (name, matches) in [
            (
                "cat",
                cat::is_canonical_host_binding as fn(crate::builtin::RuntimeBuiltinBinding) -> bool,
            ),
            (
                "horzcat",
                horzcat::is_canonical_host_binding
                    as fn(crate::builtin::RuntimeBuiltinBinding) -> bool,
            ),
            ("vertcat", vertcat::is_canonical_host_binding),
        ] {
            let selected =
                crate::builtin::registration::registered_runtime_builtin_bindings_by_name(name);
            if selected.len() != 1
                || !selected[0].registration().is_current()
                || !matches(selected[0].binding())
            {
                return false;
            }
        }
        true
    });
    if !canonical {
        return false;
    }
    rows.iter().flatten().all(host_literal_value)
}

#[cfg(not(target_arch = "wasm32"))]
fn host_literal_value(value: &Value) -> bool {
    // cat's trailing name-value parser treats a textual `like` token as a
    // control in some argument positions. Keep those literals on dispatch.
    let like_token = match value {
        Value::String(runmat_value::StringElement::Text(text)) => text.eq_ignore_ascii_case("like"),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .eq_ignore_ascii_case("like")
        }
        Value::CharArray(array) if array.rows == 1 => array
            .data()
            .iter()
            .copied()
            .map(|unit| unit.raw())
            .map(|unit| {
                u8::try_from(unit)
                    .ok()
                    .map(|byte| byte.to_ascii_lowercase())
            })
            .eq(b"like".iter().copied().map(Some)),
        _ => false,
    };
    !like_token
        && matches!(
            value,
            Value::Num(_)
                | Value::Int(_)
                | Value::Bool(_)
                | Value::Complex(_, _)
                | Value::Tensor(_)
                | Value::ComplexTensor(_)
                | Value::LogicalArray(_)
                | Value::CharArray(_)
                | Value::String(_)
                | Value::StringArray(_)
        )
}

/// Move owned rows through the same host class/shape assembly as canonical
/// horzcat/cat/vertcat. Input roots remain live until each successor is rooted.
pub fn create_host_matrix_from_values(
    rows: Vec<Vec<Value>>,
) -> BuiltinResult<crate::execution::RootedValue> {
    let _inputs_root =
        crate::execution::RootedValueLease::for_values(rows.iter().flat_map(|row| row.iter()))
            .map_err(|error| error.into_runtime_error())?;
    if rows.is_empty() {
        return crate::execution::RootedValue::new(Value::Tensor(
            Tensor::new(vec![], vec![0, 0]).map_err(concat_error)?,
        ))
        .map_err(|error| error.into_runtime_error());
    }
    let mut row_matrices = crate::execution::RootedValuesBuilder::with_capacity(rows.len());
    for row in rows {
        let value = crate::builtins::array::shape::horzcat::concatenate_host_values(row)?;
        row_matrices
            .push(value)
            .map_err(|error| error.into_runtime_error())?;
    }
    let row_matrices = row_matrices
        .finish()
        .map_err(|error| error.into_runtime_error())?;
    validate_matrix_row_widths(row_matrices.values())?;
    row_matrices.with_unbounded_values(|mut values| {
        let value = if values.len() == 1 {
            values.remove(0)
        } else {
            crate::builtins::array::shape::vertcat::concatenate_host_values(values)?
        };
        crate::execution::RootedValue::new(value).map_err(|error| error.into_runtime_error())
    })?
}
