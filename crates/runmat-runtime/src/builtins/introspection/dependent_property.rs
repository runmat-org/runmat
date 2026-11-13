use crate::build_runtime_error;
use runmat_macros::runtime_builtin;
use runmat_value::Value;
#[cfg(test)]
mod tests;
// Test-oriented dependent property handlers (global). If a class defines a Dependent
// property named 'p', the VM will try to call get.p / set.p. We provide generic
// implementations that read/write a conventional backing field 'p_backing'.
#[runtime_builtin(
    name = "get.p",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::dependent_property"
)]
pub async fn get_p_builtin(obj: Value) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::GET_P_DESCRIPTOR,
        &runmat_builtins::GET_P_ERROR_OUTPUTS,
        "get.p",
    )?;
    match obj {
        Value::Object(o) => {
            if let Some(v) = o.property("p_backing") {
                Ok(v.clone())
            } else {
                Ok(Value::Num(0.0))
            }
        }
        other => Err(build_runtime_error(format!(
            "get.p: requires object receiver (got {other:?})"
        ))
        .with_builtin("get.p")
        .with_identifier("RunMat:GetPReceiverInvalid")
        .build()),
    }
}

#[runtime_builtin(
    name = "set.p",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::dependent_property"
)]
pub async fn set_p_builtin(obj: Value, val: Value) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::SET_P_DESCRIPTOR,
        &runmat_builtins::SET_P_ERROR_OUTPUTS,
        "set.p",
    )?;
    match obj {
        Value::Object(o) => {
            let mut o = o.into_unbounded_builder().map_err(|_| {
                build_runtime_error("set.p: admitted receiver needs an admitted edit generation")
                    .with_builtin("set.p")
                    .with_identifier("RunMat:SetPReceiverInvalid")
                    .build()
            })?;
            o.insert("p_backing".to_string(), val);
            Ok(Value::Object(o.finish()))
        }
        other => Err(build_runtime_error(format!(
            "set.p: requires object receiver (got {other:?})"
        ))
        .with_builtin("set.p")
        .with_identifier("RunMat:SetPReceiverInvalid")
        .build()),
    }
}
