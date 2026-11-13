use super::*;
use runmat_builtins::{
    TINV_INTEGER_NU_EXTENSION as INTEGER_NU_EXTENSION,
    TINV_INTEGER_P_EXTENSION as INTEGER_P_EXTENSION,
    TINV_LOGICAL_INPUT_EXTENSION as LOGICAL_INPUT_EXTENSION,
};

#[runtime_builtin(
    name = "tinv",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::distributions::tinv"
)]
pub(crate) async fn tinv_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if rest.len() != 1 {
        return Err(super::normal_error("tinv", "tinv: expected p and nu"));
    }
    let nu_value = rest.into_iter().next().expect("one nu argument");
    ensure_extensions(&value, &nu_value)?;
    numeric_inputs("tinv", &[&value, &nu_value], NumericShapePolicy::Broadcast)?;
    let output = NormalOutputPlan::inspect(
        "tinv",
        &value,
        crate::call::FunctionArgumentView::from_slice(std::slice::from_ref(&nu_value)),
    )?;
    let p = super::normal_value_to_tensor("tinv", "p", value).await?;
    let nu = super::normal_value_to_tensor("tinv", "nu", nu_value).await?;
    let (p, nu, shape) = super::broadcast_pair("tinv", &p, &nu)?;
    let data = p
        .iter()
        .zip(nu.iter())
        .map(|(p, nu)| distribution_math::student_t_inv(*p, *nu))
        .collect();
    output.finish("tinv", shape, data)
}

fn ensure_extensions(p: &Value, nu: &Value) -> BuiltinResult<()> {
    if super::is_typed_integer_value(p) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_P_EXTENSION, "tinv")?;
    }
    if super::is_typed_integer_value(nu) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_NU_EXTENSION, "tinv")?;
    }
    if super::is_logical_value(p) || super::is_logical_value(nu) {
        crate::compatibility::ensure_builtin_extension_enabled(&LOGICAL_INPUT_EXTENSION, "tinv")?;
    }
    Ok(())
}
