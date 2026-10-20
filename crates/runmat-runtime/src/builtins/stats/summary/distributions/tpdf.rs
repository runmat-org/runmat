use super::*;
use runmat_builtins::{
    TPDF_INTEGER_NU_EXTENSION as INTEGER_NU_EXTENSION,
    TPDF_INTEGER_X_EXTENSION as INTEGER_X_EXTENSION,
    TPDF_LOGICAL_INPUT_EXTENSION as LOGICAL_INPUT_EXTENSION,
};

#[runtime_builtin(
    name = "tpdf",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::distributions::tpdf"
)]
pub(crate) async fn tpdf_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if rest.len() != 1 {
        return Err(super::normal_error("tpdf", "tpdf: expected x and nu"));
    }
    let nu_value = rest.into_iter().next().expect("one nu argument");
    ensure_extensions(&value, &nu_value)?;
    numeric_inputs("tpdf", &[&value, &nu_value], NumericShapePolicy::Broadcast)?;
    let output = NormalOutputPlan::inspect(
        "tpdf",
        &value,
        crate::call::FunctionArgumentView::from_slice(std::slice::from_ref(&nu_value)),
    )?;
    let x = super::normal_value_to_tensor("tpdf", "x", value).await?;
    let nu = super::normal_value_to_tensor("tpdf", "nu", nu_value).await?;
    let (x, nu, shape) = super::broadcast_pair("tpdf", &x, &nu)?;
    let data = x
        .iter()
        .zip(nu.iter())
        .map(|(x, nu)| distribution_math::student_t_pdf(*x, *nu))
        .collect();
    output.finish("tpdf", shape, data)
}

fn ensure_extensions(x: &Value, nu: &Value) -> BuiltinResult<()> {
    if super::is_typed_integer_value(x) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_X_EXTENSION, "tpdf")?;
    }
    if super::is_typed_integer_value(nu) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_NU_EXTENSION, "tpdf")?;
    }
    if super::is_logical_value(x) || super::is_logical_value(nu) {
        crate::compatibility::ensure_builtin_extension_enabled(&LOGICAL_INPUT_EXTENSION, "tpdf")?;
    }
    Ok(())
}
