use super::*;
use runmat_builtins::{
    TCDF_INTEGER_NU_EXTENSION as INTEGER_NU_EXTENSION,
    TCDF_INTEGER_X_EXTENSION as INTEGER_X_EXTENSION,
    TCDF_LOGICAL_INPUT_EXTENSION as LOGICAL_INPUT_EXTENSION,
};

#[runtime_builtin(
    name = "tcdf",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::distributions::tcdf"
)]
pub(crate) async fn tcdf_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args = parse_args(value, rest).await?;
    let data = args
        .x
        .iter()
        .zip(args.nu.iter())
        .map(|(x, nu)| {
            if args.tail == DistributionTail::Upper {
                distribution_math::student_t_cdf_upper(*x, *nu)
            } else {
                distribution_math::student_t_cdf(*x, *nu)
            }
        })
        .collect();
    args.output.finish("tcdf", args.shape, data)
}

struct Args {
    x: Vec<f64>,
    nu: Vec<f64>,
    shape: Vec<usize>,
    tail: DistributionTail,
    output: NormalOutputPlan,
}

async fn parse_args(value: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Args> {
    let (rest, tail) = super::split_upper_tail(rest)?;
    if rest.len() != 1 {
        return Err(super::normal_error(
            "tcdf",
            "tcdf: expected x and nu, optionally followed by 'upper'",
        ));
    }
    let nu_value = rest[0].clone();
    ensure_extensions(&value, &nu_value)?;
    numeric_inputs("tcdf", &[&value, &nu_value], NumericShapePolicy::Broadcast)?;
    let output = NormalOutputPlan::inspect(
        "tcdf",
        &value,
        crate::call::FunctionArgumentView::from_slice(std::slice::from_ref(&nu_value)),
    )?;
    let x = super::normal_value_to_tensor("tcdf", "x", value).await?;
    let nu = super::normal_value_to_tensor("tcdf", "nu", nu_value).await?;
    let (x, nu, shape) = super::broadcast_pair("tcdf", &x, &nu)?;
    Ok(Args {
        x,
        nu,
        shape,
        tail,
        output,
    })
}

fn ensure_extensions(x: &Value, nu: &Value) -> BuiltinResult<()> {
    if super::is_typed_integer_value(x) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_X_EXTENSION, "tcdf")?;
    }
    if super::is_typed_integer_value(nu) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_NU_EXTENSION, "tcdf")?;
    }
    if super::is_logical_value(x) || super::is_logical_value(nu) {
        crate::compatibility::ensure_builtin_extension_enabled(&LOGICAL_INPUT_EXTENSION, "tcdf")?;
    }
    Ok(())
}
