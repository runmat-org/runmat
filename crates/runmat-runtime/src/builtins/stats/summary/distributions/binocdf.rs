use super::*;
use runmat_builtins::{
    BINOCDF_INTEGER_N_EXTENSION as INTEGER_N_EXTENSION,
    BINOCDF_INTEGER_P_EXTENSION as INTEGER_P_EXTENSION,
    BINOCDF_INTEGER_X_EXTENSION as INTEGER_X_EXTENSION,
    BINOCDF_LOGICAL_INPUT_EXTENSION as LOGICAL_INPUT_EXTENSION,
};

#[runtime_builtin(
    name = "binocdf",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::distributions::binocdf"
)]
pub(crate) async fn binocdf_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args = parse_args(value, rest).await?;
    let data = args
        .x
        .iter()
        .zip(args.a.iter())
        .zip(args.b.iter())
        .map(|((x, n), p)| super::binocdf_scalar(*x, *n, *p, args.tail))
        .collect();
    args.output.finish("binocdf", args.shape, data)
}

struct BinocdfArgs {
    x: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    shape: Vec<usize>,
    tail: DistributionTail,
    output: NormalOutputPlan,
}

async fn parse_args(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<BinocdfArgs> {
    let (rest, tail) = super::split_upper_tail(rest)?;
    if rest.len() != 2 {
        return Err(super::normal_error(
            "binocdf",
            "binocdf: expected x, n, and p arguments",
        ));
    }
    let inputs = [&value, &rest[0], &rest[1]];
    ensure_extensions(inputs)?;
    numeric_inputs("binocdf", &inputs, NumericShapePolicy::Broadcast)?;
    let output = NormalOutputPlan::inspect("binocdf", &value, rest.view())?;
    let x = super::normal_value_to_tensor("binocdf", "x", value).await?;
    let n = super::normal_value_to_tensor("binocdf", "n", rest[0].clone()).await?;
    let p = super::normal_value_to_tensor("binocdf", "p", rest[1].clone()).await?;
    let (mut broadcasted, shape) = super::broadcast_tensors("binocdf", &[&x, &n, &p])?;
    Ok(BinocdfArgs {
        x: broadcasted.remove(0),
        a: broadcasted.remove(0),
        b: broadcasted.remove(0),
        shape,
        tail,
        output,
    })
}

fn ensure_extensions(inputs: [&Value; 3]) -> BuiltinResult<()> {
    for (value, extension) in inputs.into_iter().zip([
        &INTEGER_X_EXTENSION,
        &INTEGER_N_EXTENSION,
        &INTEGER_P_EXTENSION,
    ]) {
        if super::is_typed_integer_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(extension, "binocdf")?;
        }
    }
    if inputs.into_iter().any(super::is_logical_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOGICAL_INPUT_EXTENSION,
            "binocdf",
        )?;
    }
    Ok(())
}
