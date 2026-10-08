use super::nanmax_contract::{nanmax_type, NANMAX_INTEGER_EXTENSION};
use super::*;
use runmat_macros::runtime_builtin;
use runmat_value::Tensor;

#[runtime_builtin(
    name = "nanmax",
    category = "stats/summary",
    summary = "Return maximum values while omitting NaNs.",
    keywords = "nanmax,max,maximum,omitnan,statistics",
    type_resolver(nanmax_type),
    descriptor(crate::builtins::math::reduction::max::NANMAX_DESCRIPTOR),
    extensions(crate::builtins::math::reduction::max::NANMAX_EXTENSIONS),
    integer_capabilities(crate::builtins::math::reduction::max::NANMAX_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::reduction::max"
)]
pub(super) async fn nanmax_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if is_real_typed_integer_value(&value) || rest.iter().any(is_real_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &NANMAX_INTEGER_EXTENSION,
            "nanmax",
        )?;
    }
    let adjusted = nanmax_rest(rest);
    reject_pairwise_multiple_outputs(&adjusted)?;
    let eval = evaluate_for_demand(value, &adjusted, builtin_demand()).await?;
    evaluation_to_value(eval)
}

fn is_real_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(
            value,
            Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_integer_type(handle).is_some()
        )
}

fn nanmax_rest(rest: Vec<Value>) -> Vec<Value> {
    let mut args = Vec::with_capacity(rest.len() + 2);
    if rest.is_empty() {
        args.push(empty_placeholder());
    }
    args.extend(rest);
    args.push(Value::from("omitnan"));
    args
}

fn empty_placeholder() -> Value {
    Value::Tensor(Tensor::new(Vec::<f64>::new(), vec![0, 0]).expect("empty placeholder shape"))
}
