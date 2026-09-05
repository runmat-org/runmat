use super::double_error_with_detail;
use super::residency::{convert_to_gpu, convert_to_host_like};
use crate::builtins::common::random_args::keyword_of;
use crate::BuiltinResult;
use runmat_builtins::{DOUBLE_ERROR_INVALID_ARGUMENT, DOUBLE_ERROR_INVALID_INPUT};
use runmat_value::Value;

#[derive(Clone)]
pub(super) enum OutputTemplate {
    Default,
    Like(Value),
}

pub(super) fn parse_output_template(args: &[Value]) -> BuiltinResult<OutputTemplate> {
    match args.len() {
        0 => Ok(OutputTemplate::Default),
        1 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Err(double_error_with_detail(
                    &DOUBLE_ERROR_INVALID_ARGUMENT,
                    "expected prototype after 'like'",
                ))
            } else {
                Err(double_error_with_detail(
                    &DOUBLE_ERROR_INVALID_ARGUMENT,
                    "unrecognised argument for double",
                ))
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err(double_error_with_detail(
                    &DOUBLE_ERROR_INVALID_ARGUMENT,
                    "unsupported option; only 'like' is accepted",
                ))
            }
        }
        _ => Err(double_error_with_detail(
            &DOUBLE_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        )),
    }
}

pub(super) async fn apply_output_template(
    value: Value,
    template: &OutputTemplate,
) -> BuiltinResult<Value> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(proto) => match proto {
            Value::GpuTensor(prototype) => convert_to_gpu(value, prototype).await,
            Value::Tensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::LogicalArray(_) => convert_to_host_like(value).await,
            Value::Complex(_, _) | Value::ComplexTensor(_) => Err(double_error_with_detail(
                &DOUBLE_ERROR_INVALID_INPUT,
                "complex prototypes for 'like' are not supported yet",
            )),
            _ => Err(double_error_with_detail(
                &DOUBLE_ERROR_INVALID_INPUT,
                "unsupported prototype for 'like'; provide a numeric or gpuArray prototype",
            )),
        },
    }
}
