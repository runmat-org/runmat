use runmat_value::Value;

use crate::builtins::common::random_args::keyword_of;
use crate::BuiltinResult;

use super::{OutputPrototypeContext, OutputTemplate};

pub(in crate::builtins::math::elementwise::binary_arithmetic) fn parse_output_template(
    context: OutputPrototypeContext,
    args: &[Value],
) -> BuiltinResult<OutputTemplate> {
    match args {
        [] => Ok(OutputTemplate::Default),
        [option] if matches!(keyword_of(option).as_deref(), Some("like")) => {
            Err(context
                .described_error(context.invalid_argument, "expected prototype after 'like'"))
        }
        [_] => Err(context.described_error(
            context.invalid_argument,
            "unsupported option; only 'like' is accepted",
        )),
        [option, prototype] if matches!(keyword_of(option).as_deref(), Some("like")) => {
            Ok(OutputTemplate::Like(prototype.clone()))
        }
        [_, _] => Err(context.described_error(
            context.invalid_argument,
            "unsupported option; only 'like' is accepted",
        )),
        _ => Err(context.described_error(context.invalid_argument, "too many input arguments")),
    }
}
