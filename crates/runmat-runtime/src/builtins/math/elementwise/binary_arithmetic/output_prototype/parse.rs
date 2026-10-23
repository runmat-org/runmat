use crate::builtins::common::random_args::keyword_of;
use crate::BuiltinResult;

use super::{OutputPrototypeContext, OutputTemplate};

pub(in crate::builtins::math::elementwise::binary_arithmetic) fn parse_output_template(
    context: OutputPrototypeContext,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<OutputTemplate> {
    match (args.len(), args.get(0), args.get(1)) {
        (0, _, _) => Ok(OutputTemplate::Default),
        (1, Some(option), _) if matches!(keyword_of(option)?.as_deref(), Some("like")) => {
            Err(context
                .described_error(context.invalid_argument, "expected prototype after 'like'"))
        }
        (1, _, _) => Err(context.described_error(
            context.invalid_argument,
            "unsupported option; only 'like' is accepted",
        )),
        (2, Some(option), Some(prototype))
            if matches!(keyword_of(option)?.as_deref(), Some("like")) =>
        {
            Ok(OutputTemplate::Like(prototype.clone()))
        }
        (2, _, _) => Err(context.described_error(
            context.invalid_argument,
            "unsupported option; only 'like' is accepted",
        )),
        _ => Err(context.described_error(context.invalid_argument, "too many input arguments")),
    }
}
