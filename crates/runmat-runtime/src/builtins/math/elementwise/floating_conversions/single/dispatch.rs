use runmat_builtins::SINGLE_LIKE_OUTPUT_EXTENSION;
use runmat_value::Value;

use super::{execution, parse_output_template, OutputTemplate, CONVERSION};
use crate::BuiltinResult;

pub(super) async fn execute(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        runmat_builtins::SINGLE_CATALOG_ENTRY.descriptor,
        &runmat_builtins::SINGLE_ERROR_INVALID_ARGUMENT,
        CONVERSION.name(),
    )?;
    let template = parse_output_template(&rest)?;
    if !rest.is_empty() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SINGLE_LIKE_OUTPUT_EXTENSION,
            CONVERSION.name(),
        )?;
    }
    template.validate()?;
    template.validate_input(&value)?;
    let borrowed: Vec<_> = std::iter::once(&value)
        .chain(rest.iter())
        .filter_map(|value| {
            if let Value::GpuTensor(handle) = value {
                Some(handle.clone())
            } else {
                None
            }
        })
        .collect();
    let signature = usize::from(!rest.is_empty());
    let input_count = 1 + rest.len();
    super::super::preparation::execute(
        runmat_builtins::SINGLE_CATALOG_ENTRY.identity,
        value,
        signature,
        input_count,
        runmat_builtins::BuiltinPreparationFormId(if matches!(template, OutputTemplate::Default) {
            1
        } else {
            2
        }),
        |value| execution::convert(value, template, borrowed),
    )
    .await
}
