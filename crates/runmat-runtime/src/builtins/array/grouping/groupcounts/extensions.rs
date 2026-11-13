use runmat_value::Value;

use crate::builtins::common::tensor;
use crate::builtins::table::is_tabular_object;
use crate::BuiltinResult;

use super::options;

pub(super) fn validate(
    first: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    if crate::dispatcher::value_contains_explicit_gpu(first)
        || rest
            .iter()
            .any(crate::dispatcher::value_contains_explicit_gpu)
    {
        require(&runmat_builtins::GROUPCOUNTS_RESIDENT_INPUT_EXTENSION)?;
    }
    let start = options::option_start(rest)?;
    let positional = rest.range(..start).expect("groupcounts positional prefix");
    let bin = if matches!(first, Value::Object(object) if is_tabular_object(object)) {
        positional.get(1)
    } else {
        positional.first()
    };
    let typed_bin_count = bin.is_some_and(is_fixed_width_integer_scalar);
    let mut typed_boolean = false;
    for pair in rest
        .range(start..)
        .expect("groupcounts option suffix")
        .chunks_exact(2)
    {
        if options::is_boolean_option(&pair[0])? && is_fixed_width_integer_scalar(&pair[1]) {
            typed_boolean = true;
            break;
        }
    }
    if typed_bin_count || typed_boolean {
        require(&runmat_builtins::GROUPCOUNTS_INTEGER_CONTROL_EXTENSION)?;
    }
    Ok(())
}

fn is_fixed_width_integer_scalar(value: &Value) -> bool {
    match value {
        Value::Cell(value) if value.data().len() == 1 => {
            is_fixed_width_integer_scalar(&value.data()[0])
        }
        _ => {
            tensor::scalar_integer_value(value).is_some()
                || matches!(value, Value::GpuTensor(handle) if handle.shape().iter().product::<usize>() == 1 && runmat_accelerate_api::handle_integer_type(handle).is_some())
        }
    }
}

fn require(extension: &runmat_builtins::BuiltinExtensionDescriptor) -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(extension, "groupcounts")
}
