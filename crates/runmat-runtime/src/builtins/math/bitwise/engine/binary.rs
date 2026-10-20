use super::*;

pub(crate) async fn evaluate_binary_bitwise(
    name: &'static str,
    args: crate::call::BuiltinArguments,
    operator: BinaryBitwiseOperator,
    single_extension: &BuiltinExtensionDescriptor,
    gpu_undocumented_input_extension: &BuiltinExtensionDescriptor,
    gpu_assumed_type_extension: &BuiltinExtensionDescriptor,
) -> BuiltinResult<Value> {
    let protected = borrowed_gpu_inputs(args.view());
    enforce_public_binary_bitwise_compatibility(
        name,
        args.view(),
        single_extension,
        gpu_undocumented_input_extension,
        gpu_assumed_type_extension,
    )?;
    let output_source = args.iter().find_map(|value| {
        let Value::GpuTensor(handle) = value else {
            return None;
        };
        Some(handle.clone())
    });
    let logical_output = args.len() >= 2 && args.iter().take(2).all(is_logical_bitwise_value);
    let result = binary_bitwise_from_args(name, args, operator).await?;
    let result = if logical_output {
        bitwise_result_as_logical(name, result)?
    } else {
        result
    };
    restore_binary_bitwise_gpu_result(name, result, output_source.as_ref(), &protected)
}

fn enforce_public_binary_bitwise_compatibility(
    name: &str,
    args: crate::call::FunctionArgumentView<'_>,
    single_extension: &BuiltinExtensionDescriptor,
    gpu_undocumented_input_extension: &BuiltinExtensionDescriptor,
    gpu_assumed_type_extension: &BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    if !(2..=3).contains(&args.len()) {
        return Ok(());
    }
    if args.iter().take(2).any(is_single_bitwise_value) {
        crate::compatibility::ensure_builtin_extension_enabled(single_extension, name)?;
    }
    let has_gpu_input = args
        .iter()
        .take(2)
        .any(|value| matches!(value, Value::GpuTensor(_)));
    if has_gpu_input && args.len() == 3 {
        crate::compatibility::ensure_builtin_extension_enabled(gpu_assumed_type_extension, name)?;
    }
    if args.iter().take(2).any(|value| {
        matches!(value, Value::GpuTensor(handle) if !matches!(
            runmat_accelerate_api::handle_integer_type(handle),
            Some(
                runmat_accelerate_api::IntegerElementType::U8
                    | runmat_accelerate_api::IntegerElementType::U16
                    | runmat_accelerate_api::IntegerElementType::U32
            )
        ))
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(
            gpu_undocumented_input_extension,
            name,
        )?;
    }
    Ok(())
}
