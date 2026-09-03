use super::*;

pub(crate) async fn evaluate_bitshift(args: Vec<Value>) -> BuiltinResult<Value> {
    enforce_bitshift_compatibility(&args)?;
    let output_source = args.iter().find_map(|value| {
        let Value::GpuTensor(handle) = value else {
            return None;
        };
        Some(handle.clone())
    });
    let (value, shift, assumed) = value_bit_args(BITSHIFT_NAME, args)?;
    if let Value::SparseTensor(sparse) = value {
        return sparse_bitshift(sparse, shift, assumed).await;
    }
    let left = bit_buffer_from(BITSHIFT_NAME, value, assumed).await?;
    let shifts = shift_buffer_from(shift).await?;
    let plan = scalar_or_exact_size_plan(&left.shape, &shifts.shape)
        .map_err(|err| error_with_detail(BITSHIFT_NAME, &ERROR_SIZE_MISMATCH, err))?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, idx_a, idx_b) in plan.iter() {
        data.push(apply_shift(
            left.data[idx_a],
            shifts.data[idx_b],
            left.compute_class,
        ));
    }
    let result = value_from_bits_with_classes(
        data,
        plan.output_shape().to_vec(),
        left.compute_class,
        left.output_class,
        BITSHIFT_NAME,
    )?;
    restore_binary_bitwise_gpu_result(BITSHIFT_NAME, result, output_source.as_ref())
}

fn enforce_bitshift_compatibility(args: &[Value]) -> BuiltinResult<()> {
    if !(2..=3).contains(&args.len()) {
        return Ok(());
    }
    if is_single_bitwise_value(&args[0]) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::BITSHIFT_SINGLE_VALUE_EXTENSION,
            BITSHIFT_NAME,
        )?;
    }
    if is_single_bitwise_value(&args[1]) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::BITSHIFT_SINGLE_COUNT_EXTENSION,
            BITSHIFT_NAME,
        )?;
    }
    if is_logical_bitwise_value(&args[0]) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::BITSHIFT_LOGICAL_VALUE_EXTENSION,
            BITSHIFT_NAME,
        )?;
    }
    if is_logical_bitwise_value(&args[1]) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::BITSHIFT_LOGICAL_COUNT_EXTENSION,
            BITSHIFT_NAME,
        )?;
    }
    let has_gpu_input = args
        .iter()
        .take(2)
        .any(|value| matches!(value, Value::GpuTensor(_)));
    if has_gpu_input && args.len() == 3 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::BITSHIFT_GPU_ASSUMED_TYPE_EXTENSION,
            BITSHIFT_NAME,
        )?;
    }
    if has_gpu_input {
        let value_class = bitwise_integer_class(&args[0]);
        let count_class = bitwise_integer_class(&args[1]);
        let outside_public_gpu_domain = value_class.is_some_and(IntegerClass::is_signed)
            || value_class.is_some_and(|class| class.bit_width() == 64)
            || count_class.is_some_and(|class| class.bit_width() == 64)
            || (value_class.is_none() && count_class.is_none())
            || args
                .iter()
                .take(2)
                .any(|value| matches!(value, Value::SparseTensor(_)))
            || args.iter().take(2).any(is_single_bitwise_value)
            || args.iter().take(2).any(is_logical_bitwise_value);
        if outside_public_gpu_domain {
            crate::compatibility::ensure_builtin_extension_enabled(
                &runmat_builtins::BITSHIFT_GPU_UNDOCUMENTED_INPUT_EXTENSION,
                BITSHIFT_NAME,
            )?;
        }
    }
    Ok(())
}

async fn sparse_bitshift(
    sparse: runmat_value::SparseTensor,
    shift: Value,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<Value> {
    if sparse.integer_storage().is_some() {
        return Err(error_with_detail(
            BITSHIFT_NAME,
            &ERROR_INVALID_INPUT,
            "typed sparse integer storage is a RunMat extension and is not supported by bitshift",
        ));
    }
    let shifts = shift_buffer_from(shift).await?;
    let class = assumed;
    let sparse_shape = sparse.shape();
    let plan = scalar_or_exact_size_plan(&sparse_shape, &shifts.shape)
        .map_err(|err| error_with_detail(BITSHIFT_NAME, &ERROR_SIZE_MISMATCH, err))?;
    let output_shape = plan.output_shape().to_vec();
    checked_sparse_result_len(&output_shape, BITSHIFT_NAME)?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, sparse_index, shift_index) in plan.iter() {
        let row = sparse_index % sparse.rows;
        let col = sparse_index / sparse.rows;
        let bits = sparse
            .get(row, col)
            .map(|value| double_to_bits(BITSHIFT_NAME, value, class))
            .transpose()?
            .unwrap_or(0);
        data.push(apply_shift(bits, shifts.data[shift_index], class));
    }
    sparse_or_full_from_bits(
        data,
        output_shape.clone(),
        class,
        None,
        BITSHIFT_NAME,
        output_shape.len() == 2,
    )
}
