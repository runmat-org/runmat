use super::*;

pub(crate) async fn evaluate_bitcmp(args: Vec<Value>) -> BuiltinResult<Value> {
    let output_source = args.first().and_then(|value| match value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    });
    enforce_direct_bit_gpu_compatibility(DirectBitOperation::Complement, &args)?;
    let (value, assumed) = unary_args(BITCMP_NAME, args)?;
    if let Value::SparseTensor(sparse) = value {
        return sparse_bitcmp(sparse, assumed);
    }
    let input = bit_buffer_from(BITCMP_NAME, value, assumed).await?;
    let mask = input.compute_class.map_or(u64::MAX, IntegerClass::bit_mask);
    let result = value_from_bits_with_classes(
        input.data.into_iter().map(|bits| !bits & mask).collect(),
        input.shape,
        input.compute_class,
        input.output_class,
        BITCMP_NAME,
    )?;
    restore_binary_bitwise_gpu_result(BITCMP_NAME, result, output_source.as_ref())
}

fn sparse_bitcmp(
    sparse: runmat_value::SparseTensor,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<Value> {
    if sparse.integer_storage().is_some() {
        return Err(error_with_detail(
            BITCMP_NAME,
            &ERROR_INVALID_INPUT,
            "typed sparse integer storage is a RunMat extension and is not supported by bitcmp",
        ));
    }
    let class = assumed;
    let mask = class.map_or(u64::MAX, IntegerClass::bit_mask);
    map_sparse_real_values(&sparse, BITCMP_NAME, |value| {
        let bits = double_to_bits(BITCMP_NAME, value, class)?;
        let result = !bits & mask;
        Ok(match class {
            Some(class) => class.value_from_bits(result).to_i128() as f64,
            None => result as f64,
        })
    })
}

pub(crate) async fn evaluate_bitget(args: Vec<Value>) -> BuiltinResult<Value> {
    let output_source = args.first().and_then(|value| match value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    });
    enforce_direct_bit_gpu_compatibility(DirectBitOperation::Get, &args)?;
    let (value, bit, assumed) = value_bit_args(BITGET_NAME, args)?;
    if let Value::SparseTensor(sparse) = value {
        return sparse_bitget(sparse, bit, assumed).await;
    }
    let input = bit_buffer_from(BITGET_NAME, value, assumed).await?;
    let positions = shift_buffer_from(bit).await?;
    let plan = scalar_or_exact_size_plan(&input.shape, &positions.shape)
        .map_err(|err| error_with_detail(BITGET_NAME, &ERROR_SIZE_MISMATCH, err))?;
    let width = input.compute_class.map_or(64, IntegerClass::bit_width);
    let mut data = Vec::with_capacity(plan.len());
    for (_, input_index, bit_index) in plan.iter() {
        let position = positions.data[bit_index];
        if !(1..=i128::from(width)).contains(&position) {
            return Err(error_with_detail(
                BITGET_NAME,
                &ERROR_INVALID_INPUT,
                format!("bit position {position} must be between 1 and {width}"),
            ));
        }
        data.push((input.data[input_index] >> (position as u32 - 1)) & 1);
    }
    let result = value_from_bits_with_classes(
        data,
        plan.output_shape().to_vec(),
        input.compute_class,
        input.output_class,
        BITGET_NAME,
    )?;
    restore_binary_bitwise_gpu_result(BITGET_NAME, result, output_source.as_ref())
}

async fn sparse_bitget(
    sparse: runmat_value::SparseTensor,
    bit: Value,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<Value> {
    if sparse.integer_storage().is_some() {
        return Err(error_with_detail(
            BITGET_NAME,
            &ERROR_INVALID_INPUT,
            "typed sparse integer storage is a RunMat extension and is not supported by bitget",
        ));
    }
    let positions = shift_buffer_from(bit).await?;
    let class = assumed;
    let width = class.map_or(64, IntegerClass::bit_width);
    let sparse_shape = sparse.shape();
    let plan = scalar_or_exact_size_plan(&sparse_shape, &positions.shape)
        .map_err(|err| error_with_detail(BITGET_NAME, &ERROR_SIZE_MISMATCH, err))?;
    let output_shape = plan.output_shape().to_vec();
    checked_sparse_result_len(&output_shape, BITGET_NAME)?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, sparse_index, bit_index) in plan.iter() {
        let position = positions.data[bit_index];
        if !(1..=i128::from(width)).contains(&position) {
            return Err(error_with_detail(
                BITGET_NAME,
                &ERROR_INVALID_INPUT,
                format!("bit position {position} must be between 1 and {width}"),
            ));
        }
        let row = sparse_index % sparse.rows;
        let col = sparse_index / sparse.rows;
        let bits = sparse
            .get(row, col)
            .map(|value| double_to_bits(BITGET_NAME, value, class))
            .transpose()?
            .unwrap_or(0);
        data.push((bits >> (position as u32 - 1)) & 1);
    }
    sparse_or_full_from_bits(
        data,
        output_shape.clone(),
        class,
        None,
        BITGET_NAME,
        output_shape.len() == 2,
    )
}

pub(crate) async fn evaluate_bitset(args: Vec<Value>) -> BuiltinResult<Value> {
    let output_source = args.first().and_then(|value| match value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    });
    enforce_direct_bit_gpu_compatibility(DirectBitOperation::Set, &args)?;
    let (value, bit, value_to_set, assumed) = bitset_args(args)?;
    if let Value::SparseTensor(sparse) = value {
        return sparse_bitset(sparse, bit, value_to_set, assumed).await;
    }
    let input = bit_buffer_from(BITSET_NAME, value, assumed).await?;
    let positions = shift_buffer_from(bit).await?;
    let values = match value_to_set {
        Some(value) => bit_value_buffer_from(value).await?,
        None => BitValueBuffer {
            data: vec![true],
            shape: vec![1, 1],
        },
    };
    let input_positions = scalar_or_exact_size_plan(&input.shape, &positions.shape)
        .map_err(|err| error_with_detail(BITSET_NAME, &ERROR_SIZE_MISMATCH, err))?;
    let input_position_indices = input_positions
        .iter()
        .map(|(_, input_index, position_index)| (input_index, position_index))
        .collect::<Vec<_>>();
    let plan = scalar_or_exact_size_plan(input_positions.output_shape(), &values.shape)
        .map_err(|err| error_with_detail(BITSET_NAME, &ERROR_SIZE_MISMATCH, err))?;
    let width = input.compute_class.map_or(64, IntegerClass::bit_width);
    let mut data = Vec::with_capacity(plan.len());
    for (_, input_position_index, value_index) in plan.iter() {
        let (input_index, position_index) = input_position_indices[input_position_index];
        let position = positions.data[position_index];
        if !(1..=i128::from(width)).contains(&position) {
            return Err(error_with_detail(
                BITSET_NAME,
                &ERROR_INVALID_INPUT,
                format!("bit position {position} must be between 1 and {width}"),
            ));
        }
        let mask = 1_u64 << (position as u32 - 1);
        let current = input.data[input_index];
        data.push(if values.data[value_index] {
            current | mask
        } else {
            current & !mask
        });
    }
    let result = value_from_bits_with_classes(
        data,
        plan.output_shape().to_vec(),
        input.compute_class,
        input.output_class,
        BITSET_NAME,
    )?;
    restore_binary_bitwise_gpu_result(BITSET_NAME, result, output_source.as_ref())
}

async fn sparse_bitset(
    sparse: runmat_value::SparseTensor,
    bit: Value,
    value_to_set: Option<Value>,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<Value> {
    if sparse.integer_storage().is_some() {
        return Err(error_with_detail(
            BITSET_NAME,
            &ERROR_INVALID_INPUT,
            "typed sparse integer storage is a RunMat extension and is not supported by bitset",
        ));
    }
    let positions = shift_buffer_from(bit).await?;
    let values = match value_to_set {
        Some(value) => bit_value_buffer_from(value).await?,
        None => BitValueBuffer {
            data: vec![true],
            shape: vec![1, 1],
        },
    };
    let class = assumed;
    let width = class.map_or(64, IntegerClass::bit_width);
    let sparse_shape = sparse.shape();
    let input_positions = scalar_or_exact_size_plan(&sparse_shape, &positions.shape)
        .map_err(|err| error_with_detail(BITSET_NAME, &ERROR_SIZE_MISMATCH, err))?;
    checked_sparse_result_len(input_positions.output_shape(), BITSET_NAME)?;
    let input_position_indices = input_positions
        .iter()
        .map(|(_, sparse_index, position_index)| (sparse_index, position_index))
        .collect::<Vec<_>>();
    let plan = scalar_or_exact_size_plan(input_positions.output_shape(), &values.shape)
        .map_err(|err| error_with_detail(BITSET_NAME, &ERROR_SIZE_MISMATCH, err))?;
    let output_shape = plan.output_shape().to_vec();
    checked_sparse_result_len(&output_shape, BITSET_NAME)?;
    let mut data = Vec::with_capacity(plan.len());
    let mut implicit_result_nonzero = false;
    for (_, input_position_index, value_index) in plan.iter() {
        let (sparse_index, position_index) = input_position_indices[input_position_index];
        let position = positions.data[position_index];
        if !(1..=i128::from(width)).contains(&position) {
            return Err(error_with_detail(
                BITSET_NAME,
                &ERROR_INVALID_INPUT,
                format!("bit position {position} must be between 1 and {width}"),
            ));
        }
        let row = sparse_index % sparse.rows;
        let col = sparse_index / sparse.rows;
        let stored = sparse.get(row, col);
        let current = stored
            .map(|value| double_to_bits(BITSET_NAME, value, class))
            .transpose()?
            .unwrap_or(0);
        let mask = 1_u64 << (position as u32 - 1);
        let result = if values.data[value_index] {
            current | mask
        } else {
            current & !mask
        };
        implicit_result_nonzero |= stored.is_none() && result != 0;
        data.push(result);
    }
    sparse_or_full_from_bits(
        data,
        output_shape.clone(),
        class,
        None,
        BITSET_NAME,
        output_shape.len() == 2 && !implicit_result_nonzero,
    )
}

#[derive(Clone, Copy)]
enum DirectBitOperation {
    Complement,
    Get,
    Set,
}

impl DirectBitOperation {
    const fn name(self) -> &'static str {
        match self {
            Self::Complement => BITCMP_NAME,
            Self::Get => BITGET_NAME,
            Self::Set => BITSET_NAME,
        }
    }

    const fn assumed_type_arity(self) -> usize {
        match self {
            Self::Complement => 2,
            Self::Get | Self::Set => 3,
        }
    }

    const fn restricts_integer_classes(self) -> bool {
        matches!(self, Self::Complement)
    }
}

fn enforce_direct_bit_gpu_compatibility(
    operation: DirectBitOperation,
    args: &[Value],
) -> BuiltinResult<()> {
    let name = operation.name();
    let Some(Value::GpuTensor(source)) = args.first() else {
        return Ok(());
    };
    if args.len() == operation.assumed_type_arity() && matches!(args.last(), Some(Value::String(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::DIRECT_BIT_GPU_ASSUMED_TYPE_EXTENSION,
            name,
        )?;
    }
    let source_class = runmat_accelerate_api::handle_integer_type(source);
    let outside_domain = if operation.restricts_integer_classes() {
        !matches!(
            source_class,
            Some(
                runmat_accelerate_api::IntegerElementType::U8
                    | runmat_accelerate_api::IntegerElementType::U16
                    | runmat_accelerate_api::IntegerElementType::U32
            )
        )
    } else {
        source_class.is_none()
            && !args
                .iter()
                .skip(1)
                .any(|value| bitwise_integer_class(value).is_some())
    };
    if outside_domain {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::DIRECT_BIT_GPU_UNDOCUMENTED_INPUT_EXTENSION,
            name,
        )?;
    }
    Ok(())
}
