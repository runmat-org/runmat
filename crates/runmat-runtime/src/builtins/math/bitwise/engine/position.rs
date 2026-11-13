use super::*;

fn validate_position(name: &'static str, width: u32, position: i128) -> BuiltinResult<()> {
    if !(1..=i128::from(width)).contains(&position) {
        return Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            format!("bit position {position} must be between 1 and {width}"),
        ));
    }
    Ok(())
}

fn admit_host_positions(
    name: &'static str,
    input: &AdmittedBuffer<BitBuffer>,
    positions: &AdmittedBuffer<ShiftBuffer>,
    assumed: Option<IntegerClass>,
    output_len: usize,
) -> BuiltinResult<()> {
    // Existing empty-output calls do not inspect unused position contents.
    if output_len != 0 {
        if let Some(positions) = positions.host() {
            for &position in &positions.data {
                validate_position(name, input.bit_width(assumed), position)?;
            }
        }
    }
    Ok(())
}

pub(crate) async fn evaluate_bitcmp(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let protected = borrowed_gpu_inputs(args.view());
    let output_source = args.first().and_then(|value| match value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    });
    let (value, assumed) = unary_args(BITCMP_NAME, args)?;
    enforce_direct_bit_gpu_compatibility(DirectBitOperation::Complement, &value, &[], assumed)?;
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
    restore_binary_bitwise_gpu_result(BITCMP_NAME, result, output_source.as_ref(), &protected)
}

fn sparse_bitcmp(
    sparse: runmat_value::SparseTensor,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<Value> {
    admit_sparse(BITCMP_NAME, &sparse)?;
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

pub(crate) async fn evaluate_bitget(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let protected = borrowed_gpu_inputs(args.view());
    let output_source = args.first().and_then(|value| match value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    });
    let (value, bit, assumed) = value_bit_args(BITGET_NAME, args)?;
    enforce_direct_bit_gpu_compatibility(DirectBitOperation::Get, &value, &[&bit], assumed)?;
    if let Value::SparseTensor(sparse) = value {
        return sparse_bitget(sparse, bit, assumed).await;
    }
    let input = admit_bit_buffer(BITGET_NAME, value, assumed)?;
    let positions = admit_shift_buffer(bit)?;
    let plan = scalar_or_exact_size_plan(
        input.shape(|value| &value.shape),
        positions.shape(|value| &value.shape),
    )
    .map_err(|err| error_with_detail(BITGET_NAME, &ERROR_SIZE_MISMATCH, err))?;
    admit_host_positions(BITGET_NAME, &input, &positions, assumed, plan.len())?;
    let input = input
        .materialize(BITGET_NAME, |value| {
            tensor_to_bit_buffer(BITGET_NAME, value, assumed)
        })
        .await?;
    let positions = positions
        .materialize(BITGET_NAME, tensor_to_shift_buffer)
        .await?;
    let width = input.compute_class.map_or(64, IntegerClass::bit_width);
    let mut data = Vec::with_capacity(plan.len());
    for (_, input_index, bit_index) in plan.iter() {
        let position = positions.data[bit_index];
        validate_position(BITGET_NAME, width, position)?;
        data.push((input.data[input_index] >> (position as u32 - 1)) & 1);
    }
    let result = value_from_bits_with_classes(
        data,
        plan.output_shape().to_vec(),
        input.compute_class,
        input.output_class,
        BITGET_NAME,
    )?;
    restore_binary_bitwise_gpu_result(BITGET_NAME, result, output_source.as_ref(), &protected)
}

async fn sparse_bitget(
    sparse: runmat_value::SparseTensor,
    bit: Value,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<Value> {
    admit_sparse(BITGET_NAME, &sparse)?;
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
        validate_position(BITGET_NAME, width, position)?;
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

pub(crate) async fn evaluate_bitset(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let protected = borrowed_gpu_inputs(args.view());
    let output_source = args.first().and_then(|value| match value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    });
    let (value, bit, value_to_set, assumed) = bitset_args(args)?;
    let mut controls = vec![&bit];
    controls.extend(value_to_set.as_ref());
    enforce_direct_bit_gpu_compatibility(DirectBitOperation::Set, &value, &controls, assumed)?;
    if let Value::SparseTensor(sparse) = value {
        return sparse_bitset(sparse, bit, value_to_set, assumed).await;
    }
    let input = admit_bit_buffer(BITSET_NAME, value, assumed)?;
    let positions = admit_shift_buffer(bit)?;
    let values = match value_to_set {
        Some(value) => admit_bit_value_buffer(value)?,
        None => AdmittedBuffer::Host(BitValueBuffer {
            data: vec![true],
            shape: vec![1, 1],
        }),
    };
    let input_positions = scalar_or_exact_size_plan(
        input.shape(|value| &value.shape),
        positions.shape(|value| &value.shape),
    )
    .map_err(|err| error_with_detail(BITSET_NAME, &ERROR_SIZE_MISMATCH, err))?;
    let input_position_indices = input_positions
        .iter()
        .map(|(_, input_index, position_index)| (input_index, position_index))
        .collect::<Vec<_>>();
    let plan = scalar_or_exact_size_plan(
        input_positions.output_shape(),
        values.shape(|value| &value.shape),
    )
    .map_err(|err| error_with_detail(BITSET_NAME, &ERROR_SIZE_MISMATCH, err))?;
    admit_host_positions(BITSET_NAME, &input, &positions, assumed, plan.len())?;
    let input = input
        .materialize(BITSET_NAME, |value| {
            tensor_to_bit_buffer(BITSET_NAME, value, assumed)
        })
        .await?;
    let positions = positions
        .materialize(BITSET_NAME, tensor_to_shift_buffer)
        .await?;
    let values = values
        .materialize(BITSET_NAME, tensor_to_bit_value_buffer)
        .await?;
    let width = input.compute_class.map_or(64, IntegerClass::bit_width);
    let mut data = Vec::with_capacity(plan.len());
    for (_, input_position_index, value_index) in plan.iter() {
        let (input_index, position_index) = input_position_indices[input_position_index];
        let position = positions.data[position_index];
        validate_position(BITSET_NAME, width, position)?;
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
    restore_binary_bitwise_gpu_result(BITSET_NAME, result, output_source.as_ref(), &protected)
}

async fn sparse_bitset(
    sparse: runmat_value::SparseTensor,
    bit: Value,
    value_to_set: Option<Value>,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<Value> {
    admit_sparse(BITSET_NAME, &sparse)?;
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
        validate_position(BITSET_NAME, width, position)?;
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

    const fn restricts_integer_classes(self) -> bool {
        matches!(self, Self::Complement)
    }
}

fn enforce_direct_bit_gpu_compatibility(
    operation: DirectBitOperation,
    value: &Value,
    controls: &[&Value],
    assumed: Option<IntegerClass>,
) -> BuiltinResult<()> {
    let name = operation.name();
    let Value::GpuTensor(source) = value else {
        return Ok(());
    };
    if assumed.is_some() {
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
            && !controls
                .iter()
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
