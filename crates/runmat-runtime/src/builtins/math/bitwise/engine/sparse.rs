use super::*;

pub(super) async fn binary_bitwise(
    name: &'static str,
    lhs: Value,
    rhs: Value,
    assumed: Option<IntegerClass>,
    operator: BinaryBitwiseOperator,
) -> BuiltinResult<Value> {
    if matches!(&lhs, Value::SparseTensor(_)) || matches!(&rhs, Value::SparseTensor(_)) {
        return sparse_binary_bitwise(name, lhs, rhs, assumed, operator).await;
    }
    let left = bit_buffer_from(name, lhs, assumed).await?;
    let right = bit_buffer_from(name, rhs, assumed).await?;
    let output_class = binary_output_class(name, &left, &right)?;
    let compute_class = assumed.or(output_class);
    let plan = BroadcastPlan::new(&left.shape, &right.shape)
        .map_err(|err| error_with_detail(name, &ERROR_SIZE_MISMATCH, err))?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, idx_a, idx_b) in plan.iter() {
        data.push(apply_binary_operator(
            operator,
            left.data[idx_a],
            right.data[idx_b],
        ));
    }
    value_from_bits_with_classes(
        data,
        plan.output_shape().to_vec(),
        compute_class,
        output_class,
        name,
    )
}

enum SparseBitwiseOperand {
    Sparse(runmat_value::SparseTensor),
    Dense(BitBuffer),
}

impl SparseBitwiseOperand {
    fn metadata(&self) -> BitBuffer {
        match self {
            Self::Sparse(sparse) => BitBuffer {
                data: Vec::new(),
                shape: sparse.shape(),
                compute_class: None,
                output_class: None,
                is_scalar: sparse.rows == 1 && sparse.cols == 1,
            },
            Self::Dense(buffer) => BitBuffer {
                data: Vec::new(),
                shape: buffer.shape.clone(),
                compute_class: buffer.compute_class,
                output_class: buffer.output_class,
                is_scalar: buffer.is_scalar,
            },
        }
    }

    fn value_at(
        &self,
        name: &'static str,
        assumed: Option<IntegerClass>,
        index: usize,
    ) -> BuiltinResult<(u64, bool)> {
        match self {
            Self::Sparse(sparse) => {
                let row = index % sparse.rows;
                let col = index / sparse.rows;
                match sparse.get(row, col) {
                    Some(value) => Ok((double_to_bits(name, value, assumed)?, false)),
                    None => Ok((0, true)),
                }
            }
            Self::Dense(buffer) => Ok((buffer.data[index], false)),
        }
    }
}

async fn sparse_bitwise_operand_from(
    name: &'static str,
    value: Value,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<SparseBitwiseOperand> {
    match value {
        Value::SparseTensor(sparse) => {
            if sparse.integer_storage().is_some() {
                return Err(error_with_detail(
                    name,
                    &ERROR_INVALID_INPUT,
                    "typed sparse integer storage is a RunMat extension and is not supported by bitwise operations",
                ));
            }
            Ok(SparseBitwiseOperand::Sparse(sparse))
        }
        value => bit_buffer_from(name, value, assumed)
            .await
            .map(SparseBitwiseOperand::Dense),
    }
}

pub(super) async fn sparse_binary_bitwise(
    name: &'static str,
    lhs: Value,
    rhs: Value,
    assumed: Option<IntegerClass>,
    operator: BinaryBitwiseOperator,
) -> BuiltinResult<Value> {
    let left = sparse_bitwise_operand_from(name, lhs, assumed).await?;
    let right = sparse_bitwise_operand_from(name, rhs, assumed).await?;
    let left_meta = left.metadata();
    let right_meta = right.metadata();
    let output_class = binary_output_class(name, &left_meta, &right_meta)?;
    let compute_class = assumed.or(output_class);
    let plan = BroadcastPlan::new(&left_meta.shape, &right_meta.shape)
        .map_err(|err| error_with_detail(name, &ERROR_SIZE_MISMATCH, err))?;
    let output_shape = plan.output_shape().to_vec();
    checked_sparse_result_len(&output_shape, name)?;

    let can_return_sparse = output_class.is_none() && output_shape.len() == 2;
    let mut bits = Vec::with_capacity(plan.len());
    let mut implicit_result_nonzero = false;
    for (_, left_index, right_index) in plan.iter() {
        let (left_bits, left_implicit) = left.value_at(name, compute_class, left_index)?;
        let (right_bits, right_implicit) = right.value_at(name, compute_class, right_index)?;
        let result = apply_binary_operator(operator, left_bits, right_bits);
        // A sparse/sparse result can store any position present in either
        // input; only positions absent from both operands are necessarily
        // implicit. With one dense operand, every implicit sparse position
        // must stay zero or the result has to become full.
        let result_is_implicit = match (&left, &right) {
            (SparseBitwiseOperand::Sparse(_), SparseBitwiseOperand::Sparse(_)) => {
                left_implicit && right_implicit
            }
            (SparseBitwiseOperand::Sparse(_), SparseBitwiseOperand::Dense(_)) => left_implicit,
            (SparseBitwiseOperand::Dense(_), SparseBitwiseOperand::Sparse(_)) => right_implicit,
            (SparseBitwiseOperand::Dense(_), SparseBitwiseOperand::Dense(_)) => false,
        };
        implicit_result_nonzero |= result_is_implicit && result != 0;
        bits.push(result);
    }

    sparse_or_full_from_bits(
        bits,
        output_shape,
        compute_class,
        output_class,
        name,
        can_return_sparse && !implicit_result_nonzero,
    )
}

pub(super) fn sparse_or_full_from_bits(
    bits: Vec<u64>,
    output_shape: Vec<usize>,
    compute_class: Option<IntegerClass>,
    output_class: Option<IntegerClass>,
    name: &'static str,
    return_sparse: bool,
) -> BuiltinResult<Value> {
    if return_sparse {
        let rows = output_shape[0];
        let cols = output_shape[1];
        let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        col_ptrs.push(0);
        for col in 0..cols {
            for row in 0..rows {
                let value = bits[row + col * rows];
                if value != 0 {
                    row_indices.push(row);
                    values.push(bits_to_double(value, compute_class));
                }
            }
            col_ptrs.push(values.len());
        }
        return runmat_value::SparseTensor::new(rows, cols, col_ptrs, row_indices, values)
            .map(Value::SparseTensor)
            .map_err(|err| error_with_detail(name, &ERROR_INVALID_INPUT, err));
    }

    value_from_bits_with_classes(bits, output_shape, compute_class, output_class, name)
}

pub(super) fn bits_to_double(bits: u64, compute_class: Option<IntegerClass>) -> f64 {
    match compute_class {
        Some(class) => class.value_from_bits(bits).to_i128() as f64,
        None => bits as f64,
    }
}
