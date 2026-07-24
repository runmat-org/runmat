//! MATLAB-compatible `blkdiag` builtin.
//!
//! `blkdiag` constructs a block diagonal matrix by placing each input on the
//! diagonal and filling off-block entries with the class-appropriate zero.

use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexTensor, LogicalArray, NumericDType, ResolveContext, SparseTensor, Tensor,
    Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "blkdiag";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::blkdiag")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "blkdiag",
    op_kind: GpuOpKind::Custom("blkdiag"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("upload"), ProviderHook::Custom("download")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Assembles block diagonal matrices on the host; gpuArray inputs are gathered once and dense numeric, logical, or complex results are uploaded back to the active provider.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::blkdiag")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "blkdiag",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Block diagonal assembly materialises a new array and terminates fusion pipelines.",
};

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Block diagonal matrix containing each input along the diagonal.",
}];

const INPUTS_EMPTY: [BuiltinParamDescriptor; 0] = [];

const INPUTS_VARIADIC: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First scalar, vector, matrix, sparse matrix, or gpuArray block.",
    },
    BuiltinParamDescriptor {
        name: "An",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional blocks placed down and right from the previous block.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = blkdiag()",
        inputs: &INPUTS_EMPTY,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "B = blkdiag(A1, An...)",
        inputs: &INPUTS_VARIADIC,
        outputs: &OUTPUTS,
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLKDIAG.INVALID_INPUT",
    identifier: Some("RunMat:blkdiag:InvalidInput"),
    when: "An input class, dimensionality, or block conversion is unsupported.",
    message: "blkdiag: invalid input",
};

const ERROR_SIZE_OVERFLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLKDIAG.SIZE_OVERFLOW",
    identifier: Some("RunMat:blkdiag:SizeOverflow"),
    when: "The block diagonal output dimensions or allocation size exceed addressable memory.",
    message: "blkdiag: output size exceeds maximum supported size",
};

const ERROR_GPU: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLKDIAG.GPU",
    identifier: Some("RunMat:blkdiag:GpuError"),
    when: "A required gpuArray gather or upload operation fails.",
    message: "blkdiag: gpuArray operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [ERROR_INVALID_INPUT, ERROR_SIZE_OVERFLOW, ERROR_GPU];

pub const BLKDIAG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn blkdiag_type(args: &[Type], _context: &ResolveContext) -> Type {
    if args.is_empty() {
        return Type::Tensor {
            shape: Some(vec![Some(0), Some(0)]),
        };
    }

    let all_logical = args
        .iter()
        .all(|arg| matches!(arg, Type::Logical { .. } | Type::Bool));

    let mut rows: Option<usize> = Some(0);
    let mut cols: Option<usize> = Some(0);
    for arg in args {
        let Some((r, c)) = type_matrix_dims(arg) else {
            rows = None;
            cols = None;
            break;
        };
        rows = match (rows, r) {
            (Some(acc), Some(value)) => acc.checked_add(value),
            _ => None,
        };
        cols = match (cols, c) {
            (Some(acc), Some(value)) => acc.checked_add(value),
            _ => None,
        };
    }

    let shape = Some(vec![rows, cols]);
    if all_logical {
        Type::Logical { shape }
    } else {
        Type::Tensor { shape }
    }
}

fn type_matrix_dims(ty: &Type) -> Option<(Option<usize>, Option<usize>)> {
    match ty {
        Type::Num | Type::Int | Type::Bool => Some((Some(1), Some(1))),
        Type::Tensor { shape } | Type::Logical { shape } => match shape.as_deref() {
            Some([]) => Some((Some(1), Some(1))),
            Some([n]) => Some((Some(1), *n)),
            Some([r, c, rest @ ..]) => {
                if rest.iter().all(|dim| matches!(dim, Some(1))) {
                    Some((*r, *c))
                } else {
                    None
                }
            }
            None => Some((None, None)),
        },
        _ => None,
    }
}

#[runtime_builtin(
    name = "blkdiag",
    category = "array/shape",
    summary = "Construct a block diagonal matrix from scalar, vector, or matrix blocks.",
    keywords = "blkdiag,block diagonal,matrix,sparse,gpu",
    type_resolver(blkdiag_type),
    descriptor(crate::builtins::array::shape::blkdiag::BLKDIAG_DESCRIPTOR),
    builtin_path = "crate::builtins::array::shape::blkdiag"
)]
async fn blkdiag_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let has_gpu = args
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)));
    if has_gpu {
        return blkdiag_gpu(args).await;
    }
    blkdiag_host(args)
}

async fn blkdiag_gpu(args: Vec<Value>) -> BuiltinResult<Value> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        let gathered_value = gpu_helpers::gather_value_async(&value)
            .await
            .map_err(|err| error_with_detail(&ERROR_GPU, err.message().to_string()))?;
        gathered.push(gathered_value);
    }
    let host = blkdiag_host(gathered)?;
    upload_gpu_result(host)
}

fn blkdiag_host(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        let tensor = Tensor::new(Vec::new(), vec![0, 0])
            .map_err(|detail| error_with_detail(&ERROR_INVALID_INPUT, detail))?;
        return Ok(Value::Tensor(tensor));
    }

    let mut blocks = Vec::with_capacity(args.len());
    for value in args {
        blocks.push(Block::from_value(value)?);
    }
    assemble_blocks(blocks)
}

fn assemble_blocks(blocks: Vec<Block>) -> BuiltinResult<Value> {
    let class = OutputClass::for_blocks(&blocks)?;
    match class {
        OutputClass::Char => assemble_char(blocks),
        OutputClass::Complex => assemble_complex(blocks),
        OutputClass::Sparse => assemble_sparse(blocks),
        OutputClass::Logical => assemble_logical(blocks),
        OutputClass::Real(dtype) => assemble_real(blocks, dtype),
    }
}

#[derive(Clone, Debug)]
enum Block {
    Real(Tensor),
    Logical(LogicalArray),
    Complex(ComplexTensor),
    Sparse(SparseTensor),
    Char(CharArray),
}

impl Block {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Tensor(tensor) => {
                validate_matrix_shape(&tensor.shape)?;
                if tensor.shape.is_empty() {
                    return Tensor::new_with_dtype(tensor.data, vec![1, 1], tensor.dtype)
                        .map(Self::Real)
                        .map_err(|detail| error_with_detail(&ERROR_INVALID_INPUT, detail));
                }
                Ok(Self::Real(tensor))
            }
            Value::Num(value) => Tensor::new(vec![value], vec![1, 1])
                .map(Self::Real)
                .map_err(|detail| error_with_detail(&ERROR_INVALID_INPUT, detail)),
            Value::Int(value) => Tensor::new(vec![value.to_f64()], vec![1, 1])
                .map(Self::Real)
                .map_err(|detail| error_with_detail(&ERROR_INVALID_INPUT, detail)),
            Value::Bool(flag) => LogicalArray::new(vec![if flag { 1 } else { 0 }], vec![1, 1])
                .map(Self::Logical)
                .map_err(|detail| error_with_detail(&ERROR_INVALID_INPUT, detail)),
            Value::LogicalArray(array) => {
                validate_matrix_shape(&array.shape)?;
                if array.shape.is_empty() {
                    return LogicalArray::new(array.data, vec![1, 1])
                        .map(Self::Logical)
                        .map_err(|detail| error_with_detail(&ERROR_INVALID_INPUT, detail));
                }
                Ok(Self::Logical(array))
            }
            Value::Complex(re, im) => ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map(Self::Complex)
                .map_err(|detail| error_with_detail(&ERROR_INVALID_INPUT, detail)),
            Value::ComplexTensor(tensor) => {
                validate_matrix_shape(&tensor.shape)?;
                if tensor.shape.is_empty() {
                    return ComplexTensor::new(tensor.data, vec![1, 1])
                        .map(Self::Complex)
                        .map_err(|detail| error_with_detail(&ERROR_INVALID_INPUT, detail));
                }
                Ok(Self::Complex(tensor))
            }
            Value::SparseTensor(sparse) => Ok(Self::Sparse(sparse)),
            Value::CharArray(chars) => Ok(Self::Char(chars)),
            Value::String(_) | Value::StringArray(_) | Value::Cell(_) => Err(error_with_detail(
                &ERROR_INVALID_INPUT,
                "string and cell inputs are not valid blkdiag blocks",
            )),
            Value::GpuTensor(_) => Err(error_with_detail(
                &ERROR_INVALID_INPUT,
                "gpuArray values must be handled by the GPU path",
            )),
            other => Err(error_with_detail(
                &ERROR_INVALID_INPUT,
                format!("unsupported input type {other:?}"),
            )),
        }
    }

    fn rows(&self) -> usize {
        match self {
            Self::Real(tensor) => tensor.rows,
            Self::Logical(array) => matrix_dims_from_shape(&array.shape).map_or(0, |dims| dims.0),
            Self::Complex(tensor) => tensor.rows,
            Self::Sparse(sparse) => sparse.rows,
            Self::Char(chars) => chars.rows,
        }
    }

    fn cols(&self) -> usize {
        match self {
            Self::Real(tensor) => tensor.cols,
            Self::Logical(array) => matrix_dims_from_shape(&array.shape).map_or(0, |dims| dims.1),
            Self::Complex(tensor) => tensor.cols,
            Self::Sparse(sparse) => sparse.cols,
            Self::Char(chars) => chars.cols,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputClass {
    Real(NumericDType),
    Logical,
    Complex,
    Sparse,
    Char,
}

impl OutputClass {
    fn for_blocks(blocks: &[Block]) -> BuiltinResult<Self> {
        let has_char = blocks.iter().any(|block| matches!(block, Block::Char(_)));
        if has_char {
            for block in blocks {
                if !matches!(block, Block::Char(_)) {
                    return Err(error_with_detail(
                        &ERROR_INVALID_INPUT,
                        "char blkdiag inputs cannot be mixed with numeric, logical, complex, or sparse blocks",
                    ));
                }
            }
            return Ok(Self::Char);
        }

        let has_complex = blocks
            .iter()
            .any(|block| matches!(block, Block::Complex(_)));
        let has_sparse = blocks.iter().any(|block| matches!(block, Block::Sparse(_)));
        if has_sparse && has_complex && !complex_blocks_are_real(blocks) {
            return Ok(Self::Complex);
        }

        if has_sparse {
            return Ok(Self::Sparse);
        }

        if has_complex {
            return Ok(Self::Complex);
        }

        if blocks
            .iter()
            .all(|block| matches!(block, Block::Logical(_)))
        {
            return Ok(Self::Logical);
        }

        Ok(Self::Real(promoted_dense_dtype(blocks)))
    }
}

fn promoted_dense_dtype(blocks: &[Block]) -> NumericDType {
    let mut dtype: Option<NumericDType> = None;
    for block in blocks {
        match block {
            Block::Real(tensor) => match dtype {
                None => dtype = Some(tensor.dtype),
                Some(existing) if existing == tensor.dtype => {}
                Some(_) => return NumericDType::F64,
            },
            Block::Logical(_) => {}
            Block::Complex(_) | Block::Sparse(_) | Block::Char(_) => return NumericDType::F64,
        }
    }
    dtype.unwrap_or(NumericDType::F64)
}

fn complex_blocks_are_real(blocks: &[Block]) -> bool {
    blocks.iter().all(|block| match block {
        Block::Complex(tensor) => tensor.data.iter().all(|&(_, im)| im == 0.0),
        _ => true,
    })
}

fn validate_matrix_shape(shape: &[usize]) -> BuiltinResult<()> {
    if shape.len() <= 2 || shape.iter().skip(2).all(|&dim| dim == 1) {
        return Ok(());
    }
    Err(error_with_detail(
        &ERROR_INVALID_INPUT,
        "inputs must be two-dimensional matrices or have only singleton trailing dimensions",
    ))
}

fn matrix_dims_from_shape(shape: &[usize]) -> Option<(usize, usize)> {
    match shape {
        [] => Some((1, 1)),
        [n] => Some((1, *n)),
        [rows, cols, rest @ ..] if rest.iter().all(|&dim| dim == 1) => Some((*rows, *cols)),
        _ => None,
    }
}

fn output_dims(blocks: &[Block]) -> BuiltinResult<(usize, usize)> {
    let mut rows = 0usize;
    let mut cols = 0usize;
    for block in blocks {
        rows = rows.checked_add(block.rows()).ok_or_else(size_overflow)?;
        cols = cols.checked_add(block.cols()).ok_or_else(size_overflow)?;
    }
    Ok((rows, cols))
}

fn checked_len(rows: usize, cols: usize) -> BuiltinResult<usize> {
    rows.checked_mul(cols).ok_or_else(size_overflow)
}

fn assemble_real(blocks: Vec<Block>, dtype: NumericDType) -> BuiltinResult<Value> {
    let (rows, cols) = output_dims(&blocks)?;
    let len = checked_len(rows, cols)?;
    let mut data = filled_vec(0.0, len)?;
    let mut row_offset = 0usize;
    let mut col_offset = 0usize;
    for block in &blocks {
        match block {
            Block::Real(tensor) => copy_real_block(&mut data, rows, row_offset, col_offset, tensor),
            Block::Logical(array) => {
                copy_logical_to_real_block(&mut data, rows, row_offset, col_offset, array)?
            }
            Block::Sparse(sparse) => {
                copy_sparse_to_real_block(&mut data, rows, row_offset, col_offset, sparse)
            }
            Block::Complex(_) | Block::Char(_) => {
                return Err(error_with_detail(
                    &ERROR_INVALID_INPUT,
                    "invalid block type for real output",
                ))
            }
        }
        row_offset += block.rows();
        col_offset += block.cols();
    }
    let tensor = Tensor::new_with_dtype(data, vec![rows, cols], dtype)
        .map(|tensor| tensor::coerce_tensor_dtype(tensor, dtype))
        .map_err(|detail| error_with_detail(&ERROR_INVALID_INPUT, detail))?;
    Ok(tensor_into_blkdiag_value(tensor))
}

fn assemble_logical(blocks: Vec<Block>) -> BuiltinResult<Value> {
    let (rows, cols) = output_dims(&blocks)?;
    let len = checked_len(rows, cols)?;
    let mut data = filled_vec(0u8, len)?;
    let mut row_offset = 0usize;
    let mut col_offset = 0usize;
    for block in &blocks {
        match block {
            Block::Logical(array) => {
                let (block_rows, block_cols) =
                    matrix_dims_from_shape(&array.shape).ok_or_else(|| {
                        error_with_detail(&ERROR_INVALID_INPUT, "invalid logical shape")
                    })?;
                for col in 0..block_cols {
                    for row in 0..block_rows {
                        let src = row + col * block_rows;
                        let dst = (row_offset + row) + (col_offset + col) * rows;
                        data[dst] = array.data[src];
                    }
                }
            }
            _ => {
                return Err(error_with_detail(
                    &ERROR_INVALID_INPUT,
                    "invalid block type for logical output",
                ))
            }
        }
        row_offset += block.rows();
        col_offset += block.cols();
    }
    LogicalArray::new(data, vec![rows, cols])
        .map(Value::LogicalArray)
        .map_err(|detail| error_with_detail(&ERROR_INVALID_INPUT, detail))
}

fn assemble_complex(blocks: Vec<Block>) -> BuiltinResult<Value> {
    let (rows, cols) = output_dims(&blocks)?;
    let len = checked_len(rows, cols)?;
    let mut data = filled_vec((0.0, 0.0), len)?;
    let mut row_offset = 0usize;
    let mut col_offset = 0usize;
    for block in &blocks {
        match block {
            Block::Complex(tensor) => {
                for col in 0..tensor.cols {
                    for row in 0..tensor.rows {
                        let src = row + col * tensor.rows;
                        let dst = (row_offset + row) + (col_offset + col) * rows;
                        data[dst] = tensor.data[src];
                    }
                }
            }
            Block::Real(tensor) => {
                for col in 0..tensor.cols {
                    for row in 0..tensor.rows {
                        let src = row + col * tensor.rows;
                        let dst = (row_offset + row) + (col_offset + col) * rows;
                        data[dst] = (tensor.data[src], 0.0);
                    }
                }
            }
            Block::Logical(array) => {
                let (block_rows, block_cols) =
                    matrix_dims_from_shape(&array.shape).ok_or_else(|| {
                        error_with_detail(&ERROR_INVALID_INPUT, "invalid logical shape")
                    })?;
                for col in 0..block_cols {
                    for row in 0..block_rows {
                        let src = row + col * block_rows;
                        let dst = (row_offset + row) + (col_offset + col) * rows;
                        data[dst] = (f64::from(array.data[src]), 0.0);
                    }
                }
            }
            Block::Sparse(sparse) => {
                for col in 0..sparse.cols {
                    for idx in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
                        let row = sparse.row_indices[idx];
                        let dst = (row_offset + row) + (col_offset + col) * rows;
                        data[dst] = (sparse.values[idx], 0.0);
                    }
                }
            }
            Block::Char(_) => {
                return Err(error_with_detail(
                    &ERROR_INVALID_INPUT,
                    "char blocks cannot be converted to complex blkdiag output",
                ))
            }
        }
        row_offset += block.rows();
        col_offset += block.cols();
    }
    let tensor = ComplexTensor::new(data, vec![rows, cols])
        .map_err(|detail| error_with_detail(&ERROR_INVALID_INPUT, detail))?;
    Ok(complex_tensor_into_blkdiag_value(tensor))
}

fn assemble_sparse(blocks: Vec<Block>) -> BuiltinResult<Value> {
    let (rows, cols) = output_dims(&blocks)?;
    let mut sparse_blocks = Vec::with_capacity(blocks.len());
    for block in blocks {
        sparse_blocks.push(block_into_sparse(block)?);
    }

    let mut col_ptrs = Vec::new();
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs
        .try_reserve_exact(cols.checked_add(1).ok_or_else(size_overflow)?)
        .map_err(|_| allocation_error("sparse column pointer allocation failed"))?;
    let nnz = sparse_blocks
        .iter()
        .try_fold(0usize, |acc, block| acc.checked_add(block.values.len()))
        .ok_or_else(size_overflow)?;
    row_indices
        .try_reserve_exact(nnz)
        .map_err(|_| allocation_error("sparse row index allocation failed"))?;
    values
        .try_reserve_exact(nnz)
        .map_err(|_| allocation_error("sparse value allocation failed"))?;
    col_ptrs.push(0);
    let mut row_offset = 0usize;
    for block in sparse_blocks {
        for col in 0..block.cols {
            for idx in block.col_ptrs[col]..block.col_ptrs[col + 1] {
                row_indices.push(row_offset + block.row_indices[idx]);
                values.push(block.values[idx]);
            }
            col_ptrs.push(values.len());
        }
        row_offset += block.rows;
    }

    SparseTensor::new(rows, cols, col_ptrs, row_indices, values)
        .map(Value::SparseTensor)
        .map_err(|detail| error_with_detail(&ERROR_INVALID_INPUT, detail))
}

fn assemble_char(blocks: Vec<Block>) -> BuiltinResult<Value> {
    let (rows, cols) = output_dims(&blocks)?;
    let len = checked_len(rows, cols)?;
    let mut data = filled_vec(' ', len)?;
    let mut row_offset = 0usize;
    let mut col_offset = 0usize;
    for block in &blocks {
        let Block::Char(chars) = block else {
            return Err(error_with_detail(
                &ERROR_INVALID_INPUT,
                "invalid block type for char output",
            ));
        };
        for row in 0..chars.rows {
            for col in 0..chars.cols {
                let src = row * chars.cols + col;
                let dst = (row_offset + row) * cols + (col_offset + col);
                data[dst] = chars.data[src];
            }
        }
        row_offset += block.rows();
        col_offset += block.cols();
    }
    CharArray::new(data, rows, cols)
        .map(Value::CharArray)
        .map_err(|detail| error_with_detail(&ERROR_INVALID_INPUT, detail))
}

fn block_into_sparse(block: Block) -> BuiltinResult<SparseTensor> {
    match block {
        Block::Sparse(sparse) => Ok(sparse),
        Block::Real(tensor) => dense_to_sparse(&tensor.data, tensor.rows, tensor.cols),
        Block::Logical(array) => {
            let (rows, cols) = matrix_dims_from_shape(&array.shape)
                .ok_or_else(|| error_with_detail(&ERROR_INVALID_INPUT, "invalid logical shape"))?;
            logical_to_sparse(&array.data, rows, cols)
        }
        Block::Complex(tensor) => complex_to_real_sparse(&tensor),
        Block::Char(_) => Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "char blocks cannot be converted to sparse blkdiag output",
        )),
    }
}

fn dense_to_sparse(data: &[f64], rows: usize, cols: usize) -> BuiltinResult<SparseTensor> {
    let mut col_ptrs = Vec::new();
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs
        .try_reserve_exact(cols.checked_add(1).ok_or_else(size_overflow)?)
        .map_err(|_| allocation_error("sparse column pointer allocation failed"))?;
    col_ptrs.push(0);
    for col in 0..cols {
        for row in 0..rows {
            let value = data[row + col * rows];
            if value != 0.0 {
                row_indices.push(row);
                values.push(value);
            }
        }
        col_ptrs.push(values.len());
    }
    SparseTensor::new(rows, cols, col_ptrs, row_indices, values)
        .map_err(|detail| error_with_detail(&ERROR_INVALID_INPUT, detail))
}

fn logical_to_sparse(data: &[u8], rows: usize, cols: usize) -> BuiltinResult<SparseTensor> {
    let mut col_ptrs = Vec::new();
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs
        .try_reserve_exact(cols.checked_add(1).ok_or_else(size_overflow)?)
        .map_err(|_| allocation_error("sparse column pointer allocation failed"))?;
    col_ptrs.push(0);
    for col in 0..cols {
        for row in 0..rows {
            if data[row + col * rows] != 0 {
                row_indices.push(row);
                values.push(1.0);
            }
        }
        col_ptrs.push(values.len());
    }
    SparseTensor::new(rows, cols, col_ptrs, row_indices, values)
        .map_err(|detail| error_with_detail(&ERROR_INVALID_INPUT, detail))
}

fn complex_to_real_sparse(tensor: &ComplexTensor) -> BuiltinResult<SparseTensor> {
    let mut col_ptrs = Vec::new();
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs
        .try_reserve_exact(tensor.cols.checked_add(1).ok_or_else(size_overflow)?)
        .map_err(|_| allocation_error("sparse column pointer allocation failed"))?;
    col_ptrs.push(0);
    for col in 0..tensor.cols {
        for row in 0..tensor.rows {
            let (re, _) = tensor.data[row + col * tensor.rows];
            if re != 0.0 {
                row_indices.push(row);
                values.push(re);
            }
        }
        col_ptrs.push(values.len());
    }
    SparseTensor::new(tensor.rows, tensor.cols, col_ptrs, row_indices, values)
        .map_err(|detail| error_with_detail(&ERROR_INVALID_INPUT, detail))
}

fn copy_real_block(
    output: &mut [f64],
    output_rows: usize,
    row_offset: usize,
    col_offset: usize,
    tensor: &Tensor,
) {
    for col in 0..tensor.cols {
        for row in 0..tensor.rows {
            let src = row + col * tensor.rows;
            let dst = (row_offset + row) + (col_offset + col) * output_rows;
            output[dst] = tensor.data[src];
        }
    }
}

fn copy_logical_to_real_block(
    output: &mut [f64],
    output_rows: usize,
    row_offset: usize,
    col_offset: usize,
    array: &LogicalArray,
) -> BuiltinResult<()> {
    let (rows, cols) = matrix_dims_from_shape(&array.shape)
        .ok_or_else(|| error_with_detail(&ERROR_INVALID_INPUT, "invalid logical shape"))?;
    for col in 0..cols {
        for row in 0..rows {
            let src = row + col * rows;
            let dst = (row_offset + row) + (col_offset + col) * output_rows;
            output[dst] = f64::from(array.data[src]);
        }
    }
    Ok(())
}

fn copy_sparse_to_real_block(
    output: &mut [f64],
    output_rows: usize,
    row_offset: usize,
    col_offset: usize,
    sparse: &SparseTensor,
) {
    for col in 0..sparse.cols {
        for idx in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
            let row = sparse.row_indices[idx];
            let dst = (row_offset + row) + (col_offset + col) * output_rows;
            output[dst] = sparse.values[idx];
        }
    }
}

fn tensor_into_blkdiag_value(tensor: Tensor) -> Value {
    if tensor.dtype == NumericDType::F64 {
        tensor::tensor_into_value(tensor)
    } else {
        Value::Tensor(tensor)
    }
}

fn complex_tensor_into_blkdiag_value(tensor: ComplexTensor) -> Value {
    if tensor.data.len() == 1 {
        let (re, im) = tensor.data[0];
        Value::Complex(re, im)
    } else {
        Value::ComplexTensor(tensor)
    }
}

fn upload_gpu_result(value: Value) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider()
        .ok_or_else(|| error_with_detail(&ERROR_GPU, "no acceleration provider is registered"))?;
    match value {
        Value::Tensor(tensor) => {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider
                .upload(&view)
                .map_err(|err| error_with_detail(&ERROR_GPU, err))?;
            runmat_accelerate_api::set_handle_logical(&handle, false);
            runmat_accelerate_api::set_handle_storage(
                &handle,
                runmat_accelerate_api::GpuTensorStorage::Real,
            );
            runmat_accelerate_api::set_handle_precision(&handle, tensor_precision(tensor.dtype));
            Ok(gpu_helpers::resident_gpu_value(handle))
        }
        Value::Num(value) => {
            let data = [value];
            let shape = [1usize, 1usize];
            let view = HostTensorView {
                data: &data,
                shape: &shape,
            };
            let handle = provider
                .upload(&view)
                .map_err(|err| error_with_detail(&ERROR_GPU, err))?;
            runmat_accelerate_api::set_handle_logical(&handle, false);
            runmat_accelerate_api::set_handle_storage(
                &handle,
                runmat_accelerate_api::GpuTensorStorage::Real,
            );
            runmat_accelerate_api::set_handle_precision(
                &handle,
                runmat_accelerate_api::ProviderPrecision::F64,
            );
            Ok(gpu_helpers::resident_gpu_value(handle))
        }
        Value::LogicalArray(array) => {
            let data: Vec<f64> = array.data.iter().map(|&value| f64::from(value)).collect();
            let view = HostTensorView {
                data: &data,
                shape: &array.shape,
            };
            provider
                .upload(&view)
                .map(gpu_helpers::logical_gpu_value)
                .map_err(|err| error_with_detail(&ERROR_GPU, err))
        }
        Value::ComplexTensor(tensor) => gpu_helpers::upload_complex_tensor(provider, &tensor)
            .map(gpu_helpers::complex_gpu_value),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|detail| error_with_detail(&ERROR_GPU, detail))?;
            gpu_helpers::upload_complex_tensor(provider, &tensor)
                .map(gpu_helpers::complex_gpu_value)
        }
        Value::SparseTensor(sparse) => Ok(Value::SparseTensor(sparse)),
        other => Err(error_with_detail(
            &ERROR_GPU,
            format!("cannot upload blkdiag result {other:?}"),
        )),
    }
}

fn tensor_precision(dtype: NumericDType) -> runmat_accelerate_api::ProviderPrecision {
    match dtype {
        NumericDType::F32 => runmat_accelerate_api::ProviderPrecision::F32,
        NumericDType::F64 | NumericDType::U8 | NumericDType::U16 | NumericDType::U32 => {
            runmat_accelerate_api::ProviderPrecision::F64
        }
    }
}

fn size_overflow() -> RuntimeError {
    error_with_detail(&ERROR_SIZE_OVERFLOW, "output dimensions overflow usize")
}

fn allocation_error(detail: impl std::fmt::Display) -> RuntimeError {
    error_with_detail(&ERROR_SIZE_OVERFLOW, detail)
}

fn filled_vec<T: Clone>(value: T, len: usize) -> BuiltinResult<Vec<T>> {
    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(|_| allocation_error("output allocation failed"))?;
    data.resize(len, value);
    Ok(data)
}

fn error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {}", error.message, detail)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, Type};

    use crate::builtins::common::test_support;

    fn call(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(blkdiag_builtin(args))
    }

    #[test]
    fn type_resolver_sums_known_shapes() {
        let out = blkdiag_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
                Type::Num,
                Type::Logical {
                    shape: Some(vec![Some(4), Some(1)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(7), Some(5)])
            }
        );
    }

    #[test]
    fn empty_call_returns_empty_double_matrix() {
        match call(Vec::new()).unwrap() {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![0, 0]);
                assert!(tensor.data.is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn dense_blocks_use_column_major_offsets() {
        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0], vec![1, 2]).unwrap();
        match call(vec![Value::Tensor(a), Value::Tensor(b)]).unwrap() {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 4]);
                assert_eq!(
                    out.data,
                    vec![1.0, 3.0, 0.0, 2.0, 4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0]
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn vectors_and_scalars_are_blocks() {
        let row = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        match call(vec![
            Value::Int(IntValue::I32(1)),
            Value::Tensor(row),
            Value::Num(5.0),
        ])
        .unwrap()
        {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 5]);
                assert_eq!(out.data[0], 1.0);
                assert_eq!(out.data[1 + 3 * 3], 4.0);
                assert_eq!(out.data[2 + 4 * 3], 5.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn all_logical_inputs_return_logical_array() {
        let logical = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        match call(vec![Value::Bool(true), Value::LogicalArray(logical)]).unwrap() {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![3, 2]);
                assert_eq!(out.data, vec![1, 0, 0, 0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn single_dense_blocks_preserve_dtype() {
        let a = Tensor::new_with_dtype(vec![1.5, 2.5], vec![2, 1], NumericDType::F32).unwrap();
        let b = Tensor::new_with_dtype(vec![3.25], vec![1, 1], NumericDType::F32).unwrap();
        match call(vec![Value::Tensor(a), Value::Tensor(b)]).unwrap() {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 2]);
                assert_eq!(out.dtype, NumericDType::F32);
                assert_eq!(out.data[0], 1.5);
                assert_eq!(out.data[1], 2.5);
                assert_eq!(out.data[5], 3.25);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn complex_inputs_promote_real_blocks() {
        let complex = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![2, 1]).unwrap();
        match call(vec![Value::Num(9.0), Value::ComplexTensor(complex)]).unwrap() {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![3, 2]);
                assert_eq!(out.data[0], (9.0, 0.0));
                assert_eq!(out.data[4], (1.0, 2.0));
                assert_eq!(out.data[5], (3.0, -4.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn sparse_input_produces_sparse_block_diagonal() {
        let sparse = SparseTensor::new(2, 2, vec![0, 1, 2], vec![1, 0], vec![4.0, 5.0]).unwrap();
        let dense = Tensor::new(vec![0.0, 6.0], vec![2, 1]).unwrap();
        match call(vec![Value::SparseTensor(sparse), Value::Tensor(dense)]).unwrap() {
            Value::SparseTensor(out) => {
                assert_eq!(out.rows, 4);
                assert_eq!(out.cols, 3);
                assert_eq!(out.col_ptrs, vec![0, 1, 2, 3]);
                assert_eq!(out.row_indices, vec![1, 0, 3]);
                assert_eq!(out.values, vec![4.0, 5.0, 6.0]);
            }
            other => panic!("expected sparse tensor, got {other:?}"),
        }
    }

    #[test]
    fn sparse_with_real_complex_stays_sparse() {
        let sparse = SparseTensor::new(2, 1, vec![0, 1], vec![1], vec![4.0]).unwrap();
        match call(vec![Value::SparseTensor(sparse), Value::Complex(5.0, 0.0)]).unwrap() {
            Value::SparseTensor(out) => {
                assert_eq!(out.rows, 3);
                assert_eq!(out.cols, 2);
                assert_eq!(out.col_ptrs, vec![0, 1, 2]);
                assert_eq!(out.row_indices, vec![1, 2]);
                assert_eq!(out.values, vec![4.0, 5.0]);
            }
            other => panic!("expected sparse tensor, got {other:?}"),
        }
    }

    #[test]
    fn sparse_with_nonreal_complex_promotes_to_dense_complex() {
        let sparse = SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![2.0]).unwrap();
        match call(vec![Value::SparseTensor(sparse), Value::Complex(5.0, 1.0)]).unwrap() {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(
                    out.data,
                    vec![(2.0, 0.0), (0.0, 0.0), (0.0, 0.0), (5.0, 1.0)]
                );
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn sparse_with_mostly_zero_dense_blocks_records_only_nonzeros() {
        let sparse = SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![2.0]).unwrap();
        let mut data = vec![0.0; 512];
        data[511] = 9.0;
        let dense = Tensor::new(data, vec![512, 1]).unwrap();
        match call(vec![Value::SparseTensor(sparse), Value::Tensor(dense)]).unwrap() {
            Value::SparseTensor(out) => {
                assert_eq!(out.rows, 513);
                assert_eq!(out.cols, 2);
                assert_eq!(out.col_ptrs, vec![0, 1, 2]);
                assert_eq!(out.row_indices, vec![0, 512]);
                assert_eq!(out.values, vec![2.0, 9.0]);
            }
            other => panic!("expected sparse tensor, got {other:?}"),
        }
    }

    #[test]
    fn sparse_zero_width_blocks_contribute_rows_only() {
        let sparse = SparseTensor::new(2, 0, vec![0], vec![], vec![]).unwrap();
        match call(vec![Value::SparseTensor(sparse), Value::Num(7.0)]).unwrap() {
            Value::SparseTensor(out) => {
                assert_eq!(out.rows, 3);
                assert_eq!(out.cols, 1);
                assert_eq!(out.col_ptrs, vec![0, 1]);
                assert_eq!(out.row_indices, vec![2]);
                assert_eq!(out.values, vec![7.0]);
            }
            other => panic!("expected sparse tensor, got {other:?}"),
        }
    }

    #[test]
    fn scalar_shaped_arrays_are_one_by_one_blocks() {
        let tensor = Tensor::new(vec![7.0], vec![]).unwrap();
        let logical = LogicalArray::new(vec![1], vec![]).unwrap();
        let complex = ComplexTensor::new(vec![(2.0, -3.0)], vec![]).unwrap();
        match call(vec![
            Value::Tensor(tensor),
            Value::LogicalArray(logical),
            Value::ComplexTensor(complex),
        ])
        .unwrap()
        {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![3, 3]);
                assert_eq!(out.data[0], (7.0, 0.0));
                assert_eq!(out.data[4], (1.0, 0.0));
                assert_eq!(out.data[8], (2.0, -3.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn char_blocks_are_space_padded() {
        let a = CharArray::new("ab".chars().collect(), 1, 2).unwrap();
        let b = CharArray::new("cd".chars().collect(), 2, 1).unwrap();
        match call(vec![Value::CharArray(a), Value::CharArray(b)]).unwrap() {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 3);
                assert_eq!(out.cols, 3);
                let text: String = out.data.iter().collect();
                assert_eq!(text, "ab   c  d");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_and_high_dimensional_inputs_error() {
        let err = call(vec![Value::String("x".to_string())]).unwrap_err();
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);

        let tensor = Tensor::new(vec![0.0; 8], vec![2, 2, 2]).unwrap();
        let err = call(vec![Value::Tensor(tensor)]).unwrap_err();
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn gpu_numeric_inputs_roundtrip_to_gpu() {
        test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let b = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
            let ha = provider
                .upload(&HostTensorView {
                    data: &a.data,
                    shape: &a.shape,
                })
                .unwrap();
            let hb = provider
                .upload(&HostTensorView {
                    data: &b.data,
                    shape: &b.shape,
                })
                .unwrap();
            let value = call(vec![Value::GpuTensor(ha), Value::GpuTensor(hb)]).unwrap();
            assert!(matches!(value, Value::GpuTensor(_)));
            let gathered = test_support::gather(value).unwrap();
            assert_eq!(gathered.shape, vec![3, 3]);
            assert_eq!(
                gathered.data,
                vec![1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0]
            );
        });
    }

    #[test]
    fn gpu_single_dense_result_preserves_precision_metadata() {
        test_support::with_test_provider(|provider| {
            let a = Tensor::new_with_dtype(vec![1.25], vec![1, 1], NumericDType::F32).unwrap();
            let b = Tensor::new_with_dtype(vec![2.5], vec![1, 1], NumericDType::F32).unwrap();
            let ha = provider
                .upload(&HostTensorView {
                    data: &a.data,
                    shape: &a.shape,
                })
                .unwrap();
            let hb = provider
                .upload(&HostTensorView {
                    data: &b.data,
                    shape: &b.shape,
                })
                .unwrap();
            runmat_accelerate_api::set_handle_precision(
                &ha,
                runmat_accelerate_api::ProviderPrecision::F32,
            );
            runmat_accelerate_api::set_handle_precision(
                &hb,
                runmat_accelerate_api::ProviderPrecision::F32,
            );
            let value = call(vec![Value::GpuTensor(ha), Value::GpuTensor(hb)]).unwrap();
            let Value::GpuTensor(handle) = value else {
                panic!("expected gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_precision(&handle),
                Some(runmat_accelerate_api::ProviderPrecision::F32)
            );
            let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle)))
                .expect("gather");
            let Value::Tensor(out) = gathered else {
                panic!("expected gathered tensor");
            };
            assert_eq!(out.dtype, NumericDType::F32);
            assert_eq!(out.shape, vec![2, 2]);
        });
    }

    #[test]
    fn mixed_gpu_and_host_inputs_reupload() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &tensor.data,
                    shape: &tensor.shape,
                })
                .unwrap();
            let value = call(vec![Value::GpuTensor(handle), Value::Num(2.0)]).unwrap();
            assert!(matches!(value, Value::GpuTensor(_)));
            let gathered = test_support::gather(value).unwrap();
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![1.0, 0.0, 0.0, 2.0]);
        });
    }

    #[test]
    fn gpu_logical_inputs_preserve_logical_metadata() {
        test_support::with_test_provider(|provider| {
            let logical = Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &logical.data,
                    shape: &logical.shape,
                })
                .unwrap();
            let value = call(vec![
                Value::Bool(true),
                gpu_helpers::logical_gpu_value(handle),
            ])
            .unwrap();
            let Value::GpuTensor(handle) = value else {
                panic!("expected gpu tensor");
            };
            assert!(runmat_accelerate_api::handle_is_logical(&handle));
            let gathered = test_support::gather(Value::GpuTensor(handle)).unwrap();
            assert_eq!(gathered.shape, vec![3, 2]);
            assert_eq!(gathered.data, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        });
    }

    #[test]
    fn gpu_complex_inputs_preserve_complex_metadata() {
        test_support::with_test_provider(|provider| {
            let complex = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![2, 1]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &complex).unwrap();
            let value = call(vec![Value::Num(9.0), Value::GpuTensor(handle)]).unwrap();
            let Value::GpuTensor(handle) = value else {
                panic!("expected gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&handle),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let gathered = block_on(
                crate::builtins::math::fft::common::gather_gpu_complex_tensor(&handle, "blkdiag"),
            )
            .unwrap();
            assert_eq!(gathered.shape, vec![3, 2]);
            assert_eq!(gathered.data[0], (9.0, 0.0));
            assert_eq!(gathered.data[4], (1.0, 2.0));
            assert_eq!(gathered.data[5], (3.0, -4.0));
        });
    }
}
