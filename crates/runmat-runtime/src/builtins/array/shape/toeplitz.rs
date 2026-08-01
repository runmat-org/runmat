//! MATLAB-compatible `toeplitz` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, IntegerComplexStorage, IntegerStorage, NumericDType, NumericStorage, Tensor,
    Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::math::elementwise::conj::conjugate_integer_imaginary_storage;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "toeplitz";

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Toeplitz matrix with constant diagonals.",
}];

const INPUTS_ONE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "First row for the symmetric or Hermitian Toeplitz matrix.",
}];

const INPUTS_TWO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "c",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First column.",
    },
    BuiltinParamDescriptor {
        name: "r",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First row.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "T = toeplitz(c)",
        inputs: &INPUTS_ONE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "T = toeplitz(c, r)",
        inputs: &INPUTS_TWO,
        outputs: &OUTPUT,
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TOEPLITZ.INVALID_INPUT",
    identifier: Some("RunMat:toeplitz:InvalidInput"),
    when: "Inputs are not vector-valued real or complex numeric arrays.",
    message: "toeplitz: invalid input",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TOEPLITZ.INTERNAL",
    identifier: Some("RunMat:toeplitz:Internal"),
    when: "Output allocation or gpuArray gather fails.",
    message: "toeplitz: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_INTERNAL];

pub const TOEPLITZ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "toeplitz",
    category = "array/shape",
    summary = "Construct a Toeplitz matrix from first-column and first-row vectors.",
    keywords = "toeplitz,matrix,constant diagonals",
    accel = "gather",
    descriptor(crate::builtins::array::shape::toeplitz::TOEPLITZ_DESCRIPTOR),
    builtin_path = "crate::builtins::array::shape::toeplitz"
)]
async fn toeplitz_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    match args.len() {
        1 => {
            let r = InputVector::from_value(args.into_iter().next().expect("r")).await?;
            toeplitz_from_vectors(r.conjugated(), r, true)
        }
        2 => {
            let mut iter = args.into_iter();
            let c = InputVector::from_value(iter.next().expect("c")).await?;
            let r = InputVector::from_value(iter.next().expect("r")).await?;
            toeplitz_from_vectors(c, r, false)
        }
        _ => Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "expected one or two inputs",
        )),
    }
}

#[derive(Clone, Debug)]
enum InputVector {
    Real {
        values: Vec<f64>,
        dtype: NumericDType,
    },
    TypedInteger(IntegerStorage),
    Complex(Vec<(f64, f64)>),
    TypedComplex(IntegerComplexStorage),
}

impl InputVector {
    async fn from_value(value: Value) -> BuiltinResult<Self> {
        let gathered = gpu_helpers::gather_value_async(&value)
            .await
            .map_err(|err| error_with_detail(&ERROR_INTERNAL, err.message()))?;
        match gathered {
            Value::Num(value) => Ok(Self::Real {
                values: vec![value],
                dtype: NumericDType::F64,
            }),
            Value::Int(value) => Ok(Self::TypedInteger(IntegerStorage::from_scalar(value))),
            Value::Tensor(tensor) => {
                validate_vector_shape(&tensor.shape, tensor_utils::tensor_element_len(&tensor))?;
                let storage = tensor
                    .into_numeric_storage()
                    .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))?;
                Ok(match storage {
                    NumericStorage::F64(values) => Self::Real {
                        values,
                        dtype: NumericDType::F64,
                    },
                    NumericStorage::F32(values) => Self::Real {
                        values: values.into_iter().map(f64::from).collect(),
                        dtype: NumericDType::F32,
                    },
                    storage => Self::TypedInteger(
                        storage
                            .into_integer_storage()
                            .expect("integer numeric storage"),
                    ),
                })
            }
            Value::Complex(re, im) => Ok(Self::Complex(vec![(re, im)])),
            Value::ComplexTensor(tensor) => {
                validate_vector_shape(
                    &tensor.shape,
                    tensor_utils::complex_tensor_element_len(&tensor),
                )?;
                Ok(match tensor.integer_data {
                    Some(storage) => Self::TypedComplex(storage),
                    None => Self::Complex(tensor.data),
                })
            }
            other => Err(error_with_detail(
                &ERROR_INVALID_INPUT,
                format!("unsupported input {other:?}"),
            )),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Real { values, .. } => values.len(),
            Self::TypedInteger(storage) => storage.len(),
            Self::Complex(values) => values.len(),
            Self::TypedComplex(storage) => storage.len(),
        }
    }

    fn is_complex(&self) -> bool {
        matches!(self, Self::Complex(_) | Self::TypedComplex(_))
    }

    fn real_at(&self, index: usize) -> f64 {
        match self {
            Self::Real { values, .. } => values[index],
            Self::TypedInteger(storage) => storage
                .value_at(index)
                .expect("toeplitz index is in bounds")
                .to_f64(),
            Self::Complex(values) => values[index].0,
            Self::TypedComplex(_) => {
                unreachable!("typed complex integers use toeplitz_typed_complex")
            }
        }
    }

    fn complex_at(&self, index: usize) -> (f64, f64) {
        match self {
            Self::Real { values, .. } => (values[index], 0.0),
            Self::TypedInteger(storage) => (
                storage
                    .value_at(index)
                    .expect("toeplitz index is in bounds")
                    .to_f64(),
                0.0,
            ),
            Self::Complex(values) => values[index],
            Self::TypedComplex(_) => {
                unreachable!("typed complex integers use toeplitz_typed_complex")
            }
        }
    }

    fn conjugated(&self) -> Self {
        match self {
            Self::Real { values, dtype } => Self::Real {
                values: values.clone(),
                dtype: *dtype,
            },
            Self::TypedInteger(storage) => Self::TypedInteger(storage.clone()),
            Self::Complex(values) => {
                Self::Complex(values.iter().map(|&(re, im)| (re, -im)).collect())
            }
            Self::TypedComplex(storage) => Self::TypedComplex(
                IntegerComplexStorage::new(
                    storage.real.clone(),
                    conjugate_integer_imaginary_storage(storage.imag.clone()),
                )
                .expect("existing typed complex storage has matching components"),
            ),
        }
    }

    fn dtype(&self) -> Option<NumericDType> {
        match self {
            Self::Real { dtype, .. } => Some(*dtype),
            Self::TypedInteger(_) => None,
            Self::Complex(_) => None,
            Self::TypedComplex(_) => None,
        }
    }
}

fn toeplitz_from_vectors(c: InputVector, r: InputVector, one_input: bool) -> BuiltinResult<Value> {
    let rows = c.len();
    let cols = r.len();
    if matches!(&c, InputVector::TypedComplex(_)) || matches!(&r, InputVector::TypedComplex(_)) {
        return toeplitz_typed_complex(c, r, rows, cols, one_input);
    }
    if matches!(&c, InputVector::TypedInteger(_)) && matches!(&r, InputVector::TypedInteger(_)) {
        return toeplitz_typed_integer(c, r, rows, cols);
    }
    if c.is_complex() || r.is_complex() {
        return toeplitz_complex(c, r, rows, cols, one_input);
    }
    toeplitz_real(c, r, rows, cols)
}

fn toeplitz_typed_integer(
    c: InputVector,
    r: InputVector,
    rows: usize,
    cols: usize,
) -> BuiltinResult<Value> {
    let (InputVector::TypedInteger(c), InputVector::TypedInteger(r)) = (c, r) else {
        unreachable!("typed integer branch requires typed integer vectors");
    };
    if c.class_name() != r.class_name() {
        return toeplitz_real(
            InputVector::TypedInteger(c),
            InputVector::TypedInteger(r),
            rows,
            cols,
        );
    }
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| error_with_detail(&ERROR_INTERNAL, "output size overflow"))?;
    let mut values = Vec::with_capacity(len);
    for col in 0..cols {
        for row in 0..rows {
            let storage = if row >= col { &c } else { &r };
            values.push(
                storage
                    .value_at(row.abs_diff(col))
                    .expect("toeplitz index is in bounds"),
            );
        }
    }
    let storage = c
        .from_exact_values_like(values)
        .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))?;
    Tensor::new_integer(storage, vec![rows, cols])
        .map(Value::Tensor)
        .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))
}

fn toeplitz_typed_complex(
    c: InputVector,
    r: InputVector,
    rows: usize,
    cols: usize,
    one_input: bool,
) -> BuiltinResult<Value> {
    let (InputVector::TypedComplex(c), InputVector::TypedComplex(r)) = (c, r) else {
        return Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "typed complex integer inputs must use the same integer class",
        ));
    };
    if c.class_name() != r.class_name() {
        return Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "typed complex integer inputs must use the same integer class",
        ));
    }
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| error_with_detail(&ERROR_INTERNAL, "output size overflow"))?;
    let mut real = Vec::with_capacity(len);
    let mut imag = Vec::with_capacity(len);
    for col in 0..cols {
        for row in 0..rows {
            let index = row.abs_diff(col);
            let storage = if row >= col { &c } else { &r };
            real.push(
                storage
                    .real
                    .value_at(index)
                    .expect("toeplitz index is in bounds"),
            );
            imag.push(
                storage
                    .imag
                    .value_at(index)
                    .expect("toeplitz index is in bounds"),
            );
        }
    }
    if one_input && !real.is_empty() {
        let diagonal_real = r.real.value_at(0).expect("nonempty diagonal");
        let diagonal_imag = r.imag.value_at(0).expect("nonempty diagonal");
        for index in 0..rows.min(cols) {
            real[index + index * rows] = diagonal_real.clone();
            imag[index + index * rows] = diagonal_imag.clone();
        }
    }
    let real = c
        .real
        .from_exact_values_like(real)
        .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))?;
    let imag = c
        .imag
        .from_exact_values_like(imag)
        .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))?;
    let storage = IntegerComplexStorage::new(real, imag)
        .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))?;
    ComplexTensor::new_integer(storage, vec![rows, cols])
        .map(Value::ComplexTensor)
        .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))
}

fn toeplitz_real(c: InputVector, r: InputVector, rows: usize, cols: usize) -> BuiltinResult<Value> {
    let dtype = common_real_dtype(&c, &r);
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| error_with_detail(&ERROR_INTERNAL, "output size overflow"))?;
    let mut data = Vec::with_capacity(len);
    for col in 0..cols {
        for row in 0..rows {
            let value = if row >= col {
                c.real_at(row - col)
            } else {
                r.real_at(col - row)
            };
            data.push(value);
        }
    }
    if len == 1 {
        return scalar_or_tensor_real(data, rows, cols, dtype);
    }
    Tensor::new_with_dtype(data, vec![rows, cols], dtype)
        .map(Value::Tensor)
        .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))
}

fn toeplitz_complex(
    c: InputVector,
    r: InputVector,
    rows: usize,
    cols: usize,
    one_input: bool,
) -> BuiltinResult<Value> {
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| error_with_detail(&ERROR_INTERNAL, "output size overflow"))?;
    let mut data = Vec::with_capacity(len);
    for col in 0..cols {
        for row in 0..rows {
            let value = if row >= col {
                c.complex_at(row - col)
            } else {
                r.complex_at(col - row)
            };
            data.push(value);
        }
    }
    if one_input && !data.is_empty() {
        let diagonal = r.complex_at(0);
        for idx in 0..rows.min(cols) {
            data[idx + idx * rows] = diagonal;
        }
    }
    if len == 1 {
        let (re, im) = data[0];
        return Ok(Value::Complex(re, im));
    }
    ComplexTensor::new(data, vec![rows, cols])
        .map(Value::ComplexTensor)
        .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))
}

fn validate_vector_shape(shape: &[usize], len: usize) -> BuiltinResult<()> {
    if len <= 1 {
        return Ok(());
    }
    if is_vector_shape(shape) {
        return Ok(());
    }
    Err(error_with_detail(
        &ERROR_INVALID_INPUT,
        "inputs must be vectors",
    ))
}

fn is_vector_shape(shape: &[usize]) -> bool {
    match shape {
        [] => true,
        [_] => true,
        [rows, cols] => *rows == 1 || *cols == 1,
        dims => dims.iter().filter(|&&dim| dim != 1).take(2).count() <= 1,
    }
}

fn common_real_dtype(c: &InputVector, r: &InputVector) -> NumericDType {
    match (c.dtype(), r.dtype()) {
        (Some(left), Some(right)) if left == right => left,
        _ => NumericDType::F64,
    }
}

fn scalar_or_tensor_real(
    data: Vec<f64>,
    rows: usize,
    cols: usize,
    dtype: NumericDType,
) -> BuiltinResult<Value> {
    if dtype == NumericDType::F64 {
        Ok(Value::Num(data[0]))
    } else {
        Tensor::new_with_dtype(data, vec![rows, cols], dtype)
            .map(Value::Tensor)
            .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))
    }
}

fn error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {detail}", error.message)).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerComplexStorage, IntegerStorage};

    fn row(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).expect("tensor"))
    }

    #[test]
    fn toeplitz_one_real_vector_is_symmetric() {
        let out = block_on(toeplitz_builtin(vec![row(&[1.0, 2.0, 3.0])])).expect("toeplitz");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![3, 3]);
        assert_eq!(
            tensor.data,
            vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]
        );
    }

    #[test]
    fn toeplitz_column_and_row_use_first_column_diagonal() {
        let out = block_on(toeplitz_builtin(vec![
            row(&[1.0, 2.0, 3.0]),
            row(&[9.0, 4.0]),
        ]))
        .expect("toeplitz");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![3, 2]);
        assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn toeplitz_complex_single_input_uses_row_and_conjugates_first_column() {
        let r = ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap();
        let out = block_on(toeplitz_builtin(vec![Value::ComplexTensor(r)])).expect("toeplitz");
        let Value::ComplexTensor(tensor) = out else {
            panic!("expected complex tensor");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(
            tensor.data,
            vec![(1.0, 2.0), (3.0, -4.0), (3.0, 4.0), (1.0, 2.0)]
        );
    }

    #[test]
    fn toeplitz_preserves_native_single_storage() {
        let input = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let out = block_on(toeplitz_builtin(vec![Value::Tensor(input)])).expect("toeplitz");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![3, 3]);
        assert_eq!(
            tensor.into_numeric_storage().expect("single storage"),
            NumericStorage::F32(vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0])
        );
    }

    #[test]
    fn toeplitz_preserves_all_exact_integer_classes() {
        let storages = [
            IntegerStorage::I8(vec![-2, 7, 9]),
            IntegerStorage::I16(vec![-300, 400, 900]),
            IntegerStorage::I32(vec![i32::MIN, 0, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, 0, i64::MAX]),
            IntegerStorage::U8(vec![0, 7, u8::MAX]),
            IntegerStorage::U16(vec![0, 700, u16::MAX]),
            IntegerStorage::U32(vec![0, 9_007_199, u32::MAX]),
            IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX]),
        ];

        for storage in storages {
            let values = storage.exact_values();
            let expected = storage
                .from_exact_values_like(vec![
                    values[0].clone(),
                    values[1].clone(),
                    values[2].clone(),
                    values[1].clone(),
                    values[0].clone(),
                    values[1].clone(),
                    values[2].clone(),
                    values[1].clone(),
                    values[0].clone(),
                ])
                .expect("expected toeplitz storage");
            let mut input = Tensor::new_integer(storage, vec![1, 3]).expect("integer vector");
            input.data.clear();
            let Value::Tensor(output) =
                block_on(toeplitz_builtin(vec![Value::Tensor(input)])).expect("toeplitz")
            else {
                panic!("expected exact integer tensor");
            };
            assert_eq!(output.shape, vec![3, 3]);
            assert_eq!(output.integer_storage(), Some(&expected));
        }

        let mut column = Tensor::new_integer(IntegerStorage::U64(vec![7, u64::MAX]), vec![2, 1])
            .expect("column");
        column.data.clear();
        let mut row = Tensor::new_integer(
            IntegerStorage::U64(vec![7, 9_007_199_254_740_993]),
            vec![1, 2],
        )
        .expect("row");
        row.data.clear();
        let Value::Tensor(output) = block_on(toeplitz_builtin(vec![
            Value::Tensor(column),
            Value::Tensor(row),
        ]))
        .expect("toeplitz") else {
            panic!("expected exact integer tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                7,
                u64::MAX,
                9_007_199_254_740_993,
                7
            ]))
        );

        let Value::Tensor(scalar) =
            block_on(toeplitz_builtin(vec![Value::Int(IntValue::U64(u64::MAX))]))
                .expect("scalar toeplitz")
        else {
            panic!("expected exact scalar tensor");
        };
        assert_eq!(
            scalar.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX]))
        );
    }

    #[test]
    fn toeplitz_typed_complex_integer_vector_shape_uses_storage_len_not_mirror() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::I16(vec![1, 2, 3]),
            IntegerStorage::I16(vec![4, 5, 6]),
        )
        .expect("complex integer storage");
        let mut input = ComplexTensor::new_integer(storage, vec![1, 3]).expect("complex vector");
        input.data.clear();

        let Value::ComplexTensor(output) =
            block_on(toeplitz_builtin(vec![Value::ComplexTensor(input)])).expect("toeplitz")
        else {
            panic!("expected exact complex integer tensor");
        };

        assert_eq!(output.shape, vec![3, 3]);
        let storage = output.integer_data.expect("typed complex storage");
        assert_eq!(
            storage.real,
            IntegerStorage::I16(vec![1, 2, 3, 2, 1, 2, 3, 2, 1])
        );
        assert_eq!(
            storage.imag,
            IntegerStorage::I16(vec![4, -5, -6, 5, 4, -5, 6, 5, 4])
        );
    }

    #[test]
    fn toeplitz_rejects_matrix_inputs() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = block_on(toeplitz_builtin(vec![Value::Tensor(matrix)]))
            .expect_err("matrix input should fail");
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
    }
}
