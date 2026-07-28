//! Exact relational comparisons for native MATLAB integer storage.

use std::cmp::Ordering;

use runmat_builtins::{
    ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray, Value,
};

use crate::builtins::common::broadcast::BroadcastPlan;

#[derive(Clone, Copy)]
pub(crate) enum IntegerComparisonOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug)]
pub(crate) enum IntegerComparisonError {
    SizeMismatch,
    Internal,
}

/// Performs a comparison when native integer storage is compared against other
/// native integer storage or real numeric storage. This keeps integer values
/// exact even when the other operand is an f64 array.
pub(crate) fn try_integer_comparison(
    lhs: &Value,
    rhs: &Value,
    operation: IntegerComparisonOp,
) -> Result<Option<Value>, IntegerComparisonError> {
    let lhs_integer = integer_operand(lhs);
    let rhs_integer = integer_operand(rhs);
    let result = match (lhs_integer, rhs_integer) {
        (None, None) => return Ok(None),
        (Some(lhs), Some(rhs)) => compare_integer_operands(&lhs, &rhs, operation)?,
        (Some(lhs), None) => {
            let Some(rhs) = numeric_operand(rhs) else {
                return Ok(None);
            };
            compare_integer_numeric(&lhs, &rhs, true, operation)?
        }
        (None, Some(rhs)) => {
            let Some(lhs) = numeric_operand(lhs) else {
                return Ok(None);
            };
            compare_integer_numeric(&rhs, &lhs, false, operation)?
        }
    };
    Ok(Some(result))
}

/// Performs exact equality/inequality when at least one complex operand carries
/// native integer component storage.
pub(crate) fn try_complex_integer_equality_comparison(
    lhs: &Value,
    rhs: &Value,
    operation: IntegerComparisonOp,
) -> Result<Option<Value>, IntegerComparisonError> {
    debug_assert!(matches!(
        operation,
        IntegerComparisonOp::Eq | IntegerComparisonOp::Ne
    ));
    let lhs_complex = complex_operand(lhs);
    let rhs_complex = complex_operand(rhs);
    let result = match (lhs_complex, rhs_complex) {
        (Some(lhs), Some(rhs)) if lhs.has_integer_storage() || rhs.has_integer_storage() => {
            compare_complex_operands(&lhs, &rhs, operation)?
        }
        (Some(lhs), None) if lhs.has_integer_storage() => {
            let Some(rhs) = real_operand(rhs) else {
                return Ok(None);
            };
            compare_complex_real(&lhs, &rhs, operation)?
        }
        (None, Some(rhs)) if rhs.has_integer_storage() => {
            let Some(lhs) = real_operand(lhs) else {
                return Ok(None);
            };
            compare_complex_real(&rhs, &lhs, operation)?
        }
        _ => return Ok(None),
    };
    Ok(Some(result))
}

fn compare_integer_operands(
    lhs: &IntegerOperand<'_>,
    rhs: &IntegerOperand<'_>,
    operation: IntegerComparisonOp,
) -> Result<Value, IntegerComparisonError> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|_| IntegerComparisonError::SizeMismatch)?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, lhs_index, rhs_index) in plan.iter() {
        let ordering = compare_integer_values(lhs.value_at(lhs_index), rhs.value_at(rhs_index));
        data.push(matches_relation(ordering, operation) as u8);
    }
    logical_result(data, plan.output_shape().to_vec())
}

fn compare_integer_numeric(
    integer: &IntegerOperand<'_>,
    numeric: &NumericOperand<'_>,
    integer_is_left: bool,
    operation: IntegerComparisonOp,
) -> Result<Value, IntegerComparisonError> {
    let plan = BroadcastPlan::new(&integer.shape, numeric.shape())
        .map_err(|_| IntegerComparisonError::SizeMismatch)?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, integer_index, numeric_index) in plan.iter() {
        let ordering = integer_f64_order(
            integer.value_at(integer_index),
            numeric.value_at(numeric_index),
        );
        let ordering = if integer_is_left {
            ordering
        } else {
            ordering.map(Ordering::reverse)
        };
        data.push(matches_optional_relation(ordering, operation) as u8);
    }
    logical_result(data, plan.output_shape().to_vec())
}

fn logical_result(data: Vec<u8>, shape: Vec<usize>) -> Result<Value, IntegerComparisonError> {
    if data.len() == 1 {
        return Ok(Value::Bool(data[0] != 0));
    }
    Ok(Value::LogicalArray(
        LogicalArray::new(data, shape).map_err(|_| IntegerComparisonError::Internal)?,
    ))
}

struct ComplexOperand<'a> {
    source: ComplexSource<'a>,
    shape: Vec<usize>,
}

impl ComplexOperand<'_> {
    fn has_integer_storage(&self) -> bool {
        matches!(
            self.source,
            ComplexSource::Tensor {
                integer_storage: Some(_),
                ..
            }
        )
    }

    fn real_imag_at(&self, index: usize) -> ComplexValue {
        match self.source {
            ComplexSource::Scalar(real, imag) => ComplexValue::Float(real, imag),
            ComplexSource::Tensor {
                data: _,
                integer_storage: Some(storage),
            } => ComplexValue::Integer(
                storage_value(&storage.real, index),
                storage_value(&storage.imag, index),
            ),
            ComplexSource::Tensor {
                data,
                integer_storage: None,
            } => {
                let (real, imag) = data[index];
                ComplexValue::Float(real, imag)
            }
        }
    }
}

enum ComplexSource<'a> {
    Scalar(f64, f64),
    Tensor {
        data: &'a [(f64, f64)],
        integer_storage: Option<&'a IntegerComplexStorage>,
    },
}

enum ComplexValue {
    Integer(IntValue, IntValue),
    Float(f64, f64),
}

enum RealValue {
    Integer(IntValue),
    Float(f64),
}

struct RealOperand<'a> {
    source: RealSource<'a>,
    shape: Vec<usize>,
}

impl RealOperand<'_> {
    fn value_at(&self, index: usize) -> RealValue {
        match self.source {
            RealSource::ScalarInteger(ref value) => RealValue::Integer(value.clone()),
            RealSource::ScalarFloat(value) => RealValue::Float(value),
            RealSource::Tensor {
                data: _,
                integer_storage: Some(storage),
            } => RealValue::Integer(storage_value(storage, index)),
            RealSource::Tensor {
                data,
                integer_storage: None,
            } => RealValue::Float(data[index]),
            RealSource::Logical { data } => RealValue::Float(f64::from(data[index] != 0)),
        }
    }
}

enum RealSource<'a> {
    ScalarInteger(IntValue),
    ScalarFloat(f64),
    Tensor {
        data: &'a [f64],
        integer_storage: Option<&'a IntegerStorage>,
    },
    Logical {
        data: &'a [u8],
    },
}

fn compare_complex_operands(
    lhs: &ComplexOperand<'_>,
    rhs: &ComplexOperand<'_>,
    operation: IntegerComparisonOp,
) -> Result<Value, IntegerComparisonError> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|_| IntegerComparisonError::SizeMismatch)?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, lhs_index, rhs_index) in plan.iter() {
        let matches =
            complex_values_equal(lhs.real_imag_at(lhs_index), rhs.real_imag_at(rhs_index));
        data.push(matches_relation_bool(matches, operation) as u8);
    }
    logical_result(data, plan.output_shape().to_vec())
}

fn compare_complex_real(
    complex: &ComplexOperand<'_>,
    real: &RealOperand<'_>,
    operation: IntegerComparisonOp,
) -> Result<Value, IntegerComparisonError> {
    let plan = BroadcastPlan::new(&complex.shape, &real.shape)
        .map_err(|_| IntegerComparisonError::SizeMismatch)?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, complex_index, real_index) in plan.iter() {
        let matches = complex_value_equals_real(
            complex.real_imag_at(complex_index),
            real.value_at(real_index),
        );
        data.push(matches_relation_bool(matches, operation) as u8);
    }
    logical_result(data, plan.output_shape().to_vec())
}

fn complex_values_equal(lhs: ComplexValue, rhs: ComplexValue) -> bool {
    match (lhs, rhs) {
        (ComplexValue::Integer(lhs_real, lhs_imag), ComplexValue::Integer(rhs_real, rhs_imag)) => {
            compare_integer_values(lhs_real, rhs_real) == Ordering::Equal
                && compare_integer_values(lhs_imag, rhs_imag) == Ordering::Equal
        }
        (ComplexValue::Integer(real, imag), ComplexValue::Float(rhs_real, rhs_imag)) => {
            integer_value_equals_f64(real, rhs_real) && integer_value_equals_f64(imag, rhs_imag)
        }
        (ComplexValue::Float(lhs_real, lhs_imag), ComplexValue::Integer(real, imag)) => {
            integer_value_equals_f64(real, lhs_real) && integer_value_equals_f64(imag, lhs_imag)
        }
        (ComplexValue::Float(lhs_real, lhs_imag), ComplexValue::Float(rhs_real, rhs_imag)) => {
            lhs_real == rhs_real && lhs_imag == rhs_imag
        }
    }
}

fn complex_value_equals_real(complex: ComplexValue, real: RealValue) -> bool {
    match (complex, real) {
        (ComplexValue::Integer(complex_real, complex_imag), RealValue::Integer(real)) => {
            complex_imag.is_zero() && compare_integer_values(complex_real, real) == Ordering::Equal
        }
        (ComplexValue::Integer(complex_real, complex_imag), RealValue::Float(real)) => {
            complex_imag.is_zero() && integer_value_equals_f64(complex_real, real)
        }
        (ComplexValue::Float(complex_real, complex_imag), RealValue::Integer(real)) => {
            complex_imag == 0.0 && integer_value_equals_f64(real, complex_real)
        }
        (ComplexValue::Float(complex_real, complex_imag), RealValue::Float(real)) => {
            complex_imag == 0.0 && complex_real == real
        }
    }
}

fn integer_value_equals_f64(integer: IntValue, float: f64) -> bool {
    integer_f64_order(integer, float) == Some(Ordering::Equal)
}

fn matches_relation_bool(matches: bool, operation: IntegerComparisonOp) -> bool {
    match operation {
        IntegerComparisonOp::Eq => matches,
        IntegerComparisonOp::Ne => !matches,
        IntegerComparisonOp::Lt
        | IntegerComparisonOp::Le
        | IntegerComparisonOp::Gt
        | IntegerComparisonOp::Ge => {
            unreachable!("complex integer equality helper only supports eq/ne")
        }
    }
}

fn complex_operand(value: &Value) -> Option<ComplexOperand<'_>> {
    match value {
        Value::Complex(real, imag) => Some(ComplexOperand {
            source: ComplexSource::Scalar(*real, *imag),
            shape: vec![1, 1],
        }),
        Value::ComplexTensor(tensor) => Some(complex_tensor_operand(tensor)),
        _ => None,
    }
}

fn complex_tensor_operand(tensor: &ComplexTensor) -> ComplexOperand<'_> {
    ComplexOperand {
        source: ComplexSource::Tensor {
            data: &tensor.data,
            integer_storage: tensor.integer_data.as_ref(),
        },
        shape: tensor.shape.clone(),
    }
}

struct IntegerOperand<'a> {
    storage: IntegerStorageRef<'a>,
    shape: Vec<usize>,
}

impl IntegerOperand<'_> {
    fn value_at(&self, index: usize) -> IntValue {
        self.storage.value_at(index)
    }
}

enum IntegerStorageRef<'a> {
    Scalar(&'a IntValue),
    Array(&'a IntegerStorage),
}

impl IntegerStorageRef<'_> {
    fn value_at(&self, index: usize) -> IntValue {
        match self {
            Self::Scalar(value) => (*value).clone(),
            Self::Array(storage) => storage_value(storage, index),
        }
    }
}

fn integer_operand(value: &Value) -> Option<IntegerOperand<'_>> {
    match value {
        Value::Int(value) => Some(IntegerOperand {
            storage: IntegerStorageRef::Scalar(value),
            shape: vec![1, 1],
        }),
        Value::Tensor(tensor) => tensor.integer_storage().map(|storage| IntegerOperand {
            storage: IntegerStorageRef::Array(storage),
            shape: tensor.shape.clone(),
        }),
        _ => None,
    }
}

fn real_operand(value: &Value) -> Option<RealOperand<'_>> {
    match value {
        Value::Int(value) => Some(RealOperand {
            source: RealSource::ScalarInteger(value.clone()),
            shape: vec![1, 1],
        }),
        Value::Num(value) => Some(RealOperand {
            source: RealSource::ScalarFloat(*value),
            shape: vec![1, 1],
        }),
        Value::Bool(value) => Some(RealOperand {
            source: RealSource::ScalarFloat(if *value { 1.0 } else { 0.0 }),
            shape: vec![1, 1],
        }),
        Value::Tensor(tensor) => Some(RealOperand {
            source: RealSource::Tensor {
                data: &tensor.data,
                integer_storage: tensor.integer_storage(),
            },
            shape: tensor.shape.clone(),
        }),
        Value::LogicalArray(array) => Some(RealOperand {
            source: RealSource::Logical { data: &array.data },
            shape: array.shape.clone(),
        }),
        _ => None,
    }
}

pub(crate) fn compare_integer_values(lhs: IntValue, rhs: IntValue) -> Ordering {
    match (signed_value(&lhs), signed_value(&rhs)) {
        (Some(lhs), Some(rhs)) => lhs.cmp(&rhs),
        (None, None) => unsigned_value(&lhs).cmp(&unsigned_value(&rhs)),
        (Some(lhs), None) => {
            if lhs < 0 {
                Ordering::Less
            } else {
                (lhs as u64).cmp(&unsigned_value(&rhs))
            }
        }
        (None, Some(rhs)) => {
            if rhs < 0 {
                Ordering::Greater
            } else {
                unsigned_value(&lhs).cmp(&(rhs as u64))
            }
        }
    }
}

pub(crate) fn integer_f64_order(integer: IntValue, float: f64) -> Option<Ordering> {
    if float.is_nan() {
        return None;
    }
    if float == f64::INFINITY {
        return Some(Ordering::Less);
    }
    if float == f64::NEG_INFINITY {
        return Some(Ordering::Greater);
    }

    const MIN_I64: f64 = -9_223_372_036_854_775_808.0;
    const U64_EXCLUSIVE_UPPER: f64 = 18_446_744_073_709_551_616.0;
    if float < MIN_I64 {
        return Some(Ordering::Greater);
    }
    if float >= U64_EXCLUSIVE_UPPER {
        return Some(Ordering::Less);
    }

    let integer = integer_as_i128(&integer);
    let truncated = float as i128;
    let ordering = integer.cmp(&truncated);
    if float.fract() == 0.0 {
        return Some(ordering);
    }
    Some(if float.is_sign_positive() {
        if ordering == Ordering::Greater {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    } else if ordering == Ordering::Less {
        Ordering::Less
    } else {
        Ordering::Greater
    })
}

fn integer_as_i128(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => i128::from(*value),
        IntValue::I16(value) => i128::from(*value),
        IntValue::I32(value) => i128::from(*value),
        IntValue::I64(value) => i128::from(*value),
        IntValue::U8(value) => i128::from(*value),
        IntValue::U16(value) => i128::from(*value),
        IntValue::U32(value) => i128::from(*value),
        IntValue::U64(value) => i128::from(*value),
    }
}

enum NumericOperand<'a> {
    Scalar(f64),
    Tensor(&'a [f64], &'a [usize]),
    Logical(&'a [u8], &'a [usize]),
}

impl NumericOperand<'_> {
    fn shape(&self) -> &[usize] {
        match self {
            Self::Scalar(_) => &[1, 1],
            Self::Tensor(_, shape) | Self::Logical(_, shape) => shape,
        }
    }

    fn value_at(&self, index: usize) -> f64 {
        match self {
            Self::Scalar(value) => *value,
            Self::Tensor(data, _) => data[index],
            Self::Logical(data, _) => f64::from(data[index] != 0),
        }
    }
}

fn numeric_operand(value: &Value) -> Option<NumericOperand<'_>> {
    match value {
        Value::Num(value) => Some(NumericOperand::Scalar(*value)),
        Value::Bool(value) => Some(NumericOperand::Scalar(if *value { 1.0 } else { 0.0 })),
        Value::Tensor(tensor) if tensor.integer_storage().is_none() => {
            Some(NumericOperand::Tensor(&tensor.data, &tensor.shape))
        }
        Value::LogicalArray(array) => Some(NumericOperand::Logical(&array.data, &array.shape)),
        _ => None,
    }
}

fn signed_value(value: &IntValue) -> Option<i64> {
    match value {
        IntValue::I8(value) => Some(*value as i64),
        IntValue::I16(value) => Some(*value as i64),
        IntValue::I32(value) => Some(*value as i64),
        IntValue::I64(value) => Some(*value),
        IntValue::U8(_) | IntValue::U16(_) | IntValue::U32(_) | IntValue::U64(_) => None,
    }
}

fn unsigned_value(value: &IntValue) -> u64 {
    match value {
        IntValue::U8(value) => *value as u64,
        IntValue::U16(value) => *value as u64,
        IntValue::U32(value) => *value as u64,
        IntValue::U64(value) => *value,
        IntValue::I8(_) | IntValue::I16(_) | IntValue::I32(_) | IntValue::I64(_) => {
            unreachable!("unsigned conversion is only used for unsigned integer values")
        }
    }
}

pub(crate) fn storage_value(storage: &IntegerStorage, index: usize) -> IntValue {
    match storage {
        IntegerStorage::I8(values) => IntValue::I8(values[index]),
        IntegerStorage::I16(values) => IntValue::I16(values[index]),
        IntegerStorage::I32(values) => IntValue::I32(values[index]),
        IntegerStorage::I64(values) => IntValue::I64(values[index]),
        IntegerStorage::U8(values) => IntValue::U8(values[index]),
        IntegerStorage::U16(values) => IntValue::U16(values[index]),
        IntegerStorage::U32(values) => IntValue::U32(values[index]),
        IntegerStorage::U64(values) => IntValue::U64(values[index]),
    }
}

pub(crate) fn matches_relation(ordering: Ordering, operation: IntegerComparisonOp) -> bool {
    match operation {
        IntegerComparisonOp::Eq => ordering == Ordering::Equal,
        IntegerComparisonOp::Ne => ordering != Ordering::Equal,
        IntegerComparisonOp::Lt => ordering == Ordering::Less,
        IntegerComparisonOp::Le => ordering != Ordering::Greater,
        IntegerComparisonOp::Gt => ordering == Ordering::Greater,
        IntegerComparisonOp::Ge => ordering != Ordering::Less,
    }
}

pub(crate) fn matches_optional_relation(
    ordering: Option<Ordering>,
    operation: IntegerComparisonOp,
) -> bool {
    match ordering {
        Some(ordering) => matches_relation(ordering, operation),
        None => matches!(operation, IntegerComparisonOp::Ne),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn array(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        Value::Tensor(runmat_builtins::Tensor::new_integer(storage, shape).expect("integer tensor"))
    }

    #[test]
    fn compares_signed_unsigned_and_uint64_exactly() {
        let lhs = Value::Int(IntValue::U64(u64::MAX));
        let rhs = Value::Int(IntValue::I64(i64::MAX));
        assert_eq!(
            try_integer_comparison(&lhs, &rhs, IntegerComparisonOp::Gt).expect("comparison"),
            Some(Value::Bool(true))
        );
        assert_eq!(
            try_integer_comparison(
                &Value::Int(IntValue::I8(-1)),
                &Value::Int(IntValue::U8(0)),
                IntegerComparisonOp::Lt,
            )
            .expect("comparison"),
            Some(Value::Bool(true))
        );
    }

    #[test]
    fn broadcasts_exact_integer_arrays_for_all_relations() {
        let lhs = array(IntegerStorage::U64(vec![0, u64::MAX]), vec![2, 1]);
        let rhs = array(IntegerStorage::I64(vec![0, 1, i64::MAX]), vec![1, 3]);
        let result = try_integer_comparison(&lhs, &rhs, IntegerComparisonOp::Ge)
            .expect("comparison")
            .expect("integer path");
        assert_eq!(
            result,
            Value::LogicalArray(
                LogicalArray::new(vec![1, 1, 0, 1, 0, 1], vec![2, 3]).expect("logical result")
            )
        );
    }

    #[test]
    fn compares_integer_storage_to_scalar_double_without_64_bit_loss() {
        let exact = Value::Int(IntValue::U64((1_u64 << 53) + 1));
        let rounded = Value::Num((1_u64 << 53) as f64);
        assert_eq!(
            try_integer_comparison(&exact, &rounded, IntegerComparisonOp::Eq).expect("comparison"),
            Some(Value::Bool(false))
        );
        assert_eq!(
            try_integer_comparison(&exact, &rounded, IntegerComparisonOp::Gt).expect("comparison"),
            Some(Value::Bool(true))
        );

        let tensor = array(IntegerStorage::U64(vec![0, (1_u64 << 53) + 1]), vec![1, 2]);
        assert_eq!(
            try_integer_comparison(&tensor, &rounded, IntegerComparisonOp::Ne).expect("comparison"),
            Some(Value::LogicalArray(
                LogicalArray::new(vec![1, 1], vec![1, 2]).expect("logical result")
            ))
        );
    }

    #[test]
    fn compares_integer_storage_to_broadcast_float_arrays_without_64_bit_loss() {
        let integer = array(
            IntegerStorage::U64(vec![1_u64 << 53, (1_u64 << 53) + 1]),
            vec![2, 1],
        );
        let float = Value::Tensor(
            runmat_builtins::Tensor::new(
                vec![(1_u64 << 53) as f64, 0.0, (1_u64 << 53) as f64],
                vec![1, 3],
            )
            .expect("float tensor"),
        );
        let result = try_integer_comparison(&integer, &float, IntegerComparisonOp::Eq)
            .expect("comparison")
            .expect("integer path");
        assert_eq!(
            result,
            Value::LogicalArray(
                LogicalArray::new(vec![1, 0, 0, 0, 1, 0], vec![2, 3]).expect("logical result")
            )
        );

        let result = try_integer_comparison(&float, &integer, IntegerComparisonOp::Lt)
            .expect("comparison")
            .expect("integer path");
        assert_eq!(
            result,
            Value::LogicalArray(
                LogicalArray::new(vec![0, 1, 1, 1, 0, 1], vec![2, 3]).expect("logical result")
            )
        );
    }

    #[test]
    fn compares_integer_storage_to_logical_arrays() {
        let integer = array(IntegerStorage::I8(vec![0, 1]), vec![1, 2]);
        let logical =
            Value::LogicalArray(LogicalArray::new(vec![0, 1], vec![1, 2]).expect("logical array"));
        assert_eq!(
            try_integer_comparison(&integer, &logical, IntegerComparisonOp::Eq)
                .expect("comparison"),
            Some(Value::LogicalArray(
                LogicalArray::new(vec![1, 1], vec![1, 2]).expect("logical result")
            ))
        );
    }
}
