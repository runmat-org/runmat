//! MATLAB-compatible `issortedrows` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when rows are sorted according to the requested row order.",
}];

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix, character matrix, complex matrix, or table.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Column, direction, comparison, and missing-placement options.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "tf = issortedrows(A)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "tf = issortedrows(A, args...)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSORTEDROWS.INVALID_INPUT",
    identifier: Some("RunMat:issortedrows:InvalidInput"),
    when: "Input or row-sorting arguments are invalid.",
    message: "issortedrows: invalid input",
};

const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_INPUT];

pub const ISSORTEDROWS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "issortedrows",
    category = "array/sorting_sets",
    summary = "Determine whether matrix or table rows are sorted.",
    keywords = "issortedrows,sortrows,rows,sorted,table",
    accel = "sink",
    sink = true,
    descriptor(crate::builtins::array::sorting_sets::issortedrows::ISSORTEDROWS_DESCRIPTOR),
    builtin_path = "crate::builtins::array::sorting_sets::issortedrows"
)]
async fn issortedrows_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let evaluation = crate::builtins::array::sorting_sets::sortrows::evaluate(value, &rest).await?;
    let indices = evaluation.indices_value();
    let sorted = match indices {
        Value::Tensor(tensor) => tensor
            .materialize_f64()
            .iter()
            .enumerate()
            .all(|(idx, value)| *value == idx as f64 + 1.0),
        Value::Num(value) => value == 1.0,
        Value::Int(value) => value.to_i64() == 1,
        _ => false,
    };
    Ok(Value::Bool(sorted))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerStorage, Tensor};

    #[test]
    fn issortedrows_detects_sorted_and_unsorted_numeric_rows() {
        let sorted = Value::Tensor(Tensor::new(vec![1.0, 2.0, 1.0, 3.0], vec![2, 2]).unwrap());
        assert_eq!(
            block_on(issortedrows_builtin(sorted, Vec::new())).unwrap(),
            Value::Bool(true)
        );
        let unsorted = Value::Tensor(Tensor::new(vec![2.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap());
        assert_eq!(
            block_on(issortedrows_builtin(unsorted, Vec::new())).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            block_on(issortedrows_builtin(Value::Num(1.0), Vec::new())).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn issortedrows_uses_exact_integer_row_ordering() {
        let sorted = Tensor::new_integer(
            IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX, 0, 1, 2]),
            vec![3, 2],
        )
        .expect("input");
        assert_eq!(
            block_on(issortedrows_builtin(Value::Tensor(sorted), Vec::new())).unwrap(),
            Value::Bool(true)
        );

        let unsorted = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993, 0, 2, 1, 0]),
            vec![3, 2],
        )
        .expect("input");
        assert_eq!(
            block_on(issortedrows_builtin(Value::Tensor(unsorted), Vec::new())).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn issortedrows_reads_mirrorless_integer_storage_through_sortrows() {
        let sorted = Tensor::new_integer(
            IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX, 0, 1, 2]),
            vec![3, 2],
        )
        .expect("input");
        assert_eq!(
            block_on(issortedrows_builtin(Value::Tensor(sorted), Vec::new())).unwrap(),
            Value::Bool(true)
        );

        let unsorted =
            Tensor::new_integer(IntegerStorage::I64(vec![10, 3, 2, 9, 4, 1]), vec![3, 2])
                .expect("input");
        assert_eq!(
            block_on(issortedrows_builtin(Value::Tensor(unsorted), Vec::new())).unwrap(),
            Value::Bool(false)
        );
    }
}
