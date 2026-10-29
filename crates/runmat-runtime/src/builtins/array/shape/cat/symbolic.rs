//! Symbolic assembly shares the checked N-D column-major concatenation owner.

use crate::builtins::common::concatenation::symbolic::symbolic_block_from_value;
use crate::BuiltinResult;
use runmat_value::{SymbolicArray, Value};

pub(crate) fn concatenate<'a>(
    dim_zero: usize,
    values: impl Iterator<Item = &'a Value>,
) -> BuiltinResult<Value> {
    let mut blocks = Vec::new();
    for value in values {
        if let Some(block) = symbolic_block_from_value(value)? {
            blocks.push(block);
        }
    }
    if blocks.is_empty() {
        return SymbolicArray::new(Vec::new(), vec![0, 0])
            .map(Value::SymbolicArray)
            .map_err(super::cat_err);
    }
    let shapes: Vec<_> = blocks.iter().map(|block| block.shape.as_slice()).collect();
    let data: Vec<_> = blocks.iter().map(|block| block.data.as_slice()).collect();
    let rank = shapes
        .iter()
        .map(|shape| shape.len())
        .max()
        .unwrap_or(2)
        .max(dim_zero + 1);
    // Only [] is neutral for the existing symbolic conversion. Shaped empty
    // arrays still participate in non-concatenated dimension validation.
    let (data, mut shape) =
        super::concat_column_major_with_neutral(dim_zero, &shapes, &data, "cat", |shape| {
            shape[0] == 0 && shape[1] == 0
        })?;
    // Symbolic arrays preserve explicitly authored trailing dimensions.
    shape.resize(rank, 1);
    SymbolicArray::new(data, shape)
        .map(Value::SymbolicArray)
        .map_err(super::cat_err)
}

#[cfg(test)]
mod tests {
    use super::super::{
        cat_builtin as execute_cat, determine_category, CatCategory, LikeDevice, LikeSpec,
    };
    use crate::call::BuiltinArguments;
    use futures::executor::block_on;
    use runmat_value::{Tensor, Value};

    fn cat_builtin(dim: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(execute_cat(
            None,
            dim,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    #[test]
    fn symbolic_cat_shares_nd_geometry_and_preserves_symbolic_empty_rules() {
        use runmat_value::{SymbolicArray, SymbolicExpr};

        let symbolic = Value::SymbolicArray(
            SymbolicArray::new(
                ["a", "b", "c", "d"].map(SymbolicExpr::variable).to_vec(),
                vec![1, 2, 2],
            )
            .unwrap(),
        );
        let numeric = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]).unwrap());
        for (dimension, shape, expected) in [
            (
                1,
                vec![2, 2, 2],
                vec!["a", "1", "b", "2", "c", "3", "d", "4"],
            ),
            (
                2,
                vec![1, 4, 2],
                vec!["a", "b", "1", "2", "c", "d", "3", "4"],
            ),
            (
                3,
                vec![1, 2, 4],
                vec!["a", "b", "c", "d", "1", "2", "3", "4"],
            ),
            (
                4,
                vec![1, 2, 2, 2],
                vec!["a", "b", "c", "d", "1", "2", "3", "4"],
            ),
        ] {
            let Value::SymbolicArray(result) = cat_builtin(
                Value::Num(f64::from(dimension)),
                vec![symbolic.clone(), numeric.clone()],
            )
            .unwrap() else {
                panic!("expected symbolic result")
            };
            assert_eq!(result.shape, shape);
            assert_eq!(
                result
                    .data()
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                expected
            );
        }
        let empty = Value::SymbolicArray(SymbolicArray::new(Vec::new(), vec![0, 0]).unwrap());
        let Value::SymbolicArray(result) =
            cat_builtin(Value::Num(2.0), vec![empty.clone(), symbolic.clone()]).unwrap()
        else {
            panic!("expected symbolic result")
        };
        assert_eq!(result.shape, vec![1, 2, 2]);
        let Value::SymbolicArray(result) =
            cat_builtin(Value::Num(2.0), vec![empty.clone(), empty]).unwrap()
        else {
            panic!("expected symbolic empty")
        };
        assert_eq!(result.shape, vec![0, 0]);
        let shaped_empty =
            Value::SymbolicArray(SymbolicArray::new(Vec::new(), vec![0, 3, 2]).unwrap());
        assert!(cat_builtin(Value::Num(1.0), vec![symbolic.clone(), shaped_empty]).is_err());
        let trailing = Value::SymbolicArray(
            SymbolicArray::new(
                vec![SymbolicExpr::variable("x"), SymbolicExpr::variable("y")],
                vec![1, 2, 1],
            )
            .unwrap(),
        );
        let Value::SymbolicArray(result) =
            cat_builtin(Value::Num(2.0), vec![trailing.clone(), trailing.clone()]).unwrap()
        else {
            panic!("expected symbolic result")
        };
        assert_eq!(result.shape, vec![1, 4, 1]);
        let inputs = vec![trailing.clone(), trailing];
        let inputs = BuiltinArguments::from_borrowed(&inputs, None).unwrap();
        assert!(determine_category(
            inputs.view(),
            &LikeSpec::from_prototype(&Value::Num(0.0)).unwrap()
        )
        .is_err());
        assert!(LikeSpec {
            device: LikeDevice::Gpu,
            ..LikeSpec::default()
        }
        .ensure_device(CatCategory::Symbolic)
        .is_err());
        assert!(cat_builtin(Value::Num(2.0), vec![symbolic, Value::String("x".into())]).is_err());
    }
}
