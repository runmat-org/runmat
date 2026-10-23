//! Cluster selection and stable Schur-block ordering.

use super::form::SchurBlock;
use super::{
    factor_from_values, failure, invalid, option_text, output_pair, validate_pair_outputs,
    SchurContext, ORDSCHUR,
};
use crate::BuiltinResult;
use num_complex::Complex64;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

enum Selection {
    Selected(Vec<bool>),
    Clusters(Vec<u64>),
}

#[derive(Clone, Copy)]
enum Region {
    LeftHalfPlane,
    RightHalfPlane,
    UnitDiskInterior,
    UnitDiskExterior,
}

fn is_vector(shape: &[usize], length: usize) -> bool {
    match shape {
        [one] => *one == length,
        [rows, columns] => {
            (length == 0 || *rows == 1 || *columns == 1)
                && rows.checked_mul(*columns) == Some(length)
        }
        _ => false,
    }
}

fn selection(
    value: &Value,
    eigenvalues: &[Complex64],
    name: SchurContext,
) -> BuiltinResult<Selection> {
    let n = eigenvalues.len();
    if let Some(keyword) = option_text(value, name)? {
        let region = match keyword.trim() {
            "lhp" => Region::LeftHalfPlane,
            "rhp" => Region::RightHalfPlane,
            "udi" => Region::UnitDiskInterior,
            "udo" => Region::UnitDiskExterior,
            _ => {
                return Err(invalid(
                    name,
                    "selector must be 'lhp', 'rhp', 'udi', or 'udo'",
                ))
            }
        };
        let selected = eigenvalues
            .iter()
            .map(|eigenvalue| match region {
                Region::LeftHalfPlane => eigenvalue.re < 0.0,
                Region::RightHalfPlane => eigenvalue.re >= 0.0,
                Region::UnitDiskInterior => eigenvalue.norm() < 1.0,
                Region::UnitDiskExterior => eigenvalue.norm() >= 1.0,
            })
            .collect::<Vec<_>>();
        return Ok(Selection::Selected(selected));
    }
    match value {
        Value::Bool(value) if n == 1 => Ok(Selection::Selected(vec![*value])),
        Value::LogicalArray(values) if is_vector(&values.shape, n) => {
            Ok(Selection::Selected(values.data.iter().map(|&value| value != 0).collect()))
        }
        Value::Tensor(values) if is_vector(values.shape(), n) && matches!(values.numeric_dtype(), runmat_value::NumericDType::F32 | runmat_value::NumericDType::F64) => {
            let clusters = values
                .materialize_f64()
                .into_iter()
                .map(|value| {
                    if value.is_finite() && value >= 1.0 && value.fract() == 0.0 && value <= (1_u64 << 53) as f64 {
                        Ok(value as u64)
                    } else {
                        Err(invalid(name, "cluster indices must be positive integers"))
                    }
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            Ok(Selection::Clusters(clusters))
        }
        Value::Num(value) if n == 1 && value.is_finite() && *value >= 1.0 && value.fract() == 0.0 && *value <= (1_u64 << 53) as f64 => {
            Ok(Selection::Clusters(vec![*value as u64]))
        }
        _ => Err(invalid(name, "selector must be a logical vector, region keyword, or positive-integer cluster vector matching Schur order")),
    }
}

fn block_order(blocks: &[SchurBlock], selection: Selection) -> Vec<usize> {
    let mut order = (0..blocks.len()).collect::<Vec<_>>();
    match selection {
        Selection::Selected(values) => {
            order.sort_by_key(|&id| {
                let block = blocks[id];
                !values[block.start..block.start + block.size]
                    .iter()
                    .any(|&value| value)
            });
        }
        Selection::Clusters(values) => {
            order.sort_by_key(|&id| {
                let block = blocks[id];
                std::cmp::Reverse(
                    *values[block.start..block.start + block.size]
                        .iter()
                        .max()
                        .expect("Schur blocks are nonempty"),
                )
            });
        }
    }
    order
}

#[runtime_builtin(
    name = "ordschur",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::schur"
)]
async fn ordschur_builtin(
    vectors: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    const NAME: SchurContext = ORDSCHUR;
    validate_pair_outputs(NAME)?;
    let (2, Some(triangular), Some(selector)) = (rest.len(), rest.get(0), rest.get(1)) else {
        return Err(invalid(NAME, "ordschur expects U, T, and a selector"));
    };
    let (mut form, dtype) = factor_from_values(vectors, triangular.clone(), NAME)?;
    let eigenvalues = form.eigenvalues();
    let blocks = form.blocks();
    let selection = selection(selector, &eigenvalues, NAME)?;
    let order = block_order(&blocks, selection);
    form.reorder(&order)
        .map_err(|message| failure(NAME, message))?;
    output_pair(form, dtype, NAME, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_selection_and_larger_cluster_label_are_indivisible() {
        let blocks = [
            SchurBlock { start: 0, size: 1 },
            SchurBlock { start: 1, size: 2 },
        ];
        assert_eq!(
            block_order(&blocks, Selection::Selected(vec![false, false, true])),
            vec![1, 0]
        );
        assert_eq!(
            block_order(&blocks, Selection::Clusters(vec![2, 1, 3])),
            vec![1, 0]
        );
    }
}
