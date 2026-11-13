//! Exact-order solver scans and final column-major publication.

use super::{
    checkpoint, optim_error, LeastSquaresEvaluator, LeastSquaresResult, ScalarCooperation,
};
use crate::BuiltinResult;
use runmat_value::{Tensor, Value};

const SCAN_CHUNK: usize = 4096;

pub(super) async fn first_order_optimality<E: LeastSquaresEvaluator>(
    jacobian: &[f64],
    residual: &[f64],
    n: usize,
    evaluator: &mut E,
) -> BuiltinResult<f64> {
    let mut maximum = 0.0_f64;
    for column in 0..n {
        let mut value = 0.0;
        for (chunk_index, chunk) in residual.chunks(SCAN_CHUNK).enumerate() {
            let first = chunk_index * SCAN_CHUNK;
            for (offset, residual) in chunk.iter().enumerate() {
                value += jacobian[(first + offset) * n + column] * residual;
            }
            checkpoint(evaluator, chunk.len()).await?;
        }
        if !value.is_finite() {
            return Ok(value.abs());
        }
        maximum = maximum.max(value.abs());
    }
    Ok(maximum)
}

pub(super) async fn norm2_squared<E: LeastSquaresEvaluator>(
    values: &[f64],
    evaluator: &mut E,
) -> BuiltinResult<f64> {
    // Preserve Iterator::sum's floating identity and the original left-to-right accumulation.
    let mut total = -0.0;
    for chunk in values.chunks(SCAN_CHUNK) {
        for value in chunk {
            total += value * value;
        }
        checkpoint(evaluator, chunk.len()).await?;
    }
    Ok(total)
}

pub(crate) async fn residual_norm_inf(
    values: &[f64],
    mut cooperation: Option<&mut ScalarCooperation>,
) -> BuiltinResult<f64> {
    let mut maximum = 0.0_f64;
    for chunk in values.chunks(SCAN_CHUNK) {
        for value in chunk {
            if !value.is_finite() {
                return Ok(value.abs());
            }
            maximum = maximum.max(value.abs());
        }
        if let Some(cooperation) = cooperation.as_deref_mut() {
            cooperation.completed(chunk.len() as u64).await?;
        }
    }
    Ok(maximum)
}

pub(super) async fn max_abs_difference<E: LeastSquaresEvaluator>(
    lhs: &[f64],
    rhs: &[f64],
    evaluator: &mut E,
) -> BuiltinResult<f64> {
    let mut maximum = 0.0_f64;
    for (lhs, rhs) in lhs.chunks(SCAN_CHUNK).zip(rhs.chunks(SCAN_CHUNK)) {
        for (a, b) in lhs.iter().zip(rhs) {
            maximum = maximum.max((a - b).abs());
        }
        checkpoint(evaluator, lhs.len().min(rhs.len())).await?;
    }
    Ok(maximum)
}

pub(crate) async fn jacobian_value(
    name: &str,
    result: &LeastSquaresResult,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Value> {
    cooperation.completed(0).await?;
    if result.point_evidence == super::PointEvidence::Supplied {
        return crate::builtins::math::optim::common::output_callback::empty(cooperation.runtime());
    }
    if let super::PublishedJacobian::Supplied(source) = &result.jacobian {
        // The admitted variant contains only the original immutable numeric
        // carrier. Tensor/CSC clones share their coefficient/topology backing.
        return Ok(source.value().clone());
    }
    let values = result
        .jacobian
        .row_major()
        .expect("dense publication variant");
    let rows = result.residual_len;
    let columns = result.variable_len;
    let mut column_major = Vec::with_capacity(values.len());
    for column in 0..columns {
        for first in (0..rows).step_by(SCAN_CHUNK) {
            let end = rows.min(first + SCAN_CHUNK);
            for row in first..end {
                column_major.push(values[row * columns + column]);
            }
            cooperation.completed((end - first) as u64).await?;
        }
    }
    Tensor::new(column_major, vec![rows, columns])
        .map(Value::Tensor)
        .map_err(|error| optim_error(name, format!("{name}: {error}")))
}

pub(crate) async fn bound_gradient(
    result: &LeastSquaresResult,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    cooperation.completed(0).await?;
    let n = result.variable_len;
    if let Some(gradient) = &result.gradient {
        if gradient.len() != n {
            return Err(optim_error(
                "lsqnonlin",
                "accepted gradient cache geometry is inconsistent",
            ));
        }
        let mut output = runmat_value::exact_capacity_vec(n).map_err(|source| {
            crate::build_runtime_error("gradient publication allocation failed")
                .with_source(source)
                .build()
        })?;
        for chunk in gradient.chunks(SCAN_CHUNK) {
            output.extend_from_slice(chunk);
            cooperation.completed(chunk.len() as u64).await?;
        }
        return Ok(output);
    }
    let values = result.jacobian.row_major();
    let mut output = vec![0.0; n];
    if result.point_evidence == super::PointEvidence::Supplied
        || result.residual_len.checked_mul(n) != values.map(<[f64]>::len)
    {
        for chunk in output.chunks_mut(SCAN_CHUNK) {
            chunk.fill(f64::NAN);
            cooperation.completed(chunk.len() as u64).await?;
        }
        return Ok(output);
    }
    let values = values.expect("checked row-major derivative geometry");
    for (column, output) in output.iter_mut().enumerate() {
        for (chunk_index, chunk) in result.residual.chunks(SCAN_CHUNK).enumerate() {
            let first = chunk_index * SCAN_CHUNK;
            for (offset, residual) in chunk.iter().enumerate() {
                *output += values[(first + offset) * n + column] * residual;
            }
            cooperation.completed(chunk.len() as u64).await?;
        }
    }
    Ok(output)
}

pub(crate) async fn original_derivative_columns(
    name: &str,
    result: &mut LeastSquaresResult,
    bounds: &super::LeastSquaresBounds,
    mut cooperation: Option<&mut ScalarCooperation>,
) -> BuiltinResult<()> {
    if result.jacobian_source == super::JacobianSource::Supplied {
        return Ok(());
    }
    if bounds.lower.len() != result.variable_len || bounds.upper.len() != result.variable_len {
        return Err(optim_error(
            name,
            "fixed derivative geometry disagrees with the accepted point",
        ));
    }
    let super::PublishedJacobian::RowMajor(values) = &mut result.jacobian else {
        return Err(optim_error(
            name,
            "finite-difference result lost its numerical workspace",
        ));
    };
    if result.residual_len.checked_mul(result.variable_len) != Some(values.len()) {
        return Ok(());
    }
    for column in 0..result.variable_len {
        if bounds.lower[column] == bounds.upper[column] {
            for first in (0..result.residual_len).step_by(1024) {
                let end = result.residual_len.min(first + 1024);
                for row in first..end {
                    values[row * result.variable_len + column] = f64::NAN;
                }
                if let Some(c) = cooperation.as_deref_mut() {
                    c.completed((end - first) as u64).await?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    struct ScanOnly;
    impl LeastSquaresEvaluator for ScanOnly {
        fn residual<'a>(&'a mut self, _: &'a [f64]) -> super::super::ResidualFuture<'a> {
            panic!("a final traversal must not replay the callback")
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn final_traversals_preserve_column_major_publication_gradient_and_sum_order() {
        let result = LeastSquaresResult {
            point_evidence: crate::builtins::math::optim::least_squares::PointEvidence::Evaluated,
            gradient: None,
            cg_iterations: 0,
            jacobian_source:
                crate::builtins::math::optim::least_squares::JacobianSource::FiniteDifference,
            method: super::super::Method::LevenbergMarquardt,
            x: vec![0.0; 2],
            residual: vec![5.0, 6.0],
            jacobian: crate::builtins::math::optim::least_squares::PublishedJacobian::RowMajor(
                vec![1.0, 2.0, 3.0, 4.0],
            ),
            residual_len: 2,
            variable_len: 2,
            resnorm: 61.0,
            exitflag: 1,
            iterations: 0,
            func_count: 1,
            first_order_optimality: 34.0,
            step_size: 0.0,
            message: String::new(),
        };
        let mut cooperation = ScalarCooperation::new(runmat_builtins::OptimSolver::Lsqnonlin, None);
        assert_eq!(
            block_on(bound_gradient(&result, &mut cooperation)).unwrap(),
            vec![23.0, 34.0]
        );
        let Value::Tensor(jacobian) =
            block_on(jacobian_value("lsqnonlin", &result, &mut cooperation)).unwrap()
        else {
            panic!("Jacobian is a dense matrix")
        };
        assert_eq!(jacobian.shape(), &[2, 2]);
        assert_eq!(jacobian.as_f64_slice().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
        assert_eq!(
            block_on(first_order_optimality(
                result
                    .jacobian
                    .row_major()
                    .expect("fixture numerical workspace"),
                &result.residual,
                2,
                &mut ScanOnly
            ))
            .unwrap(),
            34.0
        );
        assert!(
            block_on(first_order_optimality(&[f64::NAN], &[1.], 1, &mut ScanOnly))
                .unwrap()
                .is_nan()
        );
        assert!(block_on(residual_norm_inf(&[f64::NAN], None))
            .unwrap()
            .is_nan());
        let mut values = vec![1.0; SCAN_CHUNK + 3];
        values[0] = 1.0e8;
        let expected: f64 = values.iter().map(|value| value * value).sum();
        let actual = block_on(norm2_squared(&values, &mut ScanOnly)).unwrap();
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "chunk boundaries must not regroup the sum"
        );
        assert_eq!(
            block_on(residual_norm_inf(&[-7.0, 2.0], None)).unwrap(),
            7.0
        );
        assert_eq!(
            block_on(max_abs_difference(
                &[-7.0, 2.0],
                &[1.0, -3.0],
                &mut ScanOnly
            ))
            .unwrap(),
            8.0
        );
    }
}
