//! Random and constant initializer algorithms; no topology interpretation.
use super::*;
use plan::BuiltinInitializer as I;
use runmat_builtins::DeepLearningLayerKind;

fn constant(count: usize, value: f64, function: &'static str) -> BuiltinResult<Vec<f64>> {
    let mut out = Vec::new();
    out.try_reserve_exact(count).map_err(|_| {
        deep_learning_error(
            function,
            format!("{function}: cannot allocate initialized parameter"),
        )
    })?;
    out.resize(count, value);
    Ok(out)
}

pub(super) fn fans(shape: &[usize], function: &'static str) -> BuiltinResult<(usize, usize)> {
    match shape {
        [rows, columns] => Ok((*columns, *rows)),
        [filter, channels, filters] => {
            let input = filter.checked_mul(*channels);
            let output = filter.checked_mul(*filters);
            input.zip(output).ok_or_else(|| {
                deep_learning_error(
                    function,
                    format!("{function}: initializer fan geometry overflows"),
                )
            })
        }
        _ => Err(deep_learning_error(
            function,
            format!("{function}: random initializer requires a matrix or convolution kernel"),
        )),
    }
}

fn orthogonal(
    turn: &mut crate::builtins::common::random::RandomTurn,
    shape: &[usize],
    function: &'static str,
) -> BuiltinResult<Vec<f64>> {
    let [rows, columns] = shape else {
        return Err(deep_learning_error(
            function,
            format!("{function}: orthogonal initializer requires a matrix"),
        ));
    };
    let count = rows.checked_mul(*columns).ok_or_else(|| {
        deep_learning_error(
            function,
            format!("{function}: orthogonal initializer shape overflows"),
        )
    })?;
    let mut data = crate::builtins::common::random::generate_normal(turn, count, function)?;
    // Orthogonalize the shorter axis. For tall matrices columns are unit
    // vectors; for wide matrices rows are unit vectors. The payload stays in
    // the canonical column-major layout throughout.
    let vectors = (*rows).min(*columns);
    let length = (*rows).max(*columns);
    let offset = |vector: usize, coordinate: usize| {
        if rows >= columns {
            vector * *rows + coordinate
        } else {
            coordinate * *rows + vector
        }
    };
    for vector in 0..vectors {
        let mut norm = 0.0_f64;
        for coordinate in 0..length {
            let value = data[offset(vector, coordinate)];
            norm = norm.hypot(value);
        }
        if norm == 0.0 || !norm.is_finite() {
            return Err(deep_learning_error(
                function,
                format!("{function}: orthogonal initializer could not form a finite basis"),
            ));
        }
        for coordinate in 0..length {
            data[offset(vector, coordinate)] /= norm;
        }
        // Modified Gram-Schmidt projects each remaining vector against the
        // just normalized basis vector, improving stability for tall gates.
        for later in vector + 1..vectors {
            let mut dot = 0.0;
            for coordinate in 0..length {
                dot += data[offset(vector, coordinate)] * data[offset(later, coordinate)];
            }
            for coordinate in 0..length {
                data[offset(later, coordinate)] -= dot * data[offset(vector, coordinate)];
            }
        }
    }
    Ok(data)
}

pub(super) fn builtin(
    turn: &mut crate::builtins::common::random::RandomTurn,
    initializer: I,
    kind: DeepLearningLayerKind,
    shape: &[usize],
    precision: NumericDType,
    function: &'static str,
) -> BuiltinResult<Value> {
    let count = shape
        .iter()
        .try_fold(1usize, |product, value| product.checked_mul(*value))
        .ok_or_else(|| {
            deep_learning_error(function, format!("{function}: initializer shape overflows"))
        })?;
    let values = match initializer {
        I::Zeros => constant(count, 0.0, function)?,
        I::Ones => constant(count, 1.0, function)?,
        I::UnitForget => {
            let [rows, 1] = shape else {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: unit-forget initializer requires column bias"),
                ));
            };
            let directions = if kind == DeepLearningLayerKind::Bilstm {
                2
            } else {
                1
            };
            if rows % (4 * directions) != 0 {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: invalid recurrent bias geometry"),
                ));
            }
            let units = rows / (4 * directions);
            let mut values = constant(count, 0.0, function)?;
            for direction in 0..directions {
                let start = direction * 4 * units + units;
                values[start..start + units].fill(1.0);
            }
            values
        }
        I::Glorot => {
            let (input, output) = fans(shape, function)?;
            let denominator = input.checked_add(output).ok_or_else(|| {
                deep_learning_error(
                    function,
                    format!("{function}: initializer fan geometry overflows"),
                )
            })?;
            let standard_deviation = (2.0 / denominator as f64).sqrt();
            crate::builtins::common::random::generate_normal_scaled(
                turn,
                0.0,
                standard_deviation,
                count,
                function,
            )?
        }
        I::He => {
            let (input, _) = fans(shape, function)?;
            let standard_deviation = (2.0 / input as f64).sqrt();
            crate::builtins::common::random::generate_normal_scaled(
                turn,
                0.0,
                standard_deviation,
                count,
                function,
            )?
        }
        I::NarrowNormal => crate::builtins::common::random::generate_normal_scaled(
            turn, 0.0, 0.01, count, function,
        )?,
        I::Orthogonal => orthogonal(turn, shape, function)?,
    };
    Tensor::new_with_dtype(values, shape.to_vec(), precision)
        .map(Value::Tensor)
        .map_err(|error| deep_learning_error(function, error))
}
