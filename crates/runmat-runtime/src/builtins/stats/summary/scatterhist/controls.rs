//! Scatterhist bounded numeric controls.
use super::*;

pub(super) fn control_matrix_dimensions(tensor: &Tensor) -> (usize, usize) {
    match tensor.shape() {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

pub(super) fn parse_bins(value: &Value) -> BuiltinResult<[usize; 2]> {
    let tensor = tensor::value_to_tensor(value)
        .map_err(|_| invalid("scatterhist: NBins must be numeric"))?;
    if let Some(values) = integer_values(&tensor) {
        return match values.as_slice() {
            [n] => Ok([bin_count_integer(n)?, bin_count_integer(n)?]),
            [nx, ny] => Ok([bin_count_integer(nx)?, bin_count_integer(ny)?]),
            _ => Err(invalid(
                "scatterhist: NBins must be a scalar or two-element vector",
            )),
        };
    }
    let values = tensor_values_f64(&tensor);
    match values.as_slice() {
        [n] => Ok([bin_count(*n)?, bin_count(*n)?]),
        [nx, ny] => Ok([bin_count(*nx)?, bin_count(*ny)?]),
        _ => Err(invalid(
            "scatterhist: NBins must be a scalar or two-element vector",
        )),
    }
}

pub(super) fn bin_count(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 2.0 || value.fract() != 0.0 {
        return Err(invalid(
            "scatterhist: NBins entries must be integers greater than or equal to 2",
        ));
    }
    if value > MAX_BINS as f64 {
        return Err(invalid(format!(
            "scatterhist: NBins entries must be no greater than {MAX_BINS}"
        )));
    }
    let value = value as usize;
    if value > MAX_BINS {
        return Err(invalid(format!(
            "scatterhist: NBins entries must be no greater than {MAX_BINS}"
        )));
    }
    Ok(value)
}

pub(super) fn bin_count_integer(value: &IntValue) -> BuiltinResult<usize> {
    let Some(value) = value.try_to_usize() else {
        return Err(invalid(
            "scatterhist: NBins entries must be integers greater than or equal to 2",
        ));
    };
    if value < 2 {
        return Err(invalid(
            "scatterhist: NBins entries must be integers greater than or equal to 2",
        ));
    }
    if value > MAX_BINS {
        return Err(invalid(format!(
            "scatterhist: NBins entries must be no greater than {MAX_BINS}"
        )));
    }
    Ok(value)
}

pub(super) fn parse_bandwidth_cycle(value: &Value) -> BuiltinResult<Vec<[f64; 2]>> {
    let tensor = tensor::value_to_tensor(value)
        .map_err(|_| invalid("scatterhist: Bandwidth must be numeric"))?;
    if tensor.shape().iter().skip(2).any(|&extent| extent > 1) {
        return Err(invalid("scatterhist: Bandwidth must be a 2-D matrix"));
    }
    let values = tensor_values_f64(&tensor);
    let (rows, cols) = control_matrix_dimensions(&tensor);
    // The documented 2-by-K form wins for the ambiguous 2-by-2 matrix.
    // Retain the earlier RunMat K-by-2 form when K is not two.
    if rows == 2 && cols > 0 {
        let mut bandwidths = Vec::with_capacity(cols);
        for column in 0..cols {
            bandwidths.push([
                bandwidth_value(values[2 * column])?,
                bandwidth_value(values[2 * column + 1])?,
            ]);
        }
        return Ok(bandwidths);
    }
    if rows > 0 && cols == 2 {
        let mut bandwidths = Vec::with_capacity(rows);
        for row in 0..rows {
            bandwidths.push([
                bandwidth_value(values[row])?,
                bandwidth_value(values[rows + row])?,
            ]);
        }
        return Ok(bandwidths);
    }
    match values.as_slice() {
        [bw] => {
            let bw = bandwidth_value(*bw)?;
            Ok(vec![[bw, bw]])
        }
        [bx, by] => Ok(vec![[bandwidth_value(*bx)?, bandwidth_value(*by)?]]),
        _ => Err(invalid(
            "scatterhist: Bandwidth must be a scalar, two-element vector, 2-by-K matrix, or legacy K-by-2 matrix",
        )),
    }
}

pub(super) fn bandwidth_value(value: f64) -> BuiltinResult<f64> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid("scatterhist: Bandwidth values must be positive"));
    }
    Ok(value)
}

pub(super) fn nonnegative_scalar_cycle(value: &Value, name: &str) -> BuiltinResult<Vec<f32>> {
    let tensor = tensor::value_to_tensor(value)
        .map_err(|_| invalid(format!("scatterhist: {name} must be numeric")))?;
    let values = tensor_values_f64(&tensor);
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(invalid(format!(
            "scatterhist: {name} values must be nonnegative"
        )));
    }
    Ok(values.iter().map(|value| *value as f32).collect())
}

pub(super) fn option_bool(value: &Value, name: &str) -> BuiltinResult<bool> {
    if let Some(text) = scalar_text(value)? {
        return match text.trim().to_ascii_lowercase().as_str() {
            "on" | "true" | "yes" => Ok(true),
            "off" | "false" | "no" => Ok(false),
            other => Err(invalid(format!(
                "scatterhist: unsupported {name} value '{other}'"
            ))),
        };
    }
    let tensor = tensor::value_to_tensor(value)
        .map_err(|_| invalid(format!("scatterhist: {name} must be logical")))?;
    if tensor::tensor_element_len(&tensor) != 1 {
        return Err(invalid(format!(
            "scatterhist: {name} must be a logical scalar"
        )));
    }
    if let Some(value) = tensor
        .integer_storage()
        .and_then(|storage| storage.value_at(0))
    {
        return match value.try_to_usize() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(invalid(format!("scatterhist: {name} must be 0 or 1"))),
        };
    }
    match tensor::tensor_value_f64(&tensor, 0) {
        0.0 => Ok(false),
        1.0 => Ok(true),
        _ => Err(invalid(format!("scatterhist: {name} must be 0 or 1"))),
    }
}

pub(super) fn parse_kernel(value: &Value) -> BuiltinResult<KernelMode> {
    if let Some(text) = scalar_text(value)? {
        return match text.trim().to_ascii_lowercase().as_str() {
            "on" | "true" | "yes" => Ok(KernelMode::On),
            "off" | "false" | "no" => Ok(KernelMode::Off),
            "overlay" => Ok(KernelMode::Overlay),
            other => Err(invalid(format!(
                "scatterhist: unsupported Kernel value '{other}'"
            ))),
        };
    }
    Ok(if option_bool(value, "Kernel")? {
        KernelMode::On
    } else {
        KernelMode::Off
    })
}

pub(super) fn parse_style(value: &Value) -> BuiltinResult<(KernelMode, HistStyle)> {
    let text = scalar_text(value)?.ok_or_else(|| invalid("scatterhist: Style must be text"))?;
    match text.trim().to_ascii_lowercase().as_str() {
        "bar" | "histogram" => Ok((KernelMode::Off, HistStyle::Bar)),
        "stairs" => Ok((KernelMode::Off, HistStyle::Stairs)),
        other => Err(invalid(format!(
            "scatterhist: unsupported Style value '{other}'"
        ))),
    }
}

pub(super) fn parse_location(value: &Value) -> BuiltinResult<Location> {
    let text = scalar_text(value)?.ok_or_else(|| invalid("scatterhist: Location must be text"))?;
    match text.trim().to_ascii_lowercase().as_str() {
        "northeast" => Ok(Location::NorthEast),
        "southeast" => Ok(Location::SouthEast),
        "southwest" => Ok(Location::SouthWest),
        "northwest" => Ok(Location::NorthWest),
        other => Err(invalid(format!(
            "scatterhist: unsupported Location value '{other}'"
        ))),
    }
}

pub(super) fn parse_direction(value: &Value) -> BuiltinResult<Direction> {
    let text = scalar_text(value)?.ok_or_else(|| invalid("scatterhist: Direction must be text"))?;
    match text.trim().to_ascii_lowercase().as_str() {
        "out" => Ok(Direction::Out),
        "in" => Ok(Direction::In),
        other => Err(invalid(format!(
            "scatterhist: unsupported Direction value '{other}'"
        ))),
    }
}
