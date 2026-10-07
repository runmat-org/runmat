use super::detection::{any_missing, logical_array_mask_for_rows};
use super::numeric::{is_missing_text, numeric_scalar, scalar_text, scalar_usize};
use super::*;
use runmat_accelerate_api::{
    handle_integer_type, handle_is_logical, handle_storage, GpuTensorStorage,
};
use runmat_builtins::{FillMissingMethodKind, FillMissingOptionKind};

pub(super) mod numeric;
mod lines;
mod runs;
mod text;
use lines::FillLines;

use numeric::{
    fill_linear_numeric, fill_nearest_numeric, fill_neighbor_numeric, fill_summary_numeric,
    Neighbor, Summary,
};
use text::{fill_nearest_text, fill_neighbor_text};

#[derive(Clone)]
pub(super) enum FillMethod {
    Constant(Value),
    Previous,
    Next,
    Nearest,
    Linear,
    Mean,
    Median,
}

#[derive(Clone)]
pub(super) struct FillOptions {
    method: FillMethod,
    dim: Option<usize>,
}

impl FillOptions {
    pub(super) fn parse(args: &[Value]) -> BuiltinResult<Self> {
        if args.is_empty() {
            return Err(invalid_argument("fillmissing: method is required"));
        }
        let mut idx = 0;
        let method_text = scalar_text(&args[idx])
            .ok_or_else(|| invalid_argument("fillmissing: method must be a string"))?
            .to_ascii_lowercase();
        idx += 1;
        let method_kind = FillMissingMethodKind::parse(&method_text).ok_or_else(|| {
            invalid_argument(format!("fillmissing: unsupported method '{method_text}'"))
        })?;
        let method = match method_kind {
            FillMissingMethodKind::Constant => {
                let fill = args
                    .get(idx)
                    .ok_or_else(|| {
                        invalid_argument("fillmissing: constant method needs a fill value")
                    })?
                    .clone();
                idx += 1;
                FillMethod::Constant(fill)
            }
            FillMissingMethodKind::Previous => FillMethod::Previous,
            FillMissingMethodKind::Next => FillMethod::Next,
            FillMissingMethodKind::Nearest => FillMethod::Nearest,
            FillMissingMethodKind::Linear => FillMethod::Linear,
            FillMissingMethodKind::Mean => FillMethod::Mean,
            FillMissingMethodKind::Median => FillMethod::Median,
        };
        let mut dim = None;
        while idx < args.len() {
            if let Some(text) = scalar_text(&args[idx]) {
                if let Some(FillMissingOptionKind::Dimension) = FillMissingOptionKind::parse(&text)
                {
                    let value = args
                        .get(idx + 1)
                        .ok_or_else(|| invalid_argument("fillmissing: 'dim' requires a value"))?;
                    dim = Some(scalar_usize(value, "fillmissing dimension")?);
                    idx += 2;
                    continue;
                }
                return Err(invalid_argument(format!(
                    "fillmissing: unsupported option '{text}'"
                )));
            }
            if matches!(args[idx], Value::Num(_) | Value::Int(_))
                || matches!(&args[idx], Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor))
            {
                dim = Some(scalar_usize(&args[idx], "fillmissing dimension")?);
                idx += 1;
                continue;
            }
            return Err(invalid_argument(format!(
                "fillmissing: unsupported option argument {:?}",
                args[idx]
            )));
        }
        Ok(Self { method, dim })
    }

    /// Reject host-visible invalid calls before a resident operand can be
    /// materialized. The same typed method and dimension drive execution.
    pub(super) fn preflight(&self, value: &Value) -> BuiltinResult<()> {
        match value {
            Value::Tensor(tensor) => {
                self.preflight_numeric(tensor.shape(), tensor.integer_storage().is_some())
            }
            Value::GpuTensor(handle) => {
                if handle_storage(handle) != GpuTensorStorage::Real {
                    return Err(unsupported_type(
                        "fillmissing: complex arrays are not supported yet",
                    ));
                }
                if handle_is_logical(handle) {
                    FillLines::new(handle.shape(), self.dim)?;
                    return Ok(());
                }
                self.preflight_numeric(handle.shape(), handle_integer_type(handle).is_some())
            }
            Value::LogicalArray(array) => {
                FillLines::new(&array.shape, self.dim)?;
                Ok(())
            }
            Value::StringArray(array) => {
                FillLines::new(&array.shape, self.dim)?;
                match &self.method {
                    FillMethod::Constant(fill) => {
                        scalar_text(fill).ok_or_else(|| {
                            invalid_argument("fillmissing: string constant must be text")
                        })?;
                        Ok(())
                    }
                    FillMethod::Previous | FillMethod::Next | FillMethod::Nearest => Ok(()),
                    _ => Err(unsupported_type(
                        "fillmissing: string arrays support constant, previous, next, and nearest",
                    )),
                }
            }
            Value::Cell(_) => match &self.method {
                FillMethod::Constant(_) => Ok(()),
                _ => Err(unsupported_type(
                    "fillmissing: cell arrays currently support the constant method",
                )),
            },
            Value::Object(object) if is_tabular_object(object) => {
                let variables = table_variables(object)?;
                for child in variables.fields.values() {
                    self.preflight(child)?;
                }
                Ok(())
            }
            Value::ComplexTensor(_) => Err(unsupported_type(
                "fillmissing: complex arrays are not supported yet",
            )),
            Value::SparseTensor(_) => Err(unsupported_type(
                "fillmissing: sparse arrays are not supported yet",
            )),
            other => {
                if !matches!(&self.method, FillMethod::Constant(_)) && any_missing(other)? {
                    return Err(unsupported_type(
                        "fillmissing: scalar fill requires the constant method",
                    ));
                }
                Ok(())
            }
        }
    }

    fn preflight_numeric(&self, shape: &[usize], integer: bool) -> BuiltinResult<()> {
        FillLines::new(shape, self.dim)?;
        if !integer {
            if let FillMethod::Constant(fill) = &self.method {
                numeric_scalar(fill, "fillmissing constant")?;
            }
        }
        Ok(())
    }
}

pub(super) fn fill_missing_value(
    value: Value,
    options: &FillOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    match value {
        Value::Tensor(tensor) => fill_missing_tensor(tensor, options),
        Value::LogicalArray(array) => {
            lines::FillLines::new(&array.shape, options.dim)?;
            let mask =
                LogicalArray::from_host_buffer(vec![0; array.len()].into(), array.shape.clone())
                    .map_err(internal_error)?;
            Ok((Value::LogicalArray(array), mask))
        }
        Value::StringArray(array) => fill_missing_string_array(array, options),
        Value::Object(object) if is_tabular_object(&object) => fill_missing_table(object, options),
        Value::Cell(cell) => fill_missing_cell(cell, options),
        Value::ComplexTensor(_) => Err(unsupported_type(
            "fillmissing: complex arrays are not supported yet",
        )),
        Value::SparseTensor(_) => Err(unsupported_type(
            "fillmissing: sparse arrays are not supported yet",
        )),
        other => {
            let missing = any_missing(&other)?;
            let mask =
                LogicalArray::new(vec![u8::from(missing)], vec![1, 1]).map_err(internal_error)?;
            if missing {
                match &options.method {
                    FillMethod::Constant(fill) => Ok((fill.clone(), mask)),
                    _ => Err(unsupported_type(
                        "fillmissing: scalar fill requires the constant method",
                    )),
                }
            } else {
                Ok((other, mask))
            }
        }
    }
}

pub(super) fn fill_missing_table(
    mut object: ObjectInstance,
    options: &FillOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    let height = table_height(&object)?;
    let width = table_width(&object)?;
    let names = table_variable_names_from_object(&object)?;
    let variables = table_variables(&object)?;
    let mut output_vars = StructValue::new();
    let mut mask_data = vec![0u8; height * width];
    for (col, name) in names.iter().enumerate() {
        let value = variables
            .fields
            .get(name)
            .ok_or_else(|| internal_error(format!("table missing variable {name}")))?
            .clone();
        let (filled, mask) = fill_missing_value(value, options)?;
        let row_mask = logical_array_mask_for_rows(&mask, height)?;
        for row in 0..height {
            if row_mask.get(row).copied().unwrap_or(0) != 0 {
                mask_data[row + col * height] = 1;
            }
        }
        output_vars.insert(name.clone(), filled);
    }
    object
        .properties
        .insert("__table_variables".to_string(), Value::Struct(output_vars));
    Ok((
        Value::Object(object),
        LogicalArray::new(mask_data, vec![height, width]).map_err(internal_error)?,
    ))
}

pub(super) fn fill_missing_tensor(
    tensor: Tensor,
    options: &FillOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    let lines = FillLines::new(tensor.shape(), options.dim)?;
    if tensor.integer_storage().is_some() {
        let mask = LogicalArray::new(vec![0; tensor.len()], tensor.shape().to_vec())
            .map_err(internal_error)?;
        return Ok((Value::Tensor(tensor), mask));
    }
    let dtype = tensor.numeric_dtype();
    let shape = tensor.shape().to_vec();
    let mut data = tensor_utils::tensor_into_values_f64(tensor);
    let mut mask: Vec<u8> = data.iter().map(|value| u8::from(value.is_nan())).collect();
    match &options.method {
        FillMethod::Constant(fill) => {
            let fill = numeric_scalar(fill, "fillmissing constant")?;
            for (idx, value) in data.iter_mut().enumerate() {
                if mask[idx] != 0 {
                    *value = fill;
                }
            }
        }
        FillMethod::Mean => fill_summary_numeric(&mut data, &lines, Summary::Mean),
        FillMethod::Median => fill_summary_numeric(&mut data, &lines, Summary::Median),
        FillMethod::Previous => fill_neighbor_numeric(&mut data, &lines, Neighbor::Previous),
        FillMethod::Next => fill_neighbor_numeric(&mut data, &lines, Neighbor::Next),
        FillMethod::Nearest => fill_nearest_numeric(&mut data, &lines),
        FillMethod::Linear => fill_linear_numeric(&mut data, &lines),
    }
    for (flag, value) in mask.iter_mut().zip(&data) {
        if value.is_nan() {
            *flag = 0;
        }
    }
    Ok((
        Value::Tensor(Tensor::new_with_dtype(data, shape.clone(), dtype).map_err(internal_error)?),
        LogicalArray::new(mask, shape).map_err(internal_error)?,
    ))
}

pub(super) fn fill_missing_string_array(
    array: StringArray,
    options: &FillOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    let lines = FillLines::new(&array.shape, options.dim)?;
    let mut data = array.data.clone();
    let mut mask: Vec<u8> = data
        .iter()
        .map(|text| u8::from(is_missing_text(text)))
        .collect();
    match &options.method {
        FillMethod::Constant(fill) => {
            let fill = scalar_text(fill)
                .ok_or_else(|| invalid_argument("fillmissing: string constant must be text"))?;
            for (idx, text) in data.iter_mut().enumerate() {
                if mask[idx] != 0 {
                    *text = fill.clone();
                }
            }
        }
        FillMethod::Previous => fill_neighbor_text(&mut data, &lines, Neighbor::Previous),
        FillMethod::Next => fill_neighbor_text(&mut data, &lines, Neighbor::Next),
        FillMethod::Nearest => fill_nearest_text(&mut data, &lines),
        _ => {
            return Err(unsupported_type(
                "fillmissing: string arrays support constant, previous, next, and nearest",
            ))
        }
    }
    for (flag, text) in mask.iter_mut().zip(&data) {
        if is_missing_text(text) {
            *flag = 0;
        }
    }
    Ok((
        Value::StringArray(StringArray::new(data, array.shape.clone()).map_err(internal_error)?),
        LogicalArray::new(mask, array.shape).map_err(internal_error)?,
    ))
}

pub(super) fn fill_missing_cell(
    cell: CellArray,
    options: &FillOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    let mut data = cell.data.clone();
    let mut mask = Vec::with_capacity(data.len());
    for item in &data {
        mask.push(u8::from(any_missing(item)?));
    }
    match &options.method {
        FillMethod::Constant(fill) => {
            for (idx, item) in data.iter_mut().enumerate() {
                if mask[idx] != 0 {
                    *item = fill.clone();
                }
            }
        }
        _ => {
            return Err(unsupported_type(
                "fillmissing: cell arrays currently support the constant method",
            ))
        }
    }
    for (flag, item) in mask.iter_mut().zip(&data) {
        if any_missing(item)? {
            *flag = 0;
        }
    }
    Ok((
        Value::Cell(CellArray::new(data, cell.rows, cell.cols).map_err(internal_error)?),
        LogicalArray::new(mask, vec![cell.rows, cell.cols]).map_err(internal_error)?,
    ))
}
