use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{Tensor, Value};

use crate::builtins::plotting::common::{gather_tensor_from_gpu_async, numeric_vector};
use crate::builtins::plotting::plotting_error;
use crate::BuiltinResult;

pub fn parse_surface_call_args(
    args: Vec<Value>,
    builtin: &'static str,
) -> BuiltinResult<(Value, Value, Value, Vec<Value>)> {
    match args.len() {
        0 => Err(plotting_error(
            builtin,
            format!("{builtin}: expected Z or X,Y,Z input"),
        )),
        1 => {
            let z = args.into_iter().next().expect("one arg");
            let (rows, cols) = inferred_grid_shape(&z, builtin)?;
            let x = Value::Tensor(default_surface_axis(rows));
            let y = Value::Tensor(default_surface_axis(cols));
            Ok((x, y, z, Vec::new()))
        }
        2 => Err(plotting_error(
            builtin,
            format!("{builtin}: expected Z or X,Y,Z input"),
        )),
        _ => {
            let mut it = args.into_iter();
            let x = it.next().expect("x");
            let y = it.next().expect("y");
            let z = it.next().expect("z");
            let rest = it.collect();
            Ok((x, y, z, rest))
        }
    }
}

fn inferred_grid_shape(value: &Value, builtin: &'static str) -> BuiltinResult<(usize, usize)> {
    match value {
        Value::GpuTensor(handle) => {
            if handle.shape.len() < 2 {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: Z must contain at least a 2-D grid"),
                ));
            }
            let rows = handle.shape[0].max(1);
            let cols = handle.shape[1].max(1);
            Ok((rows, cols))
        }
        other => {
            let tensor = Tensor::try_from(other)
                .map_err(|e| plotting_error(builtin, format!("{builtin}: {e}")))?;
            if tensor.rows == 0 || tensor.cols == 0 {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: Z must contain at least a 2-D grid"),
                ));
            }
            Ok((tensor.rows, tensor.cols))
        }
    }
}

fn default_surface_axis(len: usize) -> Tensor {
    Tensor {
        data: (1..=len).map(|i| i as f64).collect(),
        shape: vec![len],
        rows: len,
        cols: 1,
        dtype: runmat_builtins::NumericDType::F64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_surface_call_args_supports_z_only_shorthand() {
        let z = Value::Tensor(Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
            rows: 2,
            cols: 2,
            dtype: runmat_builtins::NumericDType::F64,
        });
        let (x, y, z_out, rest) = parse_surface_call_args(vec![z.clone()], "surf").unwrap();
        let x = Tensor::try_from(&x).unwrap();
        let y = Tensor::try_from(&y).unwrap();
        assert_eq!(x.data, vec![1.0, 2.0]);
        assert_eq!(y.data, vec![1.0, 2.0]);
        assert!(rest.is_empty());
        assert_eq!(
            Tensor::try_from(&z_out).unwrap().data,
            Tensor::try_from(&z).unwrap().data
        );
    }
}

#[derive(Clone)]
pub enum AxisSource {
    Host(Vec<f64>),
    Gpu(GpuTensorHandle),
}

impl AxisSource {
    pub fn len(&self) -> usize {
        match self {
            AxisSource::Host(v) => v.len(),
            AxisSource::Gpu(h) => vector_len_from_shape(&h.shape),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub fn is_vector_like(tensor: &Tensor) -> bool {
    if tensor.shape.is_empty() {
        return true;
    }
    tensor.rows == 1 || tensor.cols == 1
}

pub fn is_vector_like_shape(shape: &[usize]) -> bool {
    if shape.is_empty() {
        return true;
    }
    let non_singleton = shape.iter().copied().filter(|d| *d > 1).count();
    non_singleton <= 1
}

pub fn vector_len_from_shape(shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 1;
    }
    shape.iter().copied().max().unwrap_or(1)
}

pub async fn axis_sources_from_xy_values(
    x: Value,
    y: Value,
    rows: usize,
    cols: usize,
    builtin: &'static str,
) -> BuiltinResult<(AxisSource, AxisSource)> {
    match (x, y) {
        (Value::GpuTensor(xh), Value::GpuTensor(yh))
            if is_vector_like_shape(&xh.shape) && is_vector_like_shape(&yh.shape) =>
        {
            Ok((AxisSource::Gpu(xh), AxisSource::Gpu(yh)))
        }
        (x_val, y_val) => {
            let x_tensor = match x_val {
                Value::GpuTensor(handle) => gather_tensor_from_gpu_async(handle, builtin).await?,
                other => Tensor::try_from(&other)
                    .map_err(|e| plotting_error(builtin, format!("{builtin}: {e}")))?,
            };
            let y_tensor = match y_val {
                Value::GpuTensor(handle) => gather_tensor_from_gpu_async(handle, builtin).await?,
                other => Tensor::try_from(&other)
                    .map_err(|e| plotting_error(builtin, format!("{builtin}: {e}")))?,
            };
            if x_tensor.data.is_empty() || y_tensor.data.is_empty() {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: axis vectors must be non-empty"),
                ));
            }

            if is_vector_like(&x_tensor) && is_vector_like(&y_tensor) {
                Ok((
                    AxisSource::Host(numeric_vector(x_tensor)),
                    AxisSource::Host(numeric_vector(y_tensor)),
                ))
            } else {
                let (x_vec, y_vec) = extract_meshgrid_axes_from_xy_matrices(
                    &x_tensor, &y_tensor, rows, cols, builtin,
                )?;
                Ok((AxisSource::Host(x_vec), AxisSource::Host(y_vec)))
            }
        }
    }
}

pub async fn axis_sources_to_host(
    x: &AxisSource,
    y: &AxisSource,
    builtin: &'static str,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let x_vec = match x {
        AxisSource::Host(v) => v.clone(),
        AxisSource::Gpu(h) => {
            numeric_vector(gather_tensor_from_gpu_async(h.clone(), builtin).await?)
        }
    };
    let y_vec = match y {
        AxisSource::Host(v) => v.clone(),
        AxisSource::Gpu(h) => {
            numeric_vector(gather_tensor_from_gpu_async(h.clone(), builtin).await?)
        }
    };
    Ok((x_vec, y_vec))
}

fn matrix_rows_are_identical(tensor: &Tensor) -> bool {
    let rows = tensor.rows;
    let cols = tensor.cols;
    if rows == 0 || cols == 0 {
        return false;
    }
    for row in 1..rows {
        for col in 0..cols {
            let idx0 = rows * col;
            let idx = row + rows * col;
            if tensor.data[idx] != tensor.data[idx0] {
                return false;
            }
        }
    }
    true
}

fn matrix_cols_are_identical(tensor: &Tensor) -> bool {
    let rows = tensor.rows;
    let cols = tensor.cols;
    if rows == 0 || cols == 0 {
        return false;
    }
    for col in 1..cols {
        for row in 0..rows {
            let idx0 = row;
            let idx = row + rows * col;
            if tensor.data[idx] != tensor.data[idx0] {
                return false;
            }
        }
    }
    true
}

pub fn extract_meshgrid_axes_from_xy_matrices(
    x: &Tensor,
    y: &Tensor,
    rows: usize,
    cols: usize,
    builtin: &'static str,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    if x.rows != rows || x.cols != cols || y.rows != rows || y.cols != cols {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: when X and Y are matrices, they must match the shape of Z"),
        ));
    }
    if !matrix_rows_are_identical(x) || !matrix_cols_are_identical(y) {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: matrix X/Y inputs must be meshgrid-style coordinate matrices"),
        ));
    }
    let mut x_vec = Vec::with_capacity(cols);
    for col in 0..cols {
        x_vec.push(x.data[rows * col]);
    }
    let mut y_vec = Vec::with_capacity(rows);
    for row in 0..rows {
        y_vec.push(y.data[row]);
    }
    Ok((x_vec, y_vec))
}
