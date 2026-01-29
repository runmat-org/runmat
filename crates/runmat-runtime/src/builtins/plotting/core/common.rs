use futures::executor::block_on;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{Tensor, Value};
use runmat_plot::plots::Figure;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::BuiltinResult;

use super::plotting_error;

type NumericTriplet = (Vec<f64>, Vec<f64>, Vec<f64>);

/// Default error message when no plotting backend is available.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub const ERR_PLOTTING_UNAVAILABLE: &str =
    "Plotting is unavailable in this build (enable the `gui` or `plot-web` feature).";

pub fn numeric_vector(tensor: Tensor) -> Vec<f64> {
    tensor.data
}

pub fn numeric_pair(
    x: Tensor,
    y: Tensor,
    name: &'static str,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let x_vec = numeric_vector(x);
    let y_vec = numeric_vector(y);
    if x_vec.len() != y_vec.len() {
        return Err(plotting_error(
            name,
            format!("{name}: X and Y inputs must have the same number of elements"),
        ));
    }
    Ok((x_vec, y_vec))
}

pub fn numeric_triplet(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    name: &'static str,
) -> BuiltinResult<NumericTriplet> {
    let (x_vec, y_vec) = numeric_pair(x, y, name)?;
    let z_vec = numeric_vector(z);
    if z_vec.len() != x_vec.len() {
        return Err(plotting_error(
            name,
            format!("{name}: X, Y, and Z inputs must have the same number of elements"),
        ));
    }
    Ok((x_vec, y_vec, z_vec))
}

pub fn value_as_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Num(v) => Some(*v),
        Value::Int(int_val) => Some(int_val.to_f64()),
        Value::Bool(flag) => Some(if *flag { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) => tensor.data.first().copied(),
        Value::CharArray(chars) => {
            // Treat character arrays as numeric strings when possible.
            let text: String = chars.data.iter().collect();
            text.trim().parse::<f64>().ok()
        }
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

/// Helper that wraps either a host-resident tensor or a gpuArray when plotting surfaces.
#[derive(Clone)]
pub enum SurfaceDataInput {
    Host(Tensor),
    Gpu(GpuTensorHandle),
}

impl SurfaceDataInput {
    pub fn from_value(value: Value, context: &'static str) -> BuiltinResult<Self> {
        match value {
            Value::GpuTensor(handle) => Ok(Self::Gpu(handle)),
            other => {
                let tensor = Tensor::try_from(&other)
                    .map_err(|e| plotting_error(context, format!("{context}: {e}")))?;
                Ok(Self::Host(tensor))
            }
        }
    }

    pub fn gpu_handle(&self) -> Option<&GpuTensorHandle> {
        match self {
            Self::Gpu(handle) => Some(handle),
            Self::Host(_) => None,
        }
    }

    pub fn into_tensor(self, context: &'static str) -> BuiltinResult<Tensor> {
        match self {
            Self::Host(tensor) => Ok(tensor),
            Self::Gpu(handle) => gather_tensor_from_gpu(handle, context),
        }
    }

    pub fn grid_shape(&self, context: &'static str) -> BuiltinResult<(usize, usize)> {
        match self {
            Self::Host(tensor) => {
                if tensor.rows == 0 || tensor.cols == 0 {
                    Err(plotting_error(
                        context,
                        format!("{context}: Z must contain at least a 2-D grid"),
                    ))
                } else {
                    Ok((tensor.rows, tensor.cols))
                }
            }
            Self::Gpu(handle) => {
                if handle.shape.len() < 2 {
                    return Err(plotting_error(
                        context,
                        format!(
                            "{context}: gpuArray inputs must be 2-D (got shape {:?})",
                            handle.shape
                        ),
                    ));
                }
                let rows = handle.shape[0];
                let cols = handle.shape[1];
                if rows == 0 || cols == 0 {
                    Err(plotting_error(
                        context,
                        format!("{context}: Z must contain at least a 2-D grid"),
                    ))
                } else {
                    Ok((rows, cols))
                }
            }
        }
    }
}

pub async fn gather_tensor_from_gpu_async(
    handle: GpuTensorHandle,
    context: &'static str,
) -> BuiltinResult<Tensor> {
    let value = Value::GpuTensor(handle);
    let gathered = crate::gather_if_needed_async(&value)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, context))?;
    Tensor::try_from(&gathered).map_err(|e| plotting_error(context, format!("{context}: {e}")))
}

pub fn gather_tensor_from_gpu(
    handle: GpuTensorHandle,
    context: &'static str,
) -> BuiltinResult<Tensor> {
    block_on(gather_tensor_from_gpu_async(handle, context))
}

pub fn tensor_to_surface_grid(
    z: Tensor,
    x_len: usize,
    y_len: usize,
    context: &'static str,
) -> BuiltinResult<Vec<Vec<f64>>> {
    if z.data.len() != x_len * y_len {
        return Err(plotting_error(
            context,
            format!(
                "{context}: surface data must contain exactly {} values ({}×{})",
                x_len * y_len,
                x_len,
                y_len
            ),
        ));
    }
    let mut grid = vec![vec![0.0; y_len]; x_len];
    for (row, row_vec) in grid.iter_mut().enumerate() {
        for (col, cell) in row_vec.iter_mut().enumerate().take(y_len) {
            let idx = col * x_len + row; // column-major layout
            *cell = z.data[idx];
        }
    }
    Ok(grid)
}

pub fn default_figure(title: &str, x_label: &str, y_label: &str) -> Figure {
    Figure::new()
        .with_title(title)
        .with_labels(x_label, y_label)
        .with_grid(true)
}
