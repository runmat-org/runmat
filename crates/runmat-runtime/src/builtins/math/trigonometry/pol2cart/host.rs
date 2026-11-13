use super::*;

#[derive(Debug)]
pub(super) struct Pol2CartEval {
    x: Tensor,
    y: Tensor,
    z: Option<Tensor>,
    max_outputs: usize,
}

impl Pol2CartEval {
    pub(super) fn polar(theta: Value, rho: Value) -> BuiltinResult<Self> {
        let theta = value_into_real_tensor(theta)?;
        let rho = value_into_real_tensor(rho)?;
        let (x, y, _) = compute_pol2cart(&theta, &rho, None, false)?;
        Ok(Self {
            x,
            y,
            z: None,
            max_outputs: 2,
        })
    }

    pub(super) fn cylindrical(
        theta: Value,
        rho: Value,
        z: Value,
        include_z_output: bool,
    ) -> BuiltinResult<Self> {
        let theta = value_into_real_tensor(theta)?;
        let rho = value_into_real_tensor(rho)?;
        let z = value_into_real_tensor(z)?;
        let (x, y, z) = compute_pol2cart(&theta, &rho, Some(&z), include_z_output)?;
        Ok(Self {
            x,
            y,
            z,
            max_outputs: 3,
        })
    }

    fn max_outputs(&self) -> usize {
        self.max_outputs
    }

    pub(super) fn x_value(&self) -> BuiltinResult<Value> {
        Ok(tensor::tensor_into_value(self.x.clone()))
    }

    pub(super) fn output_list(&self, count: usize) -> BuiltinResult<Value> {
        if count > self.max_outputs() {
            return Err(pol2cart_error(
                &ERROR_OUTPUT_COUNT,
                format!(
                    "requested {count} outputs but this syntax supports at most {}",
                    self.max_outputs()
                ),
            ));
        }
        let mut outputs = Vec::with_capacity(count);
        if count >= 1 {
            outputs.push(tensor::tensor_into_value(self.x.clone()));
        }
        if count >= 2 {
            outputs.push(tensor::tensor_into_value(self.y.clone()));
        }
        if count >= 3 {
            outputs.push(tensor::tensor_into_value(
                self.z.clone().expect("z output available"),
            ));
        }
        Ok(Value::output_list(outputs))
    }
}

fn compute_pol2cart(
    theta: &Tensor,
    rho: &Tensor,
    z: Option<&Tensor>,
    include_z_output: bool,
) -> BuiltinResult<(Tensor, Tensor, Option<Tensor>)> {
    let theta_rho = matlab_broadcast_shape(theta.shape(), rho.shape())?;
    let final_shape = match z {
        Some(z_tensor) => matlab_broadcast_shape(&theta_rho, z_tensor.shape())?,
        None => theta_rho,
    };
    let len = if final_shape.contains(&0) {
        0
    } else {
        final_shape.iter().copied().product::<usize>()
    };

    let theta_plan = MatlabBroadcastIndexPlan::new(theta.shape(), &final_shape)?;
    let rho_plan = MatlabBroadcastIndexPlan::new(rho.shape(), &final_shape)?;
    let z_plan = match z {
        Some(z_tensor) => Some(MatlabBroadcastIndexPlan::new(
            z_tensor.shape(),
            &final_shape,
        )?),
        None => None,
    };
    let theta_data = tensor::tensor_values_f64_cow(theta);
    let rho_data = tensor::tensor_values_f64_cow(rho);
    let z_data = z.map(tensor::tensor_values_f64_cow);

    let mut x = vec![0.0; len];
    let mut y = vec![0.0; len];
    let theta_single = theta.numeric_dtype() == runmat_value::NumericDType::F32;
    let single = theta_single || rho.numeric_dtype() == runmat_value::NumericDType::F32;
    for out_idx in 0..len {
        let theta_idx = theta_plan.index(out_idx);
        let rho_idx = rho_plan.index(out_idx);
        let angle = theta_data[theta_idx];
        let radius = rho_data[rho_idx];
        let (sin, cos) = if theta_single {
            ((angle as f32).sin() as f64, (angle as f32).cos() as f64)
        } else {
            (angle.sin(), angle.cos())
        };
        if single {
            x[out_idx] = ((radius as f32) * (cos as f32)) as f64;
            y[out_idx] = ((radius as f32) * (sin as f32)) as f64;
        } else {
            x[out_idx] = radius * cos;
            y[out_idx] = radius * sin;
        }
    }

    let z_output = match (z, z_plan, include_z_output) {
        (_, _, false) => None,
        (Some(z_tensor), Some(plan), true) => {
            let mut values = vec![0.0; len];
            let z_data = z_data.as_ref().expect("z values");
            for out_idx in 0..len {
                let z_idx = plan.index(out_idx);
                values[out_idx] = z_data[z_idx];
            }
            Some(
                floating_tensor(
                    values,
                    final_shape.clone(),
                    z_tensor.numeric_dtype() == runmat_value::NumericDType::F32,
                )
                .map_err(|err| pol2cart_error(&ERROR_INTERNAL, err))?,
            )
        }
        _ => None,
    };

    Ok((
        floating_tensor(x, final_shape.clone(), single)
            .map_err(|err| pol2cart_error(&ERROR_INTERNAL, err))?,
        floating_tensor(y, final_shape, single)
            .map_err(|err| pol2cart_error(&ERROR_INTERNAL, err))?,
        z_output,
    ))
}

fn floating_tensor(values: Vec<f64>, shape: Vec<usize>, single: bool) -> Result<Tensor, String> {
    if single {
        Tensor::from_f32(values.into_iter().map(|x| x as f32).collect(), shape)
    } else {
        Tensor::new(values, shape)
    }
}

pub(super) fn matlab_broadcast_shape(left: &[usize], right: &[usize]) -> BuiltinResult<Vec<usize>> {
    let rank = left.len().max(right.len());
    let left = broadcast::align_shape(left, rank);
    let right = broadcast::align_shape(right, rank);
    let mut shape = Vec::with_capacity(rank);
    for dim in 0..rank {
        let a = left[dim];
        let b = right[dim];
        if a == b {
            shape.push(a);
        } else if a == 1 {
            shape.push(b);
        } else if b == 1 {
            shape.push(a);
        } else {
            return Err(pol2cart_error(
                &ERROR_SIZE_MISMATCH,
                format!(
                    "non-singleton dimension mismatch (dimension {}: {} vs {})",
                    dim + 1,
                    a,
                    b
                ),
            ));
        }
    }
    Ok(shape)
}

#[derive(Debug)]
struct MatlabBroadcastIndexPlan {
    out_shape: Vec<usize>,
    in_shape: Vec<usize>,
    strides: Vec<usize>,
}

impl MatlabBroadcastIndexPlan {
    fn new(input_shape: &[usize], output_shape: &[usize]) -> BuiltinResult<Self> {
        let expected = matlab_broadcast_shape(input_shape, output_shape)?;
        if expected != output_shape {
            return Err(pol2cart_error(
                &ERROR_SIZE_MISMATCH,
                "input cannot expand to final output shape",
            ));
        }
        let in_shape = broadcast::align_shape(input_shape, output_shape.len());
        let strides = column_major_strides(&in_shape);
        Ok(Self {
            out_shape: output_shape.to_vec(),
            in_shape,
            strides,
        })
    }

    fn index(&self, mut linear: usize) -> usize {
        let mut offset = 0usize;
        for dim in 0..self.out_shape.len() {
            let out_extent = self.out_shape[dim];
            let coord = if out_extent == 0 {
                0
            } else {
                let coord = linear % out_extent;
                linear /= out_extent;
                coord
            };
            if self.in_shape[dim] != 1 && out_extent != 0 {
                offset += coord * self.strides[dim];
            }
        }
        offset
    }
}

fn column_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &extent in shape {
        strides.push(stride);
        stride = stride.saturating_mul(extent.max(1));
    }
    strides
}

pub(super) fn value_into_real_tensor(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Num(value) => {
            Tensor::new(vec![value], vec![1, 1]).map_err(|err| pol2cart_error(&ERROR_INTERNAL, err))
        }
        Value::Tensor(tensor) => Ok(tensor),
        Value::Int(value) => {
            Tensor::new_integer(runmat_value::IntegerStorage::from_scalar(value), vec![1, 1])
                .map_err(|error| pol2cart_error(&ERROR_INTERNAL, error))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(pol2cart_error(&ERROR_COMPLEX_UNSUPPORTED, "complex input"))
        }
        other => Err(pol2cart_error(
            &ERROR_INVALID_INPUT,
            format!("expected real single or double input, got {other:?}"),
        )),
    }
}
