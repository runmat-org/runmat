//! Native arithmetic over authenticated payloads; no floating compatibility mirror.
use super::*;

mod kernels;
mod observation;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Operation {
    Add,
    Sub,
    Mul,
    Div,
}

impl InProcessProvider {
    pub(super) fn native_binary(
        &self,
        left: &GpuTensorHandle,
        right: &GpuTensorHandle,
        operation: Operation,
    ) -> Result<GpuTensorHandle> {
        let (data, shape, storage) = {
            let guard = numeric_registry()
                .lock_for(self.device_id)
                .map_err(|_| anyhow!("in-process allocation owner is poisoned"))?;
            // Authenticate BOTH operands before deciding representation eligibility.
            let lhs = guard
                .get(left)
                .ok_or_else(|| anyhow!("arithmetic left input is not a live owned allocation"))?;
            let rhs = guard
                .get(right)
                .ok_or_else(|| anyhow!("arithmetic right input is not a live owned allocation"))?;
            let lhs = Operand::new(lhs, left)?;
            let rhs = Operand::new(rhs, right)?;
            evaluate(lhs, rhs, operation)?
        };
        let output = self.allocate_numeric_data(data, shape, storage)?;
        self.confirm_native_arithmetic(operation);
        Ok(output)
    }

    pub(super) fn native_scalar(
        &self,
        input: &GpuTensorHandle,
        scalar: f64,
        operation: Operation,
        scalar_left: bool,
    ) -> Result<GpuTensorHandle> {
        let (data, _, storage) = {
            let guard = numeric_registry()
                .lock_for(self.device_id)
                .map_err(|_| anyhow!("in-process allocation owner is poisoned"))?;
            let data = guard
                .get(input)
                .ok_or_else(|| anyhow!("arithmetic input is not a live owned allocation"))?;
            let input = Operand::new(data, input)?;
            // Scalar ABI is Double: Single/Double arithmetic rounds the result,
            // not the scalar before the operation. Only one stack word is used.
            let scalar_data = [scalar];
            let scalar = Operand {
                data: Floating::Double(&scalar_data),
                shape: &[],
                complex: false,
            };
            if scalar_left {
                evaluate(scalar, input, operation)?
            } else {
                evaluate(input, scalar, operation)?
            }
        };
        let output = self.allocate_numeric_data(data, input.shape().to_vec(), storage)?;
        self.confirm_native_arithmetic(operation);
        Ok(output)
    }
}

#[derive(Clone, Copy)]
enum Floating<'a> {
    Single(&'a [f32]),
    Double(&'a [f64]),
}
impl Floating<'_> {
    fn len(self) -> usize {
        match self {
            Self::Single(v) => v.len(),
            Self::Double(v) => v.len(),
        }
    }
    fn f64(self, index: usize) -> f64 {
        match self {
            Self::Single(v) => f64::from(v[index]),
            Self::Double(v) => v[index],
        }
    }
    fn is_single(self) -> bool {
        matches!(self, Self::Single(_))
    }
}

#[derive(Clone, Copy)]
struct Operand<'a> {
    data: Floating<'a>,
    shape: &'a [usize],
    complex: bool,
}
impl<'a> Operand<'a> {
    fn new(data: &'a InProcessNumericData, handle: &'a GpuTensorHandle) -> Result<Self> {
        if runmat_accelerate_api::handle_is_logical(handle) {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "arithmetic logical promotion belongs to the canonical domain",
            ));
        }
        let data = match data {
            InProcessNumericData::F32(data) => Floating::Single(data),
            InProcessNumericData::F64(data) => Floating::Double(data),
            InProcessNumericData::Integer(_) => {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "native integer arithmetic uses the exact canonical integer engine",
                ))
            }
        };
        let complex = handle.physical().storage() == GpuTensorStorage::ComplexInterleaved;
        let words = logical_len_for_shape("arithmetic", handle.shape())?
            .checked_mul(if complex { 2 } else { 1 })
            .ok_or_else(|| anyhow!("arithmetic storage extent overflow"))?;
        ensure!(
            data.len() == words,
            "arithmetic payload does not match admitted view"
        );
        Ok(Self {
            data,
            shape: handle.shape(),
            complex,
        })
    }
    fn pair_f64(self, index: usize) -> (f64, f64) {
        if self.complex {
            (self.data.f64(index * 2), self.data.f64(index * 2 + 1))
        } else {
            (self.data.f64(index), 0.0)
        }
    }
}

fn evaluate(
    lhs: Operand<'_>,
    rhs: Operand<'_>,
    operation: Operation,
) -> Result<(InProcessNumericData, Vec<usize>, GpuTensorStorage)> {
    // Existing canonical geometry, with checked zero-first extent before allocation.
    let shape =
        runtime_broadcast_shapes("arithmetic", lhs.shape, rhs.shape).map_err(|e| anyhow!(e))?;
    let count = logical_len_for_shape("arithmetic", &shape)?;
    let complex = lhs.complex || rhs.complex;
    let words = count
        .checked_mul(if complex { 2 } else { 1 })
        .ok_or_else(|| anyhow!("arithmetic output extent overflow"))?;
    let ls = runtime_compute_strides(lhs.shape);
    let rs = runtime_compute_strides(rhs.shape);
    // Preserve the linear scalar/equal-shape fast path; only expanded operands
    // need per-element rank traversal.
    let index = |i, operand: Operand<'_>, strides: &[usize]| {
        if operand.data.len() == if operand.complex { 2 } else { 1 } {
            0
        } else if operand.shape == shape.as_slice() {
            i
        } else {
            runtime_broadcast_index(i, &shape, operand.shape, strides)
        }
    };
    let indices = (0..count).map(|i| (index(i, lhs, &ls), index(i, rhs, &rs)));
    let data = if lhs.data.is_single() || rhs.data.is_single() {
        let mut output = Vec::with_capacity(words);
        for (li, ri) in indices {
            let value = match (lhs.data, rhs.data) {
                (Floating::Single(left), Floating::Single(right)) => {
                    let pair = |v: &[f32], index, complex| {
                        if complex {
                            (v[index * 2], v[index * 2 + 1])
                        } else {
                            (v[index], 0.0)
                        }
                    };
                    kernels::single(
                        pair(left, li, lhs.complex),
                        pair(right, ri, rhs.complex),
                        lhs.complex,
                        rhs.complex,
                        operation,
                    )
                }
                _ => {
                    let v = kernels::double(
                        lhs.pair_f64(li),
                        rhs.pair_f64(ri),
                        lhs.complex,
                        rhs.complex,
                        operation,
                    );
                    (v.0 as f32, v.1 as f32)
                }
            };
            output.push(value.0);
            if complex {
                output.push(value.1);
            }
        }
        InProcessNumericData::F32(output)
    } else {
        let mut output = Vec::with_capacity(words);
        for (li, ri) in indices {
            let value = kernels::double(
                lhs.pair_f64(li),
                rhs.pair_f64(ri),
                lhs.complex,
                rhs.complex,
                operation,
            );
            output.push(value.0);
            if complex {
                output.push(value.1);
            }
        }
        InProcessNumericData::F64(output)
    };
    Ok((
        data,
        shape,
        if complex {
            GpuTensorStorage::ComplexInterleaved
        } else {
            GpuTensorStorage::Real
        },
    ))
}
