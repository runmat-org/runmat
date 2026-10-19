use super::*;
use runmat_accelerate_api::{MatmulAffine, MatmulOffset, MatmulScaleWork, MatmulWorkload};

pub(in crate::fusion_exec) struct Selection<'a> {
    pub inputs: Vec<&'a Value>,
    pub workload: MatmulWorkload,
    pub upload_bytes: u64,
}

impl plan::AdmittedMatmul {
    pub(super) fn selection<'a>(
        &self,
        request: &'a FusionExecutionRequest<'_>,
        provider: &'static dyn AccelProvider,
    ) -> Option<Selection<'a>> {
        let mut ids = Vec::new();
        for id in self.data_inputs() {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        let inputs = ids
            .iter()
            .map(|id| self.value(request, *id))
            .collect::<Option<Vec<_>>>()?;
        let index = |id| ids.iter().position(|candidate| *candidate == id);
        let left_shape = representation::shape(self.value(request, self.left)?, provider)?;
        let descriptor = self.scalar_descriptor();
        let affine = MatmulAffine {
            scale: descriptor.affine.scale.map(f64::to_bits),
            offset: descriptor.affine.offset.map(|offset| match offset {
                MatmulOffset::Add(v) => MatmulOffset::Add(v.to_bits()),
                MatmulOffset::Subtract(v) => MatmulOffset::Subtract(v.to_bits()),
                MatmulOffset::SubtractFrom(v) => MatmulOffset::SubtractFrom(v.to_bits()),
            }),
        };
        let width = match provider.precision() {
            ProviderPrecision::F32 => 4,
            ProviderPrecision::F64 => 8,
        };
        let upload_bytes = inputs.iter().try_fold(0u64, |total, value| {
            let elements = match value {
                Value::Tensor(tensor) => u64::try_from(tensor.len()).ok()?,
                Value::GpuTensor(_) => 0,
                _ => return None,
            };
            total.checked_add(elements.checked_mul(width)?)
        })?;
        Some(Selection {
            inputs,
            workload: MatmulWorkload {
                left: index(self.left)?,
                right: index(self.right)?,
                rows: u64::try_from(self.shape[0]).ok()?,
                inner: u64::try_from(left_shape[1]).ok()?,
                columns: u64::try_from(self.shape[1]).ok()?,
                affine,
                row: self.row.map(|(id, operation)| MatmulScaleWork {
                    input: index(id).expect("data role included"),
                    operation,
                }),
                column: self.column.map(|(id, operation)| MatmulScaleWork {
                    input: index(id).expect("data role included"),
                    operation,
                }),
                clamp_min: descriptor.clamp_min.map(f64::to_bits),
                clamp_max: descriptor.clamp_max.map(f64::to_bits),
                exponent: descriptor.pow_exponent.map(f64::to_bits),
                diagonal: self.diagonal,
            },
            upload_bytes,
        })
    }
}
