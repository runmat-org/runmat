//! Provider-independent admission for a resident logical selection.
use crate::{GpuTensorHandle, GpuTensorStorage, NumericElementType};
use anyhow::{anyhow, ensure, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentSelectPlan {
    shape: Vec<usize>,
    len: usize,
    element_type: NumericElementType,
}

impl ResidentSelectPlan {
    pub fn new(
        mask: &GpuTensorHandle,
        then_value: &GpuTensorHandle,
        else_value: &GpuTensorHandle,
    ) -> Result<Self> {
        for handle in [mask, then_value, else_value] {
            handle.allocation_lease().validate_live()?;
            ensure!(
                handle.physical().storage() == GpuTensorStorage::Real,
                "select requires real storage"
            );
            let len = checked_len(handle.shape())?;
            ensure!(
                u64::try_from(len).is_ok_and(|n| n <= handle.physical().element_capacity()),
                "select input shape exceeds its physical allocation"
            );
        }
        ensure!(
            mask.device_id() == then_value.device_id()
                && mask.device_id() == else_value.device_id(),
            "select inputs belong to different devices"
        );
        ensure!(mask.view().is_logical(), "select mask must be logical");
        ensure!(
            !then_value.view().is_logical() && !else_value.view().is_logical(),
            "select branches must be numeric, not logical"
        );
        let element_type = then_value.physical().element_type();
        ensure!(
            matches!(
                element_type,
                NumericElementType::F32 | NumericElementType::F64
            ) && else_value.physical().element_type() == element_type
                && matches!(
                    mask.physical().element_type(),
                    NumericElementType::F32 | NumericElementType::F64
                ),
            "select requires matching real floating precision"
        );
        let shape =
            runmat_types::geometry::broadcast_concrete_shape(mask.shape(), then_value.shape())?;
        let shape = runmat_types::geometry::broadcast_concrete_shape(&shape, else_value.shape())?;
        let len = checked_len(&shape)?;
        Ok(Self {
            shape,
            len,
            element_type,
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn element_type(&self) -> NumericElementType {
        self.element_type
    }

    pub fn into_shape(self) -> Vec<usize> {
        self.shape
    }
}

fn checked_len(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |len, &extent| len.checked_mul(extent))
        .ok_or_else(|| anyhow!("select tensor size overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_major_broadcast_and_empty_geometry_are_checked() {
        assert_eq!(
            runmat_types::geometry::broadcast_concrete_shape(&[2, 1], &[1, 3]).unwrap(),
            [2, 3]
        );
        assert!(checked_len(&[usize::MAX, 2]).is_err());
    }
}
