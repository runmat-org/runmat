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
        let shape = broadcast_shape(mask.shape(), then_value.shape())?;
        let shape = broadcast_shape(&shape, else_value.shape())?;
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

fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>> {
    let mut output = Vec::new();
    let rank = lhs.len().max(rhs.len());
    output.try_reserve_exact(rank)?;
    for axis in 0..rank {
        let extent_at = |shape: &[usize]| {
            if rank >= 2 && shape.len() == 1 {
                if axis == 1 {
                    shape[0]
                } else {
                    1
                }
            } else {
                shape.get(axis).copied().unwrap_or(1)
            }
        };
        let a = extent_at(lhs);
        let b = extent_at(rhs);
        let extent = if a == b {
            a
        } else if a == 1 {
            b
        } else if b == 1 {
            a
        } else {
            return Err(anyhow!(
                "select non-singleton dimension mismatch at axis {}: {} versus {}",
                axis + 1,
                a,
                b
            ));
        };
        output.push(extent);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_major_broadcast_and_empty_geometry_are_checked() {
        assert_eq!(broadcast_shape(&[2, 1], &[1, 3]).unwrap(), [2, 3]);
        assert_eq!(broadcast_shape(&[0, 2], &[1, 2]).unwrap(), [0, 2]);
        assert_eq!(broadcast_shape(&[3], &[2, 3]).unwrap(), [2, 3]);
        assert!(broadcast_shape(&[2, 2], &[3, 2]).is_err());
        assert!(checked_len(&[usize::MAX, 2]).is_err());
    }
}
