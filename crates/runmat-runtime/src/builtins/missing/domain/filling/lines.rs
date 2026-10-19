use super::super::{internal_error, invalid_argument};
use crate::BuiltinResult;

/// Column-major lines along one axis. Trailing singleton axes do not require
/// padding the shape or allocating dimension-sized metadata.
pub(super) struct FillLines {
    stride: usize,
    length: usize,
    blocks: usize,
}

pub(super) struct FillLine {
    pub(super) start: usize,
    pub(super) length: usize,
    pub(super) stride: usize,
}

impl FillLines {
    pub(super) fn new(shape: &[usize], dimension: Option<usize>) -> BuiltinResult<Self> {
        let axis = match dimension {
            Some(dimension) => dimension
                .checked_sub(1)
                .ok_or_else(|| invalid_argument("fillmissing: dimension must be positive"))?,
            None => shape.iter().position(|size| *size != 1).unwrap_or(0),
        };
        if shape.contains(&0) {
            return Ok(Self {
                stride: 0,
                length: 0,
                blocks: 0,
            });
        }
        let product = |dimensions: &[usize]| {
            dimensions.iter().try_fold(1usize, |count, size| {
                count
                    .checked_mul(*size)
                    .ok_or_else(|| internal_error("fillmissing: shape overflow"))
            })
        };
        let stride = product(&shape[..shape.len().min(axis)])?;
        let length = shape.get(axis).copied().unwrap_or(1);
        let blocks = product(&shape[shape.len().min(axis.saturating_add(1))..])?;
        stride
            .checked_mul(length)
            .and_then(|span| span.checked_mul(blocks))
            .ok_or_else(|| internal_error("fillmissing: shape overflow"))?;
        Ok(Self {
            stride,
            length,
            blocks,
        })
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = FillLine> + '_ {
        (0..self.blocks).flat_map(move |block| {
            (0..self.stride).map(move |prefix| FillLine {
                start: block * self.stride * self.length + prefix,
                length: self.length,
                stride: self.stride,
            })
        })
    }
}

#[cfg(test)]
mod tests;
