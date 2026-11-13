//! Metadata from the cumulative owner's actual public option parser.
use crate::{call::FunctionArgumentView, BuiltinResult};
use runmat_builtins::{CumulativeKind, CumulativeScanDirection};
pub(in crate::builtins::math) struct ScanGeometry {
    stride: usize,
    extent: usize,
    count: usize,
    reverse: bool,
}
pub(in crate::builtins::math) struct ScanRows {
    base: usize,
    stride: usize,
    next: usize,
    end: usize,
}
impl Iterator for ScanRows {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.next == self.end {
            return None;
        }
        let value = self.base + self.next * self.stride;
        self.next += 1;
        Some(value)
    }
}
impl ScanGeometry {
    fn rows(&self, index: usize, prefix: bool) -> Option<ScanRows> {
        if index >= self.count || self.extent == 0 {
            return None;
        }
        let coordinate = index / self.stride % self.extent;
        let base = index - coordinate * self.stride;
        let (next, end) = if prefix {
            (0, coordinate + 1)
        } else {
            (coordinate, self.extent)
        };
        Some(ScanRows {
            base,
            stride: self.stride,
            next,
            end,
        })
    }
    /// Transposed routing over the very same forward/reverse scan lane.
    pub(in crate::builtins::math) fn inputs(&self, output: usize) -> Option<ScanRows> {
        self.rows(output, !self.reverse)
    }
    pub(in crate::builtins::math) fn outputs(&self, input: usize) -> Option<ScanRows> {
        self.rows(input, self.reverse)
    }
}
pub(in crate::builtins::math) fn scan_geometry(
    kind: CumulativeKind,
    shape: &[usize],
    arguments: FunctionArgumentView<'_>,
) -> BuiltinResult<ScanGeometry> {
    let parsed = super::options::parse(arguments).map_err(|e| super::failure(kind, e))?;
    let dimension = parsed
        .dimension
        .unwrap_or_else(|| super::super::cumsum::default_dimension_from_shape(shape));
    let axis = dimension - 1;
    let count = if shape.contains(&0) {
        0
    } else {
        shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| super::failure(kind, "scan extent exceeds addressable size"))?
    };
    let extent = shape.get(axis).copied().unwrap_or(1);
    let stride = if count == 0 {
        1
    } else {
        shape
            .iter()
            .take(axis)
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| super::failure(kind, "scan stride exceeds addressable size"))?
    };
    Ok(ScanGeometry {
        stride,
        extent,
        count,
        reverse: parsed.direction == CumulativeScanDirection::Reverse,
    })
}

pub(in crate::builtins::math) struct ScanStep {
    pub index: usize,
    pub first: bool,
}
pub(in crate::builtins::math) struct ScanSteps {
    stride: usize,
    extent: usize,
    count: usize,
    lane: usize,
    position: usize,
    reverse: bool,
}
impl Iterator for ScanSteps {
    type Item = ScanStep;
    fn next(&mut self) -> Option<Self::Item> {
        if self.extent == 0 || self.lane >= self.count / self.extent {
            return None;
        }
        let coordinate = if self.reverse {
            self.extent - 1 - self.position
        } else {
            self.position
        };
        let base = self.lane % self.stride + self.lane / self.stride * self.stride * self.extent;
        let value = ScanStep {
            index: base + coordinate * self.stride,
            first: self.position == 0,
        };
        self.position += 1;
        if self.position == self.extent {
            self.position = 0;
            self.lane += 1;
        }
        Some(value)
    }
}
impl ScanGeometry {
    pub(in crate::builtins::math) fn steps(&self, transpose: bool) -> ScanSteps {
        ScanSteps {
            stride: self.stride,
            extent: self.extent,
            count: self.count,
            lane: 0,
            position: 0,
            reverse: self.reverse != transpose,
        }
    }
}
impl ExactSizeIterator for ScanRows {
    fn len(&self) -> usize {
        self.end - self.next
    }
}
