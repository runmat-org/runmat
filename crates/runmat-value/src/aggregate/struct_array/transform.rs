use super::{column_major_strides, total_len, StructArray};
use crate::UnboundedOrderedMapBuilder;

impl StructArray {
    pub fn reshape_unbounded(self, shape: Vec<usize>) -> Result<Self, String> {
        if total_len(&shape)? != self.len() {
            return Err("structure array reshape must preserve element count".into());
        }
        Ok(Self {
            fields: self.fields,
            shape: shape.into(),
        })
    }

    pub fn permute_unbounded(self, order: &[usize]) -> Result<Self, String> {
        let effective_rank = self
            .shape
            .iter()
            .rposition(|extent| *extent != 1)
            .map_or(2, |position| (position + 1).max(2));
        if order.len() < effective_rank {
            return Err("structure array permutation must name every existing dimension".into());
        }
        validate_permutation(order)?;
        let mut source_shape = self.shape.to_vec();
        source_shape.truncate(effective_rank);
        source_shape.resize(order.len(), 1);
        let output_shape = order
            .iter()
            .map(|dimension| source_shape[*dimension])
            .collect();
        let strides = column_major_strides(&source_shape)?;
        self.reorder(output_shape, |index, shape| {
            permuted_source(index, shape, order, &strides)
        })
    }

    pub fn flip_unbounded(self, dimension: usize) -> Result<Self, String> {
        let extent = self
            .shape
            .get(dimension)
            .copied()
            .ok_or_else(|| "structure array flip dimension is out of range".to_string())?;
        if extent <= 1 || self.is_empty() {
            return Ok(self);
        }
        let shape = self.shape.to_vec();
        let stride = column_major_strides(&shape)?[dimension];
        self.reorder(shape, |index, _| {
            let coordinate = index / stride % extent;
            let left = index.checked_sub(coordinate.checked_mul(stride)?)?;
            left.checked_add((extent - 1 - coordinate).checked_mul(stride)?)
        })
    }

    pub fn circular_shift_unbounded(self, shifts: &[isize]) -> Result<Self, String> {
        let rank = self.shape.len().max(shifts.len());
        let mut shape = self.shape.to_vec();
        shape.resize(rank, 1);
        let strides = column_major_strides(&shape)?;
        self.reorder(shape.clone(), |index, _| {
            let mut remainder = index;
            shape
                .iter()
                .enumerate()
                .try_fold(0usize, |source, (dimension, extent)| {
                    let coordinate = remainder % extent;
                    remainder /= extent;
                    let shift = shifts.get(dimension).copied().unwrap_or(0);
                    let source_coordinate =
                        (coordinate as isize - shift).rem_euclid(*extent as isize) as usize;
                    source_coordinate
                        .checked_mul(strides[dimension])?
                        .checked_add(source)
                })
        })
    }

    pub fn tile_unbounded(&self, repetitions: &[usize]) -> Result<Self, String> {
        let rank = self.shape.len().max(repetitions.len()).max(2);
        let mut source_shape = self.shape.to_vec();
        source_shape.resize(rank, 1);
        let mut reps = repetitions.to_vec();
        reps.resize(rank, 1);
        let output_shape = source_shape
            .iter()
            .zip(&reps)
            .map(|(extent, repetition)| {
                extent.checked_mul(*repetition).ok_or_else(|| {
                    "structure array replication exceeds platform limits".to_string()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_len = total_len(&output_shape)?;
        let strides = column_major_strides(&source_shape)?;
        let mut fields = UnboundedOrderedMapBuilder::with_capacity(self.fields.len());
        for (name, values) in &self.fields {
            let mut output = Vec::with_capacity(output_len);
            for index in 0..output_len {
                let source = tiled_source(index, &output_shape, &source_shape, &strides)?;
                output.push(values[source].clone());
            }
            fields.insert(name.to_owned(), output);
        }
        Self::from_columns(fields, output_shape)
    }

    fn reorder(
        self,
        shape: Vec<usize>,
        mut source_index: impl FnMut(usize, &[usize]) -> Option<usize>,
    ) -> Result<Self, String> {
        let length = total_len(&shape)?;
        if length != self.len() {
            return Err("structure array reorder must preserve element count".into());
        }
        let fields = self
            .into_columns_unbounded()
            .map(|(name, values)| {
                let mut source = values.into_iter().map(Some).collect::<Vec<_>>();
                let values = (0..length)
                    .map(|index| {
                        let source_index = source_index(index, &shape)
                            .ok_or_else(|| "structure array reorder index overflow".to_string())?;
                        source
                            .get_mut(source_index)
                            .and_then(Option::take)
                            .ok_or_else(|| "structure array reorder is not bijective".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((name, values))
            })
            .collect::<Result<UnboundedOrderedMapBuilder<_>, String>>()?;
        Self::from_columns(fields, shape)
    }
}

fn validate_permutation(order: &[usize]) -> Result<(), String> {
    let mut seen = vec![false; order.len()];
    for dimension in order {
        let slot = seen
            .get_mut(*dimension)
            .ok_or_else(|| "structure array permutation dimension is out of range".to_string())?;
        if *slot {
            return Err("structure array permutation contains a duplicate dimension".into());
        }
        *slot = true;
    }
    Ok(())
}

fn permuted_source(
    index: usize,
    shape: &[usize],
    order: &[usize],
    strides: &[usize],
) -> Option<usize> {
    let mut remainder = index;
    shape
        .iter()
        .enumerate()
        .try_fold(0usize, |source, (axis, extent)| {
            let coordinate = remainder % extent;
            remainder /= extent;
            coordinate
                .checked_mul(strides[order[axis]])?
                .checked_add(source)
        })
}

fn tiled_source(
    index: usize,
    output: &[usize],
    source: &[usize],
    strides: &[usize],
) -> Result<usize, String> {
    let mut remainder = index;
    output
        .iter()
        .enumerate()
        .try_fold(0usize, |result, (dimension, extent)| {
            let coordinate = remainder % extent;
            remainder /= extent;
            (coordinate % source[dimension])
                .checked_mul(strides[dimension])
                .and_then(|offset| result.checked_add(offset))
                .ok_or_else(|| "structure array replication index overflow".to_string())
        })
}
