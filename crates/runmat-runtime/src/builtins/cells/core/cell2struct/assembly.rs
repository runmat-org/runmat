use runmat_value::{CellArray, StructArray, StructValue, Value};

pub(super) fn build(
    cells: CellArray,
    fields: Vec<String>,
    dimension: usize,
) -> crate::BuiltinResult<Value> {
    let rank = cells.shape.len().max(dimension);
    let mut input_shape = cells.shape.clone();
    input_shape
        .try_reserve_exact(rank.saturating_sub(input_shape.len()))
        .map_err(|_| super::error::invalid("dim exceeds platform limits"))?;
    input_shape.resize(rank, 1);
    let field_dimension = dimension - 1;
    if input_shape[field_dimension] != fields.len() {
        return Err(super::error::shape(format!(
            "selected dimension has extent {}, but {} field names were supplied",
            input_shape[field_dimension],
            fields.len()
        )));
    }
    let mut output_shape = input_shape.clone();
    output_shape[field_dimension] = 1;
    let output_count = checked_count(&output_shape)?;
    let values = cells.into_column_major().map_err(super::error::invalid)?;
    let mut values = values.into_iter().map(Some).collect::<Vec<_>>();
    let mut structures = Vec::new();
    structures
        .try_reserve_exact(output_count)
        .map_err(|_| super::error::invalid("output shape exceeds platform limits"))?;
    for output_index in 0..output_count {
        let mut coordinates = coordinates(output_index, &output_shape);
        let mut structure = StructValue::new();
        for (field_index, field) in fields.iter().enumerate() {
            coordinates[field_dimension] = field_index;
            let input_index = linear_index(&coordinates, &input_shape)?;
            let value = values
                .get_mut(input_index)
                .and_then(Option::take)
                .ok_or_else(|| super::error::invalid("cell storage does not match its shape"))?;
            structure.insert(field.clone(), value);
        }
        structures.push(structure);
    }
    StructArray::normalize(fields, structures, output_shape).map_err(super::error::invalid)
}

fn checked_count(shape: &[usize]) -> crate::BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |count, extent| {
        count
            .checked_mul(*extent)
            .ok_or_else(|| super::error::invalid("output shape exceeds platform limits"))
    })
}

fn coordinates(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .map(|extent| {
            let coordinate = if *extent == 0 { 0 } else { linear % extent };
            if *extent != 0 {
                linear /= extent;
            }
            coordinate
        })
        .collect()
}

fn linear_index(coordinates: &[usize], shape: &[usize]) -> crate::BuiltinResult<usize> {
    coordinates
        .iter()
        .zip(shape)
        .try_fold(
            (0usize, 1usize),
            |(linear, stride), (coordinate, extent)| {
                let offset = coordinate
                    .checked_mul(stride)
                    .ok_or_else(|| super::error::invalid("cell index exceeds platform limits"))?;
                Ok((
                    linear.checked_add(offset).ok_or_else(|| {
                        super::error::invalid("cell index exceeds platform limits")
                    })?,
                    stride.checked_mul(*extent).ok_or_else(|| {
                        super::error::invalid("cell shape exceeds platform limits")
                    })?,
                ))
            },
        )
        .map(|(linear, _)| linear)
}
