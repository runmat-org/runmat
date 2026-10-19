#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EnumerationError {
    CardinalityOverflow,
    ElementLimitExceeded,
    SequenceInvariant,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct MaterializationLimit {
    pub max_input_elements: Option<usize>,
    pub max_output_elements: usize,
}

pub(super) fn permutation_rows(
    elements: usize,
    limit: MaterializationLimit,
) -> Result<usize, EnumerationError> {
    if limit
        .max_input_elements
        .is_some_and(|maximum| elements > maximum)
    {
        return Err(EnumerationError::ElementLimitExceeded);
    }
    let rows = checked_factorial(elements)?;
    let total = rows
        .checked_mul(elements)
        .ok_or(EnumerationError::CardinalityOverflow)?;
    if total > limit.max_output_elements {
        return Err(EnumerationError::ElementLimitExceeded);
    }
    Ok(rows)
}

pub(super) fn checked_factorial(value: usize) -> Result<usize, EnumerationError> {
    (2..=value).try_fold(1_usize, |product, factor| {
        product
            .checked_mul(factor)
            .ok_or(EnumerationError::CardinalityOverflow)
    })
}

pub(super) fn checked_binomial_u128(
    population: usize,
    selection: usize,
) -> Result<u128, EnumerationError> {
    if selection > population {
        return Ok(0);
    }
    let selection = selection.min(population - selection);
    (1..=selection).try_fold(1_u128, |result, index| {
        result
            .checked_mul((population - selection + index) as u128)
            .map(|product| product / index as u128)
            .ok_or(EnumerationError::CardinalityOverflow)
    })
}

pub(super) fn checked_binomial_usize(
    population: usize,
    selection: usize,
) -> Result<usize, EnumerationError> {
    usize::try_from(checked_binomial_u128(population, selection)?)
        .map_err(|_| EnumerationError::CardinalityOverflow)
}

pub(super) fn for_each_combination(
    population: usize,
    selection: usize,
    mut visit: impl FnMut(&[usize]),
) {
    if selection == 0 {
        visit(&[]);
        return;
    }
    if selection > population {
        return;
    }
    let mut indices = (0..selection).collect::<Vec<_>>();
    loop {
        visit(&indices);
        let mut position = selection;
        while position > 0 {
            position -= 1;
            if indices[position] != position + population - selection {
                break;
            }
        }
        if position == 0 && indices[position] == population - selection {
            break;
        }
        indices[position] += 1;
        for index in position + 1..selection {
            indices[index] = indices[index - 1] + 1;
        }
    }
}

pub(super) fn permuted_columns<T: Clone>(
    input: &[T],
    rows: usize,
) -> Result<Vec<T>, EnumerationError> {
    permute(input, rows, OutputLayout::ColumnMajor)
}

pub(super) fn permuted_rows<T: Clone>(
    input: &[T],
    rows: usize,
) -> Result<Vec<T>, EnumerationError> {
    permute(input, rows, OutputLayout::RowMajor)
}

#[derive(Debug, Clone, Copy)]
enum OutputLayout {
    ColumnMajor,
    RowMajor,
}

fn permute<T: Clone>(
    input: &[T],
    rows: usize,
    layout: OutputLayout,
) -> Result<Vec<T>, EnumerationError> {
    let columns = input.len();
    if columns == 0 {
        return Ok(Vec::new());
    }
    let total = rows
        .checked_mul(columns)
        .ok_or(EnumerationError::CardinalityOverflow)?;
    let mut output = vec![input[0].clone(); total];
    let mut indices: Vec<usize> = (0..columns).rev().collect();
    for row in 0..rows {
        for (column, &source) in indices.iter().enumerate() {
            let destination = match layout {
                OutputLayout::ColumnMajor => column * rows + row,
                OutputLayout::RowMajor => row * columns + column,
            };
            output[destination] = input[source].clone();
        }
        if row + 1 < rows && !previous_permutation(&mut indices) {
            return Err(EnumerationError::SequenceInvariant);
        }
    }
    Ok(output)
}

fn previous_permutation(values: &mut [usize]) -> bool {
    if values.len() < 2 {
        return false;
    }
    let Some(pivot) = (0..values.len() - 1).rfind(|&index| values[index] > values[index + 1])
    else {
        return false;
    };
    let swap_with = (pivot + 1..values.len())
        .rfind(|&index| values[index] < values[pivot])
        .expect("a descending pivot has a smaller suffix value");
    values.swap(pivot, swap_with);
    values[pivot + 1..].reverse();
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_order_is_reverse_lexicographic() {
        let rows = permutation_rows(
            3,
            MaterializationLimit {
                max_input_elements: None,
                max_output_elements: 100,
            },
        )
        .expect("rows");
        assert_eq!(
            permuted_rows(&[1, 2, 3], rows).expect("permutations"),
            vec![3, 2, 1, 3, 1, 2, 2, 3, 1, 2, 1, 3, 1, 3, 2, 1, 2, 3]
        );
    }

    #[test]
    fn limits_are_checked_before_allocation() {
        assert_eq!(
            permutation_rows(
                4,
                MaterializationLimit {
                    max_input_elements: Some(3),
                    max_output_elements: usize::MAX,
                },
            ),
            Err(EnumerationError::ElementLimitExceeded)
        );
    }
}
