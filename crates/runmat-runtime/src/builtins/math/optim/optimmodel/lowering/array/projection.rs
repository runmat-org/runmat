//! Linear row projection of compact coefficients. Input row IDs remain
//! sparse; a tiny selection never allocates a full input-row pointer table.
use super::super::terms::{Coefficients, Term};
use super::super::*;
use super::{Array, Assigned, Degree, Offset};
use runmat_value::{NumericStorage, Tensor};
pub(super) enum RowMap {
    Identity {
        rows: usize,
    },
    Gather {
        inputs: usize,
        outputs: usize,
        map: Coefficients,
    },
}
impl RowMap {
    pub(super) fn input_rows(&self) -> usize {
        match self {
            Self::Identity { rows } => *rows,
            Self::Gather { inputs, .. } => *inputs,
        }
    }
    pub(super) fn output_rows(&self) -> usize {
        match self {
            Self::Identity { rows } => *rows,
            Self::Gather { outputs, .. } => *outputs,
        }
    }
    pub(super) fn entries(&self, column: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let (single, sparse) = match self {
            Self::Identity { rows } => (
                if column < *rows {
                    Some((column, 1.))
                } else {
                    None
                },
                None,
            ),
            Self::Gather { map, .. } => (None, Some(map)),
        };
        single
            .into_iter()
            .chain(sparse.into_iter().flat_map(move |map| map.column(column)))
    }
    fn source_rows(&self) -> impl Iterator<Item = usize> + '_ {
        let (range, columns) = match self {
            Self::Identity { rows } => (0..*rows, &[][..]),
            Self::Gather { map, .. } => (0..0, map.columns()),
        };
        range.chain(columns.iter().copied())
    }
    pub(super) async fn gather(
        selected: &[usize],
        input_rows: usize,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> Lowered<Self> {
        if selected.len() == input_rows {
            let mut identity = true;
            for (row, &source) in selected.iter().enumerate() {
                identity &= row == source;
                cooperation.completed(1).await?;
            }
            if identity {
                return Ok(Ok(Self::Identity { rows: input_rows }));
            }
        }
        let mut entries =
            runmat_value::exact_capacity_vec(selected.len()).map_err(|e| policy.storage(e))?;
        for (row, &source) in selected.iter().enumerate() {
            if source >= input_rows {
                return Err(policy.invalid("compiled selector exceeds original expression rows"));
            }
            entries.push(Term {
                column: source,
                row,
                value: 1.,
            });
            cooperation.completed(1).await?;
        }
        let map = match Coefficients::from_entries(
            entries,
            selected.len(),
            input_rows,
            policy,
            cooperation,
        )
        .await?
        {
            Ok(map) => map,
            Err(stop) => return Ok(Err(stop)),
        };
        Ok(Ok(Self::Gather {
            inputs: input_rows,
            outputs: selected.len(),
            map,
        }))
    }
}
pub(super) async fn project(
    input: Array,
    map: &RowMap,
    shape: SharedShape,
    columns: usize,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> Lowered<Array> {
    if input.count(policy)? != map.input_rows()
        || evaluation::element_count(&shape, policy)? != map.output_rows()
    {
        return Err(policy.invalid("compiled row map differs from its original shape"));
    }
    if matches!(map, RowMap::Identity { .. }) {
        return Ok(Ok(Array { shape, ..input }));
    }
    // A zero polynomial offset stays a uniform symbolic coefficient, not a
    // fabricated dense array. Ordinary fully assigned input needs no mask.
    // Every gather output has exactly one unit-weight source. Uniform
    // polynomial offsets therefore stay uniform, including scalar expansion.
    let uniform = match &input.offset {
        Offset::Uniform(value) => Some(*value),
        _ => None,
    };
    let mut offset = if uniform.is_some() {
        None
    } else {
        let mut values =
            runmat_value::exact_capacity_vec(map.output_rows()).map_err(|e| policy.storage(e))?;
        for _ in 0..map.output_rows() {
            values.push(0.);
            cooperation.completed(1).await?;
        }
        Some(values)
    };
    let mut assigned = if matches!(input.assigned, Assigned::All) {
        None
    } else {
        let mut values =
            runmat_value::exact_capacity_vec(map.output_rows()).map_err(|e| policy.storage(e))?;
        for _ in 0..map.output_rows() {
            values.push(true);
            cooperation.completed(1).await?;
        }
        Some(values)
    };
    let mut has_missing = false;
    let mut entries = Vec::new();
    if offset.is_some() || assigned.is_some() {
        for source in map.source_rows() {
            let valid = input.assigned.at(source, policy)?;
            // Unassigned slots are never treated as numerical zero: their explicit
            // mask is propagated to every dependent output before publication.
            let value = if valid {
                input.offset.at(source, policy)?
            } else {
                0.
            };
            for (row, weight) in map.entries(source) {
                if weight != 0. {
                    if let Some(assigned) = &mut assigned {
                        assigned[row] &= valid;
                        has_missing |= !valid;
                    }
                    if let Some(offset) = &mut offset {
                        offset[row] += weight * value;
                        if !offset[row].is_finite() {
                            return Ok(Err(Stop::NonFinite));
                        }
                    }
                }
                cooperation.completed(1).await?;
            }
        }
    }
    for term in input.coefficients.entries() {
        for (row, weight) in map.entries(term.row) {
            let value = term.value * weight;
            if value != 0. {
                entries.try_reserve(1).map_err(|e| policy.reserve(e))?;
                entries.push(Term {
                    column: term.column,
                    row,
                    value,
                });
            }
            cooperation.completed(1).await?;
        }
    }
    let coefficients =
        match Coefficients::from_entries(entries, map.output_rows(), columns, policy, cooperation)
            .await?
        {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        };
    let assigned = if has_missing {
        Assigned::Partial(std::rc::Rc::new(
            assigned.expect("partial input retains its row mask"),
        ))
    } else {
        Assigned::All
    };
    let offset = match offset {
        None => Offset::Uniform(uniform.expect("uniform gather source")),
        Some(values) => Offset::Dense(
            Tensor::from_numeric_storage(NumericStorage::F64(values), shape.clone())
                .map_err(|source| policy.invalid(source))?,
        ),
    };
    let mut dependent = std::collections::HashSet::new();
    for source in input.degree.rows(input.count(policy)?) {
        if input.degree.at(source) {
            for (row, _) in map.entries(source) {
                dependent.try_reserve(1).map_err(|e| policy.reserve(e))?;
                dependent.insert(row);
                cooperation.completed(1).await?;
            }
        }
        cooperation.completed(1).await?;
    }
    let degree = Degree::collect(
        map.output_rows(),
        |row| Ok(dependent.contains(&row)),
        policy,
        cooperation,
    )
    .await?;
    Ok(Ok(Array {
        shape,
        offset,
        coefficients,
        assigned,
        degree,
    }))
}
