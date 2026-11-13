use super::*;

enum AssemblyStorage {
    Discovered(Vec<BTreeMap<usize, f64>>),
    Planned {
        pattern: SparsePattern,
        values: Vec<f64>,
    },
}

/// Deterministic contribution accumulation with either discovered or pre-admitted support.
pub struct RealSparseAssembler {
    columns: usize,
    storage: AssemblyStorage,
    role: OperatorRole,
    properties: MatrixProperties,
    pub(super) contributions: Vec<ContributionProvenance>,
    error: Option<RealSparseMatrixError>,
}

impl RealSparseAssembler {
    pub fn new(
        rows: usize,
        columns: usize,
        role: OperatorRole,
        properties: MatrixProperties,
    ) -> Self {
        Self {
            columns,
            storage: AssemblyStorage::Discovered((0..rows).map(|_| BTreeMap::new()).collect()),
            role,
            properties,
            contributions: Vec::new(),
            error: None,
        }
    }

    pub fn square(size: usize, role: OperatorRole, properties: MatrixProperties) -> Self {
        Self::new(size, size, role, properties)
    }

    /// Uses only the supplied canonical support and one value buffer. Nonzero contributions
    /// outside the support fail closed; finalization compacts cancelled entries in place.
    pub fn with_pattern(
        pattern: SparsePattern,
        role: OperatorRole,
        properties: MatrixProperties,
    ) -> Result<Self, RealSparseMatrixError> {
        let mut values = reserve_payload(pattern.column_indices().len())?;
        values.resize(pattern.column_indices().len(), 0.0);
        Ok(Self {
            columns: pattern.columns(),
            storage: AssemblyStorage::Planned { pattern, values },
            role,
            properties,
            contributions: Vec::new(),
            error: None,
        })
    }

    /// Wraps values already assembled on `pattern` (e.g. by parallel element assembly).
    /// Finalization applies the same zero compaction and finiteness checks as `add`.
    pub fn from_planned_values(
        pattern: SparsePattern,
        values: Vec<f64>,
        role: OperatorRole,
        properties: MatrixProperties,
    ) -> Result<Self, RealSparseMatrixError> {
        if values.len() != pattern.column_indices().len() {
            return Err(RealSparseMatrixError::ValueCountMismatch {
                expected: pattern.column_indices().len(),
                actual: values.len(),
            });
        }
        Ok(Self {
            columns: pattern.columns(),
            storage: AssemblyStorage::Planned { pattern, values },
            role,
            properties,
            contributions: Vec::new(),
            error: None,
        })
    }

    pub fn add(&mut self, row: usize, column: usize, value: f64) {
        let rows = match &self.storage {
            AssemblyStorage::Discovered(rows) => rows.len(),
            AssemblyStorage::Planned { pattern, .. } => pattern.rows(),
        };
        if row >= rows || column >= self.columns {
            self.error
                .get_or_insert(RealSparseMatrixError::ShapeMismatch);
            return;
        }
        if !value.is_finite() {
            self.error
                .get_or_insert(RealSparseMatrixError::NonFiniteValue { entry: 0 });
            return;
        }
        if value == 0.0 {
            return;
        }
        match &mut self.storage {
            AssemblyStorage::Discovered(rows) => *rows[row].entry(column).or_insert(0.0) += value,
            AssemblyStorage::Planned { pattern, values } => {
                let range = pattern.row_range(row);
                match pattern.column_indices()[range.clone()].binary_search(&(column as u64)) {
                    Ok(index) => values[range.start + index] += value,
                    Err(_) => {
                        self.error
                            .get_or_insert(RealSparseMatrixError::OutsidePlannedPattern {
                                row,
                                column,
                            });
                    }
                }
            }
        }
    }

    pub fn add_contribution(&mut self, provenance: ContributionProvenance) {
        self.contributions.push(provenance);
    }

    pub fn finish(self) -> Result<RealSparseMatrix, RealSparseMatrixError> {
        if let Some(error) = self.error {
            return Err(error);
        }
        let (pattern, values) = match self.storage {
            AssemblyStorage::Discovered(rows) => finish_discovered(rows, self.columns)?,
            AssemblyStorage::Planned { pattern, values } => finish_planned(pattern, values)?,
        };
        RealSparseMatrix::try_new(
            pattern,
            values,
            self.role,
            self.properties,
            self.contributions,
        )
    }
}

fn finish_discovered(
    rows: Vec<BTreeMap<usize, f64>>,
    columns: usize,
) -> Result<(SparsePattern, Vec<f64>), RealSparseMatrixError> {
    let row_count = rows.len();
    let offset_count = row_count
        .checked_add(1)
        .ok_or(SparsePatternError::DimensionOverflow)?;
    let entry_count = rows.iter().try_fold(0_usize, |count, row| {
        count
            .checked_add(row.values().filter(|value| **value != 0.0).count())
            .ok_or(SparsePatternError::DimensionOverflow)
    })?;
    let mut offsets = reserve_payload(offset_count)?;
    let mut column_indices = reserve_payload(entry_count)?;
    let mut values = reserve_payload(entry_count)?;
    offsets.push(0_u64);
    for row in rows {
        for (column, value) in row {
            if value != 0.0 {
                column_indices.push(
                    u64::try_from(column).map_err(|_| SparsePatternError::DimensionOverflow)?,
                );
                values.push(value);
            }
        }
        offsets
            .push(u64::try_from(values.len()).map_err(|_| SparsePatternError::DimensionOverflow)?);
    }
    Ok((
        SparsePattern::try_new(row_count as u64, columns as u64, offsets, column_indices)?,
        values,
    ))
}

fn finish_planned(
    pattern: SparsePattern,
    mut values: Vec<f64>,
) -> Result<(SparsePattern, Vec<f64>), RealSparseMatrixError> {
    let rows = pattern.rows();
    let columns = pattern.columns();
    // Unchanged support remains shared. Compaction owns its buffers and must
    // not mutate another logical operator's canonical support.
    if values.iter().all(|value| *value != 0.0) {
        return Ok((pattern, values));
    }
    let (mut offsets, mut indices) = pattern.into_index_buffers()?;
    let mut read_start = 0;
    let mut write = 0;
    for row in 0..rows {
        let read_end = offsets[row + 1] as usize;
        offsets[row] = write as u64;
        for read in read_start..read_end {
            if values[read] != 0.0 {
                indices[write] = indices[read];
                values[write] = values[read];
                write += 1;
            }
        }
        read_start = read_end;
    }
    offsets[rows] = write as u64;
    indices.truncate(write);
    values.truncate(write);
    Ok((
        SparsePattern::try_new(rows as u64, columns as u64, offsets, indices)?,
        values,
    ))
}

#[cfg(test)]
mod tests;
