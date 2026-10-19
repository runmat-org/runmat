//! Exact homogeneous selection. Eliminated vertices anchor the energy graph;
//! reconstructed coefficients are compared in the actual operator's coordinates.
use super::{EnergyError, Interval, RealSparseMatrix};

#[derive(Clone, Copy)]
pub(super) enum EnergyProjection<'a> {
    Identity(&'a [bool]),
    Selection(&'a [Option<usize>]),
}

impl EnergyProjection<'_> {
    pub fn len(self) -> usize {
        match self {
            Self::Identity(fixed) => fixed.len(),
            Self::Selection(rows) => rows.len(),
        }
    }

    pub fn row(self, vertex: usize) -> Option<usize> {
        match self {
            Self::Identity(fixed) => (!fixed[vertex]).then_some(vertex),
            Self::Selection(rows) => rows[vertex],
        }
    }

    pub fn validate(self, dimension: usize) -> Result<(), EnergyError> {
        let valid = match self {
            Self::Identity(fixed) => fixed.len() == dimension,
            Self::Selection(rows) => {
                // Called after storage admission. Exact linear-time coverage avoids
                // repeatedly scanning a large constraint inventory.
                let mut seen = vec![false; dimension];
                let mut count = 0;
                for &row in rows.iter().flatten() {
                    let Some(entry) = seen.get_mut(row) else {
                        return Err(EnergyError::Shape);
                    };
                    if *entry {
                        return Err(EnergyError::Shape);
                    }
                    *entry = true;
                    count += 1;
                }
                count == dimension
            }
        };
        if valid {
            Ok(())
        } else {
            Err(EnergyError::Shape)
        }
    }

    pub fn seed_entries(self, matrix: &RealSparseMatrix) -> Result<Vec<Interval>, EnergyError> {
        let mut entries = vec![Interval::exact(0.0); matrix.values().len()];
        match self {
            Self::Selection(_) => {}
            Self::Identity(fixed) => {
                for (row, &fixed) in fixed.iter().enumerate() {
                    if fixed {
                        let index = matrix
                            .pattern()
                            .row_range(row)
                            .find(|&i| matrix.pattern().column(i) == row)
                            .ok_or(EnergyError::Shape)?;
                        entries[index] = Interval::exact(1.0);
                    }
                }
            }
        }
        Ok(entries)
    }
}

#[cfg(test)]
impl<'a> super::AnchoredEnergyCheck<'a> {
    #[cfg(test)]
    pub fn new_selection(
        matrix: &'a RealSparseMatrix,
        rows: &'a [Option<usize>],
        admitted: super::CoercivityWorkEnvelope,
    ) -> Result<Self, EnergyError> {
        Self::new_projected(
            matrix,
            EnergyProjection::Selection(rows),
            admitted.into(),
            std::mem::size_of_val(rows) as u64,
            0,
        )
    }
}
