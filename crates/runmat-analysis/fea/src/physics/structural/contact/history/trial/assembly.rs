//! Scatter this smooth point contribution through explicit original translational rows.
//! A cancelled/failed assembly must be discarded by the owner; it is never accepted state.
use super::*;
use crate::algebra::RealSparseAssembler;
impl ContactPointTrial {
    pub fn assemble_into(
        &self,
        primary_rows: &[[usize; 3]],
        secondary_rows: &[[usize; 3]],
        residual: &mut [f64],
        tangent: &mut RealSparseAssembler,
    ) -> Result<(), ContactHistoryError> {
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled.into());
        }
        if primary_rows.len() != self.kinematics.count
            || secondary_rows.len() != self.secondary_count
            || primary_rows
                .iter()
                .chain(secondary_rows)
                .flatten()
                .any(|r| *r >= residual.len())
        {
            return Err(ContactHistoryError::InvalidScope);
        }
        for (rows, force) in primary_rows
            .iter()
            .zip(self.primary_internal_force_n())
            .chain(
                secondary_rows
                    .iter()
                    .zip(self.secondary_internal_force_n().iter()),
            )
        {
            for (&row, &value) in rows.iter().zip(force) {
                let next = residual[row] + value;
                if !next.is_finite() {
                    return Err(ContactHistoryError::InvalidScope);
                }
                residual[row] = next;
            }
        }
        for (local, rows) in primary_rows.iter().chain(secondary_rows).enumerate() {
            for (axis, &column) in rows.iter().enumerate() {
                if crate::is_fea_cancelled() {
                    return Err(BoundarySearchError::Cancelled.into());
                }
                let mut primary = [[0.; 3]; 6];
                let mut query = [0.; 3];
                if local < primary_rows.len() {
                    primary[local][axis] = 1.;
                } else {
                    query[axis] = self.secondary_shape[local - primary_rows.len()];
                }
                let action = self.tangent_action(query, &primary[..primary_rows.len()])?;
                for (rows, force) in primary_rows.iter().zip(action.primary_force_increment_n()) {
                    for (&row, &value) in rows.iter().zip(force) {
                        tangent.add(row, column, value);
                    }
                }
                for (local, rows) in secondary_rows.iter().enumerate() {
                    for (&row, &value) in rows.iter().zip(&action.query_force_increment_n) {
                        tangent.add(row, column, self.secondary_shape[local] * value);
                    }
                }
            }
        }
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled.into());
        }
        Ok(())
    }
}
