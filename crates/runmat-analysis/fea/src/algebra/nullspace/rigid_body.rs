use crate::assembly::dofs::StructuralDofKind;

use super::NullspaceEvidenceError;

#[derive(Clone, Copy, Debug)]
pub(super) struct RigidBodyFrame {
    centroid: [f64; 3],
    characteristic_length: f64,
}

impl RigidBodyFrame {
    pub(super) fn from_positions(positions: &[[f64; 3]]) -> Result<Self, NullspaceEvidenceError> {
        if positions.is_empty()
            || positions
                .iter()
                .flatten()
                .any(|coordinate| !coordinate.is_finite())
        {
            return Err(NullspaceEvidenceError::DimensionMismatch);
        }
        let mut centroid = [0.0; 3];
        for position in positions {
            for axis in 0..3 {
                centroid[axis] += position[axis];
            }
        }
        for coordinate in &mut centroid {
            *coordinate /= positions.len() as f64;
        }
        let characteristic_length = positions
            .iter()
            .map(|position| {
                position
                    .iter()
                    .zip(centroid)
                    .map(|(coordinate, center)| (coordinate - center).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(0.0_f64, f64::max);
        if !characteristic_length.is_finite() || characteristic_length <= f64::MIN_POSITIVE {
            return Err(NullspaceEvidenceError::DimensionMismatch);
        }
        Ok(Self {
            centroid,
            characteristic_length,
        })
    }

    pub(super) fn response(self, position: [f64; 3], kind: StructuralDofKind) -> [f64; 6] {
        let [x, y, z] = [
            (position[0] - self.centroid[0]) / self.characteristic_length,
            (position[1] - self.centroid[1]) / self.characteristic_length,
            (position[2] - self.centroid[2]) / self.characteristic_length,
        ];
        let inverse_length = self.characteristic_length.recip();
        match kind {
            StructuralDofKind::Ux => [1.0, 0.0, 0.0, 0.0, z, -y],
            StructuralDofKind::Uy => [0.0, 1.0, 0.0, -z, 0.0, x],
            StructuralDofKind::Uz => [0.0, 0.0, 1.0, y, -x, 0.0],
            StructuralDofKind::Rx => [0.0, 0.0, 0.0, inverse_length, 0.0, 0.0],
            StructuralDofKind::Ry => [0.0, 0.0, 0.0, 0.0, inverse_length, 0.0],
            StructuralDofKind::Rz => [0.0, 0.0, 0.0, 0.0, 0.0, inverse_length],
        }
    }
}
