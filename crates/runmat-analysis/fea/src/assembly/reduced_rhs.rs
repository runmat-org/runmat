use crate::excitation::{RealizedExcitationKind, RealizedExcitations};

pub(super) fn apply_reduced_multiphysics_rhs(excitations: &RealizedExcitations, rhs: &mut [f64]) {
    for (index, load) in excitations.iter().enumerate() {
        let base = (index * 3) % rhs.len();
        match &load.kind {
            RealizedExcitationKind::Force(value) | RealizedExcitationKind::BodyForce(value) => {
                let [x, y, z] = value.real;
                rhs[base] += x;
                if base + 1 < rhs.len() {
                    rhs[base + 1] += y;
                }
                if base + 2 < rhs.len() {
                    rhs[base + 2] += z;
                }
            }
            RealizedExcitationKind::Wrench { force, .. } => {
                let [x, y, z] = force.real;
                rhs[base] += x;
                if base + 1 < rhs.len() {
                    rhs[base + 1] += y;
                }
                if base + 2 < rhs.len() {
                    rhs[base + 2] += z;
                }
            }
            RealizedExcitationKind::Pressure(value) => {
                rhs[base] += value.real * 1.0e-3;
                if base + 1 < rhs.len() {
                    rhs[base + 1] -= value.real * 1.0e-3;
                }
            }
            RealizedExcitationKind::CurrentDensity(value) => {
                let [x, y, z] = value.real;
                rhs[base] += x * 1.0e-3;
                if base + 1 < rhs.len() {
                    rhs[base + 1] += y * 1.0e-3;
                }
                if base + 2 < rhs.len() {
                    rhs[base + 2] += z * 1.0e-3;
                }
            }
            RealizedExcitationKind::ElectricCurrent(value) => {
                rhs[base] += value.real * 1.0e-2;
            }
            _ => {}
        }
    }
}
