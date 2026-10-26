//! Authored primary-electric harmonic intent, independent of mesh/solver admission.
use super::{AuthoredModel, AuthoredModelError};
use crate::{AuthoredConstraintKind as Kind, ElectromagneticDomain};

pub(super) fn validate(model: &AuthoredModel) -> Result<(), AuthoredModelError> {
    let harmonic = match &model.electromagnetic {
        Some(ElectromagneticDomain::Harmonic {
            reference_frequency,
        }) => {
            if !reference_frequency.is_finite() || reference_frequency.canonical_value() <= 0.0 {
                return Err(AuthoredModelError::InvalidPhysics {
                    kind: "electromagnetic domain",
                    detail: "harmonic Maxwell reference frequency must be finite and positive"
                        .into(),
                });
            }
            true
        }
        None
        | Some(ElectromagneticDomain::Static { .. })
        | Some(ElectromagneticDomain::Magnetostatic { .. }) => false,
    };
    for constraint in &model.constraints {
        let pec = matches!(constraint.kind, Kind::ElectricPerfectConductor { .. });
        let incompatible = matches!(
            constraint.kind,
            Kind::ElectricalPotential { .. }
                | Kind::ElectricalEquipotential { .. }
                | Kind::ElectricalFloatingConductor { .. }
                | Kind::ElectricalIntegralCurrent { .. }
                | Kind::ElectricalPotentialReference { .. }
                | Kind::ElectricalPeriodic(_)
                | Kind::MagneticVectorPotential { .. }
                | Kind::MagneticVectorPotentialGround { .. }
                | Kind::MagneticCoulombGauge { .. }
                | Kind::MagneticPeriodic(_)
                | Kind::MagneticPhaseShiftedPeriodic(_)
        );
        if (pec && !harmonic) || (harmonic && incompatible) {
            return Err(AuthoredModelError::InvalidPhysics {
                kind: "electromagnetic constraint",
                detail: "electric PEC belongs to harmonic electric-field Maxwell; scalar-potential and magnetic constraints cannot be reinterpreted".into(),
            });
        }
    }
    Ok(())
}
