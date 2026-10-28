//! One control validation law for portable intent and actual numerical preparation.
use super::{CanonicalFluidError as Error, CanonicalFluidPolicy};
impl CanonicalFluidPolicy {
    pub fn validate(self) -> Result<(), Error> {
        if let Some(p) = self.fixed_point_acceleration {
            p.validate()
                .map_err(|_| Error::Invalid("fluid fixed point controls"))?;
        }
        if self.maximum_outer_iterations == 0
            || self.maximum_nonorthogonal_iterations == 0
            || self.maximum_linear_iterations == 0
            || self.gmres_restart == 0
            || self.maximum_resident_bytes == Some(0)
            || self.maximum_operator_work == Some(0)
            || self.maximum_authority_items == 0
            || !self.linear_relative_tolerance.is_finite()
            || !(0.0..1.0).contains(&self.linear_relative_tolerance)
            || self.linear_relative_tolerance == 0.
            || !self.physical_relative_tolerance.is_finite()
            || !(0.0..1.0).contains(&self.physical_relative_tolerance)
            || self.physical_relative_tolerance == 0.
            || !self.momentum_relaxation.is_finite()
            || !(0.0..=1.0).contains(&self.momentum_relaxation)
            || self.momentum_relaxation == 0.
            || !self.pressure_relaxation.is_finite()
            || !(0.0..=1.0).contains(&self.pressure_relaxation)
            || self.pressure_relaxation == 0.
        {
            return Err(Error::Invalid("fluid control bounds"));
        }
        Ok(())
    }
}
