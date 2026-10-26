//! Original ordered point projections, not averaged history or restart authority.
use super::*;
impl CanonicalContactOperator<'_> {
    pub fn point_outputs(
        &self,
        state: &ContactConfigurationState,
    ) -> Result<(Vec<f64>, Vec<f64>), ContactOperatorError> {
        self.validate_configuration(state, state.ordinal())?;
        let mut pressure = reserve(self.point_count)?;
        let mut gap = reserve(self.point_count)?;
        let mut ordinal = 0;
        for interface in &self.interfaces {
            for point in &interface.points {
                if crate::is_fea_cancelled() {
                    return Err(ContactOperatorError::Cancelled);
                }
                let original = state.point_history()[ordinal].point();
                pressure.push(point.normal_pressure_pa(original)?.canonical_value());
                gap.push(original.gap_m().canonical_value());
                ordinal += 1;
            }
        }
        Ok((pressure, gap))
    }
}
