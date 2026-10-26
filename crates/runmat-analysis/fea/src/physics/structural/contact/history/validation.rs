use super::*;
impl ContactPointBinding<'_> {
    pub fn validate_history(
        &self,
        history: &ContactMaterialPointHistory,
        expected_ordinal: u64,
        accepted_positions_m: &[[f64; 3]],
    ) -> Result<(), ContactHistoryError> {
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled.into());
        }
        if history.schema != 1 || history.scope != self.scope || history.ordinal != expected_ordinal
        {
            return Err(ContactHistoryError::InvalidHistory);
        }
        let primary_index = self.primary_index(history.primary_face)?;
        let Configuration {
            primary,
            secondary,
            relative,
            frame,
            gap,
            ..
        } = self.configuration(
            primary_index,
            history.primary_barycentric,
            accepted_positions_m,
        )?;
        if primary != history.primary_coordinates_m
            || secondary != history.secondary_coordinates_m
            || relative != history.relative_material_vector_m
            || frame != history.frame
            || gap != history.point.gap_m().canonical_value()
        {
            return Err(ContactHistoryError::InvalidHistory);
        }
        self.law.validate_state(&history.point)?;
        Ok(())
    }
}
