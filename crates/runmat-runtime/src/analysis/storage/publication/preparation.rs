//! Reuse an already charged borrowed input across a source-admission boundary.
use super::*;

pub(in crate::analysis) struct PreparationReservation {
    pub(super) scope: Arc<()>,
    pub(super) resident_bytes: u64,
}

impl AnalysisPublicationBuilder {
    /// The source's inclusive allowance contains this input; its original profile
    /// is subtracted by the source. Handoff releases neither the original nor
    /// temporary copy reservations. Temporary owners end only through consuming
    /// their scoped reservation after drop; original input remains charged.
    pub(in crate::analysis) fn source_workspace_with_preparation(
        &self,
        reservation: &PreparationReservation,
        input: &impl Serialize,
    ) -> Result<u64, AnalysisPublicationError> {
        if !Arc::ptr_eq(&self.preparation_scope, &reservation.scope) {
            return Err("source input reservation belongs to another publication".into());
        }
        let bytes = typed_resident(input, self.maximum_bytes)?;
        if bytes > reservation.resident_bytes {
            return Err(AnalysisPublicationError::ResourceLimit {
                message: "source input exceeds its retained preparation reservation".into(),
            });
        }
        self.source_workspace_limit()?
            .checked_add(bytes)
            .filter(|bytes| *bytes <= self.maximum_bytes)
            .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                message: "source input handoff exceeds captured publication allowance".into(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_input_handoff_is_scoped_and_does_not_release_commit_reservation() {
        let input = vec![1.25_f64; 4096];
        let definition = Digest::sha256(b"retained-input-handoff");
        let maximum = 2 << 20;
        let mut builder = AnalysisPublicationBuilder::new(definition, maximum).unwrap();
        let token = builder
            .admit_preparation(&(&input, "original metadata"))
            .unwrap();
        let charged = builder.preparation_resident_bytes;
        let available = builder.source_workspace_limit().unwrap();
        let candidate_bytes = super::super::typed_resident(&input, maximum).unwrap();
        let inclusive = builder
            .source_workspace_with_preparation(&token, &input)
            .unwrap();
        assert_eq!(inclusive - candidate_bytes, available);
        assert!(inclusive <= maximum);
        assert_eq!(builder.preparation_resident_bytes, charged);
        assert_eq!(builder.source_workspace_limit().unwrap(), available);

        // Same definition/ceiling does not grant access to another builder's charge.
        let foreign = AnalysisPublicationBuilder::new(definition, maximum).unwrap();
        assert!(foreign
            .source_workspace_with_preparation(&token, &input)
            .is_err());
        assert!(matches!(
            builder.source_workspace_with_preparation(&token, &vec![1.25_f64; 8192]),
            Err(AnalysisPublicationError::ResourceLimit { .. })
        ));
        assert_eq!(builder.preparation_resident_bytes, charged);
        assert_eq!(builder.source_workspace_limit().unwrap(), available);
    }
}
