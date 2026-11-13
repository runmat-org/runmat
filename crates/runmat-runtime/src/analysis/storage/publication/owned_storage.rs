//! Temporary typed owners have a shorter lifetime than borrowed input authority.
//! A consumed, publication-scoped token releases only those completed owners;
//! staged outputs and the original input remain charged through commit.
use super::*;
pub(in crate::analysis) struct OwnedCopyReservation {
    scope: Arc<()>,
    bytes: u64,
}
impl AnalysisPublicationBuilder {
    pub(in crate::analysis) fn admit_owned_copies(
        &mut self,
        bytes: u64,
    ) -> Result<OwnedCopyReservation, AnalysisPublicationError> {
        self.admit(bytes)?;
        self.preparation_resident_bytes = self
            .preparation_resident_bytes
            .checked_add(bytes)
            .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                message: "owned-copy storage overflow".into(),
            })?;
        Ok(OwnedCopyReservation {
            scope: self.preparation_scope.clone(),
            bytes,
        })
    }

    pub(in crate::analysis) fn admit_completion_storage(
        &mut self,
        original: u64,
        copies: u64,
    ) -> Result<(PreparationReservation, OwnedCopyReservation), AnalysisPublicationError> {
        let original_reservation = self.admit_preparation_storage(original, copies)?;
        Ok((
            original_reservation,
            OwnedCopyReservation {
                scope: self.preparation_scope.clone(),
                bytes: copies,
            },
        ))
    }
    /// Call only after every owner covered by this reservation has been dropped.
    /// The token is not Clone and cannot be reconstructed from serialized claims.
    pub(in crate::analysis) fn release_owned_copies(
        &mut self,
        reservation: OwnedCopyReservation,
    ) -> Result<(), AnalysisPublicationError> {
        if !Arc::ptr_eq(&self.preparation_scope, &reservation.scope) {
            return Err("owned copies belong to another publication".into());
        }
        self.preparation_resident_bytes = self
            .preparation_resident_bytes
            .checked_sub(reservation.bytes)
            .ok_or("owned-copy storage reservation underflow")?;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn completed_copy_release_preserves_original_charge_and_publication_scope() {
        let definition = Digest::sha256(b"completion-phase");
        let mut builder = AnalysisPublicationBuilder::new(definition, 100).unwrap();
        let (original, copies) = builder.admit_completion_storage(30, 70).unwrap();
        assert!(builder.admit(1).is_err());
        builder.release_owned_copies(copies).unwrap();
        assert_eq!(builder.preparation_resident_bytes, original.resident_bytes);
        assert!(builder.admit(70).is_ok());
        assert!(builder.admit(71).is_err());
        let (_, foreign) = AnalysisPublicationBuilder::new(definition, 100)
            .unwrap()
            .admit_completion_storage(1, 1)
            .unwrap();
        assert!(builder.release_owned_copies(foreign).is_err());
        assert_eq!(builder.preparation_resident_bytes, 30);
    }

    // witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: Two independently reserved temporary owners coexist with original input under a fixed
    //   ceiling. Failed growth preserves charges; consuming each reservation releases only that owner.
    //   Component bookkeeping only, without original fork completion, measured RSS or whole-task peak
    //   qualification.
    #[test]
    fn independent_owner_phases_remain_charged_until_each_owner_is_released() {
        let mut builder =
            AnalysisPublicationBuilder::new(Digest::sha256(b"owner-overlap"), 100).unwrap();
        builder.admit_preparation_storage(20, 0).unwrap();
        let first = builder.admit_owned_copies(30).unwrap();
        let second = builder.admit_owned_copies(40).unwrap();
        assert!(builder.admit_owned_copies(11).is_err());
        assert_eq!(builder.preparation_resident_bytes, 90);
        builder.release_owned_copies(first).unwrap();
        assert_eq!(builder.preparation_resident_bytes, 60);
        assert!(builder.admit(40).is_ok());
        assert!(builder.admit(41).is_err());
        builder.release_owned_copies(second).unwrap();
        assert_eq!(builder.preparation_resident_bytes, 20);
    }
}
