//! Requested operator/trial payload. Borrowed sources and committed state are additional.
use super::*;
#[derive(Clone, Copy, Debug)]
pub struct ContactOperatorResources {
    pub prepared_resident_bytes: u64,
    pub trial_resident_bytes: u64,
    pub trial_peak_additional_bytes: u64,
    pub identity_construction_bytes: u64,
    pub construction_peak_additional_bytes: u64,
}
fn slots<T>(count: usize) -> Result<u64, ContactOperatorError> {
    crate::topology::storage::slots::<T>(count).map_err(|_| ContactOperatorError::Budget)
}
fn add(a: u64, b: u64) -> Result<u64, ContactOperatorError> {
    a.checked_add(b).ok_or(ContactOperatorError::Budget)
}
impl CanonicalContactOperator<'_> {
    pub fn resources(&self) -> Result<ContactOperatorResources, ContactOperatorError> {
        let pattern = self.pattern.retained_storage_bytes()?;
        let mut prepared = add(
            pattern,
            slots::<PreparedInterface<'_>>(self.interfaces.len())?,
        )?;
        let mut search = 0;
        let mut scopes = self.identity_construction_bytes;
        for interface in &self.interfaces {
            prepared = add(prepared, interface.inventory.resident_bytes()?)?;
            prepared = add(
                prepared,
                slots::<ContactPreparedPoint<'_>>(interface.points.len())?,
            )?;
            scopes = scopes.max(
                interface
                    .points
                    .iter()
                    .map(ContactPreparedPoint::construction_resident_bytes)
                    .max()
                    .unwrap_or(0),
            );
            let index = BoundarySurfaceIndex::resident_bytes(
                interface.inventory.primary_face_indices().len(),
            )?;
            let narrow = interface
                .policy
                .pairing
                .narrow
                .resident_bytes()
                .map_err(ContactHistoryError::from)?;
            search = search.max(add(index, narrow)?);
        }
        let configuration = add(
            slots::<[f64; 3]>(self.projection.nodes.len())?,
            slots::<ContactConfigurationPoint>(self.point_count)?,
        )?;
        let values = crate::algebra::storage::shared_vector_storage_bytes::<f64>(
            self.pattern.column_indices().len(),
        )
        .ok_or(ContactOperatorError::Budget)?;
        let contributions = crate::algebra::storage::shared_vector_storage_bytes::<
            crate::algebra::ContributionProvenance,
        >(0)
        .ok_or(ContactOperatorError::Budget)?;
        let numeric = add(
            slots::<f64>(self.pattern.rows())?,
            add(values, contributions)?,
        )?;
        let trial = add(configuration, add(numeric, pattern)?)?;
        Ok(ContactOperatorResources {
            prepared_resident_bytes: prepared,
            trial_resident_bytes: trial,
            trial_peak_additional_bytes: add(trial, search)?,
            identity_construction_bytes: self.identity_construction_bytes,
            construction_peak_additional_bytes: add(
                prepared,
                add(pattern, add(self.construction_scratch_bytes, scopes)?)?,
            )?,
        })
    }
}
