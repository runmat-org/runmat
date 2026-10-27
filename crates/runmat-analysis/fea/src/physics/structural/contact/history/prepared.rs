//! Compile portable scope once; live bindings borrow the same immutable source authorities.
//! No pointer address is encoded into a physical scope or durable state.
use super::*;
const SCOPE_LIMITS: runmat_canonical_codec::CanonicalLimits =
    runmat_canonical_codec::CanonicalLimits::new(16384, 64, 512, 24);
#[derive(Clone, Copy)]
pub struct ContactPreparedPoint<'a> {
    projection: &'a AnalysisProjection,
    interface: &'a AnalysisInterface,
    point_index: usize,
    law: PenaltyContactLaw,
    policy: BoundaryProjectionDifferentialPolicy,
    scope: Digest,
    construction_bytes: u64,
}
impl<'a> ContactPreparedPoint<'a> {
    pub fn derive(
        inventory: &ContactReferenceInventory<'a>,
        point_index: usize,
        law: PenaltyContactLaw,
        policy: BoundaryProjectionDifferentialPolicy,
    ) -> Result<Self, ContactHistoryError> {
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled.into());
        }
        let AnalysisInterfaceKind::Contact(model) = &inventory.interface.kind else {
            return Err(ContactHistoryError::InvalidScope);
        };
        if PenaltyContactLaw::derive(model, law.policy())?.identity() != law.identity() {
            return Err(ContactHistoryError::InvalidScope);
        }
        policy.validate()?;
        let projection = inventory.projection;
        let entry = inventory
            .points
            .get(point_index)
            .ok_or(ContactHistoryError::InvalidScope)?;
        let secondary_face = entry.face;
        let point_ordinal = entry.ordinal;
        let reference = entry.point;
        let source = scope_source(
            projection.projection_digest,
            inventory.interface,
            law.identity(),
            projection.boundary_faces[secondary_face].stable_identity,
            point_ordinal,
            reference,
            policy,
        );
        let limits = SCOPE_LIMITS;
        let profile = runmat_canonical_codec::profile_serialized(&source, limits)
            .map_err(|_| ContactHistoryError::InvalidScope)?;
        let construction_bytes = profile
            .materialization_resident_bytes_estimate
            .checked_add(profile.encoded_bytes_upper_bound)
            .ok_or(ContactHistoryError::InvalidScope)?;
        let bytes = runmat_canonical_codec::encode_contract(
            b"runmat-contact-material-point/v1\0",
            "contact-material-point",
            &source,
            limits,
        )
        .map_err(|_| ContactHistoryError::InvalidScope)?;
        let scope = Digest::sha256(&bytes);
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled.into());
        }
        Ok(Self {
            projection,
            interface: inventory.interface,
            point_index,
            law,
            policy,
            scope,
            construction_bytes,
        })
    }

    pub fn bind<'b>(
        &self,
        inventory: &'b ContactReferenceInventory<'a>,
    ) -> Result<ContactPointBinding<'b>, ContactHistoryError> {
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled.into());
        }
        if !std::ptr::eq(self.projection, inventory.projection)
            || !std::ptr::eq(self.interface, inventory.interface)
        {
            return Err(ContactHistoryError::InvalidScope);
        }
        let entry = inventory
            .points
            .get(self.point_index)
            .ok_or(ContactHistoryError::InvalidScope)?;
        Ok(ContactPointBinding {
            projection: inventory.projection,
            interface: inventory.interface,
            primary_faces: &inventory.primary_faces,
            secondary_face: entry.face,
            point_ordinal: entry.ordinal,
            reference: entry.point,
            law: self.law,
            policy: self.policy,
            scope: self.scope,
        })
    }
    pub fn construction_resident_bytes(&self) -> u64 {
        self.construction_bytes
    }
    pub(crate) fn normal_pressure_pa(
        &self,
        original: &ContactPointState,
    ) -> Result<runmat_types::physical::Pressure, ContactHistoryError> {
        self.law.validate_state(original)?;
        Ok(self.law.normal_pressure_at_gap(original.gap_m())?)
    }
    pub fn scope_identity(&self) -> Digest {
        self.scope
    }
}

// The shared wire shape is profiled before constructing point scopes. Digest and
// integer values do not affect the canonical profiler's scalar inventory.
pub(crate) fn scope_construction_bytes(
    projection: Digest,
    interface: &AnalysisInterface,
    law: Digest,
    face: StableDigest,
    ordinal: usize,
    reference: BoundaryQuadraturePoint,
    policy: BoundaryProjectionDifferentialPolicy,
) -> Result<u64, ContactHistoryError> {
    let source = scope_source(projection, interface, law, face, ordinal, reference, policy);
    let profile = runmat_canonical_codec::profile_serialized(&source, SCOPE_LIMITS)
        .map_err(|_| ContactHistoryError::InvalidScope)?;
    profile
        .materialization_resident_bytes_estimate
        .checked_add(profile.encoded_bytes_upper_bound)
        .ok_or(ContactHistoryError::InvalidScope)
}
fn scope_source<'a>(
    projection: Digest,
    interface: &'a AnalysisInterface,
    law: Digest,
    face: StableDigest,
    ordinal: usize,
    reference: BoundaryQuadraturePoint,
    policy: BoundaryProjectionDifferentialPolicy,
) -> impl Serialize + 'a {
    (
        1u16,
        projection,
        interface,
        law,
        face,
        ordinal,
        reference.reference_barycentric().map(f64::to_bits),
        reference.reference_weight().to_bits(),
        reference.area_weight_m2.to_bits(),
        [
            policy.stationarity_tolerance_m.to_bits(),
            policy.minimum_barycentric.to_bits(),
            policy.minimum_metric_ratio.to_bits(),
        ],
    )
}
