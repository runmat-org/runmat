//! Source-counted preparation demand. Derivation admits curved-rule scratch first;
//! construction admits all payload before identities, inventories and graphs.
//! Caller sources, allocator metadata, execution/state/transport and RSS are separate.
use super::*;
use std::mem::size_of;
#[derive(Clone, Copy, Debug)]
pub struct ContactPreparationDemand {
    pub counting_peak_bytes: u64,
    pub construction_peak_bytes: u64,
    pub scalar_operations_upper_bound: u64,
    pub reference_points: usize,
    pub face_pairs: usize,
    pub coordinates: usize,
}
impl ContactPreparationDemand {
    pub fn derive(
        projection: &AnalysisProjection,
        interfaces: &[AnalysisInterface],
        node_rows: &[[usize; 3]],
        dof_count: usize,
        controls: &[ContactInterfacePolicy],
        policy: ContactOperatorPolicy,
    ) -> Result<Self, ContactOperatorError> {
        cancel()?;
        if policy.maximum_preparation_resident_bytes == 0
            || policy.maximum_preparation_scalar_operations == 0
        {
            return Err(ContactOperatorError::Budget);
        }
        if interfaces.is_empty()
            || interfaces.len() != controls.len()
            || node_rows.len() != projection.nodes.len()
            || dof_count == 0
        {
            return Err(ContactOperatorError::Invalid);
        }
        if dof_count > policy.maximum_dofs {
            return Err(ContactOperatorError::Budget);
        }
        let offsets = add(dof_count as u64, 1)?;
        let header = add(dof_count as u64, mul(offsets, 8)?)?;
        if header > policy.maximum_preparation_resident_bytes {
            return Err(ContactOperatorError::Budget);
        }
        let limits = policy.identity_limits();
        let physical = profile(
            &(
                1u16,
                projection.projection_digest,
                interfaces,
                node_rows,
                controls,
            ),
            limits,
        )?;
        // Digest has fixed wire/profile shape; no physical identity is synthesized here.
        let execution = profile(&(1u16, projection.projection_digest, policy), limits)?;
        let mut scopes = physical.max(execution);
        // Charge original source scans as well as selected support: unrelated
        // faces still participate in every target-resolution pass.
        let provenance = projection
            .boundary_faces
            .iter()
            .try_fold(0u64, |sum, face| add(sum, face.provenance.len() as u64))?;
        let target_probes = interfaces.iter().try_fold(0u64, |sum, interface| {
            add(
                sum,
                add(
                    interface.primary.resolved_entities().len().max(1) as u64,
                    interface.secondary.resolved_entities().len().max(1) as u64,
                )?,
            )
        })?;
        let selection_work = add(
            mul(mul(provenance, target_probes)?, 32)?,
            mul(
                mul(
                    projection.boundary_faces.len() as u64,
                    interfaces.len() as u64,
                )?,
                32,
            )?,
        )?;
        let mut primary_total = 0u64;
        let mut selected_total = 0u64;
        let mut quad = 0u64;
        let mut points = 0usize;
        let mut pairs = 0usize;
        let mut coordinates = 0usize;
        let mut refinement = 0u64;
        for (interface, control) in interfaces.iter().zip(controls) {
            cancel()?;
            let AnalysisInterfaceKind::Contact(model) = &interface.kind else {
                return Err(ContactOperatorError::Invalid);
            };
            control.pairing.validate()?;
            control
                .stationary
                .validate()
                .map_err(ContactHistoryError::from)?;
            let [primary, secondary] = super::super::surfaces::counts(projection, interface)?;
            if primary as u64 > control.pairing.maximum_faces {
                return Err(ContactOperatorError::Budget);
            }
            pairs = pairs
                .checked_add(
                    primary
                        .checked_mul(secondary)
                        .ok_or(ContactOperatorError::Budget)?,
                )
                .ok_or(ContactOperatorError::Budget)?;
            if pairs > policy.maximum_face_pairs {
                return Err(ContactOperatorError::Budget);
            }
            primary_total = add(primary_total, primary as u64)?;
            selected_total = add(selected_total, add(primary as u64, secondary as u64)?)?;
            let mut node_sums = [0u64; 2];
            let mut square_sums = [0u64; 2];
            let mut local_points = 0u64;
            scopes = scopes.max(
                PenaltyContactLaw::construction_resident_bytes(model, control.penalty)
                    .map_err(ContactHistoryError::from)?,
            );
            for face in &projection.boundary_faces {
                cancel()?;
                for (side, target) in [&interface.primary, &interface.secondary]
                    .iter()
                    .enumerate()
                {
                    if !super::super::surfaces::matches(face, target) {
                        continue;
                    }
                    if face.node_indices.iter().any(|&n| n >= node_rows.len()) {
                        return Err(ContactOperatorError::Invalid);
                    }
                    let nodes = face.node_indices.len() as u64;
                    node_sums[side] = add(node_sums[side], nodes)?;
                    square_sums[side] = add(square_sums[side], mul(nodes, nodes)?)?;
                }
                if !super::super::surfaces::matches(face, &interface.secondary) {
                    continue;
                }
                let scratch =
                    AnalysisProjection::boundary_quadrature_scratch_resident_bytes(face.order)
                        .map_err(|_| ContactOperatorError::Invalid)?;
                if scratch > policy.maximum_preparation_resident_bytes {
                    return Err(ContactOperatorError::Budget);
                }
                quad = quad.max(scratch);
                let mut error = None;
                projection
                    .visit_boundary_quadrature(face, |point| {
                        if error.is_some() {
                            return;
                        }
                        if crate::is_fea_cancelled() {
                            error = Some(ContactOperatorError::Cancelled);
                            return;
                        }
                        if points == policy.maximum_reference_points {
                            error = Some(ContactOperatorError::Budget);
                            return;
                        }
                        match super::super::history::scope_construction_bytes(
                            projection.projection_digest,
                            interface,
                            projection.projection_digest,
                            face.stable_identity,
                            local_points as usize,
                            point,
                            control.stationary.differential,
                        ) {
                            Ok(bytes) => scopes = scopes.max(bytes),
                            Err(e) => {
                                error = Some(e.into());
                                return;
                            }
                        }
                        points += 1;
                        local_points += 1;
                    })
                    .map_err(|_| ContactOperatorError::Invalid)?;
                if let Some(error) = error {
                    return Err(error);
                }
            }
            let coordinate_count = mul(
                9,
                add(
                    add(
                        mul(secondary as u64, square_sums[0])?,
                        mul(primary as u64, square_sums[1])?,
                    )?,
                    mul(2, mul(node_sums[0], node_sums[1])?)?,
                )?,
            )?;
            coordinates = coordinates
                .checked_add(
                    usize::try_from(coordinate_count).map_err(|_| ContactOperatorError::Budget)?,
                )
                .ok_or(ContactOperatorError::Budget)?;
            if coordinates > policy.maximum_coordinates {
                return Err(ContactOperatorError::Budget);
            }
            let per_point = add(
                mul(
                    control.stationary.maximum_iterations as u64,
                    add(control.stationary.maximum_backtracks as u64, 1)?,
                )?,
                1,
            )?;
            refinement = add(refinement, mul(local_points, per_point)?)?;
            if refinement > policy.maximum_refinement_evaluations_per_trial {
                return Err(ContactOperatorError::Budget);
            }
        }
        if points == 0 {
            return Err(ContactOperatorError::Invalid);
        }
        let inventory = ContactReferenceInventory::resident_bytes_for_counts(
            usize::try_from(primary_total).map_err(|_| ContactOperatorError::Budget)?,
            points,
        )?;
        let prepared = add(
            inventory,
            add(
                mul(points as u64, size_of::<ContactPreparedPoint<'_>>() as u64)?,
                mul(
                    interfaces.len() as u64,
                    size_of::<PreparedInterface<'_>>() as u64,
                )?,
            )?,
        )?;
        // At most three CSR inventories coexist during union, beside one original
        // coordinate buffer. Duplicate support reduces actual retained columns.
        let csr = SparsePattern::retained_storage_bytes_for_counts(dof_count, coordinates)?;
        let construction = add(
            prepared,
            add(
                mul(csr, 3)?,
                add(
                    mul(coordinates as u64, size_of::<(usize, usize)>() as u64)?,
                    add(
                        mul(selected_total, size_of::<usize>() as u64)?,
                        add(quad, add(scopes, dof_count as u64)?)?,
                    )?,
                )?,
            )?,
        )?;
        let log = usize::BITS - coordinates.max(1).leading_zeros();
        let work = add(
            selection_work,
            add(
                mul(coordinates as u64, 16 + 8 * u64::from(log))?,
                add(
                    mul(points as u64, add(scopes, quad)?)?,
                    add(
                        mul(selected_total, add(primary_total, points as u64)?)?,
                        mul(physical, 16)?,
                    )?,
                )?,
            )?,
        )?;
        let result = Self {
            counting_peak_bytes: quad,
            construction_peak_bytes: construction,
            scalar_operations_upper_bound: work,
            reference_points: points,
            face_pairs: pairs,
            coordinates,
        };
        cancel()?;
        result.admit(policy)?;
        Ok(result)
    }
    pub fn admit(self, policy: ContactOperatorPolicy) -> Result<(), ContactOperatorError> {
        if self.construction_peak_bytes > policy.maximum_preparation_resident_bytes
            || self.scalar_operations_upper_bound > policy.maximum_preparation_scalar_operations
        {
            Err(ContactOperatorError::Budget)
        } else {
            Ok(())
        }
    }
}
fn profile<T: Serialize>(
    source: &T,
    limits: runmat_canonical_codec::CanonicalLimits,
) -> Result<u64, ContactOperatorError> {
    let p = runmat_canonical_codec::profile_serialized_shape(source, limits)
        .map_err(|_| ContactOperatorError::Budget)?;
    add(
        p.materialization_resident_bytes_estimate,
        p.encoded_bytes_upper_bound,
    )
}
fn add(a: u64, b: u64) -> Result<u64, ContactOperatorError> {
    a.checked_add(b).ok_or(ContactOperatorError::Budget)
}
fn mul(a: u64, b: u64) -> Result<u64, ContactOperatorError> {
    a.checked_mul(b).ok_or(ContactOperatorError::Budget)
}
fn cancel() -> Result<(), ContactOperatorError> {
    if crate::is_fea_cancelled() {
        Err(ContactOperatorError::Cancelled)
    } else {
        Ok(())
    }
}
