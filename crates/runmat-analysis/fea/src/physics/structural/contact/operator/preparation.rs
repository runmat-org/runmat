use super::*;
impl<'a> CanonicalContactOperator<'a> {
    pub fn prepare(
        projection: &'a AnalysisProjection,
        interfaces: &'a [AnalysisInterface],
        node_rows: &'a [[usize; 3]],
        dof_count: usize,
        controls: &[ContactInterfacePolicy],
        policy: ContactOperatorPolicy,
    ) -> Result<Self, ContactOperatorError> {
        if crate::is_fea_cancelled() {
            return Err(ContactOperatorError::Cancelled);
        }
        if interfaces.is_empty()
            || controls.len() != interfaces.len()
            || node_rows.len() != projection.nodes.len()
            || dof_count == 0
            || policy.maximum_reference_points == 0
            || policy.maximum_patches_per_trial == 0
            || policy.maximum_refinement_evaluations_per_trial == 0
        {
            return Err(ContactOperatorError::Invalid);
        }
        if dof_count > policy.maximum_dofs {
            return Err(ContactOperatorError::Budget);
        }
        let demand = ContactPreparationDemand::derive(
            projection, interfaces, node_rows, dof_count, controls, policy,
        )?;
        demand.admit(policy)?;
        let mut seen = reserve(dof_count)?;
        seen.resize(dof_count, false);
        for &row in node_rows.iter().flatten() {
            if row >= dof_count || seen[row] {
                return Err(ContactOperatorError::Invalid);
            }
            seen[row] = true;
        }
        drop(seen);
        for (i, (interface, control)) in interfaces.iter().zip(controls).enumerate() {
            if interface.interface_id.is_empty()
                || interfaces[..i]
                    .iter()
                    .any(|p| p.interface_id == interface.interface_id)
            {
                return Err(ContactOperatorError::Invalid);
            }
            control.pairing.validate()?;
            control
                .stationary
                .validate()
                .map_err(ContactHistoryError::from)?;
            if !control.maximum_initial_penetration_m.is_finite()
                || control.maximum_initial_penetration_m < 0.0
            {
                return Err(ContactOperatorError::Invalid);
            }
        }
        let identity_source = (
            1u16,
            projection.projection_digest,
            interfaces,
            node_rows,
            controls,
        );
        let limits = policy.identity_limits();
        let profile = runmat_canonical_codec::profile_serialized(&identity_source, limits)
            .map_err(|_| ContactOperatorError::Budget)?;
        // Streamed: the support-sized identity encoding is never materialized.
        let identity = crate::contracts::canonical_digest::canonical_contract_digest(
            b"runmat-fea-contact-operator/v1\0",
            "contact-operator",
            &identity_source,
            limits,
        )
        .map_err(|_| ContactOperatorError::Budget)?;
        let identity_construction_bytes = profile
            .materialization_resident_bytes_estimate
            .checked_add(profile.encoded_bytes_upper_bound)
            .ok_or(ContactOperatorError::Budget)?;
        let execution_source = (1u16, identity, policy);
        let execution_profile =
            runmat_canonical_codec::profile_serialized(&execution_source, limits)
                .map_err(|_| ContactOperatorError::Budget)?;
        let encoded = runmat_canonical_codec::encode_contract(
            b"runmat-fea-contact-execution/v1\0",
            "contact-execution",
            &execution_source,
            limits,
        )
        .map_err(|_| ContactOperatorError::Budget)?;
        let execution_identity = Digest::sha256(&encoded);
        drop(encoded);
        let identity_construction_bytes = identity_construction_bytes.max(
            execution_profile
                .materialization_resident_bytes_estimate
                .checked_add(execution_profile.encoded_bytes_upper_bound)
                .ok_or(ContactOperatorError::Budget)?,
        );
        let mut construction_scratch_bytes = dof_count as u64;
        let mut prepared = reserve(interfaces.len())?;
        let mut pattern = SparsePattern::from_coordinates(dof_count, dof_count, Vec::new())?;
        let (mut point_count, mut pairs, mut coordinates) = (0usize, 0usize, 0usize);
        let mut refinement_bound = 0u64;
        for (interface, control) in interfaces.iter().zip(controls) {
            let AnalysisInterfaceKind::Contact(model) = &interface.kind else {
                return Err(ContactOperatorError::Invalid);
            };
            let law = PenaltyContactLaw::derive(model, control.penalty)
                .map_err(ContactHistoryError::from)?;
            let remaining_points = policy
                .maximum_reference_points
                .checked_sub(point_count)
                .ok_or(ContactOperatorError::Budget)?;
            let surfaces = CanonicalContactSurfaces::derive_bounded(
                projection,
                interface,
                usize::try_from(control.pairing.maximum_faces)
                    .map_err(|_| ContactOperatorError::Budget)?,
                remaining_points,
            )?;
            for previous in &prepared {
                let previous: &PreparedInterface<'_> = previous;
                if surfaces.primary_face_indices().iter().any(|face| {
                    previous
                        .inventory
                        .secondary_face_indices()
                        .any(|old| old == *face)
                }) || surfaces.secondary_face_indices().iter().any(|face| {
                    previous.inventory.primary_face_indices().contains(face)
                        || previous
                            .inventory
                            .secondary_face_indices()
                            .any(|old| old == *face)
                }) {
                    return Err(ContactSurfaceError::SurfaceConflict.into());
                }
            }
            let inventory = ContactReferenceInventory::derive(&surfaces, remaining_points)?;
            point_count += inventory.point_count();
            let per_point = u64::from(control.stationary.maximum_iterations)
                .checked_mul(u64::from(control.stationary.maximum_backtracks) + 1)
                .and_then(|n| n.checked_add(1))
                .ok_or(ContactOperatorError::Budget)?;
            refinement_bound = refinement_bound
                .checked_add(
                    per_point
                        .checked_mul(inventory.point_count() as u64)
                        .ok_or(ContactOperatorError::Budget)?,
                )
                .ok_or(ContactOperatorError::Budget)?;
            if refinement_bound > policy.maximum_refinement_evaluations_per_trial {
                return Err(ContactOperatorError::Budget);
            }

            let support = ContactTangentSupport::derive(
                &surfaces,
                node_rows,
                dof_count,
                ContactSupportBudget {
                    maximum_face_pairs: policy
                        .maximum_face_pairs
                        .checked_sub(pairs)
                        .ok_or(ContactOperatorError::Budget)?,
                    maximum_coordinates: policy
                        .maximum_coordinates
                        .checked_sub(coordinates)
                        .ok_or(ContactOperatorError::Budget)?,
                },
            )?;
            pairs += support.face_pair_count();
            coordinates += support.coordinate_count();
            let selection = CanonicalContactSurfaces::resident_bytes(projection, interface)?;
            let quadrature = inventory
                .construction_resident_bytes()?
                .checked_sub(inventory.resident_bytes()?)
                .ok_or(ContactOperatorError::Budget)?;
            let scratch = selection
                .checked_add(quadrature)
                .and_then(|n| n.checked_add(support.construction_resident_bytes().ok()?))
                .ok_or(ContactOperatorError::Budget)?;
            construction_scratch_bytes = construction_scratch_bytes.max(scratch);
            pattern = pattern.union(support.pattern())?;
            drop(support);
            drop(surfaces);
            let mut points = reserve(inventory.point_count())?;
            for index in 0..inventory.point_count() {
                points.push(ContactPreparedPoint::derive(
                    &inventory,
                    index,
                    law,
                    control.stationary.differential,
                )?);
            }
            prepared.push(PreparedInterface {
                inventory,
                points,
                policy: *control,
            });
        }
        if crate::is_fea_cancelled() {
            return Err(ContactOperatorError::Cancelled);
        }
        Ok(Self {
            projection,
            node_rows,
            authored_interfaces: interfaces,
            interfaces: prepared,
            pattern,
            identity,
            execution_identity,
            policy,
            point_count,
            identity_construction_bytes,
            construction_scratch_bytes,
            preparation_demand: demand,
        })
    }
}
