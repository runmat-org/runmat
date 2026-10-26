use super::*;
use crate::algebra::{MatrixProperties, OperatorRole, RealSparseAssembler};
impl CanonicalContactOperator<'_> {
    fn pairing_policy(
        &self,
        interface: &PreparedInterface<'_>,
        work: &ContactAssemblyDiagnostics,
    ) -> Result<BoundaryPairingPolicy, ContactOperatorError> {
        let remaining = self
            .policy
            .maximum_patches_per_trial
            .checked_sub(work.visited_patches)
            .filter(|n| *n > 0)
            .ok_or(ContactOperatorError::Budget)?;
        Ok(BoundaryPairingPolicy {
            maximum_total_patches: interface
                .policy
                .pairing
                .maximum_total_patches
                .min(remaining),
            ..interface.policy.pairing
        })
    }
    fn pair(
        &self,
        interface: &PreparedInterface<'_>,
        binding: &ContactPointBinding<'_>,
        index: &BoundarySurfaceIndex,
        positions: &[[f64; 3]],
        work: &mut ContactAssemblyDiagnostics,
    ) -> Result<crate::topology::BoundarySurfaceMatch, ContactOperatorError> {
        let (matched, refined) = index.closest_point_stationary(
            binding.current_query_position_m(positions)?,
            self.pairing_policy(interface, work)?,
            interface.policy.stationary,
        )?;
        work.visited_patches = work
            .visited_patches
            .checked_add(matched.visited_patches)
            .ok_or(ContactOperatorError::Budget)?;
        work.tested_faces = work
            .tested_faces
            .checked_add(matched.tested_faces)
            .ok_or(ContactOperatorError::Budget)?;
        let evaluations = u64::from(refined.refinement_steps)
            .checked_mul(2)
            .and_then(|n| n.checked_add(refined.backtracks))
            .and_then(|n| n.checked_add(1))
            .ok_or(ContactOperatorError::Budget)?;
        work.refinement_evaluations_bound = work
            .refinement_evaluations_bound
            .checked_add(evaluations)
            .ok_or(ContactOperatorError::Budget)?;
        if work.refinement_evaluations_bound > self.policy.maximum_refinement_evaluations_per_trial
        {
            return Err(ContactOperatorError::Budget);
        }
        Ok(matched)
    }
    pub fn initial_configuration(
        &self,
        positions: &[[f64; 3]],
    ) -> Result<ContactConfigurationState, ContactOperatorError> {
        self.check_positions(positions)?;
        let mut points = reserve(self.point_count)?;
        let mut work = ContactAssemblyDiagnostics::default();
        for interface in &self.interfaces {
            let index = self.index(interface, positions)?;
            for prepared in &interface.points {
                let binding = prepared.bind(&interface.inventory)?;
                let matched = self.pair(interface, &binding, &index, positions, &mut work)?;
                if matched.closest.signed_normal_offset_m
                    < -interface.policy.maximum_initial_penetration_m
                {
                    return Err(ContactOperatorError::InitialPenetration);
                }
                let history = binding.initial_history(
                    matched.face_index,
                    matched.closest.barycentric,
                    positions,
                )?;
                points.push(binding.configuration_record(&history, 0, positions)?);
            }
        }
        let mut original = reserve(positions.len())?;
        original.extend_from_slice(positions);
        if crate::is_fea_cancelled() {
            return Err(ContactOperatorError::Cancelled);
        }
        Ok(ContactConfigurationState {
            schema: 1,
            scope: self.identity,
            ordinal: 0,
            positions_m: original,
            points,
        })
    }
    /// Integrate the original point-owned physical friction work with the original
    /// reference measure. No averaged field or current surface measure is substituted.
    pub fn frictional_work_j(
        &self,
        state: &ContactConfigurationState,
    ) -> Result<f64, ContactOperatorError> {
        self.validate_configuration(state, state.ordinal())?;
        let mut total = 0.;
        let mut ordinal = 0;
        for interface in &self.interfaces {
            for prepared in &interface.points {
                let binding = prepared.bind(&interface.inventory)?;
                total += binding.reference_point().area_weight_m2
                    * state.point_history()[ordinal]
                        .point()
                        .frictional_work_j_per_m2()
                        .canonical_value();
                if !total.is_finite() || total < 0. {
                    return Err(ContactOperatorError::Invalid);
                }
                ordinal += 1;
            }
        }
        Ok(total)
    }
    /// Immutable trial against the complete original configuration. All errors discard output.
    /// Body equilibrium, contact accuracy and global admission remain the procedure's authority.
    pub fn assemble_trial(
        &self,
        committed: &ContactConfigurationState,
        expected_ordinal: u64,
        positions: &[[f64; 3]],
    ) -> Result<ContactAssemblyTrial, ContactOperatorError> {
        self.check_state_header(committed, expected_ordinal)?;
        self.check_positions(positions)?;
        let next = expected_ordinal
            .checked_add(1)
            .ok_or(ContactHistoryError::OrdinalOverflow)?;
        let mut tangent = RealSparseAssembler::with_pattern(
            self.pattern.try_clone()?,
            OperatorRole::EffectiveTangent,
            MatrixProperties::GENERAL,
        )?;
        let mut residual = reserve(self.pattern.rows())?;
        residual.resize(self.pattern.rows(), 0.0);
        let mut points = reserve(self.point_count)?;
        let mut work = ContactAssemblyDiagnostics::default();
        let mut ordinal = 0;
        for interface in &self.interfaces {
            let index = self.index(interface, positions)?;
            for prepared in &interface.points {
                let binding = prepared.bind(&interface.inventory)?;
                let accepted = binding.material_history(
                    &committed.points[ordinal],
                    expected_ordinal,
                    &committed.positions_m,
                )?;
                let matched = self.pair(interface, &binding, &index, positions, &mut work)?;
                if matched.face_identity != accepted.primary_face_identity() {
                    return Err(ContactOperatorError::FeatureTransition);
                }
                let trial = binding.trial(
                    &accepted,
                    expected_ordinal,
                    &committed.positions_m,
                    positions,
                    matched.closest.barycentric,
                )?;
                let mut primary = [[0usize; 3]; 6];
                let mut secondary = [[0usize; 3]; 6];
                let a = binding.primary_node_indices(matched.face_identity)?;
                let b = binding.secondary_node_indices();
                for (rows, &node) in primary.iter_mut().zip(a) {
                    *rows = self.node_rows[node];
                }
                for (rows, &node) in secondary.iter_mut().zip(b) {
                    *rows = self.node_rows[node];
                }
                trial.assemble_into(
                    &primary[..a.len()],
                    &secondary[..b.len()],
                    &mut residual,
                    &mut tangent,
                )?;
                work.record(
                    trial.point_response(),
                    binding.reference_point().area_weight_m2,
                )?;
                points.push(binding.configuration_record(trial.history(), next, positions)?);
                ordinal += 1;
            }
        }
        let tangent = tangent.finish()?;
        let mut original = reserve(positions.len())?;
        original.extend_from_slice(positions);
        if crate::is_fea_cancelled() {
            return Err(ContactOperatorError::Cancelled);
        }
        Ok(ContactAssemblyTrial {
            residual,
            tangent,
            configuration: ContactConfigurationState {
                schema: 1,
                scope: self.identity,
                ordinal: next,
                positions_m: original,
                points,
            },
            diagnostics: work,
        })
    }
}
impl ContactAssemblyDiagnostics {
    fn record(
        &mut self,
        response: &ContactPointResponse,
        weight: f64,
    ) -> Result<(), ContactOperatorError> {
        match response.mode {
            ContactPointMode::Open => self.open_points += 1,
            ContactPointMode::FrictionlessActive => self.active_points += 1,
            ContactPointMode::Stick => {
                self.active_points += 1;
                self.stick_points += 1;
            }
            ContactPointMode::Slip => {
                self.active_points += 1;
                self.slip_points += 1;
            }
        }
        self.maximum_penetration_m = self
            .maximum_penetration_m
            .max((-response.trial_state.gap_m().canonical_value()).max(0.0));
        for (total, value) in [
            (
                &mut self.normal_elastic_energy_j,
                response.normal_elastic_energy_j_per_m2,
            ),
            (
                &mut self.tangential_elastic_energy_j,
                response.tangential_elastic_energy_j_per_m2,
            ),
            (
                &mut self.frictional_dissipation_increment_j,
                response.frictional_dissipation_increment_j_per_m2,
            ),
            (
                &mut self.algorithmic_dissipation_increment_j,
                response.tangential_algorithmic_dissipation_increment_j_per_m2,
            ),
            (
                &mut self.separation_release_energy_j,
                response.separation_release_energy_j_per_m2,
            ),
        ] {
            *total += weight * value.canonical_value();
            if !total.is_finite() || *total < 0.0 {
                return Err(ContactOperatorError::Invalid);
            }
        }
        Ok(())
    }
}
