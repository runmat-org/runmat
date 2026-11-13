//! Material-owned irreversible work and physically scaled accepted-history replay.
use super::*;

impl CanonicalNonlinearSolidBasis {
    /// For this closed linear-isotropic-hardening law the irreversible accumulated work
    /// is sigma_y0*alpha, distinct from the stored H*alpha²/2 hardening energy.
    pub(crate) fn accumulated_plastic_dissipation_j(
        &self,
        state: &NonlinearSolidState,
    ) -> Result<f64, NonlinearSolidError> {
        self.validate_state(state)?;
        let mut ordinal = 0;
        let mut total = 0.0;
        for element in &self.elements {
            for point in &element.points {
                let density = element
                    .constitutive
                    .committed_plastic_dissipation_density_pa(&state.points[ordinal])
                    .map_err(|e| NonlinearSolidError::History(e.to_string()))?;
                total += point.weight_m3 * density;
                ordinal += 1;
            }
        }
        if !total.is_finite() || total < 0.0 {
            return Err(NonlinearSolidError::History(
                "invalid accumulated irreversible work".into(),
            ));
        }
        Ok(total)
    }

    /// Replaying the accepted total strain must not materially advance any point history.
    /// The kernel supplies a stress/strain/history-scaled floating-point bound; no fixed
    /// alpha-only floor can erase a real small-scale plastic increment.
    pub(crate) fn validate_replayed_history(
        &self,
        displacement: &[f64],
        state: &NonlinearSolidState,
    ) -> Result<(), NonlinearSolidError> {
        self.validate_state(state)?;
        if displacement.len() != self.dof_count || displacement.iter().any(|v| !v.is_finite()) {
            return Err(NonlinearSolidError::History(
                "invalid replay displacement layout".into(),
            ));
        }
        let mut ordinal = 0;
        for element in &self.elements {
            for point in &element.points {
                let strain =
                    evaluation::point_kinematics(self.kinematics, element, point, displacement)
                        .strain;
                let committed = &state.points[ordinal];
                let response = element
                    .constitutive
                    .update(strain, committed)
                    .map_err(|e| NonlinearSolidError::History(e.to_string()))?;
                let bound = element
                    .constitutive
                    .replay_increment_roundoff_bound(committed, strain)
                    .map_err(|e| NonlinearSolidError::History(e.to_string()))?;
                let alpha_increment = (response.state.equivalent_plastic_strain
                    - committed.equivalent_plastic_strain)
                    .abs();
                // sqrt(2/3 eps_p:eps_p), respecting engineering shear (twice tensor shear).
                let mut tensor_norm = 0.0_f64;
                for i in 0..6 {
                    let increment = response.state.engineering_plastic_strain[i]
                        - committed.engineering_plastic_strain[i];
                    tensor_norm = tensor_norm.hypot(if i < 3 {
                        increment
                    } else {
                        increment / std::f64::consts::SQRT_2
                    });
                }
                let equivalent_increment = tensor_norm * (2.0_f64 / 3.0).sqrt();
                if !bound.is_finite()
                    || bound < 0.0
                    || alpha_increment > bound
                    || equivalent_increment > bound
                {
                    return Err(NonlinearSolidError::History(format!("accepted point {}:{} advances on replay: alpha={alpha_increment}, tensor={equivalent_increment}, roundoff={bound}",element.element_id,point.identity.quadrature_ordinal)));
                }
                ordinal += 1;
            }
        }
        Ok(())
    }
}
