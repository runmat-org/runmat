//! One validated volume recovery for original accepted history, fresh or resumed.
use crate::{
    assembly::nonlinear_solid::{NonlinearSolidEvaluation, NonlinearSolidState},
    FeaRunError,
};

pub(super) struct RecoveredSolidFields {
    pub stress: Vec<f64>,
    pub total_strain: Vec<f64>,
    pub plastic: Vec<f64>,
    pub alpha: Vec<f64>,
    pub von_mises: Vec<f64>,
}

pub(super) fn recover(
    evaluation: &NonlinearSolidEvaluation,
    state: &NonlinearSolidState,
) -> Result<RecoveredSolidFields, FeaRunError> {
    let mut stress = Vec::new();
    let mut total_strain = Vec::new();
    let mut plastic = Vec::new();
    let mut alpha = Vec::new();
    let mut von_mises = Vec::new();
    if evaluation.point_evidence.len() != state.points.len()
        || state.points.len() != state.point_identities.len()
    {
        return Err(FeaRunError::Solve(
            "accepted J2 point recovery inventory differs".into(),
        ));
    }
    let mut owners = std::collections::BTreeMap::new();
    let mut element_ids = std::collections::BTreeSet::new();
    for (index, element) in evaluation.element_evidence.iter().enumerate() {
        if !element.volume_m3.is_finite()
            || element.volume_m3 <= 0.0
            || owners.insert(element.element_identity, index).is_some()
            || !element_ids.insert(element.element_id)
        {
            return Err(FeaRunError::Solve(
                "accepted J2 element volume/identity is invalid".into(),
            ));
        }
    }
    let mut means = vec![([0.0; 6], 0.0, 0.0, 0.0, 0usize); owners.len()];
    let mut seen = std::collections::BTreeSet::new();
    for ((point, history), identity) in evaluation
        .point_evidence
        .iter()
        .zip(&state.points)
        .zip(&state.point_identities)
    {
        if point.identity != *identity
            || !seen.insert((identity.element_identity, identity.quadrature_ordinal))
            || !point.volume_weight_m3.is_finite()
            || point.volume_weight_m3 <= 0.0
            || point
                .position_m
                .iter()
                .chain(&point.total_engineering_strain)
                .chain(&point.stress_pa)
                .any(|value| !value.is_finite())
            || [
                point.elastic_energy_density_pa,
                point.hardening_energy_density_pa,
                point.plastic_dissipation_increment_pa,
            ]
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(FeaRunError::Solve(
                "accepted J2 point order/weight/identity differs".into(),
            ));
        }
        let index = *owners
            .get(&identity.element_identity)
            .ok_or_else(|| FeaRunError::Solve("accepted J2 point has no element owner".into()))?;
        let weight = point.volume_weight_m3 / evaluation.element_evidence[index].volume_m3;
        let s = point.stress_pa;
        let vm = (0.5_f64).sqrt() * (s[0] - s[1]).hypot(s[1] - s[2]).hypot(s[2] - s[0]);
        let vm = vm
            .hypot(3.0_f64.sqrt() * s[3])
            .hypot(3.0_f64.sqrt() * s[4])
            .hypot(3.0_f64.sqrt() * s[5]);
        let (plastic, alpha, mises, volume, count) = &mut means[index];
        *mises += weight * vm;
        *alpha += weight * history.equivalent_plastic_strain;
        *volume += point.volume_weight_m3;
        *count += 1;
        for (mean, value) in plastic.iter_mut().zip(history.engineering_plastic_strain) {
            *mean += weight * value;
        }
    }
    for (element, (mean_plastic, mean_alpha, mean_vm, volume, count)) in
        evaluation.element_evidence.iter().zip(means)
    {
        if count == 0
            || (volume - element.volume_m3).abs() > 128.0 * f64::EPSILON * element.volume_m3
            || mean_plastic
                .iter()
                .chain([mean_alpha, mean_vm].iter())
                .chain(element.mean_stress_pa.iter())
                .chain(element.mean_total_engineering_strain.iter())
                .any(|v| !v.is_finite())
        {
            return Err(FeaRunError::Solve(
                "accepted J2 recovery is nonfinite or incompletely covered".into(),
            ));
        }
        stress.extend(element.mean_stress_pa);
        total_strain.extend(element.mean_total_engineering_strain);
        plastic.extend(mean_plastic);
        alpha.push(mean_alpha);
        von_mises.push(mean_vm);
    }
    Ok(RecoveredSolidFields {
        stress,
        total_strain,
        plastic,
        alpha,
        von_mises,
    })
}
