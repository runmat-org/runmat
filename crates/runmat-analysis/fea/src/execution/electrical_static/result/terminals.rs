//! Exact terminal ownership and weak nodal flux accounting; never field extrapolation.
use super::*;
use runmat_analysis_core::{AnalysisConstraint, ElectricalConstraintKind};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn validate(r: &CanonicalElectricalResponse) -> AnalysisExecutionResult<()> {
    let nodal = r
        .ordered_node_ids
        .iter()
        .copied()
        .zip(r.nodal_outward_flux.canonical_values().iter().copied())
        .collect::<BTreeMap<_, _>>();
    let mut ids = BTreeSet::new();
    let mut owned = BTreeSet::new();
    if r.terminals.is_empty() {
        return Err(invalid("electrical result requires terminal ownership"));
    }
    for terminal in &r.terminals {
        if terminal.constraint_id.trim().is_empty()
            || !ids.insert(&terminal.constraint_id)
            || !unique(&terminal.ordered_node_ids)
            || !terminal.outward_flux.canonical_value().is_finite()
            || terminal
                .ordered_node_ids
                .iter()
                .any(|n| !nodal.contains_key(n) || !owned.insert(*n))
        {
            return Err(invalid("electrical terminal ownership is invalid"));
        }
        let sum: f64 = terminal.ordered_node_ids.iter().map(|n| nodal[n]).sum();
        let scale: f64 = terminal
            .ordered_node_ids
            .iter()
            .map(|n| nodal[n].abs())
            .sum();
        if !sum.is_finite()
            || !scale.is_finite()
            || (sum - terminal.outward_flux.canonical_value()).abs()
                > 64.0 * f64::EPSILON * scale.max(f64::MIN_POSITIVE)
        {
            return Err(invalid(
                "electrical terminal flux disagrees with retained nodal flux",
            ));
        }
    }
    let sum: f64 = r
        .terminals
        .iter()
        .map(|t| t.outward_flux.canonical_value())
        .sum();
    let scale: f64 = r
        .terminals
        .iter()
        .map(|t| t.outward_flux.canonical_value().abs())
        .sum();
    if !sum.is_finite() || !scale.is_finite() {
        return Err(invalid("nonfinite electrical terminal accounting"));
    }
    let charge = r
        .charge_sources
        .as_ref()
        .map_or(0.0, |source| source.integrated_source);
    let source_scale = r
        .charge_sources
        .as_ref()
        .map_or(0.0, |source| source.absolute_source);
    let imbalance = (sum - charge).abs() / scale.max(source_scale).max(f64::MIN_POSITIVE);
    if !imbalance.is_finite() || imbalance != r.relative_flux_imbalance {
        return Err(invalid(
            "electrical flux closure disagrees with terminal accounting",
        ));
    }
    Ok(())
}

pub(super) fn validate_for(
    r: &CanonicalElectricalResponse,
    workload: &ElectricalStaticWorkload,
    mesh: &SolverMeshArtifact,
    projection: &AnalysisProjection,
) -> AnalysisExecutionResult<()> {
    if r.terminals.len() != workload.model.constraints.len() {
        return Err(invalid(
            "electrical result terminal inventory differs from source",
        ));
    }
    let nodes = projection
        .nodes
        .iter()
        .map(|n| (n.stable_identity, n.node_id))
        .collect::<BTreeMap<_, _>>();
    let potentials = r
        .ordered_node_ids
        .iter()
        .copied()
        .zip(r.potential_v.canonical_values().iter().copied())
        .collect::<BTreeMap<_, _>>();
    let mut prescribed = Vec::new();
    for (terminal, constraint) in r.terminals.iter().zip(&workload.model.constraints) {
        let AnalysisConstraint::Electrical(constraint) = constraint else {
            return Err(invalid("non-electrical constraint in static result source"));
        };
        let (target, value) = match &constraint.kind {
            ElectricalConstraintKind::PrescribedPotential { target, value } => {
                let value = crate::excitation::sample_prescribed_potential(
                    value,
                    &workload.model.signals,
                    workload.sample_point(),
                )
                .map_err(invalid)?;
                if !value.real.is_finite() || value.imaginary != 0.0 {
                    return Err(invalid("invalid electrical prescribed potential source"));
                }
                (target, value.real)
            }
            ElectricalConstraintKind::PotentialReference { target } => (target, 0.0),
            _ => return Err(invalid("unsupported electrical terminal source")),
        };
        let resolved =
            crate::constraints::resolve_nodal_target(mesh, target, &constraint.constraint_id)
                .map_err(invalid)?;
        if terminal.constraint_id != constraint.constraint_id
            || !terminal
                .ordered_node_ids
                .iter()
                .copied()
                .eq(resolved.matched_node_identities.iter().map(|id| nodes[id]))
        {
            return Err(invalid(
                "electrical terminal does not match resolved canonical source",
            ));
        }
        prescribed.extend(terminal.ordered_node_ids.iter().map(|id| (*id, value)));
    }
    // Shared PCG starts constrained identity rows at their prescribed increment,
    // so their search directions remain zero. Allow only gauge-add roundoff,
    // not a solver tolerance that grows with a physically irrelevant gauge offset.
    let magnitude = prescribed
        .iter()
        .map(|(_, value)| value.abs())
        .fold(0.0, f64::max);
    let bound = magnitude * (128.0 * f64::EPSILON);
    if !bound.is_finite()
        || prescribed
            .iter()
            .any(|(id, value)| (potentials[id] - value).abs() > bound)
    {
        return Err(invalid(
            "retained electrical potential violates prescribed source",
        ));
    }
    let prescribed_values = prescribed.into_iter().collect::<BTreeMap<_, _>>();
    for component in
        crate::algebra::connected_components(&projection.node_neighbors).map_err(invalid)?
    {
        let reference = component
            .iter()
            .find_map(|&node| prescribed_values.get(&projection.nodes[node].node_id))
            .ok_or_else(|| invalid("unanchored electrical component"))?;
        for node in component {
            let expected = r.potential_increment_v.canonical_values()[node] + reference;
            if expected != r.potential_v.canonical_values()[node] {
                return Err(invalid(
                    "retained potential disagrees with gauge departure and prescribed reference",
                ));
            }
        }
    }
    Ok(())
}
