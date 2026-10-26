//! Genuine linear thermal producer supplies the manufactured continuum's initial state.
use super::*;
use crate::execution::{InitialFieldBinding, ThermalSolverResultManifest};
use crate::results::field::RecoveredField;

pub(super) fn solved_linear_profile(mesh: &SolverMeshArtifact) -> InitialFieldBinding {
    let input = crate::physics::thermal::qualification::initial::linear_profile();
    let result = input.execute(mesh).unwrap();
    for (node, &value) in mesh
        .topology
        .nodes
        .iter()
        .zip(result.temperature_snapshots[0].as_host_f64().unwrap())
    {
        assert!((value - (400.0 - 10.0 * node.coordinates_m.canonical_value()[0])).abs() < 1.0e-7);
    }
    let manifest = ThermalSolverResultManifest::from_result(&result).unwrap();
    let entry = manifest.terminal_temperature().unwrap();
    let field = RecoveredField::new(
        entry.definition.clone(),
        entry.sample,
        result.temperature_snapshots[0].clone(),
    )
    .unwrap();
    crate::physics::thermal::qualification::initial::temperature_binding(&manifest, field, mesh)
}
