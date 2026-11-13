//! Thermal contact between two solids against the series-resistance closed form.
use runmat_analysis_core::*;
use runmat_meshing_core::{fixtures::layered_box_mesh, SolverMeshArtifact};

use crate::{
    execution::ThermalWorkload,
    fixtures::{fixture_model, FixtureId},
    ComputeBackend, ThermalProcedureControl,
};

const SPLIT: f64 = 0.4;
const K_LOWER: f64 = 40.0;
const K_UPPER: f64 = 10.0;
pub(super) const HOT: f64 = 400.0;
pub(super) const COLD: f64 = 300.0;

pub(super) fn boundary(mesh: &SolverMeshArtifact, name: &str) -> RegionTarget {
    let mut target = RegionTarget::boundary(
        runmat_meshing_core::fixtures::fixture_boundary_selection(&(name)),
    );
    let entities = mesh
        .topology
        .boundary_faces
        .iter()
        .flat_map(|f| f.provenance.iter())
        .filter(|e| target.matches_persistent_entity(e))
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    target.bind_resolved_entities(entities).unwrap();
    target
}

fn material(id: &str, k: f64) -> MaterialModel {
    let mut m = fixture_model(FixtureId::CantileverLinearStatic).materials[0].clone();
    m.material_id = id.into();
    m.name = id.into();
    m.mass_density_kg_per_m3 = Some(runmat_types::physical::Quantity::new(1000.0));
    let thermal = m.thermal.as_mut().unwrap();
    thermal.conductivity_w_per_mk = runmat_types::physical::Quantity::new(k);
    thermal.specific_heat_j_per_kgk = runmat_types::physical::Quantity::new(500.0);
    m
}

pub(super) fn workload(
    mesh: &SolverMeshArtifact,
    transfer: ThermalInterfaceTransfer,
) -> ThermalWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.structural = None;
    model.thermo_mechanical = None;
    model.constraints.clear();
    model.boundaries.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    model.initial_conditions.clear();
    model.materials = vec![material("lower", K_LOWER), material("upper", K_UPPER)];
    let mut lower = model.material_assignments[0].clone();
    lower.target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region("lower"));
    lower.expected_material_id = "lower".into();
    lower.assigned_material_id = "lower".into();
    let mut upper = lower.clone();
    upper.target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region("upper"));
    upper.expected_material_id = "upper".into();
    upper.assigned_material_id = "upper".into();
    model.material_assignments = vec![lower, upper];
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(350.0),
    });
    model.sequence =
        AnalysisSequence::from_steps("contact", vec![ProcedureStep::thermal_steady("heat")]);
    for (name, value) in [("lower_y_min", HOT), ("upper_y_max", COLD)] {
        model
            .constraints
            .push(AnalysisConstraint::Thermal(ThermalConstraint {
                constraint_id: name.into(),
                kind: ThermalConstraintKind::Prescribed {
                    target: boundary(mesh, name),
                    value: PrescribedValue::Constant(Temperature::new(value)),
                },
            }));
    }
    model.interfaces = vec![AnalysisInterface {
        interface_id: "joint".into(),
        primary: boundary(mesh, "interface_lower_upper"),
        secondary: boundary(mesh, "upper_contact"),
        kind: AnalysisInterfaceKind::ConjugateHeatTransfer(ConjugateHeatTransferInterfaceModel {
            transfer,
        }),
    }];
    ThermalWorkload {
        model,
        initial_fields: Vec::new(),
        step_id: ProcedureStepId("heat".into()),
        backend: ComputeBackend::Cpu,
        procedure: ThermalProcedureControl::Steady,
        residual_target: 1e-9,
    }
}

/// The joint as two named faces, one per body, as an imprinted CAD assembly names them.
fn mesh() -> SolverMeshArtifact {
    mesh_with([2, 5, 2])
}

/// The two-layer joint box with `cells` hexahedral cells (each split into tetrahedra).
pub(super) fn mesh_with(cells: [usize; 3]) -> SolverMeshArtifact {
    let mut mesh = layered_box_mesh(cells, [0.5, 1.0, 0.5], &[(SPLIT, "lower"), (2.0, "upper")]);
    for face in &mut mesh.topology.boundary_faces {
        if let Some(lower) = face
            .provenance
            .iter()
            .find(|entity| entity.source_topology_id == "interface_lower_upper")
            .cloned()
        {
            let mut upper = lower.clone();
            upper.source_topology_id = "upper_contact".into();
            face.provenance.push(upper);
            face.provenance.sort();
        }
    }
    mesh.canonical_digest = runmat_meshing_core::StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    mesh
}

// witness: FEA-THERMAL-DOMAIN-001/constraints @ thermal_steady/tet4/numerical_kernel; coverage:
//   partial;
//   boundary: A two-layer linear-mesh box with a contact conductance (or equivalent resistance)
//   matches the one-dimensional series-resistance solution at every node, the interface jump equals
//   flux over conductance, and the energy balance closes; Tet10 is not run.
#[test]
fn contact_conductance_carries_the_series_resistance_flux_and_jump() {
    // One-dimensional series resistance per unit area: L1/k1 + 1/h + L2/k2. Each layer's
    // profile is linear, so linear elements reproduce it exactly, with the jump q/h at the joint.
    for transfer in [
        ThermalInterfaceTransfer::Conductance(HeatTransferCoefficient::new(500.0)),
        ThermalInterfaceTransfer::Resistance(ThermalContactResistance::new(1.0 / 500.0)),
    ] {
        let mesh = mesh();
        let input = workload(&mesh, transfer);
        let result = input.execute(&mesh).unwrap();
        let h = 500.0;
        let q = (HOT - COLD) / (SPLIT / K_LOWER + 1.0 / h + (1.0 - SPLIT) / K_UPPER);
        let lower = |y: f64| HOT - q * y / K_LOWER;
        let upper = |y: f64| COLD + q * (1.0 - y) / K_UPPER;
        let temperature = result.temperature_snapshots[0].as_host_f64().unwrap();
        for (node, &value) in mesh.topology.nodes.iter().zip(temperature) {
            let y = node.coordinates_m.canonical_value()[1];
            // Interface nodes publish the first (lower) side here.
            let expected = if y <= SPLIT + 1e-12 {
                lower(y)
            } else {
                upper(y)
            };
            assert!(
                (value - expected).abs() < 1e-7 * HOT,
                "y={y} {value} vs {expected}"
            );
        }
        assert!(!result.interface_node_ids.is_empty());
        let second = result.interface_temperature_snapshots[0]
            .as_host_f64()
            .unwrap();
        for (&id, &value) in result.interface_node_ids.iter().zip(second) {
            let node = mesh
                .topology
                .nodes
                .iter()
                .find(|n| n.node_id == id)
                .unwrap();
            assert!((node.coordinates_m.canonical_value()[1] - SPLIT).abs() < 1e-12);
            let expected = upper(SPLIT);
            assert!(
                (value - expected).abs() < 1e-7 * HOT,
                "{value} vs {expected}"
            );
        }
        // The jump is the flux over the conductance.
        assert!(((lower(SPLIT) - upper(SPLIT)) - q / h).abs() < 1e-9 * HOT);
        for balance in &result.energy_balances {
            assert!(balance.relative_residual <= input.residual_target);
        }
        result.validate_complete().unwrap();
    }
}

// witness: FEA-THERMAL-DOMAIN-001/constraints @ thermal_steady/tet4/numerical_kernel; coverage:
//   partial;
//   boundary: Perfect contact shares the interface temperature across the joint.
#[test]
fn perfect_contact_shares_the_interface_temperature() {
    let mesh = mesh();
    let input = workload(&mesh, ThermalInterfaceTransfer::Perfect);
    let result = input.execute(&mesh).unwrap();
    assert!(result.interface_node_ids.is_empty());
    let q = (HOT - COLD) / (SPLIT / K_LOWER + (1.0 - SPLIT) / K_UPPER);
    let temperature = result.temperature_snapshots[0].as_host_f64().unwrap();
    for (node, &value) in mesh.topology.nodes.iter().zip(temperature) {
        let y = node.coordinates_m.canonical_value()[1];
        let expected = if y <= SPLIT {
            HOT - q * y / K_LOWER
        } else {
            COLD + q * (1.0 - y) / K_UPPER
        };
        assert!(
            (value - expected).abs() < 1e-7 * HOT,
            "y={y} {value} vs {expected}"
        );
    }
}

// witness: FEA-THERMAL-INPUT-001/boundaries @ thermal_steady/tet4/numerical_kernel; coverage: partial;
//   boundary: The manifest and published state carry second-side joint temperatures and element
//   means read each body's side, matching the closed form at every centroid; Tet4 only.
#[test]
fn published_state_and_element_means_read_each_side_of_the_joint() {
    // The manifest and the published state carry the second side's interface temperature, and
    // element means taken from the state use each body's own side: a linear field's element
    // mean is its value at the centroid.
    let mesh = mesh();
    let input = workload(
        &mesh,
        ThermalInterfaceTransfer::Conductance(HeatTransferCoefficient::new(500.0)),
    );
    let result = input.execute(&mesh).unwrap();
    let manifest = crate::execution::ThermalSolverResultManifest::from_result(&result).unwrap();
    assert_eq!(manifest.interface_node_ids, result.interface_node_ids);
    assert!(manifest.fields.iter().all(|fields| fields.len() == 5));
    let state =
        crate::execution::CanonicalSemanticState::from_thermal_result(&input.model, &mesh, &result)
            .unwrap();
    let payload = state.payload.thermal().unwrap();
    assert_eq!(payload.interface_node_ids, result.interface_node_ids);
    let (elements, _) = crate::execution::conduction_operating_temperatures(
        payload,
        &mesh,
        &input.model.interfaces,
        false,
    )
    .unwrap();
    let q = (HOT - COLD) / (SPLIT / K_LOWER + 1.0 / 500.0 + (1.0 - SPLIT) / K_UPPER);
    let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
    for (element, value) in projection.elements.iter().zip(elements) {
        let y = element.centroid_m[1];
        let expected = if element.region_id.source_topology_id == "lower" {
            HOT - q * y / K_LOWER
        } else {
            COLD + q * (1.0 - y) / K_UPPER
        };
        assert!(
            (value - expected).abs() < 1e-7 * HOT,
            "y={y} {value} vs {expected}"
        );
    }
}
