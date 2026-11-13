//! Geometric conservation: a uniform stream stays exactly uniform while the interior of the
//! mesh moves, and the swept face volumes reproduce every cell's volume change.
use super::super::transient::{CanonicalFluidFields, CanonicalFluidMeshMotion};
use super::*;
use crate::topology::{AnalysisProjection, FiniteVolumeTopology};
use runmat_analysis_core::{
    AnalysisBoundary, BoundaryOperator, FluidConstraint, FluidConstraintKind, PrescribedValue,
    Pressure, RegionTarget, Velocity,
};

#[test]
fn uniform_stream_is_preserved_on_a_moving_mesh() {
    let mut f = case(3, ReferenceCase::Couette, false);
    let stream = [0.7, -0.2, 0.1];
    let mut inflow = RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
        "walls_and_inlet",
    ));
    inflow
        .bind_resolved_entities(
            f.constraints
                .iter()
                .flat_map(|c| {
                    let FluidConstraintKind::PrescribedVelocity { target, .. } = &c.kind else {
                        unreachable!()
                    };
                    target.resolved_entities().iter().cloned()
                })
                .collect(),
        )
        .unwrap();
    let mut outlet = RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("outlet"));
    outlet
        .bind_resolved_entities(
            f.operators
                .iter()
                .flat_map(|b| b.target.resolved_entities().iter().cloned())
                .collect(),
        )
        .unwrap();
    f.constraints = vec![FluidConstraint {
        constraint_id: "stream".into(),
        kind: FluidConstraintKind::PrescribedVelocity {
            target: inflow,
            value: PrescribedValue::Constant(stream.map(Velocity::new)),
        },
    }];
    f.operators = vec![AnalysisBoundary {
        boundary_id: "outlet".into(),
        target: outlet,
        operator: BoundaryOperator::FluidOutletPressure {
            value: PrescribedValue::Constant(Pressure::new(0.0)),
        },
    }];
    // Move the interior nodes only; the boundary stays where its laws are authored.
    let mut moved = f.mesh.clone();
    for node in &mut moved.topology.nodes {
        let x = node.coordinates_m;
        if x.canonical_values()
            .iter()
            .all(|v| *v > 1e-9 && *v < 1.0 - 1e-9)
        {
            node.coordinates_m = runmat_types::physical::Quantity::new([
                x.canonical_value()[0] + 0.04 * (3.0 * x.canonical_value()[1]).sin(),
                x.canonical_value()[1] - 0.03 * (2.0 * x.canonical_value()[2]).cos(),
                x.canonical_value()[2] + 0.02 * x.canonical_value()[0],
            ]);
        }
    }
    moved.seal_canonical_digest().unwrap();
    let arrival = AnalysisProjection::derive(&moved).unwrap();
    let topology = FiniteVolumeTopology::derive(&arrival, fixture::geometry_policy()).unwrap();
    let dt = 0.05;
    let w = swept_face_flux(&f.source, &arrival, &topology, dt).unwrap();
    let previous_volume = f
        .topology
        .cells
        .iter()
        .map(|c| c.volume_m3)
        .collect::<Vec<_>>();
    let p = CanonicalFluidProblem::prepare(CanonicalFluidInput {
        source: &arrival,
        topology: &topology,
        material: &f.material,
        constraints: &f.constraints,
        boundary_operators: &f.operators,
        signals: &f.signals,
        excitations: &f.actions,
        policy: f.policy,
    })
    .unwrap();
    let velocity = vec![stream; f.topology.cells.len()];
    let pressure = vec![0.0; velocity.len()];
    let reference = vec![0.0; velocity.len()];
    let flux = f
        .topology
        .faces
        .iter()
        .map(|face| geometry_ops::dot(stream, face.owner_area_vector_m2))
        .collect::<Vec<_>>();
    let step = p
        .prepare_backward_euler_moving(
            0.0,
            dt,
            CanonicalFluidFields {
                velocity_m_per_s: runmat_types::physical::Quantity::new(velocity.as_slice()),
                pressure_reference_pa: runmat_types::physical::Quantity::new(reference.as_slice()),
                pressure_fluctuation_pa: runmat_types::physical::Quantity::new(pressure.as_slice()),
                face_volume_flux_m3_per_s: runmat_types::physical::Quantity::new(flux.as_slice()),
            },
            CanonicalFluidMeshMotion {
                previous_volume_m3: &previous_volume,
                face_mesh_flux_m3_per_s: &w,
            },
        )
        .unwrap();
    let r = step.solve().unwrap();
    step.recover(&r).unwrap();
    for u in r.velocity_m_per_s.canonical_values() {
        assert!(norm(sub(*u, stream)) < 1e-8, "stream disturbed: {u:?}");
    }
    // Without the swept fluxes the same interval violates the conservation law.
    let still = vec![0.0; w.len()];
    assert!(p
        .prepare_backward_euler_moving(
            0.0,
            dt,
            CanonicalFluidFields {
                velocity_m_per_s: runmat_types::physical::Quantity::new(velocity.as_slice()),
                pressure_reference_pa: runmat_types::physical::Quantity::new(reference.as_slice()),
                pressure_fluctuation_pa: runmat_types::physical::Quantity::new(pressure.as_slice()),
                face_volume_flux_m3_per_s: runmat_types::physical::Quantity::new(flux.as_slice()),
            },
            CanonicalFluidMeshMotion {
                previous_volume_m3: &previous_volume,
                face_mesh_flux_m3_per_s: &still,
            },
        )
        .is_err());
}
