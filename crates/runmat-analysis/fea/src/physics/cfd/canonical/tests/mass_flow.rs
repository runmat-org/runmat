//! Independent integral-source conservation and constant-flow history references.
use super::super::qualification::mass_flow::uniform;
use super::*;
use crate::excitation::ProjectionSample;
use runmat_analysis_core::*;

// witness: FEA-FLUID-INPUTS-001/signals @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   boundary: Signed rates across rotated exact faces; three modulo-index pieces demonstrate
//   additive realization only, not distributed cell assembly.
#[test]
fn signed_mass_sources_preserve_exact_faces_partitions_and_cold_original_fields() {
    for (speed, density, rotated) in [(0.001, 1.0, false), (-0.002, 3.0, true), (0.0, 2.0, false)] {
        let f = uniform(2, speed, density, rotated);
        let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
        let exact = if rotated {
            reference::rotate([speed, 0.0, 0.0])
        } else {
            [speed, 0.0, 0.0]
        };
        for id in ["mass-inlet", "mass-outlet"] {
            let mut pieces = [0.0; 3];
            let mut area = 0.0;
            for (i, (face, law)) in p.topology.faces.iter().zip(&p.boundaries).enumerate() {
                if let boundary::FaceBoundary::Velocity {
                    authority_id,
                    value,
                    ..
                } = law
                {
                    if *authority_id == id {
                        assert!(norm(sub(*value, exact)) < 1e-15);
                        pieces[i % 3] +=
                            density * geometry_ops::dot(*value, face.owner_area_vector_m2);
                        area += face.area_m2.canonical_value();
                    }
                }
            }
            assert!((area - 1.0).abs() < 1e-14);
            let outward = density * speed * if id == "mass-inlet" { -1.0 } else { 1.0 };
            assert!((pieces.iter().sum::<f64>() - outward).abs() < 1e-14);
        }
        let original = p.solve().unwrap();
        for u in original.velocity_m_per_s.canonical_values() {
            assert!(norm(sub(*u, exact)) < 1e-8);
        }
        assert!(original
            .pressure_fluctuation_pa
            .canonical_values()
            .iter()
            .all(|p| p.abs() < 1e-7));
        assert_eq!(
            original.pressure_reference_pa.canonical_values().as_slice(),
            vec![1e12; p.topology.cells.len()]
        );
        for id in ["mass-inlet", "mass-outlet"] {
            let balance = original
                .residual
                .boundary_balances
                .iter()
                .find(|b| b.authority_id == id)
                .unwrap();
            let exact = density * speed * if id == "mass-inlet" { -1.0 } else { 1.0 };
            assert!((balance.mass_flow_rate_kg_per_s.canonical_value() - exact).abs() < 1e-14);
        }
        let cold: CanonicalFluidResult =
            serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
        assert_eq!(p.recover(&cold).unwrap(), original.residual);
        let mut corrupt = cold;
        corrupt.face_volume_flux_m3_per_s = corrupt
            .face_volume_flux_m3_per_s
            .try_map_canonical_storage(|mut values| {
                values[0] += 1e-4;
                values
            })
            .unwrap();
        assert!(p.recover(&corrupt).is_err());
    }
}

// witness: FEA-FLUID-INPUTS-001/rejection @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   boundary: Selected duplicate/wrong-quantity/source authority and budget rejection only.
#[test]
fn source_authority_conflicts_quantity_and_budget_reject_before_solve() {
    for mutation in 0..11 {
        let mut f = uniform(2, 0.001, 1.0, false);
        match mutation {
            0 => f.actions.push(f.actions[0].clone()),
            1 => f.signals.push(f.signals[0].clone()),
            2 => {
                let ExcitationAction::MassFlow(a) = &mut f.actions[0] else {
                    unreachable!()
                };
                a.target = ExcitationTarget::WholeModel;
            }
            3 => {
                let ExcitationAction::MassFlow(a) = &mut f.actions[0] else {
                    unreachable!()
                };
                a.projection.preserve_moment = true;
            }
            4 => {
                let PhysicalSignal::MassFlowRate(s) = &mut f.signals[0] else {
                    unreachable!()
                };
                s.frame = Some(CoordinateFrame::Global);
            }
            5 => {
                let ExcitationTarget::Region(target) = &f.actions[0].excitation().target else {
                    unreachable!()
                };
                f.constraints.push(FluidConstraint {
                    constraint_id: "conflicting-wall".into(),
                    kind: FluidConstraintKind::NoSlipWall {
                        target: target.clone(),
                    },
                });
            }
            6 => {
                let required = CanonicalFluidProblem::prepare(f.input())
                    .unwrap()
                    .reservation();
                f.policy.maximum_operator_work = Some(required.operator_work_upper_bound - 1);
            }
            7 => {
                let mut wrong = case(2, ReferenceCase::Duct, false).signals.remove(0);
                let PhysicalSignal::BodyForceDensity(signal) = &mut wrong else {
                    unreachable!()
                };
                signal.signal_id = f.signals[0].signal_id().clone();
                f.signals[0] = wrong;
            }
            8 => {
                let ExcitationAction::MassFlow(action) = &mut f.actions[0] else {
                    unreachable!()
                };
                action.direction = Some(Direction {
                    vector: [1.0, 0.0, 0.0],
                });
            }
            9 => {
                let ExcitationAction::MassFlow(action) = &mut f.actions[0] else {
                    unreachable!()
                };
                action.distribution = ExcitationDistribution::IntensityPerArea;
            }
            10 => {
                let PhysicalSignal::MassFlowRate(signal) = &mut f.signals[0] else {
                    unreachable!()
                };
                let SignalRepresentation::Constant(SignalValue::Scalar(value)) =
                    &mut signal.representation
                else {
                    unreachable!()
                };
                *value = runmat_types::physical::Quantity::new(value.canonical_value() * (2.0));
            }
            _ => unreachable!(),
        }
        assert!(
            CanonicalFluidProblem::prepare(f.input()).is_err(),
            "mutation {mutation}"
        );
    }
}

#[test]
fn shared_mass_history_drives_original_transient_and_fluctuation_pressure() {
    let density = 3.0;
    let start = 0.001;
    let acceleration = 0.01;
    let f = super::super::qualification::mass_flow::accelerating(2, start, acceleration, density);
    let at = |value| ProjectionSample::TimeSeconds {
        value,
        approach: SignalSampleApproach::At,
    };
    assert!(CanonicalFluidProblem::prepare(f.input()).is_err());
    assert!(CanonicalFluidProblem::prepare_at(f.input(), at(2.0)).is_err());
    let initial = CanonicalFluidProblem::prepare_at(f.input(), at(0.0)).unwrap();
    let receiving = CanonicalFluidProblem::prepare_at(f.input(), at(1.0)).unwrap();
    let u = vec![[start, 0.0, 0.0]; f.topology.cells.len()];
    let mut pressure = f
        .topology
        .cells
        .iter()
        .map(|c| -density * acceleration * c.centroid_m[0])
        .collect::<Vec<_>>();
    initial
        .pressure_reference_plan
        .normalize(&initial, &mut pressure)
        .unwrap();
    let q = f
        .topology
        .faces
        .iter()
        .map(|face| geometry_ops::dot([start, 0.0, 0.0], face.owner_area_vector_m2))
        .collect::<Vec<_>>();
    let previous = CanonicalFluidFields {
        velocity_m_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Velocity,
            _,
        >::new(&u),
        pressure_fluctuation_pa: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Pressure,
            _,
        >::new(&pressure),
        pressure_reference_pa: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Pressure,
            _,
        >::new(initial.pressure_reference_pa.canonical_values()),
        face_volume_flux_m3_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::VolumeFlowRate,
            _,
        >::new(&q),
    };
    let interval = receiving
        .prepare_backward_euler(0.0, 1.0, previous)
        .unwrap();
    let result = interval.solve().unwrap();
    for v in result.velocity_m_per_s.canonical_values() {
        assert!(norm(sub(*v, [start + acceleration, 0.0, 0.0])) < 1e-8);
    }
    for g in geometry_ops::pressure_gradient(
        &receiving,
        result.pressure_fluctuation_pa.canonical_values(),
    )
    .unwrap()
    {
        assert!(norm(geometry_ops::add(g, [density * acceleration, 0.0, 0.0])) < 1e-6);
    }
    let cold: CanonicalFluidResult =
        serde_json::from_slice(&serde_json::to_vec(&result).unwrap()).unwrap();
    assert_eq!(interval.recover(&cold).unwrap(), result.residual);
    assert!(receiving.recover(&cold).is_err());
}

#[test]
fn nonuniform_face_areas_use_integral_rates_and_work_is_admitted_before_binding() {
    let mut f = uniform(3, 0.002, 4.0, false);
    // Monotone nonuniform y spacing and a 2x3 cross section: analytic total area six.
    for node in &mut f.mesh.topology.nodes {
        {
            let quantity_change = 2.0 * node.coordinates_m.canonical_values()[1].powi(2);
            let quantity_index = 1;
            let quantity_slot = &mut node.coordinates_m;
            let mut quantity_coordinates = quantity_slot.canonical_value();
            quantity_coordinates[quantity_index] = quantity_change;
            *quantity_slot = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(quantity_coordinates);
        };
        {
            let quantity_change = 3.0;
            let quantity_index = 2;
            let quantity_slot = &mut node.coordinates_m;
            let mut quantity_coordinates = quantity_slot.canonical_value();
            quantity_coordinates[quantity_index] *= quantity_change;
            *quantity_slot = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(quantity_coordinates);
        };
    }
    f.mesh.seal_canonical_digest().unwrap();
    f.source = crate::topology::AnalysisProjection::derive(&f.mesh).unwrap();
    f.topology =
        crate::topology::FiniteVolumeTopology::derive(&f.source, fixture::geometry_policy())
            .unwrap();
    for signal in &mut f.signals {
        let PhysicalSignal::MassFlowRate(signal) = signal else {
            unreachable!()
        };
        let SignalRepresentation::Constant(SignalValue::Scalar(value)) = &mut signal.representation
        else {
            unreachable!()
        };
        *value = runmat_types::physical::Quantity::new(value.canonical_value() * (6.0));
    }
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let required = p.reservation();
    for id in ["mass-inlet", "mass-outlet"] {
        let mut areas = Vec::new();
        let mut outward = 0.0;
        for (face, law) in p.topology.faces.iter().zip(&p.boundaries) {
            if let boundary::FaceBoundary::Velocity {
                authority_id,
                value,
                ..
            } = law
            {
                if *authority_id == id {
                    areas.push(face.area_m2.canonical_value());
                    assert!(norm(sub(*value, [0.002, 0.0, 0.0])) < 1e-15);
                    outward += 4.0 * geometry_ops::dot(*value, face.owner_area_vector_m2);
                }
            }
        }
        assert!(areas.iter().any(|a| (*a - areas[0]).abs() > 1e-3));
        assert!((areas.iter().sum::<f64>() - 6.0).abs() < 1e-13);
        assert!((outward - 0.048 * if id == "mass-inlet" { -1.0 } else { 1.0 }).abs() < 1e-14);
    }
    f.policy.maximum_operator_work = Some(required.operator_work_upper_bound - 1);
    assert!(matches!(
        CanonicalFluidProblem::prepare(f.input()),
        Err(CanonicalFluidError::Resource(_))
    ));
}

#[test]
fn mass_history_meets_original_native_controls_without_exact_pressure_seed() {
    let mut f = super::super::qualification::mass_flow::accelerating(2, 0.001, 0.01, 1.0);
    // Original scheduled native fixture controls, unchanged; pressure begins at its gauge.
    f.policy.maximum_outer_iterations = 200;
    f.policy.maximum_nonorthogonal_iterations = 4;
    f.policy.maximum_linear_iterations = 100;
    f.policy.gmres_restart = 8;
    f.policy.linear_relative_tolerance = 1e-10;
    f.policy.physical_relative_tolerance = 1e-8;
    let at = |value| ProjectionSample::TimeSeconds {
        value,
        approach: SignalSampleApproach::At,
    };
    let initial = CanonicalFluidProblem::prepare_at(f.input(), at(0.0)).unwrap();
    let receiving = CanonicalFluidProblem::prepare_at(f.input(), at(0.1)).unwrap();
    let u = vec![[0.001, 0.0, 0.0]; f.topology.cells.len()];
    let pressure = vec![0.0; u.len()];
    let q = f
        .topology
        .faces
        .iter()
        .map(|face| geometry_ops::dot(u[0], face.owner_area_vector_m2))
        .collect::<Vec<_>>();
    let previous = CanonicalFluidFields {
        velocity_m_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Velocity,
            _,
        >::new(&u),
        pressure_fluctuation_pa: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Pressure,
            _,
        >::new(&pressure),
        pressure_reference_pa: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Pressure,
            _,
        >::new(initial.pressure_reference_pa.canonical_values()),
        face_volume_flux_m3_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::VolumeFlowRate,
            _,
        >::new(&q),
    };
    let interval = receiving
        .prepare_backward_euler(0.0, 0.1, previous)
        .unwrap();
    let result = interval.solve().unwrap();
    for u in result.velocity_m_per_s.canonical_values() {
        assert!(norm(sub(*u, [0.002, 0.0, 0.0])) < 1e-8);
    }
    for (cell, p) in f
        .topology
        .cells
        .iter()
        .zip(result.pressure_fluctuation_pa.canonical_values())
    {
        assert!((p + 0.01 * (cell.centroid_m[0] - 1.0)).abs() < 2e-7);
    }
}
