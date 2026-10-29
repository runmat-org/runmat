//! An absolute-temperature offset cannot change constant-property transport powers.
use super::*;

#[test]
fn transient_temperature_offset_preserves_flux_storage_and_boundary_power() {
    for quadratic in [false, true] {
        let mesh = fixture::mesh(quadratic, 2);
        for convection in [false, true] {
            let mut results = Vec::new();
            for reference in [300.0, 3000.0] {
                let mut input = fixture::base();
                input.model.thermal.as_mut().unwrap().reference_temperature =
                    Temperature::new(reference);
                fixture::transient(&mut input, 32);
                fixture::generation(
                    &mut input,
                    SignalRepresentation::Constant(SignalValue::Scalar(
                        HeatGenerationDensity::new(0.125),
                    )),
                );
                if convection {
                    for face in ["slab_left", "slab_right"] {
                        input.model.constraints.push(AnalysisConstraint::Thermal(
                            ThermalConstraint {
                                constraint_id: face.into(),
                                kind: ThermalConstraintKind::Prescribed {
                                    target: RegionTarget::boundary(
                                        runmat_meshing_core::fixtures::fixture_boundary_selection(
                                            &(face),
                                        ),
                                    ),
                                    value: PrescribedValue::Constant(Temperature::new(reference)),
                                },
                            },
                        ));
                    }
                    input.model.boundaries.push(AnalysisBoundary {
                        boundary_id: "ambient".into(),
                        target: RegionTarget::boundary(
                            runmat_meshing_core::fixtures::fixture_face("slab_insulated"),
                        ),
                        operator: BoundaryOperator::ThermalConvection {
                            ambient_temperature: PrescribedValue::Constant(Temperature::new(
                                reference + 2.0,
                            )),
                            coefficient: HeatTransferCoefficient::new(0.5),
                        },
                    });
                } else {
                    fixture::insulate_all(&mut input);
                }
                let cold = ThermalWorkload::from_canonical_bytes(&input.canonical_bytes().unwrap())
                    .unwrap();
                let result = cold.execute(&mesh).unwrap();
                result.validate_complete().unwrap();
                results.push(result);
            }
            let [low, high] = results.as_slice() else {
                unreachable!()
            };
            for (a, b) in low
                .temperature_snapshots
                .iter()
                .zip(&high.temperature_snapshots)
            {
                for (&a, &b) in a
                    .as_host_f64()
                    .unwrap()
                    .iter()
                    .zip(b.as_host_f64().unwrap())
                {
                    assert!((b - a - 2700.0).abs() < 1e-10);
                }
            }
            for (a, b) in low
                .temperature_gradient_snapshots
                .iter()
                .chain(&low.heat_flux_snapshots)
                .chain(&low.energy_storage_rate_snapshots)
                .zip(
                    high.temperature_gradient_snapshots
                        .iter()
                        .chain(&high.heat_flux_snapshots)
                        .chain(&high.energy_storage_rate_snapshots),
                )
            {
                for (&a, &b) in a
                    .as_host_f64()
                    .unwrap()
                    .iter()
                    .zip(b.as_host_f64().unwrap())
                {
                    assert_eq!(a, b, "offset-dependent recovered transport");
                }
            }
            for (a, b) in low.energy_balances.iter().zip(&high.energy_balances) {
                assert_eq!(a, b, "offset-dependent physical energy balance");
            }
        }
    }
}
