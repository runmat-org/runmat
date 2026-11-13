use super::*;

#[test]
fn transient_reaction_power_and_scale_share_the_same_nodal_transfers() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut assembly = energy_assembly(&model, order, 0.0);
        let previous = vec![293.15; assembly.projection.nodes.len()];
        for (node, projected) in assembly.projection.nodes.iter().enumerate() {
            if projected.coordinates_m.canonical_value()[0] == 0.0 {
                assembly
                    .prescribed_temperature_k
                    .try_replace(node, runmat_types::physical::Quantity::new(Some(310.0)))
                    .unwrap();
            }
        }
        for increment in [0.01, 0.1, 1.0, 10.0, 16.85] {
            let temperature = assembly
                .projection
                .nodes
                .iter()
                .map(|node| 293.15 + increment * (1.0 + node.coordinates_m.canonical_value()[0]))
                .collect::<Vec<_>>();
            let balance = assembly
                .energy_balance(
                    &model,
                    ExcitationSamplePoint::Static,
                    &previous,
                    &temperature,
                    1.0,
                    0.0,
                )
                .unwrap();
            assert!(
                balance.validate().is_ok(),
                "{order:?}, {increment}: {balance:?}"
            );
        }
    }
}

fn energy_assembly(
    model: &AnalysisModel,
    order: ElementOrder,
    source_density: f64,
) -> ThermalOperatorAssembly {
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
    energy_assembly_on(model, &mesh, source_density)
}

fn energy_assembly_on(
    model: &AnalysisModel,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    source_density: f64,
) -> ThermalOperatorAssembly {
    assemble(
        model,
        mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(mesh, &[])
            .unwrap()
            .support(),
        &heat_generation(source_density),
        ExcitationSamplePoint::Static,
    )
    .unwrap()
}

// witness: FEA-THERMAL-FIELDS-001/global_energy @ thermal_steady/tet4|tet10/numerical_kernel,
//   thermal_transient/tet4|tet10/numerical_kernel;
//   boundary: Both-order physical heat-recovery identities and uniform zero flux; no arbitrary
//   regional partition balance.
#[test]
fn internal_conduction_is_not_external_power_and_uniform_flux_is_exactly_zero() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let assembly = energy_assembly(&model, order, 0.0);
        let uniform = vec![315.0; assembly.projection.nodes.len()];
        let energy = assembly
            .energy_balance(
                &model,
                ExcitationSamplePoint::Static,
                &uniform,
                &uniform,
                0.25,
                0.0,
            )
            .unwrap();
        assert_eq!(energy.residual_w.canonical_value(), 0.0);
        assert_eq!(energy.relative_residual, 0.0);
        assert_eq!(energy.boundary_loss_w.canonical_value(), 0.0);
        assert_eq!(
            assembly.recover_element_heat_flux(&uniform).unwrap(),
            vec![[0.0; 3]]
        );

        let nonuniform = assembly
            .projection
            .nodes
            .iter()
            .map(|node| 315.0 + 20.0 * node.coordinates_m.canonical_value()[0])
            .collect::<Vec<_>>();
        let energy = assembly
            .energy_balance(
                &model,
                ExcitationSamplePoint::Static,
                &nonuniform,
                &nonuniform,
                0.25,
                0.0,
            )
            .unwrap();
        assert_eq!(energy.boundary_loss_w.canonical_value(), 0.0);
        assert_eq!(energy.residual_w.canonical_value(), 0.0);
        let expected = -20.0 * assembly.element_conductivity_w_per_mk.canonical_values()[0];
        let actual = assembly.recover_element_heat_flux(&nonuniform).unwrap()[0][0];
        assert!((actual - expected).abs() <= expected.abs() * 1.0e-12);
    }
}

// witness: FEA-THERMAL-FIELDS-001/global_energy @ thermal_steady/tet4|tet10/numerical_kernel,
//   thermal_transient/tet4|tet10/numerical_kernel;
//   boundary: Both-order conductive prescribed reaction signs and internal-pair cancellation; no
//   transient continuum convergence credit.
#[test]
fn conductive_prescribed_reactions_have_analytical_sign_and_cancel_internal_pairs() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut assembly = energy_assembly(&model, order, 0.0);
        let temperature = assembly
            .projection
            .nodes
            .iter()
            .map(|node| 300.0 + 20.0 * node.coordinates_m.canonical_value()[0])
            .collect::<Vec<_>>();
        for (node, projected) in assembly.projection.nodes.iter().enumerate() {
            if projected.coordinates_m.canonical_value()[0] == 0.0 {
                assembly
                    .prescribed_temperature_k
                    .try_replace(
                        node,
                        runmat_types::physical::Quantity::new(Some(temperature[node])),
                    )
                    .unwrap();
            }
        }
        let energy = assembly
            .energy_balance(
                &model,
                ExcitationSamplePoint::Static,
                &temperature,
                &temperature,
                1.0,
                0.0,
            )
            .unwrap();
        // This imposed linear field does not satisfy the other faces' zero-natural-flux
        // conditions. Check its weak nodal reaction, not an equilibrated boundary flux:
        // R = -integral(k * grad(T) . grad(sum of prescribed shape functions)).
        // On the canonical tetrahedron that shape sum is 1-x (Tet4), or
        // 1-3x+2x² (Tet10); mean(x)=1/4 gives mean derivatives -1 and -2.
        let derivative_magnitude = match order {
            ElementOrder::Tet4 => 1.0,
            ElementOrder::Tet10 => 2.0,
        };
        let expected = derivative_magnitude
            * 20.0
            * assembly.element_conductivity_w_per_mk.canonical_values()[0]
            * assembly.projection.elements[0].signed_volume_m3;
        assert!(
            (energy.prescribed_boundary_power_w.canonical_value() - expected).abs()
                <= expected * 1.0e-12,
            "{order:?}: {energy:?}"
        );
        assert_eq!(energy.boundary_loss_w.canonical_value(), 0.0);
        for (node, &value) in temperature.iter().enumerate() {
            assembly
                .prescribed_temperature_k
                .try_replace(node, runmat_types::physical::Quantity::new(Some(value)))
                .unwrap();
        }
        let all_prescribed = assembly
            .energy_balance(
                &model,
                ExcitationSamplePoint::Static,
                &temperature,
                &temperature,
                1.0,
                0.0,
            )
            .unwrap();
        all_prescribed.validate().unwrap();
        // Equal/opposite nodal conductive transfers cancel globally, but must
        // not disappear from the physical through-flow normalization. The
        // integrated linear-field nodal gradients give L1 = 2 * expected.
        assert!(
            (all_prescribed.scale.prescribed_w.canonical_value() - 2.0 * expected).abs()
                <= 2.0 * expected * 1.0e-12,
            "{order:?}: {all_prescribed:?}"
        );
        assert!(
            all_prescribed.relative_residual < 1.0e-12,
            "{order:?}: {all_prescribed:?}"
        );
    }
}

#[test]
fn robin_net_transfer_is_physical_and_never_hides_unrecovered_tiny_heating() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for ambient in [305.0, 1.0e8] {
            let mut model = fixture_model(FixtureId::CantileverLinearStatic);
            let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
            let projection = AnalysisProjection::derive(&mesh).unwrap();
            let face = &projection.boundary_faces[0];
            let selector = face.provenance[0].analysis_selection_id().unwrap();
            model.boundaries = vec![AnalysisBoundary {
                boundary_id: "physical_robin".into(),
                target: RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_boundary_selection(&(selector)),
                ),
                operator: BoundaryOperator::ThermalConvection {
                    ambient_temperature: PrescribedValue::Constant(Temperature::new(ambient)),
                    coefficient: HeatTransferCoefficient::new(18.0),
                },
            }];
            let assembly = energy_assembly(&model, order, 0.0);
            let equilibrium = vec![ambient; assembly.projection.nodes.len()];
            let energy = assembly
                .energy_balance(
                    &model,
                    ExcitationSamplePoint::Static,
                    &equilibrium,
                    &equilibrium,
                    0.25,
                    0.0,
                )
                .unwrap();
            assert_eq!(energy.applied_power_w.canonical_value(), 0.0);
            assert_eq!(energy.boundary_loss_w.canonical_value(), 0.0);
            assert_eq!(energy.relative_residual, 0.0);
            for difference in [-20.0, 20.0] {
                let temperature = vec![ambient + difference; equilibrium.len()];
                let energy = assembly
                    .energy_balance(
                        &model,
                        ExcitationSamplePoint::Static,
                        &temperature,
                        &temperature,
                        0.25,
                        0.0,
                    )
                    .unwrap();
                let expected = 18.0 * difference * face.area_m2.canonical_value();
                assert!(
                    (energy.boundary_loss_w.canonical_value() - expected).abs()
                        <= expected.abs() * 1.0e-12
                );
                assert_eq!(energy.applied_power_w.canonical_value(), 0.0);
                assert_eq!(energy.relative_residual, 1.0);
            }
            for density in [1.0e-30, 1.0e-15, 1.0e-3] {
                let heated = energy_assembly(&model, order, density);
                let energy = heated
                    .energy_balance(
                        &model,
                        ExcitationSamplePoint::Static,
                        &equilibrium,
                        &equilibrium,
                        0.25,
                        0.0,
                    )
                    .unwrap();
                assert!(energy.applied_power_w.canonical_value() > 0.0);
                assert_eq!(energy.boundary_loss_w.canonical_value(), 0.0);
                assert_eq!(
                    energy.residual_w.canonical_value(),
                    -energy.applied_power_w.canonical_value()
                );
                // Tet10 uniform source has four corner weights -V/20 and six edge
                // weights V/5: its nodal L1 scale is 7V/5, versus V for Tet4.
                let expected = if order == ElementOrder::Tet10 {
                    5.0 / 7.0
                } else {
                    1.0
                };
                assert!((energy.relative_residual - expected).abs() < 1.0e-12);
                assert!(energy.relative_residual > 1.0e-4);
            }
        }
    }
}

#[test]
fn small_boundary_flux_is_not_lost_in_robin_forcing_regardless_of_operator_order() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let face = &projection.boundary_faces[0];
    let selector = face.provenance[0].analysis_selection_id().unwrap();
    let target = RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_boundary_selection(
        &(selector),
    ));
    model.boundaries = vec![
        AnalysisBoundary {
            boundary_id: "ambient".into(),
            target: target.clone(),
            operator: BoundaryOperator::ThermalConvection {
                ambient_temperature: PrescribedValue::Constant(Temperature::new(305.0)),
                coefficient: HeatTransferCoefficient::new(18.0),
            },
        },
        AnalysisBoundary {
            boundary_id: "tiny_flux".into(),
            target,
            operator: BoundaryOperator::ThermalHeatFlux {
                value: PrescribedValue::Constant(HeatFlux::new(1.0e-30)),
            },
        },
    ];
    for _ in 0..2 {
        let assembly = energy_assembly(&model, ElementOrder::Tet10, 0.0);
        let temperature = vec![305.0; projection.nodes.len()];
        let energy = assembly
            .energy_balance(
                &model,
                ExcitationSamplePoint::Static,
                &temperature,
                &temperature,
                1.0,
                0.0,
            )
            .unwrap();
        let expected = 1.0e-30 * face.area_m2.canonical_value();
        assert!((energy.applied_power_w.canonical_value() - expected).abs() <= expected * 1.0e-12);
        // Quadratic triangular corner integrals vanish analytically; quadrature can
        // leave roundoff-scale signed corner weights in the nodal absolute sum.
        assert!((energy.relative_residual - 1.0).abs() < 1.0e-12);
        model.boundaries.reverse();
    }
}

// witness: FEA-THERMAL-FIELDS-001/global_energy @ thermal_steady/tet4|tet10/numerical_kernel;
//   coverage: partial;
//   check: bar-heat-flow; boundary: A bar held at two end temperatures reports its closed-form through-flow
//   k*A*dT/L as the heat entering and leaving through the prescribed ends, while the net
//   prescribed reaction cancels; other boundary conditions are covered elsewhere.
#[test]
fn bar_between_two_fixed_temperatures_reports_its_closed_form_heat_flow() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    // The slab fixture is the unit cube [0, 1]^3: a prism of length 1 m along x whose
    // prescribed end faces x = 0 and x = 1 each have area 1 m^2.
    let (length_m, area_m2) = (1.0, 1.0);
    for (order, mesh) in [
        (
            ElementOrder::Tet4,
            runmat_meshing_core::fixtures::slab_mesh(2),
        ),
        (
            ElementOrder::Tet10,
            runmat_meshing_core::fixtures::quadratic_slab_mesh(2),
        ),
    ] {
        let mut assembly = energy_assembly_on(&model, &mesh, 0.0);
        let coordinates = assembly
            .projection
            .nodes
            .iter()
            .map(|node| node.coordinates_m.canonical_value())
            .collect::<Vec<_>>();
        let (cold_k, hot_k) = (300.0, 350.0);
        for (node, point) in coordinates.iter().enumerate() {
            let prescribed = if point[0] == 0.0 {
                Some(cold_k)
            } else if (point[0] - length_m).abs() <= 1.0e-12 * length_m {
                Some(hot_k)
            } else {
                None
            };
            assembly
                .prescribed_temperature_k
                .try_replace(node, runmat_types::physical::Quantity::new(prescribed))
                .unwrap();
        }
        let reference = assembly
            .steady_reference_temperatures(&model, ExcitationSamplePoint::Static, 293.15)
            .unwrap();
        let operator = assembly
            .steady_increment_operator(&model, ExcitationSamplePoint::Static, &reference)
            .unwrap();
        let solved = solve_operator_system(
            &operator,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
        );
        assert!(solved.converged);
        let mut temperature = solved.solution;
        for (node, value) in temperature.iter_mut().enumerate() {
            *value = assembly.prescribed_temperature_k.canonical_values()[node]
                .unwrap_or(*value + reference[node]);
        }
        let balance = assembly
            .energy_balance(
                &model,
                ExcitationSamplePoint::Static,
                &temperature,
                &temperature,
                1.0,
                0.0,
            )
            .unwrap();
        balance.validate().unwrap();
        // A linear field is exact in both element orders: Q = k A (T_hot - T_cold) / L.
        let conductivity = assembly.element_conductivity_w_per_mk.canonical_values()[0];
        let expected = conductivity * area_m2 * (hot_k - cold_k) / length_m;
        assert!(expected > 0.0);
        let flow = balance.prescribed_heat_flow;
        for gross in [flow.entering_w, flow.leaving_w] {
            assert!(
                (gross.canonical_value() - expected).abs() <= expected * 1.0e-8,
                "{order:?}: expected {expected} W, {balance:?}"
            );
        }
        assert!(
            balance.prescribed_boundary_power_w.canonical_value().abs() <= expected * 1.0e-8,
            "{order:?}: {balance:?}"
        );
    }
}

#[test]
fn opposing_prescribed_heat_flows_keep_their_physical_scale_after_cancellation() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut assembly = energy_assembly(&model, order, 0.0);
        for (node, projected) in assembly.projection.nodes.iter().enumerate() {
            assembly
                .prescribed_temperature_k
                .try_replace(
                    node,
                    runmat_types::physical::Quantity::new(
                        match projected.coordinates_m.canonical_value() {
                            [0.0, 0.0, 0.0] => Some(300.0),
                            [1.0, 0.0, 0.0] => Some(400.0),
                            _ => None,
                        },
                    ),
                )
                .unwrap();
        }
        let reference = assembly
            .steady_reference_temperatures(&model, ExcitationSamplePoint::Static, 293.15)
            .unwrap();
        let operator = assembly
            .steady_increment_operator(&model, ExcitationSamplePoint::Static, &reference)
            .unwrap();
        let solved = solve_operator_system(
            &operator,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
        );
        assert!(solved.converged);
        let mut temperature = solved.solution;
        for (node, value) in temperature.iter_mut().enumerate() {
            *value = assembly.prescribed_temperature_k.canonical_values()[node]
                .unwrap_or(*value + reference[node]);
        }
        let balance = assembly
            .energy_balance(
                &model,
                ExcitationSamplePoint::Static,
                &temperature,
                &temperature,
                1.0,
                0.0,
            )
            .unwrap();
        balance.validate().unwrap();
        assert!(balance.scale.prescribed_w.canonical_value() > 0.0);
        assert_eq!(balance.applied_power_w.canonical_value(), 0.0);
        assert_eq!(balance.boundary_loss_w.canonical_value(), 0.0);
        assert_eq!(balance.storage_rate_w.canonical_value(), 0.0);
        assert!(
            balance.relative_residual < 1.0e-10,
            "{order:?}: {balance:?}"
        );
        let free = assembly
            .prescribed_temperature_k
            .canonical_values()
            .iter()
            .position(Option::is_none)
            .unwrap();
        temperature[free] += 0.1;
        let perturbed = assembly
            .energy_balance(
                &model,
                ExcitationSamplePoint::Static,
                &temperature,
                &temperature,
                1.0,
                0.0,
            )
            .unwrap();
        assert!(
            perturbed.relative_residual > 1.0e-6,
            "{order:?}: {perturbed:?}"
        );
    }
}

#[test]
fn steady_increment_formulation_preserves_exact_equilibrium_without_erasing_small_loads() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for robin_anchor in [false, true] {
            let mut model = fixture_model(FixtureId::CantileverLinearStatic);
            let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
            let projection = AnalysisProjection::derive(&mesh).unwrap();
            let selector = projection.boundary_faces[0].provenance[0]
                .analysis_selection_id()
                .unwrap();
            let equilibrium_k = 350.0;
            if robin_anchor {
                model.boundaries.push(AnalysisBoundary {
                    boundary_id: "increment_robin".into(),
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(&(selector)),
                    ),
                    operator: BoundaryOperator::ThermalConvection {
                        ambient_temperature: PrescribedValue::Constant(Temperature::new(
                            equilibrium_k,
                        )),
                        coefficient: HeatTransferCoefficient::new(18.0),
                    },
                });
            } else {
                model
                    .constraints
                    .push(AnalysisConstraint::Thermal(ThermalConstraint {
                        constraint_id: "increment_fixed".into(),
                        kind: ThermalConstraintKind::Prescribed {
                            target: RegionTarget::boundary(
                                runmat_meshing_core::fixtures::fixture_boundary_selection(
                                    &(selector),
                                ),
                            ),
                            value: PrescribedValue::Constant(Temperature::new(equilibrium_k)),
                        },
                    }));
            }
            let assembly = energy_assembly(&model, order, 0.0);
            let reference = assembly
                .steady_reference_temperatures(&model, ExcitationSamplePoint::Static, 293.15)
                .unwrap();
            assert!(reference.iter().all(|&value| value == equilibrium_k));
            let operator = assembly
                .steady_increment_operator(&model, ExcitationSamplePoint::Static, &reference)
                .unwrap();
            assert!(operator.load().iter().all(|&value| value == 0.0));
            let result = solve_operator_system(
                &operator,
                SpdPreconditionerKind::Jacobi,
                LinearAlgebraBackendKind::CpuReference,
                &CpuReferenceBackend,
                None,
            );
            assert!(result.converged);
            assert!(result.solution.iter().all(|&value| value == 0.0));
            let temperature = result
                .solution
                .iter()
                .zip(&reference)
                .map(|(value, reference)| value + reference)
                .collect::<Vec<_>>();
            let energy = assembly
                .energy_balance(
                    &model,
                    ExcitationSamplePoint::Static,
                    &temperature,
                    &temperature,
                    1.0,
                    0.0,
                )
                .unwrap();
            assert_eq!(energy.relative_residual, 0.0);
            let heated = energy_assembly(&model, order, 1.0e-30);
            let heated_operator = heated
                .steady_increment_operator(&model, ExcitationSamplePoint::Static, &reference)
                .unwrap();
            assert!(heated_operator.load().iter().any(|&value| value != 0.0));
        }
    }
}
