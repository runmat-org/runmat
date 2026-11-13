use super::*;

#[test]
fn curved_tri6_normal_displacement_uses_local_midside_tangent() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    // Face 0 is the z=0 triangle. Bowing edge 1-2 toward positive z by h
    // yields the exact area vector (0, 2h, 1) at its midside node. The
    // outward normal is its negative; the corner plane alone remains z=0.
    let bow_m = 0.1;
    {
        let quantity_change = bow_m;
        let quantity_index = 2;
        let quantity_slot = &mut mesh.topology.nodes[4].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] = quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    model.geometry_revision = mesh.geometry.geometry_revision as u32;
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "curved_normal".into(),
        frame: CoordinateFrame::BoundaryNormalTangent,
        kind: StructuralConstraintKind::NormalDisplacement {
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("face:0")),
            value: PrescribedValue::Constant(Displacement::new(0.002)),
        },
    })];

    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    assert_eq!(plan.rows.len(), 6);
    let midside = mesh.topology.nodes[4].stable_identity;
    let row = plan
        .rows
        .iter()
        .find(|row| row.terms[0].owner == CanonicalStructuralDofOwner::StructuralNode(midside))
        .unwrap();
    let local = row_vector(row);
    let norm = (1.0 + 4.0 * bow_m * bow_m).sqrt();
    for (actual, expected) in local
        .into_iter()
        .zip([0.0, -2.0 * bow_m / norm, -1.0 / norm])
    {
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "actual={actual} expected={expected}"
        );
    }
    assert_eq!(row.right_hand_side, 0.002);

    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "curved_symmetry".into(),
        frame: CoordinateFrame::BoundaryNormalTangent,
        kind: StructuralConstraintKind::Symmetry {
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("face:0")),
        },
    })];
    let symmetry = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let symmetry_row = symmetry
        .rows
        .iter()
        .find(|row| row.terms[0].owner == CanonicalStructuralDofOwner::StructuralNode(midside))
        .unwrap();
    for (actual, expected) in row_vector(symmetry_row).into_iter().zip(local) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }

    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "curved_antisymmetry".into(),
        frame: CoordinateFrame::BoundaryNormalTangent,
        kind: StructuralConstraintKind::Antisymmetry {
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("face:0")),
        },
    })];
    let antisymmetry = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let tangents = antisymmetry
        .rows
        .iter()
        .filter(|row| row.terms[0].owner == CanonicalStructuralDofOwner::StructuralNode(midside))
        .collect::<Vec<_>>();
    assert_eq!(tangents.len(), 2);
    for tangent in tangents {
        let dot = row_vector(tangent)
            .into_iter()
            .zip(local)
            .map(|(component, normal)| component * normal)
            .sum::<f64>();
        assert!(dot.abs() <= 1.0e-12);
    }
}

#[test]
fn curved_tri6_normal_support_solves_tangential_traction_and_recovers_normal_reaction() {
    let warp_m = 0.1_f64;
    let youngs_pa = 100.0;
    let poisson = 0.25;
    let tangent_traction_pa = 1.0;
    let length = (1.0 + 4.0 * warp_m * warp_m).sqrt();
    let tangent = [0.0, 1.0 / length, -2.0 * warp_m / length];
    let normal = [0.0, -2.0 * warp_m / length, -1.0 / length];
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    {
        let quantity_change = warp_m;
        let quantity_index = 2;
        let quantity_slot = &mut mesh.topology.nodes[4].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] = quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let runmat_analysis_core::MaterialElasticityModel::IsotropicLinear(elasticity) =
        model.materials[0].elasticity.as_mut().unwrap()
    else {
        panic!("fixture requires isotropic elasticity");
    };
    elasticity.youngs_modulus_pa = runmat_types::physical::YoungsModulus::new(youngs_pa);
    elasticity.poisson_ratio = runmat_types::physical::PoissonRatio::new(poisson);
    model.constraints = mesh
        .topology
        .nodes
        .iter()
        .map(|node| {
            let dofs = if node.node_id == 5 {
                vec![StructuralTranslationDof::Ux]
            } else {
                vec![
                    StructuralTranslationDof::Ux,
                    StructuralTranslationDof::Uy,
                    StructuralTranslationDof::Uz,
                ]
            };
            prescribed_constraint(&format!("anchor_{}", node.node_id), node.node_id, &dofs)
        })
        .collect();
    model
        .constraints
        .push(AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "curved_face_normal".into(),
            frame: CoordinateFrame::BoundaryNormalTangent,
            kind: StructuralConstraintKind::NormalDisplacement {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "face:0",
                )),
                value: PrescribedValue::Constant(Displacement::new(0.0)),
            },
        }));
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "curved_face_tangent_traction".into(),
            region_id: "face:0".into(),
            kind: FixtureExcitationKind::Traction(
                tangent.map(|component| tangent_traction_pa * component),
            ),
        }],
    );

    let summary = assemble_structural_system(
        &model,
        &static_excitations(&model),
        Some(mesh.clone()),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    let solve = solve_linear_system(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    )
    .unwrap();
    assert!(solve.converged, "{:?}", solve.diagnostics);
    let (tangent_stiffness, normal_tangent_stiffness) =
        curved_midside_stiffness_reference(warp_m, youngs_pa, poisson);
    let (face_area_m2, midside_shape_area_m2) = curved_face_traction_reference(warp_m);
    let tangent_force_n = tangent_traction_pa * midside_shape_area_m2;
    let amplitude = tangent_force_n / tangent_stiffness;
    for (index, actual) in solve.solution.iter().enumerate() {
        let expected = if (12..15).contains(&index) {
            tangent[index - 12] * amplitude
        } else {
            0.0
        };
        assert!(
            (actual - expected).abs() <= amplitude * 1.0e-7,
            "dof={index} actual={actual} expected={expected} projected_force={} reference_force={tangent_force_n} reference_stiffness={tangent_stiffness}",
            summary.physical_load[13] / tangent[1],
        );
    }

    let fields = recover_result_fields(
        &summary,
        &solve,
        crate::planning::resource::LocalExecutionPlan::serial(),
    )
    .unwrap();
    let energy = fields
        .iter()
        .find(|field| field.field_id == crate::FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
        .unwrap()
        .as_host_f64()
        .unwrap()[0];
    assert!((energy - 0.5 * tangent_force_n * amplitude).abs() <= energy * 1.0e-7);
    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let result = CanonicalStructuralConstraintResult::recover(&plan, &mesh, &fields).unwrap();
    let face = result
        .groups
        .iter()
        .find(|group| group.constraint_id == "curved_face_normal")
        .unwrap();
    assert_eq!(face.active_row_count, 1);
    assert_eq!(face.redundant_row_count, 5);
    assert!(face.maximum_displacement_error_meters <= 1.0e-10);
    let expected_normal_reaction = normal_tangent_stiffness * amplitude;
    for (actual, direction) in face.resultant_force_newtons.iter().zip(normal) {
        assert!(
            (actual - direction * expected_normal_reaction).abs()
                <= expected_normal_reaction.abs() * 1.0e-7,
            "actual={actual} expected={}",
            direction * expected_normal_reaction
        );
    }
    assert_eq!(face.constraint_work_joules, 0.0);
    for (reaction, applied) in result
        .resultant_force_newtons
        .iter()
        .zip(tangent.map(|component| tangent_traction_pa * face_area_m2 * component))
    {
        assert!((reaction + applied).abs() <= tangent_traction_pa * face_area_m2 * 1.0e-7);
    }
}

#[test]
fn curved_tri6_foundation_adds_independent_normal_stiffness_and_recovered_energy() {
    let warp_m = 0.1_f64;
    let youngs_pa = 100.0;
    let poisson = 0.25;
    let modulus = 10_000.0;
    let length = (1.0 + 4.0 * warp_m * warp_m).sqrt();
    let tangent = [0.0, 1.0 / length, -2.0 * warp_m / length];
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    {
        let quantity_change = warp_m;
        let quantity_index = 2;
        let quantity_slot = &mut mesh.topology.nodes[4].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] = quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let runmat_analysis_core::MaterialElasticityModel::IsotropicLinear(elasticity) =
        model.materials[0].elasticity.as_mut().unwrap()
    else {
        panic!("fixture requires isotropic elasticity");
    };
    elasticity.youngs_modulus_pa = runmat_types::physical::YoungsModulus::new(youngs_pa);
    elasticity.poisson_ratio = runmat_types::physical::PoissonRatio::new(poisson);
    model.constraints = mesh
        .topology
        .nodes
        .iter()
        .map(|node| {
            let dofs = if node.node_id == 5 {
                vec![StructuralTranslationDof::Ux]
            } else {
                vec![
                    StructuralTranslationDof::Ux,
                    StructuralTranslationDof::Uy,
                    StructuralTranslationDof::Uz,
                ]
            };
            prescribed_constraint(&format!("anchor_{}", node.node_id), node.node_id, &dofs)
        })
        .collect();
    model
        .constraints
        .push(AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "curved_face_normal".into(),
            frame: CoordinateFrame::BoundaryNormalTangent,
            kind: StructuralConstraintKind::NormalDisplacement {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "face:0",
                )),
                value: PrescribedValue::Constant(Displacement::new(0.0)),
            },
        }));
    model
        .connectors
        .push(runmat_analysis_core::AnalysisConnector {
            connector_id: "curved_foundation".into(),
            kind: runmat_analysis_core::ConnectorKind::SurfaceNormalFoundation {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "face:0",
                )),
                modulus: runmat_analysis_core::FoundationModulus::new(modulus),
            },
        });
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "curved_face_tangent_traction".into(),
            region_id: "face:0".into(),
            kind: FixtureExcitationKind::Traction(tangent),
        }],
    );

    let summary = assemble_structural_system(
        &model,
        &static_excitations(&model),
        Some(mesh),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    let solve = solve_linear_system(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    )
    .unwrap();
    assert!(solve.converged, "{:?}", solve.diagnostics);
    let (solid_stiffness, _) = curved_midside_stiffness_reference(warp_m, youngs_pa, poisson);
    let (_, midside_shape_area_m2) = curved_face_traction_reference(warp_m);
    let (foundation_stiffness, foundation_resultant_per_meter) =
        curved_foundation_reference(warp_m, modulus);
    let amplitude = midside_shape_area_m2 / (solid_stiffness + foundation_stiffness);
    for (index, actual) in solve.solution.iter().enumerate() {
        let expected = if (12..15).contains(&index) {
            tangent[index - 12] * amplitude
        } else {
            0.0
        };
        assert!(
            (actual - expected).abs() <= amplitude * 1.0e-7,
            "dof={index} actual={actual} expected={expected}"
        );
    }
    let connector = crate::connectors::CanonicalStructuralConnectorResult::recover_static(
        summary.structural_connector_plan.clone().unwrap(),
        &summary.structural_dof_layout,
        &solve.solution,
    )
    .unwrap();
    assert_eq!(connector.responses.len(), 1);
    let response = &connector.responses[0];
    assert_eq!(response.connector_id(), "curved_foundation");
    let expected_energy = 0.5 * foundation_stiffness * amplitude * amplitude;
    assert!((response.stored_energy_joules() - expected_energy).abs() <= expected_energy * 1.0e-7);
    for (actual, expected_per_meter) in response
        .resultant_force_newtons()
        .iter()
        .zip(foundation_resultant_per_meter)
    {
        let expected = expected_per_meter * amplitude;
        assert!((actual - expected).abs() <= modulus * amplitude * 1.0e-7);
    }
    let fields = recover_result_fields(
        &summary,
        &solve,
        crate::planning::resource::LocalExecutionPlan::serial(),
    )
    .unwrap();
    let total_energy = fields
        .iter()
        .find(|field| field.field_id == crate::FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
        .unwrap()
        .as_host_f64()
        .unwrap()[0];
    let expected_total_energy =
        0.5 * (solid_stiffness + foundation_stiffness) * amplitude * amplitude;
    assert!((total_energy - expected_total_energy).abs() <= expected_total_energy * 1.0e-7);
}

fn curved_midside_stiffness_reference(warp: f64, youngs: f64, poisson: f64) -> (f64, f64) {
    // N_5=4x(1-x-y-z), z_physical=z+warp*N_5. Thus det J=1-4*warp*x
    // and grad_x N_5=(grad_reference N_5)/det J. Analytic integration over
    // y,z leaves a smooth one-dimensional rational integral, evaluated here
    // by an independent composite Simpson rule rather than solver quadrature.
    let mu = youngs / (2.0 * (1.0 + poisson));
    let lambda = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    let integrand = |x: f64| {
        let span = 1.0 - x;
        let first = 1.0 - 2.0 * x;
        let gradient_squared = 16.0
            * (first * first * span * span / 2.0 - 2.0 * first * span.powi(3) / 3.0
                + span.powi(4) / 4.0
                + x * x * span * span);
        let tangent_dot_squared =
            8.0 * x * x * span * span * (1.0 - 2.0 * warp).powi(2) / (1.0 + 4.0 * warp * warp);
        let normal_tangent_dot =
            -8.0 * x * x * span * span * (1.0 - 4.0 * warp * warp) / (1.0 + 4.0 * warp * warp);
        let determinant = 1.0 - 4.0 * warp * x;
        (
            (mu * gradient_squared + (lambda + mu) * tangent_dot_squared) / determinant,
            (lambda + mu) * normal_tangent_dot / determinant,
        )
    };
    let intervals = 16_384;
    let width = 1.0 / intervals as f64;
    let mut tangent = 0.0;
    let mut normal_tangent = 0.0;
    for index in 0..=intervals {
        let weight = simpson_weight(index, intervals);
        let (along, cross) = integrand(index as f64 * width);
        tangent += weight * along;
        normal_tangent += weight * cross;
    }
    (tangent * width / 3.0, normal_tangent * width / 3.0)
}

fn curved_face_traction_reference(warp: f64) -> (f64, f64) {
    // On the bowed face z=4h*x*(1-x-y). Composite Simpson integration of
    // area and N_5-weighted area independently predicts the total traction
    // and its consistent midside nodal contribution.
    let intervals = 512;
    let width = 1.0 / intervals as f64;
    let mut area = 0.0;
    let mut midside_shape_area = 0.0;
    for r_index in 0..=intervals {
        let x = r_index as f64 * width;
        let r_weight = simpson_weight(r_index, intervals);
        for s_index in 0..=intervals {
            let y = (1.0 - x) * s_index as f64 * width;
            let s_weight = simpson_weight(s_index, intervals);
            let shape = 4.0 * x * (1.0 - x - y);
            let slope_x = 4.0 * warp * (1.0 - 2.0 * x - y);
            let slope_y = -4.0 * warp * x;
            let weighted_area = r_weight
                * s_weight
                * (1.0 - x)
                * (1.0 + slope_x * slope_x + slope_y * slope_y).sqrt();
            area += weighted_area;
            midside_shape_area += weighted_area * shape;
        }
    }
    (
        area * width * width / 9.0,
        midside_shape_area * width * width / 9.0,
    )
}

fn curved_foundation_reference(warp: f64, modulus: f64) -> (f64, [f64; 3]) {
    // The outward area vector on the bowed z=0 face is
    // (dz/dx,dz/dy,-1) dxdy. A one-node tangent displacement is N_5*q*t;
    // the foundation stiffness and net reaction are independent surface
    // integrals of N_5^2(t·n)^2 and -N_5(t·n)n, respectively.
    let intervals = 512;
    let width = 1.0 / intervals as f64;
    let tangent_length = (1.0 + 4.0 * warp * warp).sqrt();
    let mut stiffness = 0.0;
    let mut resultant = [0.0; 3];
    for r_index in 0..=intervals {
        let x = r_index as f64 * width;
        let r_weight = simpson_weight(r_index, intervals);
        for s_index in 0..=intervals {
            let y = (1.0 - x) * s_index as f64 * width;
            let weight = r_weight * simpson_weight(s_index, intervals) * (1.0 - x);
            let shape = 4.0 * x * (1.0 - x - y);
            let slope_x = 4.0 * warp * (1.0 - 2.0 * x - y);
            let slope_y = -4.0 * warp * x;
            let jacobian = (1.0 + slope_x * slope_x + slope_y * slope_y).sqrt();
            let tangent_dot_normal = (slope_y + 2.0 * warp) / (tangent_length * jacobian);
            stiffness += weight * shape * shape * tangent_dot_normal.powi(2) * jacobian;
            for (component, normal_area) in resultant.iter_mut().zip([slope_x, slope_y, -1.0]) {
                *component -= weight * shape * tangent_dot_normal * normal_area;
            }
        }
    }
    (
        modulus * stiffness * width * width / 9.0,
        resultant.map(|value| modulus * value * width * width / 9.0),
    )
}

fn simpson_weight(index: usize, intervals: usize) -> f64 {
    if index == 0 || index == intervals {
        1.0
    } else if index.is_multiple_of(2) {
        2.0
    } else {
        4.0
    }
}
