//! CPU physical solve and durable result history, not worker/restart evidence.
use super::super::storage::{AnalysisArtifactStore, FilesystemAnalysisArtifactStore};
use super::*;
use runmat_analysis_core::{
    AnalysisConstraint, Displacement, MaterialPlasticModel, PrescribedStructuralDofs,
    PrescribedValue, StructuralConstraint, StructuralConstraintKind, StructuralTranslationDof,
    Traction, TranslationDofValue,
};
use runmat_types::physical::Quantity;

fn fixture(order: ElementOrder, root: &Path) -> (AnalysisModel, SolverMeshArtifact) {
    // Exercise dimensional authored material lowering; the model/mesh below is a typed fixture.
    let program = source::check_fea_source_in_root(
        root,
        &root.join("material.fea"),
        br#"material metal {
    mass_density = 1 kg/m^3
    elastic { youngs_modulus = 1200 Pa
              poisson_ratio = 0.25 }
    plastic { law = "j2_isotropic_linear_hardening"
              yield_stress = 5 Pa
              isotropic_hardening_modulus = 30 Pa }
}"#
        .to_vec(),
    )
    .unwrap();
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let material = program.materials.values().next().unwrap().clone();
    assert_eq!(
        material.plastic,
        Some(MaterialPlasticModel::J2IsotropicLinearHardening {
            yield_stress_pa: Quantity::new(5.0),
            isotropic_hardening_modulus_pa: Quantity::new(30.0),
        })
    );
    let mut model = sample_solid_model();
    model.material_assignments[0].expected_material_id = material.material_id.clone();
    model.material_assignments[0].assigned_material_id = material.material_id.clone();
    model.materials = vec![material];
    model.constraints.clear();
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.sequence =
        AnalysisSequence::from_steps("j2", vec![ProcedureStep::structural_nonlinear("j2")]);
    let mut mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
    // Coordinate-plane normal anchors allow Poisson contraction; never fix an entire face.
    for (axis, dof) in [
        StructuralTranslationDof::Ux,
        StructuralTranslationDof::Uy,
        StructuralTranslationDof::Uz,
    ]
    .into_iter()
    .enumerate()
    {
        let region = format!("anchor_{axis}");
        let topo = &mut *mesh.topology;
        let face = topo
            .boundary_faces
            .iter_mut()
            .find(|face| {
                face.node_ids.iter().all(|id| {
                    topo.nodes
                        .iter()
                        .find(|n| n.node_id == *id)
                        .unwrap()
                        .coordinates_m
                        .canonical_value()[axis]
                        == 0.0
                })
            })
            .unwrap();
        face.provenance.push(persistent_face(&region));
        model
            .constraints
            .push(AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: region.clone(),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(&(region)),
                    ),
                    translations: vec![TranslationDofValue {
                        dof,
                        value: PrescribedValue::Constant(Displacement::new(0.0)),
                    }],
                    rotations: Vec::new(),
                }),
            }));
    }
    let topo = &mut *mesh.topology;
    let face = topo
        .boundary_faces
        .iter_mut()
        .find(|face| {
            face.node_ids.iter().all(|id| {
                let x = topo
                    .nodes
                    .iter()
                    .find(|n| n.node_id == *id)
                    .unwrap()
                    .coordinates_m
                    .canonical_value();
                (x.iter().sum::<f64>() - 1.0).abs() < 1e-14
            })
        })
        .unwrap();
    face.provenance.push(persistent_face("loaded"));
    for face in &mut mesh.topology.boundary_faces {
        face.provenance.sort();
    }
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    // Inclined-face n_x=1/sqrt(3); t_x=sigma_xx*n_x. Other faces have zero free traction.
    let id = SignalId("traction.signal".into());
    model.signals.push(PhysicalSignal::Traction(Signal {
        signal_id: id.clone(),
        shape: SignalShape::Vector3,
        frame: Some(CoordinateFrame::Global),
        representation: SignalRepresentation::Constant(SignalValue::Vector3([
            Traction::new(5.6 / 3.0_f64.sqrt()),
            Traction::new(0.0),
            Traction::new(0.0),
        ])),
        precision: SignalPrecision::Real64,
        provenance: SignalProvenance::Authored,
        evidence: SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
    }));
    model
        .excitations
        .push(ExcitationAction::Traction(AnalysisExcitation {
            excitation_id: ExcitationId("traction".into()),
            target: ExcitationTarget::Region(RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_face("loaded"),
            )),
            frame: CoordinateFrame::Global,
            direction: None,
            distribution: ExcitationDistribution::IntensityPerArea,
            kinematics: ExcitationKinematics::FixedFrame,
            signal_id: id,
            active_step_ids: Vec::new(),
            combination_group: None,
            projection: ProjectionRequirement {
                preserve_resultant: false,
                preserve_moment: false,
                relative_tolerance: 1e-10,
            },
        }));
    model.excitation_cases.push(ExcitationCase {
        case_id: ExcitationCaseId("traction.case".into()),
        action_ids: vec![ExcitationId("traction".into())],
        activation: ExcitationCaseActivation::Always,
    });
    (model, mesh)
}

/// A real Tet4/J2 seed for artifact-store tests, which do not exercise the
/// retired mesh-free beam nonlinear proxy. The caller already owns/cleans root.
pub(super) fn artifact_seed(root: &Path) -> (AnalysisModel, AnalysisNonlinearRunOptions) {
    fs::create_dir_all(root).expect("create physical nonlinear seed root");
    let (model, mesh) = fixture(ElementOrder::Tet4, root);
    let path = root.join("physical_nonlinear_seed.cbor");
    fs::write(
        &path,
        mesh.canonical_encode().expect("encode physical seed mesh"),
    )
    .expect("write physical seed mesh");
    (
        model,
        AnalysisNonlinearRunOptions {
            solver_mesh_artifact_path: Some(path.to_string_lossy().into_owned()),
            increment_count: 4,
            tolerance: 1e-9,
            linear_tolerance: 1e-12,
            ..Default::default()
        },
    )
}

fn values(field: &AnalysisField) -> &[f64] {
    field.as_host_f64().unwrap()
}

#[test]
fn physical_j2_tet4_tet10_history_survives_cold_filesystem_and_query() {
    let _serial = analysis_test_guard();
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let root = temp_artifact_root(&format!("physical_j2_{order:?}"));
        fs::create_dir_all(&root).unwrap();
        let (model, mesh) = fixture(order, &root);
        let (root, path) =
            write_ready_analysis_mesh_artifact(&format!("physical_j2_run_{order:?}"), &mesh);
        let _config = scoped_fea_artifact_roots(&root);
        storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
            root: root.clone(),
        })
        .expect("configure durable physical nonlinear run store");
        let result = analysis_run_nonlinear_with_options_op(
            &model,
            ComputeBackend::Cpu,
            AnalysisNonlinearRunOptions {
                solver_mesh_artifact_path: Some(path.to_string_lossy().into_owned()),
                increment_count: 4,
                tolerance: 1e-9,
                linear_tolerance: 1e-12,
                ..AnalysisNonlinearRunOptions::default()
            },
            OperationContext::new(None, None),
        )
        .unwrap()
        .data;
        let history = result.nonlinear_results.as_ref().unwrap();
        assert_eq!(history.load_factors.last(), Some(&1.0));
        assert!(history
            .load_factors
            .windows(2)
            .all(|pair| pair[1] > pair[0]));
        assert_eq!(
            history.displacement_snapshots.len(),
            history.load_factors.len()
        );
        assert!(history
            .residual_norms
            .iter()
            .all(|r| r.is_finite() && *r <= 1e-9));
        // Independent uniaxial J2: alpha=(sigma-yield)/H; epsilon_x=sigma/E+alpha.
        let alpha = (5.6 - 5.0) / 30.0;
        let strain = [
            5.6 / 1200.0 + alpha,
            -0.25 * 5.6 / 1200.0 - alpha / 2.0,
            -0.25 * 5.6 / 1200.0 - alpha / 2.0,
        ];
        let expected = mesh
            .topology
            .nodes
            .iter()
            .flat_map(|node| {
                node.coordinates_m
                    .canonical_value()
                    .into_iter()
                    .zip(strain)
                    .map(|(x, e)| x * e)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            values(history.displacement_snapshots.last().unwrap()).len(),
            expected.len()
        );
        for (actual, expected) in values(history.displacement_snapshots.last().unwrap())
            .iter()
            .zip(expected)
        {
            assert!((actual - expected).abs() < 2e-8);
        }
        for actual in values(history.von_mises_snapshots.last().unwrap()) {
            assert!((*actual - 5.6).abs() < 2e-8);
        }
        for actual in values(history.equivalent_plastic_strain_snapshots.last().unwrap()) {
            assert!((*actual - alpha).abs() < 2e-8);
        }
        let expected_plastic = [alpha, -alpha / 2.0, -alpha / 2.0, 0.0, 0.0, 0.0];
        for chunk in values(history.plastic_strain_snapshots.last().unwrap()).chunks_exact(6) {
            for (actual, expected) in chunk.iter().zip(expected_plastic) {
                assert!((actual - expected).abs() < 2e-8);
            }
        }
        assert!(history
            .contact_pressure_snapshots
            .iter()
            .all(|f| values(f).is_empty()));
        assert!(history.thermo_mechanical_temperature_snapshots.is_empty());
        for id in [
            FEA_FIELD_STRUCTURAL_STRESS,
            FEA_FIELD_STRUCTURAL_REACTION_FORCE,
        ] {
            assert!(result.run.fields.iter().any(|field| field.field_id == id));
        }
        // Fresh store has no manifest cache; both history and immutable field chunks must decode.
        let cold = FilesystemAnalysisArtifactStore::new(root.clone())
            .load_run(&result.run_id)
            .unwrap()
            .unwrap();
        assert_eq!(cold.nonlinear_results, result.nonlinear_results);
        assert_eq!(cold.run.fields, result.run.fields);
        let query = analysis_results_by_run_id_op(
            &result.run_id,
            AnalysisResultsQuery {
                include_field_values: true,
                include_nonlinear_results: true,
                ..AnalysisResultsQuery::default()
            },
            OperationContext::new(None, None),
        )
        .unwrap()
        .data;
        assert_eq!(query.nonlinear_results, result.nonlinear_results);
        let hidden = analysis_results_by_run_id_op(
            &result.run_id,
            AnalysisResultsQuery {
                include_field_values: true,
                include_nonlinear_results: false,
                ..AnalysisResultsQuery::default()
            },
            OperationContext::new(None, None),
        )
        .unwrap()
        .data;
        assert!(hidden.nonlinear_results.is_none());
        assert_eq!(hidden.fields, query.fields);
    }
}
