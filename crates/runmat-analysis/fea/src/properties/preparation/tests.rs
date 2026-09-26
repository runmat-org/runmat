use super::*;
use crate::fixtures::{fixture_model, FixtureId};
use runmat_analysis_core::{
    LaminatePlyModel, LaminateShellSectionModel, MaterialCapability, PlyOrderDirection,
    PropertyRegionTarget, ShellReferenceSurface, ShellSectionModel,
};
use runmat_meshing_core::fixtures::{
    canonical_tetrahedron_solver_mesh, three_region_tetrahedron_solver_mesh_with_order,
};
use runmat_meshing_core::ElementOrder;

fn solid_model(mesh: &SolverMeshArtifact) -> AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.geometry_revision = u32::try_from(mesh.geometry.geometry_revision).unwrap();
    model.material_assignments[0].target = PropertyRegionTarget::all_solids();
    model
}

fn assert_profile_envelope(compiled: &CompiledPropertyMap, estimated_resident_bytes: u64) {
    let owned_profile = serialized_materialization_resident_bytes_estimate(
        compiled,
        CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64),
    )
    .unwrap();
    // Comparison of shared-codec projections, not an allocator-exact peak measurement.
    assert!(estimated_resident_bytes >= owned_profile);
}

#[test]
fn exact_admission_boundary_preserves_owned_properties_and_identities() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let model = solid_model(&mesh);
        let prepared = PreparedProperties::prepare(&model, Some(&mesh), u64::MAX).unwrap();
        let required = prepared.estimated_resident_bytes;
        let expected = prepared.materialize().unwrap();
        assert_profile_envelope(&expected, required);
        assert!(matches!(
            PreparedProperties::prepare(&model, Some(&mesh), required - 1),
            Err(PropertyCompilationError::ResourceCeiling {
                required: actual,
                maximum,
            }) if actual == required && maximum == required - 1
        ));
        let actual = PreparedProperties::prepare(&model, Some(&mesh), required)
            .unwrap()
            .materialize()
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(
            serde_json::to_vec(&actual).unwrap(),
            serde_json::to_vec(&expected).unwrap()
        );
        for element in &mesh.topology.volume_elements {
            assert_eq!(
                actual
                    .solid_material(element.stable_identity)
                    .unwrap()
                    .material,
                model.materials[0]
            );
        }
        super::super::identity::tests::assert_legacy_identity(&actual);
    }
}

#[test]
fn lookup_admission_precedes_mesh_validation() {
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let model = solid_model(&mesh);
    mesh.canonical_digest = StableDigest::ZERO;
    assert!(matches!(
        PreparedProperties::prepare(&model, Some(&mesh), 0),
        Err(PropertyCompilationError::ResourceCeiling {
            required,
            maximum: 0,
        }) if required > 0
    ));
    assert!(matches!(
        PreparedProperties::prepare(&model, Some(&mesh), u64::MAX),
        Err(PropertyCompilationError::InvalidMesh(_))
    ));
}

#[test]
fn mesh_validation_scratch_is_admitted_before_canonical_validation() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut mesh = canonical_tetrahedron_solver_mesh(order);
        let model = solid_model(&mesh);
        let validation = mesh.validation_resident_bytes_estimate().unwrap();
        let required = selection_scratch(&model).unwrap() + validation;
        // A stale seal is detected only by validation, after resource admission succeeds.
        mesh.canonical_digest = StableDigest::ZERO;
        assert!(matches!(
            PreparedProperties::prepare(&model, Some(&mesh), required - 1),
            Err(PropertyCompilationError::ResourceCeiling { required: actual, maximum })
                if actual == required && maximum == required - 1
        ));
        assert!(matches!(
            PreparedProperties::prepare(&model, Some(&mesh), required),
            Err(PropertyCompilationError::InvalidMesh(_))
        ));
    }
}

fn heterogeneous_model(mesh: &SolverMeshArtifact, large_occurrences: usize) -> AnalysisModel {
    let mut model = solid_model(mesh);
    let mut small = model.materials[0].clone();
    small.material_id = "mat_small".into();
    small.name = "m".into();
    let mut large = small.clone();
    large.material_id = "mat_large".into();
    model.materials = vec![small, large];
    let assignment = model.material_assignments[0].clone();
    model.material_assignments = mesh
        .topology
        .volume_elements
        .iter()
        .enumerate()
        .map(|(index, element)| {
            let mut selected = assignment.clone();
            selected.target =
                PropertyRegionTarget::solid(element.region_id.source_topology_id.clone());
            let id = if index < large_occurrences {
                "mat_large"
            } else {
                "mat_small"
            };
            selected.expected_material_id = id.into();
            selected.assigned_material_id = id.into();
            selected
        })
        .collect();
    model
}

#[test]
fn selected_metadata_scales_with_actual_material_occurrences() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = three_region_tetrahedron_solver_mesh_with_order(order);
        assert_eq!(mesh.topology.volume_elements.len(), 3);
        let mut deltas = Vec::new();
        for occurrences in [1, 3] {
            let mut model = heterogeneous_model(&mesh, occurrences);
            let baseline = PreparedProperties::prepare(&model, Some(&mesh), u64::MAX)
                .unwrap()
                .selected_metadata_bytes;
            model.materials[1].name = "x".repeat(1025);
            let grown = PreparedProperties::prepare(&model, Some(&mesh), u64::MAX).unwrap();
            deltas.push(grown.selected_metadata_bytes - baseline);
            let required = grown.estimated_resident_bytes;
            let compiled = grown.materialize().unwrap();
            assert_profile_envelope(&compiled, required);
            for (index, element) in mesh.topology.volume_elements.iter().enumerate() {
                let expected = if index < occurrences {
                    "mat_large"
                } else {
                    "mat_small"
                };
                assert_eq!(
                    compiled
                        .solid_material(element.stable_identity)
                        .unwrap()
                        .material
                        .material_id,
                    expected
                );
            }
            super::super::identity::tests::assert_legacy_identity(&compiled);
        }
        assert!(deltas[0] > 0);
        assert_eq!(deltas[1], 3 * deltas[0]);
    }
}

#[test]
fn unused_large_material_does_not_expand_selected_metadata() {
    let mesh = three_region_tetrahedron_solver_mesh_with_order(ElementOrder::Tet4);
    let mut model = heterogeneous_model(&mesh, 1);
    let base = PreparedProperties::prepare(&model, Some(&mesh), u64::MAX).unwrap();
    let base_metadata = base.selected_metadata_bytes;
    let base_resident = base.estimated_resident_bytes;
    let base_owned = base.materialize().unwrap();
    let mut unused = model.materials[0].clone();
    unused.material_id = "mat_unused".into();
    unused.name = "unused_material_metadata".repeat(2048);
    model.materials.push(unused);
    let grown = PreparedProperties::prepare(&model, Some(&mesh), u64::MAX).unwrap();
    assert_eq!(grown.selected_metadata_bytes, base_metadata);
    assert!(grown.estimated_resident_bytes > base_resident); // Inventory slot, not material clones.
    assert_eq!(grown.materialize().unwrap(), base_owned);
}

#[test]
fn preparation_preserves_typed_ownership_failures() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let model = solid_model(&mesh);
    let mut gap = model.clone();
    gap.material_assignments.clear();
    assert!(matches!(
        PreparedProperties::prepare(&gap, Some(&mesh), u64::MAX),
        Err(PropertyCompilationError::InvalidCoverage {
            role: "solid",
            assignment_count: 0,
            ..
        })
    ));
    let mut overlap = model.clone();
    overlap.material_assignments = vec![model.material_assignments[0].clone(); 3];
    assert!(matches!(
        PreparedProperties::prepare(&overlap, Some(&mesh), u64::MAX),
        Err(PropertyCompilationError::InvalidCoverage {
            role: "solid",
            assignment_count: 3,
            ..
        })
    ));
    let mut missing = model;
    missing.material_assignments[0].assigned_material_id = "absent".into();
    assert!(matches!(
        PreparedProperties::prepare(&missing, Some(&mesh), u64::MAX),
        Err(PropertyCompilationError::UnknownAssignedMaterial { assignment_index: 0, material_id })
            if material_id == "absent"
    ));
}

fn laminate_model() -> AnalysisModel {
    let mut model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
    model.material_assignments.clear();
    let mut upper = model.materials[0].clone();
    upper.material_id = "mat_upper".into();
    model.materials.push(upper);
    for material in &mut model.materials {
        material.name = "m".into();
    }
    model.structural.as_mut().unwrap().shell_sections =
        vec![ShellSectionModel::Laminate(LaminateShellSectionModel {
            section_id: "panel_2mm".into(),
            reference_orientation_id: "global_x".into(),
            ply_order: PlyOrderDirection::AlongSurfaceNormal,
            reference_surface: ShellReferenceSurface::MidSurface,
            plies: [
                ("lower", "mat_steel"),
                ("middle", "mat_steel"),
                ("upper", "mat_upper"),
            ]
            .map(|(ply_id, material_id)| LaminatePlyModel {
                ply_id: ply_id.into(),
                material_id: material_id.into(),
                thickness_m: 0.001,
                angle_rad: 0.0,
            })
            .to_vec(),
            shear_correction: Some(5.0 / 6.0),
        })];
    model
}

#[test]
fn explicit_profile_envelopes_cover_beam_shell_and_repeated_laminate_materials() {
    for fixture in [
        FixtureId::StructuralBeamCantileverEndMomentReference,
        FixtureId::StructuralShellPlateMomentReference,
    ] {
        let model = fixture_model(fixture);
        let prepared = PreparedProperties::prepare(&model, None, u64::MAX).unwrap();
        let required = prepared.estimated_resident_bytes;
        let compiled = prepared.materialize().unwrap();
        assert_profile_envelope(&compiled, required);
        super::super::identity::tests::assert_legacy_identity(&compiled);
    }
    let model = laminate_model();
    let baseline = PreparedProperties::prepare(&model, None, u64::MAX)
        .unwrap()
        .selected_metadata_bytes;
    let mut deltas = Vec::new();
    for material_index in [0, 1] {
        let mut grown_model = model.clone();
        grown_model.materials[material_index].name = "x".repeat(1025);
        let grown = PreparedProperties::prepare(&grown_model, None, u64::MAX).unwrap();
        deltas.push(grown.selected_metadata_bytes - baseline);
        let required = grown.estimated_resident_bytes;
        let compiled = grown.materialize().unwrap();
        assert_profile_envelope(&compiled, required);
        super::super::identity::tests::assert_legacy_identity(&compiled);
    }
    assert!(deltas[1] > 0);
    assert_eq!(deltas[0], 2 * deltas[1]);
}

#[test]
fn invalid_last_laminate_ply_emits_no_selected_owner() {
    let valid = laminate_model();
    let compiled = PreparedProperties::prepare(&valid, None, u64::MAX)
        .unwrap()
        .materialize()
        .unwrap();
    super::super::identity::tests::assert_legacy_identity(&compiled);
    for missing in [false, true] {
        let mut model = valid.clone();
        if missing {
            model.materials.pop();
        } else {
            model.materials.last_mut().unwrap().elasticity = None;
        }
        let materials = material_inventory(&model).unwrap();
        let explicit =
            PreparedExplicitTopology::prepare(model.structural.as_ref().unwrap(), u64::MAX)
                .unwrap();
        let mut emitted = 0;
        let result = visit(&model, None, &materials, Some(&explicit), &mut |_| {
            emitted += 1;
            Ok(())
        });
        assert_eq!(emitted, 0);
        let expected = if missing {
            assert!(
                matches!(&result, Err(PropertyCompilationError::UnknownPlyMaterial { material_id, .. }) if material_id == "mat_upper")
            );
            result.err().unwrap()
        } else {
            assert!(matches!(
                &result,
                Err(PropertyCompilationError::MissingCapability {
                    capability: MaterialCapability::LaminaElasticity,
                    ..
                })
            ));
            result.err().unwrap()
        };
        assert!(matches!(
            PreparedProperties::prepare(&model, None, u64::MAX),
            Err(actual) if actual == expected
        ));
    }
}

#[test]
fn explicit_cache_exact_ceiling_preserves_identity_shape_and_bytes() {
    use crate::physics::structural::topology::{
        explicit_element_identity, explicit_identity_resource_bytes, explicit_topology_digest,
    };
    for fixture in [
        FixtureId::StructuralBeamCantileverEndMomentReference,
        FixtureId::StructuralShellPlateMomentReference,
    ] {
        let model = fixture_model(fixture);
        let structural = model.structural.as_ref().unwrap();
        let prepared = PreparedExplicitTopology::prepare(structural, u64::MAX).unwrap();
        let required = prepared.estimated_resident_bytes;
        assert_eq!(
            required,
            explicit_identity_resource_bytes(structural).unwrap()
                + (structural.elements.len() * std::mem::size_of::<StableDigest>()) as u64
        );
        assert!(matches!(
            PreparedExplicitTopology::prepare(structural, required - 1),
            Err(PropertyCompilationError::ResourceCeiling { required: actual, maximum })
                if actual == required && maximum == required - 1
        ));
        let accepted = PreparedExplicitTopology::prepare(structural, required).unwrap();
        assert!(std::ptr::eq(accepted.structural(), structural));
        assert_eq!(
            accepted.topology_digest(),
            explicit_topology_digest(structural).unwrap()
        );
        assert_eq!(
            accepted.element_identities().len(),
            structural.elements.len()
        );
        assert_eq!(accepted.element_identities(), prepared.element_identities());
        for (element, identity) in structural
            .elements
            .iter()
            .zip(accepted.element_identities())
        {
            assert_eq!(
                *identity,
                explicit_element_identity(
                    accepted.topology_digest(),
                    &element.element_id,
                    &element.region_id,
                    &element.kind,
                )
                .unwrap()
            );
        }
    }
}

#[test]
fn explicit_budget_rejects_before_invalid_coordinate_validation_and_includes_selection() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    model.structural.as_mut().unwrap().nodes[0].coordinates_m[0] = f64::NAN;
    let structural = model.structural.as_ref().unwrap();
    let explicit_required = match PreparedExplicitTopology::prepare(structural, 0) {
        Err(PropertyCompilationError::ResourceCeiling {
            required,
            maximum: 0,
        }) => required,
        _ => panic!("explicit identity admission must precede coordinate validation"),
    };
    assert!(explicit_required > 0);
    assert!(matches!(
        PreparedExplicitTopology::prepare(structural, u64::MAX),
        Err(PropertyCompilationError::InvalidExplicitTopology(_))
    ));
    let selection = selection_scratch(&model).unwrap();
    let maximum = selection + explicit_required - 1;
    assert!(matches!(
        PreparedProperties::prepare(&model, None, maximum),
        Err(PropertyCompilationError::ResourceCeiling { required, maximum: actual_maximum })
            if required == selection + explicit_required && actual_maximum == maximum
    ));
}

#[test]
fn changed_topology_gets_fresh_cache_and_cannot_use_another_models_identities() {
    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let initial = PreparedProperties::prepare(&model, None, u64::MAX).unwrap();
    let initial_explicit = initial.explicit.as_ref().unwrap();
    let initial_digest = initial_explicit.topology_digest();
    let initial_identities = initial_explicit.element_identities().to_vec();
    let mut changed = model.clone();
    changed.structural.as_mut().unwrap().nodes[0].coordinates_m[0] += 0.25;
    let materials = material_inventory(&changed).unwrap();
    let mut emitted = 0;
    assert!(matches!(
        visit(
            &changed,
            None,
            &materials,
            Some(initial_explicit),
            &mut |_| {
                emitted += 1;
                Ok(())
            }
        ),
        Err(PropertyCompilationError::InvalidExplicitTopology(_))
    ));
    assert_eq!(emitted, 0);
    let fresh = PreparedProperties::prepare(&changed, None, u64::MAX).unwrap();
    let fresh_explicit = fresh.explicit.as_ref().unwrap();
    assert_ne!(fresh_explicit.topology_digest(), initial_digest);
    assert_ne!(fresh_explicit.element_identities(), initial_identities);
    let fresh_digest = fresh_explicit.topology_digest();
    let compiled = fresh.materialize().unwrap();
    assert_eq!(
        compiled.identity.explicit_topology_digest,
        Some(fresh_digest)
    );
    super::super::identity::tests::assert_legacy_identity(&compiled);
}
