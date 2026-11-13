use crate::fixtures::{fixture_model, FixtureId};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

use super::*;

#[test]
fn compiled_map_separates_semantic_identity_from_audit_confidence() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let model = matching_model(&mesh);
    let first = compile_solid_property_map(&model, &mesh).expect("compiled properties");

    let mut changed_audit = model.clone();
    changed_audit.material_assignments[0].confidence =
        runmat_analysis_core::EvidenceConfidence::Inferred;
    changed_audit.material_assignments[0].expected_material_id = "imported_steel_alias".into();
    let second = compile_solid_property_map(&changed_audit, &mesh).expect("compiled properties");

    assert_eq!(
        first.identity.semantic_digest,
        second.identity.semantic_digest
    );
    assert_ne!(first.audit_digest, second.audit_digest);
    assert_eq!(
        first.solid_materials().len(),
        mesh.topology.volume_elements.len()
    );
}

#[test]
fn compiled_map_has_a_canonical_json_safe_round_trip() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let compiled =
        compile_solid_property_map(&matching_model(&mesh), &mesh).expect("compiled properties");
    let encoded = serde_json::to_vec(&compiled).expect("encode compiled property map");
    let value = serde_json::to_value(&compiled).unwrap();
    // One shared material row; per-element data are columns.
    assert_eq!(value["materials"].as_array().unwrap().len(), 1);
    assert_eq!(
        value["solids"]["identities"].as_array().unwrap().len(),
        compiled.solid_materials().len()
    );
    assert_eq!(
        value["audit"]["elements"].as_array().unwrap().len(),
        compiled.audit().len()
    );
    let decoded: CompiledPropertyMap =
        serde_json::from_slice(&encoded).expect("decode compiled property map");
    assert_eq!(decoded, compiled);
}

#[test]
fn material_display_evidence_changes_audit_but_not_numerical_semantics() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let model = matching_model(&mesh);
    let first = compile_solid_property_map(&model, &mesh).expect("compiled properties");
    let mut changed_evidence = model;
    changed_evidence.materials[0].name = "Imported display label".into();
    let second = compile_solid_property_map(&changed_evidence, &mesh).expect("compiled properties");

    assert_eq!(
        first.identity.semantic_digest,
        second.identity.semantic_digest
    );
    assert_ne!(first.audit_digest, second.audit_digest);
}

#[test]
fn compiled_map_decode_rejects_semantic_audit_and_entry_substitution() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let compiled =
        compile_solid_property_map(&matching_model(&mesh), &mesh).expect("compiled properties");
    let encoded = serde_json::to_value(&compiled).expect("encode compiled property map");

    let mut semantic = encoded.clone();
    semantic["identity"]["semantic_digest"] =
        serde_json::to_value(runmat_execution::Digest::sha256(b"substituted semantics")).unwrap();
    assert!(serde_json::from_value::<CompiledPropertyMap>(semantic).is_err());

    let mut audit = encoded.clone();
    audit["audit_digest"] =
        serde_json::to_value(runmat_execution::Digest::sha256(b"substituted audit")).unwrap();
    assert!(serde_json::from_value::<CompiledPropertyMap>(audit).is_err());

    let mut entry = encoded.clone();
    entry["solids"]["identities"][0] = serde_json::json!("a5".repeat(32));
    assert!(serde_json::from_value::<CompiledPropertyMap>(entry).is_err());

    let mut incomplete_audit = encoded;
    incomplete_audit["audit"] = serde_json::json!({"elements": [], "owners": []});
    assert!(serde_json::from_value::<CompiledPropertyMap>(incomplete_audit).is_err());

    let mut material_evidence = serde_json::to_value(&compiled).unwrap();
    material_evidence["materials"][0]["name"] =
        serde_json::json!("substituted material display name");
    assert!(serde_json::from_value::<CompiledPropertyMap>(material_evidence).is_err());
}

// witness: FEA-SHARED-PROPERTIES-001/distinct_property_roles @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: A compiled property map rejects gaps, overlaps, wrong roles and stale scopes; the
//   full distinct-role list (orientation, section, layup) is not asserted.
#[test]
fn compiled_map_rejects_gaps_overlaps_wrong_roles_and_stale_scopes() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let model = matching_model(&mesh);

    let mut gap = model.clone();
    gap.material_assignments.clear();
    assert!(matches!(
        compile_solid_property_map(&gap, &mesh),
        Err(PropertyCompilationError::InvalidCoverage {
            assignment_count: 0,
            ..
        })
    ));

    let mut overlap = model.clone();
    let mut all = overlap.material_assignments[0].clone();
    all.target = runmat_analysis_core::PropertyRegionTarget::all_solids();
    overlap.material_assignments.push(all);
    assert!(matches!(
        compile_solid_property_map(&overlap, &mesh),
        Err(PropertyCompilationError::InvalidCoverage {
            assignment_count: 2,
            ..
        })
    ));
    overlap
        .material_assignments
        .push(overlap.material_assignments[0].clone());
    assert!(matches!(
        compile_solid_property_map(&overlap, &mesh),
        Err(PropertyCompilationError::InvalidCoverage {
            assignment_count: 3,
            ..
        })
    ));

    let mut wrong_role = model.clone();
    wrong_role.material_assignments[0].target =
        runmat_analysis_core::PropertyRegionTarget::all_cells();
    assert!(matches!(
        compile_solid_property_map(&wrong_role, &mesh),
        Err(PropertyCompilationError::InvalidCoverage {
            assignment_count: 0,
            ..
        })
    ));

    let mut stale = model;
    stale.material_assignments[0].target = runmat_analysis_core::PropertyRegionTarget::solid(
        runmat_meshing_core::fixtures::fixture_region("absent-region"),
    );
    assert!(compile_solid_property_map(&stale, &mesh).is_err());
}

#[test]
fn structural_solid_capabilities_fail_before_assembly_without_restricting_generic_properties() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let model = matching_model(&mesh);
    let generic = compile_property_map(&model, Some(&mesh)).expect("generic solid properties");
    let structural =
        compile_structural_property_map(&model, Some(&mesh)).expect("structural solid properties");
    assert_eq!(generic.identity, structural.identity);

    let mut thermal_only = model.clone();
    thermal_only.materials[0].elasticity = None;
    compile_property_map(&thermal_only, Some(&mesh))
        .expect("generic material ownership must not require elasticity");
    assert!(matches!(
        compile_structural_property_map(&thermal_only, Some(&mesh)),
        Err(PropertyCompilationError::MissingCapability {
            role: "structural solid",
            capability: runmat_analysis_core::MaterialCapability::IsotropicLinearElasticity,
            ..
        })
    ));

    let mut massless = model;
    massless.materials[0].mass_density_kg_per_m3 =
        (None).map(runmat_types::physical::MassDensity::new);
    compile_property_map(&massless, Some(&mesh))
        .expect("generic material ownership must not require density");
    assert!(matches!(
        compile_structural_property_map(&massless, Some(&mesh)),
        Err(PropertyCompilationError::MissingCapability {
            role: "structural solid",
            capability: runmat_analysis_core::MaterialCapability::MassDensity,
            ..
        })
    ));
}

#[test]
fn thermal_solid_capabilities_fail_before_assembly_without_restricting_generic_properties() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let model = matching_model(&mesh);
    let generic = compile_property_map(&model, Some(&mesh)).expect("generic solid properties");
    let thermal =
        compile_thermal_property_map(&model, Some(&mesh)).expect("thermal solid properties");
    assert_eq!(generic.identity, thermal.identity);

    let mut nonthermal = model.clone();
    nonthermal.materials[0].thermal = None;
    compile_property_map(&nonthermal, Some(&mesh)).expect("generic material ownership");
    assert!(matches!(
        compile_thermal_property_map(&nonthermal, Some(&mesh)),
        Err(PropertyCompilationError::MissingCapability {
            role: "thermal solid",
            capability: runmat_analysis_core::MaterialCapability::Thermal,
            ..
        })
    ));

    let mut massless = model;
    massless.materials[0].mass_density_kg_per_m3 =
        (None).map(runmat_types::physical::MassDensity::new);
    assert!(matches!(
        compile_thermal_property_map(&massless, Some(&mesh)),
        Err(PropertyCompilationError::MissingCapability {
            role: "thermal solid",
            capability: runmat_analysis_core::MaterialCapability::MassDensity,
            ..
        })
    ));
}

#[test]
fn compiled_map_uses_bound_entity_sets_instead_of_logical_region_names() {
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let selected = runmat_geometry_core::PersistentEntityId::new(
        runmat_geometry_core::PersistentEntityKind::Solid,
        "selected-solid".into(),
        vec!["root".into()],
    );
    mesh.topology.volume_elements[0]
        .provenance
        .push(selected.clone());
    mesh.topology.volume_elements[0].provenance.sort();
    mesh.canonical_digest = runmat_meshing_core::StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    let mut model = matching_model(&mesh);
    let mut target = runmat_analysis_core::PropertyRegionTarget::solid(
        runmat_meshing_core::fixtures::fixture_solid("logical-body"),
    );
    target
        .bind_resolved_entities(vec![
            selected,
            runmat_geometry_core::PersistentEntityId::new(
                runmat_geometry_core::PersistentEntityKind::Solid,
                "another-solid".into(),
                vec!["root".into()],
            ),
        ])
        .unwrap();
    model.material_assignments[0].target = target;

    let compiled = compile_solid_property_map(&model, &mesh).expect("compiled properties");
    assert_eq!(compiled.solid_materials().len(), 1);

    let mut wrong_kind = runmat_analysis_core::PropertyRegionTarget::solid(
        runmat_meshing_core::fixtures::fixture_solid("logical-body"),
    );
    assert!(wrong_kind
        .bind_resolved_entities(vec![mesh.topology.boundary_faces[0].provenance[0].clone()])
        .is_err());
}

// witness: FEA-SHARED-PROPERTIES-001/assignment_qualification_corpus @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Explicit beam and shell domains project without cross-role fallback; a multi-solid
//   exact-CAD corpus is not covered.
#[test]
fn compiled_map_projects_explicit_beam_and_shell_domains_without_cross_role_fallback() {
    let beam_model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let beam = compile_property_map(&beam_model, None).expect("compiled beam properties");
    assert_eq!(beam.beam_properties().len(), 1);
    assert_eq!(beam.shell_properties().len(), 0);
    assert!(beam.identity.mesh_digest.is_none());
    assert!(beam.identity.explicit_topology_digest.is_some());

    let shell_model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
    let shell = compile_property_map(&shell_model, None).expect("compiled shell properties");
    assert_eq!(shell.beam_properties().len(), 0);
    assert_eq!(shell.shell_properties().len(), 1);

    let mut wrong_role = beam_model;
    wrong_role.material_assignments[0].target =
        runmat_analysis_core::PropertyRegionTarget::all_shells();
    assert!(matches!(
        compile_property_map(&wrong_role, None),
        Err(PropertyCompilationError::InvalidCoverage {
            role: "beam",
            assignment_count: 0,
            ..
        })
    ));
}

// witness: FEA-SHARED-PROPERTIES-001/required_sections_and_layups @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Explicit properties reject duplicate and missing sections.
#[test]
fn explicit_properties_reject_duplicate_and_missing_sections() {
    for fixture in [
        FixtureId::StructuralBeamCantileverEndMomentReference,
        FixtureId::StructuralShellPlateMomentReference,
    ] {
        let model = fixture_model(fixture);
        for duplicate in [false, true] {
            let mut invalid = model.clone();
            let structural = invalid.structural.as_mut().unwrap();
            if !structural.beam_sections.is_empty() {
                if duplicate {
                    structural
                        .beam_sections
                        .push(structural.beam_sections[0].clone());
                } else {
                    structural.beam_sections.clear();
                }
            } else if duplicate {
                structural
                    .shell_sections
                    .push(structural.shell_sections[0].clone());
            } else {
                structural.shell_sections.clear();
            }
            assert!(matches!(
                compile_property_map(&invalid, None),
                Err(PropertyCompilationError::InvalidSection { .. })
            ));
        }
    }
}

// witness: FEA-SHARED-PROPERTIES-001/independent_section_coverage @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Laminate property identity round-trips and rejects competing homogeneous ownership.
#[test]
fn laminate_property_identity_round_trips_and_rejects_competing_homogeneous_ownership() {
    use runmat_analysis_core::{
        LaminatePlyModel, LaminateShellSectionModel, PlyOrderDirection, ShellReferenceSurface,
        ShellSectionModel,
    };

    let mut model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
    model.material_assignments.clear();
    model
        .structural
        .as_mut()
        .expect("shell fixture")
        .shell_sections = vec![ShellSectionModel::Laminate(LaminateShellSectionModel {
        section_id: "panel_2mm".into(),
        reference_orientation_id: "global_x".into(),
        ply_order: PlyOrderDirection::AlongSurfaceNormal,
        reference_surface: ShellReferenceSurface::MidSurface,
        plies: ["lower", "upper"]
            .map(|ply_id| LaminatePlyModel {
                ply_id: ply_id.into(),
                material_id: "mat_steel".into(),
                thickness_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Length,
                    _,
                >::new(0.001),
                angle_rad:
                    runmat_types::physical::Quantity::<runmat_types::physical::kind::Angle, _>::new(
                        0.0,
                    ),
            })
            .to_vec(),
        shear_correction: Some(5.0 / 6.0),
    })];

    let compiled = compile_property_map(&model, None).expect("compiled laminate property map");
    super::identity::tests::assert_reference_identity(&compiled);
    assert_eq!(compiled.shell_properties().len(), 1);
    assert!(matches!(
        compiled.shell_properties().next().unwrap().owner,
        CompiledShellPropertyOwner::Laminate { .. }
    ));
    assert!(matches!(
        compiled.audit()[0].owner,
        CompiledPropertyAuditOwner::Laminate { .. }
    ));
    let encoded = serde_json::to_vec(&compiled).expect("encode laminate property map");
    let decoded: CompiledPropertyMap =
        serde_json::from_slice(&encoded).expect("decode laminate property map");
    assert_eq!(decoded, compiled);

    let mut missing_elasticity = model.clone();
    missing_elasticity.materials[0].elasticity = None;
    assert!(matches!(
        compile_property_map(&missing_elasticity, None),
        Err(PropertyCompilationError::MissingCapability {
            capability: runmat_analysis_core::MaterialCapability::LaminaElasticity,
            ..
        })
    ));

    let mut overlap = model;
    overlap
        .material_assignments
        .push(runmat_analysis_core::MaterialAssignment {
            target: runmat_analysis_core::PropertyRegionTarget::all_shells(),
            expected_material_id: "mat_steel".into(),
            assigned_material_id: "mat_steel".into(),
            confidence: runmat_analysis_core::EvidenceConfidence::Verified,
        });
    assert!(matches!(
        compile_property_map(&overlap, None),
        Err(PropertyCompilationError::LaminateMaterialOverlap {
            assignment_count: 1,
            ..
        })
    ));
}

// witness: FEA-SHARED-PROPERTIES-001/compiled_assignment_identity @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Semantic identity is independent of assignment order while the audit trail is not.
#[test]
fn semantic_identity_is_independent_of_assignment_order_but_audit_is_not() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut model = matching_model(&mesh);
    let mut unused_role = model.material_assignments[0].clone();
    // Cell now has actual volume ownership and cannot be an inert unused role.
    // Beam remains disjoint when there is no explicit beam topology.
    unused_role.target = runmat_analysis_core::PropertyRegionTarget::all_beams();
    model.material_assignments.push(unused_role);

    let first = compile_solid_property_map(&model, &mesh).expect("compiled properties");
    model.material_assignments.reverse();
    let second = compile_solid_property_map(&model, &mesh).expect("compiled reordered properties");

    assert_eq!(
        first.identity.semantic_digest,
        second.identity.semantic_digest
    );
    assert_eq!(
        first.solid_materials().collect::<Vec<_>>(),
        second.solid_materials().collect::<Vec<_>>()
    );
    assert_ne!(first.audit_digest, second.audit_digest);
}

#[test]
fn overlapping_solid_and_cell_volume_intent_is_rejected_not_ignored() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut model = matching_model(&mesh);
    let mut cell = model.material_assignments[0].clone();
    cell.target = runmat_analysis_core::PropertyRegionTarget::all_cells();
    model.material_assignments.push(cell);
    assert!(matches!(
        compile_property_map(&model, Some(&mesh)),
        Err(PropertyCompilationError::InvalidCoverage {
            role: "volume",
            assignment_count: 2,
            ..
        })
    ));
}

#[test]
fn cell_ownership_cannot_satisfy_solid_constitutive_compilation() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut model = matching_model(&mesh);
    model.material_assignments[0].target = runmat_analysis_core::PropertyRegionTarget::all_cells();
    let cells = compile_property_map(&model, Some(&mesh)).unwrap();
    assert!(matches!(
        cells.validate_structural_solid_capabilities(),
        Err(PropertyCompilationError::CellVolumeForSolidPhysics)
    ));
    assert!(matches!(
        cells.validate_thermal_solid_capabilities(),
        Err(PropertyCompilationError::CellVolumeForSolidPhysics)
    ));
}

fn matching_model(
    mesh: &runmat_meshing_core::SolverMeshArtifact,
) -> runmat_analysis_core::AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.geometry_revision = u32::try_from(mesh.geometry.geometry_revision).unwrap();
    model.material_assignments[0].target = runmat_analysis_core::PropertyRegionTarget::solid(
        mesh.topology.volume_elements[0]
            .region_id
            .analysis_selection_id()
            .unwrap(),
    );
    model
}

// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Tet4/Tet10 fixtures verify that compiled property clones share original element and
//   audit inventory storage, preserve wire bytes, survive original owner drop, and independently
//   validate cold reconstruction. No whole publication admission, allocator peak or study scaling
//   claim.
#[test]
fn clones_share_original_validated_properties_without_changing_cold_wire_authority() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let compiled = compile_solid_property_map(&matching_model(&mesh), &mesh).unwrap();
        let shared = compiled.clone();
        assert!(std::ptr::eq(
            compiled.solid_materials().next().unwrap(),
            shared.solid_materials().next().unwrap(),
        ));
        assert_eq!(compiled.audit().as_ptr(), shared.audit().as_ptr());
        let encoded = serde_json::to_vec(&compiled).unwrap();
        assert_eq!(encoded, serde_json::to_vec(&shared).unwrap());
        let cold: CompiledPropertyMap = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(cold, compiled);
        cold.validate_integrity().unwrap();
        assert!(!std::ptr::eq(
            compiled.solid_materials().next().unwrap(),
            cold.solid_materials().next().unwrap(),
        ));
        drop(compiled);
        shared.validate_integrity().unwrap();
    }
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Original material/audit ownership is separate from shared clones, varies with
//   retained display buffers and survives independent cold validation. Conservative typed component
//   inventory, not RSS or whole-study scalability qualification.
#[test]
fn original_property_storage_charges_retained_materials_and_audit_once_per_owner() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let model = matching_model(&mesh);
    let properties = compile_solid_property_map(&model, &mesh).unwrap();
    let limits = runmat_canonical_codec::CanonicalLimits::new(16 << 20, 1_000_000, 65_536, 64);
    let original = properties.retained_storage_bytes_estimate(limits).unwrap();
    assert!(original > properties.shared_clone_storage_bytes().unwrap());
    let mut larger = model.clone();
    larger.materials[0].name = "display label".repeat(2048);
    let larger = compile_solid_property_map(&larger, &mesh).unwrap();
    assert_eq!(
        properties.identity.semantic_digest,
        larger.identity.semantic_digest
    );
    assert!(larger.retained_storage_bytes_estimate(limits).unwrap() > original);
    assert!(properties
        .retained_storage_bytes_estimate(runmat_canonical_codec::CanonicalLimits::new(
            1024, 1_000_000, 65_536, 64
        ))
        .is_err());
    let wire = serde_json::to_vec(&properties).unwrap();
    let cold: CompiledPropertyMap = serde_json::from_slice(&wire).unwrap();
    cold.validate_integrity().unwrap();
    assert!(
        cold.retained_storage_bytes_estimate(limits).unwrap()
            > cold.shared_clone_storage_bytes().unwrap()
    );
}
