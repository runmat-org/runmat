use std::fs;

use runmat_analysis_language::{
    GeometryInspectionRequest, GeometryInspectionSource, LogicalSourcePath,
};
use tempfile::tempdir;

use super::{
    check_fea_path_in_root, check_fea_source_in_root, check_fea_sources_in_root,
    prepare_fea_path_in_root, prepare_fea_sources_in_root, FilesystemFeaHost,
};

#[test]
fn filesystem_host_checks_a_complete_constructed_study() {
    let temporary = tempdir().unwrap();
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
}

mesh volume on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
model structure {
}
study static {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.structure
    analysis solve {
        structural static {
        }
    }
    output displacement = analysis.solve.field.node.displacement
}
"#;
    let path = temporary.path().join("main.fea");
    fs::write(&path, source).unwrap();

    let program = check_fea_path_in_root(temporary.path(), &path).unwrap();

    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    assert_eq!(program.geometries.len(), 1);
    assert_eq!(program.meshes.len(), 1);
    assert_eq!(program.models.len(), 1);
    assert_eq!(program.studies.len(), 1);
}

#[test]
fn source_check_returns_canonical_typed_objects_and_source_locations() {
    let temporary = tempdir().unwrap();
    let source = r#"
geometry bracket {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region pad : face {
        select result
        expect one_or_more
    }
}
mesh volume on geometry.bracket {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
model loaded {
    constraint support {
        structural.fixed { target = geometry.bracket.region.pad }
    }
    load downward {
        force on geometry.bracket.region.pad = [0, -1, 0] kN
    }
    load_case service {
        include load.downward
    }
}
study static {
    geometry = geometry.bracket
    mesh = mesh.volume
    model = model.loaded
    analysis solve { structural static {} }
    output displacement = analysis.solve.field.node.displacement
    output peak = max(magnitude(output.displacement), over = geometry.bracket.region.pad)
    assessment limit { value = output.peak; maximum = 1 mm; evidence = authored_samples }
}
"#;
    let path = temporary.path().join("main.fea");
    fs::write(&path, source).unwrap();

    let checked = prepare_fea_path_in_root(temporary.path(), &path)
        .unwrap()
        .check(Vec::new(), Vec::new())
        .unwrap();
    assert!(
        checked.program.is_valid(),
        "{:#?}",
        checked.program.diagnostics
    );
    let report = checked.check_result();
    assert_eq!(report.program.studies.len(), 1);
    assert_eq!(
        report.program.studies[0],
        *checked.program.studies.values().next().unwrap()
    );
    assert_eq!(report.program.models.len(), 1);
    assert_eq!(
        report.program.models[0],
        *checked.program.models.values().next().unwrap()
    );
    assert_eq!(report.program.geometries.len(), 1);
    assert_eq!(
        report.program.geometries[0],
        *checked.program.geometries.values().next().unwrap()
    );
    assert_eq!(report.program.meshes.len(), 1);
    assert_eq!(
        report.program.meshes[0],
        *checked.program.meshes.values().next().unwrap()
    );
    assert!(report.source_index.entries.iter().any(|entry| matches!(
        &entry.identity,
        super::FeaSourceIdentity::Declaration { symbol, .. }
            if symbol.kind == runmat_analysis_language::FeaSymbolKind::Study
                && symbol.name == "static"
    )));
    assert!(report.source_index.entries.iter().any(|entry| matches!(
        &entry.identity,
        super::FeaSourceIdentity::ModelMember {
            member_kind: super::FeaModelMemberKind::Load,
            member_id,
            ..
        } if member_id == "downward"
    )));
    let assessment_page = report
        .browse(&runmat_fea_source::FeaBrowseRequest {
            revision: report.revision,
            module_path: None,
            filter: runmat_fea_source::FeaQueryFilter::StudyMember {
                kind: runmat_fea_source::FeaStudyMemberKind::Assessment,
            },
            after: None,
            limit: 1,
            detail: runmat_fea_source::FeaQueryDetail::Definition,
        })
        .unwrap();
    assert_eq!(assessment_page.total, 1);
    let entry = &assessment_page.entries[0];
    assert!(source[entry.location.span.start..entry.location.span.end].contains("assessment limit"));
    let runmat_fea_source::FeaObjectDefinition::Assessment(definition) =
        &assessment_page.details[0].definition
    else {
        panic!("assessment query returned another object kind")
    };
    assert_eq!(definition, &report.program.studies[0].assessments[0]);
    let dependencies = assessment_page.details[0].setup_dependencies().unwrap();
    assert!(
        matches!(&dependencies[0], runmat_fea_source::FeaSourceIdentity::StudyMember {
        member_kind: runmat_fea_source::FeaStudyMemberKind::Output, member_id, ..
    } if member_id == "peak")
    );
    let model = &report.program.models[0];
    assert_eq!(model.constraints[0].constraint_id, "support");
    assert_eq!(model.excitations[0].excitation_id, "load.downward");
    assert_eq!(model.excitation_cases[0].case_id, "load_case.service");
    assert_eq!(
        report.program.studies[0].sequence.steps[0].step_id.0,
        "solve"
    );
    assert_eq!(
        report.program.studies[0].outputs[0].output_id,
        "displacement"
    );

    let encoded = serde_json::to_vec(&report).unwrap();
    let decoded: super::FeaSourceCheckResult = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, report);
}

#[test]
fn source_overlay_checks_unsaved_root_without_mutating_the_filesystem() {
    let temporary = tempdir().unwrap();
    let imported = temporary.path().join("materials.fea");
    let imported_source = r#"material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
}

"#;
    fs::write(&imported, imported_source).unwrap();
    let path = temporary.path().join("main.fea");
    fs::write(&path, "this saved source is intentionally invalid\n").unwrap();
    let source = r#"
import material steel from "./materials.fea"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region body : solid {
        select result
        expect exactly_one
    }
}
mesh volume on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
model structure {
    assign material.steel to geometry.part.region.body
}
study static {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.structure
    analysis solve {
        structural static {}
    }
    output displacement = analysis.solve.field.node.displacement
}
"#;

    let program = check_fea_source_in_root(temporary.path(), &path, source.as_bytes()).unwrap();

    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    assert_eq!(program.materials.len(), 1);
    assert_eq!(program.studies.len(), 1);
    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        "this saved source is intentionally invalid\n"
    );

    fs::write(&imported, "this imported source is also invalid\n").unwrap();
    let program = check_fea_sources_in_root(
        temporary.path(),
        &path,
        &[
            (path.clone(), source.as_bytes().to_vec()),
            (imported.clone(), imported_source.as_bytes().to_vec()),
        ],
    )
    .unwrap();
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    assert_eq!(
        fs::read_to_string(&imported).unwrap(),
        "this imported source is also invalid\n"
    );
}

#[test]
fn prepared_source_reports_exact_deduplicated_runmat_function_requests() {
    let temporary = tempdir().unwrap();
    let path = temporary.path().join("main.fea");
    fs::write(&path, "this saved source is intentionally invalid\n").unwrap();
    let source = b"import function impactPulse from \"./signals.m\"\n\
import function impactPulse from \"./signals.m\"\n";

    let prepared =
        prepare_fea_sources_in_root(temporary.path(), &path, &[(path.clone(), source.to_vec())])
            .unwrap();
    let requests = prepared.function_requests().unwrap();

    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].module_path,
        LogicalSourcePath::new("signals.m").unwrap()
    );
    assert_eq!(
        requests[0].symbol,
        runmat_analysis_language::SourceSymbol::new("impactPulse").unwrap()
    );
    assert_eq!(
        prepared.source_dependencies().unwrap(),
        [
            LogicalSourcePath::new("main.fea").unwrap(),
            LogicalSourcePath::new("signals.m").unwrap(),
        ]
        .into_iter()
        .collect()
    );
    assert_eq!(
        fs::read_to_string(path).unwrap(),
        "this saved source is intentionally invalid\n"
    );
}

#[test]
fn prepared_dependency_closure_keeps_unavailable_import_targets() {
    let temporary = tempdir().unwrap();
    let path = temporary.path().join("main.fea");
    let prepared = prepare_fea_sources_in_root(temporary.path(), &path, &[(
        path.clone(), b"import model supports from \"./missing.fea\"\nmodel assembly { include model.supports }\n".to_vec(),
    )]).unwrap();
    assert!(prepared
        .source_dependencies()
        .unwrap()
        .contains(&LogicalSourcePath::new("missing.fea").unwrap()));
}

#[test]
fn source_check_status_distinguishes_invalid_and_incomplete_facts() {
    let temporary = tempdir().unwrap();
    let path = temporary.path().join("main.fea");
    fs::write(&path, "this is not FEA source\n").unwrap();

    let invalid = prepare_fea_path_in_root(temporary.path(), &path)
        .unwrap()
        .check(Vec::new(), Vec::new())
        .unwrap();
    let invalid_report = invalid.check_result();
    assert_eq!(invalid_report.status, super::FeaCheckStatus::Invalid);
    assert!(!invalid_report.diagnostics.is_empty());

    let source = b"import function impactPulse from \"./signals.m\"\n";
    let incomplete =
        prepare_fea_sources_in_root(temporary.path(), &path, &[(path.clone(), source.to_vec())])
            .unwrap()
            .check(Vec::new(), Vec::new())
            .unwrap();
    let incomplete_report = incomplete.check_result();
    assert_eq!(incomplete_report.status, super::FeaCheckStatus::Incomplete);
    assert!(incomplete_report.diagnostics.iter().any(|diagnostic| {
        diagnostic.availability == runmat_analysis_language::SourceAvailability::Unavailable
    }));
}

#[test]
fn runnable_root_selection_is_typed_and_requires_disambiguation() {
    let temporary = tempdir().unwrap();
    let path = temporary.path().join("main.fea");
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
}
mesh volume on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
model structure {}
study first {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.structure
    analysis solve { structural static {} }
    output displacement = analysis.solve.field.node.displacement
}
study second {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.structure
    analysis solve { structural static {} }
    output displacement = analysis.solve.field.node.displacement
}
"#;
    fs::write(&path, source).unwrap();
    let checked = prepare_fea_path_in_root(temporary.path(), &path)
        .unwrap()
        .check(Vec::new(), Vec::new())
        .unwrap();

    let report = checked.check_result();
    assert_eq!(report.status, super::FeaCheckStatus::Valid);
    assert_eq!(report.program.studies.len(), 2);
    assert_eq!(report.program.studies[0].study_id, "first");
    assert_eq!(
        report.program.studies[0].geometry.definition.geometry_id,
        "part"
    );
    assert_eq!(report.program.studies[0].mesh.mesh_id, "volume");
    assert_eq!(
        report.program.studies[0]
            .model
            .as_ref()
            .unwrap()
            .model
            .model_id,
        "structure"
    );
    assert_eq!(
        report.program.studies[0].sequence.steps[0].step_id.0,
        "solve"
    );
    assert!(matches!(
        checked.resolve_root(None),
        Err(super::FeaRootSelectionError::MultipleRoots { .. })
    ));
    let selected = checked
        .resolve_root(Some(&super::FeaRootSelector::Study {
            id: "second".to_string(),
        }))
        .unwrap();
    assert_eq!(selected.id(), "second");
}

#[test]
fn sweeps_own_their_study_root_but_preserve_explicit_study_selection() {
    let temporary = tempdir().unwrap();
    let path = temporary.path().join("main.fea");
    let study = r#"
study response {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.structure
    input scale : real = 1
    analysis solve { structural static {} }
    output displacement = analysis.solve.field.node.displacement
}
"#;
    let source = format!(
        r#"
geometry part {{
    construct {{
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }}
}}
mesh volume on geometry.part {{
    tetrahedra {{
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }}
}}
model structure {{}}
{study}
sweep envelope over study.response {{
    axis scale {{ half = 0.5; full = 1 }}
}}
"#
    );
    for independent in [false, true] {
        let source = if independent {
            format!(
                "{source}\n{}",
                study.replace("study response", "study independent")
            )
        } else {
            source.clone()
        };
        let checked = prepare_fea_sources_in_root(
            temporary.path(),
            &path,
            &[(path.clone(), source.into_bytes())],
        )
        .unwrap()
        .check(Vec::new(), Vec::new())
        .unwrap();
        assert_eq!(checked.check_result().status, super::FeaCheckStatus::Valid);
        let explicit = super::FeaRootSelector::Study {
            id: "response".into(),
        };
        assert_eq!(
            checked.resolve_root(Some(&explicit)).unwrap().id(),
            "response"
        );
        if independent {
            assert_eq!(checked.root_selectors().len(), 2);
            assert!(matches!(
                checked.resolve_root(None),
                Err(super::FeaRootSelectionError::MultipleRoots { .. })
            ));
        } else {
            assert_eq!(
                checked.root_selectors(),
                vec![super::FeaRootSelector::Sweep {
                    id: "envelope".into()
                }]
            );
            assert!(matches!(
                checked.resolve_root(None),
                Ok(super::ResolvedFeaRoot::Sweep(_))
            ));
        }
    }
}

#[test]
fn immutable_snapshot_discovers_modules_and_geometry_before_checking() {
    let temporary = tempdir().unwrap();
    let root_path = LogicalSourcePath::new("main.fea").unwrap();
    let root = super::FeaSourceSnapshotBuilder::source_module(
        root_path,
        br#"import material steel from "./materials.fea"
geometry bracket {
    file = "./geometry/box.brep"
    units = m
}
"#,
    );
    let mut builder = super::FeaSourceSnapshotBuilder::new(temporary.path(), root).unwrap();

    assert_eq!(
        builder.missing_module_paths().unwrap(),
        vec![LogicalSourcePath::new("materials.fea").unwrap()]
    );
    builder
        .add_module(super::FeaSourceSnapshotBuilder::source_module(
            LogicalSourcePath::new("materials.fea").unwrap(),
            br#"material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
}
"#,
        ))
        .unwrap();
    assert!(builder.missing_module_paths().unwrap().is_empty());

    let mut prepared = builder.prepare().unwrap();
    let requests = prepared.requests(Vec::new(), Vec::new()).unwrap();
    assert_eq!(requests.geometry.len(), 1);
    assert_eq!(
        requests.geometry[0].logical_path,
        LogicalSourcePath::new("geometry/box.brep").unwrap()
    );
    prepared
        .add_geometry(
            requests.geometry[0].logical_path.clone(),
            include_bytes!("../../../../runmat-geometry/io/tests/fixtures/box.brep").to_vec(),
        )
        .unwrap();

    let checked = prepared.check(Vec::new(), Vec::new()).unwrap();
    let report = checked.check_result();
    #[cfg(not(feature = "occt-native"))]
    {
        assert_eq!(report.status, super::FeaCheckStatus::Invalid);
        assert!(report.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "fea.host.geometry_inspection_failed"
                && diagnostic.message.contains("GEOMETRY_BACKEND_UNAVAILABLE")
        }));
    }
    #[cfg(feature = "occt-native")]
    assert_eq!(
        report.status,
        super::FeaCheckStatus::Valid,
        "diagnostics: {:#?}",
        report.diagnostics
    );
    #[cfg(feature = "occt-native")]
    assert_eq!(checked.program.geometries.len(), 1);
}

#[test]
fn filesystem_host_confines_imports_to_the_declared_root() {
    let temporary = tempdir().unwrap();
    let source_root = temporary.path().join("project");
    fs::create_dir(&source_root).unwrap();
    let path = source_root.join("main.fea");
    fs::write(
        &path,
        "import model escaped from \"../outside.fea\"\nmodel local {}\n",
    )
    .unwrap();

    let program = check_fea_path_in_root(&source_root, &path).unwrap();

    assert!(!program.is_valid());
    assert!(program
        .bound
        .graph
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "fea.module.invalid_path"));
}

#[test]
fn filesystem_host_retains_normalized_ap242_exchange_facts() {
    let temporary = tempdir().unwrap();
    let geometry_path = temporary.path().join("part.step");
    fs::write(
        &geometry_path,
        include_bytes!(
            "../../../../runmat-geometry/io/tests/fixtures/ap242_exchange_metadata.step"
        ),
    )
    .unwrap();
    let host = FilesystemFeaHost::new(temporary.path()).unwrap();
    let exchange = host
        .inspect(&GeometryInspectionRequest {
            referrer: LogicalSourcePath::new("main.fea").unwrap(),
            logical_path: LogicalSourcePath::new("part.step").unwrap(),
            source_units: Some(runmat_geometry_core::LengthUnit::Meter),
            expected_content_digest: None,
            shared_interfaces: false,
        })
        .unwrap()
        .exchange
        .unwrap();

    assert_eq!(exchange.parameters.len(), 1);
    assert_eq!(exchange.materials.len(), 1);
    assert_eq!(exchange.engineering_properties.len(), 1);
    assert_eq!(
        exchange.parameters[0].persistent_id.as_deref(),
        Some("parameter-wall-thickness")
    );
    assert_eq!(
        exchange.materials[0].persistent_id.as_deref(),
        Some("material-structural-steel")
    );
}

#[test]
fn checked_study_realizes_shared_case_membership_without_duplicate_actions() {
    let temporary = tempdir().unwrap();
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region body : solid {
        select result
        expect exactly_one
    }
    region support : face {
        select result
        expect one_or_more
    }
    region rail : edge {
        select result
        expect one_or_more
    }
}
mesh volume on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
}
model structure {
    orientation rail_axes {
        cartesian {
            origin = [0 m, 0 m, 0 m]
            x_axis = [1, 0, 0]
            y_axis = [0, 1, 0]
        }
    }
    section rail_section {
        rectangular_beam {
            width = 20 mm
            height = 40 mm
        }
    }
    assign material.steel to geometry.part.region.body
    assign material.steel to geometry.part.region.rail
    assign section.rail_section to geometry.part.region.rail
    assign orientation.rail_axes to geometry.part.region.rail
    constraint base {
        structural.fixed { target = geometry.part.region.support }
    }
    load downward {
        force on geometry.part.region.support = [0, -1, 0] kN
    }
    load_case service { include load.downward }
    load_case proof { include load.downward }
}
study static {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.structure
    analysis solve { structural static {} }
    output displacement = analysis.solve.field.node.displacement
}
"#;
    let path = temporary.path().join("main.fea");
    fs::write(&path, source).unwrap();
    let program = check_fea_path_in_root(temporary.path(), &path).unwrap();
    assert!(program.is_valid(), "{:#?}", program.diagnostics);

    let resolved = program.resolve_study("main.fea", "static").unwrap();
    assert_eq!(resolved.mesh.mesh_id, "volume");
    assert!(resolved.execution_policy.is_none());
    let realized = resolved
        .study
        .realize_with_models(&resolved.models, &resolved.geometry, &[])
        .unwrap();
    let resolved_regions = std::collections::BTreeMap::from([
        (
            "body".into(),
            runmat_geometry_core::ResolvedGeometryRegion {
                role: runmat_geometry_core::GeometryRegionRole::Solid,
                entity_ids: vec![runmat_geometry_core::PersistentEntityId::new(
                    runmat_geometry_core::PersistentEntityKind::Solid,
                    "body".into(),
                    vec!["root".into()],
                )],
            },
        ),
        (
            "support".into(),
            runmat_geometry_core::ResolvedGeometryRegion {
                role: runmat_geometry_core::GeometryRegionRole::Face,
                entity_ids: vec![runmat_geometry_core::PersistentEntityId::new(
                    runmat_geometry_core::PersistentEntityKind::Face,
                    "support".into(),
                    vec!["root".into()],
                )],
            },
        ),
        (
            "rail".into(),
            runmat_geometry_core::ResolvedGeometryRegion {
                role: runmat_geometry_core::GeometryRegionRole::Edge,
                entity_ids: vec![runmat_geometry_core::PersistentEntityId::new(
                    runmat_geometry_core::PersistentEntityKind::Edge,
                    "rail-edge".into(),
                    vec!["root".into()],
                )],
            },
        ),
    ]);
    let materials = resolved.material_catalog;
    let missing_material = runmat_analysis_core::realize_analysis_model(
        runmat_analysis_core::AnalysisModelRealizationInput {
            study: &realized,
            geometry_id: "materialized-part",
            geometry_revision: 7,
            units: runmat_types::physical::CoordinateUnits::canonical(),
            regions: &resolved_regions,
            materials: &Default::default(),
        },
    )
    .unwrap_err();
    assert!(matches!(
        missing_material,
        runmat_analysis_core::AnalysisModelRealizationError::MissingMaterial { .. }
    ));

    let mut wrong_geometry = realized.clone();
    let runmat_analysis_core::AuthoredPropertyTarget::Solid(
        runmat_analysis_core::AuthoredRegionSource::Geometry(region),
    ) = &mut wrong_geometry.model.material_assignments[0].target
    else {
        panic!("fixture material should target its solid geometry region");
    };
    region.geometry_id = "other-part".into();
    let wrong_geometry_error = runmat_analysis_core::realize_analysis_model(
        runmat_analysis_core::AnalysisModelRealizationInput {
            study: &wrong_geometry,
            geometry_id: "materialized-part",
            geometry_revision: 7,
            units: runmat_types::physical::CoordinateUnits::canonical(),
            regions: &resolved_regions,
            materials: &materials,
        },
    )
    .unwrap_err();
    assert!(matches!(
        wrong_geometry_error,
        runmat_analysis_core::AnalysisModelRealizationError::RegionFromWrongGeometry { .. }
    ));

    let mut wrong_role_regions = resolved_regions.clone();
    wrong_role_regions.get_mut("body").unwrap().role =
        runmat_geometry_core::GeometryRegionRole::Cell;
    let wrong_role_error = runmat_analysis_core::realize_analysis_model(
        runmat_analysis_core::AnalysisModelRealizationInput {
            study: &realized,
            geometry_id: "materialized-part",
            geometry_revision: 7,
            units: runmat_types::physical::CoordinateUnits::canonical(),
            regions: &wrong_role_regions,
            materials: &materials,
        },
    )
    .unwrap_err();
    assert!(matches!(
        wrong_role_error,
        runmat_analysis_core::AnalysisModelRealizationError::InvalidRegionResolution(_)
    ));

    let realized_model = runmat_analysis_core::realize_analysis_model(
        runmat_analysis_core::AnalysisModelRealizationInput {
            study: &realized,
            geometry_id: "materialized-part",
            geometry_revision: 7,
            units: runmat_types::physical::CoordinateUnits::canonical(),
            regions: &resolved_regions,
            materials: &materials,
        },
    )
    .unwrap();
    assert_eq!(realized_model.properties.sections.len(), 1);
    assert_eq!(realized_model.properties.orientations.len(), 1);
    assert_eq!(realized_model.properties.assignments.len(), 2);
    for assignment in &realized_model.properties.assignments {
        let target = match assignment {
            runmat_analysis_core::RealizedPropertyAssignment::Section { target, .. }
            | runmat_analysis_core::RealizedPropertyAssignment::Orientation { target, .. } => {
                target
            }
            runmat_analysis_core::RealizedPropertyAssignment::Layup { .. } => {
                panic!("fixture does not declare a layup")
            }
        };
        let runmat_analysis_core::PropertyRegionScope::Region(selection) = target.scope() else {
            panic!("authored property must remain region-scoped");
        };
        assert_eq!(selection.region_id(), "rail");
        assert_eq!(
            selection.resolved_entities(),
            resolved_regions["rail"].entity_ids
        );
    }
    let model = realized_model.model;

    assert_eq!(model.materials.len(), 1);
    assert_eq!(model.material_assignments.len(), 2);
    assert_eq!(model.constraints.len(), 1);
    assert_eq!(
        model.material_assignments[0]
            .target
            .scope()
            .selected_region_id(),
        Some("body")
    );
    let runmat_analysis_core::PropertyRegionScope::Region(material_target) =
        model.material_assignments[0].target.scope()
    else {
        panic!("authored material assignment must remain region-scoped");
    };
    assert_eq!(
        material_target.resolved_entities(),
        resolved_regions["body"].entity_ids
    );
    assert_eq!(
        model.constraints[0].target().unwrap().resolved_entities(),
        resolved_regions["support"].entity_ids
    );
    let runmat_analysis_core::ExcitationTarget::Region(excitation_target) =
        &model.excitations[0].excitation().target
    else {
        panic!("authored force must remain region-scoped");
    };
    assert_eq!(
        excitation_target.resolved_entities(),
        resolved_regions["support"].entity_ids
    );
    assert_eq!(model.excitations.len(), 1);
    assert_eq!(model.excitation_cases.len(), 2);
    assert!(model.excitation_cases.iter().all(
        |case| case.action_ids == [runmat_analysis_core::ExcitationId("load.downward".into())]
    ));
    assert_eq!(model.sequence.steps[0].observations.len(), 1);
    runmat_analysis_core::validate_model(&model).unwrap();

    let (geometry, _, _) = runmat_geometry_fixtures::exact_tetrahedron();
    let mut mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(
        runmat_meshing_core::ElementOrder::Tet4,
    );
    mesh.geometry.source_digest =
        runmat_meshing_core::StableDigest::from_bytes(*geometry.source.content_digest.bytes());
    mesh.geometry.geometry_revision = geometry.revision.revision;
    mesh.geometry.persistent_mapping_version = geometry.revision.persistent_mapping_version;
    mesh.topology.volume_elements[0].region_id = resolved_regions["body"].entity_ids[0].clone();
    mesh.topology.regions[0].region_id = resolved_regions["body"].entity_ids[0].clone();
    mesh.topology.volume_elements[0].region_id.kind =
        runmat_geometry_core::PersistentEntityKind::Region;
    mesh.topology.regions[0].region_id.kind = runmat_geometry_core::PersistentEntityKind::Region;
    mesh.topology.volume_elements[0].provenance = vec![
        mesh.topology.volume_elements[0].region_id.clone(),
        resolved_regions["body"].entity_ids[0].clone(),
    ];
    mesh.topology.volume_elements[0].provenance.sort();
    mesh.topology.boundary_edges[0].provenance = resolved_regions["rail"].entity_ids.clone();
    mesh.topology.boundary_faces[0].provenance = resolved_regions["support"].entity_ids.clone();
    mesh.canonical_digest = runmat_meshing_core::StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();

    let planning_policy = runmat_analysis_fea::study::StructuralStudyPlanningPolicy {
        backend: runmat_analysis_fea::ComputeBackend::Cpu,
        numerics: runmat_analysis_fea::study::StructuralNumericalPlanningPolicy::production(),
    };
    let product = runmat_analysis_fea::study::build_structural_study_product(
        runmat_analysis_fea::study::StructuralStudyProductInput {
            study: &realized,
            geometry: &geometry,
            regions: &resolved_regions,
            materials: &materials,
            mesh: &mesh,
            planning: planning_policy,
        },
    )
    .unwrap();
    let compiled = product.compiled;
    compiled.validate_against(&mesh).unwrap();
    assert_eq!(compiled.model.geometry_id, "part");
    assert_eq!(
        compiled.model.structural.as_ref().unwrap().elements.len(),
        1
    );
    assert_eq!(compiled.properties.solid_materials().len(), 1);
    assert_eq!(compiled.properties.beam_properties().len(), 1);
    assert_ne!(compiled.definition_digest().unwrap().bytes(), &[0; 32]);
    let workloads = product.workloads;
    assert_eq!(
        workloads.step_ids().unwrap(),
        [compiled.model.sequence.steps[0].step_id.clone()]
    );

    let mut transient_compiled = compiled.clone();
    transient_compiled.model.sequence.steps[0].procedure =
        runmat_analysis_core::AnalysisProcedure::TransientResponse;
    transient_compiled.model.sequence.steps[0].independent_domain =
        runmat_analysis_core::IndependentDomain::TimeSeconds {
            origin: 0.0,
            start: 0.0,
            end: 2.0,
            interval_count: 20,
        };
    let transient =
        runmat_analysis_fea::study::plan_structural_study(&transient_compiled, planning_policy)
            .unwrap();
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::StructuralTransient {
        workload,
        block_plan,
    } = transient
    else {
        panic!("direct transient study should produce a transient workload");
    };
    assert_eq!(workload.time_step_policy.interval_count(), 20);
    assert_eq!(workload.residual_target, 1.0e-8);
    assert_eq!(block_plan.maximum_intervals_per_block(), 20);

    transient_compiled.model.sequence.steps[0]
        .accuracy
        .criteria
        .clear();
    assert!(matches!(
        runmat_analysis_fea::study::plan_structural_study(
            &transient_compiled,
            planning_policy,
        ),
        Err(
            runmat_analysis_fea::study::StructuralStudyPlanningError::MissingAlgebraicResidualTarget { .. }
        )
    ));

    let mut modal_compiled = compiled.clone();
    modal_compiled.model.sequence = runmat_analysis_core::AnalysisSequence::from_steps(
        "modal",
        vec![runmat_analysis_core::ProcedureStep::structural_modal(
            "modes",
        )],
    );
    assert!(matches!(
        runmat_analysis_fea::study::plan_structural_study(&modal_compiled, planning_policy)
            .unwrap(),
        runmat_analysis_fea::execution::StructuralStudyWorkloads::StructuralModal { .. }
    ));

    let mut harmonic_compiled = compiled.clone();
    let mut harmonic = runmat_analysis_core::ProcedureStep::structural_harmonic("harmonic");
    harmonic.independent_domain =
        runmat_analysis_core::IndependentDomain::FrequencyHertz(vec![10.0, 20.0]);
    harmonic_compiled.model.sequence =
        runmat_analysis_core::AnalysisSequence::from_steps("harmonic", vec![harmonic]);
    assert!(matches!(
        runmat_analysis_fea::study::plan_structural_study(&harmonic_compiled, planning_policy)
            .unwrap(),
        runmat_analysis_fea::execution::StructuralStudyWorkloads::StructuralFrequency { .. }
    ));

    let mut modal_transient_compiled = compiled.clone();
    let modal = runmat_analysis_core::ProcedureStep::structural_modal("modes");
    let transient = runmat_analysis_core::ProcedureStep::structural_modal_transient(
        "response",
        "modes",
        "modal_basis",
    );
    modal_transient_compiled.model.sequence.steps = vec![modal, transient];
    modal_transient_compiled.model.sequence.dependencies =
        vec![runmat_analysis_core::SequenceDependency {
            predecessor: runmat_analysis_core::ProcedureStepId("modes".into()),
            successor: runmat_analysis_core::ProcedureStepId("response".into()),
            reason: runmat_analysis_core::SequenceDependencyReason::StateTransfer,
        }];
    modal_transient_compiled.model.sequence.transitions =
        vec![runmat_analysis_core::StateTransition {
            transition_id: runmat_analysis_core::TransitionId("basis-to-response".into()),
            producer_step: runmat_analysis_core::ProcedureStepId("modes".into()),
            consumer_step: runmat_analysis_core::ProcedureStepId("response".into()),
            state_id: runmat_analysis_core::StateId("modal_basis".into()),
            method: runmat_analysis_core::StateTransitionMethod::Reuse,
            topology: runmat_analysis_core::TopologyTransitionPolicy::RequireIdentical,
            reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
        }];
    let modal_transient = runmat_analysis_fea::study::plan_structural_study(
        &modal_transient_compiled,
        planning_policy,
    )
    .unwrap();
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
        workloads,
    } = modal_transient
    else {
        panic!("modal transient should produce a heterogeneous sequence");
    };
    assert!(matches!(
        workloads.get(&runmat_analysis_core::ProcedureStepId("modes".into())),
        Some(runmat_analysis_fea::execution::StructuralStepWorkload::ModalExtraction { .. })
    ));
    assert!(matches!(
        workloads.get(&runmat_analysis_core::ProcedureStepId("response".into())),
        Some(runmat_analysis_fea::execution::StructuralStepWorkload::ModalTransient { .. })
    ));

    let mut buckling_compiled = compiled.clone();
    let preload = runmat_analysis_core::ProcedureStep::structural_preload("preload");
    let buckling = runmat_analysis_core::ProcedureStep::structural_buckling(
        "buckling",
        "preload",
        "geometric_stiffness",
    );
    buckling_compiled.model.sequence.steps = vec![preload, buckling];
    buckling_compiled.model.sequence.dependencies =
        vec![runmat_analysis_core::SequenceDependency {
            predecessor: runmat_analysis_core::ProcedureStepId("preload".into()),
            successor: runmat_analysis_core::ProcedureStepId("buckling".into()),
            reason: runmat_analysis_core::SequenceDependencyReason::StateTransfer,
        }];
    buckling_compiled.model.sequence.transitions = vec![runmat_analysis_core::StateTransition {
        transition_id: runmat_analysis_core::TransitionId("preload-to-buckling".into()),
        producer_step: runmat_analysis_core::ProcedureStepId("preload".into()),
        consumer_step: runmat_analysis_core::ProcedureStepId("buckling".into()),
        state_id: runmat_analysis_core::StateId("geometric_stiffness".into()),
        method: runmat_analysis_core::StateTransitionMethod::Reuse,
        topology: runmat_analysis_core::TopologyTransitionPolicy::RequireIdentical,
        reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
    }];
    let buckling =
        runmat_analysis_fea::study::plan_structural_study(&buckling_compiled, planning_policy)
            .unwrap();
    assert!(matches!(
        buckling,
        runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence { .. }
    ));

    let mut wrong_mesh = mesh;
    wrong_mesh.geometry.geometry_revision += 1;
    wrong_mesh.canonical_digest = runmat_meshing_core::StableDigest::ZERO;
    wrong_mesh.seal_canonical_digest().unwrap();
    let mismatch = runmat_analysis_fea::study::compile_analysis_study(
        runmat_analysis_fea::study::AnalysisStudyCompilationInput {
            study: &realized,
            geometry: &geometry,
            regions: &resolved_regions,
            materials: &materials,
            mesh: &wrong_mesh,
        },
    )
    .unwrap_err();
    assert!(matches!(
        mismatch,
        runmat_analysis_fea::study::AnalysisStudyCompilationError::GeometryMeshMismatch
    ));
}
