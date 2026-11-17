use lsp_types::{HoverContents, Url};
use runmat_lexer::fea::FeaKeyword;

use super::*;

fn check_in_roots(
    source: &str,
    physical_path: &std::path::Path,
    workspace_roots: &[PathBuf],
) -> Result<FeaDocumentAnalysis, String> {
    futures::executor::block_on(analyze_fea_source(
        source,
        physical_path,
        workspace_roots,
        &BTreeMap::new(),
        crate::core::analysis::CompatMode::RunMat,
    ))
}

#[test]
fn keyword_inventory_matches_the_lexer() {
    use runmat_analysis_language::FeaContextualWord;
    use runmat_lexer::fea::{tokenize_fea, FeaTokenKind};
    let mut names = std::collections::BTreeSet::new();
    for &keyword in FeaKeyword::ALL {
        let source = keyword.as_source();
        assert!(names.insert(source));
        assert_eq!(FeaKeyword::from_source(source), Some(keyword), "{source}");
        assert_eq!(
            tokenize_fea(source)
                .significant_tokens()
                .next()
                .unwrap()
                .kind,
            FeaTokenKind::Keyword(keyword)
        );
    }
    for &word in FeaContextualWord::ALL {
        let source = word.as_source();
        assert!(names.insert(source));
        assert_eq!(FeaContextualWord::from_source(source), Some(word));
        assert_eq!(
            FeaKeyword::from_source(source),
            None,
            "contextual {source} must not be reserved"
        );
        assert_eq!(
            tokenize_fea(source)
                .significant_tokens()
                .next()
                .unwrap()
                .kind,
            FeaTokenKind::Identifier
        );
    }
    assert_eq!(
        names.len(),
        59,
        "preserve the complete existing completion inventory"
    );
}

// witness: FEA-AUTHORING-COMPILER-002/lsp_services @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: The checked editor product drives diagnostics, navigation and signature help across
//   an imported material file; completion, rename and formatting services are not asserted.
// witness: FEA-AUTHORING-COMPILER-002/lsp_verification @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Diagnostics, definition and signature help over a complete multi-file source;
//   incomplete-file recovery and cross-language navigation are not covered here.
#[test]
fn checked_editor_product_drives_diagnostics_navigation_and_signature_help() {
    let temporary = tempfile::tempdir().unwrap();
    let material_path = temporary.path().join("materials.fea");
    std::fs::write(
        &material_path,
        r#"material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
}
"#,
    )
    .unwrap();
    let main_path = temporary.path().join("main.fea");
    let source = r#"import material steel from "./materials.fea"
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
    let analysis = check_in_roots(source, &main_path, &[temporary.path().to_path_buf()]).unwrap();

    assert!(
        analysis.diagnostics(source).is_empty(),
        "{:#?}",
        analysis.diagnostics(source)
    );
    assert_eq!(analysis.document_symbols(source).len(), 5);
    let completions = analysis.completion_items();
    assert!(completions
        .iter()
        .any(|item| item.label == "material.steel"));
    for word in runmat_analysis_language::FeaContextualWord::ALL {
        assert_eq!(
            completions
                .iter()
                .filter(|item| item.label == word.as_source())
                .count(),
            1
        );
    }
    let workspace_symbols = analysis.workspace_symbols(Some("steel"));
    assert_eq!(workspace_symbols.len(), 1);
    assert_eq!(workspace_symbols[0].name, "material.steel");
    assert_eq!(
        workspace_symbols[0].location.uri,
        Url::from_file_path(&material_path).unwrap()
    );
    assert!(analysis.semantic_tokens(source).unwrap().data.len() > 20);
    let formatting = analysis.formatting_edits(source);
    let formatted = formatting
        .first()
        .map_or_else(|| source.to_string(), |edit| edit.new_text.clone());
    assert!(analysis.formatting_edits(&formatted).is_empty());

    let material_use = source.find("material.steel to").unwrap() + "material.".len();
    let material_position = offset_to_position(source, material_use);
    let definition = analysis.definition(source, &material_position).unwrap();
    assert_eq!(definition.uri, Url::from_file_path(&material_path).unwrap());
    let references = analysis.references(source, &material_position);
    assert_eq!(references.len(), 1);
    assert_eq!(references[0].uri, Url::from_file_path(&main_path).unwrap());
    let rename_target = analysis
        .rename_target(source, &material_position)
        .expect("material reference should be renameable");
    assert_eq!(rename_target.key.name, "steel");
    let rename_locations = analysis.rename_locations(&rename_target);
    assert_eq!(rename_locations.len(), 3);
    assert!(rename_locations
        .iter()
        .any(|location| location.uri == Url::from_file_path(&material_path).unwrap()));
    assert!(!analysis.rename_has_collision(&rename_target, "aluminum"));

    let width = source.find("width = 1 m").unwrap() + 2;
    let signature = analysis
        .signature_help(source, &offset_to_position(source, width))
        .unwrap();
    assert!(signature.signatures[0].label.starts_with("box("));
    assert!(signature.signatures[0].label.contains("width"));

    let study_keyword = offset_to_position(source, source.find("study static").unwrap() + 2);
    let study_hover = analysis.hover(source, &study_keyword).unwrap();
    let HoverContents::Markup(study_hover) = study_hover.contents else {
        panic!("expected markdown hover");
    };
    assert!(study_hover.value.contains("executable analysis"));

    let material_hover = analysis.hover(source, &material_position).unwrap();
    let HoverContents::Markup(material_hover) = material_hover.contents else {
        panic!("expected markdown hover");
    };
    assert!(material_hover.value.contains("material.steel"));
    assert!(material_hover.value.contains("materials.fea"));

    let quantity_offset = source.find("0.1 m").unwrap() + 1;
    let quantity_hover = analysis
        .hover(source, &offset_to_position(source, quantity_offset))
        .unwrap();
    let HoverContents::Markup(quantity_hover) = quantity_hover.contents else {
        panic!("expected markdown hover");
    };
    assert!(quantity_hover.value.contains("**Physical quantity**"));
    assert!(quantity_hover.value.contains("0.1 m"));

    let target_size_offset = source.find("target_size").unwrap() + 2;
    let target_size_hover = analysis
        .hover(source, &offset_to_position(source, target_size_offset))
        .unwrap();
    let HoverContents::Markup(target_size_hover) = target_size_hover.contents else {
        panic!("expected markdown hover");
    };
    assert!(target_size_hover
        .value
        .contains("characteristic edge length"));
    assert!(target_size_hover.value.contains("| Dimension | length |"));
    assert!(target_size_hover.value.contains("| Canonical SI | `m` |"));

    let order_offset = source.find("order =").unwrap() + 2;
    let order_hover = analysis
        .hover(source, &offset_to_position(source, order_offset))
        .unwrap();
    let HoverContents::Markup(order_hover) = order_hover.contents else {
        panic!("expected markdown hover");
    };
    assert!(order_hover.value.contains("order: enum"));
    for contract in ["linear", "quadratic", "Tet4", "Tet10"] {
        assert!(
            order_hover.value.contains(contract),
            "missing order contract {contract}"
        );
    }

    let linear_offset = source.find("linear").unwrap() + 2;
    let linear_hover = analysis
        .hover(source, &offset_to_position(source, linear_offset))
        .unwrap();
    let HoverContents::Markup(linear_hover) = linear_hover.contents else {
        panic!("expected markdown hover");
    };
    assert!(!linear_hover.value.contains("linear: enum"));
}

#[test]
fn invalid_unsaved_source_reports_production_parser_diagnostics() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("invalid.fea");
    let source = "study broken {\n";
    let analysis = check_in_roots(source, &path, &[temporary.path().to_path_buf()]).unwrap();
    let diagnostics = analysis.diagnostics(source);
    assert!(!diagnostics.is_empty());
    assert!(diagnostics.iter().any(|diagnostic| {
        matches!(
            diagnostic.code.as_ref(),
            Some(lsp_types::NumberOrString::String(code)) if code.starts_with("fea.syntax")
        )
    }));
}

#[test]
fn project_imports_supply_callable_profile_and_source_navigation_facts() {
    let temporary = tempfile::tempdir().unwrap();
    let source_root = temporary.path();
    let source_dir = source_root.join("src");
    std::fs::create_dir(&source_dir).unwrap();
    std::fs::write(
        source_root.join("runmat.toml"),
        r#"[package]
name = "fea-editor-project"

[sources]
roots = ["src"]

[execution-profiles.engineering]
cluster = "engineering-cluster"
"#,
    )
    .unwrap();
    let function_path = source_dir.join("impactPulse.m");
    let function_source = r#"% Exact imported excitation function.

function y = impactPulse(t)
    y = t;
end
"#;
    std::fs::write(&function_path, function_source).unwrap();
    let study_path = source_dir.join("main.fea");
    let source = r#"import function impactPulse from "./impactPulse.m"
model impact_model {
    signal pulse : acceleration over time {
        generated by function.impactPulse {
            arguments {
                scale = 2
            }
            sample from 0 s to 1 s at 10 Hz
            interpolation = linear
            extrapolation = reject
            tolerance = 1e-6 m/s^2
        }
    }
}
"#;
    std::fs::write(&study_path, source).unwrap();

    let project = futures::executor::block_on(
        crate::core::project::ProjectContext::discover_from_source_name_async(study_path.to_str()),
    )
    .expect("the study's frozen project");
    let root = fea_source_root(&study_path, &[source_root.to_path_buf()], Some(&project));
    assert_eq!(root, source_root);
    let prepared = futures::executor::block_on(prepare_fea_snapshot(
        source,
        &study_path,
        &root,
        &BTreeMap::new(),
    ))
    .unwrap();
    let requests = prepared.function_requests().unwrap();
    let unsaved_function = r#"% Unsaved editor overlay.
% The saved file still has one parameter.

function y = impactPulse(t, scale)
    arguments
        t (:,1) double
        scale (1,1) double
    end
    y = t * scale;
end
"#;
    let source_overlays = BTreeMap::from([(function_path.clone(), unsaved_function.to_string())]);
    let facts = futures::executor::block_on(resolve_project_facts(
        &root,
        &requests,
        Some(&project),
        crate::core::analysis::CompatMode::RunMat,
        &source_overlays,
    ))
    .unwrap();
    assert_eq!(facts.functions.len(), 1);
    assert_eq!(facts.functions[0].inputs[0].name, "t");
    assert_eq!(facts.functions[0].inputs[1].name, "scale");
    assert_eq!(facts.profiles.len(), 1);
    assert_eq!(facts.profiles[0].profile.as_str(), "engineering");

    let checked = prepared.check(facts.functions, facts.profiles).unwrap();
    let analysis = FeaDocumentAnalysis::from_checked(checked, facts.function_sources);
    assert!(
        analysis.diagnostics(source).is_empty(),
        "{:#?}",
        analysis.diagnostics(source)
    );
    let offset = source.find("impactPulse").unwrap() + 2;
    let definition = analysis
        .definition(source, &offset_to_position(source, offset))
        .unwrap();
    assert_eq!(definition.uri, Url::from_file_path(&function_path).unwrap());
    assert_eq!(definition.range.start.line, 3);
    let rename_target = analysis
        .rename_target(source, &offset_to_position(source, offset))
        .expect("imported function should be renameable");
    assert_eq!(rename_target.key.name, "impactPulse");
    assert_eq!(
        rename_target.definition_uri,
        Url::from_file_path(&function_path).unwrap()
    );
    let rename_locations = analysis.rename_locations(&rename_target);
    assert_eq!(rename_locations.len(), 3);
    assert!(rename_locations
        .iter()
        .any(|location| { location.uri == Url::from_file_path(&study_path).unwrap() }));
    assert!(rename_locations
        .iter()
        .any(|location| { location.uri == Url::from_file_path(&function_path).unwrap() }));
    let import_paths = analysis.function_import_paths(&rename_target);
    assert_eq!(import_paths.len(), 1);
    assert_eq!(import_paths[0].source, "./impactPulse.m");
    let parameter_locations = analysis.function_parameter_locations(
        &Url::from_file_path(&function_path).unwrap(),
        "impactPulse",
        "scale",
    );
    assert_eq!(
        parameter_locations.len(),
        1,
        "sites={:#?} uses={:#?}",
        analysis.program.call_sites,
        analysis.program.bound.symbol_uses
    );
    assert_eq!(
        parameter_locations[0].uri,
        Url::from_file_path(&study_path).unwrap()
    );
    assert_eq!(
        std::fs::read_to_string(&function_path).unwrap(),
        function_source
    );
}

const STEEL_MODULE: &str = r#"material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
}
"#;

/// A project with `materials.fea` at its root, `outside.fea` above it and a study folder.
fn project_with_study_folder() -> (tempfile::TempDir, PathBuf, PathBuf) {
    let temporary = tempfile::tempdir().unwrap();
    let project = temporary.path().join("project");
    let simulation = project.join("simulation");
    std::fs::create_dir_all(&simulation).unwrap();
    std::fs::write(project.join("materials.fea"), STEEL_MODULE).unwrap();
    std::fs::write(temporary.path().join("outside.fea"), STEEL_MODULE).unwrap();
    (temporary, project, simulation.join("study.fea"))
}

fn import_codes(analysis: &FeaDocumentAnalysis, source: &str) -> Vec<String> {
    analysis
        .diagnostics(source)
        .into_iter()
        .filter_map(|diagnostic| match diagnostic.code {
            Some(lsp_types::NumberOrString::String(code)) => Some(code),
            _ => None,
        })
        .filter(|code| code.starts_with("fea.module") || code.starts_with("fea.host"))
        .collect()
}

fn imports_steel(analysis: &FeaDocumentAnalysis) -> bool {
    analysis
        .workspace_symbols(Some("steel"))
        .iter()
        .any(|symbol| symbol.location.uri.path().ends_with("/materials.fea"))
}

#[test]
fn a_study_reads_a_sibling_folder_inside_the_project_root() {
    let (_temporary, project, study) = project_with_study_folder();
    let source = "import material steel from \"../materials.fea\"\n";
    let analysis = check_in_roots(source, &study, std::slice::from_ref(&project)).unwrap();
    assert!(
        import_codes(&analysis, source).is_empty(),
        "{:#?}",
        analysis.diagnostics(source)
    );
    assert!(imports_steel(&analysis));
}

#[test]
fn a_study_cannot_read_above_the_project_root() {
    let (_temporary, project, study) = project_with_study_folder();
    let source = "import material steel from \"../../outside.fea\"\n";
    let analysis = check_in_roots(source, &study, std::slice::from_ref(&project)).unwrap();
    let diagnostics = analysis.diagnostics(source);
    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic.code
                == Some(lsp_types::NumberOrString::String(
                    "fea.module.invalid_path".into(),
                ))
                && diagnostic
                    .message
                    .contains("escapes the declared source root")
        }),
        "{diagnostics:#?}"
    );
    assert!(!imports_steel(&analysis));
}

#[test]
fn without_workspace_roots_a_study_resolves_from_the_filesystem_root() {
    let (_temporary, _project, study) = project_with_study_folder();
    let source = "import material steel from \"../materials.fea\"\n";
    let analysis = check_in_roots(source, &study, &[]).unwrap();
    assert!(
        import_codes(&analysis, source).is_empty(),
        "{:#?}",
        analysis.diagnostics(source)
    );
    assert!(imports_steel(&analysis));
    assert_eq!(
        fea_source_root(&study, &[], None),
        study.ancestors().last().unwrap()
    );
}

#[test]
fn the_deepest_workspace_root_holding_the_study_wins() {
    let (_temporary, project, study) = project_with_study_folder();
    let simulation = study.parent().unwrap().to_path_buf();
    let roots = [
        project.clone(),
        simulation.clone(),
        PathBuf::from("/elsewhere"),
    ];
    assert_eq!(fea_source_root(&study, &roots, None), simulation);
    assert_eq!(fea_source_root(&study, &roots[..1], None), project);
}

/// A study in `simulation/` may mesh a part in another folder of the project: the path climbs
/// out of the study's folder but stays inside the project root. One more `..` leaves the root.
#[test]
fn a_study_in_a_subfolder_meshes_a_part_elsewhere_in_the_project() {
    let (_temporary, project, study) = project_with_study_folder();
    let body = project.join("mechanical").join("body");
    std::fs::create_dir_all(&body).unwrap();
    std::fs::write(body.join("landing_skid.part"), "part landing_skid {}\n").unwrap();
    let escapes = |part: &str| {
        let source = format!("geometry skid {{\n    file = \"{part}\"\n    units = mm\n}}\n");
        let analysis = check_in_roots(&source, &study, std::slice::from_ref(&project)).unwrap();
        let diagnostics = analysis.diagnostics(&source);
        let escaped = diagnostics.iter().any(|diagnostic| {
            diagnostic.code
                == Some(lsp_types::NumberOrString::String(
                    "fea.inspection.invalid_geometry_path".into(),
                ))
                && diagnostic
                    .message
                    .contains("escapes the declared source root")
        });
        (escaped, diagnostics)
    };
    let (escaped, diagnostics) = escapes("../mechanical/body/landing_skid.part");
    assert!(!escaped, "{diagnostics:#?}");
    let (escaped, diagnostics) = escapes("../../mechanical/body/landing_skid.part");
    assert!(escaped, "{diagnostics:#?}");
}
