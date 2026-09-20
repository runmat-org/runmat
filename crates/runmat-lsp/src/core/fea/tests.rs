use lsp_types::Url;
use runmat_lexer::fea::FeaKeyword;

use super::*;

#[test]
fn keyword_inventory_matches_the_lexer() {
    for keyword in KEYWORDS {
        assert!(FeaKeyword::from_source(keyword).is_some(), "{keyword}");
    }
}

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
    require formulation = experimental
    analysis solve {
        structural static {}
    }
    output displacement = analysis.solve.field.node.displacement
}
"#;
    let analysis = FeaDocumentAnalysis::check_with_overlays(
        source,
        &main_path,
        &[temporary.path().to_path_buf()],
        &[],
    )
    .unwrap();

    assert!(analysis.diagnostics(source).is_empty());
    assert_eq!(analysis.document_symbols(source).len(), 5);
    assert!(analysis
        .completion_items()
        .iter()
        .any(|item| item.label == "material.steel"));
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

    let width = source.find("width = 1 m").unwrap() + 2;
    let signature = analysis
        .signature_help(source, &offset_to_position(source, width))
        .unwrap();
    assert!(signature.signatures[0].label.starts_with("box("));
    assert!(signature.signatures[0].label.contains("width"));
}

#[test]
fn invalid_unsaved_source_reports_production_parser_diagnostics() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("invalid.fea");
    let source = "study broken {\n";
    let analysis = FeaDocumentAnalysis::check_with_overlays(
        source,
        &path,
        &[temporary.path().to_path_buf()],
        &[],
    )
    .unwrap();
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
    let source = "import function impactPulse from \"./impactPulse.m\"\n";
    std::fs::write(&study_path, source).unwrap();

    let preliminary = FeaDocumentAnalysis::check_with_overlays(
        source,
        &study_path,
        &[source_root.to_path_buf()],
        &[],
    )
    .unwrap();
    let unsaved_function = r#"% Unsaved editor overlay.
% The saved file still has one parameter.

function y = impactPulse(t, scale)
    y = t * scale;
end
"#;
    let source_overlays = BTreeMap::from([(function_path.clone(), unsaved_function.to_string())]);
    let facts = resolve_project_facts(
        &preliminary,
        &study_path,
        crate::core::analysis::CompatMode::RunMat,
        &source_overlays,
    )
    .unwrap();
    assert_eq!(facts.functions.len(), 1);
    assert_eq!(facts.functions[0].inputs[0].name, "t");
    assert_eq!(facts.functions[0].inputs[1].name, "scale");
    assert_eq!(facts.profiles.len(), 1);
    assert_eq!(facts.profiles[0].profile.as_str(), "engineering");

    let analysis = FeaDocumentAnalysis::check_with_project_facts(
        source,
        &study_path,
        &[source_root.to_path_buf()],
        &[],
        facts.functions,
        facts.profiles,
        facts.function_sources,
    )
    .unwrap();
    assert!(analysis.diagnostics(source).is_empty());
    let offset = source.find("impactPulse").unwrap() + 2;
    let definition = analysis
        .definition(source, &offset_to_position(source, offset))
        .unwrap();
    assert_eq!(definition.uri, Url::from_file_path(&function_path).unwrap());
    assert_eq!(definition.range.start.line, 3);
    assert_eq!(
        std::fs::read_to_string(&function_path).unwrap(),
        function_source
    );
}
