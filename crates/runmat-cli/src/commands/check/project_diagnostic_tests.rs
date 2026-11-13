use super::*;

fn project() -> ProjectAnalysis {
    ProjectAnalysis {
        analysis: runmat_static_analysis::frontend::analyze_source(
            "",
            runmat_parser::CompatMode::default(),
            &runmat_hir::LoweringContext::empty(),
        ),
        sources: vec![
            ProjectSource {
                path: "project/driver.m".into(),
                text: "% π\nA = ones(2,3);\nC = incompatible_product(A, A);\n".into(),
            },
            ProjectSource {
                path: "project/incompatible_product.m".into(),
                text: "function C = incompatible_product(A, B)\nC = A * B;\nend\n".into(),
            },
        ],
        incomplete_sources: Vec::new(),
        call_context_budget_exhausted: false,
        call_context_precision_withheld: false,
    }
}

fn diagnostic(project: &ProjectAnalysis) -> HirDiagnostic {
    let call = project.sources[0]
        .text
        .find("incompatible_product")
        .unwrap();
    let operation = project.sources[1].text.find("A * B").unwrap();
    let mut diagnostic = HirDiagnostic::new(
        "RM-SHAPE-TEST",
        HirDiagnosticSeverity::Error,
        "incompatible matrix product",
        Span {
            start: call,
            end: call + "incompatible_product".len(),
        },
    )
    .with_source(ProgramSourceId(0))
    .with_secondary(
        Span {
            start: operation,
            end: operation + 5,
        },
        "callee operation",
    )
    .with_suggestion(
        Span {
            start: operation,
            end: operation + 5,
        },
        "A .* B",
        "elementwise alternative",
    );
    diagnostic.secondary[0].source = Some(ProgramSourceId(1));
    diagnostic.suggestions[0].source = Some(ProgramSourceId(1));
    diagnostic
}

#[test]
fn project_diagnostic_json_and_text_use_each_span_source() {
    let project = project();
    let sources = CheckSources::Project(&project, Path::new("project/driver.m"));
    let diagnostic = diagnostic(&project);
    let json = serde_json::to_value(project_diagnostic(
        &diagnostic,
        runmat_static_analysis::frontend::DiagnosticStage::Semantic,
        |id| sources.source(id),
    ))
    .unwrap();
    assert_eq!(json["primary"]["path"], "project/driver.m");
    assert_eq!(json["primary"]["source_id"], 0);
    assert_eq!(json["primary"]["line"], 3);
    assert_eq!(json["primary"]["column"], 5);
    assert_eq!(
        json["secondary"][0]["path"],
        "project/incompatible_product.m"
    );
    assert_eq!(json["secondary"][0]["source_id"], 1);
    assert_eq!(json["secondary"][0]["line"], 2);
    assert_eq!(json["secondary"][0]["column"], 5);
    assert_eq!(
        json["suggestions"][0]["span"]["path"],
        "project/incompatible_product.m"
    );
    assert_eq!(json["suggestions"][0]["span"]["line"], 2);
    let text = render_check_diagnostic_with_sources(&diagnostic, &sources, &StreamStyles::plain());
    assert!(text.contains("--> project/driver.m:3:5"));
    assert!(text.contains("3 | C = incompatible_product(A, A);"));
    assert!(text.contains("related: project/incompatible_product.m:2:5: callee operation"));
}

#[test]
fn explicit_unknown_diagnostic_source_never_uses_driver_coordinates() {
    let project = project();
    let sources = CheckSources::Project(&project, Path::new("project/driver.m"));
    let mut diagnostic = diagnostic(&project);
    diagnostic.primary.source = Some(ProgramSourceId(99));
    diagnostic.secondary[0].source = Some(ProgramSourceId(98));
    diagnostic.suggestions[0].source = Some(ProgramSourceId(97));
    let json = serde_json::to_value(project_diagnostic(
        &diagnostic,
        runmat_static_analysis::frontend::DiagnosticStage::Semantic,
        |id| sources.source(id),
    ))
    .unwrap();
    for span in [
        &json["primary"],
        &json["secondary"][0],
        &json["suggestions"][0]["span"],
    ] {
        assert!(span["path"].is_null());
        assert!(span["line"].is_null());
        assert!(span["column"].is_null());
    }
    assert_eq!(json["primary"]["source_id"], 99);
    let text = render_check_diagnostic_with_sources(&diagnostic, &sources, &StreamStyles::plain());
    assert!(text.contains("<unavailable source 99>:bytes"));
    assert!(text.contains("related: <unavailable source 98>:bytes"));
    assert!(!text.contains("project/driver.m"));
}

#[test]
fn primary_display_path_does_not_replace_absolute_analysis_identity() {
    let mut project = project();
    project.sources[0].path = "/absolute/project/driver.m".into();
    let sources = CheckSources::Project(&project, Path::new("driver.m"));
    let diagnostic = diagnostic(&project);
    let json = serde_json::to_value(project_diagnostic(
        &diagnostic,
        runmat_static_analysis::frontend::DiagnosticStage::Semantic,
        |id| sources.source(id),
    ))
    .unwrap();
    assert_eq!(json["primary"]["path"], "driver.m");
    assert_eq!(
        json["secondary"][0]["path"],
        "project/incompatible_product.m"
    );
    assert_eq!(
        project.sources[0].path,
        Path::new("/absolute/project/driver.m")
    );
    let text = render_check_diagnostic_with_sources(&diagnostic, &sources, &StreamStyles::plain());
    assert!(text.contains("--> driver.m:3:5"));
    assert!(!text.contains("/absolute/project/driver.m"));
    let paths = project_source_paths(&project, Path::new("driver.m"));
    assert_eq!(paths[0], json!({ "source_id": 0, "path": "driver.m" }));
    assert_eq!(
        paths[1],
        json!({ "source_id": 1, "path": "project/incompatible_product.m" })
    );
    assert!(paths
        .iter()
        .all(|source| source.as_object().unwrap().len() == 2));
}
