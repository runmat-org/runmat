//! A cloud project's studies have virtual paths (`/simulation/bracket.fea`). The language server
//! reads them through the installed filesystem provider, and a study may read files up to its
//! project root but not above it. Each test installs an in-memory provider; none of these paths
//! exist on the test machine's disk.
#![cfg(feature = "native")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use runmat_filesystem::MemoryFsProvider;
use runmat_lsp::core::analysis::CompatMode;
use runmat_lsp::core::fea::{analyze_fea_source, FeaDocumentAnalysis};
use tower_lsp::lsp_types::{
    DidOpenTextDocumentParams, InitializeParams, NumberOrString, TextDocumentItem, Url,
    WorkspaceSymbolParams,
};
use tower_lsp::LanguageServer;

const STEEL: &str = r#"material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
}
"#;

/// `/cloud/project/materials.fea`, a part under `/cloud/project/mechanical/`, and
/// `/cloud/outside.fea`, above the project.
fn cloud_files() -> Arc<MemoryFsProvider> {
    let provider = MemoryFsProvider::new();
    provider
        .write_project_path("/cloud/project/materials.fea", STEEL.as_bytes())
        .unwrap();
    provider
        .write_project_path("/cloud/outside.fea", STEEL.as_bytes())
        .unwrap();
    provider
        .write_project_path("/materials.fea", STEEL.as_bytes())
        .unwrap();
    provider
        .write_project_path(
            "/cloud/project/mechanical/body/landing_skid.part",
            b"part landing_skid {}\n",
        )
        .unwrap();
    Arc::new(provider)
}

fn with_cloud_files<R>(test: impl FnOnce() -> R) -> R {
    let _lock = runmat_filesystem::provider_override_lock();
    let _provider = runmat_filesystem::replace_provider(cloud_files());
    test()
}

fn check(source: &str, path: &str, roots: &[PathBuf]) -> FeaDocumentAnalysis {
    futures::executor::block_on(analyze_fea_source(
        source,
        Path::new(path),
        roots,
        &BTreeMap::new(),
        CompatMode::RunMat,
    ))
    .unwrap()
}

fn refused(analysis: &FeaDocumentAnalysis, source: &str) -> bool {
    analysis.diagnostics(source).iter().any(|diagnostic| {
        diagnostic.code == Some(NumberOrString::String("fea.module.invalid_path".into()))
            && diagnostic
                .message
                .contains("escapes the declared source root")
    })
}

/// The imported material was read, and from which file.
fn steel_source(analysis: &FeaDocumentAnalysis) -> Option<String> {
    analysis
        .workspace_symbols(Some("steel"))
        .into_iter()
        .map(|symbol| symbol.location.uri.path().to_string())
        .find(|path| path.ends_with("materials.fea"))
}

#[test]
fn a_cloud_study_reads_up_to_its_project_root() {
    with_cloud_files(|| {
        let source = "import material steel from \"../materials.fea\"\n";
        let analysis = check(
            source,
            "/cloud/project/simulation/bracket.fea",
            &[PathBuf::from("/cloud/project")],
        );
        assert!(
            !refused(&analysis, source),
            "{:#?}",
            analysis.diagnostics(source)
        );
        assert_eq!(
            steel_source(&analysis).as_deref(),
            Some("/cloud/project/materials.fea")
        );
    });
}

#[test]
fn a_cloud_study_cannot_read_above_its_project_root() {
    with_cloud_files(|| {
        let source = "import material steel from \"../../outside.fea\"\n";
        let analysis = check(
            source,
            "/cloud/project/simulation/bracket.fea",
            &[PathBuf::from("/cloud/project")],
        );
        assert!(
            refused(&analysis, source),
            "{:#?}",
            analysis.diagnostics(source)
        );
        assert_eq!(steel_source(&analysis), None);
    });
}

#[test]
fn without_roots_a_cloud_study_resolves_from_the_provider_root() {
    with_cloud_files(|| {
        let source = "import material steel from \"../materials.fea\"\n";
        let analysis = check(source, "/simulation/bracket.fea", &[]);
        assert!(
            !refused(&analysis, source),
            "{:#?}",
            analysis.diagnostics(source)
        );
        assert_eq!(steel_source(&analysis).as_deref(), Some("/materials.fea"));

        let source = "import material steel from \"../../materials.fea\"\n";
        let analysis = check(source, "/simulation/bracket.fea", &[]);
        assert!(
            refused(&analysis, source),
            "{:#?}",
            analysis.diagnostics(source)
        );
    });
}

/// A study at `simulation/skid.fea` that meshes `../mechanical/body/landing_skid.part`.
fn skid_study(part: &str) -> String {
    format!("geometry skid {{\n    file = \"{part}\"\n    units = mm\n}}\n")
}

/// Whether the study's geometry `file =` was refused as escaping its source root.
fn geometry_escapes(analysis: &FeaDocumentAnalysis, source: &str) -> bool {
    analysis.diagnostics(source).iter().any(|diagnostic| {
        diagnostic.code
            == Some(NumberOrString::String(
                "fea.inspection.invalid_geometry_path".into(),
            ))
            && diagnostic
                .message
                .contains("escapes the declared source root")
    })
}

#[test]
fn a_study_in_a_subfolder_meshes_a_part_elsewhere_in_the_project() {
    with_cloud_files(|| {
        let source = skid_study("../mechanical/body/landing_skid.part");
        let analysis = check(
            &source,
            "/cloud/project/simulation/skid.fea",
            &[PathBuf::from("/cloud/project")],
        );
        assert!(
            !geometry_escapes(&analysis, &source),
            "{:#?}",
            analysis.diagnostics(&source)
        );

        let source = skid_study("../../mechanical/body/landing_skid.part");
        let analysis = check(
            &source,
            "/cloud/project/simulation/skid.fea",
            &[PathBuf::from("/cloud/project")],
        );
        assert!(
            geometry_escapes(&analysis, &source),
            "{:#?}",
            analysis.diagnostics(&source)
        );
    });
}

#[test]
fn the_native_server_reads_a_cloud_study_through_the_provider() {
    with_cloud_files(|| {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                let (service, _socket) = runmat_lsp::native::service();
                let server = service.inner();
                server
                    .initialize(InitializeParams {
                        root_uri: Some(Url::parse("file:///cloud/project/").unwrap()),
                        ..InitializeParams::default()
                    })
                    .await
                    .unwrap();
                server
                    .did_open(DidOpenTextDocumentParams {
                        text_document: TextDocumentItem {
                            uri: Url::parse("file:///cloud/project/simulation/bracket.fea")
                                .unwrap(),
                            language_id: "runmat-fea".into(),
                            version: 1,
                            text: "import material steel from \"../materials.fea\"\n".into(),
                        },
                    })
                    .await;
                let symbols = server
                    .symbol(WorkspaceSymbolParams {
                        query: "steel".into(),
                        work_done_progress_params: Default::default(),
                        partial_result_params: Default::default(),
                    })
                    .await
                    .unwrap()
                    .unwrap_or_default();
                assert!(
                    symbols.iter().any(|symbol| {
                        symbol.location.uri.as_str() == "file:///cloud/project/materials.fea"
                    }),
                    "{symbols:#?}"
                );
            });
    });
}
