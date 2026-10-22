mod support;

use runmat_analysis_language::*;
use support::{source_module, TestHost};

// witness: FEA-AUTHORING-COMPOSITION-001/explicit_imports @
//   shared_infrastructure/mesh_independent/source_check;
//   boundary: Mock host checks three-module transitive imports, explicit model/function binding and
//   symbol-use provenance; no geometry execution, numerical result or remote equivalence.
#[test]
fn compiler_loads_transitive_modules_and_binds_explicit_symbols() {
    let host = TestHost::default()
        .with_geometry("project/unused.step", 10)
        .with_function("project/pulse.m", "pulse", Vec::new(), Vec::new())
        .with_module(
            "project/models/bracket.fea",
            "import material steel from \"../materials/steel.fea\"\nmodel bracket {\n}\n",
            2,
        )
        .with_module(
            "project/materials/steel.fea",
            "material steel {\nname = \"steel\"\n}\n",
            3,
        );
    let root = source_module(
        "project/main.fea",
        "import model bracket from \"./models/bracket.fea\"\nimport function pulse from \"./pulse.m\"\nstudy main {\nmodel = model.bracket\n}\n",
        1,
    );
    let program = FeaCompiler::new(&host).check_bindings(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root,
        },
    });

    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    assert_eq!(program.graph.modules.len(), 3);
    assert_eq!(program.graph.edges.len(), 2);
    assert_eq!(program.graph.catalog_fingerprint, [7; 32]);
    let root_symbols = &program.symbols[&LogicalSourcePath::new("project/main.fea").unwrap()];
    assert!(root_symbols.contains_key(&FeaSymbolKey {
        kind: FeaSymbolKind::Model,
        name: "bracket".to_string(),
    }));
    assert!(root_symbols.contains_key(&FeaSymbolKey {
        kind: FeaSymbolKind::Function,
        name: "pulse".to_string(),
    }));
    assert_eq!(program.functions.len(), 1);
    assert_eq!(program.symbol_uses.len(), 1);
    assert_eq!(program.symbol_uses[0].module.as_str(), "project/main.fea");
    assert_eq!(
        program.symbol_uses[0].target_module.as_str(),
        "project/models/bracket.fea"
    );
    assert_eq!(program.symbol_uses[0].target.kind, FeaSymbolKind::Model);
    assert_eq!(program.symbol_uses[0].target.name, "bracket");
}

// witness: FEA-AUTHORING-COMPOSITION-001/explicit_imports @
//   shared_infrastructure/mesh_independent/source_check;
//   boundary: Mock-host binding diagnostics identify cyclic imports, duplicate symbols and missing
//   exports; no complete semantic or numerical qualification.
#[test]
fn compiler_reports_cycles_duplicates_and_missing_exports() {
    let host = TestHost::default()
        .with_module(
            "project/a.fea",
            "import model b from \"./b.fea\"\nmodel a {\n}\n",
            2,
        )
        .with_module(
            "project/b.fea",
            "import model a from \"./a.fea\"\nmodel b {\n}\n",
            3,
        )
        .with_module("project/materials.fea", "material aluminum {\n}\n", 4);
    let root = source_module(
        "project/main.fea",
        "import model a from \"./a.fea\"\nimport material steel from \"./materials.fea\"\nmaterial steel {\n}\nmaterial steel {\n}\n",
        1,
    );
    let program = FeaCompiler::new(&host).check_bindings(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root,
        },
    });

    let codes = program
        .graph
        .diagnostics
        .iter()
        .chain(&program.diagnostics)
        .map(|diagnostic| diagnostic.code.as_str())
        .collect::<Vec<_>>();
    assert!(codes.contains(&"fea.module.import_cycle"));
    assert!(codes.contains(&"fea.binding.missing_export"));
    assert!(codes.contains(&"fea.binding.duplicate_symbol"));
    assert!(!program.is_valid());
}

// witness: FEA-AUTHORING-COMPILER-002/module_graph_pass @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: An import that escapes the declared source root is rejected with
//   fea.module.invalid_path; cycle and digest retention are covered by sibling tests.
#[test]
fn compiler_rejects_imports_outside_the_declared_source_root() {
    let host = TestHost::default();
    let root = source_module(
        "project/main.fea",
        "import material steel from \"../steel.fea\"\n",
        1,
    );
    let program = FeaCompiler::new(&host).check_bindings(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root,
        },
    });
    assert!(program
        .graph
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "fea.module.invalid_path"));
    assert!(!program.is_valid());
}
