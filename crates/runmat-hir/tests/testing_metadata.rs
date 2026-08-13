use runmat_hir::{lower, ClassKind, LoweringContext};
use runmat_types::{ExternalClassDeclaration, QualifiedName, SymbolName};

fn lower_source(source: &str) -> runmat_hir::HirAssembly {
    let program = runmat_parser::parse(source).expect("parse source");
    lower(&program, &LoweringContext::empty())
        .expect("lower source")
        .assembly
}

#[test]
fn script_sections_and_general_test_attributes_survive_semantic_lowering() {
    let source = "%% first\nx = 1;\n%% second\ny = 2;\nclassdef (TestTags={'fast','unit'}) SampleTest\n  properties(TestParameter)\n    Mode = 'fast'\n  end\n  methods(Test, ParameterCombination='sequential')\n    function testValue(obj)\n    end\n  end\nend";
    let assembly = lower_source(source);
    assert_eq!(assembly.modules[0].script_sections.len(), 2);
    assert_eq!(assembly.modules[0].script_sections[0].title, "first");
    assert_eq!(assembly.modules[0].script_sections[1].ordinal, 2);

    let class = &assembly.classes[0];
    assert_eq!(class.declaration.declared_attributes[0].name, "TestTags");
    assert_eq!(
        class.declaration.declared_attributes[0].value.as_deref(),
        Some("{'fast','unit'}")
    );
    assert_eq!(
        &source[class.declaration.declared_attributes[0].span.start
            ..class.declaration.declared_attributes[0].span.end],
        "TestTags={'fast','unit'}"
    );

    let property = &class.declaration.properties[0];
    assert_eq!(property.declared_attributes[0].name, "TestParameter");
    assert_eq!(
        &source[property.span.start..property.span.end],
        "Mode = 'fast'"
    );

    let method = &class.declaration.methods[0];
    assert_eq!(method.declared_attributes[0].name, "Test");
    assert_eq!(
        method.declared_attributes[1].value.as_deref(),
        Some("'sequential'")
    );
}

#[test]
fn external_superclass_uses_shared_immutable_hierarchy_metadata() {
    let declaration = ExternalClassDeclaration {
        name: QualifiedName(vec![
            SymbolName("runmat".into()),
            SymbolName("testing".into()),
            SymbolName("MetadataBase".into()),
        ]),
        parent: Some(QualifiedName(vec![SymbolName("handle".into())])),
        kind: ClassKind::Handle,
        is_sealed: false,
        is_abstract: false,
        properties: Vec::new(),
        methods: Vec::new(),
    };
    let program = runmat_parser::parse("classdef SampleTest < runmat.testing.MetadataBase\nend")
        .expect("parse source");
    let declarations = [declaration];
    let assembly = lower(
        &program,
        &LoweringContext::empty().with_external_class_declarations(&declarations),
    )
    .expect("lower source")
    .assembly;
    let class = &assembly.classes[0];
    assert_eq!(
        class.declaration.inheritance.builtin_super_class.as_deref(),
        Some("runmat.testing.MetadataBase")
    );
    assert!(matches!(class.declaration.kind, ClassKind::Handle));
}
