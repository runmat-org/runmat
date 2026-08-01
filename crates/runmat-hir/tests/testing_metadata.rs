use std::collections::HashMap;

use runmat_builtins::ClassDef;
use runmat_hir::{lower, ClassKind, LoweringContext};

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
    assert_eq!(class.declared_attributes[0].name, "TestTags");
    assert_eq!(
        class.declared_attributes[0].value.as_deref(),
        Some("{'fast','unit'}")
    );
    assert_eq!(
        &source[class.declared_attributes[0].span.start..class.declared_attributes[0].span.end],
        "TestTags={'fast','unit'}"
    );

    let property = &class.properties[0];
    assert_eq!(property.declared_attributes[0].name, "TestParameter");
    assert_eq!(
        &source[property.span.start..property.span.end],
        "Mode = 'fast'"
    );

    let method = &class.methods[0];
    assert_eq!(method.declared_attributes[0].name, "Test");
    assert_eq!(
        method.declared_attributes[1].value.as_deref(),
        Some("'sequential'")
    );
}

#[test]
fn any_registered_builtin_superclass_uses_shared_hierarchy_metadata() {
    runmat_builtins::register_class(ClassDef {
        name: "runmat.testing.MetadataBase".into(),
        parent: Some("handle".into()),
        properties: HashMap::new(),
        methods: HashMap::new(),
    });
    let assembly = lower_source("classdef SampleTest < runmat.testing.MetadataBase\nend");
    let class = &assembly.classes[0];
    assert_eq!(
        class.builtin_super_class.as_deref(),
        Some("runmat.testing.MetadataBase")
    );
    assert!(matches!(class.kind, ClassKind::Handle));
}
