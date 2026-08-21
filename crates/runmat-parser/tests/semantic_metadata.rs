use runmat_parser::{parse, ClassMember, Stmt};

#[test]
fn preserves_script_section_titles_ordinals_and_exact_spans() {
    let source = "shared = 1;\n%% first case\nx = 1;\n%%\ny = 2;\n";
    let program = parse(source).expect("parse sectioned script");
    assert_eq!(program.sections.len(), 2);

    let first = &program.sections[0];
    assert_eq!(first.ordinal, 1);
    assert_eq!(first.title, "first case");
    assert_eq!(
        &source[first.marker_span.start..first.marker_span.end],
        "%% first case"
    );
    assert_eq!(
        &source[first.body_span.start..first.body_span.end],
        "x = 1;\n"
    );

    let second = &program.sections[1];
    assert_eq!(second.ordinal, 2);
    assert!(second.title.is_empty());
    assert_eq!(
        &source[second.marker_span.start..second.marker_span.end],
        "%%"
    );
    assert_eq!(
        &source[second.body_span.start..second.body_span.end],
        "y = 2;\n"
    );
}

#[test]
fn preserves_general_attribute_values_and_member_spans() {
    let source = "classdef (TestTags={'fast','unit'}, SharedTestFixtures={fixtureA}) SampleTest < handle\n  properties(TestParameter)\n    Mode = 'fast'\n  end\n  methods(Test, ParameterCombination='sequential')\n    function testValue(obj)\n    end\n  end\n  events(NotifyAccess=protected)\n    Changed\n  end\nend";
    let program = parse(source).expect("parse attributed test class");
    let Stmt::ClassDef {
        attributes,
        members,
        ..
    } = &program.body[0]
    else {
        panic!("expected classdef");
    };
    assert_eq!(attributes[0].name, "TestTags");
    assert_eq!(attributes[0].value.as_deref(), Some("{'fast','unit'}"));
    assert_eq!(
        &source[attributes[0].span.start..attributes[0].span.end],
        "TestTags={'fast','unit'}"
    );
    assert_eq!(attributes[1].value.as_deref(), Some("{fixtureA}"));

    let ClassMember::Properties { names, span, .. } = &members[0] else {
        panic!("expected properties");
    };
    assert!(source[span.start..span.end].starts_with("properties(TestParameter)"));
    assert_eq!(
        &source[names[0].span.start..names[0].span.end],
        "Mode = 'fast'"
    );

    let ClassMember::Methods {
        attributes, span, ..
    } = &members[1]
    else {
        panic!("expected methods");
    };
    assert_eq!(attributes[1].value.as_deref(), Some("'sequential'"));
    assert!(source[span.start..span.end].starts_with("methods(Test,"));

    let ClassMember::Events { names, .. } = &members[2] else {
        panic!("expected events");
    };
    assert_eq!(names[0].name, "Changed");
    assert_eq!(&source[names[0].span.start..names[0].span.end], "Changed");
}
