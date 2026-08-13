use runmat_types::MemberAccess;
use std::cell::Cell;
use std::collections::HashMap;

use crate::class_registry::{RuntimeClass, RuntimeMethod, RuntimeProperty};
use runmat_value::Value;

use super::objects::{
    FUNCTION_TEST_CASE_CLASS, TEST_CASE_CLASS, TEST_RESULT_CLASS, TEST_SUITE_CLASS,
};

thread_local! {
    static REGISTERED: Cell<bool> = const { Cell::new(false) };
}

pub fn ensure_testing_classes() {
    REGISTERED.with(|registered| {
        if registered.replace(true) {
            return;
        }
        register_test_case();
        register_plain_class(
            TEST_SUITE_CLASS,
            None,
            &["Name", "ProcedureName", "TestFile", "Tags"],
        );
        register_plain_class(
            TEST_RESULT_CLASS,
            None,
            &[
                "Name",
                "TestFile",
                "Passed",
                "Failed",
                "Incomplete",
                "Duration",
                "Details",
                "Flaky",
            ],
        );
        register_class_with_methods(
            "matlab.unittest.TestRunner",
            Some("handle"),
            &["Plugins"],
            &["addPlugin", "run"],
        );
        register_plain_class(
            "matlab.unittest.fixtures.Fixture",
            Some("handle"),
            &["Name"],
        );
        register_plain_class(
            "matlab.unittest.fixtures.TemporaryFolderFixture",
            Some("matlab.unittest.fixtures.Fixture"),
            &["Folder"],
        );
        register_plain_class(
            "matlab.unittest.fixtures.PathFixture",
            Some("matlab.unittest.fixtures.Fixture"),
            &["Path"],
        );
        register_plain_class(
            "matlab.unittest.diagnostics.Diagnostic",
            None,
            &["Identifier", "Message"],
        );
        register_plain_class("matlab.unittest.constraints.Constraint", None, &[]);
        for class_name in [
            "matlab.unittest.constraints.IsEqualTo",
            "matlab.unittest.constraints.IsTrue",
            "matlab.unittest.constraints.IsFalse",
        ] {
            register_plain_class(
                class_name,
                Some("matlab.unittest.constraints.Constraint"),
                &[],
            );
        }
        register_plain_class(
            "matlab.unittest.plugins.TestRunnerPlugin",
            Some("handle"),
            &[],
        );
        register_plain_class(
            "matlab.unittest.plugins.CodeCoveragePlugin",
            Some("matlab.unittest.plugins.TestRunnerPlugin"),
            &["Folders", "IncludingSubfolders"],
        );
    });
}

fn register_test_case() {
    let methods = qualification_methods()
        .chain(["addTeardown", "applyFixture", "log"])
        .map(|name| {
            (
                name.to_string(),
                RuntimeMethod {
                    name: name.to_string(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: format!("{TEST_CASE_CLASS}.{name}"),
                    implicit_class_argument: None,
                },
            )
        })
        .collect();
    crate::class_registry::register_class(RuntimeClass {
        name: TEST_CASE_CLASS.into(),
        parent: Some("handle".into()),
        properties: properties(&["Name"]),
        methods,
    });
    crate::class_registry::register_class(RuntimeClass {
        name: FUNCTION_TEST_CASE_CLASS.into(),
        parent: Some(TEST_CASE_CLASS.into()),
        properties: HashMap::new(),
        methods: HashMap::new(),
    });
}

fn qualification_methods() -> impl Iterator<Item = &'static str> {
    [
        "verifyEqual",
        "verifyNotEqual",
        "verifyTrue",
        "verifyFalse",
        "verifyGreaterThan",
        "verifyLessThan",
        "verifyEmpty",
        "verifyNotEmpty",
        "assertEqual",
        "assertTrue",
        "assumeEqual",
        "assumeTrue",
        "fatalAssertTrue",
        "verifyThat",
        "assertThat",
    ]
    .into_iter()
}

fn register_plain_class(name: &str, parent: Option<&str>, property_names: &[&str]) {
    register_class_with_methods(name, parent, property_names, &[]);
}

fn register_class_with_methods(
    name: &str,
    parent: Option<&str>,
    property_names: &[&str],
    method_names: &[&str],
) {
    crate::class_registry::register_class(RuntimeClass {
        name: name.into(),
        parent: parent.map(str::to_owned),
        properties: properties(property_names),
        methods: method_names
            .iter()
            .map(|method_name| {
                (
                    (*method_name).into(),
                    RuntimeMethod {
                        name: (*method_name).into(),
                        is_static: false,
                        is_abstract: false,
                        is_sealed: false,
                        access: MemberAccess::Public,
                        function_name: format!("{name}.{method_name}"),
                        implicit_class_argument: None,
                    },
                )
            })
            .collect(),
    });
}

fn properties(names: &[&str]) -> HashMap<String, RuntimeProperty> {
    names
        .iter()
        .map(|name| {
            (
                (*name).into(),
                RuntimeProperty {
                    name: (*name).into(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: MemberAccess::Public,
                    set_access: MemberAccess::Private,
                    default_value: Some(Value::String(String::new())),
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_testing_inheritance_and_qualification_methods() {
        ensure_testing_classes();
        assert!(crate::class_registry::is_class_or_subclass(
            FUNCTION_TEST_CASE_CLASS,
            TEST_CASE_CLASS
        ));
        assert!(crate::class_registry::is_class_or_subclass(
            TEST_CASE_CLASS,
            "handle"
        ));
        assert!(crate::class_registry::lookup_method(TEST_CASE_CLASS, "verifyEqual").is_some());
    }
}
