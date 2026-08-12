use std::cell::Cell;
use std::collections::HashMap;

use runmat_builtins::{ClassDef, MethodDef, PropertyDef};
use runmat_value::{Access, Value};

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
                MethodDef {
                    name: name.to_string(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: Access::Public,
                    function_name: format!("{TEST_CASE_CLASS}.{name}"),
                    implicit_class_argument: None,
                },
            )
        })
        .collect();
    runmat_builtins::register_class(ClassDef {
        name: TEST_CASE_CLASS.into(),
        parent: Some("handle".into()),
        properties: properties(&["Name"]),
        methods,
    });
    runmat_builtins::register_class(ClassDef {
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
    runmat_builtins::register_class(ClassDef {
        name: name.into(),
        parent: parent.map(str::to_owned),
        properties: properties(property_names),
        methods: method_names
            .iter()
            .map(|method_name| {
                (
                    (*method_name).into(),
                    MethodDef {
                        name: (*method_name).into(),
                        is_static: false,
                        is_abstract: false,
                        is_sealed: false,
                        access: Access::Public,
                        function_name: format!("{name}.{method_name}"),
                        implicit_class_argument: None,
                    },
                )
            })
            .collect(),
    });
}

fn properties(names: &[&str]) -> HashMap<String, PropertyDef> {
    names
        .iter()
        .map(|name| {
            (
                (*name).into(),
                PropertyDef {
                    name: (*name).into(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: Access::Public,
                    set_access: Access::Private,
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
        assert!(runmat_builtins::is_class_or_subclass(
            FUNCTION_TEST_CASE_CLASS,
            TEST_CASE_CLASS
        ));
        assert!(runmat_builtins::is_class_or_subclass(
            TEST_CASE_CLASS,
            "handle"
        ));
        assert!(runmat_builtins::lookup_method(TEST_CASE_CLASS, "verifyEqual").is_some());
    }
}
