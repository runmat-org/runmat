use crate::*;

pub(super) const PYTHON_HOST: BuiltinExampleRequirements = BuiltinExampleRequirements {
    host: BuiltinExampleHostRequirement::NativeOnly,
    runtime: &[BuiltinRuntimeCapability::Python],
    toolchain: &[BuiltinToolchainCapability::PythonInterpreter],
    ..BuiltinExampleRequirements::NONE
};
const PYTHON_FILES: BuiltinFilesystemFixture = BuiltinFilesystemFixture {
    id: BuiltinExampleFixtureId {
        local_name: "python-basic-files",
    },
    root: BuiltinFilesystemRoot::IsolatedWorkspace,
    entries: &[
        BuiltinFilesystemEntry::File {
            relative_path: "answer.py",
            content: BuiltinFixtureContent::Utf8("answer = input_value + 2\n"),
        },
        BuiltinFilesystemEntry::File {
            relative_path: "pyfixture/marker.py",
            content: BuiltinFixtureContent::Utf8("def add2(value):\n    return value + 2\n"),
        },
    ],
};
const PYTHON_SOURCE: BuiltinPythonArtifact = BuiltinPythonArtifact::SourceTree {
    module_root: "pyfixture",
    modules: &["marker"],
};
const PYTHON_ENV: BuiltinPythonEnvironment = BuiltinPythonEnvironment {
    implementation: BuiltinPythonImplementation::Cpython,
    major: 3,
    minor: 12,
};
pub(super) const IN_PROCESS_FIXTURE: BuiltinExampleFixture =
    BuiltinExampleFixture::ForeignAdapter(BuiltinForeignAdapterFixture {
        id: BuiltinExampleFixtureId {
            local_name: "python-inprocess",
        },
        files: PYTHON_FILES,
        preparation: BuiltinForeignPreparation::Python(BuiltinPythonPreparation {
            isolation: BuiltinForeignIsolation::InProcess,
            environment: PYTHON_ENV,
            artifact: PYTHON_SOURCE,
        }),
    });
pub(super) const OUT_OF_PROCESS_FIXTURE: BuiltinExampleFixture =
    BuiltinExampleFixture::ForeignAdapter(BuiltinForeignAdapterFixture {
        id: BuiltinExampleFixtureId {
            local_name: "python-outofprocess",
        },
        files: PYTHON_FILES,
        preparation: BuiltinForeignPreparation::Python(BuiltinPythonPreparation {
            isolation: BuiltinForeignIsolation::OutOfProcess,
            environment: PYTHON_ENV,
            artifact: PYTHON_SOURCE,
        }),
    });
pub(super) const PYTHON_LINK: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Python interoperability",
    target: BuiltinDocumentationLinkTarget::Documentation("/docs/execution/python"),
};
pub(super) const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Python builtins and session foreign service",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/interop/python.rs"),
}];
