use crate::*;

pub(super) const NATIVE_HOST: BuiltinExampleRequirements = BuiltinExampleRequirements {
    host: BuiltinExampleHostRequirement::NativeOnly,
    runtime: &[BuiltinRuntimeCapability::NativeFfi],
    compiler: &[BuiltinCompilerCapability::C],
    toolchain: &[BuiltinToolchainCapability::CCompiler],
    ..BuiltinExampleRequirements::NONE
};

const FILES: BuiltinFilesystemFixture = BuiltinFilesystemFixture {
    id: BuiltinExampleFixtureId { local_name: "native-ffi-library" },
    root: BuiltinFilesystemRoot::IsolatedWorkspace,
    entries: &[
        BuiltinFilesystemEntry::File {
            relative_path: "fixture.c",
            content: BuiltinFixtureContent::Utf8("#include \"fixture.h\"\nint32_t fixture_add(int32_t left, int32_t right) { return left + right; }\nvoid fixture_increment(int32_t *value) { *value += 1; }\nint32_t fixture_record_sum(const fixture_record *value) { return value->left + value->right; }\nint32_t *fixture_borrowed_value(int32_t present) { static int32_t value = 17; return present ? &value : 0; }\n"),
        },
        BuiltinFilesystemEntry::File {
            relative_path: "fixture.h",
            content: BuiltinFixtureContent::Utf8("#include <stdint.h>\ntypedef struct fixture_record { int32_t left; int32_t right; } fixture_record;\nint32_t fixture_add(int32_t left, int32_t right);\nvoid fixture_increment(int32_t *value);\nint32_t fixture_record_sum(const fixture_record *value);\nint32_t *fixture_borrowed_value(int32_t present);\n"),
        },
    ],
};

pub(super) const NATIVE_FIXTURE: BuiltinExampleFixture =
    BuiltinExampleFixture::ForeignAdapter(BuiltinForeignAdapterFixture {
        id: BuiltinExampleFixtureId {
            local_name: "native-ffi-prepared",
        },
        files: FILES,
        preparation: BuiltinForeignPreparation::NativeFfi(BuiltinNativeFfiPreparation {
            isolation: BuiltinForeignIsolation::InProcess,
            library_name: "fixture",
            translation_units: &[BuiltinNativeTranslationUnit {
                relative_path: "fixture.c",
                language: BuiltinNativeSourceLanguage::C,
            }],
            include_directories: &[],
            definitions: &[],
            interface: BuiltinNativeInterfacePreparation {
                interface_name: "fixture",
                primary_header: "fixture.h",
                additional_headers: &[],
                include_directories: &[],
                definitions: &[],
            },
        }),
    });

const DISPATCH: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Native FFI public dispatch",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/interop/native_ffi.rs"),
};
pub(super) const LOAD_IMPLEMENTATION: &[BuiltinDocumentationLink] = &[
    DISPATCH,
    BuiltinDocumentationLink { label: "Prepared library and header ownership", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/foreign/native_ffi/adapter/library.rs") },
];
pub(super) const CALL_IMPLEMENTATION: &[BuiltinDocumentationLink] = &[
    DISPATCH,
    BuiltinDocumentationLink { label: "Prepared ABI invocation and output admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/foreign/native_ffi/adapter/call.rs") },
];
pub(super) const POINTER_IMPLEMENTATION: &[BuiltinDocumentationLink] = &[
    DISPATCH,
    BuiltinDocumentationLink { label: "Owned native pointer storage", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/foreign/native_ffi/adapter/pointer_storage.rs") },
    BuiltinDocumentationLink { label: "Borrowed native pointer view", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/foreign/native_ffi/adapter/pointer_view.rs") },
];
