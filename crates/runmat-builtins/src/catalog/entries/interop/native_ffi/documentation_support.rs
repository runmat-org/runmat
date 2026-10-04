use crate::*;

pub(super) const NATIVE_HOST: BuiltinExampleRequirements = BuiltinExampleRequirements {
    host: BuiltinExampleHostRequirement::NativeOnly,
    runtime: &[BuiltinRuntimeCapability::NativeFfi],
    toolchain: &[BuiltinToolchainCapability::CCompiler],
    ..BuiltinExampleRequirements::NONE
};

const FILES: BuiltinFilesystemFixture = BuiltinFilesystemFixture {
    id: BuiltinExampleFixtureId { local_name: "native-ffi-library" },
    root: BuiltinFilesystemRoot::IsolatedWorkspace,
    entries: &[
        BuiltinFilesystemEntry::File {
            relative_path: "fixture.h",
            content: BuiltinFixtureContent::Utf8("#include <stdint.h>\nint32_t fixture_add(int32_t left, int32_t right);\n"),
        },
        BuiltinFilesystemEntry::File {
            relative_path: "fixture.c",
            content: BuiltinFixtureContent::Utf8("#include \"fixture.h\"\nint32_t fixture_add(int32_t left, int32_t right) { return left + right; }\n"),
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
            include_directories: &["."],
            definitions: &[],
            interface: BuiltinNativeInterfacePreparation {
                interface_name: "fixture",
                primary_header: "fixture.h",
                additional_headers: &[],
                include_directories: &["."],
                definitions: &[],
            },
        }),
    });

pub(super) const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "Native FFI builtin dispatch",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/interop/native_ffi.rs"),
    },
    BuiltinDocumentationLink {
        label: "Native FFI session adapter",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/foreign/native_ffi/adapter.rs"),
    },
];
