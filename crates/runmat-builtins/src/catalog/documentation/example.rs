use serde::Serialize;

mod fixture;
mod requirements;

pub use fixture::*;
pub use requirements::*;

/// Execution lane used by the standalone documentation-example verifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinExampleHarness {
    /// Must agree in the native runtime and the browser/WASM runtime.
    Portable,
    Native,
    Browser,
    BrowserGraphics,
    NativeFilesystem,
    NativeLoopbackNetwork,
    Wgpu,
    NativeForeignRuntime,
    InteractiveHost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinExampleCompatibility {
    RunMat,
    Matlab,
    Strict,
}

/// Semantic oracle for a documentation example. Presentation output is kept
/// separately so formatting changes do not silently redefine correctness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinExampleVerification {
    Succeeds,
    Assertions {
        source: &'static str,
    },
    ExpectedError {
        identifier: &'static str,
    },
    Figure {
        minimum_figures: usize,
        assertions: &'static str,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinExample {
    /// Stable within the owning builtin identity; export combines both values.
    pub id: &'static str,
    pub title: &'static str,
    pub program: &'static str,
    pub display_output: Option<&'static str>,
    pub compatibility: BuiltinExampleCompatibility,
    pub harness: BuiltinExampleHarness,
    /// Deterministic resources materialized for this example. `None` is an
    /// explicit declaration that the program supplies everything it needs.
    pub fixture: BuiltinExampleFixture,
    /// Typed execution prerequisites checked before a verifier admits the
    /// example to a lane.
    pub requirements: BuiltinExampleRequirements,
    pub verification: BuiltinExampleVerification,
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILES: &[BuiltinFilesystemEntry] = &[BuiltinFilesystemEntry::File {
        relative_path: "gateway.f90",
        content: BuiltinFixtureContent::Utf8("subroutine mexFunction()\nend subroutine\n"),
    }];
    const UNITS: &[BuiltinNativeTranslationUnit] = &[BuiltinNativeTranslationUnit {
        relative_path: "gateway.f90",
        language: BuiltinNativeSourceLanguage::Fortran,
    }];

    #[test]
    fn fixture_wire_schema_is_closed_and_explicit() {
        let example = BuiltinExample {
            id: "native-source",
            title: "Compile native source",
            program: "result = adapter();",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::NativeForeignRuntime,
            fixture: BuiltinExampleFixture::ForeignAdapter(BuiltinForeignAdapterFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "native-source",
                },
                files: BuiltinFilesystemFixture {
                    id: BuiltinExampleFixtureId {
                        local_name: "sources",
                    },
                    root: BuiltinFilesystemRoot::IsolatedWorkspace,
                    entries: FILES,
                },
                preparation: BuiltinForeignPreparation::Mex(BuiltinMexPreparation {
                    module_name: "native_source",
                    api: BuiltinMexApi::R2017b,
                    translation_units: UNITS,
                    include_directories: &[],
                    definitions: &[],
                }),
            }),
            requirements: BuiltinExampleRequirements {
                host: BuiltinExampleHostRequirement::NativeOnly,
                engine: BuiltinExampleEngine::Aot,
                compiler: &[BuiltinCompilerCapability::Fortran],
                runtime: &[BuiltinRuntimeCapability::Mex],
                toolchain: &[BuiltinToolchainCapability::FortranCompiler],
            },
            verification: BuiltinExampleVerification::Succeeds,
        };
        let encoded = serde_json::to_value(example).expect("serialize example");
        assert_eq!(
            encoded["fixture"]["ForeignAdapter"]["preparation"]["Mex"]["api"],
            "R2017b"
        );
        assert!(encoded["fixture"]["ForeignAdapter"]
            .get("isolation")
            .is_none());
        assert_eq!(encoded["requirements"]["engine"], "Aot");
    }

    #[test]
    fn no_fixture_state_is_serialized_on_every_target() {
        assert_eq!(
            serde_json::to_value(BuiltinExampleFixture::None).expect("serialize no fixture"),
            "None"
        );
        assert_eq!(
            serde_json::to_value(BuiltinExampleRequirements::NONE)
                .expect("serialize no requirements"),
            serde_json::json!({
                "host": "Any",
                "engine": "Default",
                "compiler": [],
                "runtime": [],
                "toolchain": []
            })
        );
    }

    #[test]
    fn endpoint_substitutions_have_fixed_source_tokens() {
        assert_eq!(
            BuiltinEndpointSubstitution::HttpBaseUrl.source_token(),
            "__RUNMAT_HTTP_BASE_URL__"
        );
        assert_eq!(
            BuiltinEndpointSubstitution::LoopbackHost.source_token(),
            "__RUNMAT_LOOPBACK_HOST__"
        );
        assert_eq!(
            BuiltinEndpointSubstitution::LoopbackPort.source_token(),
            "__RUNMAT_LOOPBACK_PORT__"
        );
    }
}
