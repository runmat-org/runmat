use super::*;
use crate::{
    BuiltinCliTranscriptStep, BuiltinCompilerCapability, BuiltinEndpointSubstitution,
    BuiltinExampleCompatibility, BuiltinExampleRequirements, BuiltinExampleVerification,
    BuiltinFilesystemEntry, BuiltinFilesystemFixture, BuiltinFilesystemRoot, BuiltinFixtureContent,
    BuiltinForeignAdapterFixture, BuiltinForeignPreparation, BuiltinHttpExchange,
    BuiltinHttpMethod, BuiltinHttpRequest, BuiltinHttpResponse, BuiltinHttpScenario,
    BuiltinLoopbackFixture, BuiltinMexApi, BuiltinMexPreparation, BuiltinNativeSourceLanguage,
    BuiltinNativeTranslationUnit, BuiltinRuntimeCapability, BuiltinToolchainCapability,
};

const HTTP_EXCHANGES: &[BuiltinHttpExchange] = &[BuiltinHttpExchange {
    request: BuiltinHttpRequest {
        method: BuiltinHttpMethod::Get,
        path: "/value",
        body: None,
    },
    response: BuiltinHttpResponse {
        status: 200,
        headers: &[],
        body: b"ok",
    },
}];

fn loopback_example() -> BuiltinExample {
    BuiltinExample {
        id: "loopback",
        title: "Use loopback HTTP",
        program: "url = \"__RUNMAT_HTTP_BASE_URL__/value\";",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeLoopbackNetwork,
        fixture: BuiltinExampleFixture::Loopback(BuiltinLoopbackFixture {
            id: crate::BuiltinExampleFixtureId {
                local_name: "http-value",
            },
            scenario: crate::BuiltinLoopbackScenario::Http(BuiltinHttpScenario {
                exchanges: HTTP_EXCHANGES,
            }),
            endpoint_substitutions: &[BuiltinEndpointSubstitution::HttpBaseUrl],
        }),
        requirements: BuiltinExampleRequirements {
            host: BuiltinExampleHostRequirement::NativeOnly,
            ..BuiltinExampleRequirements::NONE
        },
        verification: BuiltinExampleVerification::Succeeds,
    }
}

#[test]
fn matching_loopback_fixture_is_valid() {
    let mut errors = Vec::new();
    validate("webread", &loopback_example(), &mut errors);
    assert!(errors.is_empty(), "{errors:#?}");
}

#[test]
fn fixture_validation_rejects_harness_and_substitution_mismatch() {
    let mut example = loopback_example();
    example.harness = BuiltinExampleHarness::Portable;
    example.program = "url = \"http://example.invalid/value\";";
    let mut errors = Vec::new();
    validate("webread", &example, &mut errors);
    assert!(errors.iter().any(|error| error.message.contains("harness")));
    assert!(errors
        .iter()
        .any(|error| error.message.contains("substitution token")));
}

#[test]
fn cli_transcript_requires_one_terminal_action() {
    let example = BuiltinExample {
        id: "prompt",
        title: "Prompt",
        program: "value = input('value: ');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::InteractiveHost,
        fixture: BuiltinExampleFixture::CliInteraction(crate::BuiltinCliInteractionFixture {
            id: crate::BuiltinExampleFixtureId {
                local_name: "prompt",
            },
            transcript: &[BuiltinCliTranscriptStep::SendLine("4")],
        }),
        requirements: BuiltinExampleRequirements {
            host: BuiltinExampleHostRequirement::NativeOnly,
            ..BuiltinExampleRequirements::NONE
        },
        verification: BuiltinExampleVerification::Succeeds,
    };
    let mut errors = Vec::new();
    validate("input", &example, &mut errors);
    assert!(errors
        .iter()
        .any(|error| error.message.contains("terminal action")));
}

#[test]
fn foreign_preparation_is_complete_and_references_declared_files() {
    const FILES: &[BuiltinFilesystemEntry] = &[BuiltinFilesystemEntry::File {
        relative_path: "gateway.c",
        content: BuiltinFixtureContent::Utf8("void mexFunction(void) {}"),
    }];
    const UNITS: &[BuiltinNativeTranslationUnit] = &[BuiltinNativeTranslationUnit {
        relative_path: "gateway.c",
        language: BuiltinNativeSourceLanguage::C,
    }];
    let mut example = BuiltinExample {
        id: "mex",
        title: "Build a MEX module",
        program: "value = fixture_mex();",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeForeignRuntime,
        fixture: BuiltinExampleFixture::ForeignAdapter(BuiltinForeignAdapterFixture {
            id: crate::BuiltinExampleFixtureId {
                local_name: "fixture-mex",
            },
            files: BuiltinFilesystemFixture {
                id: crate::BuiltinExampleFixtureId {
                    local_name: "fixture-files",
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: FILES,
            },
            preparation: BuiltinForeignPreparation::Mex(BuiltinMexPreparation {
                module_name: "fixture_mex",
                api: BuiltinMexApi::R2017b,
                translation_units: UNITS,
                include_directories: &[],
                definitions: &[],
            }),
        }),
        requirements: BuiltinExampleRequirements {
            host: BuiltinExampleHostRequirement::NativeOnly,
            engine: BuiltinExampleEngine::Default,
            compiler: &[BuiltinCompilerCapability::C],
            runtime: &[BuiltinRuntimeCapability::Mex],
            toolchain: &[BuiltinToolchainCapability::CCompiler],
        },
        verification: BuiltinExampleVerification::Succeeds,
    };
    let mut errors = Vec::new();
    validate("fixture_mex", &example, &mut errors);
    assert!(errors.is_empty(), "{errors:#?}");

    const MISSING_UNITS: &[BuiltinNativeTranslationUnit] = &[BuiltinNativeTranslationUnit {
        relative_path: "missing.c",
        language: BuiltinNativeSourceLanguage::C,
    }];
    let BuiltinExampleFixture::ForeignAdapter(mut fixture) = example.fixture else {
        unreachable!();
    };
    fixture.preparation = BuiltinForeignPreparation::Mex(BuiltinMexPreparation {
        module_name: "fixture_mex",
        api: BuiltinMexApi::R2017b,
        translation_units: MISSING_UNITS,
        include_directories: &[],
        definitions: &[],
    });
    example.fixture = BuiltinExampleFixture::ForeignAdapter(fixture);
    errors.clear();
    validate("fixture_mex", &example, &mut errors);
    assert!(errors
        .iter()
        .any(|error| error.message.contains("not a declared fixture file")));
}
