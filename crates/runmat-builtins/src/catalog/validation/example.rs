use crate::{
    BuiltinExample, BuiltinExampleEngine, BuiltinExampleFixture, BuiltinExampleHarness,
    BuiltinExampleHostRequirement, BuiltinForeignPreparation, BuiltinRuntimeCapability,
};

use super::BuiltinCatalogValidationError;

mod foreign;
mod interaction;
mod resource;
#[cfg(test)]
mod tests;

pub(super) fn validate(
    builtin: &'static str,
    example: &BuiltinExample,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    validate_requirements(builtin, example, errors);
    if !matches!(example.fixture, BuiltinExampleFixture::Loopback(_)) {
        for substitution in crate::BuiltinEndpointSubstitution::ALL {
            if example.program.contains(substitution.source_token()) {
                push(
                    errors,
                    builtin,
                    "endpoint substitution token requires a loopback fixture",
                );
            }
        }
    }
    match example.fixture {
        BuiltinExampleFixture::None => {}
        BuiltinExampleFixture::Filesystem(fixture) => {
            require_harness(
                builtin,
                example,
                &[
                    BuiltinExampleHarness::Portable,
                    BuiltinExampleHarness::Browser,
                    BuiltinExampleHarness::NativeFilesystem,
                ],
                errors,
            );
            resource::validate_filesystem(builtin, fixture, errors);
        }
        BuiltinExampleFixture::Loopback(fixture) => {
            require_harness(
                builtin,
                example,
                &[BuiltinExampleHarness::NativeLoopbackNetwork],
                errors,
            );
            require_native_host(builtin, example, errors);
            resource::validate_fixture_id(builtin, fixture.id, errors);
            validate_sorted_unique(
                builtin,
                fixture.endpoint_substitutions,
                "endpoint substitutions",
                errors,
            );
            for substitution in fixture.endpoint_substitutions {
                if !example.program.contains(substitution.source_token()) {
                    push(
                        errors,
                        builtin,
                        "loopback endpoint substitution token is absent from the program",
                    );
                }
            }
            for substitution in crate::BuiltinEndpointSubstitution::ALL {
                if example.program.contains(substitution.source_token())
                    && !fixture.endpoint_substitutions.contains(&substitution)
                {
                    push(
                        errors,
                        builtin,
                        "program uses an undeclared loopback endpoint substitution token",
                    );
                }
            }
            if matches!(fixture.scenario, crate::BuiltinLoopbackScenario::Tcp(_))
                && fixture
                    .endpoint_substitutions
                    .contains(&crate::BuiltinEndpointSubstitution::HttpBaseUrl)
            {
                push(
                    errors,
                    builtin,
                    "TCP fixture cannot declare the HTTP base URL substitution",
                );
            }
            resource::validate_loopback(builtin, fixture.scenario, errors);
        }
        BuiltinExampleFixture::ForeignAdapter(fixture) => {
            require_harness(
                builtin,
                example,
                &[BuiltinExampleHarness::NativeForeignRuntime],
                errors,
            );
            require_native_host(builtin, example, errors);
            resource::validate_fixture_id(builtin, fixture.id, errors);
            resource::validate_filesystem(builtin, fixture.files, errors);
            let required = match fixture.preparation {
                BuiltinForeignPreparation::Mex(_) => BuiltinRuntimeCapability::Mex,
                BuiltinForeignPreparation::NativeFfi(_) => BuiltinRuntimeCapability::NativeFfi,
                BuiltinForeignPreparation::Java(_) => BuiltinRuntimeCapability::JavaVirtualMachine,
                BuiltinForeignPreparation::Python(_) => BuiltinRuntimeCapability::Python,
            };
            if !example.requirements.runtime.contains(&required) {
                push(
                    errors,
                    builtin,
                    "foreign fixture does not declare its adapter runtime capability",
                );
            }
            foreign::validate_preparation(builtin, example, fixture, errors);
        }
        BuiltinExampleFixture::CliInteraction(fixture) => {
            require_harness(
                builtin,
                example,
                &[BuiltinExampleHarness::InteractiveHost],
                errors,
            );
            require_native_host(builtin, example, errors);
            resource::validate_fixture_id(builtin, fixture.id, errors);
            interaction::validate_cli_transcript(builtin, fixture.transcript, errors);
        }
        BuiltinExampleFixture::DesktopHostOnly(fixture) => {
            require_harness(
                builtin,
                example,
                &[BuiltinExampleHarness::InteractiveHost],
                errors,
            );
            resource::validate_fixture_id(builtin, fixture.id, errors);
            if example.requirements.host != BuiltinExampleHostRequirement::DesktopHostOnly {
                push(
                    errors,
                    builtin,
                    "desktop fixture requires the desktop-host-only boundary",
                );
            }
        }
    }
}

fn validate_requirements(
    builtin: &'static str,
    example: &BuiltinExample,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    validate_sorted_unique(
        builtin,
        example.requirements.compiler,
        "compiler capabilities",
        errors,
    );
    validate_sorted_unique(
        builtin,
        example.requirements.runtime,
        "runtime capabilities",
        errors,
    );
    validate_sorted_unique(
        builtin,
        example.requirements.toolchain,
        "toolchain capabilities",
        errors,
    );
    match example.requirements.host {
        BuiltinExampleHostRequirement::Any => {}
        BuiltinExampleHostRequirement::NativeOnly
            if !matches!(
                example.harness,
                BuiltinExampleHarness::Native
                    | BuiltinExampleHarness::NativeFilesystem
                    | BuiltinExampleHarness::NativeLoopbackNetwork
                    | BuiltinExampleHarness::NativeForeignRuntime
                    | BuiltinExampleHarness::InteractiveHost
            ) =>
        {
            push(
                errors,
                builtin,
                "native-only example requirements select a non-native harness",
            );
        }
        BuiltinExampleHostRequirement::DesktopHostOnly
            if example.harness != BuiltinExampleHarness::InteractiveHost =>
        {
            push(
                errors,
                builtin,
                "desktop-host-only requirements select a non-interactive harness",
            );
        }
        _ => {}
    }
    if example.requirements.engine != BuiltinExampleEngine::Default
        && example.requirements.host != BuiltinExampleHostRequirement::NativeOnly
    {
        push(
            errors,
            builtin,
            "an explicit execution engine requires a native-only host boundary",
        );
    }
}

fn require_harness(
    builtin: &'static str,
    example: &BuiltinExample,
    allowed: &[BuiltinExampleHarness],
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    if !allowed.contains(&example.harness) {
        push(
            errors,
            builtin,
            "example fixture is incompatible with its execution harness",
        );
    }
}

fn require_native_host(
    builtin: &'static str,
    example: &BuiltinExample,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    if example.requirements.host != BuiltinExampleHostRequirement::NativeOnly {
        push(
            errors,
            builtin,
            "native fixture requires an explicit native-only host boundary",
        );
    }
}

fn validate_sorted_unique<T: Ord>(
    builtin: &'static str,
    values: &[T],
    label: &str,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    if !values.windows(2).all(|pair| pair[0] < pair[1]) {
        push(
            errors,
            builtin,
            format!("{label} must be sorted and unique"),
        );
    }
}

pub(super) fn push(
    errors: &mut Vec<BuiltinCatalogValidationError>,
    builtin: &'static str,
    message: impl Into<String>,
) {
    errors.push(BuiltinCatalogValidationError {
        identity: Some(builtin),
        message: message.into(),
    });
}
