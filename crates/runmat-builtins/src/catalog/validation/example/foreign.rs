use std::collections::BTreeSet;

use crate::{
    BuiltinCompilerCapability, BuiltinExample, BuiltinFilesystemEntry,
    BuiltinForeignAdapterFixture, BuiltinForeignPreparation, BuiltinJavaPreparation,
    BuiltinMexPreparation, BuiltinNativeFfiPreparation, BuiltinNativeSourceLanguage,
    BuiltinPreprocessorDefinition, BuiltinPythonArtifact, BuiltinPythonImplementation,
    BuiltinPythonPreparation, BuiltinPythonWheelCompatibility, BuiltinToolchainCapability,
};

use super::{push, resource::valid_relative_path, BuiltinCatalogValidationError};

pub(super) fn validate_preparation(
    builtin: &'static str,
    example: &BuiltinExample,
    fixture: BuiltinForeignAdapterFixture,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    let files = fixture
        .files
        .entries
        .iter()
        .filter_map(|entry| match entry {
            BuiltinFilesystemEntry::File { relative_path, .. } => Some(*relative_path),
            BuiltinFilesystemEntry::Directory { .. } => None,
        })
        .collect::<BTreeSet<_>>();
    match fixture.preparation {
        BuiltinForeignPreparation::Mex(preparation) => {
            require_capabilities(
                builtin,
                example,
                &languages(preparation.translation_units),
                errors,
            );
            validate_mex(builtin, preparation, &files, errors);
        }
        BuiltinForeignPreparation::NativeFfi(preparation) => {
            require_capabilities(
                builtin,
                example,
                &languages(preparation.translation_units),
                errors,
            );
            validate_native_ffi(builtin, preparation, &files, errors);
        }
        BuiltinForeignPreparation::Java(preparation) => {
            require(
                builtin,
                example
                    .requirements
                    .compiler
                    .contains(&BuiltinCompilerCapability::JavaBytecode)
                    && example
                        .requirements
                        .toolchain
                        .contains(&BuiltinToolchainCapability::JavaDevelopmentKit),
                "Java fixture requires Java bytecode and JDK capabilities",
                errors,
            );
            validate_java(builtin, preparation, &files, errors);
        }
        BuiltinForeignPreparation::Python(preparation) => {
            require(
                builtin,
                example
                    .requirements
                    .toolchain
                    .contains(&BuiltinToolchainCapability::PythonInterpreter),
                "Python fixture requires the Python interpreter capability",
                errors,
            );
            validate_python(builtin, preparation, &files, errors);
        }
    }
}

fn validate_mex(
    builtin: &'static str,
    preparation: BuiltinMexPreparation,
    files: &BTreeSet<&str>,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    validate_identifier(builtin, preparation.module_name, "MEX module name", errors);
    validate_translation_units(builtin, preparation.translation_units, files, errors);
    validate_paths(
        builtin,
        preparation.include_directories,
        "MEX include directories",
        errors,
    );
    validate_definitions(builtin, preparation.definitions, errors);
}

fn validate_native_ffi(
    builtin: &'static str,
    preparation: BuiltinNativeFfiPreparation,
    files: &BTreeSet<&str>,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    validate_identifier(
        builtin,
        preparation.library_name,
        "native library name",
        errors,
    );
    validate_translation_units(builtin, preparation.translation_units, files, errors);
    validate_paths(
        builtin,
        preparation.include_directories,
        "native build include directories",
        errors,
    );
    validate_definitions(builtin, preparation.definitions, errors);
    validate_identifier(
        builtin,
        preparation.interface.interface_name,
        "native interface name",
        errors,
    );
    require_file(
        builtin,
        preparation.interface.primary_header,
        "native interface primary header",
        files,
        errors,
    );
    validate_file_paths(
        builtin,
        preparation.interface.additional_headers,
        "native interface additional headers",
        files,
        errors,
    );
    validate_paths(
        builtin,
        preparation.interface.include_directories,
        "native interface include directories",
        errors,
    );
    validate_definitions(builtin, preparation.interface.definitions, errors);
}

fn validate_java(
    builtin: &'static str,
    preparation: BuiltinJavaPreparation,
    files: &BTreeSet<&str>,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    validate_artifact_name(
        builtin,
        preparation.artifact_name,
        "Java artifact name",
        errors,
    );
    require(
        builtin,
        (8..=99).contains(&preparation.release),
        "Java release must be between 8 and 99",
        errors,
    );
    require(
        builtin,
        !preparation.source_files.is_empty(),
        "Java preparation requires at least one source file",
        errors,
    );
    validate_file_paths(
        builtin,
        preparation.source_files,
        "Java source files",
        files,
        errors,
    );
    if !preparation
        .resources
        .windows(2)
        .all(|pair| pair[0].artifact_path < pair[1].artifact_path)
    {
        push(
            errors,
            builtin,
            "Java resources must use unique canonical artifact-path order",
        );
    }
    for resource in preparation.resources {
        require_file(
            builtin,
            resource.source_path,
            "Java resource source",
            files,
            errors,
        );
        validate_relative_path(
            builtin,
            resource.artifact_path,
            "Java resource artifact path",
            errors,
        );
    }
    validate_file_paths(
        builtin,
        preparation.compile_classpath,
        "Java compile classpath",
        files,
        errors,
    );
    if preparation.source_files.iter().any(|path| {
        preparation
            .resources
            .iter()
            .any(|resource| resource.source_path == *path)
    }) {
        push(
            errors,
            builtin,
            "Java source and resource files must be disjoint",
        );
    }
}

fn validate_python(
    builtin: &'static str,
    preparation: BuiltinPythonPreparation,
    files: &BTreeSet<&str>,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    require(
        builtin,
        preparation.environment.implementation == BuiltinPythonImplementation::Cpython,
        "Python fixture requires a supported interpreter implementation",
        errors,
    );
    require(
        builtin,
        preparation.environment.major == 3 && preparation.environment.minor <= 99,
        "Python fixture requires an explicit CPython 3 minor version",
        errors,
    );
    match preparation.artifact {
        BuiltinPythonArtifact::SourceTree {
            module_root,
            modules,
        } => {
            validate_relative_path(builtin, module_root, "Python module root", errors);
            require(
                builtin,
                !modules.is_empty(),
                "Python source preparation requires at least one module",
                errors,
            );
            validate_ordered_unique(builtin, modules, "Python modules", errors);
            for module in modules {
                if !valid_qualified_identifier(module) {
                    push(errors, builtin, "Python module name is invalid");
                }
            }
            if !files.iter().any(|path| {
                path.starts_with(module_root)
                    && path.as_bytes().get(module_root.len()) == Some(&b'/')
            }) {
                push(
                    errors,
                    builtin,
                    "Python module root contains no declared fixture files",
                );
            }
        }
        BuiltinPythonArtifact::Wheel {
            artifact_name,
            relative_path,
            module,
            compatibility,
        } => {
            validate_artifact_name(builtin, artifact_name, "Python artifact name", errors);
            require_file(builtin, relative_path, "Python wheel", files, errors);
            if !relative_path.ends_with(".whl") {
                push(errors, builtin, "Python wheel path must end in .whl");
            }
            if !valid_qualified_identifier(module) {
                push(errors, builtin, "Python wheel module name is invalid");
            }
            if let BuiltinPythonWheelCompatibility::Native {
                abi_tag,
                platform_tag,
            } = compatibility
            {
                if !valid_tag(abi_tag) || !valid_tag(platform_tag) {
                    push(
                        errors,
                        builtin,
                        "native Python wheel ABI and platform tags are invalid",
                    );
                }
            }
        }
    }
}

fn languages(
    units: &[crate::BuiltinNativeTranslationUnit],
) -> BTreeSet<BuiltinNativeSourceLanguage> {
    units.iter().map(|unit| unit.language).collect()
}

fn require_capabilities(
    builtin: &'static str,
    example: &BuiltinExample,
    languages: &BTreeSet<BuiltinNativeSourceLanguage>,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    for language in languages {
        let (compiler, toolchain) = match language {
            BuiltinNativeSourceLanguage::C => (
                BuiltinCompilerCapability::C,
                BuiltinToolchainCapability::CCompiler,
            ),
            BuiltinNativeSourceLanguage::Cxx => (
                BuiltinCompilerCapability::Cxx,
                BuiltinToolchainCapability::CxxCompiler,
            ),
            BuiltinNativeSourceLanguage::Fortran => (
                BuiltinCompilerCapability::Fortran,
                BuiltinToolchainCapability::FortranCompiler,
            ),
            BuiltinNativeSourceLanguage::Cuda => (
                BuiltinCompilerCapability::Cuda,
                BuiltinToolchainCapability::CudaToolkit,
            ),
        };
        require(
            builtin,
            example.requirements.compiler.contains(&compiler)
                && example.requirements.toolchain.contains(&toolchain),
            "foreign preparation is missing a translation-unit compiler or toolchain capability",
            errors,
        );
    }
}

fn validate_translation_units(
    builtin: &'static str,
    units: &[crate::BuiltinNativeTranslationUnit],
    files: &BTreeSet<&str>,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    require(
        builtin,
        !units.is_empty(),
        "native preparation requires at least one translation unit",
        errors,
    );
    let paths = units
        .iter()
        .map(|unit| unit.relative_path)
        .collect::<Vec<_>>();
    validate_ordered_unique(builtin, &paths, "native translation units", errors);
    for path in paths {
        require_file(builtin, path, "native translation unit", files, errors);
    }
}

fn validate_definitions(
    builtin: &'static str,
    definitions: &[BuiltinPreprocessorDefinition],
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    if !definitions
        .windows(2)
        .all(|pair| pair[0].name < pair[1].name)
    {
        push(
            errors,
            builtin,
            "preprocessor definitions must be sorted and unique",
        );
    }
    for definition in definitions {
        if !valid_identifier(definition.name)
            || definition
                .value
                .is_some_and(|value| value.contains('\0') || value.contains(['\n', '\r']))
        {
            push(errors, builtin, "preprocessor definition is invalid");
        }
    }
}

fn validate_file_paths(
    builtin: &'static str,
    paths: &[&str],
    label: &str,
    files: &BTreeSet<&str>,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    validate_ordered_unique(builtin, paths, label, errors);
    for path in paths {
        require_file(builtin, path, label, files, errors);
    }
}

fn validate_paths(
    builtin: &'static str,
    paths: &[&str],
    label: &str,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    validate_ordered_unique(builtin, paths, label, errors);
    for path in paths {
        validate_relative_path(builtin, path, label, errors);
    }
}

fn require_file(
    builtin: &'static str,
    path: &str,
    label: &str,
    files: &BTreeSet<&str>,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    validate_relative_path(builtin, path, label, errors);
    if !files.contains(path) {
        push(
            errors,
            builtin,
            format!("{label} is not a declared fixture file"),
        );
    }
}

fn validate_relative_path(
    builtin: &'static str,
    path: &str,
    label: &str,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    if !valid_relative_path(path) {
        push(
            errors,
            builtin,
            format!("{label} must be a portable relative path"),
        );
    }
}

fn validate_identifier(
    builtin: &'static str,
    value: &str,
    label: &str,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    if !valid_identifier(value) {
        push(errors, builtin, format!("{label} is invalid"));
    }
}

fn validate_artifact_name(
    builtin: &'static str,
    value: &str,
    label: &str,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    if value.is_empty()
        || value.len() > 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        push(errors, builtin, format!("{label} is invalid"));
    }
}

fn validate_ordered_unique<T: Ord>(
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

fn valid_identifier(value: &str) -> bool {
    let mut bytes = value.bytes();
    bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        && value.len() <= 128
}

fn valid_qualified_identifier(value: &str) -> bool {
    !value.is_empty() && value.split('.').all(valid_identifier)
}

fn valid_tag(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

fn require(
    builtin: &'static str,
    condition: bool,
    message: &str,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    if !condition {
        push(errors, builtin, message);
    }
}
