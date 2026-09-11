use serde::Serialize;

use super::{BuiltinExampleFixtureId, BuiltinFilesystemFixture};

/// Complete, declarative preparation contract for a foreign-runtime example.
/// Runners materialize `files` and execute exactly one typed preparation plan;
/// they never infer build inputs or commands from the example program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinForeignAdapterFixture {
    pub id: BuiltinExampleFixtureId,
    pub files: BuiltinFilesystemFixture,
    pub preparation: BuiltinForeignPreparation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinForeignPreparation {
    Mex(BuiltinMexPreparation),
    NativeFfi(BuiltinNativeFfiPreparation),
    Java(BuiltinJavaPreparation),
    Python(BuiltinPythonPreparation),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinForeignIsolation {
    InProcess,
    OutOfProcess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinMexPreparation {
    pub module_name: &'static str,
    pub api: BuiltinMexApi,
    pub translation_units: &'static [BuiltinNativeTranslationUnit],
    pub include_directories: &'static [&'static str],
    pub definitions: &'static [BuiltinPreprocessorDefinition],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinMexApi {
    R2017b,
    R2018a,
    LargeArrayDims,
    CompatibleArrayDims,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinNativeTranslationUnit {
    pub relative_path: &'static str,
    pub language: BuiltinNativeSourceLanguage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum BuiltinNativeSourceLanguage {
    C,
    Cxx,
    Fortran,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinPreprocessorDefinition {
    pub name: &'static str,
    pub value: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinNativeFfiPreparation {
    pub isolation: BuiltinForeignIsolation,
    pub library_name: &'static str,
    pub translation_units: &'static [BuiltinNativeTranslationUnit],
    pub include_directories: &'static [&'static str],
    pub definitions: &'static [BuiltinPreprocessorDefinition],
    pub interface: BuiltinNativeInterfacePreparation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinNativeInterfacePreparation {
    pub interface_name: &'static str,
    pub primary_header: &'static str,
    pub additional_headers: &'static [&'static str],
    pub include_directories: &'static [&'static str],
    pub definitions: &'static [BuiltinPreprocessorDefinition],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinJavaPreparation {
    pub artifact_name: &'static str,
    pub release: u16,
    pub source_files: &'static [&'static str],
    pub resources: &'static [BuiltinJavaResource],
    pub compile_classpath: &'static [&'static str],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinJavaResource {
    pub source_path: &'static str,
    pub artifact_path: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinPythonPreparation {
    pub isolation: BuiltinForeignIsolation,
    pub environment: BuiltinPythonEnvironment,
    pub artifact: BuiltinPythonArtifact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinPythonEnvironment {
    pub implementation: BuiltinPythonImplementation,
    pub major: u16,
    pub minor: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinPythonImplementation {
    Cpython,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinPythonArtifact {
    SourceTree {
        module_root: &'static str,
        modules: &'static [&'static str],
    },
    Wheel {
        artifact_name: &'static str,
        relative_path: &'static str,
        module: &'static str,
        compatibility: BuiltinPythonWheelCompatibility,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinPythonWheelCompatibility {
    Pure,
    Native {
        abi_tag: &'static str,
        platform_tag: &'static str,
    },
}
