use crate::*;
pub const TABLE2STRUCT_INTEGER_TO_SCALAR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "table2struct-integer-toscalar",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "table2struct with a typed-integer ToScalar value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Table2StructIntegerToScalarExtension"),
    };
pub const TABLE2STRUCT_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [TABLE2STRUCT_INTEGER_TO_SCALAR_EXTENSION];
pub const ARRAY2TABLE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "array2table-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "array2table with an interactive resident GPU input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Array2TableGpuInputExtension"),
    };

pub const ARRAY2TABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [ARRAY2TABLE_GPU_INPUT_EXTENSION];
