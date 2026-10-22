pub const EXECUTION_SCHEMA_V1: u16 = 1;
/// Task V2 distinguishes compiled artifacts from admitted owned builtin work.
pub const EXECUTION_SCHEMA_V2: u16 = 2;
pub const EXECUTABLE_UNIT_SCHEMA_V2: u16 = 2;
pub const EXECUTABLE_UNIT_SCHEMA_V4: u16 = 4;
pub const PROGRAM_REVISION_SCHEMA_V1: u16 = 1;
pub const VALUE_PAYLOAD_SCHEMA_V1: u16 = 1;
pub const VALUE_PAYLOAD_SCHEMA_V2: u16 = 2;
pub const VALUE_PAYLOAD_SCHEMA_V3: u16 = 3;
/// Value payload V4 stores character arrays as exact UTF-16 code units.
pub const VALUE_PAYLOAD_SCHEMA_V4: u16 = 4;
pub const VALUE_PAYLOAD_SCHEMA_VERSION: u16 = VALUE_PAYLOAD_SCHEMA_V4;
pub const PROTOCOL_MAJOR_V1: u16 = 1;
pub const PROTOCOL_MINOR_V1: u16 = 0;

/// Current source-language semantic contract understood by RunMat compilers
/// and execution hosts.
pub const PROGRAM_SEMANTIC_SCHEMA_V2: u32 = 2;
/// Current compiler contract for portable executable products.
pub const PROGRAM_COMPILER_SCHEMA_V2: u32 = 2;
