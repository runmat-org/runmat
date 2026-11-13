use crate::BuiltinErrorDescriptor;

pub const TABLE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TABLE.INVALID_ARGUMENT",
    identifier: Some("RunMat:table:InvalidArgument"),
    when: "Arguments or table metadata are invalid.",
    message: "table: invalid argument",
};
pub const TABLE_ERROR_INVALID_INDEX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TABLE.INVALID_INDEX",
    identifier: Some("RunMat:table:InvalidIndex"),
    when: "Table indexing is invalid.",
    message: "table: invalid index",
};
pub const TABLE_ERROR_INVALID_VARIABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TABLE.INVALID_VARIABLE",
    identifier: Some("RunMat:table:InvalidVariable"),
    when: "A table variable name or value is invalid.",
    message: "table: invalid variable",
};
pub const TABLE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READTABLE.IO",
    identifier: Some("RunMat:readtable:IOError"),
    when: "readtable cannot open or read the requested file.",
    message: "readtable: file read failed",
};
pub const TABLE_ERROR_UNSUPPORTED_FILE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READTABLE.UNSUPPORTED_FILE",
    identifier: Some("RunMat:readtable:UnsupportedFileType"),
    when: "readtable receives a file type outside the text or spreadsheet import backends.",
    message: "readtable: unsupported file type",
};
pub const TABLE_ERRORS: [BuiltinErrorDescriptor; 5] = [
    TABLE_ERROR_INVALID_ARGUMENT,
    TABLE_ERROR_INVALID_INDEX,
    TABLE_ERROR_INVALID_VARIABLE,
    TABLE_ERROR_IO,
    TABLE_ERROR_UNSUPPORTED_FILE,
];
