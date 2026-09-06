use crate::*;

macro_rules! error {
    ($name:ident, $code:literal, $id:literal, $when:literal, $message:literal) => {
        pub const $name: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: $code,
            identifier: Some($id),
            when: $when,
            message: $message,
        };
    };
}

error!(
    COPYFILE_ERROR_NOT_ENOUGH_INPUTS,
    "RM.COPYFILE.NOT_ENOUGH_INPUTS",
    "RunMat:copyfile:NotEnoughInputs",
    "Fewer than two inputs are provided.",
    "copyfile: not enough input arguments"
);
error!(
    COPYFILE_ERROR_TOO_MANY_INPUTS,
    "RM.COPYFILE.TOO_MANY_INPUTS",
    "RunMat:copyfile:TooManyInputs",
    "More than three inputs are provided.",
    "copyfile: too many input arguments"
);
error!(
    COPYFILE_ERROR_SOURCE_ARG,
    "RM.COPYFILE.SOURCE_ARG",
    "RunMat:copyfile:SourceArgType",
    "Source is not a character row or string scalar.",
    "copyfile: source must be a character vector or string scalar"
);
error!(
    COPYFILE_ERROR_DEST_ARG,
    "RM.COPYFILE.DEST_ARG",
    "RunMat:copyfile:DestinationArgType",
    "Destination is not a character row or string scalar.",
    "copyfile: destination must be a character vector or string scalar"
);
error!(
    COPYFILE_ERROR_FLAG_ARG,
    "RM.COPYFILE.FLAG_ARG",
    "RunMat:copyfile:FlagArgType",
    "The optional flag is not 'f'.",
    "copyfile: flag must be the character 'f' supplied as a char vector or string scalar"
);
error!(
    COPYFILE_RESULT_OS_ERROR,
    "RM.COPYFILE.OS_ERROR",
    "RunMat:copyfile:OSError",
    "The filesystem copy fails.",
    "copyfile: unable to copy"
);
error!(
    COPYFILE_RESULT_SOURCE_NOT_FOUND,
    "RM.COPYFILE.SOURCE_NOT_FOUND",
    "RunMat:copyfile:FileDoesNotExist",
    "The source or wildcard match does not exist.",
    "copyfile: source not found"
);
error!(
    COPYFILE_RESULT_DEST_EXISTS,
    "RM.COPYFILE.DEST_EXISTS",
    "RunMat:copyfile:DestinationExists",
    "The target exists without the force flag.",
    "copyfile: destination already exists"
);
error!(
    COPYFILE_RESULT_DEST_MISSING,
    "RM.COPYFILE.DEST_MISSING",
    "RunMat:copyfile:DestinationNotFound",
    "A wildcard transfer targets a missing directory.",
    "copyfile: destination folder not found"
);
error!(
    COPYFILE_RESULT_DEST_NOT_DIR,
    "RM.COPYFILE.DEST_NOT_DIR",
    "RunMat:copyfile:DestinationNotDirectory",
    "A wildcard or directory transfer targets a non-directory.",
    "copyfile: destination is not a directory"
);
error!(
    COPYFILE_RESULT_EMPTY_SOURCE,
    "RM.COPYFILE.EMPTY_SOURCE",
    "RunMat:copyfile:EmptySource",
    "The source path is empty.",
    "Source file or folder name must not be empty."
);
error!(
    COPYFILE_RESULT_EMPTY_DEST,
    "RM.COPYFILE.EMPTY_DEST",
    "RunMat:copyfile:EmptyDestination",
    "The destination path is empty.",
    "Destination file or folder name must not be empty."
);
error!(
    COPYFILE_RESULT_PATTERN_ERROR,
    "RM.COPYFILE.PATTERN_ERROR",
    "RunMat:copyfile:InvalidPattern",
    "The source wildcard pattern is invalid.",
    "copyfile: invalid source pattern"
);
error!(
    COPYFILE_RESULT_SAME_PATH,
    "RM.COPYFILE.SAME_PATH",
    "RunMat:copyfile:SourceEqualsDestination",
    "Source and destination resolve to the same path.",
    "copyfile: source and destination are the same"
);

pub(super) const ERRORS: &[BuiltinErrorDescriptor] = &[
    COPYFILE_ERROR_NOT_ENOUGH_INPUTS,
    COPYFILE_ERROR_TOO_MANY_INPUTS,
    COPYFILE_ERROR_SOURCE_ARG,
    COPYFILE_ERROR_DEST_ARG,
    COPYFILE_ERROR_FLAG_ARG,
    COPYFILE_RESULT_OS_ERROR,
    COPYFILE_RESULT_SOURCE_NOT_FOUND,
    COPYFILE_RESULT_DEST_EXISTS,
    COPYFILE_RESULT_DEST_MISSING,
    COPYFILE_RESULT_DEST_NOT_DIR,
    COPYFILE_RESULT_EMPTY_SOURCE,
    COPYFILE_RESULT_EMPTY_DEST,
    COPYFILE_RESULT_PATTERN_ERROR,
    COPYFILE_RESULT_SAME_PATH,
];
