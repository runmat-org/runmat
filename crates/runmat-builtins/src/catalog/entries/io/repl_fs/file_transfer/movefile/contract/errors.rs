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
    MOVEFILE_ERROR_NOT_ENOUGH_INPUTS,
    "RM.MOVEFILE.NOT_ENOUGH_INPUTS",
    "RunMat:movefile:NotEnoughInputs",
    "Fewer than two inputs are provided.",
    "movefile: not enough input arguments"
);
error!(
    MOVEFILE_ERROR_TOO_MANY_INPUTS,
    "RM.MOVEFILE.TOO_MANY_INPUTS",
    "RunMat:movefile:TooManyInputs",
    "More than three inputs are provided.",
    "movefile: too many input arguments"
);
error!(
    MOVEFILE_ERROR_SOURCE_ARG,
    "RM.MOVEFILE.SOURCE_ARG",
    "RunMat:movefile:SourceArgType",
    "Source is not a character row or string scalar.",
    "movefile: source must be a character vector or string scalar"
);
error!(
    MOVEFILE_ERROR_DEST_ARG,
    "RM.MOVEFILE.DEST_ARG",
    "RunMat:movefile:DestinationArgType",
    "Destination is not a character row or string scalar.",
    "movefile: destination must be a character vector or string scalar"
);
error!(
    MOVEFILE_ERROR_FLAG_ARG,
    "RM.MOVEFILE.FLAG_ARG",
    "RunMat:movefile:FlagArgType",
    "The optional flag is not 'f'.",
    "movefile: flag must be the character 'f' supplied as a char vector or string scalar"
);
error!(
    MOVEFILE_RESULT_OS_ERROR,
    "RM.MOVEFILE.OS_ERROR",
    "RunMat:movefile:OSError",
    "The filesystem move fails.",
    "movefile: unable to move"
);
error!(
    MOVEFILE_RESULT_SOURCE_NOT_FOUND,
    "RM.MOVEFILE.SOURCE_NOT_FOUND",
    "RunMat:movefile:FileDoesNotExist",
    "The source or wildcard match does not exist.",
    "movefile: source not found"
);
error!(
    MOVEFILE_RESULT_DEST_EXISTS,
    "RM.MOVEFILE.DEST_EXISTS",
    "RunMat:movefile:DestinationExists",
    "The target exists without the force flag.",
    "movefile: destination already exists"
);
error!(
    MOVEFILE_RESULT_DEST_MISSING,
    "RM.MOVEFILE.DEST_MISSING",
    "RunMat:movefile:DestinationNotFound",
    "A wildcard transfer targets a missing directory.",
    "movefile: destination folder not found"
);
error!(
    MOVEFILE_RESULT_DEST_NOT_DIR,
    "RM.MOVEFILE.DEST_NOT_DIR",
    "RunMat:movefile:DestinationNotDirectory",
    "A wildcard transfer targets a non-directory.",
    "movefile: destination is not a directory"
);
error!(
    MOVEFILE_RESULT_EMPTY_SOURCE,
    "RM.MOVEFILE.EMPTY_SOURCE",
    "RunMat:movefile:EmptySource",
    "The source path is empty.",
    "Source file or folder name must not be empty."
);
error!(
    MOVEFILE_RESULT_EMPTY_DEST,
    "RM.MOVEFILE.EMPTY_DEST",
    "RunMat:movefile:EmptyDestination",
    "The destination path is empty.",
    "Destination file or folder name must not be empty."
);
error!(
    MOVEFILE_RESULT_PATTERN_ERROR,
    "RM.MOVEFILE.PATTERN_ERROR",
    "RunMat:movefile:InvalidPattern",
    "The source wildcard pattern is invalid.",
    "movefile: invalid source pattern"
);

pub(super) const ERRORS: &[BuiltinErrorDescriptor] = &[
    MOVEFILE_ERROR_NOT_ENOUGH_INPUTS,
    MOVEFILE_ERROR_TOO_MANY_INPUTS,
    MOVEFILE_ERROR_SOURCE_ARG,
    MOVEFILE_ERROR_DEST_ARG,
    MOVEFILE_ERROR_FLAG_ARG,
    MOVEFILE_RESULT_OS_ERROR,
    MOVEFILE_RESULT_SOURCE_NOT_FOUND,
    MOVEFILE_RESULT_DEST_EXISTS,
    MOVEFILE_RESULT_DEST_MISSING,
    MOVEFILE_RESULT_DEST_NOT_DIR,
    MOVEFILE_RESULT_EMPTY_SOURCE,
    MOVEFILE_RESULT_EMPTY_DEST,
    MOVEFILE_RESULT_PATTERN_ERROR,
];
