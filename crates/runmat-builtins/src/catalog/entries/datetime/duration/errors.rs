use crate::BuiltinErrorDescriptor;

/// Stable diagnostic categories for the host duration implementation.
/// Framework arity failures and delegated indexing/provider identifiers are
/// intentionally outside this set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurationErrorKind {
    InvalidArgument,
    InvalidFormat,
    ShapeMismatch,
    InvalidIndex,
    InvalidObject,
    OutOfRange,
    Internal,
}

impl DurationErrorKind {
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::InvalidArgument => "RunMat:duration:InvalidArgument",
            Self::InvalidFormat => "RunMat:duration:InvalidFormat",
            Self::ShapeMismatch => "RunMat:duration:ShapeMismatch",
            Self::InvalidIndex => "RunMat:duration:InvalidIndex",
            Self::InvalidObject => "RunMat:duration:InvalidObject",
            Self::OutOfRange => "RunMat:duration:OutOfRange",
            Self::Internal => "RunMat:duration:Internal",
        }
    }
}

const fn descriptor(kind: DurationErrorKind) -> BuiltinErrorDescriptor {
    match kind {
        DurationErrorKind::InvalidArgument => BuiltinErrorDescriptor { code: "RM.DURATION.INVALID_ARGUMENT", identifier: Some(kind.identifier()), when: "An input is not a supported real numeric, logical, or duration value, or a constructor form is invalid.", message: "duration: invalid argument" },
        DurationErrorKind::InvalidFormat => BuiltinErrorDescriptor { code: "RM.DURATION.INVALID_FORMAT", identifier: Some(kind.identifier()), when: "A textual Format option or property names an unsupported display format.", message: "duration: invalid Format" },
        DurationErrorKind::ShapeMismatch => BuiltinErrorDescriptor { code: "RM.DURATION.SHAPE_MISMATCH", identifier: Some(kind.identifier()), when: "Component or operand shapes cannot be combined.", message: "duration: incompatible sizes" },
        DurationErrorKind::InvalidIndex => BuiltinErrorDescriptor { code: "RM.DURATION.INVALID_INDEX", identifier: Some(kind.identifier()), when: "A duration index path or member selector is unsupported.", message: "duration: invalid index" },
        DurationErrorKind::InvalidObject => BuiltinErrorDescriptor { code: "RM.DURATION.INVALID_OBJECT", identifier: Some(kind.identifier()), when: "A duration receiver has invalid or missing internal day storage.", message: "duration: invalid object" },
        DurationErrorKind::OutOfRange => BuiltinErrorDescriptor { code: "RM.DURATION.OUT_OF_RANGE", identifier: Some(kind.identifier()), when: "A finite unit conversion exceeds the supported day-count range.", message: "duration: value out of range" },
        DurationErrorKind::Internal => BuiltinErrorDescriptor { code: "RM.DURATION.INTERNAL", identifier: Some(kind.identifier()), when: "A validated duration result cannot be assembled.", message: "duration: internal operation failed" },
    }
}

pub const CONSTRUCTOR_ERRORS: &[BuiltinErrorDescriptor] = &[
    descriptor(DurationErrorKind::InvalidArgument),
    descriptor(DurationErrorKind::InvalidFormat),
    descriptor(DurationErrorKind::ShapeMismatch),
    descriptor(DurationErrorKind::Internal),
];
pub const UNIT_ERRORS: &[BuiltinErrorDescriptor] = &[
    descriptor(DurationErrorKind::InvalidArgument),
    descriptor(DurationErrorKind::InvalidObject),
    descriptor(DurationErrorKind::OutOfRange),
    descriptor(DurationErrorKind::Internal),
];
pub const SUBREF_ERRORS: &[BuiltinErrorDescriptor] = &[
    descriptor(DurationErrorKind::InvalidArgument),
    descriptor(DurationErrorKind::InvalidIndex),
    descriptor(DurationErrorKind::InvalidObject),
    descriptor(DurationErrorKind::Internal),
];
pub const SUBSASGN_ERRORS: &[BuiltinErrorDescriptor] = &[
    descriptor(DurationErrorKind::InvalidArgument),
    descriptor(DurationErrorKind::InvalidFormat),
    descriptor(DurationErrorKind::InvalidIndex),
    descriptor(DurationErrorKind::InvalidObject),
    descriptor(DurationErrorKind::Internal),
];
pub const OPERATOR_ERRORS: &[BuiltinErrorDescriptor] = &[
    descriptor(DurationErrorKind::InvalidArgument),
    descriptor(DurationErrorKind::InvalidObject),
    descriptor(DurationErrorKind::ShapeMismatch),
    descriptor(DurationErrorKind::Internal),
];
