use runmat_geometry_core::GeometryRegionRole;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoredModelError {
    #[error("{0} reference temperature must be finite and nonnegative kelvin")]
    InvalidDomainReferenceTemperature(&'static str),
    #[error("{0} must not be empty")]
    InvalidToken(&'static str),
    #[error("model input `{0}` is duplicated")]
    DuplicateInput(String),
    #[error("model input is invalid: {0}")]
    InvalidInput(String),
    #[error("model cannot include itself")]
    SelfInclude,
    #[error("model include is duplicated")]
    DuplicateInclude,
    #[error("material assignment is duplicated")]
    DuplicateMaterialAssignment,
    #[error("section `{0}` is duplicated")]
    DuplicateSection(String),
    #[error("layup `{0}` is duplicated")]
    DuplicateLayup(String),
    #[error("orientation `{0}` is duplicated")]
    DuplicateOrientation(String),
    #[error("property definition is invalid: {0}")]
    InvalidProperty(String),
    #[error("property assignment is duplicated")]
    DuplicatePropertyAssignment,
    #[error("multiple section assignments control the same target")]
    ConflictingSectionAssignment,
    #[error("multiple layup assignments control the same target")]
    ConflictingLayupAssignment,
    #[error("multiple orientation assignments control the same target")]
    ConflictingOrientationAssignment,
    #[error("a shell target cannot have both homogeneous material and laminate ownership")]
    CompositeAndHomogeneousOwnership,
    #[error("constraint `{0}` is duplicated")]
    DuplicateConstraint(String),
    #[error("constraint is invalid: {0}")]
    InvalidConstraint(String),
    #[error("{kind} `{id}` is duplicated")]
    DuplicatePhysics { kind: &'static str, id: String },
    #[error("{kind} definition is invalid: {detail}")]
    InvalidPhysics { kind: &'static str, detail: String },
    #[error("a region-typed model input is required")]
    RegionInputRequired,
    #[error("model input `{0}` is referenced with a different kind")]
    InputKindMismatch(String),
    #[error("unknown section `{0}`")]
    UnknownSection(String),
    #[error("unknown layup `{0}`")]
    UnknownLayup(String),
    #[error("unknown orientation `{0}`")]
    UnknownOrientation(String),
    #[error("model-local property reference has a different module or model owner")]
    ForeignPropertyReference,
    #[error("section assignments require beam or shell targets")]
    WrongSectionTarget,
    #[error("layup assignments require shell-face targets")]
    WrongLayupTarget,
    #[error("orientation assignments do not support cell targets")]
    WrongOrientationTarget,
    #[error("material source input is not typed as material")]
    WrongMaterialInputKind,
    #[error("material target input is not typed as a region")]
    WrongRegionInputKind,
    #[error("model input `{0}` is not declared as a property input")]
    UnknownInput(String),
    #[error("material target role expected {expected:?} topology, found {actual:?}")]
    WrongPropertyTargetRole {
        expected: GeometryRegionRole,
        actual: GeometryRegionRole,
    },
}
