use crate::{
    ClassId, ClassIdentity, FunctionId, MemberName, MethodName, ProgramSourceId, QualifiedName,
    Span,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassDeclaration {
    pub id: ClassId,
    /// Original program-source coordinate of this declaration's HIR module.
    pub source: ProgramSourceId,
    pub name: QualifiedName,
    pub inheritance: InheritanceDeclaration,
    pub kind: ClassKind,
    pub is_sealed: bool,
    pub is_abstract: bool,
    pub declared_attributes: Vec<SemanticAttribute>,
    pub properties: Vec<PropertyDeclaration>,
    pub methods: Vec<MethodDeclaration>,
    pub events: Vec<EventDeclaration>,
    pub enumerations: Vec<EnumerationDeclaration>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct InheritanceDeclaration {
    pub declared_super_class: Option<String>,
    pub resolved_super_class: Option<ClassId>,
    pub builtin_super_class: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassKind {
    Value,
    Handle,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropertyDeclaration {
    pub name: MemberName,
    pub attributes: PropertyAttributes,
    pub declared_attributes: Vec<SemanticAttribute>,
    pub has_default: bool,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MethodDeclaration {
    pub function: FunctionId,
    pub name: MethodName,
    pub is_static: bool,
    pub role: crate::ClassMethodRole,
    pub attributes: MethodAttributes,
    pub declared_attributes: Vec<SemanticAttribute>,
    pub span: Span,
}

/// Immutable declaration for a class whose identity is not local to one HIR
/// assembly (for example a standard-library or package-provided class).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalClassDeclaration {
    pub name: QualifiedName,
    pub parent: Option<QualifiedName>,
    pub kind: ClassKind,
    pub is_sealed: bool,
    pub is_abstract: bool,
    pub properties: Vec<ExternalPropertyDeclaration>,
    pub methods: Vec<ExternalMethodDeclaration>,
    /// Structural metadata for a value-backed array wrapper. This does not
    /// admit numerical operations or invoke user getters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub array_shape: Option<ClassArrayShape>,
}

/// Array geometry supplied by stored properties, independently of the scalar
/// object container. Parallel payloads must describe the same array.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassArrayShape {
    Property {
        property: MemberName,
        payload: ClassArrayPayload,
    },
    ConsistentProperties {
        primary: MemberName,
        others: Vec<MemberName>,
        payload: ClassArrayPayload,
    },
}

/// Structural storage admission, not an operation's dtype/value policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassArrayPayload {
    RealNumeric,
    NumericOrLogical,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalPropertyDeclaration {
    pub name: MemberName,
    pub attributes: PropertyAttributes,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalMethodDeclaration {
    pub name: MethodName,
    pub attributes: MethodAttributes,
    pub is_static: bool,
    pub callable: crate::CallableIdentity,
    pub implicit_class_argument: Option<ClassIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventDeclaration {
    pub name: MemberName,
    #[serde(default)]
    pub attributes: EventAttributes,
    pub declared_attributes: Vec<SemanticAttribute>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct EventAttributes {
    pub is_hidden: bool,
    pub listen_access: EventAccess,
    pub notify_access: EventAccess,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum EventAccess {
    #[default]
    Public,
    Private,
    Protected,
    Classes(Vec<ClassIdentity>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnumerationDeclaration {
    pub name: MemberName,
    pub declared_attributes: Vec<SemanticAttribute>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticAttribute {
    pub name: String,
    pub value: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PropertyAttributes {
    pub is_static: bool,
    pub is_constant: bool,
    pub is_dependent: bool,
    pub is_transient: bool,
    pub is_hidden: bool,
    pub access: MemberAccess,
    pub get_access: MemberAccess,
    pub set_access: MemberAccess,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct MethodAttributes {
    pub access: MemberAccess,
    pub is_hidden: bool,
    pub is_abstract: bool,
    pub is_sealed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MemberAccess {
    #[default]
    Public,
    Private,
    Protected,
}
