//! Stable XML diagnostic identities shared by the three public forms.

use crate::BuiltinErrorDescriptor;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XmlOperation {
    ReadStruct,
    XmlRead,
    XmlWrite,
}

impl XmlOperation {
    pub const fn name(self) -> &'static str {
        match self {
            Self::ReadStruct => "readstruct",
            Self::XmlRead => "xmlread",
            Self::XmlWrite => "xmlwrite",
        }
    }

    pub const fn identifier(self, kind: XmlDiagnosticKind) -> &'static str {
        match (self, kind) {
            (Self::ReadStruct, XmlDiagnosticKind::InvalidArgument) => {
                "RunMat:readstruct:InvalidArgument"
            }
            (Self::ReadStruct, XmlDiagnosticKind::InvalidOption) => {
                "RunMat:readstruct:InvalidOption"
            }
            (Self::ReadStruct, XmlDiagnosticKind::MalformedXml) => "RunMat:readstruct:MalformedXml",
            (Self::ReadStruct, XmlDiagnosticKind::Projection) => {
                "RunMat:readstruct:ProjectionFailed"
            }
            (Self::ReadStruct, XmlDiagnosticKind::Io) => "RunMat:readstruct:IoFailed",
            (Self::ReadStruct, XmlDiagnosticKind::Cancelled) => "RunMat:readstruct:Cancelled",
            (Self::ReadStruct, XmlDiagnosticKind::Execution) => "RunMat:readstruct:ExecutionFailed",
            (Self::XmlRead, XmlDiagnosticKind::InvalidArgument) => "RunMat:xmlread:InvalidArgument",
            (Self::XmlRead, XmlDiagnosticKind::InvalidOption) => "RunMat:xmlread:InvalidArgument",
            (Self::XmlRead, XmlDiagnosticKind::MalformedXml) => "RunMat:xmlread:MalformedXml",
            (Self::XmlRead, XmlDiagnosticKind::Projection) => "RunMat:xmlread:ProjectionFailed",
            (Self::XmlRead, XmlDiagnosticKind::Io) => "RunMat:xmlread:IoFailed",
            (Self::XmlRead, XmlDiagnosticKind::Cancelled) => "RunMat:xmlread:Cancelled",
            (Self::XmlRead, XmlDiagnosticKind::Execution) => "RunMat:xmlread:ExecutionFailed",
            (Self::XmlWrite, XmlDiagnosticKind::InvalidArgument) => {
                "RunMat:xmlwrite:InvalidArgument"
            }
            (Self::XmlWrite, XmlDiagnosticKind::InvalidOption) => "RunMat:xmlwrite:InvalidArgument",
            (Self::XmlWrite, XmlDiagnosticKind::MalformedXml) => "RunMat:xmlwrite:InvalidDocument",
            (Self::XmlWrite, XmlDiagnosticKind::Projection) => "RunMat:xmlwrite:InvalidDocument",
            (Self::XmlWrite, XmlDiagnosticKind::Io) => "RunMat:xmlwrite:IoFailed",
            (Self::XmlWrite, XmlDiagnosticKind::Cancelled) => "RunMat:xmlwrite:Cancelled",
            (Self::XmlWrite, XmlDiagnosticKind::Execution) => "RunMat:xmlwrite:ExecutionFailed",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XmlDiagnosticKind {
    InvalidArgument,
    InvalidOption,
    MalformedXml,
    Projection,
    Io,
    Cancelled,
    Execution,
}

pub const fn error(
    code: &'static str,
    identifier: &'static str,
    when: &'static str,
    message: &'static str,
) -> BuiltinErrorDescriptor {
    BuiltinErrorDescriptor {
        code,
        identifier: Some(identifier),
        when,
        message,
    }
}
