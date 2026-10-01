use crate::catalog::inference::argument_error;
use runmat_types::InferenceDiagnostic;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::catalog) enum NetworkInferenceCaller {
    Forward,
    Predict,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum NetworkInputIssue {
    UnsupportedStorage,
    UnsupportedResidentData,
    UnsupportedComplexLayers,
    InvalidRawGeometry,
    InvalidData,
}

impl NetworkInputIssue {
    pub(super) fn diagnostic(self, caller: NetworkInferenceCaller) -> InferenceDiagnostic {
        let (prefix, function) = match caller {
            NetworkInferenceCaller::Forward => ("RM-DL-FORWARD", "forward"),
            NetworkInferenceCaller::Predict => ("RM-DL-PREDICT", "predict"),
        };
        let (suffix, message) = match self {
            Self::UnsupportedStorage => ("STORAGE", "requires dense numeric or logical input storage"),
            Self::UnsupportedResidentData => (
                "RESIDENT-CLASS",
                "provider-resident input data must have real single or double storage",
            ),
            Self::UnsupportedComplexLayers => (
                "COMPLEX-LAYER",
                "known network layer does not support complex input",
            ),
            Self::InvalidRawGeometry => ("SHAPE", "input rank is incompatible with the network's input layer"),
            Self::InvalidData => (
                "DATA",
                "input must be a supported real or complex numeric/logical array or a dlarray containing one",
            ),
        };
        let message = match self {
            Self::UnsupportedStorage => format!("{function} {message}"),
            _ => message.to_owned(),
        };
        argument_error(format!("{prefix}-{suffix}"), message, 1)
    }
}
