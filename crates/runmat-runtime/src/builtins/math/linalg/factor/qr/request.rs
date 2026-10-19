//! QR factor output demand, column ordering, and permutation format are distinct.

use runmat_accelerate_api::ProviderQrColumnOrder;

use super::{PivotMode, QrMode};

/// A full envelope is an explicit Rust request, not the ambient script output count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QrOutputRequest {
    R,
    Qr,
    Qrp,
    FullEnvelope,
    FullEnvelopeUnpivoted,
}

impl QrOutputRequest {
    pub(super) fn column_order(self) -> ProviderQrColumnOrder {
        match self {
            Self::R | Self::Qr | Self::FullEnvelopeUnpivoted => ProviderQrColumnOrder::Unpivoted,
            Self::Qrp | Self::FullEnvelope => ProviderQrColumnOrder::Pivoted,
        }
    }

    pub(super) fn is_full_envelope(self) -> bool {
        matches!(self, Self::FullEnvelope | Self::FullEnvelopeUnpivoted)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct QrOptions {
    pub(super) mode: QrMode,
    pub(super) format: QrPermutationFormat,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) enum QrPermutationFormat {
    #[default]
    MatrixDefault,
    LegacyZeroVector,
    MatrixExplicit,
    VectorExplicit,
}

impl QrPermutationFormat {
    pub(super) fn pivot_mode(self) -> PivotMode {
        match self {
            Self::MatrixDefault | Self::MatrixExplicit => PivotMode::Matrix,
            Self::LegacyZeroVector | Self::VectorExplicit => PivotMode::Vector,
        }
    }

    pub(super) fn requires_permutation_output(self) -> bool {
        matches!(self, Self::MatrixExplicit | Self::VectorExplicit)
    }
}
