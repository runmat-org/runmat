//! Structural response solvers that operate on the canonical second-order system.

mod dense;
mod direct_harmonic;
mod harmonic_balance;
mod modal_harmonic;
mod modal_projection;
mod modal_transient;
mod random_response;
mod response_spectrum;

pub use direct_harmonic::{
    solve_direct_harmonic, DirectHarmonicError, DirectHarmonicPolicy, DirectHarmonicResponse,
};
pub use harmonic_balance::{
    evaluate_harmonic_physical_state, HarmonicBalanceError, HarmonicPhysicalState,
};
pub use modal_harmonic::{
    solve_modal_harmonic, ModalHarmonicError, ModalHarmonicLoad, ModalHarmonicResponse,
};
pub use modal_projection::{ModalCoordinateQuantity, ModalProjectionError, ModalReducedSystem};
pub use modal_transient::{
    advance_modal_transient_state, continue_modal_transient, initialize_modal_transient_state,
    refresh_modal_transient_state, solve_modal_transient, ModalTransientError,
    ModalTransientHistory, ModalTransientInitialState, ModalTransientSample, ModalTransientStep,
};
pub(crate) use random_response::random_excitation_spectrum_digest;
pub use random_response::{
    assemble_described_random_response_intervals, assemble_modal_random_response_intervals,
    describe_modal_random_response, random_response_interval_breakpoints,
    solve_modal_random_response, solve_modal_random_response_interval,
    ModalRandomResponseIntervalIdentity, ModalRandomResponseIntervalResult,
    ModalRandomResponseResult, ModalRandomResponseResultIdentity, RandomResponseFieldLayout,
    RandomResponseFieldStatistics, RandomResponseFieldVariance, RandomResponseIntegrationEvidence,
    RandomResponseIntegrationPolicy, RandomResponseIntervalEvidence, RandomResponsePlanDescription,
};
pub use response_spectrum::{
    assemble_modal_response_spectrum_direction_results,
    assemble_modal_response_spectrum_directions, recover_modal_response_spectrum_fields,
    response_spectrum_deformation_recovery_unavailability, solve_modal_response_spectrum,
    solve_modal_response_spectrum_direction_with_operator,
    solve_modal_response_spectrum_with_operator, ModalResponseSpectrumContribution,
    ModalResponseSpectrumDirectionResult, ModalResponseSpectrumDirectionResultIdentity,
    ModalResponseSpectrumModeContribution, ModalResponseSpectrumModeKind,
    ModalResponseSpectrumResult, ModalResponseSpectrumResultIdentity,
    ModalResponseSpectrumTruncationEvidence, ResponseSpectrumDeformationRecovery,
    ResponseSpectrumDeformationRecoveryUnavailability, ResponseSpectrumDisplacementConversion,
    ResponseSpectrumFieldRecovery, ResponseSpectrumFieldRecoveryIdentity,
    ResponseSpectrumLinearField, ResponseSpectrumNodalEnvelope, ResponseSpectrumNodalField,
    ResponseSpectrumNodalQuantity, ResponseSpectrumResultSemantics, StructuralLinearFieldQuantity,
};
