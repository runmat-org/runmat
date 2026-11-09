//! Native process composition for the portable RunMat execution driver.

#[cfg(test)]
mod test_support;

mod accelerator_placement;
mod attempt_scratch;
mod machine_capacity;
pub(crate) mod console_bridge;
pub use console_bridge::{console_admission_from_subgrant, console_source_subgrant};
pub(crate) mod console_retention;
mod fea_delegation;
pub mod fea_proxy;
pub use accelerator_placement::active_provider_resource;
pub use machine_capacity::publication_staging_bytes;
mod analysis_acoustic;
mod analysis_acoustic_response;
mod analysis_electrical;
mod analysis_fluid;
pub use analysis_fluid::{execute_fluid_study_locally, FluidStudyExecutionPolicy};
mod analysis_fluid_transient;
pub use analysis_fluid_transient::{
    execute_fluid_transient_block_locally, CompletedFluidTransientBlockExecution,
    FluidTransientBlockExecutionPolicy,
};
mod analysis_nonlinear;
pub use analysis_nonlinear::source::nonlinear_solid_study_completion_from_source;
pub use analysis_nonlinear::{
    execute_nonlinear_solid_study_locally, NonlinearSolidStudyExecutionPolicy,
};
mod analysis_magnetic;
mod analysis_maxwell;
pub use analysis_electrical::source::electrical_study_completion_from_source;
pub use analysis_fluid::source::fluid_study_completion_from_source;
pub use analysis_magnetic::source::magnetic_study_completion_from_source;
pub use analysis_maxwell::source::maxwell_study_completion_from_source;
mod analysis_buckling;
mod analysis_frequency;
mod analysis_frequency_observations;
mod analysis_host;
mod analysis_kernel;
mod analysis_modal;
mod analysis_modal_frequency;
mod analysis_modal_transient;
mod analysis_placement;
mod analysis_prestressed_frequency;
mod analysis_prestressed_modal;
mod analysis_prestressed_modal_frequency;
mod analysis_progress;
mod analysis_random_response;
mod analysis_response_spectrum;
mod analysis_statistical_observations;
mod analysis_sequence;
mod analysis_task;
mod analysis_task_pool;
mod bounded_object_read;
pub use analysis_acoustic::source::acoustic_study_completion_from_source;
pub use analysis_acoustic_response::source::acoustic_response_study_completion_from_source;
mod analysis_thermal;
mod analysis_thermal_prestress;
mod analysis_transient;
mod checkpoint_evidence;
mod config;
pub use checkpoint_evidence::NativeCheckpointEvidence;
mod driver;
mod graph_submission;
mod durable;
mod error;
mod exact_geometry_admission;
mod exact_meshing_executor;
mod exact_meshing_job;
mod exact_meshing_root;
mod exact_structural_study_root;
mod exact_structural_sweep_root;
mod geometry_materialization;
mod geometry_meshing_root;
mod input_materialization;
mod input_materialization_contract;
mod local_store;
mod local_structural_study;
mod local_study_geometry;
mod materialized_project;
mod meshing_blob_cache;
mod meshing_cache_identity;
mod meshing_evaluator;
mod meshing_host;
mod meshing_platform;
mod meshing_subtask_cache;
mod metadata_identity;
mod native_pool;
mod program_host;
mod process_provider;
mod process_output;
pub use process_output::{RawOutputError, RawOutputState};
mod program_object_source;
mod program_session;
mod protocol;
mod remote;
mod remote_exact_meshing_executor;
mod runtime_meshing_provider;
mod service;
mod structural_study_root;
mod study_family;
pub mod cooperative;
mod family_study_publication;
pub use family_study_publication::import_family_study_result;
pub use study_family::{CompletedFamilyStudy, StudyFamilyKind};
pub mod supervisor;
mod task_outcome;
pub use task_outcome::NativeTaskQuiescence;
mod program_workload;
mod test_workload;
mod typed_fea;
pub use typed_fea::{plan_checked_fea_locally, plan_checked_fea_locally_with_timing};
mod worker;

pub use analysis_buckling::{
    execute_structural_buckling_locally, execute_structural_buckling_remotely,
    execute_structural_buckling_study_locally, execute_structural_buckling_study_remotely,
    CompletedStructuralBucklingSequenceStudy, CompletedStructuralBucklingStudy,
};
pub use analysis_frequency::{
    execute_structural_frequency_locally, execute_structural_frequency_locally_with_source,
    execute_structural_frequency_remotely, CompletedStructuralFrequencyEntry,
    CompletedStructuralFrequencyStudy, StructuralFrequencyExecutionPreparation,
};
pub use analysis_frequency_observations::{
    CompletedStructuralFrequencyHistory, CompletedStructuralFrequencyObservationSample,
};
pub use analysis_host::{
    execute_analysis_program_request, execute_reserved_analysis_program_request,
    NativeAnalysisHostLimits,
};
pub use analysis_kernel::NativeAnalysisKernel;
pub use analysis_modal::{
    execute_structural_modal_locally, execute_structural_modal_remotely,
    CompletedStructuralModalStudy,
};
pub use analysis_modal_frequency::{
    execute_structural_modal_frequency_study_locally_with_source,
    execute_structural_modal_frequency_study_remotely, CompletedStructuralModalFrequencyStudy,
};
pub use analysis_modal_transient::{
    execute_structural_modal_transient_locally, execute_structural_modal_transient_remotely,
    CompletedStructuralModalTransientStudy,
};
pub use analysis_placement::{
    execute_structural_workloads_locally, thermal_study_completion_from_source,
    NativeStructuralStudyPlacement, StructuralWorkloadLocalExecution,
};
pub use analysis_prestressed_frequency::{
    execute_structural_prestressed_frequency_study_locally,
    execute_structural_prestressed_frequency_study_locally_with_source,
    execute_structural_prestressed_frequency_study_remotely,
    CompletedStructuralPrestressedFrequencyStudy,
};
pub use analysis_prestressed_modal::{
    execute_structural_prestressed_modal_study_locally,
    execute_structural_prestressed_modal_study_remotely, CompletedStructuralPrestressedModalStudy,
};
pub use analysis_prestressed_modal_frequency::{
    execute_structural_prestressed_modal_frequency_study_locally_with_source,
    execute_structural_prestressed_modal_frequency_study_remotely,
    CompletedStructuralPrestressedModalFrequencyStudy,
};
pub use analysis_random_response::{
    execute_structural_random_response_locally, execute_structural_random_response_remotely,
    CompletedStructuralRandomResponseStudy,
};
pub use analysis_response_spectrum::{
    execute_structural_modal_response_spectrum_locally,
    execute_structural_modal_response_spectrum_remotely,
    CompletedStructuralModalResponseSpectrumStudy,
};
pub use analysis_sequence::{
    execute_structural_sequence_locally, execute_structural_sequence_remotely,
    StructuralSequenceExecutionPolicy,
};
pub use analysis_thermal::{
    execute_thermal_study_locally, execute_thermal_study_locally_with_source,
    execute_thermal_study_remotely, CompletedThermalStudyWithSource, ThermalStudyExecutionPolicy,
};
pub use analysis_thermal_prestress::{
    execute_thermal_prestress_study_locally_with_source, execute_thermal_prestress_study_remotely,
    CompletedThermalPrestressStudy, ThermalPrestressStructuralCompletion,
    ThermalPrestressStudyPreparation,
};
pub use analysis_transient::{
    execute_structural_transient_locally, execute_structural_transient_remotely,
    CompletedStructuralTransientSequence, RetainedStructuralTransientRecovery,
    StructuralTransientRecoveryPublisher,
};
pub use config::NativeExecutionConfig;
pub use error::{MetadataIoOperation, NativeExecutionError, NativeExecutionResult};
pub use exact_geometry_admission::{
    admit_prepared_exact_geometry, prepare_exact_geometry_admission, ExactGeometryAdmissionError,
    NativeMeshingContactDefinition, PreparedExactGeometryAdmission,
};
pub use exact_meshing_executor::{MeshingExecutionPolicy, NativeExactMeshingExecutor};
pub use exact_meshing_job::{
    mesh_exact_geometry, NativeExactMeshingJob, NativeExactMeshingJobError,
    NativeExactMeshingResult,
};
pub use exact_meshing_root::{
    decode_exact_meshing_root, prepare_exact_meshing_root, DecodedExactMeshingRoot,
    ExactMeshingRootPreparation, ExactMeshingRootRequest, PreparedExactMeshingRoot,
    EXACT_MESHING_ROOT_EXECUTION_MODE, EXACT_MESHING_ROOT_TARGET_PROFILE,
};
pub use exact_structural_study_root::{
    decode_exact_structural_study_root, import_exact_structural_study_result,
    import_structural_study_result, prepare_exact_structural_study_root,
    DecodedExactStructuralStudyRoot, ExactStructuralBucklingAnalysisResult,
    ExactStructuralFrequencyAnalysisResult, ExactStructuralModalResponseSpectrumAnalysisResult,
    ExactStructuralModalTransientAnalysisResult, ExactStructuralPrestressedFrequencyAnalysisResult,
    ExactStructuralPrestressedModalAnalysisResult, ExactStructuralRandomResponseAnalysisResult,
    ExactStructuralStudyAnalysisResult, ExactStructuralStudyResult,
    ExactStructuralStudyResultExpectation, ExactStructuralStudyRootPreparation,
    PreparedExactStructuralStudyRoot, RequiredStudyProgram, TopologyRegionResolution,
    EXACT_STRUCTURAL_STUDY_ROOT_EXECUTION_MODE, EXACT_STRUCTURAL_STUDY_ROOT_TARGET_PROFILE,
};
pub use exact_structural_sweep_root::{
    decode_exact_structural_sweep_root, import_exact_structural_sweep_result,
    import_exact_structural_sweep_result_catalog, import_structural_sweep_result,
    import_study_sweep_entries, prepare_exact_structural_sweep_root, prepare_structural_sweep_root,
    DecodedExactStructuralSweepRoot, ExactStructuralSweepEntryResult,
    ExactStructuralSweepEntryStatus, ExactStructuralSweepResult, ExactStructuralSweepResultCatalog,
    ExactStructuralSweepResultExpectation, ExactStructuralSweepRootPreparation,
    ExactStructuralSweepTerminal, PreparedExactStructuralSweepRoot, PreparedStructuralSweepRoot,
    StructuralSweepResultExpectation, StructuralSweepRootPreparation, StudySweepEntry,
    StudySweepTerminal, EXACT_STRUCTURAL_SWEEP_ROOT_EXECUTION_MODE,
    EXACT_STRUCTURAL_SWEEP_ROOT_TARGET_PROFILE,
};
pub use geometry_materialization::{
    execute_geometry_materialization_program_request, native_geometry_provider_profile,
    NativeGeometryMaterializationLimits,
};
pub use geometry_meshing_root::{
    decode_geometry_meshing_root, prepare_geometry_meshing_root, DecodedGeometryMeshingRoot,
    GeometryMeshingRootPreparation, GeometryMeshingRootRequest, PreparedGeometryMeshingRoot,
    RequiredGeometryMeshingProgram, GEOMETRY_MESHING_ROOT_EXECUTION_MODE,
    GEOMETRY_MESHING_ROOT_TARGET_PROFILE,
};
pub use local_structural_study::{
    execute_prepared_structural_study_locally,
    execute_prepared_structural_study_locally_with_observers,
    execute_prepared_study_locally_with_observers, LocalStructuralStudyObservers,
    LocalStudyCompletion,
};
pub use meshing_evaluator::{native_meshing_kernel_dispatcher, NativeMeshingEvaluatorProvider};
pub use meshing_host::{
    execute_meshing_program_request, run_meshing_worker_stdio, NativeMeshingHostLimits,
};
pub use meshing_platform::native_meshing_capability_cohort;
pub use process_provider::NATIVE_OBJECT_STORE_ROOT_ENV;
pub use program_object_source::NativeProgramObjectSource;
pub use program_session::{
    NativeProgramCompletion, NativeProgramFailure, NativeProgramSession, NativeProgramTask,
};
pub use protocol::{ProgramProgress, WorkerProcessMessage, WorkerResponse};
pub use remote::structural_study::{execute_study_root_on_pool, RemoteStudyRootOutcome};
pub use remote::{
    run_remote_driver_from_env, run_remote_meshing_worker_quic, run_remote_worker_from_env,
    run_remote_worker_quic, run_remote_worker_quic_ready_v1, run_remote_worker_relay,
    AuthenticatedPhysicalRetirement, QuicRemoteWorkerChannel, RelayRemoteWorkerChannel,
    RemoteAttempt, RemoteBundleReceipt, RemoteGraphTransfer, RemoteMeshingWorkerQuicRequest,
    RemoteObjectReceipt, RemotePoolCheckpoint, RemotePoolDriver, RemotePoolRestoreAuthority,
    RemoteSpmdGangCompletion, RemoteSpmdGangProgram, RemoteTaskCompletion, RemoteValueReceipt,
    RemoteWorkerChannel, RemoteWorkerChannelConfig, RemoteWorkerQuicRequest,
    RemoteWorkerRelayRequest,
};
pub use remote_exact_meshing_executor::RemoteExactMeshingExecutor;
pub use runmat_execution_artifact::cache::FilesystemObjectStore;
pub use runtime_meshing_provider::{
    mesh_source_with_native_execution, native_meshing_execution_provider,
    resolve_native_meshing_request, NativeMeshingPartitionPolicy, NativeMeshingSourceError,
    NativeMeshingSourceRequest,
};
pub use service::NativeExecutionService;
pub use structural_study_root::{
    decode_structural_study_root, prepare_structural_study_root, DecodedStructuralStudyRoot,
    PreparedStructuralStudyRoot, RequiredStructuralStudyProgram, StructuralStudyRootPreparation,
    STRUCTURAL_STUDY_ROOT_EXECUTION_MODE, STRUCTURAL_STUDY_ROOT_TARGET_PROFILE,
};
pub use structural_study_root::{
    NonlinearSolidStudyPlanning, StructuralStudyMaterialExpectation,
    StructuralStudyResultExpectation, StudyRootPlanning,
};
pub use test_workload::{
    execute_host_graph_request_with_graphics, execute_host_program_request,
    execute_host_program_request_with_graphics,
};
pub use typed_fea::{
    checked_fea_artifact_access, complete_frozen_fea_assessments, complete_local_fea_assessments,
    complete_remote_fea_assessments, freeze_fea_assessment_functions,
    materialize_checked_fea_study_geometry, prepare_checked_fea_study, reassess_saved_fea_locally,
    MaterializedFeaGeometry, PreparedFeaStudy,
};
pub use typed_fea::{
    execute_checked_fea_locally, execute_checked_fea_locally_with_timing,
    execute_prepared_fea_study_locally, imported_family_result, prepare_study_publication,
    CompletedAuthoredAcousticResponseStudy, CompletedAuthoredAcousticStudy,
    CompletedAuthoredElectricalStudy, CompletedAuthoredFluidStudy, CompletedAuthoredMagneticStudy,
    CompletedAuthoredMaxwellStudy, CompletedAuthoredNonlinearSolidStudy,
    CompletedAuthoredThermalStudy, CompletedFeaStudyResult, CompletedLineParametersStudy,
    CompletedLocalFeaCase,
    FamilyResultImportLimits, LocalFeaExecution, LocalFeaExecutionOptions, LocalFeaMeshObserver,
    LocalFeaProgressObserver, LocalSweepCase,
};
pub use worker::run_worker_stdio;
