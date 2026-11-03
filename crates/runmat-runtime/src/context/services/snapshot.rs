//! Immutable host-service composition shared by invocation clones.
//! Workspace remains an inline caller projection in RuntimeServicePorts.
use super::*;

#[derive(Clone)]
pub(super) struct ServicePortSnapshot {
    pub(super) call: Option<Rc<dyn RuntimeCallService>>,
    pub(super) builtin: Option<Rc<dyn RuntimeBuiltinService>>,
    pub(super) base_workspace: Option<Rc<dyn RuntimeBaseWorkspaceService>>,
    pub(super) object: Option<Rc<dyn RuntimeObjectService>>,
    pub(super) host: Option<Rc<dyn RuntimeHostService>>,
    pub(super) graphics_presentation: Option<Rc<dyn RuntimeGraphicsPresentationService>>,
    pub(super) graphics_callbacks: Option<crate::context::WeakRuntimeGraphicsCallbackQueue>,
    pub(super) http: Option<Rc<dyn RuntimeHttpService>>,
    pub(super) preferences: Option<Rc<dyn RuntimePreferenceService>>,
    pub(super) user_path: Option<Rc<dyn RuntimeUserPathService>>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) process: Option<Rc<dyn RuntimeProcessService>>,
    pub(super) tcp: Option<Rc<dyn RuntimeTcpService>>,
    pub(super) error: Option<Rc<dyn RuntimeErrorService>>,
    pub(super) acceleration: Option<Rc<dyn RuntimeAccelerationService>>,
    pub(super) placement: Option<Rc<dyn RuntimePlacementService>>,
    pub(super) host_effect_resources: Option<Rc<dyn RuntimeHostEffectResourceService>>,
    pub(super) native: Option<Rc<dyn RuntimeNativeService>>,
    pub(super) parallel: Option<Rc<dyn RuntimeParallelService>>,
    pub(super) spmd: Option<Rc<dyn RuntimeSpmdService>>,
    pub(super) distributed: Option<Rc<dyn RuntimeDistributedService>>,
    pub(super) collective: Option<Rc<dyn RuntimeCollectiveService>>,
    pub(super) timing: RuntimeTimingPort,
    pub(super) local_tasks: Option<Rc<dyn crate::context::RuntimeLocalTaskService>>,
    pub(super) source_evaluation: Option<Rc<dyn crate::context::RuntimeSourceEvaluationService>>,
    pub(super) selected_source_evaluation:
        Option<runmat_execution::host::VerifiedNativeServiceIssuer>,
    pub(super) selected_local: Option<Rc<RuntimeSelectedLocalServices>>,
    pub(super) selected_network: Option<selected_network::RuntimeSelectedNetworkService>,
    pub(super) selected_ui: Option<selected_external::SelectedHostPort>,
    pub(super) selected_callback: Option<selected_external::SelectedCallbackPort>,
    pub(super) selected_dynamic: Option<selected_external::SelectedDynamicPort>,
    pub(super) blocking_cpu: Option<Rc<dyn crate::execution::RuntimeBlockingCpuService>>,
    pub(super) owned_builtin_work: Option<Rc<dyn crate::execution::RuntimeOwnedBuiltinWorkService>>,
    pub(super) callable_artifacts: Option<Rc<dyn crate::context::RuntimeCallableArtifactService>>,
    pub(super) program: Option<Rc<dyn RuntimeProgramService>>,
    pub(super) analysis: Option<Rc<dyn RuntimeAnalysisService>>,
    pub(super) fea_run_admission:
        Option<Rc<dyn crate::analysis::fea_run_admission::AsyncFeaRunAdmission>>,
    pub(super) sync_fea_run_admission:
        Option<Arc<dyn crate::analysis::fea_run_admission::FeaRunAdmission>>,
}
