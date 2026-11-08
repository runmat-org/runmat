export { physicalLengthUnit, physicalLengthUnitName, isPhysicalLengthUnit, resolvedSourceLengthUnit, sourceLengthUnitLabel, PHYSICAL_LENGTH_UNITS } from "./physical.js";
export type { LengthUnit, LengthUnitEvidence } from "./physical.js";
import { resolveNativeRuntimeSettings, editNativeUserRuntimeSettings } from "./settings.js";
import type { HostRuntimeSettings, ResolvedRuntimeSettings, UserRuntimeSettingsEdit, RuntimeSettingsNativeExports, PreparedClusterProgram } from "./settings.js";
export type { HostRuntimeSettings, ResolvedRuntimeSettings, RuntimeSettingsConfig, RuntimeSettingSource, UserRuntimeSettingsEdit, SettingsJsonValue, PreparedClusterProgram } from "./settings.js";
import type { CapturedProjectResolver } from "./package-cache/browser-resolver.js";
import { invokeOpenTelemetry } from "./observability.js";
import type { OpenTelemetryNativeExports, OpenTelemetryExportStats, OpenTelemetryStartupHeaders } from "./observability.js";
export type { OpenTelemetryExportStats, OpenTelemetryStartupHeaders } from "./observability.js";
import { checkNativeSource } from "./source-check.js";
import { listNativeProjectEntrypoints, runNativeProjectEntrypoint, prepareNativeNamedClusterProgram } from "./project-entrypoints.js";
import type {
  NamedProjectEntrypoint, ProjectEntrypointContext, ProjectEntrypointDescription,
  RunProjectEntrypointOptions, ProjectEntrypointRunResult,
  ProjectEntrypointSessionExports, ProjectEntrypointModuleExports,
} from "./project-entrypoints.js";
export type {
  NamedProjectEntrypoint, ProjectEntrypointContext, ProjectEntrypointDescription,
  ProjectEntrypointFailure, ProjectEntrypointSupport, RunProjectEntrypointOptions,
  ProjectEntrypointRunResult,
} from "./project-entrypoints.js";
import type { SourceCheckNativeExports, SourceCheckRequest, SourceCheckReport } from "./source-check.js";
export type {
  SourceCheckTarget, SourceCheckRequest, SourceCheckOutcome, SourceCheckLocation,
  SourceCheckDiagnostic, SourceCheckSummary, SourceCheckDocument, SourceCheckReport,
  SourceAnalysisCompleteness,
} from "./source-check.js";
import type {
  RunMatProgramTaskCancellation,
  RunMatProgramTaskTerminal,
  RunMatWorkerProgramInvocation,
  RunMatProgramWorkerInventory,
} from "./execution/program-terminal.js";
export type {
  RunMatProgramExecutionAssignment,
  RunMatProgramAcceleratorLease,
  RunMatWorkerAcceleratorDevice,
  RunMatWorkerHostCapability,
  RunMatProgramWorkerInventory,
  RunMatProgramCancellationReason,
  RunMatProgramTaskCancellation,
  RunMatProgramTaskRetirement,
  RunMatProgramTaskTerminal,
  RunMatProgramRuntimeFailure,
  RunMatProgramExecutionResponse,
  RunMatWorkerProgramInvocation,
} from "./execution/program-terminal.js";
export {
  sameProgramAssignment,
  isRetiredProgramTerminal,
} from "./execution/program-terminal.js";
import type { InputManifest } from "./generated/collaboration-contracts.js";
export type { InputManifest, AtomicCreateResult } from "./generated/collaboration-contracts.js";
export { MAX_WORKSPACE_PUBLICATION_EDITS, MAX_WORKSPACE_PUBLICATION_BYTES } from "./generated/fea-workspace-contracts.js";
export type {
  WorkspaceCaptureRequest,
  WorkspaceSnapshot as SourceWorkspaceSnapshot,
  WorkspacePublicationRequest,
  WorkspacePublicationOutcome,
  WorkspaceEdit as SourceWorkspaceEdit,
  WorkspaceRevision as SourceWorkspaceRevision,
} from "./generated/fea-workspace-contracts.js";
import type {
  RunMatFilesystemProvider
} from "./fs/provider-types.js";
import type {
  FeaCapabilities,
  FeaExecutionChoices,
  FeaExecutionCandidate,
  FeaExecutionDuration,
  FeaExecutionMode,
  FeaExecutionReadiness,
  FeaSourceCheckResult,
  FeaBrowseRequest,
  FeaBrowseOutcome,
  FeaWorkingCheckRequest,
  FeaChangeSetId,
  FeaChangeSetRequest,
  FeaChangeSetReviewRequest,
  FeaChangeSetReview,
  FeaChangeSetCheck,
  FeaChangeSetPublication,
  FeaWorkspaceCheckRequest,
  FeaWorkspaceCheckResult,
  SourceDigest,
  FeaSemanticProgram,
  FeaSourceIndex,
  FeaSourceIndexEntry,
  FeaSourceIdentity,
  AuthoredStudy,
  AuthoredModel,
  GeometryDefinition,
  GeneratorCallable,
  MeshDefinition,
  FeaDeclaredMaterial,
  MaterialModel,
  FeaDiagnosticsArtifactPayload,
  FeaDocumentKind,
  FeaFieldBinding,
  FeaFieldBindingsArtifactPayload,
  FeaFieldKind,
  FeaFieldDescriptor,
  FeaFieldLocation,
  FeaFieldPagingDescriptor,
  FeaFieldRequestOptions,
  FeaFieldResult,
  FeaFieldStorage,
  FeaFieldStorageRef,
  FeaObjectArtifactMetadata,
  FeaProgressEvent,
  FeaProgressPhase,
  FeaProgressStatus,
  FeaRunDatasetFieldPagingPolicy,
  FeaRunDatasetPayload,
  FeaRunDatasetStudyRef,
  FeaRunResult
} from "./fea-contracts.js";
import {
  isFeaSourceCheckContract,
  isFeaSemanticProgram,
  FEA_ARTIFACT_MANIFEST_KIND,
  FEA_DATASET_ARTIFACT_KIND,
  FEA_DIAGNOSTICS_ARTIFACT_KIND,
  FEA_DIAGNOSTICS_SCHEMA_VERSION,
  FEA_FIELD_DEFAULT_MATERIALIZE_LIMIT,
  FEA_FIELD_DEFAULT_PAGE_SIZE,
  FEA_FIELD_BINDINGS_ARTIFACT_KIND,
  FEA_FIELD_BINDINGS_SCHEMA_VERSION,
  FEA_OBJECT_ARTIFACT_METADATA_SCHEMA_VERSION,
  FEA_RUN_DATASET_KIND,
  FEA_RUN_DATASET_SCHEMA_VERSION,
} from "./fea-contracts.js";
import { createDefaultFsProvider } from "./fs/default.js";
import type { FreezeTestSnapshotInput, ProjectTestLayout, RunMatTestPreparationNative } from "./testing/types.js";
export type { FreezeTestSnapshotInput, ProjectTestLayout } from "./testing/types.js";
import type { PlotPresenceFrame, PlotPointerPosition } from "./plot-presence.js";
export type { PlotPresenceFrame, PlotPointerPosition } from "./plot-presence.js";
import { createIndexedDbPackageCache } from "./package-cache/indexeddb.js";
import { fetchGitTreeInventoryWire } from "./package-cache/git-gateway.js";
import { BrowserProjectResolver } from "./package-cache/browser-resolver.js";
import { BrowserProjectSession } from "./package-cache/project-session.js";
import type {
  BrowserProjectResolverConfig,
  BrowserProjectSessionConfig,
  IndexedDbPackageCacheOptions,
  PackageCacheSnapshot,
  PackageCacheGcPlan,
  PackageCacheStatus,
  RunMatPackageCacheProvider
} from "./package-cache/index.js";
import type {
  GitAcquisitionPlan,
  GitAcquisitionPlanRequest,
  GitGatewayRequest,
  GitSnapshotWire,
  GitTreeInventoryWire,
  ServerGitGatewayOptions
} from "./package-cache/git-gateway.js";
import { __internals as workspaceHoverInternals } from "./workspace-hover.js";
import { installWebGpuCompatibilityShims } from "./webgpu-shims.js";
export {
  FEA_ARTIFACT_MANIFEST_KIND,
  FEA_DATASET_ARTIFACT_KIND,
  FEA_DIAGNOSTICS_ARTIFACT_KIND,
  FEA_DIAGNOSTICS_SCHEMA_VERSION,
  FEA_FIELD_DEFAULT_MATERIALIZE_LIMIT,
  FEA_FIELD_DEFAULT_PAGE_SIZE,
  FEA_FIELD_BINDINGS_ARTIFACT_KIND,
  FEA_FIELD_BINDINGS_SCHEMA_VERSION,
  FEA_OBJECT_ARTIFACT_METADATA_SCHEMA_VERSION,
  FEA_RUN_DATASET_KIND,
  FEA_RUN_DATASET_SCHEMA_VERSION,
  FEA_RUN_CELL_ID,
  FEA_RUN_KIND,
  FEA_RUN_MANIFEST_METADATA_SCHEMA_VERSION,
  feaRunArtifactRefId,
} from "./fea-contracts.js";
export type {
  AuthoredBoundary,
  AuthoredConnector,
  AuthoredConstraint,
  AuthoredExcitation,
  AuthoredExcitationCase,
  AuthoredInitialCondition,
  AuthoredInterface,
  AuthoredMaterialAssignment,
  AuthoredModel,
  AuthoredOutput,
  AuthoredPropertyAssignment,
  AuthoredStudy,
  FeaDeclaredMaterial,
  FeaModelMemberKind,
  FeaSemanticProgram,
  FeaSourceIdentity,
  FeaSourceIndex,
  FeaSourceIndexEntry,
  GeometryDefinition,
  GeometryRegionRef,
  GeneratorCallable,
  MaterialModel,
  MeshDefinition,
  PhysicalSignal,
  ProcedureStep,
  FeaAnalysisProcedure,
  FeaBoundaryInput,
  FeaCanonicalSequence,
  FeaCapabilities,
  FeaExecutionChoices,
  FeaExecutionCandidate,
  FeaExecutionDuration,
  FeaExecutionMode,
  FeaExecutionReadiness,
  FeaSourceCheckResult,
  FeaBrowseRequest,
  FeaBrowseOutcome,
  FeaWorkingCheckRequest,
  FeaChangeSetId,
  FeaChangeSetRequest,
  FeaChangeSetReviewRequest,
  FeaChangeSetReview,
  FeaChangeSetCheck,
  FeaChangeSetPublication,
  FeaWorkspaceCheckRequest,
  FeaWorkspaceCheckResult,
  SourceDigest,
  FeaDiagnosticsArtifactPayload,
  FeaDocumentKind,
  FeaFieldBinding,
  FeaFieldBindingsArtifactPayload,
  FeaFieldKind,
  FeaFieldDescriptor,
  FeaFieldLocation,
  FeaFieldPagingDescriptor,
  FeaFieldRequestOptions,
  FeaFieldResult,
  FeaFieldStorage,
  FeaFieldStorageRef,
  FeaObjectArtifactMetadata,
  FeaAcceleratorAdmissionPolicy,
  FeaComputeBackend,
  FeaConnectorInput,
  FeaConstraintInput,
  FeaDomainsInput,
  FeaElementOrder,
  FeaExecutionInput,
  FeaMaxwellHarmonicOptions,
  FeaExcitationCaseInput,
  FeaExcitationInput,
  FeaGeometryUnits,
  FeaInitialConditionInput,
  FeaInterfaceInput,
  FeaMaterialAssignmentInput,
  FeaMaterialInput,
  FeaPhysicsFamily,
  FeaProcedureStep,
  FeaRegionInput,
  FeaRegionSelector,
  FeaRegionTarget,
  FeaResponseSpectrumInput,
  FeaSignalInput,
  FeaProgressEvent,
  FeaProgressPhase,
  FeaProgressStatus,
  FeaRunDatasetFieldPagingPolicy,
  FeaRunDatasetPayload,
  FeaRunDatasetStudyRef,
  FeaRunResult,
  FeaRunArtifactRole
} from "./fea-contracts.js";
export {
  createInMemoryFsProvider,
  createIndexedDbFsHandle,
  createIndexedDbFsProvider,
  createDefaultFsProvider,
  createRemoteFsProvider,
  MemoryVolume
} from "./fs/index.js";
export type {
  InMemoryFsProviderOptions,
  IndexedDbFsHandle,
  IndexedDbProviderOptions,
  RemoteProviderOptions,
  VolumeSnapshotEntry
} from "./fs/index.js";
export type {
  RunMatFsFileType,
  RunMatFilesystemDirEntry,
  RunMatFilesystemMetadata,
  RunMatFilesystemProvider,
  RunMatOpenFileDialogFilter,
  RunMatOpenFileDialogRequest,
  RunMatOpenFileDialogSelection
} from "./fs/provider-types.js";
export {
  BrowserPackageMountFilesystem,
  createAndRegisterBrowserRecipientKey,
  createBrowserPrivatePackageKeyStore,
  createIndexedDbPackageCache,
  generateBrowserRecipientKey,
  invalidateBrowserPrivatePackageArtifacts,
  ImmutableBrowserPackageMount,
  revokeAndRemoveBrowserRecipientKey
} from "./package-cache/index.js";
export { BrowserProjectResolver, BrowserProjectSession };
export type {
  BrowserProjectResolveOptions,
  BrowserProjectResolveRequest,
  BrowserProjectResolverConfig,
  BrowserProjectResolverNative,
  BrowserResolvedProject,
  BrowserProjectSessionConfig,
  BrowserProjectSessionHandle,
  BrowserProjectSessionNative,
  BrowserProjectSessionResolution,
  BrowserProjectSessionResolveRequest,
  BrowserRecipientKeyRegistration,
  BrowserPrivatePackageKeyStore,
  GeneratedBrowserRecipientKey,
  RegisterBrowserRecipientKey,
  RevokeBrowserRecipientKey,
  BrowserMountEntry,
  BrowserTreeEntry,
  BrowserTreeManifest,
  IndexedDbPackageCacheHandle,
  IndexedDbPackageCacheOptions,
  PackageCacheCommitOutcome,
  PackageCacheGcPlan,
  PackageCacheFaultInjector,
  PackageCacheRevision,
  PackageCacheSnapshot,
  PackageCacheStatus,
  PackageCacheTransaction,
  RunMatPackageCacheProvider,
  GitGatewayRequest,
  GitGatewaySelector,
  GitAcquisitionIntent,
  GitAcquisitionPlan,
  GitAcquisitionPlanRequest,
  GitAcquisitionPolicy,
  GitSnapshotWire,
  GitSourceWire,
  GitTreeInventoryWire,
  ServerGitGatewayOptions
} from "./package-cache/index.js";
export { createWorkspaceHoverProvider } from "./workspace-hover.js";
export type {
  WorkspaceHoverOptions,
  WorkspaceHoverController,
  HoverFormatState
} from "./workspace-hover.js";
export { createFusionPlanAdapter } from "./fusion-plan.js";
export type {
  FusionPlanAdapter,
  FusionPlanAdapterOptions
} from "./fusion-plan.js";
export type LanguageCompatMode = "runmat" | "matlab" | "strict";
/** Preset logging level; module-filter expressions are not accepted. */
export type RunMatLogLevel = "trace" | "debug" | "info" | "warn" | "error";

export type RunmatConfigFormat = "toml" | "json";
export type DesktopRunHistoryMode = "off" | "budgeted" | "full";
export type DesktopRunLogMode = "off" | "errors" | "all";
export type DesktopNotebookOnError = "stop" | "continue";
export type DesktopNotebookRerunAfterCancel = "remaining" | "all";

export interface ResolvedDesktopConfig {
  artifacts: {
    root: string;
  };
  run_history: {
    mode: DesktopRunHistoryMode;
    trace: boolean;
    logs: DesktopRunLogMode;
  };
  script: {
    clear_workspace_before_run: boolean;
    clear_figures_before_run: boolean;
  };
  notebook: {
    on_error: DesktopNotebookOnError;
    rerun_after_cancel: DesktopNotebookRerunAfterCancel;
  };
}

export interface ResolvedRunmatConfig {
  desktop: ResolvedDesktopConfig;
  runtime: Record<string, unknown> & {
    error_namespace: string;
    language: Record<string, unknown> & { compat: LanguageCompatMode };
    accelerate: Record<string, unknown> & { enabled: boolean };
    plotting: Record<string, unknown> & {
      export?: Record<string, unknown> & { scene_budget_bytes: number };
    };
  };
}

export interface RunmatConfigPatch {
  desktop?: {
    artifacts?: { root?: string };
    run_history?: {
      mode?: DesktopRunHistoryMode;
      trace?: boolean;
      logs?: DesktopRunLogMode;
    };
    script?: {
      clear_workspace_before_run?: boolean;
      clear_figures_before_run?: boolean;
    };
    notebook?: {
      on_error?: DesktopNotebookOnError;
      rerun_after_cancel?: DesktopNotebookRerunAfterCancel;
    };
  };
  runtime?: {
    accelerate_enabled?: boolean;
    scene_budget_bytes?: number;
  };
}

export interface LegacyRunmatConfigMigration {
  source: string;
  changed: boolean;
  removedKeys: string[];
}

export type WasmInitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

/** Explicit session-isolated settings; omission retains browser persistence. */
export type ApplicationSettingsMode = "isolatedMemory";

export interface RunMatInitOptions {
  runtimeSettings?: HostRuntimeSettings;
  /** Runtime-only collector credentials; never stored in runtimeSettings or user/project files. */
  openTelemetryHeaders?: OpenTelemetryStartupHeaders;
  applicationSettings?: ApplicationSettingsMode;
  enableGpu?: boolean;
  enableJit?: boolean;
  verbose?: boolean;
  logLevel?: RunMatLogLevel;
  gpuBufferPoolMaxPerKey?: number;
  telemetryConsent?: boolean;
  telemetryId?: string;
  telemetryRunKind?: "script" | "repl" | "benchmark" | "install";
  telemetryEmitter?: (envelope: unknown) => void;
  wgpuPowerPreference?: "auto" | "high-performance" | "low-power";
  wgpuForceFallbackAdapter?: boolean;
  wasmModule?: WasmInitInput;
  fsProvider?: RunMatFilesystemProvider;
  packageCache?: IndexedDbPackageCacheOptions & {
    provider?: RunMatPackageCacheProvider;
  };
  executionHost?: RunMatExecutionHost;
  plotCanvas?: HTMLCanvasElement;
  scatterTargetPoints?: number;
  surfaceVertexBudget?: number;
  emitFusionPlan?: boolean;
  callstackLimit?: number;
  errorNamespace?: string;
  language?: {
    compat?: LanguageCompatMode;
  };
}

export type BrowserExecutionTopology = "coordinator" | "flat" | "serial";

export interface BrowserExecutionHostCapabilities {
  topology: BrowserExecutionTopology;
  maxWorkers: number;
  /** Exact owned builtin route; successful launch responses and cancelOwned acknowledgments certify task settlement. */
  ownedBuiltinWork?: boolean;
  /** Selected Direct/Graph route with an exact Runtime-owned retirement terminal. Rejection is not proof. */
  selectedProgramWork?: boolean;
}

export interface BrowserProgramLaunchRequest {
  taskId: string;
  workerId: string;
  program: Record<string, unknown>;
}

export interface RunMatExecutionHost {
  capabilities: BrowserExecutionHostCapabilities;
  /** Actual module/factory host+device inventory; required for selectedProgramWork. */
  workerInventory?: RunMatProgramWorkerInventory;
  launch(request: BrowserProgramLaunchRequest): Promise<Record<string, unknown>>;
  cancel(taskId: string): void | Promise<void>;
  /** Required only for ownedBuiltinWork: literal task-bound physical-exit proof. */
  cancelOwned?(taskId: string): Promise<{
    taskId: string;
    state: "terminated" | "notRunning";
  }>;
  /** Required for selectedProgramWork. The terminal must preserve the actual assignment. */
  launchProgram?(request: BrowserProgramLaunchRequest): Promise<RunMatProgramTaskTerminal>;
  cancelProgram?(request: RunMatProgramTaskCancellation): Promise<RunMatProgramTaskTerminal>;
}

export type FigureEventKind = "created" | "updated" | "cleared" | "closed";

export interface FigureEvent {
  handle: number;
  kind: FigureEventKind;
  fingerprint?: string;
  figure?: FigureSnapshot;
}

export interface FigureSnapshot {
  layout: FigureLayout;
  metadata: FigureMetadata;
  plots: FigurePlotDescriptor[];
}

export interface FigureTextStyle {
  colorRgba?: [number, number, number, number];
  fontSize?: number;
  fontWeight?: string;
  fontAngle?: string;
  interpreter?: string;
  visible: boolean;
}

export interface FigureLayout {
  axesRows: number;
  axesCols: number;
  axesIndices: number[];
  axesGridCells?: Array<AxesGridCell | null>;
}

export interface AxesGridCell {
  rows: number;
  columns: number;
  index: number;
}

export interface FigureMetadata {
  title?: string;
  sgTitle?: string;
  sgTitleStyle?: FigureTextStyle;
  xLabel?: string;
  yLabel?: string;
  gridEnabled: boolean;
  legendEnabled: boolean;
  colorbarEnabled: boolean;
  axisEqual: boolean;
  backgroundRgba: [number, number, number, number];
  colormap?: string;
  colorLimits?: [number, number];
  legendEntries: FigureLegendEntry[];
}

export interface FigureLegendEntry {
  label: string;
  plotType: FigurePlotKind;
  colorRgba: [number, number, number, number];
}

export interface FigurePlotDescriptor {
  kind: FigurePlotKind;
  label?: string;
  axesIndex: number;
  colorRgba: [number, number, number, number];
  visible: boolean;
}

export type FigurePlotKind =
  | "line"
  | "line3"
  | "scatter"
  | "bar"
  | "error_bar"
  | "stairs"
  | "stem"
  | "area"
  | "quiver"
  | "pie"
  | "surface"
  | "mesh"
  | "scatter3"
  | "contour"
  | "contour_fill";

export type FigureEventListener = (event: FigureEvent) => void;
export type HoldMode = "on" | "off" | "toggle" | boolean;

/** Omitted fields preserve the runtime's existing navigation defaults. */
export interface PlotNavigationOptions {
  preset?: "legacy" | "cad" | "dcc";
  orbitCenter?: "viewCenter" | "origin" | "underCursor";
  zoomSpeed?: number;
  reverseZoom?: boolean;
  zoomToCursor?: boolean;
  trackpadPan?: boolean;
}
export type PlotAntialiasing = "off" | "x2" | "x4" | "x8";

export type PlotSurfaceEvent =
  | {
      kind: "mouseDown";
      x: number;
      y: number;
      button: number;
      shiftKey?: boolean;
      ctrlKey?: boolean;
      altKey?: boolean;
      metaKey?: boolean;
    }
  | {
      kind: "mouseUp";
      x: number;
      y: number;
      button: number;
      shiftKey?: boolean;
      ctrlKey?: boolean;
      altKey?: boolean;
      metaKey?: boolean;
    }
  | {
      kind: "mouseMove";
      x: number;
      y: number;
      dx: number;
      dy: number;
      shiftKey?: boolean;
      ctrlKey?: boolean;
      altKey?: boolean;
      metaKey?: boolean;
    }
  | {
      /** Hosts classify two-finger scroll separately; Ctrl/Cmd pinch remains zoom. */
      kind: "wheel" | "trackpadScroll";
      x: number;
      y: number;
    wheelDeltaX: number;
    wheelDeltaY: number;
      wheelDeltaMode?: number;
      /** Override pixel-scroll inference for precise mice or known trackpads. */
      trackpad?: boolean;
      shiftKey?: boolean;
      ctrlKey?: boolean;
      altKey?: boolean;
      metaKey?: boolean;
    };

export type PlotSurfaceHostAction = {
  kind: "createFeaStudy";
};

export type StdoutStreamKind = "stdout" | "stderr" | "clear";

export interface StdoutEntry {
  stream: StdoutStreamKind;
  text: string;
  timestampMs: number;
}

export type StdoutListener = (entry: StdoutEntry) => void;

export interface RuntimeLogEntry {
  ts: string;
  level: string;
  target: string;
  message: string;
  traceId?: string;
  spanId?: string;
  fields?: Record<string, unknown>;
}

export type RuntimeLogListener = (entry: RuntimeLogEntry) => void;

export interface TraceEvent {
  name: string;
  cat: string;
  ph: string;
  ts: number;
  dur?: number;
  pid?: number;
  tid?: number;
  traceId?: string;
  spanId?: string;
  parentSpanId?: string;
  args?: Record<string, unknown>;
}

export type TraceEventListener = (entries: TraceEvent[]) => void;

export type SignalTraceHandler = <T>(traceId: string, name: string, fn: () => T) => T;

let signalTraceHandler: SignalTraceHandler | null = null;

export function setSignalTraceHandler(handler: SignalTraceHandler | null): void {
  signalTraceHandler = handler;
}

export function withSignalTrace<T>(traceId: string | undefined, name: string, fn: () => T): T {
  if (traceId && signalTraceHandler) {
    return signalTraceHandler(traceId, name, fn);
  }
  return fn();
}

export type InputRequest =
  | { kind: "line"; prompt: string; echo: boolean }
  | { kind: "keyPress"; prompt: string; echo: boolean };

export interface ResumeInputPayload {
  value?: string | number | boolean;
  line?: string | number | boolean;
  kind?: "line" | "keyPress";
  error?: string;
}

export type ResumeInputScalar = string | number | boolean | null | undefined;
export type ResumeInputValue = ResumeInputScalar | ResumeInputPayload;

export type InputHandlerResult = ResumeInputValue | Promise<ResumeInputValue>;
export type InputHandler = (request: InputRequest) => InputHandlerResult;

export interface AxesInfo {
  handle: number;
  axesRows: number;
  axesCols: number;
  activeIndex: number;
}

export interface FigureImageOptions {
  handle?: number;
  width?: number;
  height?: number;
  cameraState?: FigureImageCameraState;
  textmark?: string;
}

/** Observation options are preserved through this JS wrapper and validated by WASM. */
export const GEOMETRY_OBSERVATION_API_VERSION = 1;

export interface GeometrySceneImageOptions {
  handle: number;
  selectedRegionIds?: string[];
  width?: number;
  height?: number;
  view?: string | null;
  visibility?: GeometrySceneImageVisibilityState | null;
  section?: GeometrySceneImageSectionState | null;
}

export interface GeometrySceneImageVisibilityState {
  mode: "all" | "hidden" | "isolated" | string;
  hidden_ids?: string[];
  isolated_ids?: string[];
}

export interface GeometrySceneImageSectionState {
  plane: GeometrySceneImageSectionPlane;
  /** A second cutting plane: only what lies on the removed side of both planes is cut away. */
  secondPlane?: GeometrySceneImageSectionPlane | null;
  /** Fill the solid's cut faces (the default). False leaves the cut open. */
  cap?: boolean;
}

export interface GeometrySceneImageSectionPlane {
  normal: [number, number, number];
  origin?: [number, number, number] | null;
  offset?: number | null;
  label?: string | null;
}

export type FigureImageCameraProjectionState =
  | {
      kind: "perspective";
      fov: number;
      near: number;
      far: number;
    }
  | {
      kind: "orthographic";
      left: number;
      right: number;
      bottom: number;
      top: number;
      near: number;
      far: number;
    };

export interface FigureImageAxesCameraState {
  position: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
  zoom: number;
  aspectRatio: number;
  projection: FigureImageCameraProjectionState;
}

export interface FigureImageCameraState {
  activeAxes: number;
  axes: FigureImageAxesCameraState[];
}

export interface FigureBindingError extends Error {
  code: "InvalidHandle" | "InvalidSubplotGrid" | "InvalidSubplotIndex" | "RenderFailure" | "Unknown";
  handle?: number;
  rows?: number;
  cols?: number;
  index?: number;
  details?: string;
}

export interface MatlabWarning {
  identifier: string;
  message: string;
}

export interface StdinEventLog {
  prompt: string;
  kind: "line" | "keyPress";
  echo: boolean;
  value?: string;
  error?: string;
}

export type RuntimeFlow =
  | { kind: "no-value" }
  | { kind: "single"; value: unknown }
  | { kind: "output-list"; values: unknown[] }
  | { kind: "comma-list"; values: unknown[] }
  | { kind: "dynamic-list"; id: string };

export interface ExecuteRequestTextSource {
  kind: "text";
  name: string;
  text: string;
}

export interface ExecuteRequestPathSource {
  kind: "path";
  path: string;
}

export type ExecuteRequestSource = ExecuteRequestTextSource | ExecuteRequestPathSource;

export interface ExecuteRequest {
  inputManifest?: InputManifest;
  source: ExecuteRequestSource;
  compatibility?: LanguageCompatMode;
  hostPolicy?: {
    topLevelAwait?: boolean;
  };
  requestedOutputs?: number;
  retainFigures?: boolean;
}

export interface ExecuteResult {
  /** Canonical lock for the dependency graph actually used for this captured run. */
  dependencyLock?: string | null;
  flow: RuntimeFlow;
  displayEvents: DisplayEvent[];
  valueText?: string;
  valueJson?: unknown;
  typeInfo?: string;
  error?: RunMatErrorDetails;
  executionTimeMs: number;
  usedJit: boolean;
  stdout: StdoutEntry[];
  workspace: WorkspaceSnapshot;
  figuresTouched: number[];
  warnings: MatlabWarning[];
  stdinEvents: StdinEventLog[];
  profiling?: ProfilingSummary;
  fusionPlan?: FusionPlanSnapshot;
}

export interface DisplayEvent {
  label: string;
  valueText: string;
  valueJson: unknown;
  span: {
    start: number;
    end: number;
  };
}

export type RunMatErrorKind = "syntax" | "semantic" | "compile" | "runtime";

export interface RunMatErrorSpan {
  start: number;
  end: number;
  line: number;
  column: number;
}

export interface RunMatErrorDetails {
  kind: RunMatErrorKind;
  message: string;
  identifier?: string;
  diagnostic: string;
  /** Canonical meshing diagnostic, retained without translating its domain contract. */
  meshingFailure?: unknown;
  span?: RunMatErrorSpan;
  sourceId?: number;
  sourceName?: string;
  sourcePath?: string;
  callstack: string[];
  callstackElided?: number;
}

export class RunMatExecutionError extends Error {
  readonly kind: RunMatErrorKind;
  readonly identifier?: string;
  readonly diagnostic: string;
  readonly meshingFailure?: unknown;
  readonly span?: RunMatErrorSpan;
  readonly sourceId?: number;
  readonly sourceName?: string;
  readonly sourcePath?: string;
  readonly callstack: string[];
  readonly callstackElided?: number;

  constructor(details: RunMatErrorDetails) {
    super(details.message);
    this.name = "RunMatExecutionError";
    this.kind = details.kind;
    this.identifier = details.identifier;
    this.diagnostic = details.diagnostic;
    this.meshingFailure = details.meshingFailure;
    this.span = details.span;
    this.sourceId = details.sourceId;
    this.sourceName = details.sourceName;
    this.sourcePath = details.sourcePath;
    this.callstack = details.callstack;
    this.callstackElided = details.callstackElided;
  }
}

export type WorkspaceResidency = "cpu" | "gpu" | "unknown";

/** A language string element. Every string is literal text; only null is missing. */
export type StringElement = string | null;

/** String projections returned in MaterializedVariable.valueJson. */
export interface StringValueJson {
  kind: "string";
  value: StringElement;
  shape: [1, 1];
}

/** Exact browser admission revision; older sentinel-based runtimes are incompatible. */
export const STRING_ELEMENT_WIRE_SCHEMA_VERSION = 1 as const;

export interface StringArrayValueJson {
  kind: "string-array";
  shape: number[];
  rows: number;
  cols: number;
  preview: StringElement[];
  length: number;
  truncated: boolean;
}

export interface WorkspaceSnapshot {
  full: boolean;
  version: number;
  values: WorkspaceEntry[];
  removals?: string[];
}

export interface WorkspaceEntry {
  name: string;
  className: string;
  dtype?: string;
  shape: number[];
  isGpu: boolean;
  sizeBytes?: number;
  preview?: WorkspacePreview;
  residency: WorkspaceResidency;
  previewToken?: string;
}

export interface WorkspacePreview {
  values: number[];
  truncated: boolean;
}

export type WorkspaceMaterializeSelector =
  | string
  | {
      name?: string;
      previewToken?: string;
    };

export interface MaterializeSliceOptions {
  start: number[];
  shape: number[];
}

export interface MaterializeVariableOptions {
  limit?: number;
  slice?: MaterializeSliceOptions;
}

export interface MaterializedVariable {
  name: string;
  className: string;
  dtype?: string;
  shape: number[];
  isGpu: boolean;
  residency: WorkspaceResidency;
  sizeBytes?: number;
  preview?: WorkspacePreview;
  valueText: string;
  valueJson: unknown;
}

export interface ProfilingSummary {
  totalMs: number;
  cpuMs?: number;
  gpuMs?: number;
  kernelCount?: number;
}

export interface FusionPlanSnapshot {
  nodes: FusionPlanNode[];
  edges: FusionPlanEdge[];
  shaders: FusionPlanShader[];
  decisions: FusionPlanDecision[];
}

export interface FusionPlanNode {
  id: string;
  kind: string;
  label: string;
  shape: number[];
  residency?: string;
}

export interface FusionPlanEdge {
  from: string;
  to: string;
  reason?: string;
}

export interface FusionPlanShader {
  name: string;
  stage: string;
  workgroupSize?: [number, number, number];
  sourceHash?: string;
}

export interface FusionPlanDecision {
  nodeId: string;
  fused: boolean;
  reason?: string;
  thresholds?: string;
}

export interface SessionStats {
  totalExecutions: number;
  jitCompiled: number;
  interpreterFallback: number;
  totalExecutionTimeMs: number;
  averageExecutionTimeMs: number;
}

export interface GpuStatus {
  requested: boolean;
  active: boolean;
  error?: string;
  adapter?: GpuAdapterInfo;
}

export interface GpuAdapterInfo {
  deviceId: number;
  name: string;
  vendor: string;
  backend?: string;
  memoryBytes?: number;
  precision?: "single" | "double";
}

export interface MemoryUsage {
  bytes: number;
  pages: number;
}

export interface GeometryPreviewBudget {
  maxBytes?: number;
  maxTriangles?: number;
  maxVertices?: number;
  timeoutMs?: number;
  budgetPolicy?: "truncate" | "strict";
  tessellationProfile?: GeometryPreviewTessellationProfile;
  xray?: boolean;
  allowCreateFeaStudy?: boolean;
}

export interface GeometryPreviewTessellationProfile {
  profileId?: string;
  chordTolerance?: number;
  angleToleranceDeg?: number;
}

export interface GeometryStats {
  meshCount: number;
  totalVertices: number;
  totalElements: number;
  regionCount: number;
}

export interface GeometryBoundsSummary {
  min: [number, number, number];
  max: [number, number, number];
}

export interface GeometryMeshSummary {
  meshId: string;
  kind: string;
  vertexCount: number;
  elementCount: number;
  surfaceVertexCount?: number | null;
  surfaceTriangleCount?: number | null;
  bounds?: GeometryBoundsSummary | null;
}

export interface GeometryRegionMappingSummaryEntry {
  regionId: string;
  meshId: string;
  entityKind: string;
  rangeCount: number;
  entityCount: number;
  rangePreview: { start: number; count: number }[];
  truncated: boolean;
}

export interface GeometryRegionMappingSummary {
  mappingCount: number;
  mappedRegionCount: number;
  totalEntityCount: number;
  rangePreviewLimit: number;
  entries: GeometryRegionMappingSummaryEntry[];
}

export interface GeometryCadSummary {
  backend?: string | null;
  sourceFormat?: string | null;
  faceRegionCount: number;
  mappedFaceRegionCount: number;
  semanticRegionCount: number;
  mappedSemanticRegionCount: number;
  regionStatus: "not_cad" | "metadata_only" | "generic_face_topology" | "semantic_regions";
}

export interface GeometrySourceSummary {
  path: string;
  sha256: string;
  importer_version: string;
}

export interface GeometryAssemblyNodeSummary {
  node_id: string;
  label: string;
  children: GeometryAssemblyNodeSummary[];
}

export interface GeometryMaterialEvidenceSummary {
  source_key: string;
  normalized_key: string;
  value: string;
  confidence: "high" | "medium" | "low";
  unit_basis?: string | null;
  assumptions: string[];
}

export interface GeometrySourceGeometrySummary {
  kind: "mesh" | "cad";
  assembly?: GeometryAssemblyNodeSummary | null;
  material_evidence: GeometryMaterialEvidenceSummary[];
}

export interface GeometryTessellationProfileSummary {
  profile_id: string;
  chord_tolerance?: number | null;
  angle_tolerance_deg?: number | null;
  healing_mode: "safe" | "off" | "aggressive";
}

export interface GeometryAssetSummary {
  geometryId: string;
  revision: number;
  source: GeometrySourceSummary;
  sourceGeometry: GeometrySourceGeometrySummary;
  tessellationProfile: GeometryTessellationProfileSummary;
  units: string;
  meshes: GeometryMeshSummary[];
  mappingSummary: GeometryRegionMappingSummary;
  cad: GeometryCadSummary;
}

export interface GeometryInspectResult {
  path: string;
  format: string;
  byteCount: number;
  supported: boolean;
  stats?: GeometryStats | null;
  geometrySummary?: GeometryAssetSummary | null;
  regions?: unknown[];
  diagnostics?: unknown[];
  degradedReason?: string | null;
}

export interface GeometryPreviewResult extends GeometryInspectResult {
  sceneKind: "mesh" | "summary" | "unavailable";
  figureHandle: number | null;
  geometrySceneHandle: number | null;
  truncated: boolean;
  previewMessage?: string | null;
}

export type { LocalFeaWorkerContext, FeaWorkerWorkClass } from "./fea-run-context.js";
import type { LocalFeaWorkerContext } from "./fea-run-context.js";
export { createFeaRunAdmissionHook, FeaRunAdmissionError, feaRunAdmissionError } from "./fea-run-admission.js";
export type { FeaRunAdmission, FeaRunAdmissionHook, FeaRunWorkerAdmission, FeaRunWorkerOwner, FeaRunWorkerGrantRequest, FeaWorkerAssignment, FeaRunRetirementReceipt, FeaRunWorkerLeaseState, FeaRunRequest, FeaRunLease, FeaRunLeaseRenewal, FeaRunUsage, FeaRunOutcome, FeaRunDenied, FeaRunStopReason, FeaRunSettlement, FeaRunWorkerReport, CompletedSimulationUsage, SimulationOperationCount } from "./fea-run-admission.js";
export { simulationOperationCount, emptySimulationUsage, parseCompletedSimulationUsage } from "./simulation-usage.js";
import { createFeaRunAdmissionHook, FeaRunAdmissionError, type FeaRunAdmission, type FeaRunAdmissionHook, type FeaRunWorkerAdmission, type FeaRunLease, type FeaRunLeaseState } from "./fea-run-admission.js";

export interface RunMatSessionHandle {
  listProjectEntrypoints(projectPath: string, context?: ProjectEntrypointContext): Promise<ProjectEntrypointDescription[]>;
  runProjectEntrypoint(selection: NamedProjectEntrypoint, options?: RunProjectEntrypointOptions): Promise<ProjectEntrypointRunResult>;
  executeRequest(request: ExecuteRequest): Promise<ExecuteResult>;
  resetSession(): Promise<void>;
  stats(): Promise<SessionStats>;
  clearWorkspace(): void;
  exportWorkspaceState(options?: { includeVariables?: "off" | "auto" | "force" }): Promise<Uint8Array | null>;
  importWorkspaceState(state: Uint8Array): Promise<boolean>;
  workspaceSnapshot(): Promise<WorkspaceSnapshot>;
  inspectDataFile(path: string): Promise<WorkspaceEntry[]>;
  previewGeometry(path: string, budget?: GeometryPreviewBudget | null): Promise<GeometryPreviewResult>;
  disposeGeometryPreview?(
    figureHandle?: number | null,
    geometrySceneHandle?: number | null,
  ): Promise<void> | void;
  feaCapabilities(): Promise<FeaCapabilities>;
  planFeaSource(path: string, inputManifest?: InputManifest): Promise<FeaExecutionChoices>;
  runFeaSource(path: string, planId: string, inputManifest?: InputManifest): Promise<FeaRunResult>;
  setFeaRunAdmission(admission: FeaRunAdmission | FeaRunAdmissionHook | null): void;
  retryPendingFeaSettlement(runId: string): Promise<void>;
  renewFeaRunLease(runId: string, lease: FeaRunLease): void;
  checkFeaSource(path: string, inputManifest?: InputManifest): Promise<FeaSourceCheckResult>;
  checkFeaWorkspace(request: FeaWorkspaceCheckRequest): Promise<FeaWorkspaceCheckResult>;
  browseFeaSource(request: FeaBrowseRequest): Promise<FeaBrowseOutcome>;
  inspectFeaChanges(id: FeaChangeSetId): Promise<FeaChangeSetCheck>;
  reviewFeaChanges(request: FeaChangeSetReviewRequest): Promise<FeaChangeSetReview>;
  stageFeaChanges(request: FeaChangeSetRequest): Promise<FeaChangeSetCheck>;
  publishFeaChanges(id: FeaChangeSetId): Promise<FeaChangeSetPublication>;
  discardFeaChanges(id: FeaChangeSetId): Promise<void>;
  checkFeaWorkingSource(request: FeaWorkingCheckRequest): Promise<FeaSourceCheckResult>;
  feaField(runId: string, fieldId: string, options?: FeaFieldRequestOptions): Promise<FeaFieldResult>;
  materializeDataFileVariable(
    path: string,
    selector: WorkspaceMaterializeSelector,
    options?: MaterializeVariableOptions
  ): Promise<MaterializedVariable>;
  exportFigureScene?(handle: number): Promise<Uint8Array | null>;
  exportGeometryScene?(handle: number): Promise<Uint8Array | null>;
  importFigureScene?(scene: Uint8Array): Promise<number | null>;
  importGeometryScene?(scene: Uint8Array): Promise<number | null>;
  createFeaFieldScene?(runId: string, fieldId: string | null, shapeMode: "undeformed" | "true" | "exaggerated", exaggeration: number, showUndeformedOutline: boolean, view?: FeaFieldSceneView | null, replaceHandle?: number | null): Promise<{ handle: number; deformationScale: number; exact?: boolean }>;
  importFigureSceneFromPath?(path: string): Promise<number | null>;
  currentFigureHandle?(): Promise<number>;
  dispose(): void;
  telemetryConsent(): boolean;
  memoryUsage(): Promise<MemoryUsage>;
  telemetryClientId(): string | undefined;
  gpuStatus(): GpuStatus;
  cancelExecution(): void;
  setInputHandler(handler: InputHandler | null): Promise<void>;
  materializeVariable(
    selector: WorkspaceMaterializeSelector,
    options?: MaterializeVariableOptions
  ): Promise<MaterializedVariable>;
  setFusionPlanEnabled(enabled: boolean): void;
  setLanguageCompat(mode: LanguageCompatMode): void;
  setErrorNamespace(namespace: string): void;
  fusionPlanForSource?(source: string): Promise<FusionPlanSnapshot | null>;
  setFsProvider?(provider: RunMatFilesystemProvider): Promise<void>;
  packageCacheSnapshot(): Promise<PackageCacheSnapshot | null>;
  prepareCapturedProject: CapturedProjectResolver;
  installProjectHandoff(handoff: unknown): Promise<unknown>;
  clearProjectHandoff(): Promise<void>;
  projectRevision(): Promise<unknown | null>;
  prepareTests(
    snapshot: Record<string, unknown>,
    selector: Record<string, unknown>
  ): Promise<{
    snapshot: Record<string, unknown>;
    discovery: Record<string, unknown>;
    plan: Record<string, unknown>;
  }>;
  executeTestAttempt(input: {
    plan: Record<string, unknown>;
    snapshot: Record<string, unknown>;
    testId: string;
    attempt: number;
  }): Promise<{
    result: unknown;
    events: unknown[];
    coverage?: unknown[];
  }>;
}

interface NativeInitOptions {
  runtimeSettings?: HostRuntimeSettings;
  openTelemetryHeaders?: OpenTelemetryStartupHeaders;
  applicationSettings?: ApplicationSettingsMode;
  enableGpu?: boolean;
  enableJit?: boolean;
  verbose?: boolean;
  logLevel?: RunMatLogLevel;
  gpuBufferPoolMaxPerKey?: number;
  telemetryConsent?: boolean;
  telemetryId?: string;
  telemetryRunKind?: string;
  telemetryEmitter?: (envelope: unknown) => void;
  wgpuPowerPreference?: string;
  wgpuForceFallbackAdapter?: boolean;
  scatterTargetPoints?: number;
  surfaceVertexBudget?: number;
  emitFusionPlan?: boolean;
  callstackLimit?: number;
  errorNamespace?: string;
  languageCompat?: LanguageCompatMode;
  fsProvider?: RunMatFilesystemProvider;
  packageCacheProvider?: RunMatPackageCacheProvider;
  executionHost?: RunMatExecutionHost;
}

interface RunMatNativeSession extends ProjectEntrypointSessionExports {
  executeRequest(request: ExecuteRequest): Promise<ExecuteResult>;
  resetSession(): void;
  stats(): SessionStats;
  clearWorkspace(): void;
  exportWorkspaceState?: (includeVariables?: string) => Promise<Uint8Array | null>;
  importWorkspaceState?: (state: Uint8Array) => boolean;
  workspaceSnapshot?: () => WorkspaceSnapshot;
  inspectDataFile?: (path: string) => WorkspaceEntry[] | Promise<WorkspaceEntry[]>;
  previewGeometry?: (path: string, budget?: GeometryPreviewBudget | null) => GeometryPreviewResult | Promise<GeometryPreviewResult>;
  disposeGeometryPreview?: (
    figureHandle?: number | null,
    geometrySceneHandle?: number | null,
  ) => void | Promise<void>;
  feaCapabilities?: () => FeaCapabilities | Promise<FeaCapabilities>;
  planFeaSource?: (path: string, inputManifest?: InputManifest) => FeaExecutionChoices | Promise<FeaExecutionChoices>;
  runFeaSource?: (path: string, planId: string, inputManifest?: InputManifest) => FeaRunResult | Promise<FeaRunResult>;
  setFeaRunAdmission?: (admission: FeaRunAdmissionHook | null) => void;
  renewFeaRunLease?: (runId: string, lease: FeaRunLease) => void;
  delegateFeaWorker?: (context: LocalFeaWorkerContext) => FeaRunLeaseState;
  startFeaWorker?: (context: LocalFeaWorkerContext) => void;
  acquireFeaWorkerGrant?: (context: LocalFeaWorkerContext, requiredOps: import("./simulation-usage.js").SimulationOperationCount) => Promise<FeaRunLeaseState>;
  reportFeaWorker?: (context: LocalFeaWorkerContext, usage: import("./fea-run-admission.js").FeaRunUsage, finalReport: boolean) => import("./fea-run-admission.js").FeaRunUsage;
  retireFeaWorker?: (context: LocalFeaWorkerContext, receipt: import("./fea-run-admission.js").FeaRunRetirementReceipt) => void;
  discardUndispatchedFeaWorker?: (context: LocalFeaWorkerContext, receipt: import("./fea-run-admission.js").FeaRunRetirementReceipt) => void;
  inspectFeaChanges?: (id: FeaChangeSetId) => FeaChangeSetCheck | Promise<FeaChangeSetCheck>;
  reviewFeaChanges?: (request: FeaChangeSetReviewRequest) => FeaChangeSetReview | Promise<FeaChangeSetReview>;
  stageFeaChanges?: (request: FeaChangeSetRequest) => FeaChangeSetCheck | Promise<FeaChangeSetCheck>;
  publishFeaChanges?: (id: FeaChangeSetId) => FeaChangeSetPublication | Promise<FeaChangeSetPublication>;
  discardFeaChanges?: (id: FeaChangeSetId) => void | Promise<void>;
  checkFeaWorkingSource?: (request: FeaWorkingCheckRequest) => FeaSourceCheckResult | Promise<FeaSourceCheckResult>;
  browseFeaSource?: (request: FeaBrowseRequest) => FeaBrowseOutcome | Promise<FeaBrowseOutcome>;
  checkFeaWorkspace?: (request: FeaWorkspaceCheckRequest) => FeaWorkspaceCheckResult | Promise<FeaWorkspaceCheckResult>;
  checkFeaSource?: (path: string, inputManifest?: InputManifest) => FeaSourceCheckResult | Promise<FeaSourceCheckResult>;
  feaField?: (
    runId: string,
    fieldId: string,
    options?: FeaFieldRequestOptions,
  ) => FeaFieldResult | Promise<FeaFieldResult>;
  materializeDataFileVariable?: (
    path: string,
    array: string,
    options?: MaterializeVariableOptionsWire
  ) => MaterializedVariable | Promise<MaterializedVariable>;
  exportFigureScene?: (handle: number) => Uint8Array | null | Promise<Uint8Array | null>;
  exportGeometryScene?: (handle: number) => Uint8Array | null;
  importFigureScene?: (scene: Uint8Array) => number | null | Promise<number | null>;
  importGeometryScene?: (scene: Uint8Array) => number | null | Promise<number | null>;
  createFeaFieldScene?: (runId: string, fieldId: string | null, shapeMode: "undeformed" | "true" | "exaggerated", exaggeration: number, showUndeformedOutline: boolean, view?: FeaFieldSceneView | null, replaceHandle?: number | null) => { handle: number; deformationScale: number; exact?: boolean } | Promise<{ handle: number; deformationScale: number; exact?: boolean }>;
  importFigureSceneFromPath?: (path: string) => number | null | Promise<number | null>;
  currentFigureHandle?: () => number;
  dispose?: () => void;
  telemetryConsent(): boolean;
  memoryUsage?: () => MemoryUsage;
  telemetryClientId?: () => string | undefined;
  gpuStatus(): GpuStatus;
  cancelExecution?: () => void;
  setInputHandler?: (handler: InputHandler | null) => void;
  materializeVariable?: (
    selector: WorkspaceMaterializeSelectorWire,
    options?: MaterializeVariableOptionsWire
  ) => MaterializedVariable;
  setFusionPlanEnabled?: (enabled: boolean) => void;
  setLanguageCompat?: (mode: LanguageCompatMode) => void;
  setErrorNamespace?: (namespace: string) => void;
  fusionPlanForSource?: (source: string) => FusionPlanSnapshot | null;
  setFsProvider?: (provider: RunMatFilesystemProvider) => void;
  packageCacheSnapshot?: () => Promise<PackageCacheSnapshot | null>;
  prepareCapturedProject?: CapturedProjectResolver;
  installProjectHandoff?: (handoff: unknown) => unknown;
  clearProjectHandoff?: () => void;
  projectRevision?: () => unknown | null;
  prepareTests?: (
    snapshot: Record<string, unknown>,
    selector: Record<string, unknown>
  ) => {
    snapshot: Record<string, unknown>;
    discovery: Record<string, unknown>;
    plan: Record<string, unknown>;
  };
  executeTestAttempt?: (input: {
    plan: Record<string, unknown>;
    snapshot: Record<string, unknown>;
    testId: string;
    attempt: number;
  }) => Promise<{
    result: unknown;
    events: unknown[];
    coverage?: unknown[];
  }>;
}

type WorkspaceMaterializeSelectorWire =
  | string
  | {
      name?: string;
      token?: string;
      previewToken?: string;
    };

interface MaterializeVariableOptionsWire {
  limit?: number;
  slice?: {
    start: number[];
    shape: number[];
  };
}

export type PlotCameraProjectionState =
  | {
    kind: "perspective";
    fov: number;
    near: number;
    far: number;
  }
  | {
    kind: "orthographic";
    left: number;
    right: number;
    bottom: number;
    top: number;
    near: number;
    far: number;
  };

export interface PlotCameraState {
  position: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
  zoom: number;
  aspectRatio: number;
  projection: PlotCameraProjectionState;
}

export interface PlotSurfaceCameraState {
  activeAxes: number;
  axes: PlotCameraState[];
}

export interface FeaFieldSceneView {
  cameraPosition: [number, number, number];
  cameraTarget: [number, number, number];
  cameraUp: [number, number, number];
  fovYRadians: number;
  orthographicHeight?: number;
  aspectRatio: number;
  viewportHeightPx: number;
  triangleBudget: number;
  refineAbovePx: number;
}

export type GeometrySceneDisplayMode = "shaded" | "edges" | "wireframe";
export type GeometrySceneViewPreset =
  | "perspective"
  | "isometric"
  | "front"
  | "back"
  | "left"
  | "right"
  | "top"
  | "bottom";

export interface GeometrySceneRegionHighlightState {
  regionId: string;
  color?: [number, number, number, number] | null;
  role?: string | null;
  label?: string | null;
}

export interface GeometrySceneRegionAnnotationState {
  regionId: string;
  color?: [number, number, number, number] | null;
  kind: GeometrySceneAnnotationKind;
  label?: string | null;
  direction?: [number, number, number] | null;
  size?: number | null;
}

export type GeometrySceneAnnotationKind =
  | "label"
  | "force"
  | "moment"
  | "pressure"
  | "traction"
  | "body_force"
  | "heat_flux"
  | "current_density"
  | "normal_velocity"
  | "flow_velocity"
  | "mass_flow"
  | "wall_motion"
  | "fixed_support"
  | "symmetry_support"
  | "prescribed_motion"
  | "boundary_condition"
  | "thermal_condition"
  | "electrical_condition"
  | "magnetic_condition"
  | "acoustic_condition"
  | "fluid_condition"
  | "connector"
  | "interface"
  | "initial_condition";

export interface GeometryScenePresentationState {
  selectedRegionId?: string | null;
  selectedRegionIds?: string[] | null;
  selectedTopology?: GeometrySceneTopologySelectionState[];
  hoveredRegionId?: string | null;
  regionHighlights?: GeometrySceneRegionHighlightState[];
  regionAnnotations?: GeometrySceneRegionAnnotationState[];
  displayMode?: GeometrySceneDisplayMode;
  edgeOverlayEnabled?: boolean;
  showGrid?: boolean;
  hiddenOwnerNodeIds?: string[] | null;
  isolatedOwnerNodeIds?: string[] | null;
  /** Regions left out of the view; picks pass through them. */
  hiddenRegionIds?: string[];
  section?: GeometrySceneImageSectionState | null;
  viewPreset?: GeometrySceneViewPreset | null;
  /** Edge and wireframe line width in CSS px; above 1 px edges draw as screen-space strokes. */
  edgeWidthPx?: number | null;
  /** The 3D view's projection: with perspective, or parallel (orthographic). Unset keeps it. */
  projection?: "perspective" | "orthographic" | null;
}

export type GeometrySceneTopologySelectionState =
  | { kind: "face"; chunkId: string; triangleIndex: number }
  | { kind: "edge"; chunkId: string; vertexIndices: [number, number]; segments?: [number, number][] }
  | { kind: "point"; chunkId: string; vertexIndex: number };

export interface GeometryScenePickResult {
  meshId: string | null;
  chunkId: string;
  triangleIndex: number;
  regionId: string | null;
  regionLabel: string | null;
  regionTag: string | null;
  distance: number;
  position: [number, number, number];
  vertexIndices?: [number, number, number];
  barycentricCoordinates?: [number, number, number];
  topologySelection?: GeometrySceneTopologySelectionState;
  /** The owner nodes (bodies, assembly components) of what was hit, outermost first. */
  ownerNodeIds?: string[];
}


interface RunMatNativeModule extends OpenTelemetryNativeExports, RuntimeSettingsNativeExports, SourceCheckNativeExports, ProjectEntrypointModuleExports {
  default: (
    module?:
      | { module_or_path: WasmInitInput | Promise<WasmInitInput> }
      | WasmInitInput
      | Promise<WasmInitInput>
  ) => Promise<unknown>;
  initRunMat(options: NativeInitOptions): Promise<RunMatNativeSession>;
  stringElementWireSchemaVersion?: () => number;
  executeProgramArtifact?: (
    request: Record<string, unknown>
  ) => Promise<Record<string, unknown>>;
  executePreparedFeaStudy?: (json: string, memoryBytes: number, context?: LocalFeaWorkerContext, admission?: FeaRunWorkerAdmission) => Promise<string>;
  executeOwnedFeaStudyStep?: (json: string, memoryBytes: number, context?: LocalFeaWorkerContext, admission?: FeaRunWorkerAdmission) => Promise<string>;
  executeWorkerProgram?: (
    request: Record<string, unknown>
  ) => Promise<Record<string, unknown>>;
  createWorkerProgramInvocation?: (
    request: Record<string, unknown>
  ) => RunMatWorkerProgramInvocation;
  workerProgramInventory?: () => RunMatProgramWorkerInventory;
  buildGitSnapshot?: (
    repository: string,
    subdir: string,
    inventory: GitTreeInventoryWire
  ) => GitSnapshotWire;
  planGitAcquisition?: (request: GitAcquisitionPlanRequest) => GitAcquisitionPlan;
  validateGitSnapshot?: (snapshot: GitSnapshotWire) => GitSnapshotWire;
  projectTestLayout?: RunMatTestPreparationNative["projectTestLayout"];
  freezeTestSnapshot?: RunMatTestPreparationNative["freezeTestSnapshot"];
  runTests?: (input: unknown, backend: unknown) => Promise<unknown>;
  runTestsWithEvents?: (
    input: unknown,
    backend: unknown,
    observer: (event: Record<string, unknown>) => void
  ) => Promise<unknown>;
  resolveRunmatConfig?: (source: string, format: RunmatConfigFormat) => ResolvedRunmatConfig;
  patchRunmatConfig?: (
    source: string,
    format: RunmatConfigFormat,
    patch: RunmatConfigPatch
  ) => string;
  migrateLegacyRunmatConfig?: (
    source: string,
    format: RunmatConfigFormat
  ) => LegacyRunmatConfigMigration;
  migrateLegacyRunmatConfigInto?: (
    legacySource: string,
    destinationSource: string,
    format: RunmatConfigFormat
  ) => LegacyRunmatConfigMigration;
  decodePackageLock?: (input: string) => unknown;
  encodePackageLock?: (value: unknown) => string;
  handoffFromFrozenProject?: (project: unknown) => unknown;
  resolveProject?: BrowserProjectResolverConfig["native"]["resolveProject"];
  packageCacheStatus?: (
    provider: RunMatPackageCacheProvider
  ) => Promise<PackageCacheStatus>;
  packageCacheGc?: (
    provider: RunMatPackageCacheProvider,
    targetBytes: bigint,
    retainRecentMs: bigint
  ) => Promise<PackageCacheGcPlan>;
  packageCacheRenewLease?: BrowserProjectResolverConfig["native"]["packageCacheRenewLease"];
  packageCacheReleaseLease?: BrowserProjectResolverConfig["native"]["packageCacheReleaseLease"];
  plotRendererReady?: () => boolean;
  renderCurrentFigureScene?: (handle: number) => void;
  exportFigureScene?: (handle: number) => Uint8Array | null | Promise<Uint8Array | null>;
  exportGeometryScene?: (handle: number) => Uint8Array | null;
  importFigureScene?: (scene: Uint8Array) => number | null | Promise<number | null>;
  importGeometryScene?: (scene: Uint8Array) => number | null | Promise<number | null>;
  importFigureSceneFromPath?: (path: string) => number | null | Promise<number | null>;
  exportWorkspaceState?: (includeVariables?: string) => Promise<Uint8Array | null>;
  importWorkspaceState?: (state: Uint8Array) => boolean;
  workspaceSnapshot?: () => WorkspaceSnapshot;
  createPlotSurface?: (canvas: HTMLCanvasElement | OffscreenCanvas) => Promise<number>;
  destroyPlotSurface?: (surfaceId: number) => void;
  resizePlotSurface?: (
    surfaceId: number,
    width: number,
    height: number,
    pixelsPerPoint?: number
  ) => void;
  bindSurfaceToFigure?: (surfaceId: number, handle: number) => void;
  presentSurface?: (surfaceId: number) => void;
  presentFigureOnSurface?: (surfaceId: number, handle: number) => void;
  setSurfacePresence?: (surfaceId: number, frame: PlotPresenceFrame) => void;
  captureSurfacePointer?: (surfaceId: number, x: number, y: number) => unknown;
  presentGeometrySceneOnSurface?: (surfaceId: number, handle: number) => void;
  setGeometryScenePresentation?: (surfaceId: number, presentation: GeometryScenePresentationState) => void;
  pickGeometrySceneRegion?: (surfaceId: number, x: number, y: number, mode: string) => GeometryScenePickResult | null;
  handlePlotSurfaceEvent?: (surfaceId: number, event: PlotSurfaceEvent) => void;
  configurePlotSurfaceViewer?: (surfaceId: number, navigation: PlotNavigationOptions, antialiasing: PlotAntialiasing) => void;
  plotSurfaceAntialiasingModes?: (surfaceId: number) => PlotAntialiasing[];
  takePlotSurfaceHostActions?: (surfaceId: number) => unknown;
  fitPlotSurfaceExtents?: (surfaceId: number) => void;
  resetPlotSurfaceCamera?: (surfaceId: number) => void;
  getPlotSurfaceCameraState?: (surfaceId: number) => unknown;
  setPlotSurfaceCameraState?: (surfaceId: number, state: PlotSurfaceCameraState, expectedTarget?: NonNullable<PlotPresenceFrame["target"]>) => void;
  setPlotThemeConfig?: (theme: unknown) => void;
  onFigureEvent?: (callback: ((event: FigureEvent) => void) | null) => void;
  newFigureHandle?: () => number;
  selectFigure?: (handle: number) => void;
  currentFigureHandle?: () => number;
  setHoldMode?: (mode: HoldMode) => boolean;
  configureSubplot?: (rows: number, cols: number, index: number) => void;
  clearFigure?: (handle: number | null) => number;
  resetPlotState?: () => void;
  closeFigure?: (handle: number | null) => number;
  currentAxesInfo?: () => AxesInfo;
  renderFigureImage?: (handle: number | null, width: number, height: number) => Promise<Uint8Array>;
  renderFigureImageWithTextmark?: (
    handle: number | null,
    width: number,
    height: number,
    textmark?: string
  ) => Promise<Uint8Array>;
  renderFigureImageWithCameraState?: (
    handle: number | null,
    width: number,
    height: number,
    cameraState: FigureImageCameraState,
    textmark?: string
  ) => Promise<Uint8Array>;
  renderGeometrySceneImage?: (
    handle: number,
    width: number,
    height: number,
    view?: string | null
  ) => Promise<Uint8Array>;
  renderGeometrySceneObservationImage?: (
    handle: number, width: number, height: number, view: string | null,
    observation: Pick<GeometrySceneImageOptions, "visibility" | "section" | "selectedRegionIds">
  ) => Promise<Uint8Array>;
  measureGeometryScene?: (handle: number, kind: string, regionIds: string[]) => unknown;
  subscribeStdout?: (listener: (entry: StdoutEntry) => void) => number;
  unsubscribeStdout?: (id: number) => void;
  subscribeRuntimeLog?: (listener: (entry: RuntimeLogEntry) => void) => number;
  unsubscribeRuntimeLog?: (id: number) => void;
  setLogFilter?: (filter: string) => void;
  subscribeTraceEvents?: (listener: (entries: TraceEvent[]) => void) => number;
  unsubscribeTraceEvents?: (id: number) => void;
}

let loadPromise: Promise<RunMatNativeModule> | null = null;
let nativeModuleOverride: RunMatNativeModule | RunMatNativeSession | null = null;

function admitStringElementWire(native: RunMatNativeModule): RunMatNativeModule {
  const actual = typeof native.stringElementWireSchemaVersion === "function"
    ? native.stringElementWireSchemaVersion()
    : undefined;
  if (actual !== STRING_ELEMENT_WIRE_SCHEMA_VERSION) {
    throw new Error(`Incompatible RunMat string wire schema ${String(actual)}; expected ${STRING_ELEMENT_WIRE_SCHEMA_VERSION}. Rebuild the browser runtime assets with typed missing-string support.`);
  }
  return native;
}

async function loadNativeModule(wasmModule?: WasmInitInput): Promise<RunMatNativeModule> {
  installWebGpuCompatibilityShims();
  if (nativeModuleOverride) {
    if (isNativeSession(nativeModuleOverride)) {
      return {
        default: async () => {},
        stringElementWireSchemaVersion: () => STRING_ELEMENT_WIRE_SCHEMA_VERSION,
        initRunMat: async () => nativeModuleOverride
      } as RunMatNativeModule;
    }
    return admitStringElementWire(nativeModuleOverride);
  }
  if (!loadPromise) {
    loadPromise = (async () => {
      const moduleBaseUrl = import.meta.url;
      const wasmModuleUrl = `${moduleBaseUrl.slice(0, moduleBaseUrl.lastIndexOf("/") + 1)}pkg-web/runmat_wasm_web.js`;
      const native = (await import(
        /* webpackIgnore: true */ wasmModuleUrl
      )) as unknown as RunMatNativeModule;
      if (typeof native.default === "function") {
        await native.default(
          wasmModule === undefined ? undefined : { module_or_path: wasmModule }
        );
      }
      return admitStringElementWire(native);
    })();
  }
  return loadPromise;
}

export async function initRunMat(options: RunMatInitOptions = {}): Promise<RunMatSessionHandle> {
  const native = await loadNativeModule(options.wasmModule);
  const fsProvider = await resolveFsProvider(options.fsProvider);
  const packageCache = await resolvePackageCache(options.packageCache);
  const layeredSettings = options.runtimeSettings
    ? resolveNativeRuntimeSettings(native, options.runtimeSettings)
    : undefined;
  const supportsWebGpu = typeof navigator !== "undefined" && typeof (navigator as any).gpu !== "undefined";
  const hasExplicitEnableFlag = Object.prototype.hasOwnProperty.call(options, "enableGpu");
  const requestedGpu = (options.enableGpu ?? layeredSettings?.config.accelerate.enabled ?? true)
    && layeredSettings?.config.accelerate.provider !== "inprocess";
  const allowCpuFallback = layeredSettings?.config.accelerate.allow_inprocess_fallback ?? true;
  let effectiveEnableGpu: boolean;
  if (hasExplicitEnableFlag) {
    if (requestedGpu && !supportsWebGpu && allowCpuFallback) {
      console.warn(
        "[runmat] GPU acceleration was explicitly requested, but WebGPU APIs are unavailable in this context."
      );
      effectiveEnableGpu = !allowCpuFallback;
    } else {
      effectiveEnableGpu = requestedGpu;
    }
  } else {
    effectiveEnableGpu = requestedGpu && (supportsWebGpu || !allowCpuFallback);
    if (requestedGpu && !supportsWebGpu && allowCpuFallback) {
      console.warn("[runmat] WebGPU is not available in this environment; falling back to CPU execution.");
    }
  }
  const session = await native.initRunMat({
    runtimeSettings: options.runtimeSettings,
    openTelemetryHeaders: options.openTelemetryHeaders,
    applicationSettings: options.applicationSettings,
    enableGpu: effectiveEnableGpu,
    enableJit: options.enableJit,
    verbose: options.verbose,
    logLevel: options.logLevel,
    gpuBufferPoolMaxPerKey: options.gpuBufferPoolMaxPerKey,
    telemetryConsent: options.telemetryConsent,
    telemetryId: options.telemetryId,
    telemetryRunKind: options.telemetryRunKind,
    telemetryEmitter: options.telemetryEmitter,
    wgpuPowerPreference: options.wgpuPowerPreference,
    wgpuForceFallbackAdapter: options.wgpuForceFallbackAdapter,
    scatterTargetPoints: options.scatterTargetPoints,
    surfaceVertexBudget: options.surfaceVertexBudget,
    emitFusionPlan: options.emitFusionPlan ?? false,
    callstackLimit: options.callstackLimit,
    errorNamespace: options.errorNamespace,
    languageCompat: options.language?.compat,
    fsProvider,
    packageCacheProvider: packageCache.provider,
    executionHost: options.executionHost
  });
  // initRunMat resolves collector policy and installs logging before any renderer
  // can initialize the process-global subscriber with an unconfigured layer.
  if (options.plotCanvas) {
    if (typeof native.createPlotSurface === "function") {
      await native.createPlotSurface(options.plotCanvas);
    } else {
      const err = new Error(
        "The loaded runmat-wasm module does not support WebGPU plotting surfaces yet."
      ) as Error & { code?: string };
      err.code = "PlotCanvas";
      throw err;
    }
  }
  return new WebRunMatSession(session, packageCache.close);
}

/** Emit a correlated test span+log and await actual collector acceptance for both. */
export async function sendOpenTelemetryTest(wasmModule?: WasmInitInput): Promise<OpenTelemetryExportStats> {
  return invokeOpenTelemetry(await loadNativeModule(wasmModule), "sendOpenTelemetryTest");
}

export async function flushOpenTelemetry(wasmModule?: WasmInitInput): Promise<OpenTelemetryExportStats> {
  return invokeOpenTelemetry(await loadNativeModule(wasmModule), "flushOpenTelemetry");
}

/** Runtime-wide: flush before terminating the shared worker/runtime, not after each session. */
export async function shutdownOpenTelemetry(wasmModule?: WasmInitInput): Promise<OpenTelemetryExportStats> {
  return invokeOpenTelemetry(await loadNativeModule(wasmModule), "shutdownOpenTelemetry");
}

export async function openTelemetryStats(wasmModule?: WasmInitInput): Promise<OpenTelemetryExportStats> {
  return invokeOpenTelemetry(await loadNativeModule(wasmModule), "openTelemetryStats");
}

export async function executeProgramArtifact(
  request: Record<string, unknown>,
  wasmModule?: WasmInitInput
): Promise<Record<string, unknown>> {
  const native = await loadNativeModule(wasmModule);
  if (typeof native.executeProgramArtifact !== "function") {
    throw new Error("The loaded RunMat module does not support portable program execution.");
  }
  return native.executeProgramArtifact(request);
}

/** Resolve user/project fields and exact provenance without creating a session. */
export async function resolveRuntimeSettings(settings: HostRuntimeSettings, wasmModule?: WasmInitInput): Promise<ResolvedRuntimeSettings> {
  return resolveNativeRuntimeSettings(await loadNativeModule(wasmModule), settings);
}

/** Edit only the user layer. Persist the returned JSON using the host settings store. */
export async function editUserRuntimeSettings(source: string, edits: UserRuntimeSettingsEdit[], wasmModule?: WasmInitInput): Promise<string> {
  return editNativeUserRuntimeSettings(await loadNativeModule(wasmModule), source, edits);
}

/** Check scripts, functions or a project without executing user code. */
export async function checkSource(request: SourceCheckRequest, wasmModule?: WasmInitInput): Promise<SourceCheckReport> {
  return checkNativeSource(await loadNativeModule(wasmModule), request);
}

/** Freeze a project and compile under project-only cluster policy, never laptop user defaults. */
export async function prepareClusterProgram(sourcePath: string | NamedProjectEntrypoint, functionName?: string, wasmModule?: WasmInitInput): Promise<PreparedClusterProgram> {
  const native = await loadNativeModule(wasmModule);
  if (typeof sourcePath !== "string") {
    if (functionName !== undefined) throw new TypeError("A named entrypoint cannot override its declared function");
    return prepareNativeNamedClusterProgram(native, sourcePath);
  }
  if (!native.prepareClusterProgram) throw new Error("The loaded RunMat module lacks project-only cluster program preparation; rebuild the browser runtime assets.");
  return native.prepareClusterProgram(sourcePath, functionName);
}

/** Bind project policy to a selected Graph without reinterpreting compiled language/foreign semantics. */
export async function prepareClusterGraph(programJson: Uint8Array, project: NonNullable<HostRuntimeSettings["project"]>, wasmModule?: WasmInitInput): Promise<Uint8Array> {
  const native = await loadNativeModule(wasmModule);
  if (!native.prepareClusterGraph) throw new Error("The loaded RunMat module lacks project-only cluster Graph preparation; rebuild the browser runtime assets.");
  return native.prepareClusterGraph(programJson, project);
}

/** Execute only an owned worker's exact admitted root delegation. */
export async function executePreparedFeaStudy(json: string, memoryBytes: number, wasmModule?: WasmInitInput, context?: LocalFeaWorkerContext, admission?: FeaRunWorkerAdmission): Promise<string> {
  const native = await loadNativeModule(wasmModule);
  if (!native.executePreparedFeaStudy) throw new Error("Rebuild the browser numerical runtime assets with prepared FEA execution support.");
  if (!context || !admission) throw new FeaRunAdmissionError({ reason: "signInRequired" });
  return native.executePreparedFeaStudy(json, memoryBytes, context, admission);
}

/** Execute one original typed study step inside its supervised numerical worker. */
export async function executeOwnedFeaStudyStep(json: string, memoryBytes: number, wasmModule?: WasmInitInput, context?: LocalFeaWorkerContext, admission?: FeaRunWorkerAdmission): Promise<string> {
  const native = await loadNativeModule(wasmModule);
  if (!native.executeOwnedFeaStudyStep) throw new Error("Rebuild the browser numerical runtime assets with owned FEA study-step support.");
  if (!context || !admission) throw new FeaRunAdmissionError({ reason: "signInRequired" });
  return native.executeOwnedFeaStudyStep(json, memoryBytes, context, admission);
}

export async function executeWorkerProgram(
  request: Record<string, unknown>,
  wasmModule?: WasmInitInput
): Promise<Record<string, unknown>> {
  const native = await loadNativeModule(wasmModule);
  if (typeof native.executeWorkerProgram !== "function") {
    throw new Error("The loaded RunMat module does not support worker program execution.");
  }
  return native.executeWorkerProgram(request);
}

/** Construct one selected invocation inside an actual isolated CPU worker. */
export async function createWorkerProgramInvocation(
  request: Record<string, unknown>,
  wasmModule?: WasmInitInput
): Promise<RunMatWorkerProgramInvocation> {
  const native = await loadNativeModule(wasmModule);
  if (typeof native.createWorkerProgramInvocation !== "function") {
    throw new Error("The loaded RunMat module does not support controlled worker programs.");
  }
  return native.createWorkerProgramInvocation(request);
}

/** Describe the stock CPU worker factory from this exact loaded module. */
export async function workerProgramInventory(
  wasmModule?: WasmInitInput
): Promise<RunMatProgramWorkerInventory> {
  const native = await loadNativeModule(wasmModule);
  if (typeof native.workerProgramInventory !== "function") {
    throw new Error("The loaded RunMat module has no qualified worker factory inventory.");
  }
  return native.workerProgramInventory();
}

export async function fetchGitSnapshot(
  request: GitGatewayRequest,
  options: ServerGitGatewayOptions
): Promise<GitSnapshotWire> {
  const [inventory, native] = await Promise.all([
    fetchGitTreeInventoryWire(request, options),
    loadNativeModule()
  ]);
  requireNativeFunction(native, "buildGitSnapshot");
  return native.buildGitSnapshot(request.repository, request.subdir ?? ".", inventory);
}

export async function planGitAcquisition(
  request: GitAcquisitionPlanRequest
): Promise<GitAcquisitionPlan> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "planGitAcquisition");
  return native.planGitAcquisition(request);
}

export async function projectTestLayout(input: {
  manifestPath: string;
  manifestContent: string;
}): Promise<ProjectTestLayout> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "projectTestLayout");
  return native.projectTestLayout(input);
}

export async function freezeTestSnapshot(input: FreezeTestSnapshotInput): Promise<Record<string, unknown>> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "freezeTestSnapshot");
  return native.freezeTestSnapshot(input);
}

export async function runTests(input: unknown, backend: unknown): Promise<unknown> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "runTests");
  return native.runTests(input, backend);
}

export async function runTestsWithEvents(
  input: unknown,
  backend: unknown,
  observer: (event: Record<string, unknown>) => void
): Promise<unknown> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "runTestsWithEvents");
  return native.runTestsWithEvents(input, backend, observer);
}

export async function resolveRunmatConfig(
  source: string,
  format: RunmatConfigFormat
): Promise<ResolvedRunmatConfig> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "resolveRunmatConfig");
  return native.resolveRunmatConfig(source, format);
}

export async function patchRunmatConfig(
  source: string,
  format: RunmatConfigFormat,
  patch: RunmatConfigPatch
): Promise<string> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "patchRunmatConfig");
  return native.patchRunmatConfig(source, format, patch);
}

export async function migrateLegacyRunmatConfig(
  source: string,
  format: RunmatConfigFormat
): Promise<LegacyRunmatConfigMigration> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "migrateLegacyRunmatConfig");
  return native.migrateLegacyRunmatConfig(source, format);
}

export async function migrateLegacyRunmatConfigInto(
  legacySource: string,
  destinationSource: string,
  format: RunmatConfigFormat
): Promise<LegacyRunmatConfigMigration> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "migrateLegacyRunmatConfigInto");
  return native.migrateLegacyRunmatConfigInto(
    legacySource,
    destinationSource,
    format
  );
}

export async function createBrowserProjectResolver(
  config: Omit<BrowserProjectResolverConfig, "native">
): Promise<BrowserProjectResolver> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "resolveProject");
  requireNativeFunction(native, "packageCacheRenewLease");
  requireNativeFunction(native, "packageCacheReleaseLease");
  return new BrowserProjectResolver({
    ...config,
    native: {
      resolveProject: native.resolveProject,
      packageCacheRenewLease: native.packageCacheRenewLease,
      packageCacheReleaseLease: native.packageCacheReleaseLease
    }
  });
}

export async function createBrowserProjectSession(
  config: Omit<BrowserProjectSessionConfig, "native">
): Promise<BrowserProjectSession> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "resolveProject");
  requireNativeFunction(native, "packageCacheRenewLease");
  requireNativeFunction(native, "packageCacheReleaseLease");
  requireNativeFunction(native, "decodePackageLock");
  requireNativeFunction(native, "encodePackageLock");
  requireNativeFunction(native, "handoffFromFrozenProject");
  requireNativeFunction(native, "packageCacheStatus");
  requireNativeFunction(native, "packageCacheGc");
  return new BrowserProjectSession({
    ...config,
    native: {
      resolveProject: native.resolveProject,
      packageCacheRenewLease: native.packageCacheRenewLease,
      packageCacheReleaseLease: native.packageCacheReleaseLease,
      decodePackageLock: native.decodePackageLock,
      encodePackageLock: native.encodePackageLock,
      handoffFromFrozenProject: native.handoffFromFrozenProject,
      packageCacheStatus: native.packageCacheStatus,
      packageCacheGc: native.packageCacheGc
    }
  });
}

export async function decodePackageLock(input: string): Promise<unknown> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "decodePackageLock");
  return native.decodePackageLock(input);
}

export async function encodePackageLock(value: unknown): Promise<string> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "encodePackageLock");
  return native.encodePackageLock(value);
}

export async function handoffFromFrozenProject(project: unknown): Promise<unknown> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "handoffFromFrozenProject");
  return native.handoffFromFrozenProject(project);
}

export async function browserPackageCacheStatus(
  provider: RunMatPackageCacheProvider
): Promise<PackageCacheStatus> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "packageCacheStatus");
  return native.packageCacheStatus(provider);
}

export async function browserPackageCacheGc(
  provider: RunMatPackageCacheProvider,
  targetBytes: bigint,
  retainRecentMs = 0n
): Promise<PackageCacheGcPlan> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "packageCacheGc");
  return native.packageCacheGc(provider, targetBytes, retainRecentMs);
}

export async function plotRendererReady(): Promise<boolean> {
  const native = await loadNativeModule();
  if (typeof native.plotRendererReady !== "function") {
    return false;
  }
  return native.plotRendererReady();
}

export async function createPlotSurface(canvas: HTMLCanvasElement | OffscreenCanvas): Promise<number> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "createPlotSurface");
  return native.createPlotSurface(canvas);
}

export async function destroyPlotSurface(surfaceId: number): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "destroyPlotSurface");
  native.destroyPlotSurface(surfaceId);
}

export async function resizePlotSurface(
  surfaceId: number,
  widthPx: number,
  heightPx: number,
  pixelsPerPoint?: number
): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "resizePlotSurface");
  native.resizePlotSurface(surfaceId, widthPx, heightPx, pixelsPerPoint);
}

export async function bindSurfaceToFigure(surfaceId: number, handle: number): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "bindSurfaceToFigure");
  native.bindSurfaceToFigure(surfaceId, handle);
}

export async function presentSurface(surfaceId: number): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "presentSurface");
  native.presentSurface(surfaceId);
}

export async function setSurfacePresence(surfaceId: number, frame: PlotPresenceFrame): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "setSurfacePresence");
  native.setSurfacePresence(surfaceId, frame);
}

export async function captureSurfacePointer(surfaceId: number, x: number, y: number): Promise<PlotPointerPosition | null> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "captureSurfacePointer");
  return (normalizeWasmStructuredValue(native.captureSurfacePointer(surfaceId, x, y)) ?? null) as PlotPointerPosition | null;
}

export async function presentFigureOnSurface(surfaceId: number, handle: number): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "presentFigureOnSurface");
  native.presentFigureOnSurface(surfaceId, handle);
}

export async function presentGeometrySceneOnSurface(surfaceId: number, handle: number): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "presentGeometrySceneOnSurface");
  native.presentGeometrySceneOnSurface(surfaceId, handle);
}

export async function setGeometryScenePresentation(
  surfaceId: number,
  presentation: GeometryScenePresentationState,
): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "setGeometryScenePresentation");
  native.setGeometryScenePresentation(surfaceId, presentation);
}

export async function pickGeometrySceneRegion(
  surfaceId: number,
  x: number,
  y: number,
  mode = "face",
): Promise<GeometryScenePickResult | null> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "pickGeometrySceneRegion");
  return native.pickGeometrySceneRegion(surfaceId, x, y, mode) ?? null;
}

export async function renderCurrentFigureScene(handle: number): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "renderCurrentFigureScene");
  native.renderCurrentFigureScene(handle);
}

export async function handlePlotSurfaceEvent(surfaceId: number, event: PlotSurfaceEvent): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "handlePlotSurfaceEvent");
  native.handlePlotSurfaceEvent(surfaceId, event);
}

/** Unsupported AA modes are rejected without changing navigation settings. */
export async function configurePlotSurfaceViewer(surfaceId: number, navigation: PlotNavigationOptions, antialiasing: PlotAntialiasing): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "configurePlotSurfaceViewer");
  native.configurePlotSurfaceViewer(surfaceId, navigation, antialiasing);
}

export async function plotSurfaceAntialiasingModes(surfaceId: number): Promise<PlotAntialiasing[]> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "plotSurfaceAntialiasingModes");
  return native.plotSurfaceAntialiasingModes(surfaceId);
}

export async function takePlotSurfaceHostActions(surfaceId: number): Promise<PlotSurfaceHostAction[]> {
  const native = await loadNativeModule();
  if (typeof native.takePlotSurfaceHostActions !== "function") {
    return [];
  }
  const actions = normalizeWasmStructuredValue(native.takePlotSurfaceHostActions(surfaceId));
  return Array.isArray(actions) ? actions.filter(isPlotSurfaceHostAction) : [];
}

export async function fitPlotSurfaceExtents(surfaceId: number): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "fitPlotSurfaceExtents");
  native.fitPlotSurfaceExtents(surfaceId);
}

export async function resetPlotSurfaceCamera(surfaceId: number): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "resetPlotSurfaceCamera");
  native.resetPlotSurfaceCamera(surfaceId);
}

export async function getPlotSurfaceCameraState(surfaceId: number): Promise<PlotSurfaceCameraState> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "getPlotSurfaceCameraState");
  return native.getPlotSurfaceCameraState(surfaceId) as PlotSurfaceCameraState;
}

export async function setPlotSurfaceCameraState(
  surfaceId: number,
  state: PlotSurfaceCameraState,
  expectedTarget?: NonNullable<PlotPresenceFrame["target"]>
): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "setPlotSurfaceCameraState");
  native.setPlotSurfaceCameraState(surfaceId, state, expectedTarget);
}

export async function setPlotThemeConfig(theme: unknown): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "setPlotThemeConfig");
  native.setPlotThemeConfig(theme);
}

export async function onFigureEvent(listener: FigureEventListener | null): Promise<void> {
  const native = await loadNativeModule();
  if (typeof native.onFigureEvent !== "function") {
    throw new Error("The loaded runmat-wasm module does not expose figure events yet.");
  }
  native.onFigureEvent(listener ? (event: FigureEvent) => listener(event) : null);
}

export async function subscribeStdout(listener: StdoutListener): Promise<number> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "subscribeStdout");
  return native.subscribeStdout((entry: StdoutEntry) => listener(entry));
}

export async function unsubscribeStdout(id: number): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "unsubscribeStdout");
  native.unsubscribeStdout(id);
}

export async function subscribeRuntimeLog(listener: RuntimeLogListener): Promise<number> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "subscribeRuntimeLog");
  return native.subscribeRuntimeLog((entry: RuntimeLogEntry) => listener(entry));
}

export async function unsubscribeRuntimeLog(id: number): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "unsubscribeRuntimeLog");
  native.unsubscribeRuntimeLog(id);
}

export async function setLogFilter(filter: string): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "setLogFilter");
  native.setLogFilter(filter);
}

export async function subscribeTraceEvents(listener: TraceEventListener): Promise<number> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "subscribeTraceEvents");
  return native.subscribeTraceEvents((entries: TraceEvent[]) => listener(entries));
}

export async function unsubscribeTraceEvents(id: number): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "unsubscribeTraceEvents");
  native.unsubscribeTraceEvents(id);
}

export async function figure(handle?: number): Promise<number> {
  const native = await loadNativeModule();
  if (typeof handle === "number") {
    requireNativeFunction(native, "selectFigure");
    native.selectFigure(handle);
    return handle;
  }
  requireNativeFunction(native, "newFigureHandle");
  return native.newFigureHandle();
}

export async function newFigureHandle(): Promise<number> {
  return figure();
}

export async function currentFigureHandle(): Promise<number> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "currentFigureHandle");
  return native.currentFigureHandle();
}

export async function setHoldMode(mode: HoldMode = "toggle"): Promise<boolean> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "setHoldMode");
  return native.setHoldMode(mode);
}

export async function hold(mode: HoldMode = "toggle"): Promise<boolean> {
  return setHoldMode(mode);
}

export async function holdOn(): Promise<boolean> {
  return setHoldMode("on");
}

export async function holdOff(): Promise<boolean> {
  return setHoldMode("off");
}

export async function configureSubplot(rows: number, cols: number, index = 0): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "configureSubplot");
  try {
    native.configureSubplot(rows, cols, index);
  } catch (error) {
    throw coerceFigureError(error);
  }
}

export async function subplot(rows: number, cols: number, index = 0): Promise<void> {
  return configureSubplot(rows, cols, index);
}

export async function clearFigure(handle?: number): Promise<number> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "clearFigure");
  try {
    return native.clearFigure(handle ?? null);
  } catch (error) {
    throw coerceFigureError(error);
  }
}

export async function resetPlotState(): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "resetPlotState");
  try {
    native.resetPlotState();
  } catch (error) {
    throw coerceFigureError(error);
  }
}

export async function closeFigure(handle?: number): Promise<number> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "closeFigure");
  try {
    return native.closeFigure(handle ?? null);
  } catch (error) {
    throw coerceFigureError(error);
  }
}

export async function currentAxesInfo(): Promise<AxesInfo> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "currentAxesInfo");
  return native.currentAxesInfo();
}

export async function renderFigureImage(options: FigureImageOptions = {}): Promise<Uint8Array> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "renderFigureImage");
  const handle = typeof options.handle === "number" ? options.handle : null;
  const width = options.width ?? 0;
  const height = options.height ?? 0;
  const hasTextmark = typeof options.textmark === "string";
  const textmark = hasTextmark ? options.textmark : undefined;
  try {
    let bytes: Uint8Array | ArrayLike<number> | null | undefined;
    if (options.cameraState) {
        if (typeof native.renderFigureImageWithCameraState !== "function") {
          throw new Error(
            "The loaded runmat-wasm module does not support renderFigureImageWithCameraState yet."
          );
        }
        bytes = await native.renderFigureImageWithCameraState(
          handle,
          width,
          height,
          options.cameraState,
          textmark
        );
    } else if (hasTextmark && typeof native.renderFigureImageWithTextmark === "function") {
      bytes = await native.renderFigureImageWithTextmark(handle, width, height, textmark);
    } else {
      bytes = await native.renderFigureImage(handle, width, height);
    }
    if (bytes instanceof Uint8Array) {
      return bytes;
    }
    return new Uint8Array(bytes ?? []);
  } catch (error) {
    throw coerceFigureError(error);
  }
}

export async function renderGeometrySceneImage(
  options: GeometrySceneImageOptions
): Promise<Uint8Array> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "renderGeometrySceneImage");
  const handle = options.handle;
  const width = options.width ?? 0;
  const height = options.height ?? 0;
  const view = options.view ?? null;
  try {
    const observation = {visibility: options.visibility ?? null, section: options.section ?? null, selectedRegionIds: options.selectedRegionIds ?? []};
    const needsObservation = Boolean(observation.selectedRegionIds.length || observation.section
      || observation.visibility?.hidden_ids?.length || observation.visibility?.isolated_ids?.length
      || observation.visibility?.mode && observation.visibility.mode !== "all");
    if (needsObservation && typeof native.renderGeometrySceneObservationImage !== "function") {
      throw new Error("The loaded runtime predates geometry observation support. Reload the latest runtime.");
    }
    const bytes = native.renderGeometrySceneObservationImage
      ? await native.renderGeometrySceneObservationImage(handle, width, height, view, observation)
      : await native.renderGeometrySceneImage?.(handle, width, height, view);
    if (bytes instanceof Uint8Array) {
      return bytes;
    }
    return new Uint8Array(bytes ?? []);
  } catch (error) {
    throw coerceFigureError(error);
  }
}

export async function measureGeometryScene(options: {handle: number; kind: string; regionIds: string[]}): Promise<unknown> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "measureGeometryScene");
  try {
    return await native.measureGeometryScene!(options.handle, options.kind, options.regionIds);
  } catch (error) {
    throw coerceFigureError(error);
  }
}

export async function exportFigureScene(handle: number): Promise<Uint8Array | null> {
  const native = await loadNativeModule();
  if (typeof native.exportFigureScene !== "function") {
    return null;
  }
  try {
    const scene = await native.exportFigureScene(handle);
    if (!scene) {
      return null;
    }
    return scene instanceof Uint8Array ? scene : new Uint8Array(scene);
  } catch (error) {
    throw coerceFigureError(error);
  }
}

export async function exportGeometryScene(handle: number): Promise<Uint8Array | null> {
  const native = await loadNativeModule();
  if (typeof native.exportGeometryScene !== "function") {
    return null;
  }
  try {
    return native.exportGeometryScene(handle) ?? null;
  } catch {
    return null;
  }
}

export async function importFigureScene(scene: Uint8Array): Promise<number | null> {
  const native = await loadNativeModule();
  if (typeof native.importFigureScene !== "function") {
    return null;
  }
  try {
    const handle = await native.importFigureScene(scene);
    return typeof handle === "number" ? handle : null;
  } catch (error) {
    throw coerceFigureError(error);
  }
}

export async function importGeometryScene(scene: Uint8Array): Promise<number | null> {
  const native = await loadNativeModule();
  if (typeof native.importGeometryScene !== "function") {
    return null;
  }
  try {
    const handle = await native.importGeometryScene(scene);
    return typeof handle === "number" ? handle : null;
  } catch (error) {
    throw coerceFigureError(error);
  }
}

export async function importFigureSceneFromPath(path: string): Promise<number | null> {
  const native = await loadNativeModule();
  if (typeof native.importFigureSceneFromPath !== "function") {
    return null;
  }
  try {
    const handle = await native.importFigureSceneFromPath(path);
    return typeof handle === "number" ? handle : null;
  } catch (error) {
    throw coerceFigureError(error);
  }
}

export async function exportWorkspaceState(options: {
  includeVariables?: "off" | "auto" | "force";
} = {}): Promise<Uint8Array | null> {
  const native = await loadNativeModule();
  if (typeof native.exportWorkspaceState !== "function") {
    return null;
  }
  const mode = options.includeVariables ?? "auto";
  const state = await native.exportWorkspaceState(mode);
  return state ?? null;
}

export async function importWorkspaceState(state: Uint8Array): Promise<boolean> {
  const native = await loadNativeModule();
  if (typeof native.importWorkspaceState !== "function") {
    return false;
  }
  try {
    return native.importWorkspaceState(state) === true;
  } catch {
    return false;
  }
}

class WebRunMatSession implements RunMatSessionHandle {
  private disposed = false;
  private feaAdmissionHook: FeaRunAdmissionHook | null = null;

  constructor(
    private readonly native: RunMatNativeSession,
    private readonly closePackageCache: () => void = () => {}
  ) {}

  private ensureActive(): void {
    if (this.disposed) {
      throw new Error("RunMat session has been disposed");
    }
  }

  async executeRequest(request: ExecuteRequest): Promise<ExecuteResult> {
    this.ensureActive();
    try {
      return await this.native.executeRequest(request);
    } catch (error) {
      throw coerceRunMatError(error);
    }
  }

  async resetSession(): Promise<void> {
    this.ensureActive();
    this.native.resetSession();
  }

  async stats(): Promise<SessionStats> {
    this.ensureActive();
    return this.native.stats();
  }

  clearWorkspace(): void {
    this.ensureActive();
    this.native.clearWorkspace();
  }

  async exportWorkspaceState(options: { includeVariables?: "off" | "auto" | "force" } = {}): Promise<Uint8Array | null> {
    this.ensureActive();
    if (typeof this.native.exportWorkspaceState !== "function") {
      return null;
    }
    const mode = options.includeVariables ?? "auto";
    const state = await this.native.exportWorkspaceState(mode);
    return state ?? null;
  }

  async importWorkspaceState(state: Uint8Array): Promise<boolean> {
    this.ensureActive();
    if (typeof this.native.importWorkspaceState !== "function") {
      return false;
    }
    try {
      return this.native.importWorkspaceState(state) === true;
    } catch {
      return false;
    }
  }

  async workspaceSnapshot(): Promise<WorkspaceSnapshot> {
    this.ensureActive();
    requireNativeFunction(this.native, "workspaceSnapshot");
    return this.native.workspaceSnapshot();
  }

  async inspectDataFile(path: string): Promise<WorkspaceEntry[]> {
    this.ensureActive();
    if (typeof this.native.inspectDataFile !== "function") {
      throw new Error("The loaded runmat-wasm module does not expose inspectDataFile yet.");
    }
    const entries = await this.native.inspectDataFile(path);
    return Array.isArray(entries) ? entries : [];
  }

  async previewGeometry(path: string, budget?: GeometryPreviewBudget | null): Promise<GeometryPreviewResult> {
    this.ensureActive();
    if (typeof this.native.previewGeometry !== "function") {
      throw new Error("The loaded runmat-wasm module does not expose previewGeometry yet.");
    }
    return normalizeWasmStructuredValue(await this.native.previewGeometry(path, budget ?? null)) as GeometryPreviewResult;
  }

  async disposeGeometryPreview(
    figureHandle?: number | null,
    geometrySceneHandle?: number | null,
  ): Promise<void> {
    this.ensureActive();
    if (typeof this.native.disposeGeometryPreview !== "function") {
      if (typeof figureHandle === "number") {
        await closeFigure(figureHandle);
      }
      return;
    }
    await this.native.disposeGeometryPreview(
      typeof figureHandle === "number" ? figureHandle : null,
      typeof geometrySceneHandle === "number" ? geometrySceneHandle : null,
    );
  }

  async listProjectEntrypoints(projectPath: string, context?: ProjectEntrypointContext): Promise<ProjectEntrypointDescription[]> {
    this.ensureActive();
    return listNativeProjectEntrypoints(this.native, projectPath, context);
  }

  async runProjectEntrypoint(selection: NamedProjectEntrypoint, options?: RunProjectEntrypointOptions): Promise<ProjectEntrypointRunResult> {
    this.ensureActive();
    return runNativeProjectEntrypoint(this.native, selection, options);
  }

  async feaCapabilities(): Promise<FeaCapabilities> {
    this.ensureActive();
    if (typeof this.native.feaCapabilities !== "function") {
      throw new Error("The loaded runmat-wasm module does not expose feaCapabilities yet.");
    }
    return this.native.feaCapabilities();
  }

  async planFeaSource(path: string, inputManifest?: InputManifest): Promise<FeaExecutionChoices> {
    this.ensureActive();
    if (!this.native.planFeaSource) throw new Error("Reload the current runtime to prepare simulation execution.");
    return admitFeaExecutionChoices(await this.native.planFeaSource(path, inputManifest));
  }

  async runFeaSource(path: string, planId: string, inputManifest?: InputManifest): Promise<FeaRunResult> {
    this.ensureActive();
    if (!this.native.runFeaSource) throw new Error("Reload the current runtime to run this simulation.");
    return this.native.runFeaSource(path, planId, inputManifest);
  }

  setFeaRunAdmission(admission: FeaRunAdmission | FeaRunAdmissionHook | null): void {
    this.ensureActive();
    if (!this.native.setFeaRunAdmission) throw new Error("Reload the current runtime to install FEA lease admission.");
    if (admission === null) { this.native.setFeaRunAdmission(null); this.feaAdmissionHook = null; return; }
    const hook = "check" in admission ? admission
      : createFeaRunAdmissionHook(admission, () => this.cancelExecution(), (runId, lease) => this.renewFeaRunLease(runId, lease));
    const native = this.native;
    if (!native.delegateFeaWorker || !native.startFeaWorker || !native.acquireFeaWorkerGrant || !native.reportFeaWorker || !native.retireFeaWorker || !native.discardUndispatchedFeaWorker) throw new Error("Reload the current runtime to install owned FEA worker accounting.");
    hook.bindOwner({
      delegate: async context => native.delegateFeaWorker!(context),
      start: async context => { native.startFeaWorker!(context); },
      acquire: (context, requiredOps) => native.acquireFeaWorkerGrant!(context, requiredOps),
      report: async (context, usage, finalReport) => native.reportFeaWorker!(context, usage, finalReport),
      retire: async (context, receipt) => { native.retireFeaWorker!(context, receipt); },
      discardUndispatched: async (context, receipt) => { native.discardUndispatchedFeaWorker!(context, receipt); },
    });
    native.setFeaRunAdmission!(hook);
    this.feaAdmissionHook = hook;
  }

  async retryPendingFeaSettlement(runId: string): Promise<void> {
    this.ensureActive();
    if (!this.feaAdmissionHook) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
    await this.feaAdmissionHook.retryPendingSettlement(runId);
  }

  renewFeaRunLease(runId: string, lease: FeaRunLease): void {
    this.ensureActive();
    if (!this.native.renewFeaRunLease) throw new Error("Reload the current runtime to renew an FEA lease.");
    this.native.renewFeaRunLease(runId, lease);
  }

  async checkFeaSource(path: string, inputManifest?: InputManifest): Promise<FeaSourceCheckResult> {
    this.ensureActive();
    if (typeof this.native.checkFeaSource !== "function") {
      throw new Error("The loaded runmat-wasm module does not expose checkFeaSource yet.");
    }
    const result = await (inputManifest ? this.native.checkFeaSource(path, inputManifest) : this.native.checkFeaSource(path));
    return admitFeaSourceCheckResult(result);
  }

  async inspectFeaChanges(id: FeaChangeSetId): Promise<FeaChangeSetCheck> {
    this.ensureActive();
    if (!this.native.inspectFeaChanges) throw new Error("The runtime does not support study drafts.");
    const check = await this.native.inspectFeaChanges(id);
    return { ...check, roots: check.roots.map(admitFeaSourceCheckResult) };
  }

  async reviewFeaChanges(request: FeaChangeSetReviewRequest): Promise<FeaChangeSetReview> {
    this.ensureActive();
    if (!this.native.reviewFeaChanges) throw new Error("The runtime does not support checked source proposals.");
    return this.native.reviewFeaChanges(request);
  }
  async stageFeaChanges(request: FeaChangeSetRequest): Promise<FeaChangeSetCheck> {
    this.ensureActive();
    if (!this.native.stageFeaChanges) throw new Error("The runtime does not support checked source proposals.");
    return this.native.stageFeaChanges(request);
  }
  async publishFeaChanges(id: FeaChangeSetId): Promise<FeaChangeSetPublication> {
    this.ensureActive();
    if (!this.native.publishFeaChanges) throw new Error("The runtime does not support coherent source publication.");
    return this.native.publishFeaChanges(id);
  }
  async discardFeaChanges(id: FeaChangeSetId): Promise<void> {
    this.ensureActive();
    if (!this.native.discardFeaChanges) throw new Error("The runtime does not support discarding source proposals.");
    return this.native.discardFeaChanges(id);
  }

  async checkFeaWorkingSource(request: FeaWorkingCheckRequest): Promise<FeaSourceCheckResult> {
    this.ensureActive();
    if (!this.native.checkFeaWorkingSource) throw new Error("The runtime does not expose working FEA checks.");
    const result = await this.native.checkFeaWorkingSource(request);
    return admitFeaSourceCheckResult(result);
  }

  async browseFeaSource(request: FeaBrowseRequest): Promise<FeaBrowseOutcome> {
    this.ensureActive();
    if (typeof this.native.browseFeaSource !== "function") throw new Error("The runtime does not expose checked FEA browsing.");
    return this.native.browseFeaSource(request);
  }

  async checkFeaWorkspace(request: FeaWorkspaceCheckRequest): Promise<FeaWorkspaceCheckResult> {
    this.ensureActive();
    if (typeof this.native.checkFeaWorkspace !== "function") {
      throw new Error("The loaded runtime does not support candidate workspace checks.");
    }
    return this.native.checkFeaWorkspace(request);
  }

  async feaField(runId: string, fieldId: string, options?: FeaFieldRequestOptions): Promise<FeaFieldResult> {
    this.ensureActive();
    if (typeof this.native.feaField !== "function") {
      throw new Error("The loaded runmat-wasm module does not expose feaField yet.");
    }
    return this.native.feaField(runId, fieldId, options);
  }

  async materializeDataFileVariable(
    path: string,
    selector: WorkspaceMaterializeSelector,
    options?: MaterializeVariableOptions
  ): Promise<MaterializedVariable> {
    this.ensureActive();
    if (typeof this.native.materializeDataFileVariable !== "function") {
      throw new Error("The loaded runmat-wasm module does not expose materializeDataFileVariable yet.");
    }
    const normalized = normalizeMaterializeSelector(selector);
    const arrayName =
      typeof normalized === "string" ? normalized : normalized.name ?? normalized.previewToken;
    if (!arrayName) {
      throw new Error("materializeDataFileVariable selector requires name");
    }
    const wireOptions = normalizeMaterializeOptions(options);
    return this.native.materializeDataFileVariable(
      path,
      arrayName,
      wireOptions as MaterializeVariableOptionsWire | undefined
    );
  }

  async exportFigureScene(handle: number): Promise<Uint8Array | null> {
    this.ensureActive();
    if (typeof this.native.exportFigureScene !== "function") {
      return null;
    }
    try {
      return (await this.native.exportFigureScene(handle)) ?? null;
    } catch (error) {
      throw coerceFigureError(error);
    }
  }

  async exportGeometryScene(handle: number): Promise<Uint8Array | null> {
    this.ensureActive();
    if (typeof this.native.exportGeometryScene !== "function") {
      return null;
    }
    try {
      return this.native.exportGeometryScene(handle) ?? null;
    } catch {
      return null;
    }
  }

  async importFigureScene(scene: Uint8Array): Promise<number | null> {
    this.ensureActive();
    if (typeof this.native.importFigureScene !== "function") {
      return null;
    }
    try {
      const handle = await this.native.importFigureScene(scene);
      return typeof handle === "number" ? handle : null;
    } catch (error) {
      throw coerceFigureError(error);
    }
  }

  async importGeometryScene(scene: Uint8Array): Promise<number | null> {
    this.ensureActive();
    if (typeof this.native.importGeometryScene !== "function") {
      return null;
    }
    try {
      const handle = await this.native.importGeometryScene(scene);
      return typeof handle === "number" ? handle : null;
    } catch {
      return null;
    }
  }

  async createFeaFieldScene(runId: string, fieldId: string | null, shapeMode: "undeformed" | "true" | "exaggerated", exaggeration: number, showUndeformedOutline: boolean, view?: FeaFieldSceneView | null, replaceHandle?: number | null): Promise<{ handle: number; deformationScale: number; exact?: boolean }> {
    this.ensureActive();
    if (typeof this.native.createFeaFieldScene !== "function") {
      throw new Error("FEA field visualization is unavailable in this runtime");
    }
    return this.native.createFeaFieldScene(runId, fieldId, shapeMode, exaggeration, showUndeformedOutline, view, replaceHandle);
  }

  async importFigureSceneFromPath(path: string): Promise<number | null> {
    this.ensureActive();
    if (typeof this.native.importFigureSceneFromPath !== "function") {
      return null;
    }
    try {
      const handle = await this.native.importFigureSceneFromPath(path);
      return typeof handle === "number" ? handle : null;
    } catch (error) {
      throw coerceFigureError(error);
    }
  }

  async currentFigureHandle(): Promise<number> {
    this.ensureActive();
    requireNativeFunction(this.native, "currentFigureHandle");
    return this.native.currentFigureHandle();
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    if (typeof this.native.dispose === "function") {
      this.native.dispose();
    }
    this.closePackageCache();
    this.disposed = true;
  }

  telemetryConsent(): boolean {
    this.ensureActive();
    return this.native.telemetryConsent();
  }

  telemetryClientId(): string | undefined {
    this.ensureActive();
    if (typeof this.native.telemetryClientId !== "function") {
      return undefined;
    }
    return this.native.telemetryClientId() ?? undefined;
  }

  async memoryUsage(): Promise<MemoryUsage> {
    this.ensureActive();
    if (typeof this.native.memoryUsage !== "function") {
      return { bytes: 0, pages: 0 };
    }
    return this.native.memoryUsage();
  }

  gpuStatus(): GpuStatus {
    this.ensureActive();
    return this.native.gpuStatus();
  }

  cancelExecution(): void {
    if (this.disposed) {
      return;
    }
    if (typeof this.native.cancelExecution === "function") {
      this.native.cancelExecution();
    }
  }

  async setInputHandler(handler: InputHandler | null): Promise<void> {
    this.ensureActive();
    if (typeof this.native.setInputHandler !== "function") {
      throw new Error("The loaded runmat-wasm module does not expose setInputHandler yet.");
    }
    this.native.setInputHandler(handler ? (request: InputRequest) => handler(request) : null);
  }

  async materializeVariable(
    selector: WorkspaceMaterializeSelector,
    options?: MaterializeVariableOptions
  ): Promise<MaterializedVariable> {
    this.ensureActive();
    requireNativeFunction(this.native, "materializeVariable");
    const wireSelector = normalizeMaterializeSelector(selector);
    const wireOptions = normalizeMaterializeOptions(options);
    return this.native.materializeVariable(
      wireSelector,
      wireOptions as MaterializeVariableOptionsWire | undefined
    );
  }

  setFusionPlanEnabled(enabled: boolean): void {
    this.ensureActive();
    requireNativeFunction(this.native, "setFusionPlanEnabled");
    this.native.setFusionPlanEnabled(enabled);
  }

  setLanguageCompat(mode: LanguageCompatMode): void {
    this.ensureActive();
    requireNativeFunction(this.native, "setLanguageCompat");
    this.native.setLanguageCompat(mode);
  }

  setErrorNamespace(namespace: string): void {
    this.ensureActive();
    requireNativeFunction(this.native, "setErrorNamespace");
    this.native.setErrorNamespace(namespace);
  }

  async fusionPlanForSource(source: string): Promise<FusionPlanSnapshot | null> {
    this.ensureActive();
    if (typeof this.native.fusionPlanForSource !== "function") {
      throw new Error("The loaded runmat-wasm module does not expose fusionPlanForSource yet.");
    }
    try {
      return this.native.fusionPlanForSource(source) ?? null;
    } catch (error) {
      throw coerceRunMatError(error);
    }
  }

  async setFsProvider(provider: RunMatFilesystemProvider): Promise<void> {
    this.ensureActive();
    if (typeof this.native.setFsProvider !== "function") {
      throw new Error("The loaded runmat-wasm module does not expose setFsProvider yet.");
    }
    this.native.setFsProvider(provider);
  }

  async packageCacheSnapshot(): Promise<PackageCacheSnapshot | null> {
    this.ensureActive();
    if (typeof this.native.packageCacheSnapshot !== "function") {
      return null;
    }
    return (await this.native.packageCacheSnapshot()) ?? null;
  }

  prepareCapturedProject: CapturedProjectResolver = async (request, provider, filesystem) => {
    if (!this.native.prepareCapturedProject) {
      throw new Error("The loaded runtime does not support captured package resolution");
    }
    return this.native.prepareCapturedProject(request, provider, filesystem);
  };

  async installProjectHandoff(handoff: unknown): Promise<unknown> {
    this.ensureActive();
    if (typeof this.native.installProjectHandoff !== "function") {
      throw new Error("The loaded runmat-wasm module does not expose installProjectHandoff yet.");
    }
    return this.native.installProjectHandoff(handoff);
  }

  async clearProjectHandoff(): Promise<void> {
    this.ensureActive();
    if (typeof this.native.clearProjectHandoff !== "function") {
      throw new Error("The loaded runmat-wasm module does not expose clearProjectHandoff yet.");
    }
    this.native.clearProjectHandoff();
  }

  async projectRevision(): Promise<unknown | null> {
    this.ensureActive();
    if (typeof this.native.projectRevision !== "function") {
      return null;
    }
    return this.native.projectRevision() ?? null;
  }

  async prepareTests(
    snapshot: Record<string, unknown>,
    selector: Record<string, unknown>
  ): Promise<{
    snapshot: Record<string, unknown>;
    discovery: Record<string, unknown>;
    plan: Record<string, unknown>;
  }> {
    this.ensureActive();
    if (typeof this.native.prepareTests !== "function") {
      throw new Error("The loaded runmat-wasm module does not expose prepareTests yet.");
    }
    return this.native.prepareTests(snapshot, selector);
  }

  async executeTestAttempt(input: {
    plan: Record<string, unknown>;
    snapshot: Record<string, unknown>;
    testId: string;
    attempt: number;
  }): Promise<{
    result: unknown;
    events: unknown[];
    coverage?: unknown[];
  }> {
    this.ensureActive();
    if (typeof this.native.executeTestAttempt !== "function") {
      throw new Error("The loaded runmat-wasm module does not expose executeTestAttempt yet.");
    }
    return this.native.executeTestAttempt(input);
  }
}

function ensureFsProvider(provider: RunMatFilesystemProvider): void {
  const requiredMethods: (keyof RunMatFilesystemProvider)[] = [
    "readFile",
    "writeFile",
    "removeFile",
    "metadata",
    "readDir"
  ];
  for (const method of requiredMethods) {
    if (typeof provider[method] !== "function") {
      throw new Error(`fsProvider.${String(method)} must be a function`);
    }
  }
}

function admitFeaSourceCheckResult(result: FeaSourceCheckResult): FeaSourceCheckResult {
  if (!isFeaSourceCheckContract(result.contract) || !isFeaSemanticProgram(result.program)) {
    throw new Error("The runtime returned an incompatible study format. Update RunMat and try again.");
  }
  return result;
}

function requireNativeFunction<T, K extends keyof T>(
  native: T,
  method: K
): asserts native is T & Required<Pick<T, K>> {
  if (typeof native[method] !== "function") {
    throw new Error(`The loaded runmat-wasm module does not expose ${String(method)} yet.`);
  }
}

type FigureErrorPayload = {
    code?: string;
    message?: string;
    handle?: number;
    rows?: number;
  cols?: number;
  index?: number;
  details?: string;
};

type RunMatErrorPayload = {
  kind: RunMatErrorKind;
  message: string;
  identifier?: string;
  diagnostic: string;
  /** Canonical meshing diagnostic, retained without translating its domain contract. */
  meshingFailure?: unknown;
  span?: RunMatErrorSpan;
  sourceId?: number;
  sourceName?: string;
  sourcePath?: string;
  callstack?: string[];
  callstackElided?: number;
};

function isRunMatErrorPayload(value: unknown): value is RunMatErrorPayload {
  if (!value || typeof value !== "object") {
    return false;
  }
  const payload = value as RunMatErrorPayload;
  return (
    typeof payload.kind === "string" &&
    typeof payload.message === "string" &&
    typeof payload.diagnostic === "string"
  );
}

function coerceRunMatError(value: unknown): RunMatExecutionError {
  if (isRunMatErrorPayload(value)) {
    return new RunMatExecutionError({
      kind: value.kind,
      message: value.message,
      identifier: value.identifier,
      diagnostic: value.diagnostic,
      meshingFailure: value.meshingFailure,
      span: value.span,
      sourceId: value.sourceId,
      sourceName: value.sourceName,
      sourcePath: value.sourcePath,
      callstack: value.callstack ?? [],
      callstackElided: value.callstackElided
    });
  }
  if (value instanceof RunMatExecutionError) {
    return value;
  }
  if (value instanceof Error) {
    return new RunMatExecutionError({
      kind: "runtime",
      message: value.message,
      diagnostic: value.message,
      callstack: []
    });
  }
  const message = String(value ?? "RunMat execution failed");
  return new RunMatExecutionError({
    kind: "runtime",
    message,
    diagnostic: message,
    callstack: []
  });
}

function isFigureErrorPayload(value: unknown): value is FigureErrorPayload {
  return (
    typeof value === "object" &&
    value !== null &&
    "code" in value &&
    typeof (value as FigureErrorPayload).code === "string"
  );
}

function isPlotSurfaceHostAction(value: unknown): value is PlotSurfaceHostAction {
  return (
    typeof value === "object" &&
    value !== null &&
    (value as { kind?: unknown }).kind === "createFeaStudy"
  );
}

function normalizeWasmStructuredValue(value: unknown): unknown {
  if (value instanceof Map) {
    const normalized: Record<string, unknown> = {};
    for (const [key, entryValue] of value.entries()) {
      if (typeof key === "string" || typeof key === "number" || typeof key === "boolean") {
        normalized[String(key)] = normalizeWasmStructuredValue(entryValue);
      }
    }
    return normalized;
  }
  if (Array.isArray(value)) {
    return value.map((entry) => normalizeWasmStructuredValue(entry));
  }
  if (value instanceof ArrayBuffer || ArrayBuffer.isView(value)) {
    return value;
  }
  if (value && typeof value === "object") {
    const prototype: unknown = Object.getPrototypeOf(value);
    if (prototype === Object.prototype || prototype === null) {
      const normalized: Record<string, unknown> = {};
      for (const [key, entryValue] of Object.entries(value as Record<string, unknown>)) {
        normalized[key] = normalizeWasmStructuredValue(entryValue);
      }
      return normalized;
    }
  }
  return value;
}

function coerceFigureError(value: unknown): FigureBindingError {
  if (isFigureErrorPayload(value)) {
    const err = new Error(value.message ?? value.code ?? "Figure error") as FigureBindingError;
    err.code = (value.code as FigureBindingError["code"]) ?? "Unknown";
    if (typeof value.handle === "number") {
      err.handle = value.handle;
    }
    if (typeof value.rows === "number") {
      err.rows = value.rows;
    }
    if (typeof value.cols === "number") {
      err.cols = value.cols;
    }
    if (typeof value.index === "number") {
      err.index = value.index;
    }
    if (typeof value.details === "string") {
      err.details = value.details;
    }
    return err;
  }
  if (value instanceof Error) {
    const err = value as FigureBindingError;
    if (!err.code) {
      err.code = "Unknown";
    }
    return err;
  }
  const err = new Error(String(value)) as FigureBindingError;
  err.code = "Unknown";
  return err;
}

async function resolveFsProvider(
  provided?: RunMatFilesystemProvider
): Promise<RunMatFilesystemProvider | undefined> {
  if (provided) {
    ensureFsProvider(provided);
    return provided;
  }
  try {
    const autoProvider = await createDefaultFsProvider();
    ensureFsProvider(autoProvider);
    return autoProvider;
  } catch (error) {
    console.warn("[runmat] Unable to initialize default filesystem provider.", error);
    return undefined;
  }
}

async function resolvePackageCache(
  options?: RunMatInitOptions["packageCache"]
): Promise<{ provider?: RunMatPackageCacheProvider; close: () => void }> {
  if (options?.provider) {
    ensurePackageCacheProvider(options.provider);
    return { provider: options.provider, close: () => {} };
  }
  if (typeof indexedDB === "undefined") {
    return { close: () => {} };
  }
  try {
    const handle = await createIndexedDbPackageCache(options);
    ensurePackageCacheProvider(handle.provider);
    return { provider: handle.provider, close: handle.close };
  } catch (error) {
    console.warn("[runmat] Unable to initialize the package cache.", error);
    return { close: () => {} };
  }
}

function ensurePackageCacheProvider(provider: RunMatPackageCacheProvider): void {
  for (const method of ["snapshot", "initialize", "commit", "readObjectBytes"] as const) {
    if (typeof provider[method] !== "function") {
      throw new Error(`packageCache.provider.${method} must be a function`);
    }
  }
}

function normalizeMaterializeSelector(
  selector: WorkspaceMaterializeSelector
): WorkspaceMaterializeSelectorWire {
  if (typeof selector === "string") {
    const trimmed = selector.trim();
    if (!trimmed) {
      throw new Error("materializeVariable selector string must not be empty");
    }
    return trimmed;
  }
  if (!selector || typeof selector !== "object") {
    throw new Error("materializeVariable selector must be a string or object");
  }
  if (typeof selector.previewToken === "string" && selector.previewToken.trim()) {
    return { previewToken: selector.previewToken.trim() };
  }
  const payload: { name?: string } = {};
  if (typeof selector.name === "string" && selector.name.trim()) {
    payload.name = selector.name.trim();
  }
  if (!payload.name) {
    throw new Error("materializeVariable selector requires name or previewToken");
  }
  return payload;
}

function normalizeMaterializeOptions(
  options?: MaterializeVariableOptions
): MaterializeVariableOptionsWire | undefined {
  if (!options) {
    return undefined;
  }
  const payload: MaterializeVariableOptionsWire = {};
  if (typeof options.limit === "number" && Number.isFinite(options.limit)) {
    const limit = Math.floor(options.limit);
    if (limit > 0) {
      payload.limit = limit;
    }
  }
  if (options.slice) {
    const start = normalizeSliceVector(options.slice.start, false);
    const shape = normalizeSliceVector(options.slice.shape, true);
    if (start && shape) {
      payload.slice = { start, shape };
    }
  }
  return payload;
}

function normalizeSliceVector(values?: number[], requirePositive = false): number[] | undefined {
  if (!Array.isArray(values) || values.length === 0) {
    return undefined;
  }
  const normalized: number[] = [];
  for (const raw of values) {
    if (typeof raw !== "number" || !Number.isFinite(raw)) {
      return undefined;
    }
    const base = Math.floor(raw);
    if (requirePositive) {
      if (base <= 0) {
        return undefined;
      }
      normalized.push(base);
    } else {
      normalized.push(Math.max(0, base));
    }
  }
  return normalized;
}

function normalizeResumeInputValue(value: ResumeInputValue): ResumeInputPayload {
  if (value && typeof value === "object") {
    const payload = value as ResumeInputPayload;
    if (payload.error) {
      return { error: String(payload.error) };
    }
    if (payload.kind === "keyPress") {
      return { kind: "keyPress" };
    }
    if (payload.kind === "line") {
      const raw = payload.value ?? payload.line ?? "";
      return { kind: "line", value: String(raw ?? "") };
    }
  }
  // Scalars and nulls are treated as line inputs.
  if (value === null || value === undefined) {
    return { kind: "line", value: "" };
  }
  return { kind: "line", value: String(value) };
}

export const __internals = {
  coerceFigureError,
  normalizeResumeInputValue,
  workspaceHover: workspaceHoverInternals as unknown as Record<string, unknown>,
  setNativeModuleOverride(module: RunMatNativeModule | RunMatNativeSession | null): void {
    nativeModuleOverride = module;
    if (!module) {
      loadPromise = null;
    }
  }
};

function isNativeSession(value: unknown): value is RunMatNativeSession {
  return Boolean(value && typeof (value as RunMatNativeSession).executeRequest === "function");
}

export { readImmutableFilesystemResponse } from "./fs/immutable-input.js";

/** Execution options are display/admission contracts, never editable solver inputs. */
function admitFeaExecutionChoices(value: FeaExecutionChoices): FeaExecutionChoices {
  const modes = new Set<string>();
  if (!value || value.schemaVersion !== 1 || typeof value.planId !== "string" || !value.planId || !Array.isArray(value.revision) || value.revision.length !== 32 || value.revision.some(byte => !Number.isInteger(byte) || byte < 0 || byte > 255) || !Array.isArray(value.candidates) || value.candidates.length > 2) throw new Error("Invalid simulation execution options returned by the runtime.");
  for (const candidate of value.candidates) {
    if (!candidate || !["single_core", "multi_core"].includes(candidate.mode) || modes.has(candidate.mode) || (candidate.cores !== null && (!Number.isSafeInteger(candidate.cores) || candidate.cores < 1)) || !["ready", "desktop_required", "unavailable"].includes(candidate.readiness) || [candidate.memoryBytes, candidate.estimatedMillis].some(number => number !== null && (!Number.isSafeInteger(number) || number < 0))) throw new Error("Invalid simulation execution candidate returned by the runtime.");
    const evidence = candidate.durationEvidence;
    if (evidence != null && ([evidence.lowerMillis, evidence.upperMillis, evidence.sampleCount].some(number => !Number.isSafeInteger(number) || number < 1) || evidence.sampleCount < 3 || candidate.estimatedMillis == null || evidence.lowerMillis > candidate.estimatedMillis || evidence.upperMillis < candidate.estimatedMillis)) throw new Error("Invalid simulation duration evidence returned by the runtime.");
    modes.add(candidate.mode);
  }
  return value;
}
