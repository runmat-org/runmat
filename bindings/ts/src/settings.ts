/** Canonical settings input. No ambient home directory or environment is read. */
export interface HostRuntimeSettings {
  userSettingsJson?: string | null;
  project?: { source: string; format: "toml" | "json" } | null;
  /** Cluster runs ignore the user layer even when supplied. */
  policy?: "local" | "clusterWorker";
}

export type RuntimeSettingSource =
  | { layer: "built_in" }
  | { layer: "user" | "project"; path: string | null }
  | { layer: "explicit"; path: string }
  | { layer: "environment"; variable: string }
  | { layer: "command_line"; argument: string };

export interface RuntimeSettingsConfig {
  runtime: { callstack_limit: number; error_namespace: string; verbose: boolean };
  jit: { enabled: boolean; threshold: number; optimization_level: "none" | "size" | "speed" | "aggressive" };
  gc: { preset: "low-latency" | "high-throughput" | "low-memory" | "debug" | null; young_size_mb: number | null; threads: number | null; collect_stats: boolean };
  language: { compat: "runmat" | "matlab" | "strict" };
  accelerate: {
    enabled: boolean; provider: "auto" | "wgpu" | "inprocess";
    allow_inprocess_fallback: boolean;
    wgpu_power_preference: "auto" | "high-performance" | "low-power";
    wgpu_force_fallback_adapter: boolean;
    auto_offload: { enabled: boolean; calibrate: boolean; log_level: "off" | "info" | "trace" };
  };
  foreign: {
    python: {
      executable: string | null; version: string | null; minimum_version: string;
      maximum_version: string | null; execution_mode: "in_process" | "out_of_process";
      exact_version: { major: number; minor: number; patch: number } | null;
      required_abi_tag: string | null; required_platform_tag: string | null;
      module_paths: string[]; artifact_identities: string[];
      installed_module_paths: { artifact_identities: string[]; installed_path_indices: number[] } | null;
    };
    java: { home: string | null; minimum_version: number; maximum_version: number | null; classpath: string[]; options: string[] };
    native: { isolation: "process" | "in_process"; timeout_ms: number | null; timeout: { secs: number; nanos: number } | null };
    mex: { unmanifested: "isolate" | "deny"; timeout_ms: number | null };
  };
  fea: { artifact_store: "filesystem" | "in_memory" | null; artifact_root: string | null; study_artifact_root: string | null; artifact_max_runs: number | null; artifact_max_runs_per_kind: number | null };
  logging: {
    level: "error" | "warn" | "info" | "debug" | "trace"; debug: boolean; file: string | null;
    otel: { enabled: boolean; endpoint: string; protocol: "http/protobuf" | "grpc"; signals: ("traces" | "logs" | "metrics")[]; headers_from_env: string | null; service_name: string; resource_attributes: Record<string, string> };
  };
  telemetry: { enabled: boolean; show_payloads: boolean; http_endpoint: string | null; udp_endpoint: string | null; queue_size: number; sync_mode: boolean; drain_mode: "none" | "all"; drain_timeout_ms: number; require_ingestion_key: boolean };
  plotting: {
    mode: "auto" | "gui" | "headless"; force_headless: boolean; backend: "auto" | "wgpu" | "static" | "web";
    scatter_target_points: number | null; surface_vertex_budget: number | null;
    gui: { width: number; height: number; vsync: boolean; maximized: boolean; maximum_pending_opens: number; maximum_windows: number; maximum_snapshot_bytes: number; maximum_pending_callbacks: number; maximum_callback_registrations: number } | null;
    export: { format: "png" | "svg" | "pdf" | "html"; dpi: number; output_dir: string | null; scene_budget_bytes: number } | null;
  };
}

export interface ResolvedRuntimeSettings {
  config: RuntimeSettingsConfig;
  /** Canonical runtime.* leaf paths; literal dots/backslashes in keys are escaped. */
  sources: Record<string, RuntimeSettingSource>;
}

export type SettingsJsonValue = null | boolean | number | string | SettingsJsonValue[] | { [key: string]: SettingsJsonValue };
export type UserRuntimeSettingsEdit =
  | { operation: "set"; path: string[]; value: SettingsJsonValue }
  | { operation: "reset"; path: string[] };

export interface RuntimeSettingsNativeExports {
  resolveRuntimeSettings?: (settings: HostRuntimeSettings) => ResolvedRuntimeSettings;
  editUserRuntimeSettings?: (source: string, edits: UserRuntimeSettingsEdit[]) => string;
  prepareClusterProgram?: (sourcePath: string, functionName?: string) => Promise<PreparedClusterProgram>;
  prepareClusterGraph?: (programJson: Uint8Array, project: NonNullable<HostRuntimeSettings["project"]>) => Uint8Array;
}

/** Encrypted/uploaded by the host using its existing cluster-run transport. */
export interface PreparedClusterProgram {
  bundleArchive: Uint8Array;
  descriptor: Uint8Array;
}

export function resolveNativeRuntimeSettings(native: RuntimeSettingsNativeExports, settings: HostRuntimeSettings): ResolvedRuntimeSettings {
  if (!native.resolveRuntimeSettings) throw new Error("The loaded RunMat module lacks layered runtime settings; rebuild the browser runtime assets.");
  return native.resolveRuntimeSettings(settings);
}

export function editNativeUserRuntimeSettings(native: RuntimeSettingsNativeExports, source: string, edits: UserRuntimeSettingsEdit[]): string {
  if (!native.editUserRuntimeSettings) throw new Error("The loaded RunMat module lacks user runtime settings editing; rebuild the browser runtime assets.");
  return native.editUserRuntimeSettings(source, edits);
}
