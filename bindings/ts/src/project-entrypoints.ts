import type { ExecuteResult } from "./index.js";
import type { InputManifest } from "./generated/collaboration-contracts.js";
import type { FeaRunResult } from "./generated/fea-analysis-contracts.js";
import type { PreparedClusterProgram } from "./settings.js";

/** Resolve by name in this project, never relative to the JavaScript working directory. */
export interface NamedProjectEntrypoint {
  projectPath: string;
  name: string;
  inputManifest?: InputManifest;
}

export interface ProjectEntrypointContext {
  userSettingsJson?: string;
  inputManifest?: InputManifest;
}

export interface ProjectEntrypointFailure {
  code: string;
  message: string;
}

export interface ProjectEntrypointSupport {
  supported: boolean;
  reason: ProjectEntrypointFailure | null;
}

export interface ProjectEntrypointDescription {
  name: string;
  kind: "script" | "typedFunction" | "feaStudy";
  target:
    | { kind: "path"; path: string }
    | { kind: "moduleFunction"; module: string; function: string };
  /** Actual browser-host support, not capabilities of an unrelated Desktop. */
  local: ProjectEntrypointSupport;
  cluster: ProjectEntrypointSupport;
}

export interface RunProjectEntrypointOptions extends ProjectEntrypointContext {
  /** UTF-8 canonical JSON Vec<runmat_execution::value::ValuePayload>.
   * Bytes avoid rounding u64/f64-bit payloads through JavaScript numbers.
   * The Rust codec validates the entire aggregate under one shared limit.
   */
  argumentsJson?: Uint8Array;
  /** Typed function output arity; default 1. Scripts require 0. */
  requestedOutputs?: number;
  /** Existing immutable-input owner for scripts and checked FEA studies. */
  inputManifest?: InputManifest;
}

export type ProjectEntrypointRunResult =
  | { kind: "script"; result: ExecuteResult }
  | { kind: "feaStudy"; result: FeaRunResult }
  /** UTF-8 canonical JSON ordinal Vec<ValuePayload>, not a preview or cell. */
  | { kind: "typedFunction"; result: Uint8Array };

/** Implemented by the owning RunMatWasm session. No replacement runtime is made. */
export interface ProjectEntrypointSessionExports {
  listProjectEntrypoints?: (projectPath: string, context?: ProjectEntrypointContext) => Promise<ProjectEntrypointDescription[]>;
  runProjectEntrypoint?: (projectPath: string, name: string, options?: RunProjectEntrypointOptions) => Promise<ProjectEntrypointRunResult>;
}

export interface ProjectEntrypointModuleExports {
  prepareNamedClusterProgram?: (projectPath: string, name: string, inputs?: InputManifest) => Promise<PreparedClusterProgram>;
}

export function listNativeProjectEntrypoints(native: ProjectEntrypointSessionExports, projectPath: string, context?: ProjectEntrypointContext): Promise<ProjectEntrypointDescription[]> {
  if (!native.listProjectEntrypoints) throw new Error("Reload the current RunMat runtime to list named project entrypoints.");
  return native.listProjectEntrypoints(projectPath, context);
}

export function runNativeProjectEntrypoint(native: ProjectEntrypointSessionExports, selection: NamedProjectEntrypoint, options?: RunProjectEntrypointOptions): Promise<ProjectEntrypointRunResult> {
  if (!native.runProjectEntrypoint) throw new Error("Reload the current RunMat runtime to run named project entrypoints.");
  if (options?.requestedOutputs !== undefined && (!Number.isSafeInteger(options.requestedOutputs) || options.requestedOutputs < 0)) {
    throw new RangeError("requestedOutputs must be a non-negative safe integer");
  }
  if (selection.inputManifest !== undefined) {
    if (options?.inputManifest !== undefined) throw new TypeError("Supply one inputManifest, not both selection and run options");
    options = { ...options, inputManifest: selection.inputManifest };
  }
  return native.runProjectEntrypoint(selection.projectPath, selection.name, options);
}

export function prepareNativeNamedClusterProgram(native: ProjectEntrypointModuleExports, selection: NamedProjectEntrypoint): Promise<PreparedClusterProgram> {
  if (!native.prepareNamedClusterProgram) throw new Error("Reload the current RunMat runtime to prepare named project entrypoints for the cluster.");
  return native.prepareNamedClusterProgram(selection.projectPath, selection.name, selection.inputManifest);
}
