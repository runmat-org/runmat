import type { InputManifest } from './generated/collaboration-contracts.js';

export type SourceCheckTarget =
  | { kind: 'script'; path: string }
  | { kind: 'function'; path: string; function: string }
  | { kind: 'project'; path: string };

export interface SourceCheckRequest {
  target: SourceCheckTarget;
  /** Host-owned user layer; the actual selected manifest supplies project settings. */
  userSettingsJson?: string | null;
  policy?: 'local' | 'clusterWorker';
  denyWarnings?: boolean;
  /** Relative search roots use the original captured filesystem CWD, as in CLI. */
  searchPaths?: string[];
  inputManifest?: InputManifest | null;
}

export type SourceCheckOutcome = 'clean' | 'warnings' | 'failed';
export interface SourceCheckLocation {
  source_id: number;
  path: string | null;
  /** UTF-8 byte offsets, not UTF-16 offsets. */
  start: number;
  end: number;
  /** One-based Unicode-scalar coordinates, preserving CLI's human contract. */
  line: number | null;
  column: number | null;
  end_line: number | null;
  end_column: number | null;
  label: string | null;
}
export interface SourceCheckDiagnostic {
  code: string;
  severity: 'error' | 'warning' | 'information' | 'help';
  stage: 'syntax' | 'resolution' | 'semantic' | 'mir_lowering' | 'compile' | 'analysis' | 'unknown';
  message: string;
  category: string | null;
  primary: SourceCheckLocation;
  location: SourceCheckLocation;
  secondary: SourceCheckLocation[];
  related: SourceCheckLocation[];
  notes: string[];
  help: string | null;
  suggestions: { span: SourceCheckLocation; replacement: string; message: string }[];
}
export interface SourceCheckSummary { errors: number; warnings: number; warnings_denied: boolean }
export type SourceAnalysisCompleteness = 'complete' | 'partial' | 'runtime_dependent' | 'unavailable' | 'not_applicable';
export interface SourceCheckDocument {
  path: string;
  function: string | null;
  outcome: SourceCheckOutcome;
  diagnostics: SourceCheckDiagnostic[];
  /** Diagnostic/resolution source IDs are local to this document's source table. */
  sources: { source_id: number; path: string }[];
  analysis: Record<'syntax' | 'name_resolution' | 'definite_assignment' | 'effects' | 'types' | 'shapes' | 'async_safety', SourceAnalysisCompleteness>;
  resolution: { name: string; span: { start: number; end: number }; source_id?: number;
    state: 'resolved' | 'unresolved' | 'runtime_dependent'; reason: string;
    definition?: { name: string; qualified_name: string; source_path: string; package_name: string;
      dependency_alias?: string; package_instance?: string; source_id?: { package_instance: string; relative_path: string; content_digest: string }; is_private: boolean } }[];
  analysis_incomplete_sources: string[];
  call_context_budget_exhausted: boolean;
  call_context_precision_withheld: boolean;
  summary: SourceCheckSummary;
}
export interface SourceCheckReport {
  schema_version: 1;
  document_kind: 'script' | 'function' | 'project';
  path: string;
  outcome: SourceCheckOutcome;
  language_compatibility: 'runmat' | 'matlab' | 'strict';
  project_revision: { graph_digest: string; source_revision: string } | null;
  /** One reached-closure report per indexed root; shared diagnostics can recur. */
  documents: SourceCheckDocument[];
  summary: SourceCheckSummary;
}

export interface SourceCheckNativeExports {
  checkSource?: (request: SourceCheckRequest) => Promise<SourceCheckReport>;
}
/** Native WASM performs all parsing, resolution and analysis. Errors are not swallowed. */
export async function checkNativeSource(native: SourceCheckNativeExports, request: SourceCheckRequest): Promise<SourceCheckReport> {
  if (typeof native.checkSource !== 'function') {
    throw new Error('The loaded RunMat module lacks ordinary source checks; rebuild the browser runtime assets.');
  }
  return await native.checkSource(request);
}
