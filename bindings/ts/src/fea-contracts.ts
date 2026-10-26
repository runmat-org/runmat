export * from "./physical.js";
export * from "./generated/fea-analysis-contracts.js";
export * from "./generated/fea-semantic-contracts.js";
export * from "./generated/fea-workspace-contracts.js";
export * from "./generated/fea-assessment-guards.js";

export const FEA_RUN_KIND = "fea-study" as const;
export const FEA_RUN_CELL_ID = "__fea__" as const;
export const FEA_RUN_MANIFEST_METADATA_SCHEMA_VERSION = 1 as const;

export type FeaRunArtifactRole =
  | "dataset"
  | "fields"
  | "diagnostics"
  | "artifact_manifest"
  | "result_scene"
  | "workspace"
  | "stderr";

export function feaRunArtifactRefId(role: FeaRunArtifactRole): string {
  return `${FEA_RUN_CELL_ID}:${role}`;
}
