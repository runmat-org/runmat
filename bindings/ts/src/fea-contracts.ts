export * from "./generated/fea-study-document-contracts.js";

export const FEA_RUN_KIND = "fea-study" as const;
export const FEA_RUN_CELL_ID = "__fea__" as const;
export const FEA_RUN_MANIFEST_METADATA_SCHEMA_VERSION = 1 as const;

export type FeaRunArtifactRole =
  | "dataset"
  | "fields"
  | "diagnostics"
  | "artifact_manifest"
  | "workspace"
  | "stderr";

export function feaRunArtifactRefId(role: FeaRunArtifactRole): string {
  return `${FEA_RUN_CELL_ID}:${role}`;
}
