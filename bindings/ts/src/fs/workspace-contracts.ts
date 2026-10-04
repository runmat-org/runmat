//! Validation at the untrusted filesystem transport boundary.
import { z } from "zod";
import type { WorkspaceSnapshot, WorkspacePublicationOutcome } from "../generated/fea-workspace-contracts.js";
import { MAX_WORKSPACE_PUBLICATION_BYTES, MAX_WORKSPACE_PUBLICATION_EDITS } from "../generated/fea-workspace-contracts.js";

const byte = z.number().int().min(0).max(255);
export const workspaceRevisionSchema = z.tuple([
  byte, byte, byte, byte, byte, byte, byte, byte,
  byte, byte, byte, byte, byte, byte, byte, byte,
  byte, byte, byte, byte, byte, byte, byte, byte,
  byte, byte, byte, byte, byte, byte, byte, byte,
]);
const revision = workspaceRevisionSchema;
const id = z.tuple([
  byte, byte, byte, byte, byte, byte, byte, byte,
  byte, byte, byte, byte, byte, byte, byte, byte,
]);
const path = z.string().min(1).max(4096);
const receipt = z.object({ id, revision }).strict();
const sourceBytes = z.union([
  z.custom<Uint8Array>((value) => value instanceof Uint8Array && value.byteLength <= MAX_WORKSPACE_PUBLICATION_BYTES),
  z.array(byte).max(MAX_WORKSPACE_PUBLICATION_BYTES),
]);

export const workspaceSnapshotSchema = z.object({
  root: path,
  excluded_subtrees: z.array(path).max(1024),
  revision,
  files: z.array(z.object({ path, version: z.string().max(4096),
    bytes: sourceBytes, readonly: z.boolean() }).strict()).max(MAX_WORKSPACE_PUBLICATION_EDITS),
}).strict().superRefine((snapshot, ctx) => {
  let bytes = 0;
  const paths = new Set<string>();
  for (const file of snapshot.files) {
    bytes += file.bytes.length;
    if (paths.has(file.path)) ctx.addIssue({ code: z.ZodIssueCode.custom, message: "Duplicate workspace path" });
    paths.add(file.path);
  }
  if (bytes > MAX_WORKSPACE_PUBLICATION_BYTES) ctx.addIssue({ code: z.ZodIssueCode.custom, message: "Workspace exceeds byte budget" });
}) satisfies z.ZodType<WorkspaceSnapshot>;

export const workspacePublicationOutcomeSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("published"), receipt }).strict(),
  z.object({ type: z.literal("committed_pending_recovery"), receipt }).strict(),
  z.object({ type: z.literal("durability_indeterminate"), receipt }).strict(),
  z.object({ type: z.literal("conflict"), current: revision }).strict(),
  z.object({ type: z.literal("id_reused") }).strict(),
]) satisfies z.ZodType<WorkspacePublicationOutcome>;
