import { createFeaRunAdmissionHook, type FeaRunAdmission, type FeaRunLease, type FeaRunOutcome, type FeaRunRequest, type FeaRunStopReason, type FeaRunUsage } from "../fea-run-admission.js";
import type { LocalFeaWorkerContext } from "../fea-run-context.js";
import { simulationOperationCount } from "../simulation-usage.js";

/** Hermetic lease transport for qualification, using the production supervisor.
 * It does not prove Server billing. The caller must supply actual owned-work
 * retirement; no fixture callback certifies a live worker as retired.
 */
export function createHermeticFeaAdmission(options: {
  cancelExecution(): void;
  renewFeaRunLease(runId: string, lease: FeaRunLease): void;
  stopOwnedWork(request: FeaRunRequest, reason: FeaRunStopReason): void | Promise<void>;
}) {
  const starts: FeaRunRequest[] = [];
  const settlements: { leaseId: string; outcome: FeaRunOutcome; usage: FeaRunUsage }[] = [];
  const retirements: { request: FeaRunRequest; reason: FeaRunStopReason }[] = [];
  let issued: FeaRunLease | undefined;
  const host: FeaRunAdmission = {
    async start(request) {
      if (issued) throw new Error("The fixture admits one original root only.");
      starts.push(structuredClone(request));
      issued = { leaseId: `fealease-hermetic-${request.runId}`, state: "active", budgetOps: simulationOperationCount("120000"), grantedOps: simulationOperationCount("120000"), expiresAt: new Date(Date.now() + 120_000).toISOString(), heartbeatSeconds: 30 };
      return issued;
    },
    async heartbeat(leaseId) {
      if (!issued || issued.leaseId !== leaseId) throw new Error("Unknown original fixture lease.");
      return issued;
    },
    async requestGrant(leaseId, requiredOps) {
      if (!issued || issued.leaseId !== leaseId || BigInt(requiredOps) > BigInt(issued.budgetOps)) throw new Error("Unknown or over-budget fixture grant.");
      issued = { ...issued, grantedOps: simulationOperationCount(BigInt(requiredOps) > BigInt(issued.grantedOps) ? requiredOps : issued.grantedOps) };
      return issued;
    },
    async settle(leaseId, outcome, usage) {
      if (!issued || issued.leaseId !== leaseId || settlements.length) throw new Error("The original fixture lease must settle once.");
      settlements.push({ leaseId, outcome, usage: structuredClone(usage) });
    },
    async stopOwnedWork(request, reason, delegationId) {
      await options.stopOwnedWork(request, reason);
      retirements.push({ request: structuredClone(request), reason: structuredClone(reason) });
      return { runId: request.runId, delegationId, workerId: delegationId ? `fixture:${delegationId}` : null, retired: true as const };
    },
  };
  const hook = createFeaRunAdmissionHook(host, options.cancelExecution, options.renewFeaRunLease);
  // Contract fixture only. Production ownership lives in the parent WASM
  // ledger; these callbacks exercise the UI admission protocol in isolation.
  hook.bindOwner({
    async delegate(context) { return { lease: context.lease, rootStartedAtUnixMillis: context.rootStartedAtUnixMillis }; },
    async start() {},
    async acquire(context, requiredOps) { return { lease: { ...context.lease, grantedOps: requiredOps }, rootStartedAtUnixMillis: context.rootStartedAtUnixMillis }; },
    async report(_context, usage) { return usage; },
    async retire() {},
    async discardUndispatched() {},
  });
  return {
    hook, starts, settlements, retirements,
    async delegate(request: FeaRunRequest, payload: string): Promise<LocalFeaWorkerContext> {
      const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(payload)));
      const preparedStudyIdentity = `sha256:${Array.from(digest, byte => byte.toString(16).padStart(2, "0")).join("")}`;
      return { request, preparedStudyIdentity, workClass: "numerical", ...await hook.delegate(request, { preparedStudyIdentity, workClass: "numerical" }) };
    },
  };
}
