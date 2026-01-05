import type { ExecuteResult, FusionPlanSnapshot, RunMatSessionHandle } from "./index.js";
export interface FusionPlanAdapterOptions {
    session: RunMatSessionHandle;
    onPlanChange?: (plan: FusionPlanSnapshot | null) => void;
}
export interface FusionPlanAdapter {
    readonly enabled: boolean;
    readonly plan: FusionPlanSnapshot | null;
    setEnabled(enabled: boolean): void;
    handleExecutionResult(result: ExecuteResult): void;
    subscribe(listener: (plan: FusionPlanSnapshot | null) => void): () => void;
    dispose(): void;
}
export declare function createFusionPlanAdapter(options: FusionPlanAdapterOptions): FusionPlanAdapter;
//# sourceMappingURL=fusion-plan.d.ts.map