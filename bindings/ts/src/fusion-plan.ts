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

export function createFusionPlanAdapter(options: FusionPlanAdapterOptions): FusionPlanAdapter {
  let enabled = false;
  let currentPlan: FusionPlanSnapshot | null = null;
  const listeners = new Set<(plan: FusionPlanSnapshot | null) => void>();

  const notify = () => {
    options.onPlanChange?.(currentPlan);
    for (const listener of listeners) {
      try {
        listener(currentPlan);
      } catch (error) {
        console.error("FusionPlanAdapter listener failed", error);
      }
    }
  };

  const setEnabledInternal = (next: boolean) => {
    if (enabled === next) {
      return;
    }
    enabled = next;
    try {
      options.session.setFusionPlanEnabled(next);
    } catch (error) {
      console.warn("Failed to toggle fusion plan emission", error);
    }
  };

  return {
    get enabled() {
      return enabled;
    },
    get plan() {
      return currentPlan;
    },
    setEnabled(next: boolean) {
      setEnabledInternal(next);
      if (!next) {
        currentPlan = null;
        notify();
      }
    },
    handleExecutionResult(result: ExecuteResult) {
      currentPlan = result.fusionPlan ?? null;
      notify();
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    dispose() {
      if (enabled) {
        setEnabledInternal(false);
      }
      listeners.clear();
      currentPlan = null;
    },
  };
}
