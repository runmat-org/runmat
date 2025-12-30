export function createFusionPlanAdapter(options) {
    let enabled = false;
    let currentPlan = null;
    const listeners = new Set();
    const notify = () => {
        options.onPlanChange?.(currentPlan);
        for (const listener of listeners) {
            try {
                listener(currentPlan);
            }
            catch (error) {
                console.error("FusionPlanAdapter listener failed", error);
            }
        }
    };
    const setEnabledInternal = (next) => {
        if (enabled === next) {
            return;
        }
        enabled = next;
        try {
            options.session.setFusionPlanEnabled(next);
        }
        catch (error) {
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
        setEnabled(next) {
            setEnabledInternal(next);
            if (!next) {
                currentPlan = null;
                notify();
            }
        },
        handleExecutionResult(result) {
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
//# sourceMappingURL=fusion-plan.js.map