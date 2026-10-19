// Self-contained functions are also embedded in the browser worker.
export async function withFigureSurface(module, session, required, Canvas, run) {
    let surfaceId = null;
    try {
        try {
            if (required) {
                for (const name of ["createPlotSurface", "resizePlotSurface", "presentFigureOnSurface", "destroyPlotSurface"]) {
                    if (typeof module[name] !== "function") throw new Error(`Figure surface API ${name} is unavailable`);
                }
                if (typeof Canvas !== "function") throw new Error("Figure OffscreenCanvas is unavailable");
            }
            if (typeof Canvas === "function" && typeof module.createPlotSurface === "function") {
                surfaceId = await module.createPlotSurface(new Canvas(16, 16));
                if (!Number.isSafeInteger(surfaceId) || surfaceId < 0 || surfaceId > 0xffffffff) {
                    throw new Error("Figure surface creation returned an invalid handle");
                }
                if (typeof module.resizePlotSurface === "function") {
                    await module.resizePlotSurface(surfaceId, 16, 16, 1);
                }
            }
        } catch (error) {
            if (required) throw error;
            // Optional legacy setup does not establish presentation evidence.
            const failedSurface = surfaceId;
            surfaceId = null;
            if (Number.isSafeInteger(failedSurface) && failedSurface >= 0 && failedSurface <= 0xffffffff
                && typeof module.destroyPlotSurface === "function") {
                await module.destroyPlotSurface(failedSurface);
            }
        }
        return await run(surfaceId);
    } finally {
        try {
            if (Number.isSafeInteger(surfaceId) && surfaceId >= 0 && surfaceId <= 0xffffffff
                && typeof module.destroyPlotSurface === "function") {
                await module.destroyPlotSurface(surfaceId);
            }
        } finally {
            if (typeof session.dispose === "function") await session.dispose();
        }
    }
}

export function createFigureObserver() {
    const handles = new Set();
    let failure = "";
    return {
        observe(event) {
            if (!event || !Number.isSafeInteger(event.handle) || event.handle < 0
                || !["created", "updated", "cleared", "closed"].includes(event.kind)) {
                failure = "Invalid figure lifecycle event";
                return;
            }
            if (event.kind === "closed") handles.delete(event.handle);
            else handles.add(event.handle);
        },
        snapshot() {
            if (failure) throw new Error(failure);
            return [...handles].sort((a, b) => a - b);
        }
    };
}

export async function captureFigureEvidence(module, handles, surfaceId) {
    if (typeof module.renderFigureImage !== "function") {
        throw new Error("Figure image rendering is unavailable");
    }
    let image = new Uint8Array();
    let renderedFigureCount = 0;
    for (const handle of handles) {
        if (Number.isFinite(surfaceId)) {
            if (typeof module.presentFigureOnSurface !== "function") {
                throw new Error("Figure surface presentation is unavailable");
            }
            await module.presentFigureOnSurface(surfaceId, handle);
        }
        const payload = await module.renderFigureImage(handle, 1280, 720);
        image = payload instanceof Uint8Array ? payload : new Uint8Array(payload || []);
        const signature = [137, 80, 78, 71, 13, 10, 26, 10];
        if (image.length <= signature.length || !signature.every((byte, i) => image[i] === byte)) {
            throw new Error(`Figure ${handle} did not produce a PNG image`);
        }
        renderedFigureCount += 1;
    }
    return { image, renderedFigureCount };
}
