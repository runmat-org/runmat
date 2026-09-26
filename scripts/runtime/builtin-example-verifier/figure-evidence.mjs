// Self-contained functions are also embedded in the browser worker.
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
