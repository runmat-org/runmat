// @ts-check

export const ENDPOINT_TOKENS = Object.freeze({
    HttpBaseUrl: "__RUNMAT_HTTP_BASE_URL__",
    LoopbackHost: "__RUNMAT_LOOPBACK_HOST__",
    LoopbackPort: "__RUNMAT_LOOPBACK_PORT__"
});

export function substituteLoopbackEndpoints(program, fixture, endpoint) {
    let result = program;
    const declared = new Set(fixture.endpoint_substitutions);
    for (const [identity, token] of Object.entries(ENDPOINT_TOKENS)) {
        const present = result.includes(token);
        if (present !== declared.has(identity)) {
            throw new Error(`${identity} endpoint token and fixture declaration do not agree`);
        }
        if (!present) continue;
        const replacement = endpointValue(identity, endpoint);
        result = result.split(token).join(replacement);
    }
    return result;
}

function endpointValue(identity, endpoint) {
    if (identity === "HttpBaseUrl") {
        if (endpoint.protocol !== "http") throw new Error("HttpBaseUrl requires an HTTP loopback fixture");
        return `http://${endpoint.host}:${endpoint.port}`;
    }
    if (identity === "LoopbackHost") return endpoint.host;
    if (identity === "LoopbackPort") return String(endpoint.port);
    throw new Error(`Unknown endpoint substitution: ${identity}`);
}
