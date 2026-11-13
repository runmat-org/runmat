import assert from "node:assert/strict";
import "fake-indexeddb/auto";
import { readFile } from "node:fs/promises";
import { initRunMat, createInMemoryFsProvider, createIndexedDbFsHandle, createIndexedDbPackageCache, createBrowserProjectSession } from "../dist/index.js";

// Actual WASM checking with a frozen browser project, including imported callable facts.
// Browser checking is qualified here; it does not imply browser solve placement.
const wasmModule = await readFile(new URL("../dist/pkg-web/runmat_wasm_web_bg.wasm", import.meta.url));
const encoder = new TextEncoder();
const source = `import function verify from "./verify.m"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region body : solid { select result; expect exactly_one }
}
mesh part on geometry.part {
    tetrahedra { order = linear; target_size = 0.1 m; maximum_deviation = 0.01 m }
}
model structure {}
study response {
    geometry = geometry.part
    mesh = mesh.part
    model = model.structure
    analysis solve { structural static {} }
    output peak = max(magnitude(analysis.solve.field.node.displacement), over = geometry.part.region.body)
    assessment deflection { value = output.peak; maximum = 1 mm; evidence = authored_samples }
    assessment custom {
        value = output.peak
        function = function.verify
        parameters = [1 mm]
        evidence = authored_samples
    }
}
`;
const functionSource = "function result = verify(value, limit)\narguments\n value (1,1) double\n limit (1,1) double\nend\nresult = struct('evaluated', true, 'accepted', value <= limit);\nend\n";
async function qualify(filesystem, providerName) {
    await filesystem.writeFile("/runmat.toml", encoder.encode('[package]\nname = "browser-assessment-qualification"\n[sources]\nroots = ["."]\n'));
    await filesystem.writeFile("/study.fea", encoder.encode(source));
    await filesystem.writeFile("/verify.m", encoder.encode(functionSource));
    const session = await initRunMat({ applicationSettings: "isolatedMemory", enableGpu: false, enableJit: false, telemetryConsent: false, fsProvider: filesystem, wasmModule });
    const cache = await createIndexedDbPackageCache({ dbName: `assessment-functions-${crypto.randomUUID()}` });
    const project = await createBrowserProjectSession({ session, filesystem, packageCache: cache.provider, gitGateway: { baseUrl: "http://127.0.0.1:9" } });
    try {
        await project.resolve({ manifestPath: "/runmat.toml", lockPath: null, options: {
            target: "wasm32-unknown-unknown", default_server_origin: "http://127.0.0.1:9",
            default_registry_index: "http://127.0.0.1:9", groups: ["runtime"], root_features: [],
            host_capabilities: ["browser-filesystem", "worker"], source_intent: "execute", source_policy: { offline: true },
        } });
        const checked = await session.checkFeaWorkingSource({ root: "study.fea", overlays: [] });
        assert.equal(checked.status, "valid", JSON.stringify(checked));
        assert.equal(checked.program.studies[0].assessments.length, 2, "The loaded browser runtime must retain both authored criteria");
        const wrongDimension = await session.checkFeaWorkingSource({ root: "study.fea", overlays: [
            { path: "study.fea", bytes: Array.from(encoder.encode(source.replace("maximum = 1 mm", "maximum = 1 s"))) },
        ] });
        assert.equal(wrongDimension.status, "invalid", JSON.stringify(wrongDimension));
        assert.equal(new TextDecoder().decode(await filesystem.readFile("/study.fea")), source);
        const impure = await session.checkFeaWorkingSource({ root: "study.fea", overlays: [
            { path: "verify.m", bytes: Array.from(encoder.encode(functionSource.replace("result = struct", "disp(value);\nresult = struct"))) },
        ] });
        assert.equal(impure.status, "invalid", JSON.stringify(impure));
        assert.equal(new TextDecoder().decode(await filesystem.readFile("/verify.m")), functionSource);
        const revisedSource = source.replace("maximum = 1 mm", "maximum = 0.5 mm");
        const revisedFunction = functionSource.replace("value <= limit", "value < limit");
        const proposal = await session.stageFeaChanges({ base: null, changes: [
            { type: "write", path: "study.fea", source: revisedSource },
            { type: "write", path: "verify.m", source: revisedFunction },
        ] });
        assert.equal(proposal.publication_valid, true, JSON.stringify(proposal));
        assert.equal(new TextDecoder().decode(await filesystem.readFile("/study.fea")), source);
        assert.equal(new TextDecoder().decode(await filesystem.readFile("/verify.m")), functionSource);
        const publication = await session.publishFeaChanges(proposal.id);
        assert.equal(publication.type, "published", JSON.stringify(publication));
        assert.equal(new TextDecoder().decode(await filesystem.readFile("/study.fea")), revisedSource);
        assert.equal(new TextDecoder().decode(await filesystem.readFile("/verify.m")), revisedFunction);
        const published = await session.checkFeaWorkingSource({ root: "study.fea", overlays: [] });
        assert.equal(published.status, "valid", JSON.stringify(published));
        assert.equal(published.program.studies[0].assessments.length, 2);
        assert.notDeepEqual(published.revision, checked.revision);
        console.log(`Actual WASM (${providerName}): retained assessments, dimensional rejection, pure functions, unsaved overlays and checked multi-file publication passed`);
    } finally {
        await project.dispose();
        await session.dispose();
        cache.close();
    }
}

await qualify(createInMemoryFsProvider(), "memory");
const durable = await createIndexedDbFsHandle({ dbName: `assessment-project-${crypto.randomUUID()}` });
try {
    await qualify(durable.provider, "IndexedDB");
} finally {
    durable.close();
}
