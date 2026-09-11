import assert from "node:assert/strict";
import { readFileSync, rmSync, mkdtempSync } from "node:fs";
import { request } from "node:http";
import { connect } from "node:net";
import { join } from "node:path";
import { tmpdir } from "node:os";
import test from "node:test";

import { fixturePath, materializeFilesystemFixture } from "./filesystem.mjs";
import { runInteraction } from "./interaction.mjs";
import { withLoopbackFixture } from "./loopback.mjs";
import { substituteLoopbackEndpoints } from "./substitutions.mjs";

const id = { local_name: "fixture" };

test("filesystem fixtures materialize exact catalog bytes below a fresh root", () => {
    const parent = mkdtempSync(join(tmpdir(), "runmat-fixture-"));
    const root = join(parent, "workspace");
    try {
        const evidence = materializeFilesystemFixture(root, {
            id,
            root: "IsolatedWorkspace",
            entries: [
                { Directory: { relative_path: "data" } },
                { File: { relative_path: "data/value.txt", content: { Utf8: "forty-two\n" } } },
                { File: { relative_path: "raw.bin", content: { Bytes: [0, 1, 255] } } }
            ]
        });
        assert.equal(readFileSync(join(root, "data/value.txt"), "utf8"), "forty-two\n");
        assert.deepEqual([...readFileSync(join(root, "raw.bin"))], [0, 1, 255]);
        assert.deepEqual(evidence.entries.map((entry) => entry.relativePath), ["data", "data/value.txt", "raw.bin"]);
        assert.throws(() => materializeFilesystemFixture(root, { id, root: "IsolatedWorkspace", entries: [] }), /already exists/);
        assert.throws(() => fixturePath(root, "../escape"), /Invalid fixture/);
    } finally {
        rmSync(parent, { recursive: true, force: true });
    }
});

test("endpoint substitutions require an exact declaration", () => {
    const fixture = {
        id,
        scenario: { Http: { exchanges: [] } },
        endpoint_substitutions: ["HttpBaseUrl", "LoopbackPort"]
    };
    const output = substituteLoopbackEndpoints(
        'base = "__RUNMAT_HTTP_BASE_URL__"; port = __RUNMAT_LOOPBACK_PORT__;',
        fixture,
        { protocol: "http", host: "127.0.0.1", port: 4123 }
    );
    assert.equal(output, 'base = "http://127.0.0.1:4123"; port = 4123;');
    assert.throws(() => substituteLoopbackEndpoints("__RUNMAT_LOOPBACK_HOST__", fixture, { protocol: "http", host: "127.0.0.1", port: 1 }), /do not agree/);
});

test("HTTP loopback fixtures require every exact exchange", async () => {
    const fixture = {
        id,
        scenario: { Http: { exchanges: [{
            request: { method: "Post", path: "/value", body: [1, 2] },
            response: { status: 201, headers: [{ name: "content-type", value: "application/octet-stream" }], body: [3, 4] }
        }] } },
        endpoint_substitutions: ["HttpBaseUrl"]
    };
    const result = await withLoopbackFixture('url = "__RUNMAT_HTTP_BASE_URL__";', fixture, async ({ program, host, port }) => {
        const response = await httpRequest(host, port, "/value", Buffer.from([1, 2]));
        assert.equal(response.status, 201);
        assert.deepEqual([...response.body], [3, 4]);
        return program;
    });
    assert.match(result.value, /http:\/\/127\.0\.0\.1:/);
    assert.deepEqual(result.evidence, { protocol: "http", consumedExchanges: 1 });
    await assert.rejects(
        withLoopbackFixture("port = __RUNMAT_LOOPBACK_PORT__;", { ...fixture, endpoint_substitutions: ["LoopbackPort"] }, async () => undefined),
        /consumed 0 of 1/
    );
});

test("TCP loopback fixtures match exact request and response bytes", async () => {
    const fixture = {
        id,
        scenario: { Tcp: { exchanges: [{ client_bytes: [9, 8], server_bytes: [7, 6] }] } },
        endpoint_substitutions: ["LoopbackHost", "LoopbackPort"]
    };
    const result = await withLoopbackFixture(
        'host = "__RUNMAT_LOOPBACK_HOST__"; port = __RUNMAT_LOOPBACK_PORT__;',
        fixture,
        async ({ host, port }) => tcpExchange(host, port, Buffer.from([9, 8]))
    );
    assert.deepEqual([...result.value], [7, 6]);
    assert.deepEqual(result.evidence, { protocol: "tcp", consumedExchanges: 1 });
});

test("interactive adapters drive the reviewed transcript in order", async () => {
    const script = [
        "process.stdout.write('value: ');",
        "process.stdin.setEncoding('utf8');",
        "let input = '';",
        "process.stdin.on('data', chunk => { input += chunk; });",
        "process.stdin.on('end', () => process.stdout.write('received=' + input.trim()));"
    ].join("");
    const result = await runInteraction(
        process.execPath,
        ["-e", script],
        { cwd: process.cwd(), env: process.env, timeout: 2_000, maxBuffer: 4_096 },
        [{ ExpectOutput: "value: " }, { SendLine: "42" }, "SendEndOfInput", { ExpectOutput: "received=42" }]
    );
    assert.equal(result.error, null);
    assert.equal(result.status, 0);
    assert.equal(result.stdout, "value: received=42");
});

test("interactive adapters fail when an ordered expectation is absent", async () => {
    const result = await runInteraction(
        process.execPath,
        ["-e", "process.stdout.write('different');"],
        { cwd: process.cwd(), env: process.env, timeout: 2_000, maxBuffer: 4_096 },
        ["SendEndOfInput", { ExpectOutput: "expected" }]
    );
    assert.match(result.error?.message ?? "", /closed before output/);
});

function httpRequest(host, port, path, body) {
    return new Promise((resolve, reject) => {
        const outgoing = request({ host, port, path, method: "POST" }, (incoming) => {
            const chunks = [];
            incoming.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
            incoming.on("end", () => resolve({ status: incoming.statusCode, body: Buffer.concat(chunks) }));
        });
        outgoing.once("error", reject);
        outgoing.end(body);
    });
}

function tcpExchange(host, port, body) {
    return new Promise((resolve, reject) => {
        const socket = connect({ host, port });
        const chunks = [];
        socket.once("connect", () => socket.write(body));
        socket.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
        socket.once("end", () => resolve(Buffer.concat(chunks)));
        socket.once("error", reject);
    });
}
