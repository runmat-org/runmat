// @ts-check

import { createServer as createHttpServer } from "node:http";
import { createServer as createTcpServer } from "node:net";

import { substituteLoopbackEndpoints } from "./substitutions.mjs";

const LOOPBACK_HOST = "127.0.0.1";

/**
 * Run an example against one exact, catalog-owned loopback scenario. The
 * fixture succeeds only when every declared exchange is consumed once and in
 * order.
 *
 * @template T
 * @param {string} program
 * @param {import("../../../metadata/BuiltinMetadataSpecification").BuiltinLoopbackFixture} fixture
 * @param {(prepared: {program: string, host: string, port: number}) => Promise<T>} execute
 * @returns {Promise<{value: T, evidence: object}>}
 */
export async function withLoopbackFixture(program, fixture, execute) {
    const scenario = fixture.scenario;
    const host = "Http" in scenario
        ? createHttpFixtureHost(scenario.Http.exchanges)
        : createTcpFixtureHost(scenario.Tcp.exchanges);
    const endpoint = await host.start();
    let value;
    let executionError;
    try {
        value = await execute({
            program: substituteLoopbackEndpoints(program, fixture, endpoint),
            host: endpoint.host,
            port: endpoint.port
        });
    } catch (error) {
        executionError = error;
    }
    let evidence;
    let fixtureError;
    try {
        evidence = await host.finish();
    } catch (error) {
        fixtureError = error;
    }
    if (executionError && fixtureError) {
        throw new AggregateError([executionError, fixtureError], "Example execution and loopback fixture both failed");
    }
    if (executionError) throw executionError;
    if (fixtureError) throw fixtureError;
    return { value, evidence };
}

function createHttpFixtureHost(exchanges) {
    let index = 0;
    let mismatch = null;
    const server = createHttpServer((request, response) => {
        const chunks = [];
        request.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
        request.on("end", () => {
            const expected = exchanges[index];
            if (!expected) {
                mismatch ??= new Error("HTTP fixture received an undeclared exchange");
                response.statusCode = 500;
                response.end();
                return;
            }
            const actualBody = Buffer.concat(chunks);
            const expectedBody = expected.request.body === null ? Buffer.alloc(0) : Buffer.from(expected.request.body);
            const actualPath = String(request.url ?? "");
            if (request.method !== expected.request.method.toUpperCase()
                || actualPath !== expected.request.path
                || !actualBody.equals(expectedBody)) {
                mismatch ??= new Error(`HTTP fixture exchange ${index + 1} did not match its catalog request`);
                response.statusCode = 500;
                response.end();
                return;
            }
            index += 1;
            response.statusCode = expected.response.status;
            for (const header of expected.response.headers) response.setHeader(header.name, header.value);
            response.end(Buffer.from(expected.response.body));
        });
    });
    return lifecycle(server, "http", () => ({ consumedExchanges: index, expectedExchanges: exchanges.length, mismatch }));
}

function createTcpFixtureHost(exchanges) {
    let index = 0;
    let mismatch = null;
    const server = createTcpServer((socket) => {
        const expected = exchanges[index];
        if (!expected) {
            mismatch ??= new Error("TCP fixture received an undeclared exchange");
            socket.destroy();
            return;
        }
        const chunks = [];
        socket.on("data", (chunk) => {
            chunks.push(Buffer.from(chunk));
            const received = Buffer.concat(chunks);
            const wanted = Buffer.from(expected.client_bytes);
            if (received.byteLength < wanted.byteLength) return;
            if (received.byteLength !== wanted.byteLength || !received.equals(wanted)) {
                mismatch ??= new Error(`TCP fixture exchange ${index + 1} did not match its catalog request`);
                socket.destroy();
                return;
            }
            index += 1;
            socket.end(Buffer.from(expected.server_bytes));
        });
    });
    return lifecycle(server, "tcp", () => ({ consumedExchanges: index, expectedExchanges: exchanges.length, mismatch }));
}

function lifecycle(server, protocol, state) {
    return {
        start: () => new Promise((resolve, reject) => {
            server.once("error", reject);
            server.listen(0, LOOPBACK_HOST, () => {
                server.removeListener("error", reject);
                const address = server.address();
                if (!address || typeof address === "string") {
                    reject(new Error("Loopback fixture did not bind an IP endpoint"));
                    return;
                }
                resolve({ protocol, host: LOOPBACK_HOST, port: address.port });
            });
        }),
        finish: () => new Promise((resolve, reject) => {
            const snapshot = state();
            server.close((error) => {
                if (error) {
                    reject(error);
                } else if (snapshot.mismatch) {
                    reject(snapshot.mismatch);
                } else if (snapshot.consumedExchanges !== snapshot.expectedExchanges) {
                    reject(new Error(`Loopback fixture consumed ${snapshot.consumedExchanges} of ${snapshot.expectedExchanges} exchanges`));
                } else {
                    resolve({ protocol, consumedExchanges: snapshot.consumedExchanges });
                }
            });
        })
    };
}
