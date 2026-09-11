// @ts-check

import { spawn } from "node:child_process";

/**
 * Drive one closed catalog transcript against a child process. Expectations
 * consume output in order; writes and the single terminal action are taken
 * only from the reviewed fixture.
 */
export function runInteraction(executable, args, options, transcript) {
    return new Promise((resolveInteraction) => {
        const child = spawn(executable, args, {
            cwd: options.cwd,
            env: options.env,
            stdio: ["pipe", "pipe", "pipe"]
        });
        const stdout = [];
        const stderr = [];
        let combined = "";
        let outputBytes = 0;
        let processError = null;
        let status = null;
        let signal = null;
        let closed = false;
        let settled = false;
        let cursor = 0;
        let wake = () => {};
        const changed = () => new Promise((resolveChange) => { wake = resolveChange; });
        const notify = () => { const current = wake; wake = () => {}; current(); };
        const capture = (target) => (chunk) => {
            const bytes = Buffer.from(chunk);
            outputBytes += bytes.byteLength;
            if (outputBytes > options.maxBuffer) {
                processError ??= new Error(`process output exceeded ${options.maxBuffer} bytes`);
                child.kill("SIGKILL");
                notify();
                return;
            }
            target.push(bytes);
            combined += bytes.toString("utf8");
            notify();
        };
        child.stdout.on("data", capture(stdout));
        child.stderr.on("data", capture(stderr));
        child.once("error", (error) => { processError ??= error; notify(); });
        child.once("close", (code, receivedSignal) => {
            status = code;
            signal = receivedSignal;
            closed = true;
            notify();
        });
        const timer = setTimeout(() => {
            processError ??= new Error(`interactive process timed out after ${options.timeout}ms`);
            child.kill("SIGKILL");
            notify();
        }, options.timeout);

        const finish = () => {
            if (settled) return;
            settled = true;
            clearTimeout(timer);
            resolveInteraction({
                status,
                signal,
                stdout: Buffer.concat(stdout).toString("utf8"),
                stderr: Buffer.concat(stderr).toString("utf8"),
                error: processError
            });
        };

        (async () => {
            try {
                for (const step of transcript) {
                    if (step === "SendEndOfInput") {
                        child.stdin.end();
                    } else if (step === "SendInterrupt") {
                        child.kill("SIGINT");
                    } else if ("SendLine" in step) {
                        await write(child.stdin, `${step.SendLine}\n`);
                    } else if ("SendBytes" in step) {
                        await write(child.stdin, Buffer.from(step.SendBytes));
                    } else {
                        const expected = step.ExpectOutput;
                        while (true) {
                            const location = combined.indexOf(expected, cursor);
                            if (location >= 0) {
                                cursor = location + expected.length;
                                break;
                            }
                            if (processError) throw processError;
                            if (closed) throw new Error(`interactive process closed before output ${JSON.stringify(expected)}`);
                            await changed();
                        }
                    }
                }
                while (!closed && !processError) await changed();
            } catch (error) {
                processError ??= error instanceof Error ? error : new Error(String(error));
                if (!closed) child.kill("SIGKILL");
                while (!closed) await changed();
            }
            finish();
        })();
    });
}

function write(stream, content) {
    return new Promise((resolveWrite, rejectWrite) => {
        stream.write(content, (error) => error ? rejectWrite(error) : resolveWrite());
    });
}
