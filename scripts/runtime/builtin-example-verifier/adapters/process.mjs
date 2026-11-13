// @ts-check

import { spawn } from "node:child_process";

export function runBoundedProcess(executable, args, options) {
    return new Promise((resolveProcess) => {
        const child = spawn(executable, args, {
            cwd: options.cwd,
            env: options.env,
            stdio: ["ignore", "pipe", "pipe"]
        });
        const stdout = [];
        const stderr = [];
        let outputBytes = 0;
        let processError = null;
        let timedOut = false;
        const capture = (target) => (chunk) => {
            outputBytes += chunk.byteLength;
            if (outputBytes <= options.maxBuffer) target.push(Buffer.from(chunk));
            else if (!processError) {
                processError = new Error(`process output exceeded ${options.maxBuffer} bytes`);
                child.kill("SIGKILL");
            }
        };
        child.stdout.on("data", capture(stdout));
        child.stderr.on("data", capture(stderr));
        child.once("error", (error) => { processError = error; });
        const timer = setTimeout(() => {
            timedOut = true;
            child.kill("SIGKILL");
        }, options.timeout);
        child.once("close", (status, signal) => {
            clearTimeout(timer);
            if (timedOut) processError = new Error(`process timed out after ${options.timeout}ms`);
            else if (signal && !processError) processError = new Error(`process terminated by ${signal}`);
            resolveProcess({
                status,
                stdout: Buffer.concat(stdout).toString("utf8"),
                stderr: Buffer.concat(stderr).toString("utf8"),
                error: processError
            });
        });
    });
}
