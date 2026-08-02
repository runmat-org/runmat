import {
  BrowserWorkerBackend,
  type BrowserWorkerBackendOptions
} from "./browser-worker-backend.js";
import type {
  BrowserTestRunInput,
  BrowserTestRunOutput,
  BrowserTestEvent,
  RunMatTestNative
} from "./types.js";

export interface BrowserTestRunnerOptions
  extends BrowserWorkerBackendOptions {
  native: RunMatTestNative;
  artifactStore?: BrowserTestArtifactStore;
  onEvent?: (event: BrowserTestEvent) => void;
}

export interface BrowserTestArtifactStore {
  put(
    runId: string,
    report: BrowserTestRunOutput["reports"][number]
  ): Promise<void>;
}

export class BrowserTestRunner {
  constructor(private readonly options: BrowserTestRunnerOptions) {}

  async run(input: BrowserTestRunInput): Promise<BrowserTestRunOutput> {
    const backend = new BrowserWorkerBackend({
      ...this.options,
      requestedIsolation: input.options?.isolation
    });
    try {
      const output =
        this.options.onEvent && this.options.native.runTestsWithEvents
          ? await this.options.native.runTestsWithEvents(
              input,
              backend,
              this.options.onEvent
            )
          : await this.options.native.runTests(input, backend);
      if (this.options.onEvent && !this.options.native.runTestsWithEvents) {
        for (const event of output.events) this.options.onEvent(event);
      }
      if (this.options.artifactStore) {
        await Promise.all(
          output.reports.map((report) =>
            this.options.artifactStore!.put(output.result.run_id, report)
          )
        );
      }
      return output;
    } finally {
      await backend.dispose();
    }
  }
}
