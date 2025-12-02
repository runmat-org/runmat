import { describe, expect, it, afterEach, vi } from "vitest";
import worker from "./worker.js";

const basePayload = {
  event_label: "runtime_value",
  cid: "cid-test",
  session_id: "session-test",
  run_kind: "script",
  os: "darwin",
  arch: "arm64",
  release: "v0.0.1",
  payload: {
    success: true,
  },
};

afterEach(() => {
  vi.restoreAllMocks();
});

describe("telemetry worker", () => {
  it("forwards events to PostHog", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response("{}", { status: 200 }));

    const env = {
      POSTHOG_API_KEY: "phc_test",
      POSTHOG_HOST: "https://posthog.test",
      GA_MEASUREMENT_ID: "",
      GA_API_SECRET: "",
    };

    const req = new Request("https://telemetry.runmat.org/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(basePayload),
    });

    const res = await worker.fetch(req, env);
    expect(res.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith(
      "https://posthog.test/capture/",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("rejects unauthorized requests when key is set", async () => {
    const env = {
      POSTHOG_API_KEY: "phc_test",
      POSTHOG_HOST: "https://posthog.test",
      INGESTION_KEY: "secret",
      GA_MEASUREMENT_ID: "",
      GA_API_SECRET: "",
    };

    const req = new Request("https://telemetry.runmat.org/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(basePayload),
    });

    const res = await worker.fetch(req, env);
    expect(res.status).toBe(401);
  });

  it("validates path and method", async () => {
    const env = {
      POSTHOG_API_KEY: "phc_test",
      POSTHOG_HOST: "https://posthog.test",
      GA_MEASUREMENT_ID: "",
      GA_API_SECRET: "",
    };

    const badPath = new Request("https://telemetry.runmat.org/", {
      method: "POST",
      body: JSON.stringify(basePayload),
    });
    const badMethod = new Request("https://telemetry.runmat.org/ingest", {
      method: "GET",
    });

    const resPath = await worker.fetch(badPath, env);
    const resMethod = await worker.fetch(badMethod, env);

    expect(resPath.status).toBe(404);
    expect(resMethod.status).toBe(405);
  });
});

