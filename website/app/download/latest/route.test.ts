import { describe, expect, it } from "vitest";

import {
  isSafeDownloadUrl,
  resolveChannelFromHost,
  resolvePlatformFromInputs,
} from "./resolver";

describe("download latest route helpers", () => {
  describe("resolvePlatformFromInputs", () => {
    it("defaults Mac user agents without Client Hints to Apple Silicon", () => {
      expect(
        resolvePlatformFromInputs({
          userAgent:
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        })
      ).toBe("darwin-aarch64");
    });

    it("defaults Mac Chrome with platform Client Hint but no arch Client Hint to Apple Silicon", () => {
      expect(
        resolvePlatformFromInputs({
          secChUaPlatform: '"macOS"',
          userAgent:
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        })
      ).toBe("darwin-aarch64");
    });

    it("resolves Mac Apple Silicon from Client Hints", () => {
      expect(
        resolvePlatformFromInputs({
          secChUaPlatform: '"macOS"',
          secChUaArch: '"arm"',
          userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        })
      ).toBe("darwin-aarch64");
    });

    it("resolves Mac Intel from Client Hints", () => {
      expect(
        resolvePlatformFromInputs({
          secChUaPlatform: '"macOS"',
          secChUaArch: '"x86"',
          userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        })
      ).toBe("darwin-x86_64");
    });

    it("sends ARM Linux users to the chooser", () => {
      expect(
        resolvePlatformFromInputs({
          userAgent: "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36",
        })
      ).toBeNull();
      expect(
        resolvePlatformFromInputs({
          secChUaPlatform: '"Linux"',
          secChUaArch: '"arm"',
        })
      ).toBeNull();
    });

    it("uses an explicit supported platform before header detection", () => {
      expect(
        resolvePlatformFromInputs({
          explicitPlatform: "windows-x86_64",
          secChUaPlatform: '"macOS"',
          secChUaArch: '"arm"',
        })
      ).toBe("windows-x86_64");
    });
  });

  describe("isSafeDownloadUrl", () => {
    it("rejects manifest URLs outside the selected channel host", () => {
      expect(isSafeDownloadUrl("https://updates.runmat.com/prod/downloads/runmat.dmg", "prod")).toBe(true);
      expect(isSafeDownloadUrl("https://downloads.example.invalid/dev/downloads/runmat.dmg", "prod")).toBe(false);
      expect(isSafeDownloadUrl("https://example.com/runmat.dmg", "prod")).toBe(false);
      expect(isSafeDownloadUrl("not a url", "prod")).toBe(false);
    });
  });

  describe("resolveChannelFromHost", () => {
    it("selects prod for canonical production hosts with optional ports", () => {
      expect(resolveChannelFromHost("runmat.com:443", null, {})).toBe("prod");
      expect(resolveChannelFromHost("www.runmat.com", null, {})).toBe("prod");
    });

    it("selects dev for non-production hosts", () => {
      expect(resolveChannelFromHost("preview.example.invalid", null, {})).toBe("dev");
    });

    it("supports explicitly configured production aliases", () => {
      expect(
        resolveChannelFromHost("download.runmat-cdn.com", null, {
          RUNMAT_DOWNLOAD_PROD_HOSTS: "download.runmat-cdn.com",
        })
      ).toBe("prod");
    });

    it("does not allow query channel overrides on production hosts", () => {
      expect(resolveChannelFromHost("runmat.com", "dev", {})).toBe("prod");
    });

    it("allows query channel overrides on localhost", () => {
      expect(resolveChannelFromHost("localhost:3000", "prod", {})).toBe("prod");
      expect(resolveChannelFromHost("localhost:3000", "dev", {})).toBe("dev");
    });

    it("allows query channel overrides on explicitly configured hosts", () => {
      expect(
        resolveChannelFromHost("preview.example.invalid", "prod", {
          RUNMAT_DOWNLOAD_CHANNEL_OVERRIDE_HOSTS: "preview.example.invalid",
        })
      ).toBe("prod");
    });

    it("allows environment channel configuration to override host selection", () => {
      expect(resolveChannelFromHost("runmat.com", null, { RUNMAT_DOWNLOAD_CHANNEL: "dev" })).toBe("dev");
      expect(resolveChannelFromHost("preview.example.invalid", null, { RUNMAT_DOWNLOAD_CHANNEL: "prod" })).toBe("prod");
    });
  });
});
