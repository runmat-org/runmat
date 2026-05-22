export type ReleaseChannel = "dev" | "prod";
export type Platform =
  | "darwin-aarch64"
  | "darwin-x86_64"
  | "linux-x86_64"
  | "windows-x86_64";

const PROD_DOWNLOAD_HOST = "https://updates.runmat.com";

const VALID_PLATFORMS = new Set<Platform>([
  "darwin-aarch64",
  "darwin-x86_64",
  "linux-x86_64",
  "windows-x86_64",
]);

export function resolveChannelFromHost(
  host: string,
  requestedChannel?: string | null,
  env: Record<string, string | undefined> = process.env
): ReleaseChannel {
  const configuredChannel = env["RUNMAT_DOWNLOAD_CHANNEL"];
  if (configuredChannel === "dev" || configuredChannel === "prod") {
    return configuredChannel;
  }

  const normalizedHost = normalizeHost(host);
  if (
    (requestedChannel === "dev" || requestedChannel === "prod") &&
    allowsChannelOverride(normalizedHost, env)
  ) {
    return requestedChannel;
  }

  return isProductionHost(normalizedHost, env) ? "prod" : "dev";
}

export function resolvePlatformFromInputs({
  explicitPlatform,
  secChUaPlatform,
  secChUaArch,
  userAgent: rawUserAgent,
}: {
  explicitPlatform?: string | null;
  secChUaPlatform?: string | null;
  secChUaArch?: string | null;
  userAgent?: string | null;
}): Platform | null {
  if (isPlatform(explicitPlatform)) {
    return explicitPlatform;
  }

  const uaPlatform = normalizeHeader(secChUaPlatform ?? null);
  const uaArch = normalizeHeader(secChUaArch ?? null);

  if (uaPlatform.includes("windows")) return "windows-x86_64";
  // ARM Linux installers are not published yet; send those users to the chooser.
  if (uaPlatform.includes("linux")) return uaArch.includes("arm") ? null : "linux-x86_64";

  const userAgent = rawUserAgent?.toLowerCase() ?? "";
  if (uaPlatform.includes("mac")) {
    return resolveMacPlatform(uaArch) ?? resolveMacPlatformFromUserAgent(userAgent);
  }
  if (userAgent.includes("windows")) return "windows-x86_64";
  // ARM Linux installers are not published yet; send those users to the chooser.
  if (userAgent.includes("linux")) {
    return userAgent.includes("aarch64") || userAgent.includes("arm64") ? null : "linux-x86_64";
  }
  if (userAgent.includes("mac os x") || userAgent.includes("macintosh")) {
    return resolveMacPlatform(uaArch) ?? resolveMacPlatformFromUserAgent(userAgent);
  }

  return null;
}

export function isSafeDownloadUrl(
  url: string,
  channel: ReleaseChannel,
  env: Record<string, string | undefined> = process.env
) {
  try {
    const parsed = new URL(url);
    return parsed.origin === getDownloadHost(channel, env);
  } catch {
    return false;
  }
}

export function getDownloadHost(channel: ReleaseChannel, env: Record<string, string | undefined> = process.env) {
  const configuredHost =
    channel === "prod"
      ? env["RUNMAT_DOWNLOAD_PROD_ORIGIN"]
      : env["RUNMAT_DOWNLOAD_DEV_ORIGIN"];
  const fallbackHost = channel === "prod" ? PROD_DOWNLOAD_HOST : null;
  const host = configuredHost ?? fallbackHost;
  if (!host) {
    throw new Error(`Missing ${channel} download origin`);
  }
  return new URL(host).origin;
}

function allowsChannelOverride(host: string, env: Record<string, string | undefined>) {
  return (
    !isProductionHost(host, env) &&
    (host.startsWith("localhost") ||
      host.startsWith("127.0.0.1") ||
      getChannelOverrideHosts(env).has(host))
  );
}

function resolveMacPlatform(arch: string): Platform | null {
  if (arch.includes("arm") || arch.includes("aarch64")) return "darwin-aarch64";
  if (arch.includes("x86") || arch.includes("amd64")) return "darwin-x86_64";

  return null;
}

function resolveMacPlatformFromUserAgent(userAgent: string): Platform {
  if (userAgent.includes("arm") || userAgent.includes("aarch64")) return "darwin-aarch64";
  if (userAgent.includes("x86_64")) return "darwin-x86_64";

  // Safari and Firefox do not send Client Hints, and modern macOS UAs do not
  // reliably expose CPU architecture. Default to Apple Silicon for the direct
  // download path; the download page still offers explicit Intel and Apple links.
  return "darwin-aarch64";
}

function normalizeHeader(value: string | null) {
  return (value ?? "").replaceAll('"', "").toLowerCase();
}

function isPlatform(value: string | null | undefined): value is Platform {
  return Boolean(value && VALID_PLATFORMS.has(value as Platform));
}

function isProductionHost(host: string, env: Record<string, string | undefined>) {
  return getProductionHosts(env).has(host);
}

function getProductionHosts(env: Record<string, string | undefined>) {
  const hosts = new Set(["runmat.com", "www.runmat.com"]);
  for (const value of [env["RUNMAT_DOWNLOAD_PROD_HOSTS"]]) {
    for (const host of parseHostList(value)) {
      hosts.add(host);
    }
  }
  return hosts;
}

function getChannelOverrideHosts(env: Record<string, string | undefined>) {
  return new Set(parseHostList(env["RUNMAT_DOWNLOAD_CHANNEL_OVERRIDE_HOSTS"]));
}

function parseHostList(value: string | undefined) {
  return (value ?? "")
    .split(",")
    .map((host) => normalizeHost(host))
    .filter(Boolean);
}

function normalizeHost(value: string | null | undefined) {
  const trimmed = (value ?? "").trim().toLowerCase();
  if (!trimmed) {
    return "";
  }

  try {
    return new URL(trimmed.includes("://") ? trimmed : `https://${trimmed}`).hostname;
  } catch {
    return trimmed.split(":")[0] ?? "";
  }
}
