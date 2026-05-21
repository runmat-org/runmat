import { NextRequest, NextResponse } from "next/server";

type ReleaseChannel = "dev" | "prod";
type Platform =
  | "darwin-aarch64"
  | "darwin-x86_64"
  | "linux-x86_64"
  | "windows-x86_64";

type DownloadEntry = {
  platform: Platform;
  target: "darwin" | "linux" | "windows";
  arch: "aarch64" | "x86_64";
  kind: string;
  file_name: string;
  url: string;
  size_bytes?: number;
};

type DownloadManifest = {
  channel: ReleaseChannel;
  version: string;
  published_at: string;
  downloads: Partial<Record<Platform, DownloadEntry>>;
};

const DOWNLOAD_HOSTS: Record<ReleaseChannel, string> = {
  dev: "https://updates.runmat.dev",
  prod: "https://updates.runmat.com",
};

const VALID_PLATFORMS = new Set<Platform>([
  "darwin-aarch64",
  "darwin-x86_64",
  "linux-x86_64",
  "windows-x86_64",
]);

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const channel = resolveChannel(request);
  const platform = resolvePlatform(request);

  if (!platform) {
    return redirectToChooser(request, "macos");
  }

  try {
    const manifest = await fetchDownloadManifest(channel);
    const entry = manifest.downloads[platform];

    if (!entry || !isSafeDownloadUrl(entry.url, channel)) {
      return redirectToChooser(request, platform.split("-")[0]);
    }

    return NextResponse.redirect(entry.url, 302);
  } catch {
    return redirectToChooser(request, platform.split("-")[0], "unavailable");
  }
}

function resolveChannel(request: NextRequest): ReleaseChannel {
  const host = (
    request.headers.get("x-forwarded-host") ??
    request.headers.get("host") ??
    ""
  ).toLowerCase();

  const requestedChannel = request.nextUrl.searchParams.get("channel");
  if (
    (requestedChannel === "dev" || requestedChannel === "prod") &&
    allowsChannelOverride(host)
  ) {
    return requestedChannel;
  }

  return host === "runmat.com" || host === "www.runmat.com" ? "prod" : "dev";
}

function allowsChannelOverride(host: string) {
  return (
    host.startsWith("localhost") ||
    host.startsWith("127.0.0.1") ||
    host.endsWith(".vercel.app") ||
    host === "runmat.dev" ||
    host.endsWith(".runmat.dev")
  );
}

function resolvePlatform(request: NextRequest): Platform | null {
  const explicitPlatform = request.nextUrl.searchParams.get("platform");
  if (isPlatform(explicitPlatform)) {
    return explicitPlatform;
  }

  const uaPlatform = normalizeHeader(request.headers.get("sec-ch-ua-platform"));
  const uaArch = normalizeHeader(request.headers.get("sec-ch-ua-arch"));

  if (uaPlatform.includes("windows")) return "windows-x86_64";
  // ARM Linux installers are not published yet; send those users to the chooser.
  if (uaPlatform.includes("linux")) return uaArch.includes("arm") ? null : "linux-x86_64";
  if (uaPlatform.includes("mac")) return resolveMacPlatform(uaArch);

  const userAgent = request.headers.get("user-agent")?.toLowerCase() ?? "";
  if (userAgent.includes("windows")) return "windows-x86_64";
  // ARM Linux installers are not published yet; send those users to the chooser.
  if (userAgent.includes("linux")) return userAgent.includes("aarch64") || userAgent.includes("arm64") ? null : "linux-x86_64";
  if (userAgent.includes("mac os x") || userAgent.includes("macintosh")) {
    return resolveMacPlatform(uaArch) ?? resolveMacPlatformFromUserAgent(userAgent);
  }

  return null;
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

async function fetchDownloadManifest(channel: ReleaseChannel): Promise<DownloadManifest> {
  const baseUrl = DOWNLOAD_HOSTS[channel];
  const response = await fetch(`${baseUrl}/${channel}/downloads/latest.json`, {
    cache: "no-store",
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch ${channel} download manifest`);
  }

  return response.json() as Promise<DownloadManifest>;
}

function redirectToChooser(
  request: NextRequest,
  target: string,
  reason?: string
) {
  const url = request.nextUrl.clone();
  url.pathname = "/download";
  url.search = "";
  url.searchParams.set("os", normalizeChooserTarget(target));
  if (reason) {
    url.searchParams.set("download", reason);
  }
  return NextResponse.redirect(url, 302);
}

function normalizeChooserTarget(target: string) {
  if (target === "darwin") return "macos";
  if (target === "windows") return "windows";
  if (target === "linux") return "linux";
  return "macos";
}

function normalizeHeader(value: string | null) {
  return (value ?? "").replaceAll('"', "").toLowerCase();
}

function isPlatform(value: string | null): value is Platform {
  return Boolean(value && VALID_PLATFORMS.has(value as Platform));
}

function isSafeDownloadUrl(url: string, channel: ReleaseChannel) {
  try {
    const parsed = new URL(url);
    return parsed.origin === DOWNLOAD_HOSTS[channel];
  } catch {
    return false;
  }
}
