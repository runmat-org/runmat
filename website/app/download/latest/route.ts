import { NextRequest, NextResponse } from "next/server";

import {
  getDownloadHost,
  isSafeDownloadUrl,
  resolveChannelFromHost,
  resolvePlatformFromInputs,
  type Platform,
  type ReleaseChannel,
} from "./resolver";

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
  return resolveChannelFromHost(
    request.nextUrl.host,
    request.nextUrl.searchParams.get("channel")
  );
}

function resolvePlatform(request: NextRequest): Platform | null {
  return resolvePlatformFromInputs({
    explicitPlatform: request.nextUrl.searchParams.get("platform"),
    secChUaPlatform: request.headers.get("sec-ch-ua-platform"),
    secChUaArch: request.headers.get("sec-ch-ua-arch"),
    userAgent: request.headers.get("user-agent"),
  });
}

async function fetchDownloadManifest(channel: ReleaseChannel): Promise<DownloadManifest> {
  const baseUrl = getDownloadHost(channel);
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
