"use client";

import { useState } from "react";
import { Copy, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { trackEvent } from "@/components/GoogleAnalytics";
import { cn } from "@/lib/utils";

type Platform = "macos" | "windows" | "linux";
type PowerShellVersion = "ps7" | "ps5";

interface CLITabProps {
  detectedPlatform: Platform;
}

function getInstallCommand(platform: Platform, psVersion: PowerShellVersion): string {
  if (platform === "windows") {
    if (psVersion === "ps7") {
      return 'powershell -c "irm runmat.com/install.ps1|iex"';
    } else {
      return 'powershell -c "irm runmat.com/install-legacy.ps1|iex"';
    }
  }
  return "curl -fsSL https://runmat.com/install.sh | sh";
}

function getPlatformLabel(platform: Platform): string {
  switch (platform) {
    case "macos":
      return "macOS";
    case "windows":
      return "Windows";
    case "linux":
      return "Linux";
  }
}

export function CLITab({ detectedPlatform }: CLITabProps) {
  const [platform, setPlatform] = useState<Platform>(detectedPlatform);
  const [psVersion, setPsVersion] = useState<PowerShellVersion>("ps7");
  const [copied, setCopied] = useState(false);

  const command = getInstallCommand(platform, psVersion);
  const platformLabel = getPlatformLabel(platform);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      trackEvent("copy_install_command", "installation", platform);
    } catch {
      // Fallback for browsers that don't support clipboard API
      const textArea = document.createElement("textarea");
      textArea.value = command;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand("copy");
      document.body.removeChild(textArea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      trackEvent("copy_install_command", "installation", platform);
    }
  };

  const handlePlatformChange = (newPlatform: Platform) => {
    setPlatform(newPlatform);
    trackEvent("select_os", "installation", newPlatform);
  };

  const handlePsVersionChange = (version: PowerShellVersion) => {
    setPsVersion(version);
  };

  return (
    <div className="flex flex-col gap-4 pt-4">
      <div className="text-sm font-medium text-foreground">
        Install for {platformLabel}
      </div>

      <div className="relative group">
        <div
          className={cn(
            "font-mono text-sm",
            "flex items-center justify-between gap-3",
            "min-h-[3rem]"
          )}
        >
          <code className="flex-1 break-all select-all text-primary">{command}</code>
          <button
            type="button"
            onClick={handleCopy}
            className={cn(
              "p-2 transition-opacity flex-shrink-0",
              "opacity-0 group-hover:opacity-100 focus:opacity-100",
              "text-muted-foreground hover:text-foreground",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            )}
            aria-label="Copy install command"
          >
            {copied ? (
              <Check className="h-4 w-4 text-green-500" />
            ) : (
              <Copy className="h-4 w-4" />
            )}
          </button>
        </div>
        {copied && (
          <div className="absolute -top-8 right-2 bg-green-600 text-white text-xs px-2 py-1 rounded">
            Copied!
          </div>
        )}
      </div>

      {platform === "windows" && (
        <div className="flex items-center justify-start gap-2 text-xs text-muted-foreground">
          <span>Need a different PowerShell?</span>
          <button
            type="button"
            onClick={() => handlePsVersionChange("ps7")}
            className={cn(
              "px-2 py-1 rounded transition-colors",
              psVersion === "ps7"
                ? "bg-primary text-primary-foreground"
                : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
            )}
          >
            PS 7+
          </button>
          <span>|</span>
          <button
            type="button"
            onClick={() => handlePsVersionChange("ps5")}
            className={cn(
              "px-2 py-1 rounded transition-colors",
              psVersion === "ps5"
                ? "bg-primary text-primary-foreground"
                : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
            )}
          >
            PS 5.x
          </button>
        </div>
      )}

      <div className="flex flex-col gap-2">
        <p className="text-xs text-muted-foreground">
          Not {platformLabel}? Pick your platform:
        </p>
        <div className="flex flex-wrap gap-2">
          {platform !== "windows" && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => handlePlatformChange("windows")}
              className="text-xs"
            >
              Windows
            </Button>
          )}
          {platform !== "macos" && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => handlePlatformChange("macos")}
              className="text-xs"
            >
              macOS
            </Button>
          )}
          {platform !== "linux" && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => handlePlatformChange("linux")}
              className="text-xs"
            >
              Linux
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
