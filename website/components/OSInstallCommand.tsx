"use client";

import { useEffect, useState } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Terminal, Monitor, Copy, Check, ExternalLink } from "lucide-react";
import { trackWebsiteEvent } from "@/components/GoogleAnalytics";

type OS = 'mac' | 'linux' | 'windows';

function CopyableCommand({ command, ariaLabel }: { command: string; ariaLabel: string }) {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);

      trackWebsiteEvent("website.install.command_copied", {
        category: "installation",
        label: command.includes("curl") ? "unix" : "windows",
      });
    } catch {
      const textArea = document.createElement('textarea');
      textArea.value = command;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);

      trackWebsiteEvent("website.install.command_copied", {
        category: "installation",
        label: command.includes("curl") ? "unix" : "windows",
      });
    }
  };

  return (
    <div
      className="bg-[var(--editor-background)] rounded-md px-4 py-4 sm:px-5 sm:py-5 font-mono text-white relative group cursor-pointer"
      onClick={copyToClipboard}
      role="button"
      tabIndex={0}
      aria-label={ariaLabel}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          copyToClipboard();
        }
      }}
    >
      <div className="flex items-center gap-3 min-w-0">
        <span className="select-all text-sm sm:text-base md:text-lg whitespace-nowrap overflow-hidden text-ellipsis flex-1 text-left">
          <span className="text-white/40 select-none">$ </span>
          {command}
        </span>
        <button
          type="button"
          aria-label={copied ? "Copied" : "Copy install command"}
          className="p-1.5 opacity-70 group-hover:opacity-100 transition-opacity bg-white/10 hover:bg-white/20 rounded flex-shrink-0"
          onClick={(e) => {
            e.stopPropagation();
            copyToClipboard();
          }}
        >
          {copied ? (
            <Check className="h-4 w-4 text-green-400" aria-hidden="true" />
          ) : (
            <Copy className="h-4 w-4" aria-hidden="true" />
          )}
        </button>
      </div>
      <span className="sr-only" aria-live="polite">
        {copied ? "Install command copied to clipboard" : ""}
      </span>
    </div>
  );
}

function detectOS(): OS {
  if (typeof window === 'undefined') return 'mac';
  const userAgent = window.navigator.userAgent.toLowerCase();
  if (userAgent.includes('win')) return 'windows';
  if (userAgent.includes('linux')) return 'linux';
  return 'mac';
}

interface OSInstallCommandProps {
  variant?: 'compact' | 'full';
  className?: string;
}

const OS_META: Record<OS, { title: string; icon: typeof Terminal; iconColor: string; scriptPath: string }> = {
  mac: { title: 'macOS', icon: Terminal, iconColor: 'text-green-500', scriptPath: '/install.sh' },
  linux: { title: 'Linux', icon: Terminal, iconColor: 'text-green-500', scriptPath: '/install.sh' },
  windows: { title: 'Windows', icon: Monitor, iconColor: 'text-[hsl(var(--brand))]', scriptPath: '/install.ps1' },
};

function commandFor(os: OS, winFlavor: 'ps7' | 'ps5'): string {
  if (os === 'windows') {
    return winFlavor === 'ps7'
      ? 'iwr https://runmat.com/install.ps1 | iex'
      : 'iwr https://runmat.com/install.ps1 -UseBasicParsing | iex';
  }
  return 'curl -fsSL https://runmat.com/install.sh | sh';
}

export function OSInstallCommand({ variant = 'full', className = '' }: OSInstallCommandProps) {
  const [selectedOS, setSelectedOS] = useState<OS>('mac');
  const [mounted, setMounted] = useState(false);
  const [winFlavor, setWinFlavor] = useState<'ps7' | 'ps5'>('ps7');

  useEffect(() => {
    setSelectedOS(detectOS());
    setMounted(true);
  }, []);

  const handleSelect = (os: OS) => {
    setSelectedOS(os);
    trackWebsiteEvent("website.install.os_selected", {
      category: "installation",
      label: os,
    });
  };

  const meta = OS_META[selectedOS];
  const Icon = meta.icon;
  const command = commandFor(selectedOS, winFlavor);

  const tabs = (
    <div
      role="tablist"
      aria-label="Select operating system"
      className="inline-flex items-center rounded-md border border-border bg-card p-1 text-sm"
    >
      {(['mac', 'linux', 'windows'] as OS[]).map((os) => {
        const isActive = mounted && selectedOS === os;
        const TabIcon = OS_META[os].icon;
        return (
          <button
            key={os}
            type="button"
            role="tab"
            aria-selected={isActive}
            onClick={() => handleSelect(os)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded transition-colors ${
              isActive
                ? 'bg-foreground text-background font-medium'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <TabIcon className="h-3.5 w-3.5" aria-hidden="true" />
            <span>{OS_META[os].title}</span>
          </button>
        );
      })}
    </div>
  );

  const inspectLink = (
    <a
      href={meta.scriptPath}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground underline-offset-4 hover:underline"
    >
      Inspect this script
      <ExternalLink className="h-3 w-3" aria-hidden="true" />
    </a>
  );

  if (variant === 'compact') {
    return (
      <div className={className}>
        <div className="flex items-center justify-between gap-3 mb-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
            <Icon className={`h-4 w-4 ${meta.iconColor}`} aria-hidden="true" />
            <span>Install for {meta.title}</span>
          </div>
          {tabs}
        </div>
        <CopyableCommand
          command={command}
          ariaLabel={`Install command for ${meta.title}. Click to copy.`}
        />
        {selectedOS === 'windows' && (
          <div className="mt-3 flex items-center justify-end gap-2 text-xs text-muted-foreground">
            <span>PowerShell:</span>
            <button
              type="button"
              className={`px-2 py-1 rounded ${winFlavor === 'ps7' ? 'bg-[hsl(var(--brand))] text-white' : 'bg-secondary'}`}
              onClick={() => setWinFlavor('ps7')}
            >7+</button>
            <button
              type="button"
              className={`px-2 py-1 rounded ${winFlavor === 'ps5' ? 'bg-[hsl(var(--brand))] text-white' : 'bg-secondary'}`}
              onClick={() => setWinFlavor('ps5')}
            >5.x</button>
          </div>
        )}
        <div className="mt-2 flex justify-end">{inspectLink}</div>
      </div>
    );
  }

  return (
    <Card className={`border-border bg-card ${className}`}>
      <CardContent className="p-5 sm:p-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
            <Icon className={`h-4 w-4 ${meta.iconColor}`} aria-hidden="true" />
            <span>Install for {meta.title}</span>
          </div>
          {tabs}
        </div>

        <CopyableCommand
          command={command}
          ariaLabel={`Install command for ${meta.title}. Click to copy.`}
        />

        <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
          <div className="text-xs text-muted-foreground">
            {selectedOS === 'windows'
              ? 'Runs in PowerShell. ~30 seconds.'
              : 'Runs in your terminal. ~30 seconds.'}
          </div>
          <div className="flex items-center gap-3">
            {selectedOS === 'windows' && (
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <span className="hidden sm:inline">PowerShell:</span>
                <button
                  type="button"
                  className={`px-2 py-1 rounded ${winFlavor === 'ps7' ? 'bg-[hsl(var(--brand))] text-white' : 'bg-secondary'}`}
                  onClick={() => setWinFlavor('ps7')}
                >7+</button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded ${winFlavor === 'ps5' ? 'bg-[hsl(var(--brand))] text-white' : 'bg-secondary'}`}
                  onClick={() => setWinFlavor('ps5')}
                >5.x</button>
              </div>
            )}
            {inspectLink}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
