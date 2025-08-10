"use client";

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Terminal, Monitor, Copy, Check } from "lucide-react";
import { trackEvent } from "@/components/GoogleAnalytics";

type OS = 'windows' | 'mac' | 'linux' | 'unknown';

function CopyableCommand({ command, bgColor }: { command: string; bgColor: string }) {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);

      // Track the copy event
      trackEvent('copy_install_command', 'installation', command.includes('curl') ? 'unix' : 'windows');
    } catch {
      // Fallback for browsers that don't support clipboard API
      const textArea = document.createElement('textarea');
      textArea.value = command;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);

      // Track the copy event
      trackEvent('copy_install_command', 'installation', command.includes('curl') ? 'unix' : 'windows');
    }
  };

  return (
    <div className={`${bgColor} rounded-md p-4 font-mono text-sm text-white relative group cursor-pointer`} onClick={copyToClipboard}>
      <div className="flex items-center justify-center min-w-0">
        <span className="select-all whitespace-nowrap overflow-hidden text-ellipsis text-center max-w-full">{command}</span>
        <button
          className="ml-3 p-1 opacity-70 hover:opacity-100 transition-opacity bg-white/10 hover:bg-white/20 rounded flex-shrink-0"
          onClick={(e) => {
            e.stopPropagation();
            copyToClipboard();
          }}
        >
          {copied ? (
            <Check className="h-4 w-4 text-green-400" />
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
  );
}

function detectOS(): OS {
  if (typeof window === 'undefined') return 'unknown';
  
  const userAgent = window.navigator.userAgent.toLowerCase();
  
  if (userAgent.includes('win')) return 'windows';
  if (userAgent.includes('mac')) return 'mac';
  if (userAgent.includes('linux')) return 'linux';
  
  return 'unknown';
}

interface OSInstallCommandProps {
  variant?: 'compact' | 'full';
  className?: string;
}

export function OSInstallCommand({ variant = 'full', className = '' }: OSInstallCommandProps) {
  const [selectedOS, setSelectedOS] = useState<OS>('unknown');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setSelectedOS(detectOS());
    setMounted(true);
  }, []);

  if (!mounted) {
    // Show both commands during SSR/hydration
    return (
      <div className={`grid grid-cols-1 lg:grid-cols-2 gap-6 ${className}`}>
        <Card>
          <CardHeader className="text-center pb-3">
            <Terminal className="h-8 w-8 mx-auto mb-2 text-green-600" />
            <CardTitle className="text-lg">Linux & macOS</CardTitle>
          </CardHeader>
          <CardContent>
            <CopyableCommand 
              command="curl -fsSL https://runmat.org/install.sh | sh"
              bgColor="bg-gray-900"
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="text-center pb-3">
            <Monitor className="h-8 w-8 mx-auto mb-2 text-blue-600" />
            <CardTitle className="text-lg">Windows</CardTitle>
          </CardHeader>
          <CardContent>
            <CopyableCommand 
              command="iwr https://runmat.org/install.ps1 | iex"
              bgColor="bg-blue-900"
            />
          </CardContent>
        </Card>
      </div>
    );
  }

  // Show OS-specific command after hydration
  const getOSInfo = () => {
    switch (selectedOS) {
      case 'windows':
        return {
          title: 'Windows',
          icon: Monitor,
          command: 'iwr https://runmat.org/install.ps1 | iex',
          bgColor: 'bg-blue-900',
          iconColor: 'text-blue-600',
          description: 'PowerShell 3.0+ required'
        };
      case 'mac':
        return {
          title: 'macOS',
          icon: Terminal,
          command: 'curl -fsSL https://runmat.org/install.sh | sh',
          bgColor: 'bg-gray-900',
          iconColor: 'text-green-600',
          description: null
        };
      case 'linux':
        return {
          title: 'Linux',
          icon: Terminal,
          command: 'curl -fsSL https://runmat.org/install.sh | sh',
          bgColor: 'bg-gray-900',
          iconColor: 'text-green-600',
          description: null
        };
      default:
        return {
          title: 'Your System',
          icon: Terminal,
          command: 'curl -fsSL https://runmat.org/install.sh | sh',
          bgColor: 'bg-gray-900',
          iconColor: 'text-green-600',
          description: null
        };
    }
  };

  const osInfo = getOSInfo();
  const Icon = osInfo.icon;

  if (variant === 'compact') {
    return (
      <div className={className}>
        <div className="text-center mb-4">
          <Icon className={`h-6 w-6 mx-auto mb-2 ${osInfo.iconColor}`} />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Install for {osInfo.title}
          </h3>
        </div>
        <CopyableCommand 
          command={osInfo.command}
          bgColor={osInfo.bgColor}
        />
        {osInfo.description && (
          <p className="text-sm text-gray-600 dark:text-gray-300 text-center mt-2">
            {osInfo.description}
          </p>
        )}
      </div>
    );
  }

  return (
    <Card className={className}>
      <CardHeader className="text-center pb-3">
        <Icon className={`h-8 w-8 mx-auto mb-2 ${osInfo.iconColor}`} />
        <CardTitle className="text-lg">Install for {osInfo.title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className={osInfo.description ? 'mb-4' : ''}>
          <CopyableCommand 
            command={osInfo.command}
            bgColor={osInfo.bgColor}
          />
        </div>
        {osInfo.description && (
          <p className="text-sm text-gray-600 dark:text-gray-300 text-center">
            {osInfo.description}
          </p>
        )}
        
        {selectedOS !== 'unknown' && (
          <div className="mt-4 pt-4">
            <p className="text-xs text-gray-500 dark:text-gray-400 text-center mb-3">
              Not {getOSInfo().title}? Pick your platform:
            </p>
            <div className="flex justify-center gap-2">
              {selectedOS !== 'windows' && (
                <button
                  onClick={() => {
                    setSelectedOS('windows');
                    trackEvent('select_os', 'installation', 'windows');
                  }}
                  className="px-3 py-1 text-xs bg-gray-200 hover:bg-gray-300 text-gray-800 dark:bg-gray-700 dark:hover:bg-gray-600 dark:text-gray-200 rounded-md transition-colors"
                >
                  Windows
                </button>
              )}
              {selectedOS !== 'mac' && (
                <button
                  onClick={() => {
                    setSelectedOS('mac');
                    trackEvent('select_os', 'installation', 'mac');
                  }}
                  className="px-3 py-1 text-xs bg-gray-200 hover:bg-gray-300 text-gray-800 dark:bg-gray-700 dark:hover:bg-gray-600 dark:text-gray-200 rounded-md transition-colors"
                >
                  macOS
                </button>
              )}
              {selectedOS !== 'linux' && (
                <button
                  onClick={() => {
                    setSelectedOS('linux');
                    trackEvent('select_os', 'installation', 'linux');
                  }}
                  className="px-3 py-1 text-xs bg-gray-200 hover:bg-gray-300 text-gray-800 dark:bg-gray-700 dark:hover:bg-gray-600 dark:text-gray-200 rounded-md transition-colors"
                >
                  Linux
                </button>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}