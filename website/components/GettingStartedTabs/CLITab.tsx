"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Play, Download } from "lucide-react";
import Link from "next/link";
import { OSInstallCommand } from "@/components/OSInstallCommand";

export function CLITabContent() {
  return (
    <div className="space-y-6">
      <p className="text-muted-foreground">
        Install RunMat on your machine for the terminal REPL, local scripts, and full native GPU access. Use the installer below or see{" "}
        <Link href="/download" className="text-primary underline underline-offset-4">More installation options</Link>.
      </p>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Install</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <OSInstallCommand />
          <Button asChild size="sm" variant="outline">
            <Link href="/download">
              <Download className="mr-2 h-4 w-4" />
              More Installation Options
            </Link>
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center text-base">
            <Play className="h-4 w-4 mr-2 text-green-600" />
            Run the REPL and scripts
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-muted-foreground text-sm">
            Start the interactive REPL with <code className="rounded bg-muted px-1 py-0.5">runmat</code>. Run a script with <code className="rounded bg-muted px-1 py-0.5">runmat script.m</code>.
          </p>
          <div className="bg-gray-900 rounded-md p-4 font-mono text-sm text-white overflow-x-auto">
            <div className="text-gray-400">$ runmat</div>
            <div className="text-green-400 mt-1">RunMat v0.0.1 by Dystr …</div>
            <div className="text-green-400">Type &apos;help&apos; for help, &apos;exit&apos; to quit</div>
            <div className="mt-2 text-white">runmat&gt;</div>
            <div className="mt-2 text-blue-400">runmat&gt;</div> <span>A = [1, 2, 3; 4, 5, 6]; B = A * 2</span>
            <div className="text-gray-400 mt-1">ans = [2 4 6; 8 10 12]</div>
            <div className="mt-2 text-gray-400"># Or run a file:</div>
            <div className="text-gray-400">$ runmat script.m</div>
          </div>
          <p className="text-muted-foreground text-sm">
            RunMat fuses operations and keeps data on the GPU automatically. Most MATLAB/Octave scripts run with few or no changes—see the{" "}
            <Link href="/docs/language-coverage" className="text-primary underline underline-offset-4">compatibility guide</Link>.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
