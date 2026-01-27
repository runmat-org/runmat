"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TryInBrowserButton } from "@/components/TryInBrowserButton";
import MatlabInlineCodeBlock from "@/components/MatlabInlineCodeBlock";
import Link from "next/link";

const DEMO_CODE = `a = 0:pi/100:2*pi;
b = sin(a);
c = cos(a);
g = b.^2 + c.^2;
sum_g = sum(g);
max_g = max(g);
disp([ ...
   'Sum of g: ', num2str(sum_g), ' | ', ...
   'Max of g: ', num2str(max_g) ...
]);`;

export function BrowserTabContent() {
  return (
    <div className="space-y-6">
      <p className="text-muted-foreground">
        RunMat in the browser gives you a full IDE—editor, file explorer, console, and live plots—all running locally. No installation. Visit{" "}
        <Link href="/sandbox" className="text-primary underline underline-offset-4" target="_blank" rel="noopener noreferrer">
          runmat.org/sandbox
        </Link>{" "}
        to start. Works in Chrome, Edge, Firefox, and Safari. For GPU acceleration, use a browser with WebGPU (Chrome 113+, Edge 113+, Safari 18+, Firefox 139+).
      </p>

      <div>
        <h3 className="text-lg font-semibold mb-2 text-foreground">The interface</h3>
        <p className="text-muted-foreground text-sm mb-2">
          Three main areas: <strong>Sidebar</strong> (file tree, + to add files), <strong>Editor</strong> (code, Cmd/Ctrl+S to save), <strong>Runtime Panel</strong> (Run, Figures, Console, Variables). Panels are resizable.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center text-base">
            <span className="bg-blue-100 dark:bg-blue-900/30 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold text-blue-600 mr-3">1</span>
            Run the demo
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-muted-foreground text-sm">
            When you open the sandbox, <code className="rounded bg-muted px-1 py-0.5">demo.m</code> is already loaded:
          </p>
          <div className="relative group">
            <div className="absolute top-3 right-3 z-10">
              <TryInBrowserButton
                code={DEMO_CODE}
                size="sm"
                className="bg-code-surface/80 backdrop-blur-sm"
              />
            </div>
            <div className="rounded-md overflow-hidden">
              <MatlabInlineCodeBlock code={DEMO_CODE} preClassName="rounded-md p-4 !my-0 language-matlab" />
            </div>
          </div>
          <p className="text-muted-foreground text-sm">
            Click the purple <strong>▶ Run demo.m</strong> button in the Runtime Panel. The Console shows output; the Variables tab shows workspace variables with types, shapes, and CPU/GPU residency.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center text-base">
            <span className="bg-orange-100 dark:bg-orange-900/30 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold text-orange-600 mr-3">2</span>
            Edit and add a plot
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-muted-foreground text-sm">
            Add these lines to <code className="rounded bg-muted px-1 py-0.5">demo.m</code> and run again:
          </p>
          <div className="rounded-md overflow-hidden">
            <MatlabInlineCodeBlock code={'plot(a, b);\ntitle("Sine Wave");'} preClassName="rounded-md p-4 !my-0 language-matlab" />
          </div>
          <p className="text-muted-foreground text-sm">
            A new Figure tab appears in the Runtime Panel. Use <strong>Ctrl+Enter</strong> (Windows/Linux) or <strong>Cmd+Enter</strong> (macOS) to run without clicking the button.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center text-base">
            <span className="bg-green-100 dark:bg-green-900/30 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold text-green-600 mr-3">3</span>
            Create your own script
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-muted-foreground text-sm">
            Click <strong>+</strong> in the sidebar → <strong>Create file (.m)</strong> → name it (e.g. <code className="rounded bg-muted px-1 py-0.5">my_script.m</code>). The editor supports standard MATLAB syntax; large arrays get automatic GPU acceleration via WebGPU.
          </p>
        </CardContent>
      </Card>

      <div className="rounded-lg border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-900/20 p-4 text-sm text-muted-foreground">
        <strong className="text-foreground">Sandbox storage:</strong> Files live in your browser tab. No account required; your code never leaves your machine. Files are cleared when you close or refresh the tab—copy code out or use the CLI for persistent work.
      </div>
    </div>
  );
}
