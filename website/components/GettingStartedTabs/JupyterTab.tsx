"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircle } from "lucide-react";

export function JupyterTabContent() {
  return (
    <div className="space-y-6">
      <div id="jupyter-notebook-integration" className="scroll-mt-24" />
      <p className="text-muted-foreground">
        Use RunMat as a Jupyter kernel for interactive notebooks with full MATLAB syntax and automatic GPU acceleration. Requires RunMat CLI installed first.
      </p>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center text-base">
            <span className="bg-blue-100 dark:bg-blue-900/30 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold text-blue-600 mr-3">1</span>
            Install RunMat as a Jupyter kernel
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-muted-foreground text-sm">
            One-time setup that works with existing Jupyter installations:
          </p>
          <div className="bg-gray-900 rounded-md p-4 font-mono text-sm text-white overflow-x-auto">
            <div>runmat --install-kernel</div>
            <div className="text-green-400 mt-1">RunMat Jupyter kernel installed successfully!</div>
            <div className="text-green-400">Kernel directory: ~/.local/share/jupyter/kernels/runmat</div>
          </div>
          <div className="flex items-center space-x-2 text-sm text-green-600 dark:text-green-400">
            <CheckCircle className="h-4 w-4 flex-shrink-0" />
            <span>One-time setup that works with existing Jupyter installations</span>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center text-base">
            <span className="bg-orange-100 dark:bg-orange-900/30 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold text-orange-600 mr-3">2</span>
            Start Jupyter and select RunMat
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-muted-foreground text-sm">
            Launch Jupyter Notebook or Jupyter Lab, then choose &quot;RunMat&quot; when creating a new notebook:
          </p>
          <div className="bg-gray-900 rounded-md p-4 font-mono text-sm text-white overflow-x-auto">
            <div className="text-gray-400"># Start Jupyter Notebook</div>
            <div>jupyter notebook</div>
            <div className="mt-2 text-gray-400"># Or Jupyter Lab</div>
            <div>jupyter lab</div>
            <div className="mt-2 text-gray-400"># Then select &quot;RunMat&quot; when creating a new notebook</div>
          </div>
          <div className="flex items-center space-x-2 text-sm text-blue-600 dark:text-blue-400">
            <CheckCircle className="h-4 w-4 flex-shrink-0" />
            <span>Full MATLAB syntax support with automatic GPU acceleration</span>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center text-base">
            <span className="bg-green-100 dark:bg-green-900/30 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold text-green-600 mr-3">3</span>
            Verify installation
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-muted-foreground text-sm">
            Check that the RunMat kernel is listed:
          </p>
          <div className="bg-gray-900 rounded-md p-4 font-mono text-sm text-white overflow-x-auto">
            <div>jupyter kernelspec list</div>
            <div className="text-gray-400 mt-1">Available kernels:</div>
            <div className="text-gray-400">  python3    /usr/local/share/jupyter/kernels/python3</div>
            <div className="text-green-400">  runmat    ~/.local/share/jupyter/kernels/runmat</div>
          </div>
          <p className="text-muted-foreground text-sm">
            If RunMat doesn&apos;t appear, ensure Jupyter is installed and run the install command again.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
