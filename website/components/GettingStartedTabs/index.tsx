"use client";

import { useState } from "react";
import { Globe, Terminal, BookOpen } from "lucide-react";
import { cn } from "@/lib/utils";
import { BrowserTabContent } from "./BrowserTab";
import { CLITabContent } from "./CLITab";
import { JupyterTabContent } from "./JupyterTab";

type TabOption = "browser" | "cli" | "jupyter";

const tabs: { id: TabOption; label: string; icon: React.ReactNode }[] = [
  { id: "browser", label: "Browser", icon: <Globe className="h-4 w-4" /> },
  { id: "cli", label: "CLI", icon: <Terminal className="h-4 w-4" /> },
  { id: "jupyter", label: "Jupyter", icon: <BookOpen className="h-4 w-4" /> },
];

export function GettingStartedTabs() {
  const [activeTab, setActiveTab] = useState<TabOption>("browser");

  const tabId = (tab: TabOption) => `getting-started-tab-${tab}`;
  const panelId = (tab: TabOption) => `getting-started-panel-${tab}`;

  return (
    <div className="w-full">
      <div
        role="tablist"
        className="flex flex-wrap gap-2 border-b border-border mb-6"
        aria-label="Get started with RunMat"
      >
        {tabs.map(({ id, label, icon }) => {
          const isActive = activeTab === id;
          return (
            <button
              key={id}
              type="button"
              role="tab"
              id={tabId(id)}
              aria-selected={isActive}
              aria-controls={panelId(id)}
              onClick={() => setActiveTab(id)}
              className={cn(
                "relative inline-flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors cursor-pointer",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                "rounded-t-md",
                isActive
                  ? "text-foreground bg-muted/10"
                  : "text-muted-foreground bg-muted/70 hover:text-foreground hover:bg-muted/60"
              )}
            >
              <span aria-hidden="true">{icon}</span>
              <span>{label}</span>
              {isActive && (
                <span
                  className="absolute bottom-0 left-0 right-0 h-[3px] bg-[#a78bfa]"
                  aria-hidden="true"
                />
              )}
              {isActive && (
                <span
                  className="absolute top-0 left-2 right-2 h-px bg-primary/70"
                  aria-hidden="true"
                />
              )}
            </button>
          );
        })}
      </div>

      <div
        role="tabpanel"
        id={panelId(activeTab)}
        aria-labelledby={tabId(activeTab)}
        className="min-h-[200px]"
      >
        {activeTab === "browser" && <BrowserTabContent />}
        {activeTab === "cli" && <CLITabContent />}
        {activeTab === "jupyter" && <JupyterTabContent />}
      </div>
    </div>
  );
}
