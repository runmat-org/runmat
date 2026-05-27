"use client";

import { useEffect, useState } from "react";
import { Globe, Terminal } from "lucide-react";
import { cn } from "@/lib/utils";
import { BrowserTabContent } from "./BrowserTab";
import { CLITabContent } from "./CLITab";

type TabOption = "browser" | "cli";

const tabs: { id: TabOption; label: string; icon: React.ReactNode }[] = [
  { id: "browser", label: "Browser", icon: <Globe className="h-4 w-4" /> },
  { id: "cli", label: "CLI", icon: <Terminal className="h-4 w-4" /> },
];

const tabId = (tab: TabOption) => `getting-started-tab-${tab}`;
const panelId = (tab: TabOption) => `getting-started-panel-${tab}`;
const tabFromHash = () => {
  const hash = window.location.hash.slice(1);
  return tabs.find(({ id }) => hash === tabId(id))?.id;
};

export function GettingStartedTabs() {
  const [activeTab, setActiveTab] = useState<TabOption>("browser");

  useEffect(() => {
    const syncTabFromHash = () => {
      const tab = tabFromHash();
      if (tab) {
        setActiveTab(tab);
      }
    };

    syncTabFromHash();
    window.addEventListener("hashchange", syncTabFromHash);

    return () => {
      window.removeEventListener("hashchange", syncTabFromHash);
    };
  }, []);

  const handleTabChange = (tab: TabOption) => {
    setActiveTab(tab);
    window.history.replaceState(null, "", `#${tabId(tab)}`);
  };

  return (
    <div className="w-full">
      <div
        role="tablist"
        className="flex gap-0 border-b border-border mb-6"
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
              onClick={() => handleTabChange(id)}
              className={cn(
                "relative inline-flex items-center gap-2 px-4 py-2.5 text-sm font-medium transition-colors cursor-pointer -mb-px",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                isActive
                  ? "text-foreground border-b-2 border-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <span aria-hidden="true">{icon}</span>
              <span>{label}</span>
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
      </div>
    </div>
  );
}
