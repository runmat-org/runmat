"use client";

import { useEffect, useState } from "react";
import { Globe, Monitor, Terminal, Download } from "lucide-react";
import { TabButton } from "./TabButton";
import { BrowserTab } from "./BrowserTab";
import { DesktopTab } from "./DesktopTab";
import { CLITab } from "./CLITab";
import { OtherTab } from "./OtherTab";
import { trackEvent } from "@/components/GoogleAnalytics";

type TabOption = "browser" | "desktop" | "cli" | "other";
type Platform = "macos" | "windows" | "linux";

function detectOS(): Platform {
  if (typeof window === "undefined") return "macos";

  const userAgent = window.navigator.userAgent.toLowerCase();

  if (userAgent.includes("mac")) return "macos";
  if (userAgent.includes("win")) return "windows";
  if (userAgent.includes("linux")) return "linux";

  return "macos";
}

export function HeroTabs() {
  const [activeTab, setActiveTab] = useState<TabOption>("browser");
  const [detectedPlatform, setDetectedPlatform] = useState<Platform>("macos");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const platform = detectOS();
    setDetectedPlatform(platform);
    setMounted(true);
  }, []);

  const handleTabChange = (tab: TabOption) => {
    setActiveTab(tab);
    trackEvent("hero_tab_select", "hero_tabs", tab);
  };

  const tabId = (tab: TabOption) => `hero-tab-${tab}`;
  const panelId = (tab: TabOption) => `hero-tabpanel-${tab}`;

  return (
    <div className="w-full">
      <div
        role="tablist"
        className="flex flex-wrap gap-2 border-b border-border"
        aria-label="Access RunMat"
      >
        <TabButton
          label="Browser"
          icon={<Globe className="h-4 w-4" />}
          isActive={activeTab === "browser"}
          onClick={() => handleTabChange("browser")}
          id={tabId("browser")}
          aria-controls={panelId("browser")}
          className="pl-0"
        />
        <TabButton
          label="Desktop"
          icon={<Monitor className="h-4 w-4" />}
          isActive={activeTab === "desktop"}
          onClick={() => handleTabChange("desktop")}
          id={tabId("desktop")}
          aria-controls={panelId("desktop")}
        />
        <TabButton
          label="CLI"
          icon={<Terminal className="h-4 w-4" />}
          isActive={activeTab === "cli"}
          onClick={() => handleTabChange("cli")}
          id={tabId("cli")}
          aria-controls={panelId("cli")}
        />
        <TabButton
          label="Other"
          icon={<Download className="h-4 w-4" />}
          isActive={activeTab === "other"}
          onClick={() => handleTabChange("other")}
          id={tabId("other")}
          aria-controls={panelId("other")}
        />
      </div>

      <div
        role="tabpanel"
        id={panelId(activeTab)}
        aria-labelledby={tabId(activeTab)}
        className="min-h-[120px]"
      >
        {activeTab === "browser" && <BrowserTab />}
        {activeTab === "desktop" && <DesktopTab />}
        {activeTab === "cli" && mounted && <CLITab detectedPlatform={detectedPlatform} />}
        {activeTab === "other" && <OtherTab />}
      </div>
    </div>
  );
}
