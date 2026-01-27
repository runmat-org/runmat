"use client";

import { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface TabButtonProps {
  label: string;
  icon?: ReactNode;
  isActive: boolean;
  onClick: () => void;
  className?: string;
  id?: string;
  "aria-controls"?: string;
}

export function TabButton({ label, icon, isActive, onClick, className, id, "aria-controls": ariaControls }: TabButtonProps) {
  return (
    <button
      type="button"
      role="tab"
      id={id}
      aria-selected={isActive}
      aria-controls={ariaControls}
      onClick={onClick}
      className={cn(
        "pr-4 pl-4 py-2 text-sm transition-all relative",
        "flex items-center gap-2",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        isActive
          ? "text-foreground font-semibold"
          : "text-gray-600 dark:text-gray-700 font-normal hover:text-gray-500 dark:hover:text-gray-600",
        className
      )}
    >
      {icon && (
        <span
          className={cn(
            "flex-shrink-0 transition-colors",
            isActive ? "text-foreground" : "text-gray-600 dark:text-gray-700"
          )}
          aria-hidden="true"
        >
          {icon}
        </span>
      )}
      <span>{label}</span>
      {isActive && (
        <span
          className="absolute -bottom-[1px] left-0 right-0 h-[3px] bg-primary z-10 rounded-t"
          aria-hidden="true"
        />
      )}
    </button>
  );
}
