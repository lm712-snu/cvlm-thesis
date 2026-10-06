"use client";

import { create } from "zustand";
import type { UploadedImage } from "@/components/cvlm/ImageUploader";

interface AppState {
  /** Currently active module id. */
  activeModule: string;
  setActiveModule: (id: string) => void;

  /** Theme: 'light' | 'dark'. Persisted to localStorage on toggle. */
  theme: "light" | "dark";
  toggleTheme: () => void;
  setTheme: (t: "light" | "dark") => void;

  /** Shared uploaded image — used by Modules 1, 2, 3, 18, 21. */
  sharedImage: UploadedImage | null;
  setSharedImage: (img: UploadedImage | null) => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  activeModule: "home",
  setActiveModule: (id) => set({ activeModule: id }),

  theme: "light",
  toggleTheme: () => {
    const next = get().theme === "light" ? "dark" : "light";
    set({ theme: next });
    if (typeof document !== "undefined") {
      document.documentElement.classList.toggle("dark", next === "dark");
      try {
        localStorage.setItem("cvlm-theme", next);
      } catch {
        /* ignore */
      }
    }
  },
  setTheme: (t) => {
    set({ theme: t });
    if (typeof document !== "undefined") {
      document.documentElement.classList.toggle("dark", t === "dark");
      try {
        localStorage.setItem("cvlm-theme", t);
      } catch {
        /* ignore */
      }
    }
  },

  sharedImage: null,
  setSharedImage: (img) => set({ sharedImage: img }),
}));
