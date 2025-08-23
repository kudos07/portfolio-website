// src/app/ClientProviders.tsx
"use client";

import { useEffect } from "react";
import smoothscroll from "smoothscroll-polyfill";

// This file is for any client-only code that needs hooks
export default function ClientProviders({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    smoothscroll.polyfill();  // 👈 makes scrolling smooth on all browsers
  }, []);

  return <>{children}</>;
}
