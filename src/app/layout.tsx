// layout.tsx (clean, no navbar)
import type { Metadata } from "next";
import "./globals.css";
import Background from "./components/Background";

export const metadata: Metadata = {
  title: "Saransh Surana | Portfolio",
  description: "Portfolio of Saransh Surana - Data Science, Machine Learning, AI",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <head>
        {/* ✅ Umami Analytics Script */}
        <script
          defer
          src="https://cloud.umami.is/script.js"
          data-website-id="707eb819-269b-4d44-b441-cf7008915528"
        ></script>
      </head>
      <body className="m-0 antialiased bg-gradient-to-br from-gray-900 via-black to-gray-800 text-white">
        <Background />
        {children}
      </body>
    </html>
  );
}
