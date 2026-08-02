import type { Metadata } from "next";
import { Instrument_Sans, Fraunces } from "next/font/google";
import "./globals.css";
import Background from "./components/Background";

const instrumentSans = Instrument_Sans({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-sans",
});

const fraunces = Fraunces({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-display",
});

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
    <html lang="en" className={`scroll-smooth ${instrumentSans.variable} ${fraunces.variable}`}>
      <head>
        <script
          defer
          src="https://cloud.umami.is/script.js"
          data-website-id="707eb819-269b-4d44-b441-cf7008915528"
        ></script>
      </head>
      <body className="m-0 antialiased text-[var(--text-main)]">
        <Background />
        {children}
      </body>
    </html>
  );
}
