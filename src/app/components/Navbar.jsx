"use client";
import { Link as ScrollLink } from "react-scroll";

export default function Navbar() {
  return (
    <nav className="flex justify-end space-x-8 p-6 bg-black bg-opacity-70 text-white fixed top-0 left-0 right-0 shadow-md z-10">
<ScrollLink to="home" smooth duration={600} className="cursor-pointer hover:text-teal-300">
  Home
</ScrollLink>
<ScrollLink to="experience" smooth duration={600} offset={-80} className="cursor-pointer hover:text-teal-300">
  Experience
</ScrollLink>
<ScrollLink to="projects" smooth duration={600} offset={-80} className="cursor-pointer hover:text-teal-300">
  Projects
</ScrollLink>
<ScrollLink to="skills" smooth duration={600} offset={-80} className="cursor-pointer hover:text-teal-300">
  Skills
</ScrollLink>
<ScrollLink to="contact" smooth duration={600} offset={-80} className="cursor-pointer hover:text-teal-300">
  Leave a Message
</ScrollLink>

    </nav>
  );
}
