"use client";

export default function Contact() {
  return (
    <section id="contact" className="min-h-screen pt-28 sm:pt-32 px-4 sm:px-6 pb-16 section-top">
      <div className="mx-auto max-w-xl">
        <h2 className="section-title text-3xl sm:text-4xl text-center mb-8">
          <span className="section-title-accent">Leave a Message</span>
        </h2>

        <form
          action="https://formspree.io/f/xgvzgroj"  // <-- replace with your Formspree endpoint
          method="POST"
          className="space-y-5 bg-white/5 p-6 rounded-2xl ring-1 ring-white/10 card-glow"
        >
          <div>
            <label className="block mb-2 text-sm text-gray-300">Name</label>
            <input
              type="text"
              name="name"
              required
              className="w-full px-4 py-2 rounded-lg bg-black/30 text-white ring-1 ring-white/10 focus:ring-teal-400 focus:outline-none"
            />
          </div>

          <div>
            <label className="block mb-2 text-sm text-gray-300">Subject</label>
            <input
              type="text"
              name="subject"
              required
              className="w-full px-4 py-2 rounded-lg bg-black/30 text-white ring-1 ring-white/10 focus:ring-teal-400 focus:outline-none"
            />
          </div>

          <div>
            <label className="block mb-2 text-sm text-gray-300">Message</label>
            <textarea
              name="message"
              rows="5"
              required
              className="w-full px-4 py-2 rounded-lg bg-black/30 text-white ring-1 ring-white/10 focus:ring-teal-400 focus:outline-none"
            />
          </div>

          <button
            type="submit"
            className="w-full py-3 rounded-lg bg-gradient-to-r from-teal-400 via-cyan-300 to-indigo-400 text-black font-bold hover:opacity-90 transition"
          >
            Send Message
          </button>
        </form>
      </div>
    </section>
  );
}
