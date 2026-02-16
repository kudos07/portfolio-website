"use client";

export default function Contact() {
  return (
    <section id="contact" className="pt-20 sm:pt-24 px-4 sm:px-8 lg:px-12 pb-16 section-top">
      <div className="w-full max-w-3xl mx-auto">
        <h2 className="section-title text-3xl sm:text-4xl text-center mb-8">
          <span className="section-title-accent">Leave a Message</span>
        </h2>

        <form
          action="https://formspree.io/f/xgvzgroj"
          method="POST"
          className="space-y-5 flash-card p-6 rounded-2xl"
        >
          <div>
            <label className="block mb-2 text-sm text-slate-700">Name</label>
            <input
              type="text"
              name="name"
              required
              className="w-full px-4 py-2 rounded-lg bg-white/90 text-slate-900 ring-1 ring-slate-200 hover:ring-slate-300 focus:ring-blue-400 focus:outline-none"
            />
          </div>

          <div>
            <label className="block mb-2 text-sm text-slate-700">Subject</label>
            <input
              type="text"
              name="subject"
              required
              className="w-full px-4 py-2 rounded-lg bg-white/90 text-slate-900 ring-1 ring-slate-200 hover:ring-slate-300 focus:ring-blue-400 focus:outline-none"
            />
          </div>

          <div>
            <label className="block mb-2 text-sm text-slate-700">Message</label>
            <textarea
              name="message"
              rows="5"
              required
              className="w-full px-4 py-2 rounded-lg bg-white/90 text-slate-900 ring-1 ring-slate-200 hover:ring-slate-300 focus:ring-blue-400 focus:outline-none"
            />
          </div>

          <button
            type="submit"
            className="w-full py-3 rounded-lg bg-slate-900 text-white font-bold shadow-[0_10px_24px_rgba(15,23,42,0.2)] hover:bg-slate-800 transition"
          >
            Send Message
          </button>
        </form>
      </div>
    </section>
  );
}
