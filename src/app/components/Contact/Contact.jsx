"use client";

export default function Contact() {
  return (
    <section id="contact" className="section-top px-4 pb-20 pt-20 sm:px-8 sm:pt-24 lg:px-12">
      <div className="mx-auto w-full max-w-xl">
        <h2 className="section-title reveal-title mb-8 text-center text-3xl sm:text-4xl">
          <span className="section-title-accent">Leave a Message</span>
        </h2>

        <form
          action="https://formspree.io/f/xgvzgroj"
          method="POST"
          className="space-y-5 border border-[var(--line-soft)] bg-[var(--bg-1)] p-6"
        >
          <div>
            <label className="mb-2 block text-sm text-[var(--text-soft)]">Name</label>
            <input
              type="text"
              name="name"
              required
              className="w-full rounded-md border border-[var(--line-soft)] bg-[var(--bg-0)] px-4 py-2.5 text-[var(--text-main)] focus:border-[var(--accent)] focus:outline-none"
            />
          </div>

          <div>
            <label className="mb-2 block text-sm text-[var(--text-soft)]">Subject</label>
            <input
              type="text"
              name="subject"
              required
              className="w-full rounded-md border border-[var(--line-soft)] bg-[var(--bg-0)] px-4 py-2.5 text-[var(--text-main)] focus:border-[var(--accent)] focus:outline-none"
            />
          </div>

          <div>
            <label className="mb-2 block text-sm text-[var(--text-soft)]">Message</label>
            <textarea
              name="message"
              rows={5}
              required
              className="w-full rounded-md border border-[var(--line-soft)] bg-[var(--bg-0)] px-4 py-2.5 text-[var(--text-main)] focus:border-[var(--accent)] focus:outline-none"
            />
          </div>

          <button
            type="submit"
            className="w-full rounded-md bg-[var(--text-main)] py-3 font-semibold text-white transition hover:bg-[var(--accent)]"
          >
            Send Message
          </button>
        </form>
      </div>
    </section>
  );
}
