export default function Background() {
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-50 overflow-hidden">
      <div className="absolute inset-0 bg-canvas" />
      <div className="absolute -top-32 left-[-10rem] h-[28rem] w-[28rem] rounded-full bg-accent-soft blur-3xl opacity-70" />
      <div className="absolute bottom-[-8rem] right-[-12rem] h-[26rem] w-[26rem] rounded-full bg-accent-muted blur-3xl opacity-80" />
      <div className="absolute inset-0 bg-noise opacity-[0.35]" />
    </div>
  );
}
