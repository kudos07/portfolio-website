export default function Background() {
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-50 overflow-hidden">
      <div className="absolute inset-0 bg-canvas" />
      <div className="absolute -top-40 left-[-14rem] h-[32rem] w-[32rem] rounded-full bg-accent-soft blur-3xl opacity-90" />
      <div className="absolute top-[20%] right-[-16rem] h-[34rem] w-[34rem] rounded-full bg-accent-muted blur-3xl opacity-85" />
      <div className="absolute bottom-[-10rem] left-[30%] h-[28rem] w-[28rem] rounded-full bg-[radial-gradient(circle_at_center,rgba(59,130,246,0.10),transparent_70%)] blur-3xl" />
      <div className="absolute inset-0 bg-noise opacity-20" />
    </div>
  );
}
