import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Stars, Sphere } from "@react-three/drei";
import { useRef, useMemo } from "react";
import * as THREE from "three";

function Earth({ intensity = 1 }: { intensity?: number }) {
  const ref = useRef<THREE.Mesh>(null!);
  useFrame((_, dt) => { ref.current.rotation.y += dt * 0.08; });
  return (
    <Sphere ref={ref} args={[1.4, 64, 64]}>
      <meshStandardMaterial
        color={"#0e2a1f"}
        emissive={"#1bd96b"}
        emissiveIntensity={0.18 * intensity}
        wireframe
      />
    </Sphere>
  );
}

function Rain({ count = 600 }: { count?: number }) {
  const ref = useRef<THREE.Points>(null!);
  const positions = useMemo(() => {
    const a = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const r = 1.6 + Math.random() * 1.4;
      const t = Math.random() * Math.PI * 2;
      const p = Math.acos(2 * Math.random() - 1);
      a[i * 3] = r * Math.sin(p) * Math.cos(t);
      a[i * 3 + 1] = r * Math.cos(p);
      a[i * 3 + 2] = r * Math.sin(p) * Math.sin(t);
    }
    return a;
  }, [count]);
  useFrame((_, dt) => { ref.current.rotation.y -= dt * 0.05; });
  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.022} color={"#7af5b2"} transparent opacity={0.85} />
    </points>
  );
}

export function EarthHero({ intensity = 1 }: { intensity?: number }) {
  return (
    <Canvas camera={{ position: [0, 0, 4.2], fov: 50 }} dpr={[1, 2]}>
      <ambientLight intensity={0.4} />
      <pointLight position={[3, 3, 3]} intensity={1.2} color={"#7af5b2"} />
      <pointLight position={[-3, -2, -2]} intensity={0.6} color={"#1bd96b"} />
      <Stars radius={40} depth={40} count={1500} factor={3} fade speed={0.6} />
      <Earth intensity={intensity} />
      <Rain count={500 + Math.floor(intensity * 400)} />
      <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={0.4} />
    </Canvas>
  );
}
