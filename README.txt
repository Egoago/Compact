Az egyszerűség kedvéért az egész projektet beadom, mivel tartalmaz beállításokat a megfelelő fordításhoz.
A futtatáshoz telepíteni kell a CUDA toolkit-et, mely elérhető ezen a linken: https://developer.nvidia.com/cuda-downloads

A feladatot generikus módon oldottam meg, így bármilyen adathalmazra illetve tetszőleges predikátor függvénnyel futtatható.
Az implementációt a Compact(compact.cu/cuh) osztály tartalmazza.
A main.cu fájl példát mutat ennek használatára.

Rendszer:
Visual Studio 2019
CUDA Toolkit 11.3
NVIDIA GTX 960M
CUDA Compute Capability 5.0
Windows 10