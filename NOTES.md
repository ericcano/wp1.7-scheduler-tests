# Development notes

Sequential version of traccc algorithm is working properly but crashing with CUDA. It may or may not crash
depending on whether it is run in debug or release mode.

The promise type of the coroutine is parametrized with respect to co_yield and co_return parameters. Promise<void, void>
means there is no co_yield and empty co_return. We use Promise<StatusCode, StatusCode>.

There is TracccAlgorithm and TracccAlgs files. TracccAlgs is supposed to be a replacement for TracccAlgorithm but
using EventStore. At the moment, EventStore only records unique pointer objects. However, some types are not
copyable or they are expensive to copy. Therefore, another record function should be added that stores raw pointer objects.
In particular, it happens inside execute function in TracccAlgs.cpp.

Algorithm dependencies are set up using protected member functions addDependency and addProduct of AlgorithmBase base class.

## Building patatrack standalone

Get [pixeltrack-standalone](https://github.com/cms-patatrack/pixeltrack-standalone) project:

```sh
git clone git@github.com:cms-patatrack/pixeltrack-standalone.git
cd pixeltrack-standalone
```

Patatrack is using Makefile. Edit following lines to point to local packages:

```makefile
CUDA_BASE := /usr/local/cuda
TBB_BASE      := $(ONEAPI_BASE)/tbb/latest
TBB_LIBDIR    := $(TBB_BASE)/lib
BOOST_BASE := /usr
```

Then, build:

```sh
make cuda
```

## Building on AlmaLinux8
 AlmaLinux 8 has an old default compiler, so we want to use a newer one. GCC 14 is easily available in the package `gcc-toolset-14`. In turn, the host compiler should be indicated to `nvcc`.

Also the default Boost version (1.68) is too old and wee need the newer boost1.78-devel package, which is not in the include path. (This is now reflected in the CMake file as well).

 The CMake comfiguration hence becomes:
 ```bash
 Patatrack_ROOT=/home/cano/NGT/pixeltrack-standalone/ CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++ CUDACXX="/usr/local/cuda-12.9/bin/nvcc -ccbin ${CXX}" cmake -S wp1.7-scheduler-tests/ -B build/wp1.7-scheduler-test/ -DBoost_INCLUDE_DIR=/usr/include/boost1.78/
 ```
## Building on Ubuntu

(As of 25th september 2025, on Ubuntu LTS 24.04)

Configuring required a few extra packages on top of the vanilla `libboost-dev`, I needed to install:

```
sudo apt install libboost-program-options-dev libboost-filesystem
```
Leading to:
```
$ dpkg -l *boost* | grep ^ii
ii  libboost-atomic1.83-dev:amd64          1.83.0-2.1ubuntu3.1 amd64        atomic data types, operations, and memory ordering constraints
ii  libboost-atomic1.83.0:amd64            1.83.0-2.1ubuntu3.1 amd64        atomic data types, operations, and memory ordering constraints
ii  libboost-dev:amd64                     1.83.0.1ubuntu2     amd64        Boost C++ Libraries development files (default version)
ii  libboost-filesystem1.83-dev:amd64      1.83.0-2.1ubuntu3.1 amd64        filesystem operations (portable paths, iteration over directories, etc) in C++
ii  libboost-filesystem1.83.0:amd64        1.83.0-2.1ubuntu3.1 amd64        filesystem operations (portable paths, iteration over directories, etc) in C++
ii  libboost-iostreams1.83.0:amd64         1.83.0-2.1ubuntu3.1 amd64        Boost.Iostreams Library
ii  libboost-program-options-dev:amd64     1.83.0.1ubuntu2     amd64        program options library for C++ (default version)
ii  libboost-program-options1.83-dev:amd64 1.83.0-2.1ubuntu3.1 amd64        program options library for C++
ii  libboost-program-options1.83.0:amd64   1.83.0-2.1ubuntu3.1 amd64        program options library for C++
ii  libboost-system1.83-dev:amd64          1.83.0-2.1ubuntu3.1 amd64        Operating system (e.g. diagnostics support) library
ii  libboost-system1.83.0:amd64            1.83.0-2.1ubuntu3.1 amd64        Operating system (e.g. diagnostics support) library
ii  libboost-thread1.83.0:amd64            1.83.0-2.1ubuntu3.1 amd64        portable C++ multi-threading
ii  libboost1.83-dev:amd64                 1.83.0-2.1ubuntu3.1 amd64        Boost C++ Libraries development files
```
Confguring also required to install CUDA and to indicate the location of `nvcc`

```
cmake -B ../build/wp1.7-scheduler-tests/ -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc  -DCMAKE_BUILD_TYPE=Debug .
```

When compiling in debug version, cuda chokes on comments with non-ascii caracter. The workaround is to:
```
cmake --build ../build/wp1.7-scheduler-tests/ -j 10 -- -k
```
to compile as far as possible.

The error looks like:
```
ptxas fatal   : Unexpected non-ASCII character encountered on line 419728
ptxas error   : Debug information not found in presence of .target debug
ptxas error   : Debug information not found in presence of .target debug
ptxas error   : Debug information not found in presence of .target debug
ptxas error   : Debug information not found in presence of .target debug
ptxas fatal   : Ptx assembly aborted due to errors
```
 and we can find the reason here, because `traccc` compiles with the `--keep` option which keeps all intermediate files:
```
$ find ../build/ -name *.ptx | grep -Pn '[^[:ascii:]]' `cat`
../build/wp1.7-scheduler-tests/_deps/traccc-build/device/cuda/finding_algorithm.ptx:437093:///home/cano/NGT/freshbuild/build/wp1.7-scheduler-tests/_deps/detray-src/core/include/detray/materials/detail/relativistic_quantities.hpp:54           // beta²/q² = (p/E)²/q² = p²/(q²m² + q²p²) = 1/(q² + (m²(q/p)²)
../build/wp1.7-scheduler-tests/_deps/traccc-build/device/cuda/finding_algorithm.ptx:437094:///home/cano/NGT/freshbuild/build/wp1.7-scheduler-tests/_deps/detray-src/core/include/detray/materials/detail/relativistic_quantities.hpp:55           // q²/beta² = q² + m²(q/p)²
../build/wp1.7-scheduler-tests/_deps/traccc-build/device/cuda/finding_algorithm.ptx:437239:///home/cano/NGT/freshbuild/build/wp1.7-scheduler-tests/_deps/detray-src/core/include/detray/materials/detail/relativistic_quantities.hpp:65         // beta² = p²/E² = p²/(m² + p²) = 1/(1 + (m/p)²)
../build/wp1.7-scheduler-tests/_deps/traccc-build/device/cuda/finding_algorithm.ptx:437263:///home/cano/NGT/freshbuild/build/wp1.7-scheduler-tests/_deps/detray-src/core/include/detray/materials/detail/relativistic_quantities.hpp:68         // beta*gamma = (p/sqrt(m² + p²))*(sqrt(m² + p²)/m) = p/m
../build/wp1.7-scheduler-tests/_deps/traccc-build/device/cuda/finding_algorithm.ptx:437268:///home/cano/NGT/freshbuild/build/wp1.7-scheduler-tests/_deps/detray-src/core/include/detray/materials/detail/relativistic_quantities.hpp:70         // gamma = sqrt(m² + p²)/m = sqrt(1 + (p/m)²)
../build/wp1.7-scheduler-tests/_deps/traccc-build/device/cuda/finding_algorithm.ptx:441221:///home/cano/NGT/freshbuild/build/wp1.7-scheduler-tests/_deps/detray-src/core/include/detray/materials/interaction.hpp:225         //           / q² = (1/p)^4 * (q/beta)² * var(E)
../build/wp1.7-scheduler-tests/_deps/traccc-build/device/cuda/finding_algorithm.ptx:447588:///home/cano/NGT/freshbuild/build/wp1.7-scheduler-tests/_deps/detray-src/core/include/detray/materials/interaction.hpp:261         // RPP2018 eq. 33.15 (treats beta and q² consistenly)
../build/wp1.7-scheduler-tests/_deps/traccc-build/device/cuda/finding_algorithm.ptx:447605:///home/cano/NGT/freshbuild/build/wp1.7-scheduler-tests/_deps/detray-src/core/include/detray/materials/interaction.hpp:263         // log((x/X0) * (q²/beta²)) = log((sqrt(x/X0) * (q/beta))²)
../build/wp1.7-scheduler-tests/_deps/traccc-build/device/cuda/finding_algorithm.ptx:544902:///home/cano/NGT/freshbuild/build/wp1.7-scheduler-tests/_deps/detray-src/core/include/detray/materials/detail/relativistic_quantities.hpp:84     /// @return 2 * mass * (beta * gamma)² mass term.
../build/wp1.7-scheduler-tests/_deps/traccc-build/device/cuda/fitting_algorithm.ptx:419728:///home/cano/NGT/freshbuild/build/wp1.7-scheduler-tests/_deps/detray-src/core/include/detray/materials/detail/relativistic_quantities.hpp:54           // beta²/q² = (p/E)²/q² = p²/(q²m² + q²p²) = 1/(q² + (m²(q/p)²)
../build/wp1.7-scheduler-tests/_deps/traccc-build/device/cuda/fitting_algorithm.ptx:419729:///home/cano/NGT/freshbuild/build/wp1.7-scheduler-tests/_deps/detray-src/core/include/detray/materials/detail/relativistic_quantities.hpp:55           // q²/beta² = q² + m²(q/p)²
../build/wp1.7-scheduler-tests/_deps/traccc-build/device/cuda/fitting_algorithm.ptx:419874:///home/cano/NGT/freshbuild/build/wp1.7-scheduler-tests/_deps/detray-src/core/include/detray/materials/detail/relativistic_quantities.hpp:65         // beta² = p²/E² = p²/(m² + p²) = 1/(1 + (m/p)²)
../build/wp1.7-scheduler-tests/_deps/traccc-build/device/cuda/fitting_algorithm.ptx:419898:///home/cano/NGT/freshbuild/build/wp1.7-scheduler-tests/_deps/detray-src/core/include/detray/materials/detail/relativistic_quantities.hpp:68         // beta*gamma = (p/sqrt(m² + p²))*(sqrt(m² + p²)/m) = p/m
../build/wp1.7-scheduler-tests/_deps/traccc-build/device/cuda/fitting_algorithm.ptx:419903:///home/cano/NGT/freshbuild/build/wp1.7-scheduler-tests/_deps/detray-src/core/include/detray/materials/detail/relativistic_quantities.hpp:70         // gamma = sqrt(m² + p²)/m = sqrt(1 + (p/m)²)
../build/wp1.7-scheduler-tests/_deps/traccc-build/device/cuda/fitting_algorithm.ptx:458285:///home/cano/NGT/freshbuild/build/wp1.7-scheduler-tests/_deps/detray-src/core/include/detray/materials/interaction.hpp:225         //           / q² = (1/p)^4 * (q/beta)² * var(E)
../build/wp1.7-scheduler-tests/_deps/traccc-build/device/cuda/fitting_algorithm.ptx:486904:///home/cano/NGT/freshbuild/build/wp1.7-scheduler-tests/_deps/detray-src/core/include/detray/materials/interaction.hpp:261         // RPP2018 eq. 33.15 (treats beta and q² consistenly)
../build/wp1.7-scheduler-tests/_deps/traccc-build/device/cuda/fitting_algorithm.ptx:486921:///home/cano/NGT/freshbuild/build/wp1.7-scheduler-tests/_deps/detray-src/core/include/detray/materials/interaction.hpp:263         // log((x/X0) * (q²/beta²)) = log((sqrt(x/X0) * (q/beta))²)
../build/wp1.7-scheduler-tests/_deps/traccc-build/device/cuda/fitting_algorithm.ptx:543144:///home/cano/NGT/freshbuild/build/wp1.7-scheduler-tests/_deps/detray-src/core/include/detray/materials/detail/relativistic_quantities.hpp:84     /// @return 2 * mass * (beta * gamma)² mass term
```

We can simply massage the `detray` source (included by `traccc`):
```
find ../build/ -name *.ptx | grep -Pn '[^[:ascii:]]' `cat` | cut -d : -f 3 | cut -c 3- | sort -u | perl -i -pe 's/[^[:ascii:]]/X/g' `cat`
```
and compilation completes in debug mode.

