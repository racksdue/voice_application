CXX = g++
CXXFLAGS = -std=c++17 -Iinstall/include
RPATH = -Wl,-rpath,@executable_path/install -Wl,-rpath,@executable_path/install/lib

all: v1-demo

v1-demo: demo.o
	$(CXX) demo.o install/libpiper.dylib install/lib/libonnxruntime.dylib $(RPATH) -o v1-demo

demo.o: demo.cpp
	$(CXX) $(CXXFLAGS) -c demo.cpp

run: v1-demo
	./v1-demo

.PHONY: clean run

clean:
	rm -f demo.o v1-demo output.raw
