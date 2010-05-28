WFLAGS= -w
OPTIMIZATION= -O3
CFLAGS= $(OPTIMIZATION) $(WFLAGS) -fPIC -ffast-math -funroll-loops
LFLAGS= -shared
PREFIX=/usr
TARGET=nnet.so

all:
	g++ -c $(CFLAGS) nnet.cpp
	g++ $(LFLAGS) -o $(TARGET) *.o -lhybris

debug:
	g++ -c -p -g -w -fPIC nnet.cpp
	g++ -shared -o $(TARGET) *.o -lhybris

clean:
	rm -f *.o $(TARGET)

install:
	cp $(TARGET) $(PREFIX)/lib/hybris/library/
	cp nnetwork.hy	$(PREFIX)/lib/hybris/include/
